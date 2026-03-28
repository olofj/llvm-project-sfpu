//===-- RISCVXttSFPUReplay.cpp - SFPU REPLAY Optimization -----------------===//
//
// MachineFunctionPass implementing the REPLAY instruction optimization for
// the Tenstorrent SFPU.
//
// The SFPU has a 32-entry instruction replay buffer that can record and
// replay instruction sequences. When the same sequence of instructions
// appears multiple times (e.g., in unrolled loops), the REPLAY instruction
// can replace subsequent occurrences, saving code size and reducing
// instruction fetch pressure.
//
// Algorithm:
// 1. Identify repeating instruction sequences >= 4 instructions long
// 2. Score candidates by savings: (num_clones - 1) * (seq_len - 1) - 1
// 3. Partition the 32-entry replay buffer among best candidates (knapsack)
// 4. Record the first occurrence and replace subsequent clones with REPLAY
//
// The savings formula accounts for:
// - The REPLAY instruction itself costs 1 cycle
// - Each clone after the first saves (seq_len - 1) instructions
// - The recording overhead is already in the first occurrence
//
// Reference: ttsim-analysis/FUNCTIONAL_UNITS.md Section 1 (MOP Expander)
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-xttsfpu-replay"

namespace {

/// Represents a sequence of SFPU instructions that could be replayed.
struct ReplayCandidate {
  unsigned StartIdx;       // Index into the instruction list
  unsigned Length;         // Number of instructions in the sequence
  SmallVector<unsigned, 8> CloneStarts;  // Indices of clone occurrences
  int Savings;             // Net instruction savings if replayed

  unsigned numClones() const { return CloneStarts.size(); }

  void computeSavings() {
    // Each clone saves (Length - 1) instructions (REPLAY replaces the clone
    // but costs 1 instruction itself).
    // Total savings: numClones * (Length - 1) - 1 (for the buffer setup)
    if (numClones() > 0 && Length >= 4) {
      Savings = static_cast<int>(numClones()) *
                    static_cast<int>(Length - 1) - 1;
    } else {
      Savings = 0;
    }
  }
};

class RISCVXttSFPUReplay : public MachineFunctionPass {
public:
  static char ID;

  RISCVXttSFPUReplay() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "RISC-V Tenstorrent SFPU REPLAY Optimization";
  }

private:
  static constexpr unsigned ReplayBufferSize = 32;
  static constexpr unsigned MinSequenceLength = 4;

  const RISCVSubtarget *STI = nullptr;
  const RISCVInstrInfo *TII = nullptr;

  /// Compute a hash for an instruction (opcode + operands).
  uint64_t hashInstruction(const MachineInstr &MI) const;

  /// Check if two instruction sequences are identical.
  bool sequencesMatch(ArrayRef<MachineInstr *> Seq1,
                      ArrayRef<MachineInstr *> Seq2) const;

  /// Find repeating SFPU instruction sequences in a basic block.
  void findCandidates(MachineBasicBlock &MBB,
                      SmallVectorImpl<ReplayCandidate> &Candidates);

  /// Greedy knapsack: allocate replay buffer entries to best candidates.
  void allocateBuffer(SmallVectorImpl<ReplayCandidate> &Candidates,
                      SmallVectorImpl<ReplayCandidate *> &Selected);
};

} // end anonymous namespace

char RISCVXttSFPUReplay::ID = 0;

uint64_t RISCVXttSFPUReplay::hashInstruction(const MachineInstr &MI) const {
  uint64_t Hash = MI.getOpcode();
  for (const MachineOperand &MO : MI.operands()) {
    Hash = Hash * 31;
    if (MO.isReg())
      Hash += MO.getReg().id();
    else if (MO.isImm())
      Hash += static_cast<uint64_t>(MO.getImm());
  }
  return Hash;
}

bool RISCVXttSFPUReplay::sequencesMatch(ArrayRef<MachineInstr *> Seq1,
                                          ArrayRef<MachineInstr *> Seq2) const {
  if (Seq1.size() != Seq2.size())
    return false;

  for (size_t I = 0, E = Seq1.size(); I < E; ++I) {
    if (Seq1[I]->getOpcode() != Seq2[I]->getOpcode())
      return false;

    // Compare all operands
    if (Seq1[I]->getNumOperands() != Seq2[I]->getNumOperands())
      return false;

    for (unsigned J = 0, JE = Seq1[I]->getNumOperands(); J < JE; ++J) {
      const MachineOperand &Op1 = Seq1[I]->getOperand(J);
      const MachineOperand &Op2 = Seq2[I]->getOperand(J);

      if (Op1.getType() != Op2.getType())
        return false;

      if (Op1.isReg() && Op1.getReg() != Op2.getReg())
        return false;
      if (Op1.isImm() && Op1.getImm() != Op2.getImm())
        return false;
    }
  }
  return true;
}

void RISCVXttSFPUReplay::findCandidates(
    MachineBasicBlock &MBB,
    SmallVectorImpl<ReplayCandidate> &Candidates) {

  // Collect all SFPU instructions in the block
  SmallVector<MachineInstr *, 64> SFPUInstrs;
  for (MachineInstr &MI : MBB) {
    // Check if this is an SFPU instruction by opcode range
    unsigned Opc = MI.getOpcode();
    if (Opc >= RISCV::SFPLOAD_BH && Opc <= RISCV::SFPMOV_CONFIG)
      SFPUInstrs.push_back(&MI);
  }

  if (SFPUInstrs.size() < MinSequenceLength * 2)
    return;  // Not enough instructions for any replay candidate

  // Hash-based sequence matching:
  // For each possible sequence length (4..16), hash each starting position
  // and find matches.
  for (unsigned Len = MinSequenceLength;
       Len <= std::min<unsigned>(16, SFPUInstrs.size() / 2); ++Len) {

    DenseMap<uint64_t, SmallVector<unsigned, 4>> HashToPositions;

    for (unsigned I = 0; I + Len <= SFPUInstrs.size(); ++I) {
      uint64_t Hash = 0;
      for (unsigned J = 0; J < Len; ++J)
        Hash = Hash * 37 + hashInstruction(*SFPUInstrs[I + J]);

      HashToPositions[Hash].push_back(I);
    }

    // For each group of matching hashes, verify actual match
    for (auto &[Hash, Positions] : HashToPositions) {
      if (Positions.size() < 2)
        continue;

      // Use first position as the "original", rest as clones
      ArrayRef<MachineInstr *> Original(SFPUInstrs.data() + Positions[0], Len);

      ReplayCandidate Cand;
      Cand.StartIdx = Positions[0];
      Cand.Length = Len;

      for (unsigned K = 1; K < Positions.size(); ++K) {
        // Verify non-overlapping
        if (Positions[K] < Positions[0] + Len)
          continue;

        ArrayRef<MachineInstr *> Clone(SFPUInstrs.data() + Positions[K], Len);
        if (sequencesMatch(Original, Clone))
          Cand.CloneStarts.push_back(Positions[K]);
      }

      if (!Cand.CloneStarts.empty()) {
        Cand.computeSavings();
        if (Cand.Savings > 0)
          Candidates.push_back(std::move(Cand));
      }
    }
  }
}

void RISCVXttSFPUReplay::allocateBuffer(
    SmallVectorImpl<ReplayCandidate> &Candidates,
    SmallVectorImpl<ReplayCandidate *> &Selected) {

  // Sort by savings descending
  llvm::sort(Candidates,
             [](const ReplayCandidate &A, const ReplayCandidate &B) {
               return A.Savings > B.Savings;
             });

  unsigned BufferUsed = 0;

  for (ReplayCandidate &Cand : Candidates) {
    if (BufferUsed + Cand.Length > ReplayBufferSize)
      continue;  // Doesn't fit

    Selected.push_back(&Cand);
    BufferUsed += Cand.Length;

    if (BufferUsed >= ReplayBufferSize)
      break;  // Buffer full
  }
}

bool RISCVXttSFPUReplay::runOnMachineFunction(MachineFunction &MF) {
  STI = &MF.getSubtarget<RISCVSubtarget>();

  if (!STI->hasVendorXttSFPU())
    return false;

  TII = STI->getInstrInfo();

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    SmallVector<ReplayCandidate, 8> Candidates;
    findCandidates(MBB, Candidates);

    if (Candidates.empty())
      continue;

    SmallVector<ReplayCandidate *, 4> Selected;
    allocateBuffer(Candidates, Selected);

    // Emit REPLAY instructions for selected candidates.
    //
    // The REPLAY instruction encoding (from sfpu-ops-bh.h):
    //   TT_OP_BH(0x04, (start_idx << 14) + (len << 4) +
    //            (execute_while_loading << 1) + (load_mode << 0))
    //
    // Protocol:
    // 1. First occurrence: emit normally (hardware records into replay buffer)
    //    Mark with REPLAY(start_idx, length, 0, 1) = "load" mode
    // 2. Each clone: replace entire sequence with single REPLAY instruction
    //    REPLAY(start_idx, length, 1, 0) = "execute" mode
    //
    // Collect all SFPU instructions to map candidates back to MachineInstrs
    SmallVector<MachineInstr *, 64> SFPUInstrs;
    for (MachineInstr &MI : MBB) {
      unsigned Opc = MI.getOpcode();
      if (Opc >= RISCV::SFPLOAD_BH && Opc <= RISCV::SFPMOV_CONFIG)
        SFPUInstrs.push_back(&MI);
    }

    unsigned NextSlot = 0;
    for (ReplayCandidate *Cand : Selected) {
      unsigned Slot = NextSlot;
      NextSlot += Cand->Length;

      LLVM_DEBUG(dbgs() << "REPLAY: slot " << Slot << ", "
                        << Cand->Length << " insns, "
                        << Cand->numClones() << " clones, saves "
                        << Cand->Savings << " insns\n");

      // Mark original sequence: insert REPLAY(slot, len, 0, 1) before it
      // load_mode=1 means "record the following instructions into the buffer"
      if (Cand->StartIdx < SFPUInstrs.size()) {
        MachineInstr *FirstInOriginal = SFPUInstrs[Cand->StartIdx];
        DebugLoc DL = FirstInOriginal->getDebugLoc();

        // TTREPLAY encoding: (slot << 14) | (len << 4) | (ewl << 1) | load_mode
        // Emit REPLAY in "load" mode: record following instructions.
        BuildMI(MBB, *FirstInOriginal, DL, TII->get(RISCV::TTREPLAY))
            .addImm(Slot)            // start_idx
            .addImm(Cand->Length)    // len
            .addImm(0)               // exec_while_load = 0
            .addImm(1);              // load_mode = 1 (record)
      }

      // Replace each clone with REPLAY(slot, len, 1, 0) = "execute" mode
      for (unsigned CloneStart : Cand->CloneStarts) {
        if (CloneStart + Cand->Length > SFPUInstrs.size())
          continue;

        MachineInstr *FirstInClone = SFPUInstrs[CloneStart];
        DebugLoc DL = FirstInClone->getDebugLoc();

        // Insert REPLAY in "execute" mode before the clone, then delete it.
        BuildMI(MBB, *FirstInClone, DL, TII->get(RISCV::TTREPLAY))
            .addImm(Slot)            // start_idx
            .addImm(Cand->Length)    // len
            .addImm(1)               // exec_while_load = 1 (execute)
            .addImm(0);              // load_mode = 0 (not recording)

        // Delete the clone instructions
        for (unsigned J = 0; J < Cand->Length && CloneStart + J < SFPUInstrs.size(); ++J) {
          MachineInstr *CloneInstr = SFPUInstrs[CloneStart + J];
          CloneInstr->eraseFromParent();
        }

        Changed = true;
        LLVM_DEBUG(dbgs() << "  Replaced clone at idx " << CloneStart
                          << " with REPLAY execute\n");
      }
    }
  }

  return Changed;
}

INITIALIZE_PASS(RISCVXttSFPUReplay, DEBUG_TYPE,
                "RISC-V Tenstorrent SFPU REPLAY Optimization", false, false)

FunctionPass *llvm::createRISCVXttSFPUReplayPass() {
  return new RISCVXttSFPUReplay();
}
