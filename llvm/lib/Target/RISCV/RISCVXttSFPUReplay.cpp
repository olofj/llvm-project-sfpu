//===-- RISCVXttSFPUReplay.cpp - SFPU REPLAY Optimization -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
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
#include "MCTargetDesc/RISCVBaseInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-xttsfpu-replay"

namespace {

/// Represents a sequence of instructions that could be replayed.
struct ReplayCandidate {
  unsigned StartIdx;         // Index into the SFPU instruction list
  unsigned Length;           // Number of SFPU instructions in the sequence
  unsigned HardwareLength;   // Total Tensix instructions (SFPU + interleaved
                             // non-SFPU like INCRWC) — this is what the
                             // hardware replay buffer actually counts
  SmallVector<unsigned, 8> CloneStarts;  // SFPU-list indices of clone starts
  int Savings;               // Net instruction savings if replayed

  unsigned numClones() const { return CloneStarts.size(); }

  void computeSavings() {
    // Each clone saves (HardwareLength - 1) instructions (REPLAY replaces the
    // entire clone sequence but costs 1 instruction itself).
    if (numClones() > 0 && HardwareLength >= 4) {
      Savings = static_cast<int>(numClones()) *
                    static_cast<int>(HardwareLength - 1) - 1;
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
  // The 32-entry replay buffer is shared: slots 0-15 for FPU, 16-31 for SFPU.
  static constexpr unsigned ReplayBufferOffset = 16;  // SFPU starts here
  static constexpr unsigned ReplayBufferSize = 16;    // SFPU gets 16 slots
  static constexpr unsigned MinSequenceLength = 4;

  const RISCVSubtarget *STI = nullptr;
  const RISCVInstrInfo *TII = nullptr;

  /// Compute a hash for an instruction (opcode + operands).
  uint64_t hashInstruction(const MachineInstr &MI) const;

  /// Check if two instruction sequences are identical.
  bool sequencesMatch(ArrayRef<MachineInstr *> Seq1,
                      ArrayRef<MachineInstr *> Seq2) const;

  /// Check if a MachineInstr is a Tensix coprocessor instruction (SFPU or
  /// inline asm emitting .word/.ttinsn).  The hardware replay buffer records
  /// ALL Tensix instructions, not just SFPU.
  static bool isTensixInstr(const MachineInstr &MI) {
    if (MI.getDesc().TSFlags & RISCVII::IsXttSFPUMask)
      return true;
    // Inline asm in compute kernels emits Tensix instructions (INCRWC,
    // SETRWC, SETC16, etc.) via .word/.ttinsn directives.
    return MI.isInlineAsm();
  }

  /// Count total Tensix instructions from First to Last (inclusive) in the
  /// MBB.  Returns 0 if any non-Tensix (RISC-V) instruction is found in
  /// the range, which means the sequence can't be replayed.
  static unsigned countTensixInRange(const MachineInstr *First,
                                     const MachineInstr *Last) {
    unsigned Count = 0;
    for (auto It = First->getIterator();; ++It) {
      if (isTensixInstr(*It))
        ++Count;
      else
        return 0;  // RISC-V instruction breaks the Tensix sequence
      if (&*It == Last)
        break;
    }
    return Count;
  }

  /// Check that ALL instructions (SFPU + interleaved non-SFPU) match between
  /// two MBB ranges.  Both ranges must have the same number and types of
  /// instructions.
  bool fullRangeMatches(const MachineInstr *OrigFirst,
                        const MachineInstr *OrigLast,
                        const MachineInstr *CloneFirst,
                        const MachineInstr *CloneLast) const;

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

bool RISCVXttSFPUReplay::fullRangeMatches(
    const MachineInstr *OrigFirst, const MachineInstr *OrigLast,
    const MachineInstr *CloneFirst, const MachineInstr *CloneLast) const {
  auto OIt = OrigFirst->getIterator();
  auto CIt = CloneFirst->getIterator();
  auto OEnd = std::next(OrigLast->getIterator());
  auto CEnd = std::next(CloneLast->getIterator());

  while (OIt != OEnd && CIt != CEnd) {
    const MachineInstr &OMI = *OIt;
    const MachineInstr &CMI = *CIt;

    if (OMI.isInlineAsm() && CMI.isInlineAsm()) {
      // Compare inline asm by their asm string and immediate operands.
      // Two inline asm instructions match if they produce the same bytes.
      if (OMI.getNumOperands() != CMI.getNumOperands())
        return false;
      for (unsigned I = 0, E = OMI.getNumOperands(); I < E; ++I) {
        const MachineOperand &OOp = OMI.getOperand(I);
        const MachineOperand &COp = CMI.getOperand(I);
        if (OOp.getType() != COp.getType())
          return false;
        if (OOp.isImm() && OOp.getImm() != COp.getImm())
          return false;
      }
    } else if (OMI.getDesc().TSFlags & RISCVII::IsXttSFPUMask) {
      // SFPU instructions — already matched by sequencesMatch, skip
    } else {
      // Type mismatch (one is inline asm, other is SFPU, or neither)
      return false;
    }

    ++OIt;
    ++CIt;
  }
  return OIt == OEnd && CIt == CEnd;
}

void RISCVXttSFPUReplay::findCandidates(
    MachineBasicBlock &MBB,
    SmallVectorImpl<ReplayCandidate> &Candidates) {

  // Collect all SFPU instructions in the block (excluding TTREPLAY itself).
  // Use TSFlags to identify SFPU instructions — covers all current and future
  // SFPU opcodes without relying on fragile enum ordering.
  SmallVector<MachineInstr *, 64> SFPUInstrs;
  for (MachineInstr &MI : MBB) {
    if ((MI.getDesc().TSFlags & RISCVII::IsXttSFPUMask) &&
        MI.getOpcode() != RISCV::TTREPLAY)
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
      Cand.HardwareLength = Len;  // Updated later in runOnMachineFunction

      // Track the end of the last accepted region to ensure no overlaps.
      unsigned LastEnd = Positions[0] + Len;
      for (unsigned K = 1; K < Positions.size(); ++K) {
        // Verify non-overlapping with original and all previous clones
        if (Positions[K] < LastEnd)
          continue;

        ArrayRef<MachineInstr *> Clone(SFPUInstrs.data() + Positions[K], Len);
        if (sequencesMatch(Original, Clone)) {
          Cand.CloneStarts.push_back(Positions[K]);
          LastEnd = Positions[K] + Len;
        }
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

  // Track which instruction indices are already claimed by a selected
  // candidate (original or clone). This prevents overlapping selections.
  DenseSet<unsigned> UsedIndices;
  unsigned BufferUsed = 0;

  for (ReplayCandidate &Cand : Candidates) {
    if (BufferUsed + Cand.HardwareLength > ReplayBufferSize)
      continue;  // Doesn't fit in replay buffer

    // Check that the original doesn't overlap with already-selected regions
    bool OriginalOverlaps = false;
    for (unsigned I = Cand.StartIdx; I < Cand.StartIdx + Cand.Length; ++I) {
      if (UsedIndices.count(I)) {
        OriginalOverlaps = true;
        break;
      }
    }
    if (OriginalOverlaps)
      continue;

    // Filter out clones that overlap with already-selected regions
    SmallVector<unsigned, 8> ValidClones;
    for (unsigned CloneStart : Cand.CloneStarts) {
      bool Overlaps = false;
      for (unsigned I = CloneStart; I < CloneStart + Cand.Length; ++I) {
        if (UsedIndices.count(I)) {
          Overlaps = true;
          break;
        }
      }
      if (!Overlaps)
        ValidClones.push_back(CloneStart);
    }

    if (ValidClones.empty())
      continue;

    // Update the candidate with only valid (non-overlapping) clones
    Cand.CloneStarts = std::move(ValidClones);
    Cand.computeSavings();
    if (Cand.Savings <= 0)
      continue;

    // Mark all indices as used
    for (unsigned I = Cand.StartIdx; I < Cand.StartIdx + Cand.Length; ++I)
      UsedIndices.insert(I);
    for (unsigned CloneStart : Cand.CloneStarts)
      for (unsigned I = CloneStart; I < CloneStart + Cand.Length; ++I)
        UsedIndices.insert(I);

    Selected.push_back(&Cand);
    BufferUsed += Cand.HardwareLength;

    if (BufferUsed >= ReplayBufferSize)
      break;  // Buffer full
  }
}

bool RISCVXttSFPUReplay::runOnMachineFunction(MachineFunction &MF) {
  STI = &MF.getSubtarget<RISCVSubtarget>();

  LLVM_DEBUG(dbgs() << getPassName() << " on " << MF.getName() << "\n");
  if (!STI->hasVendorXttSFPU())
    return false;

  TII = STI->getInstrInfo();

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Collect SFPU instructions for candidate finding (matching is still
    // SFPU-only — the interleaved non-SFPU instructions are verified
    // separately via fullRangeMatches).
    SmallVector<MachineInstr *, 64> SFPUInstrs;
    for (MachineInstr &MI : MBB) {
      if ((MI.getDesc().TSFlags & RISCVII::IsXttSFPUMask) &&
          MI.getOpcode() != RISCV::TTREPLAY)
        SFPUInstrs.push_back(&MI);
    }

    SmallVector<ReplayCandidate, 8> Candidates;
    findCandidates(MBB, Candidates);

    if (Candidates.empty())
      continue;

    // Compute HardwareLength for each candidate: count ALL Tensix
    // instructions (SFPU + inline asm like INCRWC) in the range from the
    // first to last SFPU instruction.  The hardware replay buffer records
    // all Tensix instructions, not just SFPU ones.
    //
    // Also verify that the interleaved non-SFPU instructions match between
    // original and each clone.  Reject candidates/clones that don't match.
    for (ReplayCandidate &Cand : Candidates) {
      if (Cand.StartIdx + Cand.Length > SFPUInstrs.size()) {
        Cand.Savings = 0;
        continue;
      }

      MachineInstr *OrigFirst = SFPUInstrs[Cand.StartIdx];
      MachineInstr *OrigLast = SFPUInstrs[Cand.StartIdx + Cand.Length - 1];
      unsigned HWLen = countTensixInRange(OrigFirst, OrigLast);

      if (HWLen == 0 || HWLen > ReplayBufferSize) {
        // RISC-V instruction in range or too long — can't replay
        Cand.Savings = 0;
        continue;
      }
      Cand.HardwareLength = HWLen;

      // Verify each clone: interleaved instructions must match, and the
      // clone range must also be pure Tensix.
      SmallVector<unsigned, 8> ValidClones;
      for (unsigned CloneStart : Cand.CloneStarts) {
        if (CloneStart + Cand.Length > SFPUInstrs.size())
          continue;
        MachineInstr *CloneFirst = SFPUInstrs[CloneStart];
        MachineInstr *CloneLast = SFPUInstrs[CloneStart + Cand.Length - 1];

        unsigned CloneHWLen = countTensixInRange(CloneFirst, CloneLast);
        if (CloneHWLen != HWLen)
          continue;  // Different interleaving — can't replay

        if (!fullRangeMatches(OrigFirst, OrigLast, CloneFirst, CloneLast))
          continue;  // Interleaved instructions differ

        ValidClones.push_back(CloneStart);
      }
      Cand.CloneStarts = std::move(ValidClones);
      Cand.computeSavings();
    }

    SmallVector<ReplayCandidate *, 4> Selected;
    allocateBuffer(Candidates, Selected);

    if (Selected.empty())
      continue;

    // Emit REPLAY instructions for selected candidates.
    //
    // Protocol:
    // 1. First occurrence: insert REPLAY(slot, hwlen, ewl=1, load=1)
    //    Records the next hwlen Tensix instructions AND executes them.
    //    ewl=1 (execute_while_load) is essential — without it, the
    //    instructions are recorded but not executed, skipping the first
    //    occurrence entirely.
    // 2. Each clone: replace ALL instructions (SFPU + non-SFPU Tensix)
    //    with a single REPLAY(slot, hwlen, 0, 0) instruction.
    unsigned NextSlot = ReplayBufferOffset;
    for (ReplayCandidate *Cand : Selected) {
      unsigned Slot = NextSlot;
      NextSlot += Cand->HardwareLength;

      LLVM_DEBUG(dbgs() << "REPLAY: slot " << Slot << ", "
                        << Cand->HardwareLength << " hw insns ("
                        << Cand->Length << " SFPU), "
                        << Cand->numClones() << " clones, saves "
                        << Cand->Savings << " insns\n");

      // Mark original: insert REPLAY(record) before first instruction.
      MachineInstr *FirstInOriginal = SFPUInstrs[Cand->StartIdx];
      if (!FirstInOriginal || FirstInOriginal->getParent() != &MBB)
        continue;
      DebugLoc DL = FirstInOriginal->getDebugLoc();

      BuildMI(MBB, *FirstInOriginal, DL, TII->get(RISCV::TTREPLAY))
          .addImm(Slot)
          .addImm(Cand->HardwareLength)
          .addImm(1)   // exec_while_load = 1 (record AND execute)
          .addImm(1);  // load_mode = 1 (record)

      // Replace each clone: delete ALL instructions from the first to last
      // SFPU instruction in the clone (including interleaved non-SFPU),
      // and insert a single REPLAY(execute) instruction.
      for (unsigned CloneStart : Cand->CloneStarts) {
        if (CloneStart + Cand->Length > SFPUInstrs.size())
          continue;

        MachineInstr *CloneFirst = SFPUInstrs[CloneStart];
        MachineInstr *CloneLast = SFPUInstrs[CloneStart + Cand->Length - 1];
        if (!CloneFirst || CloneFirst->getParent() != &MBB)
          continue;

        DL = CloneFirst->getDebugLoc();

        BuildMI(MBB, *CloneFirst, DL, TII->get(RISCV::TTREPLAY))
            .addImm(Slot)
            .addImm(Cand->HardwareLength)
            .addImm(0)
            .addImm(0);  // load_mode = 0 (replay)

        // Erase ALL instructions from CloneFirst through CloneLast.
        // This includes SFPU instructions AND any interleaved non-SFPU
        // Tensix instructions (like INCRWC), since the replay buffer
        // replays ALL of them.
        auto It = CloneFirst->getIterator();
        auto End = std::next(CloneLast->getIterator());
        while (It != End) {
          MachineInstr &MI = *It++;
          // Null out SFPUInstrs entries for erased SFPU instructions
          if (MI.getDesc().TSFlags & RISCVII::IsXttSFPUMask) {
            for (unsigned J = CloneStart;
                 J < CloneStart + Cand->Length && J < SFPUInstrs.size(); ++J) {
              if (SFPUInstrs[J] == &MI) {
                SFPUInstrs[J] = nullptr;
                break;
              }
            }
          }
          MI.eraseFromParent();
        }

        Changed = true;
        LLVM_DEBUG(dbgs() << "  Replaced clone at SFPU idx " << CloneStart
                          << " (" << Cand->HardwareLength
                          << " instructions) with REPLAY execute\n");
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
