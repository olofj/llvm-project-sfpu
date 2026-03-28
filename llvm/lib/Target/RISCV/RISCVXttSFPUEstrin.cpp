//===-- RISCVXttSFPUEstrin.cpp - Horner→Estrin Polynomial Transform --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// MachineFunctionPass that transforms sequential Horner polynomial evaluation
// chains into parallel Estrin form, creating ILP that the scheduler can use
// to fill the 2-cycle MAD delay slots.
//
// Horner's method evaluates a_n*x^n + ... + a_1*x + a_0 as:
//   t = a_n
//   t = t*x + a_{n-1}    ← each depends on previous (no ILP)
//   t = t*x + a_{n-2}
//   ...
//
// Estrin's method pairs adjacent terms:
//   For degree 3: result = (a1*x + a0) + x^2*(a3*x + a2)
//     lo = a1*x + a0     ← independent
//     hi = a3*x + a2     ← independent (fills lo's delay slot)
//     x2 = x*x           ← independent
//     result = hi*x2 + lo
//
// On the SFPU with 2-cycle MAD, Horner costs 2*N cycles (each MAD depends on
// previous, causing a stall or NOP). Estrin costs ~1.5*N because independent
// MADs interleave naturally.
//
// This pass runs pre-RA in addMachineSSAOptimization(), so virtual registers
// are available for the restructured computation.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-xttsfpu-estrin"

namespace {

class RISCVXttSFPUEstrin : public MachineFunctionPass {
public:
  static char ID;

  RISCVXttSFPUEstrin() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "RISC-V Tenstorrent SFPU Horner-to-Estrin Transform";
  }

private:
  const RISCVSubtarget *STI = nullptr;
  const RISCVInstrInfo *TII = nullptr;
  MachineRegisterInfo *MRI = nullptr;

  struct HornerChain {
    SmallVector<MachineInstr *, 8> MADs;
    Register XReg;    // The common "x" variable
    unsigned Degree;  // Polynomial degree (== MADs.size())
  };

  bool findHornerChain(MachineInstr &StartMI, MachineBasicBlock &MBB,
                       HornerChain &Chain);
  bool transformToEstrin(HornerChain &Chain, MachineBasicBlock &MBB);
};

} // end anonymous namespace

char RISCVXttSFPUEstrin::ID = 0;

/// Detect a Horner chain starting at StartMI.
///
/// Horner pattern: a sequence of SFPMAD where each result feeds into the
/// next as a multiply operand, and the same X register appears throughout.
///
/// SFPMAD operand layout: dest(0), src_a(1), src_b(2), src_c(3), mod1(4)
/// Semantics: dest = src_a * src_b + src_c
bool RISCVXttSFPUEstrin::findHornerChain(MachineInstr &StartMI,
                                           MachineBasicBlock &MBB,
                                           HornerChain &Chain) {
  Chain.MADs.clear();
  Chain.Degree = 0;
  Chain.XReg = Register();

  if (StartMI.getOpcode() != RISCV::SFPMAD)
    return false;

  MachineInstr *Current = &StartMI;
  Register PrevResult;

  while (Current && Current->getOpcode() == RISCV::SFPMAD) {
    Chain.MADs.push_back(Current);

    Register Dest = Current->getOperand(0).getReg();
    Register SrcA = Current->getOperand(1).getReg();
    Register SrcB = Current->getOperand(2).getReg();

    if (Chain.MADs.size() == 1) {
      // First MAD: can't determine X yet (both src_a and src_b could be X).
      // Defer X identification until we see the second MAD, where the chain
      // dependency reveals which operand is the accumulated result and which is X.
    } else if (Chain.MADs.size() == 2) {
      // Second MAD reveals X: one of src_a/src_b is the previous result (chain),
      // the other is X. X is the one that also appeared in the first MAD.
      Register MAD0_SrcA = Chain.MADs[0]->getOperand(1).getReg();
      Register MAD0_SrcB = Chain.MADs[0]->getOperand(2).getReg();
      bool ChainedA = (SrcA == PrevResult);
      bool ChainedB = (SrcB == PrevResult);
      if (!ChainedA && !ChainedB)
        break;
      Register OtherSrc = ChainedA ? SrcB : SrcA;
      // X is OtherSrc if it also appeared in MAD0
      if (OtherSrc == MAD0_SrcA || OtherSrc == MAD0_SrcB) {
        Chain.XReg = OtherSrc;
        LLVM_DEBUG(dbgs() << "  Estrin: identified X = "
                          << printReg(Chain.XReg) << "\n");
      } else {
        break; // X changed — not a Horner chain
      }
      PrevResult = Dest;
      // Find next MAD
      MachineInstr *NextMAD = nullptr;
      if (Dest.isVirtual()) {
        for (auto &Use : MRI->use_instructions(Dest)) {
          if (Use.getOpcode() == RISCV::SFPMAD && Use.getParent() == &MBB) {
            NextMAD = &Use;
            break;
          }
        }
      }
      Current = NextMAD;
      continue;
    } else {
      // Verify the chain dependency: the previous result must be one of
      // the multiply operands (src_a or src_b).
      bool ChainedA = (SrcA == PrevResult);
      bool ChainedB = (SrcB == PrevResult);
      if (!ChainedA && !ChainedB)
        break;

      // Verify X is the other multiply operand.
      Register OtherSrc = ChainedA ? SrcB : SrcA;
      if (Chain.XReg && OtherSrc != Chain.XReg)
        break;
    }

    PrevResult = Dest;

    // Find the next MAD that consumes this result.
    MachineInstr *NextMAD = nullptr;
    if (Dest.isVirtual()) {
      for (auto &Use : MRI->use_instructions(Dest)) {
        if (Use.getOpcode() == RISCV::SFPMAD && Use.getParent() == &MBB) {
          NextMAD = &Use;
          break;
        }
      }
    }
    Current = NextMAD;
  }

  Chain.Degree = Chain.MADs.size();
  return Chain.Degree >= 3;
}

/// Transform a Horner chain of degree N into Estrin form.
///
/// Degree 3 (3 MADs, coefficients a0..a3):
///   Horner: t0 = a3*x + a2 → t1 = t0*x + a1 → t2 = t1*x + a0
///   Estrin: lo = a1*x + a0;  hi = a3*x + a2;  x2 = x*x;  result = hi*x2 + lo
///
/// We handle this by peeling off pairs from the bottom of the Horner chain
/// and restructuring them into independent sub-chains.
bool RISCVXttSFPUEstrin::transformToEstrin(HornerChain &Chain,
                                             MachineBasicBlock &MBB) {
  unsigned N = Chain.Degree;
  if (N < 3)
    return false;

  // Only handle degree 3 for now (3 dependent MADs → 2 independent + 1 combine).
  // Higher degrees use the same principle recursively.
  if (N > 4) {
    LLVM_DEBUG(dbgs() << "  Estrin: degree " << N << " > 4, skipping\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "  Estrin: transforming degree-" << N
                    << " Horner chain\n");

  // Horner chain: MADs[0] → MADs[1] → MADs[2] (→ MADs[3] for degree 4)
  //
  // MADs[0]: t0 = coeff_high * x + coeff_next
  // MADs[1]: t1 = t0 * x + coeff_next
  // MADs[2]: t2 = t1 * x + coeff_low
  //
  // Reading the chain bottom-up:
  // MADs[N-1] is the last MAD (produces the final result).
  // MADs[N-1]->src_c is a0 (the constant term or accumulator).
  // MADs[N-2]->src_c is a1 (or a1*x if degree > 3).

  // For degree 3 Estrin:
  // We need to build:
  //   lo  = MADs[1]->non_chain_multiply_op * x + MADs[2]->src_c   (a1*x + a0)
  //   hi  = MADs[0]                                                  (a3*x + a2, unchanged)
  //   x2  = x * x
  //   out = hi * x2 + lo

  MachineInstr *MAD0 = Chain.MADs[0]; // a3*x + a2 (or top of chain)
  MachineInstr *MAD1 = Chain.MADs[1]; // t0*x + a1
  MachineInstr *MAD2 = Chain.MADs[2]; // t1*x + a0

  Register X = Chain.XReg;
  Register FinalDest = MAD2->getOperand(0).getReg();
  Register A0 = MAD2->getOperand(3).getReg(); // src_c of last MAD
  Register A1 = MAD1->getOperand(3).getReg(); // src_c of second-to-last MAD
  unsigned Mod1 = MAD0->getOperand(4).getImm();
  DebugLoc DL = MAD0->getDebugLoc();

  // We need L9 (zero constant) for the MUL: x * x + 0
  Register L9 = RISCV::L9;

  // Create virtual registers for intermediates.
  const TargetRegisterClass *RC = &RISCV::SFPURegsRegClass;
  Register LoReg = MRI->createVirtualRegister(RC);
  Register X2Reg = MRI->createVirtualRegister(RC);
  Register HiReg = MAD0->getOperand(0).getReg(); // reuse MAD0's output

  // Insert before MAD1 (the first instruction we're replacing).
  MachineBasicBlock::iterator InsertPt = MachineBasicBlock::iterator(*MAD1);

  // Build: lo = A1 * X + A0   (independent of hi)
  BuildMI(MBB, InsertPt, DL, TII->get(RISCV::SFPMAD), LoReg)
      .addReg(A1)
      .addReg(X)
      .addReg(A0)
      .addImm(Mod1);

  // Build: x2 = X * X + L9   (independent of both lo and hi)
  // SFPMUL is src_a * src_b + L9 — but it's a 3-op with src_c = L9.
  // Actually SFPMUL: dest = src_a * src_b (src_c must be L9 on BH).
  BuildMI(MBB, InsertPt, DL, TII->get(RISCV::SFPMUL), X2Reg)
      .addReg(X)
      .addReg(X)
      .addReg(L9)
      .addImm(0);

  // Build: result = HiReg * X2 + LoReg
  // This replaces MAD2's output.
  BuildMI(MBB, InsertPt, DL, TII->get(RISCV::SFPMAD), FinalDest)
      .addReg(HiReg)
      .addReg(X2Reg)
      .addReg(LoReg)
      .addImm(Mod1);

  LLVM_DEBUG({
    dbgs() << "  Estrin: replaced " << N << " sequential MADs with:\n"
           << "    hi  = (kept) " << *MAD0
           << "    lo  = MAD  A1*X+A0\n"
           << "    x2  = MUL  X*X\n"
           << "    out = MAD  hi*x2+lo\n";
  });

  // Remove the old MADs (except MAD0 which we kept as "hi").
  MAD2->eraseFromParent();
  MAD1->eraseFromParent();

  // Handle degree 4: chain has 4 MADs. We've consumed MADs[1] and MADs[2].
  // MADs[3] (if present) was: t2*x + final_a0.
  // After restructuring, t2 is now produced by our new final MAD.
  // MADs[3] naturally consumes FinalDest and doesn't need modification
  // (the SSA value it reads is the same virtual register).
  // Actually for degree 4, we need to also handle MADs[3].
  if (N >= 4) {
    MachineInstr *MAD3 = Chain.MADs[3];
    // MAD3 reads MAD2's original output. Since we replaced MAD2 with our
    // new MAD that writes to FinalDest (same register), MAD3 automatically
    // picks up the new value. But we need to verify MAD3 is now reading
    // the correct register. If MAD3's chain input was the old MAD2 dest,
    // and we wrote to FinalDest = old MAD2 dest, it should be correct.
    //
    // However, FinalDest was MAD2's dest. MAD3 reads that. Our new SFPMAD
    // writes to FinalDest. So the def-use chain is preserved.
    LLVM_DEBUG(dbgs() << "  Estrin: degree 4, MAD3 now reads restructured "
                      << "output: " << *MAD3);
  }

  return true;
}

bool RISCVXttSFPUEstrin::runOnMachineFunction(MachineFunction &MF) {
  STI = &MF.getSubtarget<RISCVSubtarget>();

  if (!STI->hasVendorXttSFPU())
    return false;

  TII = STI->getInstrInfo();
  MRI = &MF.getRegInfo();

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    SmallVector<MachineInstr *, 4> ChainStarts;
    // Collect potential chain starts first to avoid iterator invalidation.
    for (MachineInstr &MI : MBB) {
      if (MI.getOpcode() == RISCV::SFPMAD) {
        ChainStarts.push_back(&MI);
        LLVM_DEBUG(dbgs() << "  Estrin: found SFPMAD: " << MI);
      }
    }

    for (MachineInstr *MI : ChainStarts) {
      // Skip if this instruction was already erased by a previous transform.
      if (MI->getParent() != &MBB)
        continue;

      HornerChain Chain;
      if (findHornerChain(*MI, MBB, Chain))
        Changed |= transformToEstrin(Chain, MBB);
    }
  }

  return Changed;
}

INITIALIZE_PASS(RISCVXttSFPUEstrin, DEBUG_TYPE,
                "RISC-V Tenstorrent SFPU Horner-to-Estrin Transform",
                false, false)

FunctionPass *llvm::createRISCVXttSFPUEstrinPass() {
  return new RISCVXttSFPUEstrin();
}
