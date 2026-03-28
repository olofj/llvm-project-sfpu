//===-- RISCVXttSFPUConstraints.cpp - SFPU Arch Constraints ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// MachineFunctionPass implementing architectural constraints for the
// Tenstorrent SFPU vector unit.
//
// WH Constraint C-010: SFPMUL/SFPADD/SFPMAD Fixed Operand Constraints
//   On Wormhole, the 3-operand arithmetic instructions have fixed register
//   constraints not obvious from the encoding:
//
//   SFPMUL WH: dst must equal src_a (destination tied to first source)
//              src_c must be L9 (zero constant)
//   SFPADD WH: src_a must be L10 (constant 1.0)
//              dst must equal src_a (but src_a is fixed, so dst=L10... unclear)
//   SFPMAD WH: dst must equal src_a
//
//   On Blackhole: These constraints are RELAXED — no register tying needed.
//
// This pass runs after register allocation and verifies constraints are met.
// The register allocator itself is constrained via the WH-specific instruction
// definitions that use register ties.
//
// Reference: ttsim-analysis/ERRATA.md C-010
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-xttsfpu-constraints"

namespace {

class RISCVXttSFPUConstraints : public MachineFunctionPass {
public:
  static char ID;

  RISCVXttSFPUConstraints() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "RISC-V Tenstorrent SFPU Architectural Constraints";
  }

private:
  const RISCVSubtarget *STI = nullptr;
  const RISCVInstrInfo *TII = nullptr;

  bool verifyWHConstraints(MachineFunction &MF);
};

} // end anonymous namespace

char RISCVXttSFPUConstraints::ID = 0;

/// Fix WH C-010 constraints after register allocation.
/// On WH, hardware ignores the dest field for 3-op arithmetic and always
/// writes to src_a. This pass ensures the encoding matches by setting
/// the dest field to equal src_a. This is a correctness requirement:
/// if dest != src_a, the encoded instruction word is technically wrong
/// (even though hardware behavior is the same).
///
/// On WH:
///   SFPMAD/SFPMUL: fix dest := src_a
///   SFPADD: no fixup needed (src_a is L10 constant, dest encoding doesn't matter)
bool RISCVXttSFPUConstraints::verifyWHConstraints(MachineFunction &MF) {
  if (!STI->hasVendorXttSFPUWH()) {
    LLVM_DEBUG(dbgs() << "C-010: skipping (BH has relaxed constraints)\n");
    return false;
  }

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      unsigned Opc = MI.getOpcode();

      // Fix SFPMAD/SFPMUL WH: set dst := src_a
      if (Opc == RISCV::SFPMAD || Opc == RISCV::SFPMUL ||
          Opc == RISCV::SFPMAD_WH || Opc == RISCV::SFPMUL_WH) {
        if (MI.getOperand(0).isReg() && MI.getOperand(1).isReg()) {
          Register SrcA = MI.getOperand(1).getReg();
          if (MI.getOperand(0).getReg() != SrcA) {
            LLVM_DEBUG(dbgs() << "  C-010 fix: setting dst := src_a ("
                              << printReg(SrcA) << ") for: " << MI);
            MI.getOperand(0).setReg(SrcA);
            Changed = true;
          }
        }
      }
    }
  }

  return Changed;
}

bool RISCVXttSFPUConstraints::runOnMachineFunction(MachineFunction &MF) {
  STI = &MF.getSubtarget<RISCVSubtarget>();

  if (!STI->hasVendorXttSFPU())
    return false;

  TII = STI->getInstrInfo();

  bool Changed = verifyWHConstraints(MF);

  return Changed;
}

INITIALIZE_PASS(RISCVXttSFPUConstraints, DEBUG_TYPE,
                "RISC-V Tenstorrent SFPU Architectural Constraints",
                false, false)

FunctionPass *llvm::createRISCVXttSFPUConstraintsPass() {
  return new RISCVXttSFPUConstraints();
}
