//===-- RISCVXttSFPUConstraints.cpp - SFPU Arch Constraints ---------------===//
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

/// Verify WH C-010 constraints after register allocation.
/// On WH:
///   SFPMAD: dst must equal src_a
///   SFPMUL: dst must equal src_a, src_c must be L9
///   SFPADD: src_a must be L10, dst must equal src_a (implied: dst=L10)
///
/// These constraints are enforced at the TableGen level for WH-specific
/// instruction variants (SFPMAD_WH, etc.) but this pass provides a
/// safety-net verification.
bool RISCVXttSFPUConstraints::verifyWHConstraints(MachineFunction &MF) {
  if (!STI->hasVendorXttSFPUWH())
    return false;  // BH has relaxed constraints

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      unsigned Opc = MI.getOpcode();

      // Check SFPMAD WH: dst == src_a
      if (Opc == RISCV::SFPMAD) {
        // In 3-Op format: operand 0 = dest, operand 1 = src_a
        if (MI.getOperand(0).isReg() && MI.getOperand(1).isReg()) {
          if (MI.getOperand(0).getReg() != MI.getOperand(1).getReg()) {
            MI.emitError("C-010: WH SFPMAD requires dst == src_a");
          }
        }
      }

      // Check SFPMUL WH: dst == src_a, src_c == L9
      if (Opc == RISCV::SFPMUL) {
        if (MI.getOperand(0).isReg() && MI.getOperand(1).isReg()) {
          if (MI.getOperand(0).getReg() != MI.getOperand(1).getReg()) {
            MI.emitError("C-010: WH SFPMUL requires dst == src_a");
          }
        }
        // src_c is operand 3 in our 3-Op format
        if (MI.getNumOperands() > 3 && MI.getOperand(3).isReg()) {
          if (MI.getOperand(3).getReg() != RISCV::L9) {
            MI.emitError("C-010: WH SFPMUL requires src_c == L9 (zero)");
          }
        }
      }

      // Check SFPADD WH: src_a == L10
      if (Opc == RISCV::SFPADD) {
        if (MI.getNumOperands() > 1 && MI.getOperand(1).isReg()) {
          if (MI.getOperand(1).getReg() != RISCV::L10) {
            MI.emitError("C-010: WH SFPADD requires src_a == L10 (one)");
          }
        }
      }
    }
  }

  return false;  // Verification-only pass, doesn't modify code
}

bool RISCVXttSFPUConstraints::runOnMachineFunction(MachineFunction &MF) {
  STI = &MF.getSubtarget<RISCVSubtarget>();

  if (!STI->hasVendorXttSFPU())
    return false;

  TII = STI->getInstrInfo();

  verifyWHConstraints(MF);

  return false;
}

INITIALIZE_PASS(RISCVXttSFPUConstraints, DEBUG_TYPE,
                "RISC-V Tenstorrent SFPU Architectural Constraints",
                false, false)

FunctionPass *llvm::createRISCVXttSFPUConstraintsPass() {
  return new RISCVXttSFPUConstraints();
}
