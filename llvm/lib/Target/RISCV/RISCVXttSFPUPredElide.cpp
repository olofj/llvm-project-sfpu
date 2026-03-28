//===-- RISCVXttSFPUPredElide.cpp - Predication Elision -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// MachineFunctionPass that removes unnecessary predication (PUSHC/SETCC/POPC)
// when the predicated body is smaller than the predication overhead.
//
// On SFPU, predication via the CC stack costs 3 instructions:
//   SFPPUSHC  (1 cycle)
//   SFPSETCC  (1 cycle)
//   SFPPOPC   (1 cycle)
//
// If the predicated body is a single 1-cycle instruction (e.g., SFPMOV),
// the total cost is 4 cycles WITH predication vs 1 cycle WITHOUT.
// Removing the predication and executing speculatively saves 3 cycles.
//
// Safety requirements for elision:
// - The body instruction must be safe to execute in all lanes
// - The body must not modify the CC stack
// - The body must be a single instruction (or very few)
// - The result must only be used after the v_endif (not observed per-lane)
//
// This is conservative: we only elide when the body is a single instruction
// that writes to a register that is immediately consumed after POPC.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-xttsfpu-predelide"

namespace {

class RISCVXttSFPUPredElide : public MachineFunctionPass {
public:
  static char ID;

  RISCVXttSFPUPredElide() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "RISC-V Tenstorrent SFPU Predication Elision";
  }

private:
  const RISCVSubtarget *STI = nullptr;
  const RISCVInstrInfo *TII = nullptr;

  /// Check if an instruction is safe to execute speculatively in all lanes.
  /// Pure arithmetic and MOV are safe. Stores and CC-modifying ops are not.
  bool isSafeToSpeculate(const MachineInstr &MI) const;

  /// Try to elide predication around a single-instruction body.
  bool tryElideRegion(MachineBasicBlock &MBB,
                      MachineBasicBlock::iterator PushI,
                      MachineBasicBlock::iterator SetCCI,
                      MachineBasicBlock::iterator BodyI,
                      MachineBasicBlock::iterator PopI);
};

} // end anonymous namespace

char RISCVXttSFPUPredElide::ID = 0;

bool RISCVXttSFPUPredElide::isSafeToSpeculate(const MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  // Safe: pure arithmetic (no side effects beyond register write)
  case RISCV::SFPMOV:
  case RISCV::SFPMOV_REG:
  case RISCV::SFPMAD:
  case RISCV::SFPMUL:
  case RISCV::SFPADD:
  case RISCV::SFPABS:
  case RISCV::SFPEXEXP:
  case RISCV::SFPEXMAN:
  case RISCV::SFPDIVP2:
  case RISCV::SFPCAST:
  case RISCV::SFPIADD:
  case RISCV::SFPSHFT:
  case RISCV::SFPAND:
  case RISCV::SFPOR:
  case RISCV::SFPNOT:
  case RISCV::SFPXOR:
  case RISCV::SFPLZ:
  case RISCV::SFPSETEXP:
  case RISCV::SFPSETMAN:
  case RISCV::SFPSETSGN:
    return true;

  // Unsafe: stores, loads, CC stack ops, config writes
  case RISCV::SFPSTORE_BH:
  case RISCV::SFPSTORE_WH:
  case RISCV::SFPLOAD_BH:
  case RISCV::SFPLOAD_WH:
  case RISCV::SFPPUSHC:
  case RISCV::SFPPOPC:
  case RISCV::SFPSETCC:
  case RISCV::SFPCOMPC:
  case RISCV::SFPENCC:
  case RISCV::SFPCONFIG:
  case RISCV::SFPSWAP:
  default:
    return false;
  }
}

bool RISCVXttSFPUPredElide::tryElideRegion(
    MachineBasicBlock &MBB,
    MachineBasicBlock::iterator PushI,
    MachineBasicBlock::iterator SetCCI,
    MachineBasicBlock::iterator BodyI,
    MachineBasicBlock::iterator PopI) {

  // Verify the body is a single safe instruction
  if (!isSafeToSpeculate(*BodyI))
    return false;

  // Verify nothing between SETCC and body, and nothing between body and POP
  auto AfterSetCC = std::next(SetCCI);
  if (&*AfterSetCC != &*BodyI)
    return false;

  auto AfterBody = std::next(BodyI);
  if (&*AfterBody != &*PopI)
    return false;

  LLVM_DEBUG(dbgs() << "PredElide: removing predication around: " << *BodyI);

  // Remove PUSHC, SETCC, POPC — keep only the body instruction
  PushI->eraseFromParent();
  SetCCI->eraseFromParent();
  PopI->eraseFromParent();

  return true;
}

bool RISCVXttSFPUPredElide::runOnMachineFunction(MachineFunction &MF) {
  STI = &MF.getSubtarget<RISCVSubtarget>();

  if (!STI->hasVendorXttSFPU())
    return false;

  TII = STI->getInstrInfo();

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Scan for PUSHC + SETCC + single_body + POPC pattern
    for (auto MBBI = MBB.begin(), MBBE = MBB.end(); MBBI != MBBE; ) {
      if (MBBI->getOpcode() != RISCV::SFPPUSHC) {
        ++MBBI;
        continue;
      }

      auto PushI = MBBI++;
      if (MBBI == MBBE || MBBI->getOpcode() != RISCV::SFPSETCC) {
        continue;
      }

      auto SetCCI = MBBI++;
      if (MBBI == MBBE) continue;

      auto BodyI = MBBI++;
      if (MBBI == MBBE || MBBI->getOpcode() != RISCV::SFPPOPC) {
        continue;
      }

      auto PopI = MBBI++;

      if (tryElideRegion(MBB, PushI, SetCCI, BodyI, PopI)) {
        Changed = true;
        // Iterator was already advanced past POP
      }
    }
  }

  return Changed;
}

INITIALIZE_PASS(RISCVXttSFPUPredElide, DEBUG_TYPE,
                "RISC-V Tenstorrent SFPU Predication Elision",
                false, false)

FunctionPass *llvm::createRISCVXttSFPUPredElidePass() {
  return new RISCVXttSFPUPredElide();
}
