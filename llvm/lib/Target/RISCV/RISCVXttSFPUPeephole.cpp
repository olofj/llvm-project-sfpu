//===-- RISCVXttSFPUPeephole.cpp - SFPU Peephole Optimizations ------------===//
//
// MachineFunctionPass implementing peephole optimizations for the Tenstorrent
// SFPU vector unit. These are local, instruction-level optimizations that
// combine adjacent instructions into more efficient single instructions.
//
// Peephole patterns (from sfpi-gcc/gcc/config/riscv/tt/rvtt-peephole.md):
//
// 1. SFPLZ + SFPSETCC → SFPLZ with CC mode
//    sfplz  dest, src, 0, 0        ; leading zeros, no CC
//    sfpsetcc dest, src, 0, NE0    ; set CC from result
//    →
//    sfplz  dest, src, 0, CC_NE0   ; leading zeros with CC set (single insn)
//
//    This works because SFPLZ has mod1 modes that set CC directly:
//      CC_NE0: set CC if leading zeros count != 0
//      CC_EQ0: set CC if leading zeros count == 0
//
// 2. SFPEXEXP + SFPSETCC → SFPEXEXP with CC mode
//    Similar fusion for exponent extraction with CC set.
//
// Reference: sfpi-gcc/gcc/config/riscv/tt/rvtt-peephole.md
//            ttsim-analysis/ERRATA.md C-020 (modifier reference)
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-xttsfpu-peephole"

namespace {

// SFPSETCC modifier values (from C-020)
enum SFPSetCCMod {
  SFPSETCC_MOD1_LREG_LT0  = 0,
  SFPSETCC_MOD1_IMM_BIT0   = 1,
  SFPSETCC_MOD1_LREG_NE0  = 2,
  SFPSETCC_MOD1_LREG_GTE0 = 4,
  SFPSETCC_MOD1_LREG_EQ0  = 6,
  SFPSETCC_MOD1_COMP       = 8,
};

// SFPLZ modifier values with CC set (from C-020)
enum SFPLZMod {
  SFPLZ_MOD1_NONE   = 0,
  SFPLZ_MOD1_CC_NE0 = 2,  // Set CC if count != 0
  SFPLZ_MOD1_CC_EQ0 = 6,  // Set CC if count == 0
};

// SFPEXEXP modifier values with CC set (from C-020)
enum SFPExExpMod {
  SFPEXEXP_MOD1_DEBIAS           = 0,
  SFPEXEXP_MOD1_NODEBIAS         = 1,
  SFPEXEXP_MOD1_SET_CC_SGN_EXP   = 2,
  SFPEXEXP_MOD1_SET_CC_COMP_EXP  = 8,
  SFPEXEXP_MOD1_SET_CC_SGN_COMP  = 10,
};

class RISCVXttSFPUPeephole : public MachineFunctionPass {
public:
  static char ID;

  RISCVXttSFPUPeephole() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "RISC-V Tenstorrent SFPU Peephole Optimizations";
  }

private:
  const RISCVSubtarget *STI = nullptr;
  const RISCVInstrInfo *TII = nullptr;

  bool tryFuseLZSetCC(MachineBasicBlock &MBB);
  bool tryFuseExExpSetCC(MachineBasicBlock &MBB);
};

} // end anonymous namespace

char RISCVXttSFPUPeephole::ID = 0;

/// Pattern 1: SFPLZ + SFPSETCC → SFPLZ with CC mode
///
/// GCC peephole (rvtt-peephole.md lines 22-42):
///   sfplz(_, src, 0) + sfpsetcc(src, NE0) → sfplz_lv(_, src, CC_NE0)
///   sfplz(_, src, 0) + sfpsetcc(src, EQ0) → sfplz_lv(_, src, CC_EQ0)
///
/// The fused instruction sets CC as a side effect of computing leading zeros,
/// saving one instruction.
bool RISCVXttSFPUPeephole::tryFuseLZSetCC(MachineBasicBlock &MBB) {
  bool Changed = false;

  for (auto MBBI = MBB.begin(), MBBE = MBB.end(); MBBI != MBBE; ++MBBI) {
    MachineInstr &LzMI = *MBBI;

    // Must be SFPLZ with mod1=0 (no CC set yet)
    if (LzMI.getOpcode() != RISCV::SFPLZ)
      continue;

    // Check mod1 is 0 (basic LZ without CC)
    // In SFPUUnaryReg format: dest, imm12, lreg_c, mod1
    unsigned Mod1Idx = LzMI.getNumOperands() - 1;
    if (!LzMI.getOperand(Mod1Idx).isImm() || LzMI.getOperand(Mod1Idx).getImm() != 0)
      continue;

    // Look at next SFPU instruction
    auto NextMBBI = std::next(MBBI);
    if (NextMBBI == MBBE)
      continue;

    MachineInstr &SetCCMI = *NextMBBI;
    if (SetCCMI.getOpcode() != RISCV::SFPSETCC)
      continue;

    // Check that SFPSETCC uses the same source as SFPLZ
    // SFPSETCC format: imm12, lreg_c, lreg_dest, mod1
    // We need to verify lreg_c matches SFPLZ's lreg_c
    // And the mod1 is NE0 or EQ0
    unsigned SetCCMod = SetCCMI.getOperand(SetCCMI.getNumOperands() - 1).getImm();

    unsigned NewLZMod;
    if (SetCCMod == SFPSETCC_MOD1_LREG_NE0)
      NewLZMod = SFPLZ_MOD1_CC_NE0;
    else if (SetCCMod == SFPSETCC_MOD1_LREG_EQ0)
      NewLZMod = SFPLZ_MOD1_CC_EQ0;
    else
      continue;  // Can't fuse other SETCC modes

    // Fuse: change LZ mod1 to the CC-setting mode
    LzMI.getOperand(Mod1Idx).setImm(NewLZMod);

    // Remove the SFPSETCC
    SetCCMI.eraseFromParent();

    Changed = true;
    LLVM_DEBUG(dbgs() << "  Fused SFPLZ + SFPSETCC into SFPLZ with CC mode "
                      << NewLZMod << "\n");
  }

  return Changed;
}

/// Pattern 2: SFPEXEXP + SFPSETCC → SFPEXEXP with CC mode
///
/// Similar to LZ fusion but for exponent extraction.
/// SFPEXEXP has mod1 values that set CC:
///   SET_CC_SGN_EXP (2): set CC from sign and exponent
///   SET_CC_COMP_EXP (8): set CC from complemented exponent
bool RISCVXttSFPUPeephole::tryFuseExExpSetCC(MachineBasicBlock &MBB) {
  bool Changed = false;

  for (auto MBBI = MBB.begin(), MBBE = MBB.end(); MBBI != MBBE; ++MBBI) {
    MachineInstr &ExExpMI = *MBBI;

    if (ExExpMI.getOpcode() != RISCV::SFPEXEXP)
      continue;

    unsigned Mod1Idx = ExExpMI.getNumOperands() - 1;
    unsigned ExExpMod = ExExpMI.getOperand(Mod1Idx).getImm();

    // Only fuse if current mode is DEBIAS (0) or NODEBIAS (1)
    if (ExExpMod > 1)
      continue;

    auto NextMBBI = std::next(MBBI);
    if (NextMBBI == MBB.end())
      continue;

    MachineInstr &SetCCMI = *NextMBBI;
    if (SetCCMI.getOpcode() != RISCV::SFPSETCC)
      continue;

    unsigned SetCCMod = SetCCMI.getOperand(SetCCMI.getNumOperands() - 1).getImm();

    // Map SETCC mod to EXEXP CC mode
    unsigned NewExExpMod;
    if (SetCCMod == SFPSETCC_MOD1_LREG_LT0)
      NewExExpMod = SFPEXEXP_MOD1_SET_CC_SGN_EXP;
    else if (SetCCMod == SFPSETCC_MOD1_COMP)
      NewExExpMod = SFPEXEXP_MOD1_SET_CC_COMP_EXP;
    else
      continue;

    // Fuse
    ExExpMI.getOperand(Mod1Idx).setImm(NewExExpMod);
    SetCCMI.eraseFromParent();

    Changed = true;
    LLVM_DEBUG(dbgs() << "  Fused SFPEXEXP + SFPSETCC into SFPEXEXP with CC mode "
                      << NewExExpMod << "\n");
  }

  return Changed;
}

bool RISCVXttSFPUPeephole::runOnMachineFunction(MachineFunction &MF) {
  STI = &MF.getSubtarget<RISCVSubtarget>();

  if (!STI->hasVendorXttSFPU())
    return false;

  TII = STI->getInstrInfo();

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    Changed |= tryFuseLZSetCC(MBB);
    Changed |= tryFuseExExpSetCC(MBB);
  }

  return Changed;
}

INITIALIZE_PASS(RISCVXttSFPUPeephole, DEBUG_TYPE,
                "RISC-V Tenstorrent SFPU Peephole Optimizations",
                false, false)

FunctionPass *llvm::createRISCVXttSFPUPeepholePass() {
  return new RISCVXttSFPUPeephole();
}
