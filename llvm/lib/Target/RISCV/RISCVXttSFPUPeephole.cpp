//===-- RISCVXttSFPUPeephole.cpp - SFPU Peephole Optimizations ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
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
#include "RISCVXttSFPUUtil.h"
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
  bool tryEliminateSelfMov(MachineBasicBlock &MBB);
  bool tryEliminateMovStore(MachineBasicBlock &MBB);
  bool tryEliminateLutSpill(MachineBasicBlock &MBB);
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

    // Verify SFPSETCC operates on the value SFPLZ produced.
    // SFPLZ format:   dest(0), imm12(1), lreg_c(2), mod1(3)
    // SFPSETCC format: imm12(0), lreg_c(1), lreg_dest(2), mod1(3)
    Register LzDest = LzMI.getOperand(0).getReg();
    Register SetCCSrc = SetCCMI.getOperand(1).getReg(); // lreg_c
    if (LzDest != SetCCSrc)
      continue; // SETCC reads a different register — don't fuse

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

    // Verify SFPSETCC operates on the value SFPEXEXP produced.
    Register ExExpDest = ExExpMI.getOperand(0).getReg();
    Register SetCCSrc = SetCCMI.getOperand(1).getReg(); // lreg_c
    if (ExExpDest != SetCCSrc)
      continue; // SETCC reads a different register — don't fuse

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

/// Pattern: Self-MOV elimination (SFPMOV Lx, Lx, 0, 0 → delete)
///
/// After register allocation with WH C-010 constraints, the coalescer may
/// leave behind identity MOVs where src == dest. These are safe to delete.
bool RISCVXttSFPUPeephole::tryEliminateSelfMov(MachineBasicBlock &MBB) {
  bool Changed = false;

  for (auto MBBI = MBB.begin(), MBBE = MBB.end(); MBBI != MBBE; ) {
    MachineInstr &MI = *MBBI++;

    if (MI.getOpcode() != RISCV::SFPMOV_REG)
      continue;

    // SFPMOV_REG format: dest, src, mod1
    // If dest == src and mod1 == 0, it's a no-op
    if (MI.getNumOperands() < 3)
      continue;

    const MachineOperand &Dst = MI.getOperand(0);
    const MachineOperand &Src = MI.getOperand(1);
    const MachineOperand &Mod = MI.getOperand(2);

    if (Dst.isReg() && Src.isReg() &&
        Dst.getReg() == Src.getReg() &&
        Mod.isImm() && Mod.getImm() == 0) {
      LLVM_DEBUG(dbgs() << "  Eliminating self-MOV: " << MI);
      MI.eraseFromParent();
      Changed = true;
    }
  }

  return Changed;
}

/// Eliminate LUT spill pattern: restructure code around SFPLUTFP32 to remove
/// redundant DEST spills. The RA spills half_in because it thinks the dead
/// output clobbers L3, but hardware writes to L7 only (lreg_dest=7).
///
/// Before: SFPMUL L3 → SFPSTORE L3 → [SFPLOAD L3] → SFPLUTFP32 →
///         SFPMOV L3,L7 → SFPLOAD L7,spill → SFPADD L7,L10,L7,L3
///
/// After:  SFPMUL L3 → SFPLUTFP32 → SFPADD L3,L10,L3,L7
///
/// L3 is preserved across SFPLUTFP32 (hardware writes L7 only). The SFPMOV
/// is removed. The SFPADD reads L3 (half_in) and L7 (LUT result) directly.
bool RISCVXttSFPUPeephole::tryEliminateLutSpill(MachineBasicBlock &MBB) {
  bool Changed = false;

  for (auto MBBI = MBB.begin(); MBBI != MBB.end(); ) {
    // Match: SFPSTORE(spill) -> SFPLOAD(same) -> SFPLUTFP32 ->
    //        SFPMOV(dest,L7) -> SFPLOAD(spill) -> SFPADD -> SFPMOV -> SFPSTORE(data)
    MachineInstr &MI = *MBBI;
    unsigned Opc = MI.getOpcode();
    if (Opc != RISCV::SFPSTORE_BH && Opc != RISCV::SFPSTORE_WH) {
      ++MBBI;
      continue;
    }

    // Check if this is a spill store (addr >= 128)
    unsigned AddrIdx = MI.getNumOperands() - 1;
    if (!MI.getOperand(AddrIdx).isImm() || MI.getOperand(AddrIdx).getImm() < 128) {
      ++MBBI;
      continue;
    }
    int SpillAddr = MI.getOperand(AddrIdx).getImm();
    Register SpillReg = MI.getOperand(0).getReg();

    // Match the exact sequence of 8 instructions
    auto I1 = MBBI; // SFPSTORE(spill)
    auto I2 = std::next(I1); if (I2 == MBB.end()) { ++MBBI; continue; }
    auto I3 = std::next(I2); if (I3 == MBB.end()) { ++MBBI; continue; }
    auto I4 = std::next(I3); if (I4 == MBB.end()) { ++MBBI; continue; }
    auto I5 = std::next(I4); if (I5 == MBB.end()) { ++MBBI; continue; }
    auto I6 = std::next(I5); if (I6 == MBB.end()) { ++MBBI; continue; }
    auto I7 = std::next(I6); if (I7 == MBB.end()) { ++MBBI; continue; }

    // I2: SFPLOAD from same spill addr (redundant reload)
    if (I2->getOpcode() != RISCV::SFPLOAD_BH && I2->getOpcode() != RISCV::SFPLOAD_WH) { ++MBBI; continue; }
    if (!I2->getOperand(I2->getNumOperands()-1).isImm() ||
        I2->getOperand(I2->getNumOperands()-1).getImm() != SpillAddr) { ++MBBI; continue; }

    // I3: SFPLUTFP32
    if (I3->getOpcode() != RISCV::SFPLUTFP32) { ++MBBI; continue; }

    // I4: SFPMOV dest, L7 (copy LUT result)
    if (I4->getOpcode() != RISCV::SFPMOV) { ++MBBI; continue; }
    if (!I4->getOperand(2).isReg() || I4->getOperand(2).getReg() != RISCV::L7) { ++MBBI; continue; }

    // I5: SFPLOAD from same spill addr (reload half_in)
    if (I5->getOpcode() != RISCV::SFPLOAD_BH && I5->getOpcode() != RISCV::SFPLOAD_WH) { ++MBBI; continue; }
    if (!I5->getOperand(I5->getNumOperands()-1).isImm() ||
        I5->getOperand(I5->getNumOperands()-1).getImm() != SpillAddr) { ++MBBI; continue; }
    Register ReloadReg = I5->getOperand(0).getReg();

    // I6: SFPADD using both LUT result and reloaded half_in
    if (I6->getOpcode() != RISCV::SFPADD) { ++MBBI; continue; }

    // Replace with SFPADD: half_in + LUT(half_in).
    // This matches the C++ source: result = half_in + lut2_sign(...)
    BuildMI(MBB, I6, I6->getDebugLoc(), TII->get(RISCV::SFPADD), SpillReg)
        .addReg(RISCV::L10)
        .addReg(SpillReg)
        .addReg(RISCV::L7)
        .addImm(0);

    // Compute safe iterator BEFORE any erasure. The new SFPADD was inserted
    // before I6. After erasing I4-I7, we want to continue from the SFPSTORE
    // (data store) which follows I7 (or the new SFPADD if I7 was the last).
    auto SafeIt = I7;
    if (I7 != MBB.end() && (I7->getOpcode() == RISCV::SFPMOV || I7->getOpcode() == RISCV::SFPMOV_LV)) {
      SafeIt = std::next(I7);
      I7->eraseFromParent();
    }
    MBBI = SafeIt;

    // Erase in reverse order
    I6->eraseFromParent();
    I5->eraseFromParent();
    I4->eraseFromParent();
    I2->eraseFromParent();
    I1->eraseFromParent();

    Changed = true;
  }
  return Changed;
}


/// Eliminate SFPMOV + SFPSTORE: store from the MOV source directly.
///
/// Pattern:
///   SFPMOV Ldst, Lsrc, 0, 0       ; Ldst = Lsrc (CC-predicated!)
///   SFPSTORE Ldst, ...             ; store Ldst
///
/// All SFPU MOVs are hardware-predicated by the CC register. If a previous
/// kernel left CC with some lanes masked, the MOV only updates active lanes.
/// The SFPSTORE then reads a mix of old and new values, corrupting output.
///
/// Fix: bypass the MOV and store from Lsrc directly, which has the correct
/// value for ALL lanes (not just CC-active ones).
///
/// Conditions:
/// - SFPMOV dest is only used by the immediately following SFPSTORE
/// - Lsrc is a valid store source (in SFPUStoreRegs)
/// - mod1 is 0 (no transformation)
bool RISCVXttSFPUPeephole::tryEliminateMovStore(MachineBasicBlock &MBB) {
  bool Changed = false;

  for (auto MBBI = MBB.begin(), MBBE = MBB.end(); MBBI != MBBE; ) {
    MachineInstr &MovMI = *MBBI;

    // Match SFPMOV (non-_lv) or SFPMOV_LV
    unsigned Opc = MovMI.getOpcode();
    if (Opc != RISCV::SFPMOV && Opc != RISCV::SFPMOV_LV) {
      ++MBBI;
      continue;
    }

    // Get the source register.
    // SFPMOV format:    dest(0), imm12(1), lreg_c(2), mod1(3)
    // SFPMOV_LV format: dest(0), live(1), imm12(2), lreg_c(3), mod1(4)
    unsigned SrcIdx = (Opc == RISCV::SFPMOV_LV) ? 3 : 2;
    unsigned ModIdx = (Opc == RISCV::SFPMOV_LV) ? 4 : 3;

    if (!MovMI.getOperand(SrcIdx).isReg())  { ++MBBI; continue; }
    if (!MovMI.getOperand(ModIdx).isImm() ||
        MovMI.getOperand(ModIdx).getImm() != 0) { ++MBBI; continue; }

    Register MovDst = MovMI.getOperand(0).getReg();
    Register MovSrc = MovMI.getOperand(SrcIdx).getReg();

    // Next instruction must be SFPSTORE using MovDst
    auto NextMBBI = std::next(MBBI);
    if (NextMBBI == MBBE) { ++MBBI; continue; }

    MachineInstr &StoreMI = *NextMBBI;
    unsigned StoreOpc = StoreMI.getOpcode();
    if (StoreOpc != RISCV::SFPSTORE_BH && StoreOpc != RISCV::SFPSTORE_WH) {
      ++MBBI;
      continue;
    }

    // Store's source operand (operand 0) must be MovDst
    if (!StoreMI.getOperand(0).isReg() ||
        StoreMI.getOperand(0).getReg() != MovDst) {
      ++MBBI;
      continue;
    }

    // Replace store's source with MovSrc
    StoreMI.getOperand(0).setReg(MovSrc);

    // Erase the MOV
    MBBI = NextMBBI;
    ++MBBI;  // advance past store before erasing mov
    MovMI.eraseFromParent();

    Changed = true;
    LLVM_DEBUG(dbgs() << "  Eliminated MOV before STORE, now storing from "
                      << printReg(MovSrc) << "\n");
  }

  return Changed;
}

bool RISCVXttSFPUPeephole::runOnMachineFunction(MachineFunction &MF) {
  STI = &MF.getSubtarget<RISCVSubtarget>();

  LLVM_DEBUG(dbgs() << getPassName() << " on " << MF.getName() << "\n");
  if (!STI->hasVendorXttSFPU())
    return false;

  TII = STI->getInstrInfo();

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    Changed |= tryFuseLZSetCC(MBB);
    Changed |= tryFuseExExpSetCC(MBB);
    Changed |= tryEliminateSelfMov(MBB);
    Changed |= tryEliminateMovStore(MBB);
    Changed |= tryEliminateLutSpill(MBB);
  }

  return Changed;
}

INITIALIZE_PASS(RISCVXttSFPUPeephole, DEBUG_TYPE,
                "RISC-V Tenstorrent SFPU Peephole Optimizations",
                false, false)

FunctionPass *llvm::createRISCVXttSFPUPeepholePass() {
  return new RISCVXttSFPUPeephole();
}
