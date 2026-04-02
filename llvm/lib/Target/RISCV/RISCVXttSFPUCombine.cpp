//===-- RISCVXttSFPUCombine.cpp - SFPU DAG Combining ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// Custom DAG combining patterns for the Tenstorrent SFPU.
// These implement the same optimizations as GCC's gimple-rvtt-combine.cc
// but using LLVM's DAG combiner infrastructure.
//
// Combining patterns:
//
// 1. MUL+ADD → MAD (GH-Q-002 fix)
//    sfpmul(a, b, L9, mod) → tmp; sfpadd(L10, tmp, c, mod) → sfpmad(a, b, c, mod)
//    Saves 1 instruction + potentially 1 NOP cycle.
//
// 2. SFPLOADI + SFPMUL → SFPMULI (immediate fold)
//    sfploadi(imm16) → tmp; sfpmul(tmp, x, L9, mod) → sfpmuli(x, imm16, mod)
//    Saves 1 instruction + frees 1 register.
//
// 3. SFPLOADI + SFPADD → SFPADDI (immediate fold)
//    sfploadi(imm16) → tmp; sfpadd(L10, tmp, x, mod) → sfpaddi(x, imm16, mod)
//    Saves 1 instruction + frees 1 register.
//
// 4. Negated operand folding (BH only)
//    sfpmul(a, L11(-1.0), L9, 0) → negate; sfpmul(negate, b, L9, mod)
//    → sfpmul(a, b, L9, mod ^ NEGATE_A)
//    Saves 1 instruction (the negation multiply).
//
// Reference: sfpi-gcc/gcc/config/riscv/tt/gimple-rvtt-combine.cc
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-xttsfpu-combine"

namespace {

class RISCVXttSFPUCombine : public MachineFunctionPass {
public:
  static char ID;

  RISCVXttSFPUCombine() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "RISC-V Tenstorrent SFPU Instruction Combining";
  }

private:
  const RISCVSubtarget *STI = nullptr;
  const RISCVInstrInfo *TII = nullptr;

  bool tryCombineMulAdd(MachineBasicBlock &MBB);
  bool tryCombineLoadiIntoImm(MachineBasicBlock &MBB);
  bool tryCombineNegatedOperands(MachineBasicBlock &MBB);
  bool tryCombineLutInput(MachineBasicBlock &MBB);

  /// Check if a register has exactly one use in the function.
  bool hasOneUse(Register Reg, const MachineRegisterInfo &MRI) const;

  /// Find the defining instruction for a register.
  MachineInstr *getDefInst(Register Reg, const MachineRegisterInfo &MRI) const;
};

} // end anonymous namespace

char RISCVXttSFPUCombine::ID = 0;

bool RISCVXttSFPUCombine::hasOneUse(Register Reg,
                                      const MachineRegisterInfo &MRI) const {
  if (!Reg.isVirtual())
    return false;
  return MRI.hasOneUse(Reg);
}

MachineInstr *RISCVXttSFPUCombine::getDefInst(
    Register Reg, const MachineRegisterInfo &MRI) const {
  if (!Reg.isVirtual())
    return nullptr;
  return MRI.getVRegDef(Reg);
}

/// Pattern 1: MUL + ADD → MAD
///
/// Look for:
///   %mul = SFPMUL src_a, src_b, L9, mod_mul
///   %result = SFPADD L10, %mul, src_c, mod_add
/// Or:
///   %result = SFPADD L10, src_c, %mul, mod_add
///
/// Replace with:
///   %result = SFPMAD src_a, src_b, src_c, (mod_mul ^ mod_add)
///
/// GCC does this in gimple-rvtt-combine.cc:try_combine_mul_add()
/// but ONLY for non-_lv MUL feeding into ADD. _lv MUL can't propagate
/// through MAD_lv because the live value semantics differ.
bool RISCVXttSFPUCombine::tryCombineMulAdd(MachineBasicBlock &MBB) {
  const MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  bool Changed = false;

  for (auto MBBI = MBB.begin(), MBBE = MBB.end(); MBBI != MBBE; ) {
    MachineInstr &AddMI = *MBBI;
    ++MBBI;  // Advance before potential erase

    // Must be SFPADD (not _lv for now)
    if (AddMI.getOpcode() != RISCV::SFPADD)
      continue;

    // SFPADD operands: dest, src_a, src_b, src_c, mod1
    // Try both src_b and src_c as the MUL result
    for (unsigned OpIdx : {2u, 3u}) {
      const MachineOperand &MulResultOp = AddMI.getOperand(OpIdx);
      if (!MulResultOp.isReg())
        continue;

      Register MulResult = MulResultOp.getReg();
      if (!hasOneUse(MulResult, MRI))
        continue;

      MachineInstr *MulMI = getDefInst(MulResult, MRI);
      if (!MulMI || MulMI->getOpcode() != RISCV::SFPMUL)
        continue;

      // Must be in the same basic block (GCC checks this too)
      if (MulMI->getParent() != &MBB)
        continue;

      // Get the "other" operand of SFPADD (the one that's not from MUL)
      unsigned OtherIdx = (OpIdx == 2) ? 3 : 2;
      const MachineOperand &OtherOp = AddMI.getOperand(OtherIdx);

      // Build SFPMAD: dest = MUL.src_a * MUL.src_b + ADD.other
      // SFPMAD operands: dest, src_a, src_b, src_c, mod1
      unsigned AddMod = AddMI.getOperand(4).getImm();
      unsigned MulMod = MulMI->getOperand(4).getImm();
      unsigned MadMod = AddMod ^ MulMod;

      BuildMI(MBB, AddMI, AddMI.getDebugLoc(), TII->get(RISCV::SFPMAD))
          .add(AddMI.getOperand(0))        // dest (from ADD)
          .add(MulMI->getOperand(1))       // src_a (from MUL)
          .add(MulMI->getOperand(2))       // src_b (from MUL)
          .add(OtherOp)                     // src_c (from ADD other)
          .addImm(MadMod);                  // mod1 = add_mod ^ mul_mod

      // Remove the ADD and MUL (iterator already advanced past AddMI)
      AddMI.eraseFromParent();
      MulMI->eraseFromParent();

      Changed = true;
      LLVM_DEBUG(dbgs() << "  Combined MUL+ADD into MAD\n");
      break;  // Exit inner loop, outer loop already advanced
    }
  }

  return Changed;
}

/// Pattern 2 & 3: SFPLOADI + SFPMUL → SFPMULI, SFPLOADI + SFPADD → SFPADDI
///
/// Look for:
///   %imm = SFPLOADI lreg, mod0, imm16
///   %result = SFPMUL %imm, other, L9, mod    (or ADD: L10, %imm, other, mod)
///
/// Replace with:
///   %result = SFPMULI other, imm16, mod       (or SFPADDI other, imm16, mod)
///
/// Conditions (from GCC gimple-rvtt-combine.cc:try_gen_muli_or_addi):
/// - SFPLOADI result has single use (otherwise we'd need the register anyway)
/// - Must be in same BB
/// - No intervening CC state change
/// - Not _lv variant of SFPLOADI (live value may differ per lane)
bool RISCVXttSFPUCombine::tryCombineLoadiIntoImm(MachineBasicBlock &MBB) {
  const MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  bool Changed = false;

  for (auto MBBI = MBB.begin(), MBBE = MBB.end(); MBBI != MBBE; ) {
    MachineInstr &MI = *MBBI++;

    // Must be SFPMUL or SFPADD (not _lv)
    unsigned Opc = MI.getOpcode();
    bool IsMul = (Opc == RISCV::SFPMUL);
    bool IsAdd = (Opc == RISCV::SFPADD);
    if (!IsMul && !IsAdd)
      continue;

    // Check each source operand for SFPLOADI
    for (unsigned OpIdx = 1; OpIdx <= 3; ++OpIdx) {
      const MachineOperand &SrcOp = MI.getOperand(OpIdx);
      if (!SrcOp.isReg())
        continue;

      Register SrcReg = SrcOp.getReg();
      if (!hasOneUse(SrcReg, MRI))
        continue;

      MachineInstr *LoadiMI = getDefInst(SrcReg, MRI);
      if (!LoadiMI || LoadiMI->getOpcode() != RISCV::SFPLOADI)
        continue;

      if (LoadiMI->getParent() != &MBB)
        continue;

      // Get the immediate from SFPLOADI
      // SFPLOADI format: lreg_ind, mod0, imm16
      unsigned Imm16 = LoadiMI->getOperand(2).getImm();
      unsigned Mod = MI.getOperand(4).getImm();

      // Find the "other" source operand (the non-LOADI one)
      // For MUL: operands are dest, src_a, src_b, src_c, mod1
      //   src_a or src_b could be the LOADI result
      // For ADD: operands are dest, src_a, src_b, src_c, mod1
      //   src_a(L10) is fixed, src_b or src_c could be LOADI result

      unsigned ImmOpc = IsMul ? RISCV::SFPMULI : RISCV::SFPADDI;

      // SFPMULI/SFPADDI format: dest, imm16, mod1
      // The "other" register becomes an implicit source (dst-as-src)
      BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(ImmOpc))
          .add(MI.getOperand(0))     // dest
          .addImm(Imm16)             // imm16 from LOADI
          .addImm(Mod);              // mod1

      // Remove the MUL/ADD and SFPLOADI
      MI.eraseFromParent();
      LoadiMI->eraseFromParent();

      Changed = true;
      LLVM_DEBUG(dbgs() << "  Combined LOADI+"
                        << (IsMul ? "MUL" : "ADD")
                        << " into " << (IsMul ? "MULI" : "ADDI") << "\n");
      break;
    }
  }

  return Changed;
}

/// Pattern 4: Negated operand folding (BH only)
///
/// On Blackhole, SFPMAD/SFPMUL/SFPADD have mod1 bits to negate operands:
///   SFPMAD_MOD1_BH_COMPL_A = 1 (bit 0) — negate src_a
///   SFPMAD_MOD1_BH_COMPL_C = 2 (bit 1) — negate src_c
///
/// Look for:
///   %neg = SFPMUL %x, L11(-1.0), L9, 0   ; negation via multiply by -1
///   %result = SFPMAD %neg, %b, %c, mod    ; uses negated value
/// Replace with:
///   %result = SFPMAD %x, %b, %c, mod ^ COMPL_A  ; toggle negate bit
///
/// Also handles:
///   - Negation as input to SFPADD (toggle COMPL_C for addend)
///   - Negated result: -(a*b+c) → toggle both COMPL_A and COMPL_C
///
/// GCC: gimple-rvtt-combine.cc:try_combine_negated_operands()
bool RISCVXttSFPUCombine::tryCombineNegatedOperands(MachineBasicBlock &MBB) {
  if (!STI->hasVendorXttSFPUBH())
    return false;  // BH-only optimization (mod1 complement bits)

  const MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  bool Changed = false;

  // BH mod1 complement bits (from rvtt-protos.h)
  constexpr unsigned COMPL_A = 1;  // Negate operand A
  constexpr unsigned COMPL_C = 2;  // Negate operand C

  for (auto MBBI = MBB.begin(), MBBE = MBB.end(); MBBI != MBBE; ++MBBI) {
    MachineInstr &MI = *MBBI;

    unsigned Opc = MI.getOpcode();
    bool IsMul = (Opc == RISCV::SFPMUL);
    bool IsAdd = (Opc == RISCV::SFPADD);
    bool IsMad = (Opc == RISCV::SFPMAD);
    if (!IsMul && !IsAdd && !IsMad)
      continue;

    // Check each source operand for negation (multiply by L11 = -1.0)
    unsigned NumSrcOps = IsMad ? 3 : 2;
    for (unsigned OpOffset = 0; OpOffset < NumSrcOps; ++OpOffset) {
      unsigned OpIdx = 1 + OpOffset;  // Skip dest (operand 0)
      const MachineOperand &SrcOp = MI.getOperand(OpIdx);
      if (!SrcOp.isReg())
        continue;

      Register SrcReg = SrcOp.getReg();
      if (!hasOneUse(SrcReg, MRI))
        continue;

      // Check if the source is a negation: SFPMUL(x, L11(-1.0), L9, 0)
      MachineInstr *NegMI = getDefInst(SrcReg, MRI);
      if (!NegMI || NegMI->getOpcode() != RISCV::SFPMUL)
        continue;
      if (NegMI->getParent() != &MBB)
        continue;

      // Check that src_b of the MUL is L11 (the -1.0 constant)
      const MachineOperand &MulSrcB = NegMI->getOperand(2);
      if (!MulSrcB.isReg() || MulSrcB.getReg() != RISCV::L11)
        continue;

      // Check mod1 is 0 (pure negation, no other modifiers)
      if (NegMI->getOperand(4).getImm() != 0)
        continue;

      // Found a negation! Get the un-negated input
      Register OrigReg = NegMI->getOperand(1).getReg();

      // Determine which complement bit to toggle
      // For MUL: always COMPL_A (negate first operand)
      // For MAD: COMPL_A for op 0 or 1, COMPL_C for op 2
      // For ADD: COMPL_C for the addend
      unsigned ComplBit;
      if (IsMul || (IsMad && OpOffset <= 1)) {
        ComplBit = COMPL_A;
      } else {
        ComplBit = COMPL_C;
      }

      // Replace the negated operand with the original
      MI.getOperand(OpIdx).setReg(OrigReg);

      // Toggle the complement bit in mod1
      unsigned Mod1Idx = MI.getNumOperands() - 1;  // mod1 is always last
      unsigned OldMod = MI.getOperand(Mod1Idx).getImm();
      MI.getOperand(Mod1Idx).setImm(OldMod ^ ComplBit);

      // Remove the negation MUL
      NegMI->eraseFromParent();

      Changed = true;
      LLVM_DEBUG(dbgs() << "  Folded negation into mod1 bit " << ComplBit
                        << " for: " << MI);
      break;  // Restart operand scan for this instruction
    }
  }

  return Changed;
}

/// Redirect SFPLUTFP32 to read from SFPMUL output instead of original input.
/// Pattern: %1 = SFPMUL %in, C, L9 → half_in
///          %2 = SFPLUTFP32 %in, mod → LUT(in)
/// Transform: %2 = SFPLUTFP32 %1, mod → LUT(half_in)
/// This eliminates simultaneous liveness of %in and %1, matching GCC's
/// gelu register usage (LUT applied to half_in, not original input).
bool RISCVXttSFPUCombine::tryCombineLutInput(MachineBasicBlock &MBB) {
  MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
  bool Changed = false;

  for (MachineInstr &LutMI : MBB) {
    if (LutMI.getOpcode() != RISCV::SFPLUTFP32)
      continue;

    // SFPLUTFP32 operands: (outs dest), (ins lreg_c, mod1)
    // With tied constraint, operand 0 = dest (tied to lreg_c), operand 1 = lreg_c
    Register LutInput = LutMI.getOperand(1).getReg();
    if (!LutInput.isVirtual())
      continue;

    // Find an SFPMUL that uses the same input as src_a
    MachineInstr *MulMI = nullptr;
    for (auto &UseMI : MRI.use_nodbg_instructions(LutInput)) {
      if (UseMI.getOpcode() == RISCV::SFPMUL &&
          UseMI.getOperand(1).getReg() == LutInput) {
        MulMI = &UseMI;
        break;
      }
    }
    if (!MulMI) continue;

    // Verify SFPMUL output is a virtual register
    Register MulOut = MulMI->getOperand(0).getReg();
    if (!MulOut.isVirtual()) continue;

    // Redirect SFPLUTFP32 input from %in to %mul_out.
    // Only change lreg_c (operand 1). The tied output (operand 0) keeps
    // its own vreg — the tie constraint tells the RA to assign the same
    // physical register. This maintains SSA (one def per vreg).
    LutMI.getOperand(1).setReg(MulOut);

    LLVM_DEBUG(dbgs() << "  Combined LUT input: " << LutMI);
    Changed = true;
  }
  return Changed;
}

bool RISCVXttSFPUCombine::runOnMachineFunction(MachineFunction &MF) {
  STI = &MF.getSubtarget<RISCVSubtarget>();

  LLVM_DEBUG(dbgs() << getPassName() << " on " << MF.getName() << "\n");
  if (!STI->hasVendorXttSFPU())
    return false;

  TII = STI->getInstrInfo();

  bool Changed = false;

  // Run combining passes over each basic block
  for (MachineBasicBlock &MBB : MF) {
    // Negated operand folding (BH only, must run before MUL+ADD→MAD)
    Changed |= tryCombineNegatedOperands(MBB);

    // Redirect LUT input to MUL output (gelu pattern, before MUL+ADD→MAD)
    Changed |= tryCombineLutInput(MBB);

    // MUL+ADD → MAD (most impactful optimization)
    Changed |= tryCombineMulAdd(MBB);

    // LOADI + MUL/ADD → MULI/ADDI
    Changed |= tryCombineLoadiIntoImm(MBB);
  }

  return Changed;
}

INITIALIZE_PASS(RISCVXttSFPUCombine, DEBUG_TYPE,
                "RISC-V Tenstorrent SFPU Instruction Combining",
                false, false)

FunctionPass *llvm::createRISCVXttSFPUCombinePass() {
  return new RISCVXttSFPUCombine();
}
