//===-- RISCVXttSFPUSynth.cpp - SFPU Immediate Synthesis ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// Custom lowering pass for SFPU immediate values that exceed field widths.
// When an immediate operand doesn't fit in the instruction encoding, this
// pass expands it to a SFPLOADI + register-form instruction sequence.
//
// Per-instruction immediate field widths:
//   SFPLOAD/SFPSTORE addr:  14 bits (WH) / 13 bits (BH)
//   SFPLOADI imm:           16 bits
//   SFPMULI/SFPADDI imm:    16 bits
//   SFPIADD/SFPDIVP2 imm:   12 bits
//   SFPSHFT imm:            12 bits
//
// When an immediate exceeds these widths, the pass:
// 1. Loads the full constant via SFPLOADI (16-bit) or SFPLOADI+SFPIADD
// 2. Replaces the immediate-form instruction with its register-form equivalent
//
// Reference: ttsim-analysis/ERRATA.md C-007 (SFPLOADI modes)
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-xttsfpu-synth"

namespace {

class RISCVXttSFPUSynth : public MachineFunctionPass {
public:
  static char ID;

  RISCVXttSFPUSynth() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "RISC-V Tenstorrent SFPU Immediate Synthesis";
  }

private:
  const RISCVSubtarget *STI = nullptr;
  const RISCVInstrInfo *TII = nullptr;

  /// Get the maximum immediate value for a given instruction.
  /// Returns 0 if the instruction does not have an immediate field.
  unsigned getImmFieldWidth(unsigned Opcode) const;

  /// Expand an out-of-range immediate operand.
  bool expandImmediate(MachineInstr &MI, unsigned ImmOpIdx,
                       unsigned FieldWidth);

  /// Get the register-form variant for an immediate-form instruction.
  /// Returns 0 if no register-form exists.
  unsigned getRegFormVariant(unsigned Opcode) const;

  /// Substitute immediate-form instruction with register-form using ScratchReg.
  bool substituteRegForm(MachineInstr &MI, Register ScratchReg);
};

} // end anonymous namespace

char RISCVXttSFPUSynth::ID = 0;

unsigned RISCVXttSFPUSynth::getImmFieldWidth(unsigned Opcode) const {
  switch (Opcode) {
  // 12-bit immediate (Standard Unary format)
  case RISCV::SFPDIVP2:
  case RISCV::SFPEXEXP:
  case RISCV::SFPEXMAN:
  case RISCV::SFPIADD:
  case RISCV::SFPSHFT:
  case RISCV::SFPSETCC:
  case RISCV::SFPMOV:
  case RISCV::SFPABS:
  case RISCV::SFPAND:
  case RISCV::SFPOR:
  case RISCV::SFPNOT:
  case RISCV::SFPLZ:
  case RISCV::SFPSETEXP:
  case RISCV::SFPSETMAN:
  case RISCV::SFPSETSGN:
  case RISCV::SFPSWAP:
  case RISCV::SFPSHFT2:
  case RISCV::SFPCAST:
    return 12;

  // 16-bit immediate (Imm16 format)
  case RISCV::SFPLOADI:
  case RISCV::SFPMULI:
  case RISCV::SFPADDI:
  case RISCV::SFPCONFIG:
    return 16;

  // 13-bit address (BH Load/Store format)
  case RISCV::SFPLOAD_BH:
  case RISCV::SFPSTORE_BH:
  case RISCV::SFPLOADMACRO_BH:
    return 13;

  // 14-bit address (WH Load/Store format)
  case RISCV::SFPLOAD_WH:
  case RISCV::SFPSTORE_WH:
  case RISCV::SFPLOADMACRO_WH:
    return 14;

  // SFPLUT uses LUT format (16-bit dest_reg_addr, no addr_mode)
  case RISCV::SFPLUT:
    return 16;

  // 8-bit immediate (StochRnd format)
  case RISCV::SFP_STOCH_RND:
    return 8;

  default:
    return 0;  // No immediate field
  }
}

bool RISCVXttSFPUSynth::expandImmediate(MachineInstr &MI, unsigned ImmOpIdx,
                                          unsigned FieldWidth) {
  MachineOperand &ImmOp = MI.getOperand(ImmOpIdx);
  if (!ImmOp.isImm())
    return false;

  int64_t ImmVal = ImmOp.getImm();
  int64_t MaxVal = (1LL << FieldWidth) - 1;

  if (ImmVal >= 0 && ImmVal <= MaxVal)
    return false;  // Fits in field, no expansion needed

  // For values that don't fit:
  // 1. If <= 16 bits: use SFPLOADI to load into a temp register
  // 2. If > 16 bits: use SFPLOADI (upper) + SFPIADD (lower)

  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();

  if (ImmVal >= 0 && ImmVal <= 0xFFFF) {
    // Single SFPLOADI can handle it
    // Load immediate into a scratch register (L7 by convention)
    BuildMI(MBB, MI, DL, TII->get(RISCV::SFPLOADI))
        .addReg(RISCV::L7, RegState::Define)
        .addImm(ImmVal)
        .addImm(0);  // mod1 = FLOATB

    LLVM_DEBUG(dbgs() << "  Synthesized imm " << ImmVal
                      << " via SFPLOADI for: " << MI);
    // Replace instruction with register-form variant using L7
    if (substituteRegForm(MI, RISCV::L7))
      return true;
    // No register-form variant — cannot silently truncate, report error.
    {
      std::string Msg;
      raw_string_ostream OS(Msg);
      OS << "SFPU immediate " << ImmVal << " exceeds "
         << FieldWidth << "-bit field; no register-form exists";
      MI.emitError(OS.str());
    }
    return false;
  } else {
    // Value > 16 bits: SFPLOADI UPPER (bits [31:16]) + SFPLOADI LOWER (bits [15:0])
    // UPPER mode (mod0=8): loads imm16 into bits [31:16], zeros [15:0]
    // LOWER mode (mod0=10): OR's imm16 into bits [15:0], preserving [31:16]
    unsigned Upper = (ImmVal >> 16) & 0xFFFF;
    unsigned Lower = ImmVal & 0xFFFF;

    BuildMI(MBB, MI, DL, TII->get(RISCV::SFPLOADI))
        .addReg(RISCV::L7, RegState::Define)
        .addImm(Upper)
        .addImm(8);  // mod0 = UPPER

    if (Lower != 0) {
      BuildMI(MBB, MI, DL, TII->get(RISCV::SFPLOADI))
          .addReg(RISCV::L7, RegState::Define)
          .addImm(Lower)
          .addImm(10);  // mod0 = LOWER
    }

    LLVM_DEBUG(dbgs() << "  Synthesized wide imm " << ImmVal
                      << " via SFPLOADI UPPER+LOWER for: " << MI);
    // Replace instruction with register-form variant using L7
    if (substituteRegForm(MI, RISCV::L7))
      return true;
    {
      std::string Msg;
      raw_string_ostream OS(Msg);
      OS << "SFPU wide immediate " << ImmVal << " exceeds "
         << FieldWidth << "-bit field; no register-form exists";
      MI.emitError(OS.str());
    }
    return false;
  }

  return true;
}

unsigned RISCVXttSFPUSynth::getRegFormVariant(unsigned Opcode) const {
  switch (Opcode) {
  case RISCV::SFPMULI: return RISCV::SFPMUL;
  case RISCV::SFPADDI: return RISCV::SFPADD;
  default: return 0;  // No register-form variant
  }
}

bool RISCVXttSFPUSynth::substituteRegForm(MachineInstr &MI,
                                            Register ScratchReg) {
  // Special case: SFPSETCC — replace immediate with the loaded register
  // as the source, zeroing the immediate. The comparison becomes:
  //   sfpsetcc 0, L7, lreg_dest, mod1  (compare lreg_c=L7 against imm=0)
  // This works because sfpsetcc compares src against imm, and we've
  // loaded the constant into L7.
  if (MI.getOpcode() == RISCV::SFPSETCC) {
    MachineBasicBlock &MBB = *MI.getParent();
    DebugLoc DL = MI.getDebugLoc();
    // SFPSETCC format: (imm12, lreg_c, lreg_dest, mod1, chain)
    // Replace: imm12 with 0, lreg_c with ScratchReg
    Register SrcReg = MI.getOperand(1).getReg();
    unsigned LregDest = MI.getOperand(2).getImm();
    unsigned Mod1 = MI.getOperand(3).getImm();

    // Load the original src into another temp if needed, then compare
    // scratch (constant) against src. But sfpsetcc compares lreg_c against
    // imm12, not the other way around. We need to swap: use SFPIADD to
    // subtract and check sign, or use a different approach.
    //
    // Actually, sfpsetcc sets CC based on (lreg_c <op> imm12) where <op>
    // depends on mod1. If we want (src <op> constant), we can do:
    // sfpsetcc 0, src, dest, mod1  — but this compares src against 0, not
    // the constant. The constant was supposed to be in imm12.
    //
    // Alternative: load constant into L7, then use sfpiadd to compute
    // (src - L7) and compare that against 0. But this changes semantics.
    //
    // Simpler: just use sfpsetcc with the loaded value as lreg_c and set
    // imm12=0 with the right mod1 adjustment. The comparison modes for
    // float compare are: LT, GE etc. Comparing L7 (=constant) against
    // 0 with src in lreg_c doesn't make sense.
    //
    // The real answer: for float comparisons with large constants, the
    // constant should be loaded into a register and the comparison should
    // use a register-register form. But SFPSETCC only has reg-imm form.
    //
    // Best approach: swap the operands — put the loaded constant as lreg_c
    // and the original source as the immediate (0), then flip the comparison
    // direction in mod1. OR: just use sfpxfcmpv (vector-vector compare)
    // pattern from the sfpi_builtins.h which maps to sfpsetcc(a, 0, mod).
    //
    // For now: swap src ↔ constant. sfpsetcc(0, L7, dest, mod1) compares
    // L7 against 0 — not what we want. We need (original_src <op> constant).
    //
    // Actually the sfpi compat header defines:
    //   sfpxfcmps(v, f, mod) → sfpsetcc(v, f, mod), 0
    // So sfpsetcc args are: (imm=f, src=v, dest=0, mod1=mod)
    // The hardware does: compare src(=v) against imm(=f) with mode mod1.
    //
    // To use register form: sfpsetcc(imm=0, src=v, dest=0, mod1=mod)
    // But this compares v against 0, not against the constant.
    //
    // The only way: load constant into a register, then subtract and compare
    // against 0. This requires SFPIADD or similar. Too complex for now.
    // Fall through to the error.
  }

  unsigned RegOpc = getRegFormVariant(MI.getOpcode());
  if (!RegOpc)
    return false;

  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();

  // SFPMULI/SFPADDI format: (dest, imm16, mod1)
  // SFPMUL/SFPADD format:   (dest, src_a, src_b, src_c, mod1)
  Register DestReg = MI.getOperand(0).getReg();
  unsigned Mod1 = 0;
  for (unsigned I = MI.getNumOperands(); I > 0; --I) {
    if (MI.getOperand(I - 1).isImm()) {
      Mod1 = MI.getOperand(I - 1).getImm();
      break;
    }
  }

  BuildMI(MBB, MI, DL, TII->get(RegOpc))
      .addReg(DestReg, RegState::Define)
      .addReg(ScratchReg)
      .addReg(DestReg)
      .addReg(RISCV::L9)
      .addImm(Mod1);

  LLVM_DEBUG(dbgs() << "  Substituted register-form: " << *std::prev(
                 MachineBasicBlock::iterator(MI)));

  MI.eraseFromParent();
  return true;
}

bool RISCVXttSFPUSynth::runOnMachineFunction(MachineFunction &MF) {
  STI = &MF.getSubtarget<RISCVSubtarget>();

  LLVM_DEBUG(dbgs() << getPassName() << " on " << MF.getName() << "\n");
  if (!STI->hasVendorXttSFPU())
    return false;

  TII = STI->getInstrInfo();

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto MBBI = MBB.begin(), MBBE = MBB.end(); MBBI != MBBE; ++MBBI) {
      unsigned FieldWidth = getImmFieldWidth(MBBI->getOpcode());
      if (FieldWidth == 0)
        continue;

      // Find the immediate operand (varies by format)
      for (unsigned I = 0, E = MBBI->getNumOperands(); I < E; ++I) {
        if (MBBI->getOperand(I).isImm()) {
          Changed |= expandImmediate(*MBBI, I, FieldWidth);
          break;  // Only one immediate per instruction
        }
      }
    }
  }

  return Changed;
}

INITIALIZE_PASS(RISCVXttSFPUSynth, DEBUG_TYPE,
                "RISC-V Tenstorrent SFPU Immediate Synthesis", false, false)

FunctionPass *llvm::createRISCVXttSFPUSynthPass() {
  return new RISCVXttSFPUSynth();
}
