//===-- RISCVXttSFPULiveness.cpp - SFPU CC Stack Liveness -----------------===//
//
// MachineFunctionPass implementing liveness analysis for the SFPU condition
// code (CC) stack. Determines when SFPU instructions need the "_lv" (live
// value) variant to preserve per-lane values in disabled lanes.
//
// The SFPU uses a per-lane push-down flag stack for SIMT-style divergent
// control flow:
//   v_if(cond)  → SFPPUSHC + SFPSETCC   (push & set predicate)
//   v_else      → SFPPOPC + SFPCOMPC + SFPPUSHC  (complement for else)
//   v_endif     → SFPPOPC              (pop, rejoin)
//
// When a register is "live" across a predicated region boundary (i.e., it was
// written before the v_if and is read after the v_endif), any write to that
// register inside the predicated region must use the "_lv" instruction variant.
// The "_lv" variant preserves the register value in lanes that are disabled by
// the predicate, rather than clobbering them.
//
// Implementation: replaces non-_lv instructions with _lv variants that have
// a tied $live_val = $lreg_dest constraint. This tells the register allocator
// to keep the destination register's old value available.
//
// Reference: ttsim-analysis/ERRATA.md Section 3 (Architectural Notes)
//            ttsim-analysis/FUNCTIONAL_UNITS.md Section 3.3
//            sfpi-gcc: rtl-rvtt-liveness.cc
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-xttsfpu-liveness"

namespace {

class RISCVXttSFPULiveness : public MachineFunctionPass {
public:
  static char ID;

  RISCVXttSFPULiveness() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "RISC-V Tenstorrent SFPU CC Stack Liveness Analysis";
  }

private:
  const RISCVSubtarget *STI = nullptr;
  const RISCVInstrInfo *TII = nullptr;

  int computeCCStackDelta(const MachineBasicBlock &MBB) const;
  bool isLiveAcrossPredication(const MachineInstr &MI, Register Reg,
                                const MachineBasicBlock &MBB) const;
  bool selectLiveValueVariant(MachineInstr &MI, int CCDepth,
                               MachineBasicBlock &MBB);

  /// Map non-_lv opcode to _lv opcode. Returns 0 if no _lv variant.
  unsigned getLVVariant(unsigned Opcode) const;
};

} // end anonymous namespace

char RISCVXttSFPULiveness::ID = 0;

unsigned RISCVXttSFPULiveness::getLVVariant(unsigned Opcode) const {
  switch (Opcode) {
  // Unary _lv
  case RISCV::SFPMOV:    return RISCV::SFPMOV_LV;
  case RISCV::SFPABS:    return RISCV::SFPABS_LV;
  case RISCV::SFPEXEXP:  return RISCV::SFPEXEXP_LV;
  case RISCV::SFPEXMAN:  return RISCV::SFPEXMAN_LV;
  case RISCV::SFPNOT:    return RISCV::SFPNOT_LV;
  case RISCV::SFPLZ:     return RISCV::SFPLZ_LV;
  case RISCV::SFPDIVP2:  return RISCV::SFPDIVP2_LV;
  case RISCV::SFPSETEXP: return RISCV::SFPSETEXP_LV;
  case RISCV::SFPSETMAN: return RISCV::SFPSETMAN_LV;
  case RISCV::SFPSETSGN: return RISCV::SFPSETSGN_LV;
  case RISCV::SFPCAST:   return RISCV::SFPCAST_LV;
  case RISCV::SFPSHFT2:  return RISCV::SFPSHFT2_LV;
  // 3-Op _lv
  case RISCV::SFPMAD:    return RISCV::SFPMAD_LV;
  case RISCV::SFPADD:    return RISCV::SFPADD_LV;
  case RISCV::SFPMUL:    return RISCV::SFPMUL_LV;
  // Load _lv
  case RISCV::SFPLOAD_BH: return RISCV::SFPLOAD_BH_LV;
  case RISCV::SFPLOAD_WH: return RISCV::SFPLOAD_WH_LV;
  // BH-only _lv
  case RISCV::SFPARECIP: return RISCV::SFPARECIP_LV;
  case RISCV::SFPMUL24:  return RISCV::SFPMUL24_LV;
  default: return 0;
  }
}

int RISCVXttSFPULiveness::computeCCStackDelta(
    const MachineBasicBlock &MBB) const {
  int Delta = 0;
  for (const MachineInstr &MI : MBB) {
    switch (MI.getOpcode()) {
    case RISCV::SFPPUSHC:
      Delta++;
      break;
    case RISCV::SFPPOPC:
      Delta--;
      break;
    default:
      break;
    }
  }
  return Delta;
}

bool RISCVXttSFPULiveness::isLiveAcrossPredication(
    const MachineInstr &MI, Register Reg,
    const MachineBasicBlock &MBB) const {
  // Walk backwards from MI to find the most recent SFPPUSHC.
  // If Reg was defined before that SFPPUSHC, it's live across the boundary.
  bool FoundPush = false;
  for (auto I = MachineBasicBlock::const_reverse_iterator(MI),
            E = MBB.rend();
       I != E; ++I) {
    if (I->getOpcode() == RISCV::SFPPUSHC) {
      FoundPush = true;
      break;
    }
    // If Reg is defined between MI and the PUSHC, it's not live-across
    for (const MachineOperand &MO : I->defs()) {
      if (MO.isReg() && MO.getReg() == Reg)
        return false;
    }
  }

  // If we found a PUSHC and didn't find a definition of Reg between it and MI,
  // then Reg is live across the predication boundary.
  return FoundPush;
}

bool RISCVXttSFPULiveness::selectLiveValueVariant(MachineInstr &MI,
                                                    int CCDepth,
                                                    MachineBasicBlock &MBB) {
  if (CCDepth <= 0)
    return false;

  if (MI.getNumDefs() == 0)
    return false;

  const MachineOperand &DestOp = MI.getOperand(0);
  if (!DestOp.isReg())
    return false;

  Register DestReg = DestOp.getReg();

  if (!isLiveAcrossPredication(MI, DestReg, MBB))
    return false;

  unsigned LVOpc = getLVVariant(MI.getOpcode());
  if (!LVOpc) {
    // No _lv variant — fall back to mod1 bit 3 (encoding-level flag).
    unsigned Mod1Idx = MI.getNumOperands() - 1;
    MachineOperand &Mod1Op = MI.getOperand(Mod1Idx);
    if (Mod1Op.isImm()) {
      constexpr unsigned MOD1_LV_FLAG = 0x8;
      Mod1Op.setImm(Mod1Op.getImm() | MOD1_LV_FLAG);
      LLVM_DEBUG(dbgs() << "  Set _lv flag (mod1 |= 0x8, no _lv variant) for: "
                        << MI);
      return true;
    }
    return false;
  }

  // Build the _lv variant: insert $live_val operand (tied to dest).
  // _lv variants have dest as operand 0 and live_val as operand 1,
  // then the remaining operands from the original instruction.
  DebugLoc DL = MI.getDebugLoc();
  MachineInstrBuilder MIB = BuildMI(MBB, MI, DL, TII->get(LVOpc), DestReg)
                                .addReg(DestReg);  // live_val = dest (tied)

  // Copy remaining operands (skip operand 0 which is the dest)
  for (unsigned I = 1, E = MI.getNumOperands(); I < E; ++I) {
    MachineOperand &Op = MI.getOperand(I);
    if (Op.isReg())
      MIB.addReg(Op.getReg(), getRegState(Op));
    else if (Op.isImm())
      MIB.addImm(Op.getImm());
    else
      MIB.add(Op);
  }

  LLVM_DEBUG(dbgs() << "  Replaced with _lv variant: " << *MIB << "\n");

  MI.eraseFromParent();
  return true;
}

bool RISCVXttSFPULiveness::runOnMachineFunction(MachineFunction &MF) {
  STI = &MF.getSubtarget<RISCVSubtarget>();

  if (!STI->hasVendorXttSFPU())
    return false;

  TII = STI->getInstrInfo();

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    int CCDepth = 0;

    for (auto MBBI = MBB.begin(), MBBE = MBB.end(); MBBI != MBBE; ) {
      MachineInstr &MI = *MBBI++;

      // Track CC stack depth
      switch (MI.getOpcode()) {
      case RISCV::SFPPUSHC:
        CCDepth++;
        break;
      case RISCV::SFPPOPC:
        CCDepth--;
        if (CCDepth < 0)
          CCDepth = 0;
        break;
      default:
        break;
      }

      Changed |= selectLiveValueVariant(MI, CCDepth, MBB);
    }
  }

  return Changed;
}

INITIALIZE_PASS(RISCVXttSFPULiveness, DEBUG_TYPE,
                "RISC-V Tenstorrent SFPU CC Stack Liveness Analysis",
                false, false)

FunctionPass *llvm::createRISCVXttSFPULivenessPass() {
  return new RISCVXttSFPULiveness();
}
