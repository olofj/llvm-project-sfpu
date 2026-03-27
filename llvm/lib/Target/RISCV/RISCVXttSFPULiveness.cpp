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
// This pass tracks:
// 1. CC stack depth (push/pop balance) to know when we're in predicated code
// 2. Register liveness across predicated region boundaries
// 3. Which instructions need "_lv" selection
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

  /// Track CC stack depth through a basic block.
  /// Returns the net change in stack depth (positive = more pushes than pops).
  int computeCCStackDelta(const MachineBasicBlock &MBB) const;

  /// Check if a register is live-in to the predicated region containing MI.
  bool isLiveAcrossPredication(const MachineInstr &MI, Register Reg,
                                const MachineBasicBlock &MBB) const;

  /// Select "_lv" variant for an instruction if needed.
  bool selectLiveValueVariant(MachineInstr &MI, int CCDepth);
};

} // end anonymous namespace

char RISCVXttSFPULiveness::ID = 0;

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
                                                    int CCDepth) {
  // Only need _lv variants when inside predicated code (CC depth > 0)
  if (CCDepth <= 0)
    return false;

  // Only SFPU instructions that write a destination register need _lv
  if (MI.getNumDefs() == 0)
    return false;

  const MachineOperand &DestOp = MI.getOperand(0);
  if (!DestOp.isReg())
    return false;

  Register DestReg = DestOp.getReg();

  // Check if destination is live across the predication boundary
  if (!isLiveAcrossPredication(MI, DestReg, *MI.getParent()))
    return false;

  // Replace the instruction opcode with its _lv variant.
  // _lv variants have an extra "live" operand (operand 0) that specifies the
  // register value to preserve in disabled lanes.
  //
  // The _lv instruction definitions are in RISCVInstrInfoXttSFPU.td:
  //   SFPMOV_LV, SFPMAD_LV, SFPMUL_LV, SFPADD_LV, SFPCAST_LV, etc.
  //
  // Since _lv variants have a different operand count (extra live-in operand),
  // we cannot simply change the opcode. Instead, we set the mod1 bit 3
  // (MOD1_LV_FLAG = 0x8) to signal live-value preservation in the encoding.
  // This matches the hardware behavior: bit 3 of mod1 tells the SFPU to
  // merge the result with the existing register value per-lane.
  unsigned Mod1Idx = MI.getNumOperands() - 1;
  MachineOperand &Mod1Op = MI.getOperand(Mod1Idx);
  if (Mod1Op.isImm()) {
    unsigned OldMod = Mod1Op.getImm();
    constexpr unsigned MOD1_LV_FLAG = 0x8;
    Mod1Op.setImm(OldMod | MOD1_LV_FLAG);
    LLVM_DEBUG(dbgs() << "  Set _lv flag (mod1 |= 0x8) for: " << MI);
    return true;
  }

  LLVM_DEBUG(dbgs() << "  Could not set _lv flag (no imm mod1) for: " << MI);
  return false;
}

bool RISCVXttSFPULiveness::runOnMachineFunction(MachineFunction &MF) {
  STI = &MF.getSubtarget<RISCVSubtarget>();

  if (!STI->hasVendorXttSFPU())
    return false;

  TII = STI->getInstrInfo();

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    int CCDepth = 0;

    for (MachineInstr &MI : MBB) {
      // Track CC stack depth
      switch (MI.getOpcode()) {
      case RISCV::SFPPUSHC:
        CCDepth++;
        break;
      case RISCV::SFPPOPC:
        CCDepth--;
        if (CCDepth < 0)
          CCDepth = 0;  // Unbalanced pop — reset
        break;
      default:
        break;
      }

      // Select _lv variants for instructions inside predicated regions
      Changed |= selectLiveValueVariant(MI, CCDepth);
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
