//===-- RISCVXttSFPULiveness.cpp - SFPU CC Stack Liveness -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
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
// The pass uses global CC stack depth analysis across the CFG to handle
// predicated regions that span multiple basic blocks (common at -O3).
//
// Reference: ttsim-analysis/ERRATA.md Section 3 (Architectural Notes)
//            ttsim-analysis/FUNCTIONAL_UNITS.md Section 3.3
//            sfpi-gcc: rtl-rvtt-liveness.cc
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include <queue>

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
  MachineRegisterInfo *MRI = nullptr;

  /// CC stack depth at entry to each basic block, keyed by MBB number.
  DenseMap<unsigned, int> BBEntryCCDepth;

  /// Compute the net CC stack depth change for a basic block.
  int computeCCStackDelta(const MachineBasicBlock &MBB) const;

  /// Compute CC stack entry depth for every block via forward dataflow.
  void computeGlobalCCDepths(MachineFunction &MF);

  /// Get the CC depth at a specific instruction within its block.
  int getCCDepthAtInstr(const MachineInstr &MI) const;

  /// Determine if Reg needs _lv preservation at instruction MI.
  /// Uses SSA def-use analysis: compares CC depth at definition vs use.
  bool isLiveAcrossPredication(const MachineInstr &MI, Register Reg,
                                int CCDepthAtMI) const;

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

void RISCVXttSFPULiveness::computeGlobalCCDepths(MachineFunction &MF) {
  BBEntryCCDepth.clear();

  // Initialize all blocks as unvisited (-1 sentinel).
  DenseMap<unsigned, int> BBExitCCDepth;
  for (MachineBasicBlock &MBB : MF) {
    BBEntryCCDepth[MBB.getNumber()] = -1;
    BBExitCCDepth[MBB.getNumber()] = -1;
  }

  // Entry block starts at depth 0.
  MachineBasicBlock &EntryBB = MF.front();
  BBEntryCCDepth[EntryBB.getNumber()] = 0;
  int Delta = computeCCStackDelta(EntryBB);
  BBExitCCDepth[EntryBB.getNumber()] = Delta;

  // Worklist-driven forward propagation.
  // The CC stack is always balanced (each SFPPUSHC has a matching SFPPOPC),
  // so all predecessors of a block agree on the entry depth. A single forward
  // pass suffices — no fixpoint iteration needed.
  std::queue<MachineBasicBlock *> WorkList;
  for (MachineBasicBlock *Succ : EntryBB.successors())
    WorkList.push(Succ);

  while (!WorkList.empty()) {
    MachineBasicBlock *MBB = WorkList.front();
    WorkList.pop();

    // Compute entry depth from predecessors.
    int EntryDepth = 0;
    bool HasComputedPred = false;
    for (MachineBasicBlock *Pred : MBB->predecessors()) {
      int PredExit = BBExitCCDepth[Pred->getNumber()];
      if (PredExit < 0)
        continue;  // Predecessor not yet processed.
      if (!HasComputedPred) {
        EntryDepth = PredExit;
        HasComputedPred = true;
      } else {
        // All predecessors should agree (balanced CC stack).
        // Use max as conservative fallback for safety.
        EntryDepth = std::max(EntryDepth, PredExit);
      }
    }

    if (!HasComputedPred)
      continue;  // No computed predecessors yet; will be revisited via worklist.

    // Skip if already computed with this value.
    if (BBEntryCCDepth[MBB->getNumber()] == EntryDepth)
      continue;

    BBEntryCCDepth[MBB->getNumber()] = EntryDepth;
    int ExitDepth = EntryDepth + computeCCStackDelta(*MBB);
    assert(ExitDepth >= 0 && "CC stack underflow");
    BBExitCCDepth[MBB->getNumber()] = ExitDepth;

    for (MachineBasicBlock *Succ : MBB->successors())
      WorkList.push(Succ);
  }

  LLVM_DEBUG({
    dbgs() << "  CC stack depth map:\n";
    for (MachineBasicBlock &MBB : MF) {
      int Entry = BBEntryCCDepth[MBB.getNumber()];
      dbgs() << "    " << printMBBReference(MBB) << ": entry="
             << (Entry >= 0 ? std::to_string(Entry) : "?") << "\n";
    }
  });
}

int RISCVXttSFPULiveness::getCCDepthAtInstr(const MachineInstr &MI) const {
  const MachineBasicBlock *MBB = MI.getParent();
  auto It = BBEntryCCDepth.find(MBB->getNumber());
  if (It == BBEntryCCDepth.end() || It->second < 0)
    return 0;  // Unknown block (unreachable); treat as depth 0.
  int Depth = It->second;

  for (const MachineInstr &Cur : *MBB) {
    if (&Cur == &MI)
      return Depth;
    if (Cur.getOpcode() == RISCV::SFPPUSHC)
      Depth++;
    else if (Cur.getOpcode() == RISCV::SFPPOPC)
      Depth--;
  }
  llvm_unreachable("Instruction not found in its parent block");
}

bool RISCVXttSFPULiveness::isLiveAcrossPredication(
    const MachineInstr &MI, Register Reg, int CCDepthAtMI) const {
  // Only virtual registers have single definitions in SSA form.
  if (!Reg.isVirtual())
    return false;

  // Find the unique definition of this register (SSA guarantee).
  MachineInstr *DefMI = MRI->getVRegDef(Reg);
  if (!DefMI)
    return false;

  // Compute CC depth at the definition point.
  int DefCCDepth = getCCDepthAtInstr(*DefMI);

  // If the current instruction is at a greater CC depth than the definition,
  // the value was defined outside (before) a SFPPUSHC that encloses MI.
  // The write inside the predicated region must use _lv to preserve
  // the value in disabled lanes.
  //
  // Examples:
  //   depth=0: %x = SFPMOV ...       (DefCCDepth = 0)
  //   depth=0: SFPPUSHC              (depth -> 1)
  //   depth=1: %x = SFPMAD %x, ...  (CCDepthAtMI = 1 > 0, needs _lv)
  //
  // Cross-block:
  //   BB0 depth=0: %x = SFPMOV; SFPPUSHC  (exits at depth=1)
  //   BB1 entry=1: %x = SFPMAD %x, ...    (CCDepthAtMI=1, DefCCDepth=0, needs _lv)
  //
  // Same depth (no _lv needed):
  //   depth=1: %y = SFPLOADI         (DefCCDepth = 1)
  //   depth=1: %z = SFPMAD %y, ...   (CCDepthAtMI = 1 == 1, no _lv)
  return CCDepthAtMI > DefCCDepth;
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

  if (!isLiveAcrossPredication(MI, DestReg, CCDepth))
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

  LLVM_DEBUG(dbgs() << getPassName() << " on " << MF.getName() << "\n");
  if (!STI->hasVendorXttSFPU())
    return false;

  TII = STI->getInstrInfo();
  MRI = &MF.getRegInfo();

  // Phase 1: Compute global CC stack depths across the CFG.
  computeGlobalCCDepths(MF);

  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    // Start from the precomputed entry depth for this block.
    int CCDepth = BBEntryCCDepth.lookup(MBB.getNumber());
    if (CCDepth < 0)
      CCDepth = 0;  // Unreachable block; treat as depth 0.

    for (auto MBBI = MBB.begin(), MBBE = MBB.end(); MBBI != MBBE; ) {
      MachineInstr &MI = *MBBI++;

      // Track CC stack depth within the block.
      switch (MI.getOpcode()) {
      case RISCV::SFPPUSHC:
        CCDepth++;
        break;
      case RISCV::SFPPOPC:
        CCDepth--;
        if (CCDepth < 0) {
          LLVM_DEBUG(dbgs() << "  WARNING: unbalanced SFPPOPC in "
                            << MF.getName() << "\n");
          CCDepth = 0;
        }
        break;
      default:
        break;
      }

      Changed |= selectLiveValueVariant(MI, CCDepth, MBB);
    }
  }

  BBEntryCCDepth.clear();
  return Changed;
}

INITIALIZE_PASS(RISCVXttSFPULiveness, DEBUG_TYPE,
                "RISC-V Tenstorrent SFPU CC Stack Liveness Analysis",
                false, false)

FunctionPass *llvm::createRISCVXttSFPULivenessPass() {
  return new RISCVXttSFPULiveness();
}
