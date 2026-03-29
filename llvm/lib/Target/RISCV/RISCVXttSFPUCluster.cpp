//===-- RISCVXttSFPUCluster.cpp - SFPU TTI Instruction Clustering ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// Post-RA pass that groups SFPU instructions together within basic blocks.
//
// The Baby RISC-V frontend fuses up to 4 adjacent TTI (Tensix Thread
// Instructions) into a single wide fetch — 4x dispatch throughput. Scalar
// RISC-V instructions interleaved between SFPU instructions break this
// fusion. This pass hoists/sinks independent scalar instructions out of
// SFPU clusters to maximize fusion opportunities.
//
// Algorithm:
//   For each basic block:
//   1. Identify SFPU instruction regions
//   2. For each scalar instruction between two SFPU instructions, check if
//      it can be safely moved before the SFPU cluster (no data dependencies)
//   3. Hoist movable scalar instructions above the cluster
//
// Safety:
//   - Only reorder within a single basic block
//   - Scalar instructions with side effects or memory ops are not moved
//   - Data dependencies (def-use chains) are respected
//   - SFPU instructions are never reordered relative to each other
//
// Reference: ttsim-analysis/PIPELINE.md §2.7 (TTI fetch fusion)
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "RISCVXttSFPUUtil.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-xttsfpu-cluster"

namespace {

class RISCVXttSFPUCluster : public MachineFunctionPass {
public:
  static char ID;

  RISCVXttSFPUCluster() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "RISC-V Tenstorrent SFPU TTI Instruction Clustering";
  }

private:
  const RISCVSubtarget *STI = nullptr;
  const RISCVInstrInfo *TII = nullptr;

  bool canHoist(const MachineInstr &MI,
                MachineBasicBlock::iterator InsertBefore,
                const MachineBasicBlock &MBB) const;
  bool clusterBlock(MachineBasicBlock &MBB);
};

} // end anonymous namespace

char RISCVXttSFPUCluster::ID = 0;

// RISCVXttSFPU::isSFPUInstr() is now in RISCVXttSFPUUtil.h (shared with Errata pass).

/// Check if MI can be safely hoisted above InsertBefore without violating
/// data dependencies. Only considers register deps within the same block.
bool RISCVXttSFPUCluster::canHoist(const MachineInstr &MI,
                                     MachineBasicBlock::iterator InsertBefore,
                                     const MachineBasicBlock &MBB) const {
  // Don't move instructions with side effects or memory operations
  if (MI.hasUnmodeledSideEffects() || MI.isCall() || MI.isTerminator())
    return false;
  if (MI.mayLoad() || MI.mayStore())
    return false;

  // Collect registers defined and used by MI
  SmallVector<Register, 4> Defs, Uses;
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.getReg().isValid())
      continue;
    if (MO.isDef())
      Defs.push_back(MO.getReg());
    if (MO.isUse())
      Uses.push_back(MO.getReg());
  }

  // Check all instructions between InsertBefore and MI for conflicts
  for (auto I = MachineBasicBlock::const_iterator(InsertBefore),
            E = MachineBasicBlock::const_iterator(MI);
       I != E; ++I) {
    // If any instruction between defines a register MI uses → can't hoist
    for (const MachineOperand &MO : I->operands()) {
      if (!MO.isReg() || !MO.getReg().isValid())
        continue;
      if (MO.isDef()) {
        for (Register U : Uses)
          if (MO.getReg() == U)
            return false;
      }
      // If any instruction between uses a register MI defines → can't hoist
      if (MO.isUse()) {
        for (Register D : Defs)
          if (MO.getReg() == D)
            return false;
      }
    }
  }

  return true;
}

bool RISCVXttSFPUCluster::clusterBlock(MachineBasicBlock &MBB) {
  bool Changed = false;

  // Find SFPU clusters and hoist independent scalar instructions above them.
  // Work from the beginning of the block forward.
  for (auto MBBI = MBB.begin(), MBBE = MBB.end(); MBBI != MBBE; ) {
    // Find start of an SFPU cluster
    if (!RISCVXttSFPU::isSFPUInstr(*MBBI)) {
      ++MBBI;
      continue;
    }

    // MBBI points to the first SFPU instruction. Remember the insert point.
    auto ClusterStart = MBBI;

    // Scan forward: collect the cluster (SFPU + interleaved scalar)
    auto ClusterEnd = MBBI;
    unsigned SFPUCount = 0;
    while (ClusterEnd != MBBE) {
      if (RISCVXttSFPU::isSFPUInstr(*ClusterEnd)) {
        SFPUCount++;
        ++ClusterEnd;
      } else if (ClusterEnd->isDebugInstr()) {
        ++ClusterEnd;
      } else {
        // Scalar instruction — check if there's more SFPU ahead (within 4)
        auto Lookahead = ClusterEnd;
        bool MoreSFPU = false;
        unsigned ScalarGap = 0;
        while (++Lookahead != MBBE && ScalarGap < 4) {
          if (RISCVXttSFPU::isSFPUInstr(*Lookahead)) {
            MoreSFPU = true;
            break;
          }
          ScalarGap++;
        }
        if (MoreSFPU && SFPUCount < 8)
          ++ClusterEnd;  // Include scalar in cluster range
        else
          break;  // End of cluster
      }
    }

    // Now try to hoist scalar instructions within [ClusterStart, ClusterEnd)
    // above the cluster start.
    for (auto I = std::next(MachineBasicBlock::iterator(ClusterStart));
         I != ClusterEnd; ) {
      if (RISCVXttSFPU::isSFPUInstr(*I) || I->isDebugInstr()) {
        ++I;
        continue;
      }
      // Scalar instruction in the middle of SFPU cluster
      if (canHoist(*I, ClusterStart, MBB)) {
        auto ToMove = I++;
        MBB.splice(ClusterStart, &MBB, ToMove);
        Changed = true;
        LLVM_DEBUG(dbgs() << "  Hoisted scalar above SFPU cluster: "
                          << *ToMove);
      } else {
        ++I;
      }
    }

    MBBI = ClusterEnd;
  }

  return Changed;
}

bool RISCVXttSFPUCluster::runOnMachineFunction(MachineFunction &MF) {
  STI = &MF.getSubtarget<RISCVSubtarget>();

  LLVM_DEBUG(dbgs() << getPassName() << " on " << MF.getName() << "\n");
  if (!STI->hasVendorXttSFPU())
    return false;

  TII = STI->getInstrInfo();

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    Changed |= clusterBlock(MBB);

  return Changed;
}

INITIALIZE_PASS(RISCVXttSFPUCluster, DEBUG_TYPE,
                "RISC-V Tenstorrent SFPU TTI Instruction Clustering",
                false, false)

FunctionPass *llvm::createRISCVXttSFPUClusterPass() {
  return new RISCVXttSFPUCluster();
}
