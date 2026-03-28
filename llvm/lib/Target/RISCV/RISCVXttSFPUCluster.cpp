//===-- RISCVXttSFPUCluster.cpp - SFPU TTI Instruction Clustering ---------===//
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

  bool isSFPUInstr(const MachineInstr &MI) const;
  bool canHoist(const MachineInstr &MI,
                MachineBasicBlock::iterator InsertBefore,
                const MachineBasicBlock &MBB) const;
  bool clusterBlock(MachineBasicBlock &MBB);
};

} // end anonymous namespace

char RISCVXttSFPUCluster::ID = 0;

bool RISCVXttSFPUCluster::isSFPUInstr(const MachineInstr &MI) const {
  unsigned Opc = MI.getOpcode();
  // SFPU opcodes are in the Tensix custom encoding space.
  // Check by opcode ranges defined in RISCVInstrInfoXttSFPU.td.
  switch (Opc) {
  case RISCV::SFPLOAD_BH: case RISCV::SFPLOAD_WH:
  case RISCV::SFPLOADI:
  case RISCV::SFPSTORE_BH: case RISCV::SFPSTORE_WH:
  case RISCV::SFPLUT: case RISCV::SFPLUTFP32:
  case RISCV::SFPLOADMACRO_BH:
  case RISCV::SFPMAD: case RISCV::SFPMAD_WH:
  case RISCV::SFPADD: case RISCV::SFPADD_WH:
  case RISCV::SFPMUL: case RISCV::SFPMUL_WH:
  case RISCV::SFPMULI: case RISCV::SFPADDI:
  case RISCV::SFPMOV: case RISCV::SFPABS:
  case RISCV::SFPEXEXP: case RISCV::SFPEXMAN:
  case RISCV::SFPDIVP2: case RISCV::SFPIADD:
  case RISCV::SFPSHFT: case RISCV::SFPSETCC:
  case RISCV::SFPAND: case RISCV::SFPOR:
  case RISCV::SFPNOT: case RISCV::SFPLZ:
  case RISCV::SFPSETEXP: case RISCV::SFPSETMAN:
  case RISCV::SFPSETSGN: case RISCV::SFPCAST:
  case RISCV::SFPSWAP: case RISCV::SFPSHFT2:
  case RISCV::SFPTRANSP: case RISCV::SFPXOR:
  case RISCV::SFP_STOCH_RND:
  case RISCV::SFPCONFIG:
  case RISCV::SFPNOP:
  case RISCV::SFPPUSHC: case RISCV::SFPPOPC: case RISCV::SFPCOMPC:
  case RISCV::SFPENCC:
  case RISCV::SFPMUL24: case RISCV::SFPARECIP:
  case RISCV::SFPGT: case RISCV::SFPLE:
  // _lv variants
  case RISCV::SFPMOV_LV: case RISCV::SFPABS_LV:
  case RISCV::SFPMAD_LV: case RISCV::SFPADD_LV: case RISCV::SFPMUL_LV:
  case RISCV::SFPLOAD_BH_LV: case RISCV::SFPLOAD_WH_LV:
  case RISCV::SFPEXEXP_LV: case RISCV::SFPEXMAN_LV:
  case RISCV::SFPDIVP2_LV: case RISCV::SFPCAST_LV:
  case RISCV::SFPLZ_LV: case RISCV::SFPNOT_LV:
  case RISCV::SFPSETEXP_LV: case RISCV::SFPSETMAN_LV:
  case RISCV::SFPSETSGN_LV: case RISCV::SFPSHFT2_LV:
  case RISCV::SFPARECIP_LV: case RISCV::SFPMUL24_LV:
  // Tensix instructions emitted as .word
  case RISCV::TTREPLAY:
    return true;
  default:
    return false;
  }
}

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
    if (!isSFPUInstr(*MBBI)) {
      ++MBBI;
      continue;
    }

    // MBBI points to the first SFPU instruction. Remember the insert point.
    auto ClusterStart = MBBI;

    // Scan forward: collect the cluster (SFPU + interleaved scalar)
    auto ClusterEnd = MBBI;
    unsigned SFPUCount = 0;
    while (ClusterEnd != MBBE) {
      if (isSFPUInstr(*ClusterEnd)) {
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
          if (isSFPUInstr(*Lookahead)) {
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
      if (isSFPUInstr(*I) || I->isDebugInstr()) {
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
