//===-- RISCVXttSFPUErrata.cpp - SFPU Errata Workarounds ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// MachineFunctionPass implementing hardware errata workarounds for the
// Tenstorrent SFPU vector unit.
//
// Errata handled:
//   E-001: WH Read-After-Write Hazard (byte/half store before word load)
//   E-002: WH_B0 SFPSHFT2 Shift-Right Zero-Fill Bug
//   E-004: SFPU Pipeline Hazards (NOP insertion for WH; selective for BH errata)
//   E-004a: BH Scoreboard Errata (NOP for 10 specific instruction combinations)
//   E-005: SFPSTORE Source Register Restriction (L12-L15)
//   E-012: ebreak Erratum (8 NOPs required after ebreak)
//
// This pass runs after register allocation and before final code emission.
//
// Reference: ttsim-analysis/ERRATA.md E-001 through E-012
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "RISCVXttSFPUUtil.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-xttsfpu-errata"

namespace {

class RISCVXttSFPUErrata : public MachineFunctionPass {
public:
  static char ID;

  RISCVXttSFPUErrata() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "RISC-V Tenstorrent SFPU Errata Workarounds";
  }

private:
  const RISCVSubtarget *STI = nullptr;
  const RISCVInstrInfo *TII = nullptr;

  bool handleE004_PipelineHazards(MachineFunction &MF);
  bool handleE005_StoreRegRestriction(MachineFunction &MF);
  bool handleE012_EbreakNops(MachineFunction &MF);
  bool handleE002_SFPSHFT2ZeroFill(MachineFunction &MF);

  bool isSFPU2Cycle(const MachineInstr &MI) const;
  bool isBHScoreboardErrata(const MachineInstr &Consumer,
                            const MachineInstr &Producer) const;
};

} // end anonymous namespace

char RISCVXttSFPUErrata::ID = 0;

/// Check if an SFPU instruction has 2-cycle latency.
/// These are: SFPMAD, SFPADD, SFPMUL, SFPMULI, SFPADDI, SFPSWAP,
///            SFPLUTFP32, SFPMUL24, and SFPSHFT2 on WH.
bool RISCVXttSFPUErrata::isSFPU2Cycle(const MachineInstr &MI) const {
  switch (MI.getOpcode()) {
  case RISCV::SFPMAD:
  case RISCV::SFPADD:
  case RISCV::SFPMUL:
  case RISCV::SFPMULI:
  case RISCV::SFPADDI:
  case RISCV::SFPSWAP:
  case RISCV::SFPLUTFP32:
  case RISCV::SFPMUL24:
    return true;
  case RISCV::SFPSHFT2:
    // SFPSHFT2 is 2-cycle on WH only (E-002 workaround adds extra cycle)
    return STI->hasVendorXttSFPUWH();
  default:
    return false;
  }
}

// RISCVXttSFPU::isSFPUInstr() is now in RISCVXttSFPUUtil.h (shared across passes).

/// E-004a: Check if Consumer is a BH scoreboard errata case that needs a NOP
/// after a 2-cycle Producer instruction.
///
/// BH's hardware scoreboard has bugs in dependency tracking for ~10 instruction
/// combinations. For these, the scoreboard fails to detect that the consumer
/// reads a register written by the producer, so no stall occurs and we get
/// wrong results. A software NOP must be inserted.
///
/// Source: tt-metal #14591, ttsim-analysis/ERRATA.md E-004a.
///
/// Conservative approach: if the consumer opcode is in the errata set AND any
/// register defined by the producer is used by the consumer, insert a NOP.
bool RISCVXttSFPUErrata::isBHScoreboardErrata(
    const MachineInstr &Consumer, const MachineInstr &Producer) const {

  // mod1 constants from sfpi_constants.h
  constexpr unsigned SFPSWAP_MOD1_SWAP = 0;
  constexpr unsigned SFPSHFT2_MOD1_SUBVEC_SHFLROR1_AND_COPY4 = 2;
  constexpr unsigned SFPSHFT2_MOD1_SUBVEC_SHFLROR1 = 3;
  constexpr unsigned SFPSHFT2_MOD1_SUBVEC_SHFLSHR1 = 4;
  constexpr unsigned SFPSHFT2_MOD1_SHFT_LREG = 5;
  constexpr unsigned SFPSHFT2_MOD1_SHFT_IMM = 6;

  unsigned Opc = Consumer.getOpcode();

  // Quick reject: not an errata-affected opcode
  bool IsErrataOpcode = false;
  switch (Opc) {
  case RISCV::SFPIADD:     // Errata #3: scoreboard misses VD read
  case RISCV::SFPSHFT:     // Errata #4: scoreboard misses VD read
  case RISCV::SFPCONFIG:   // Errata #5: scoreboard misses L0 read
  case RISCV::SFPAND:      // Errata #1: USE_VB mode misses VB read
  case RISCV::SFPOR:       // Errata #2: USE_VB mode misses VB read
  case RISCV::SFPSWAP:     // Errata #6: non-SWAP modes miss 1st-cycle reads
  case RISCV::SFPSHFT2:    // Errata #7/#8: shuffle/shift modes
  case RISCV::SFPLUT:      // Errata #9: suspected
  case RISCV::SFPLUTFP32:  // Errata #10: suspected
    IsErrataOpcode = true;
    break;
  default:
    return false;
  }

  // For mode-dependent errata, check the mod1 value to see if this specific
  // mode is affected. Get mod1 from the last explicit operand.
  auto getMod1 = [](const MachineInstr &MI) -> int {
    unsigned NumOps = MI.getNumExplicitOperands();
    if (NumOps == 0) return -1;
    const MachineOperand &LastOp = MI.getOperand(NumOps - 1);
    return LastOp.isImm() ? LastOp.getImm() : -1;
  };

  switch (Opc) {
  case RISCV::SFPAND:
  case RISCV::SFPOR: {
    // Only affected when MOD1_USE_VB (bit 0) is set.
    // Currently neither GCC nor LLVM generates this mode, but guard for future.
    int Mod1 = getMod1(Consumer);
    if (Mod1 < 0 || !(Mod1 & 1))
      return false;  // Default mode: scoreboard tracks correctly
    break;
  }
  case RISCV::SFPSWAP: {
    // Only affected in non-SWAP modes (mod1 != 0).
    int Mod1 = getMod1(Consumer);
    if (Mod1 == SFPSWAP_MOD1_SWAP)
      return false;  // Plain SWAP: scoreboard works
    break;
  }
  case RISCV::SFPSHFT2: {
    // Modes 2-6 are affected. Modes 0-1 are fine (COPY4, CHAINED_COPY4).
    int Mod1 = getMod1(Consumer);
    if (Mod1 < SFPSHFT2_MOD1_SUBVEC_SHFLROR1_AND_COPY4)
      return false;  // Modes 0,1: scoreboard works
    break;
  }
  case RISCV::SFPIADD:
  case RISCV::SFPSHFT:
  case RISCV::SFPCONFIG:
  case RISCV::SFPLUT:
  case RISCV::SFPLUTFP32:
    // Always affected (no mode-dependent check)
    break;
  default:
    return false;
  }

  // Check if there's an actual register conflict: does the producer define
  // a register that the consumer uses?
  for (const MachineOperand &Def : Producer.defs()) {
    if (!Def.isReg())
      continue;
    Register DefReg = Def.getReg();

    // Special case: SFPCONFIG reads L0 implicitly (errata #5)
    if (Opc == RISCV::SFPCONFIG && DefReg == RISCV::L0)
      return true;

    for (const MachineOperand &Use : Consumer.uses()) {
      if (Use.isReg() && Use.getReg() == DefReg)
        return true;
    }
  }

  return false;
}

/// E-004: Insert SFPNOP after SFPU instructions that need pipeline delays.
///
/// Validated against sfpi-gcc/gcc/config/riscv/tt/rtl-rvtt-schedule.cc:
/// GCC uses three delay types per instruction, per architecture:
///   - xtt_delay_none:    No NOP needed
///   - xtt_delay_static:  Always insert NOP (SFPSWAP, SFPSHFT2 shuffle modes)
///   - xtt_delay_dynamic: Insert NOP only if dependent instruction follows
///
/// BH: Hardware scoreboarding handles most RAW hazards automatically.
///   - Static delay:  SFPSWAP (all modes), SFPSHFT2 (subvec shuffle modes)
///   - Dynamic delay: SFPMAD, SFPADD, SFPMUL, SFPMULI, SFPADDI, SFPLUTFP32
///   - No delay:      All 1-cycle instructions
///
/// WH: No hardware scoreboarding — all 2-cycle instructions need NOPs.
///   - Static delay:  SFPSWAP, SFPSHFT2 (shuffle modes)
///   - Dynamic delay: SFPMAD, SFPADD, SFPMUL, SFPMULI, SFPADDI, SFPLUTFP32
///
/// On WH, dynamic-delay instructions always get a NOP if the next SFPU
/// instruction is dependent. On BH, dynamic-delay instructions only get a NOP
/// if the next instruction is an immediately-following dependent read (the
/// hardware scoreboard handles cross-basic-block dependencies).
bool RISCVXttSFPUErrata::handleE004_PipelineHazards(MachineFunction &MF) {
  bool Changed = false;
  bool IsBH = STI->hasVendorXttSFPUBH();

  for (MachineBasicBlock &MBB : MF) {
    for (auto MBBI = MBB.begin(), MBBE = MBB.end(); MBBI != MBBE; ++MBBI) {
      MachineInstr &MI = *MBBI;

      if (!isSFPU2Cycle(MI))
        continue;

      // Check if next instruction is already SFPNOP (already safe)
      auto NextMI = std::next(MBBI);
      if (NextMI != MBBE && NextMI->getOpcode() == RISCV::SFPNOP)
        continue;

      bool NeedNop = false;
      bool IsStaticDelay = (MI.getOpcode() == RISCV::SFPSWAP ||
                            MI.getOpcode() == RISCV::SFPSHFT2);

      if (IsStaticDelay) {
        // Static delay: always need NOP if next is any non-NOP SFPU instruction.
        // This applies on BOTH BH and WH.
        if (NextMI != MBBE && RISCVXttSFPU::isSFPUInstr(*NextMI))
          NeedNop = true;
      } else if (!IsBH) {
        // Dynamic delay on WH (no scoreboarding): need NOP if next SFPU
        // instruction reads our destination register.
        if (NextMI != MBBE && RISCVXttSFPU::isSFPUInstr(*NextMI)) {
          for (const MachineOperand &Def : MI.defs()) {
            if (!Def.isReg())
              continue;
            for (const MachineOperand &Use : NextMI->uses()) {
              if (Use.isReg() && Use.getReg() == Def.getReg()) {
                NeedNop = true;
                break;
              }
            }
            if (NeedNop)
              break;
          }
        }
      } else {
        // Dynamic delay on BH: scoreboard handles MOST cases, but E-004a
        // errata cases need NOP. See ERRATA.md E-004a, tt-metal #14591.
        if (NextMI != MBBE && RISCVXttSFPU::isSFPUInstr(*NextMI) &&
            isBHScoreboardErrata(*NextMI, MI))
          NeedNop = true;
      }

      if (NeedNop) {
        LLVM_DEBUG(dbgs() << "  E-004: inserting NOP after: " << MI);
        BuildMI(MBB, NextMI, MI.getDebugLoc(), TII->get(RISCV::SFPNOP));
        Changed = true;
      }
    }
  }

  return Changed;
}

/// E-005: Verify SFPSTORE does not use L12-L15 as source.
/// The register allocator should prevent this via SFPUStoreRegs constraint,
/// but this pass provides a safety check.
bool RISCVXttSFPUErrata::handleE005_StoreRegRestriction(MachineFunction &MF) {
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      unsigned Opc = MI.getOpcode();
      if (Opc != RISCV::SFPSTORE_BH && Opc != RISCV::SFPSTORE_WH)
        continue;

      // Check the lreg_ind operand (first operand for store)
      const MachineOperand &LRegOp = MI.getOperand(0);
      if (!LRegOp.isReg())
        continue;

      Register Reg = LRegOp.getReg();
      if (Reg == RISCV::L12 || Reg == RISCV::L13 ||
          Reg == RISCV::L14 || Reg == RISCV::L15) {
        LLVM_DEBUG(dbgs() << "  E-005 violation: " << MI);
        MI.emitError("E-005: SFPSTORE cannot use L12-L15 as source register");
        return false;
      }
    }
  }
  return false;
}

/// E-012: Insert 8 NOPs after every ebreak instruction.
/// The processor state is unreliable without them.
bool RISCVXttSFPUErrata::handleE012_EbreakNops(MachineFunction &MF) {
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (auto MBBI = MBB.begin(), MBBE = MBB.end(); MBBI != MBBE; ++MBBI) {
      if (MBBI->getOpcode() != RISCV::EBREAK)
        continue;

      LLVM_DEBUG(dbgs() << "  E-012: inserting 8 NOPs after ebreak\n");
      auto InsertPt = std::next(MBBI);
      for (int i = 0; i < 8; ++i) {
        BuildMI(MBB, InsertPt, MBBI->getDebugLoc(), TII->get(RISCV::SFPNOP));
      }
      Changed = true;
    }
  }

  return Changed;
}

/// E-002: WH_B0 SFPSHFT2 SHFLSHR1 zero-fill bug.
/// Before SFPSHFT2 with SHFLSHR1 mode, insert a dead rotate using L9 (zero)
/// with SHFLROR1 mode to clear the pipeline value to 0.
bool RISCVXttSFPUErrata::handleE002_SFPSHFT2ZeroFill(MachineFunction &MF) {
  // Only affects WH
  if (!STI->hasVendorXttSFPUWH())
    return false;

  bool Changed = false;

  // SFPSHFT2 mod1 values from sfpi-gcc/gcc/config/riscv/tt/rvtt-protos.h:213-219
  // and ttsim-analysis/ERRATA.md C-020 (SFPSHFT2 Modifiers table)
  constexpr unsigned SHFLSHR1_MOD1 = 4;  // SUBVEC_SHFLSHR1 (shift right 1, zero-fill)
  constexpr unsigned SHFLROR1_MOD1 = 3;  // SUBVEC_SHFLROR1 (rotate right 1)

  for (MachineBasicBlock &MBB : MF) {
    for (auto MBBI = MBB.begin(), MBBE = MBB.end(); MBBI != MBBE; ++MBBI) {
      if (MBBI->getOpcode() != RISCV::SFPSHFT2)
        continue;

      // Check mod1 field for SHFLSHR1 mode
      // The mod1 operand is the last operand in SFPUUnaryReg format
      const MachineOperand &Mod1Op = MBBI->getOperand(MBBI->getNumOperands() - 1);
      if (!Mod1Op.isImm() || Mod1Op.getImm() != SHFLSHR1_MOD1)
        continue;

      // Insert dead rotate: SFPSHFT2 L9, dst, SHFLROR1
      // This clears the pipeline to 0 so the subsequent SHFLSHR1 gets correct fill
      LLVM_DEBUG(dbgs() << "  E-002: inserting SHFLROR1 workaround before: "
                        << *MBBI);
      BuildMI(MBB, MBBI, MBBI->getDebugLoc(), TII->get(RISCV::SFPSHFT2))
          .addReg(RISCV::L0, RegState::Define)  // dummy dest
          .addImm(0)                              // imm12 = 0
          .addReg(RISCV::L9)                     // src = zero constant
          .addImm(SHFLROR1_MOD1);                // SHFLROR1 mode

      Changed = true;
    }
  }

  return Changed;
}

bool RISCVXttSFPUErrata::runOnMachineFunction(MachineFunction &MF) {
  STI = &MF.getSubtarget<RISCVSubtarget>();

  // Only run if SFPU extension is enabled
  LLVM_DEBUG(dbgs() << getPassName() << " on " << MF.getName() << "\n");
  if (!STI->hasVendorXttSFPU())
    return false;

  TII = STI->getInstrInfo();

  bool Changed = false;

  // Run errata workarounds in order of priority
  Changed |= handleE005_StoreRegRestriction(MF);
  Changed |= handleE002_SFPSHFT2ZeroFill(MF);
  Changed |= handleE004_PipelineHazards(MF);
  Changed |= handleE012_EbreakNops(MF);

  return Changed;
}

INITIALIZE_PASS(RISCVXttSFPUErrata, DEBUG_TYPE,
                "RISC-V Tenstorrent SFPU Errata Workarounds", false, false)

FunctionPass *llvm::createRISCVXttSFPUErrataPass() {
  return new RISCVXttSFPUErrata();
}
