//===-- RISCVXttSFPUUtil.h - SFPU shared utility functions ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// Shared utilities used by multiple SFPU optimization passes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVXTTSFPUUTIL_H
#define LLVM_LIB_TARGET_RISCV_RISCVXTTSFPUUTIL_H

#include "RISCVInstrInfo.h"
#include "llvm/CodeGen/MachineInstr.h"

namespace llvm {
namespace RISCVXttSFPU {

/// Return true if MI is an SFPU instruction (Tensix custom encoding space).
/// Covers all base, BH-only, WH-only, and _lv variant instructions.
inline bool isSFPUInstr(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  // Load/Store
  case RISCV::SFPLOAD_BH: case RISCV::SFPLOAD_WH:
  case RISCV::SFPLOADI:
  case RISCV::SFPSTORE_BH: case RISCV::SFPSTORE_WH:
  case RISCV::SFPLUT: case RISCV::SFPLUTFP32:
  case RISCV::SFPLOADMACRO_BH:
  // 3-Operand arithmetic
  case RISCV::SFPMAD: case RISCV::SFPMAD_WH:
  case RISCV::SFPADD: case RISCV::SFPADD_WH:
  case RISCV::SFPMUL: case RISCV::SFPMUL_WH:
  // Immediate arithmetic
  case RISCV::SFPMULI: case RISCV::SFPADDI:
  // Standard unary
  case RISCV::SFPDIVP2: case RISCV::SFPEXEXP: case RISCV::SFPEXMAN:
  case RISCV::SFPIADD: case RISCV::SFPSHFT:
  case RISCV::SFPMOV: case RISCV::SFPABS:
  case RISCV::SFPAND: case RISCV::SFPOR: case RISCV::SFPNOT:
  case RISCV::SFPLZ: case RISCV::SFPSETEXP:
  case RISCV::SFPSETMAN: case RISCV::SFPSETSGN:
  // CC stack / predication
  case RISCV::SFPSETCC:
  case RISCV::SFPPUSHC: case RISCV::SFPPOPC: case RISCV::SFPCOMPC:
  case RISCV::SFPENCC:
  // Cross-lane / transpose
  case RISCV::SFPTRANSP: case RISCV::SFPXOR:
  // Rounding / cast
  case RISCV::SFP_STOCH_RND: case RISCV::SFPCAST:
  // Config / control
  case RISCV::SFPCONFIG: case RISCV::SFPNOP:
  // Swap / shift2
  case RISCV::SFPSWAP: case RISCV::SFPSHFT2:
  // BH-only
  case RISCV::SFPMUL24: case RISCV::SFPARECIP:
  case RISCV::SFPGT: case RISCV::SFPLE:
  // _lv variants
  case RISCV::SFPMOV_LV: case RISCV::SFPABS_LV:
  case RISCV::SFPEXEXP_LV: case RISCV::SFPEXMAN_LV:
  case RISCV::SFPDIVP2_LV: case RISCV::SFPCAST_LV:
  case RISCV::SFPLZ_LV: case RISCV::SFPNOT_LV:
  case RISCV::SFPSETEXP_LV: case RISCV::SFPSETMAN_LV:
  case RISCV::SFPSETSGN_LV: case RISCV::SFPSHFT2_LV:
  case RISCV::SFPMAD_LV: case RISCV::SFPADD_LV: case RISCV::SFPMUL_LV:
  case RISCV::SFPLOAD_BH_LV: case RISCV::SFPLOAD_WH_LV:
  case RISCV::SFPARECIP_LV: case RISCV::SFPMUL24_LV:
  // Tensix replay
  case RISCV::TTREPLAY:
    return true;
  default:
    return false;
  }
}

/// Return true if MI is a 2-cycle SFPU instruction (MAD unit or SWAP).
/// These require NOP insertion on WH and scoreboard-errata checking on BH.
inline bool isSFPU2Cycle(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case RISCV::SFPMAD: case RISCV::SFPMAD_WH:
  case RISCV::SFPADD: case RISCV::SFPADD_WH:
  case RISCV::SFPMUL: case RISCV::SFPMUL_WH:
  case RISCV::SFPMULI: case RISCV::SFPADDI:
  case RISCV::SFPSWAP:
  case RISCV::SFPLUTFP32:
  case RISCV::SFPMUL24:
  // _lv variants of 2-cycle instructions
  case RISCV::SFPMAD_LV: case RISCV::SFPADD_LV: case RISCV::SFPMUL_LV:
  case RISCV::SFPMUL24_LV:
    return true;
  default:
    return false;
  }
}

} // namespace RISCVXttSFPU
} // namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVXTTSFPUUTIL_H
