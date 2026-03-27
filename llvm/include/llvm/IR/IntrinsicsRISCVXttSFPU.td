//===-- IntrinsicsRISCVXttSFPU.td - XttSFPU Intrinsics -----*- tablegen -*-===//
//
// LLVM intrinsic definitions for Tenstorrent SFPU instructions.
// Each SFPU instruction maps to one intrinsic (1:1 correspondence).
//
// Intrinsic naming: int_riscv_tt_sfp<mnemonic>
// All operands are i32 (register indices and immediates).
// Immediate operands are marked with ImmArg<ArgIndex<N>> so they become
// TargetConstant nodes in the DAG, enabling proper type-checked ISel matching.
//
// Reference: ttsim-analysis/FUNCTIONAL_UNITS.md Section 3.4
//            ttsim-analysis/ERRATA.md C-006
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "riscv" in {

//--- Load/Store Intrinsics ---

// SFPLOAD: load from Dst to LReg — all args are immediates
def int_riscv_tt_sfpload
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], // mod0, addr_mode, addr
                [IntrReadMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

// SFPLOADI: load immediate to LReg — both args are immediates
def int_riscv_tt_sfploadi
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty],             // mod0, imm16
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

// SFPSTORE: store LReg to Dst — arg 0 is register, rest are immediates
def int_riscv_tt_sfpstore
    : Intrinsic<[],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                // lreg_src, mod0, addr_mode, addr
                [IntrWriteMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

// SFPLUT: LUT lookup — 2 immediates (caller packs addr_mode|addr into dest_reg_addr)
def int_riscv_tt_sfplut
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty], // mod0, dest_reg_addr (packed)
                [IntrReadMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

// SFPLOADMACRO: macro load (writes L16) — arg 0 is register, rest immediates
def int_riscv_tt_sfploadmacro
    : Intrinsic<[],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                // lreg, mod0, addr_mode, addr
                [IntrHasSideEffects,
                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

//--- Immediate-16 Arithmetic ---

// SFPMULI: arg 0 is register (dest/source), args 1-2 are immediates
def int_riscv_tt_sfpmuli
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], // lreg_dest, imm16, mod1
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

// SFPADDI: arg 0 is register, args 1-2 are immediates
def int_riscv_tt_sfpaddi
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], // lreg_dest, imm16, mod1
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

//--- Standard Unary Arithmetic ---
// Pattern: arg 0 = register (lreg_c), arg 1 = imm12, arg 2 = mod1 (imm4)

def int_riscv_tt_sfpdivp2
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_tt_sfpexexp
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_tt_sfpexman
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_tt_sfpiadd
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_tt_sfpshft
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_tt_sfpsetcc
    : Intrinsic<[],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], // lreg_c, imm12, mod1
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_tt_sfpmov
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_tt_sfpabs
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_tt_sfpand
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_tt_sfpor
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_tt_sfpnot
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_tt_sfplz
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_tt_sfpsetexp
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_tt_sfpsetman
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_tt_sfpsetsgn
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

//--- 3-Operand Arithmetic ---
// Pattern: args 0-2 = registers, arg 3 = mod1 (imm4)

def int_riscv_tt_sfpmad
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects, ImmArg<ArgIndex<3>>]>;

def int_riscv_tt_sfpadd
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects, ImmArg<ArgIndex<3>>]>;

def int_riscv_tt_sfpmul
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects, ImmArg<ArgIndex<3>>]>;

//--- CC Stack / Predication ---

def int_riscv_tt_sfppushc
    : Intrinsic<[], [], [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfppopc
    : Intrinsic<[], [], [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpcompc
    : Intrinsic<[], [], [IntrNoMem, IntrHasSideEffects]>;

// SFPENCC: both args are immediates
def int_riscv_tt_sfpencc
    : Intrinsic<[], [llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

//--- Cross-Lane / Transpose ---
// Same pattern as unary: arg 0 = register, arg 1 = imm12, arg 2 = mod1

def int_riscv_tt_sfptransp
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_tt_sfpxor
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

//--- Rounding / Cast ---

// SFP_STOCH_RND: args 0,1,4 are immediates; args 2,3 are registers
def int_riscv_tt_sfpstochrnd
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                // rnd_mode, imm5, lreg_src_b, lreg_src_c, mod1
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<4>>]>;

// SFPCAST: same as unary
def int_riscv_tt_sfpcast
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

//--- Config / Control ---

// SFPCONFIG: all args are immediates
def int_riscv_tt_sfpconfig
    : Intrinsic<[],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], // imm16, vd, mod1
                [IntrHasSideEffects,
                 ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

// SFPNOP
def int_riscv_tt_sfpnop
    : Intrinsic<[], [], [IntrNoMem, IntrHasSideEffects]>;

//--- SFPSWAP / SFPSHFT2 / SFPLUTFP32 ---

def int_riscv_tt_sfpswap
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_tt_sfpshft2
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

// SFPLUTFP32: arg 0 = register, arg 1 = mod1 (imm4)
def int_riscv_tt_sfplutfp32
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects, ImmArg<ArgIndex<1>>]>;

//--- BH-Only Intrinsics ---

// SFPMUL24: same as 3-op (3 registers + mod1 immediate)
def int_riscv_tt_sfpmul24
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects, ImmArg<ArgIndex<3>>]>;

// SFPARECIP: same as unary
def int_riscv_tt_sfparecip
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

// SFPGT / SFPLE: same as setcc (register, imm12, mod1)
def int_riscv_tt_sfpgt
    : Intrinsic<[],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;
def int_riscv_tt_sfple
    : Intrinsic<[],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

//--- Live-Value (_lv) Intrinsics ---
// _lv variants: arg 0 = live_val (register), then same args as base.
// The immediate positions shift by +1 from the base intrinsic.

// _lv unary: (live_val, lreg_c, imm12, mod1) — imm at 2,3
def int_riscv_tt_sfpmov_lv
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

// _lv 3-op: (live_val, src_a, src_b, src_c, mod1) — imm at 4
def int_riscv_tt_sfpmad_lv
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects, ImmArg<ArgIndex<4>>]>;

def int_riscv_tt_sfpmul_lv
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects, ImmArg<ArgIndex<4>>]>;

def int_riscv_tt_sfpadd_lv
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects, ImmArg<ArgIndex<4>>]>;

// _lv load: (live_val, mod0, addr_mode, addr) — imm at 1,2,3
def int_riscv_tt_sfpload_lv
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrReadMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

// _lv unary variants: (live_val, lreg_c, imm12, mod1) — imm at 2,3
def int_riscv_tt_sfpcast_lv
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_tt_sfparecip_lv
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_tt_sfpdivp2_lv
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_tt_sfpexexp_lv
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_tt_sfpexman_lv
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_tt_sfpabs_lv
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_tt_sfpnot_lv
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_tt_sfplz_lv
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects,
                 ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

// _lv 3-op: (live_val, src_a, src_b, src_c, mod1) — imm at 4
def int_riscv_tt_sfpmul24_lv
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects, ImmArg<ArgIndex<4>>]>;

} // TargetPrefix = "riscv"
