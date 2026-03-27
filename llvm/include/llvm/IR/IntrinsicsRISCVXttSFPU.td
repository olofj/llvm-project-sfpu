//===-- IntrinsicsRISCVXttSFPU.td - XttSFPU Intrinsics -----*- tablegen -*-===//
//
// LLVM intrinsic definitions for Tenstorrent SFPU instructions.
//
// IMPORTANT: Almost NO ImmArg annotations. When sfpi.h's C++ API dispatches
// to these builtins, most "immediate" parameters are computed at runtime
// (e.g., __f32asui(float) for comparisons, bitwise OR for mod1 flags).
// Only mod0 in load/store (format selector) is truly compile-time constant.
// All ISel patterns use custom ISel in RISCVISelDAGToDAG.cpp which handles
// both constant and runtime values.
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "riscv" in {

//--- Load/Store --- (mod0 is always compile-time)

def int_riscv_tt_sfpload
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrReadMem, IntrHasSideEffects, ImmArg<ArgIndex<0>>]>;

def int_riscv_tt_sfploadi
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects, ImmArg<ArgIndex<0>>]>;

def int_riscv_tt_sfpstore
    : Intrinsic<[],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrWriteMem, IntrHasSideEffects, ImmArg<ArgIndex<1>>]>;

def int_riscv_tt_sfplut
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty],
                [IntrReadMem, IntrHasSideEffects, ImmArg<ArgIndex<0>>]>;

def int_riscv_tt_sfploadmacro
    : Intrinsic<[],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrHasSideEffects, ImmArg<ArgIndex<1>>]>;

//--- Immediate-16 Arithmetic --- (no ImmArg — imm16 and mod1 can be runtime)

def int_riscv_tt_sfpmuli
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;

def int_riscv_tt_sfpaddi
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;

//--- Standard Unary --- (no ImmArg — imm12 and mod1 can be runtime)

def int_riscv_tt_sfpdivp2
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpexexp
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpexman
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpiadd
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpshft
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpsetcc
    : Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpmov
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpabs
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpand
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpor
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpnot
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfplz
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpsetexp
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpsetman
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpsetsgn
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;

//--- 3-Operand Arithmetic --- (no ImmArg — mod1 can be runtime)

def int_riscv_tt_sfpmad
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpadd
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpmul
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;

//--- CC Stack / Predication ---

def int_riscv_tt_sfppushc : Intrinsic<[], [], [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfppopc  : Intrinsic<[], [], [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpcompc : Intrinsic<[], [], [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpencc
    : Intrinsic<[], [llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;

//--- Cross-Lane / Transpose ---

def int_riscv_tt_sfptransp
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpxor
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;

//--- Rounding / Cast ---

def int_riscv_tt_sfpstochrnd
    : Intrinsic<[llvm_i32_ty],
                [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpcast
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;

//--- Config / Control ---

def int_riscv_tt_sfpconfig
    : Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrHasSideEffects]>;
def int_riscv_tt_sfpnop
    : Intrinsic<[], [], [IntrNoMem, IntrHasSideEffects]>;

//--- SFPSWAP / SFPSHFT2 / SFPLUTFP32 ---

def int_riscv_tt_sfpswap
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpshft2
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfplutfp32
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;

//--- BH-Only ---

def int_riscv_tt_sfpmul24
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfparecip
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpgt
    : Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfple
    : Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;

//--- Live-Value (_lv) --- (no ImmArg)

def int_riscv_tt_sfpmov_lv
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpmad_lv
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpmul_lv
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpadd_lv
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpload_lv
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrReadMem, IntrHasSideEffects, ImmArg<ArgIndex<1>>]>;
def int_riscv_tt_sfpcast_lv
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfparecip_lv
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpdivp2_lv
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpexexp_lv
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpexman_lv
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpabs_lv
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpnot_lv
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfplz_lv
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;
def int_riscv_tt_sfpmul24_lv
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrHasSideEffects]>;

} // TargetPrefix = "riscv"
