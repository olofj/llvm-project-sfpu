//===-- RISCVXttSFPUISelPatterns.td - ISel DAG Patterns -----*- tablegen -*-===//
//
// ISel DAG patterns for the Tenstorrent SFPU extension.
// Maps intrinsic calls to SFPU machine instructions.
//
// IMPORTANT: On the match side (left), intrinsic operands use:
//   i32:$x    — for register operands
//   timm:$x   — for immediate operands (target-independent matching)
// On the emit side (right), instruction operands use:
//   SFPUAllRegs:$x, SFPURegs:$x — for register operands
//   uimmN_sfpu:$x                — for immediate operands (encoding)
//
// Reference: ttsim-analysis/ERRATA.md C-009 (BH-only instructions)
//
//===----------------------------------------------------------------------===//

let Predicates = [HasXttSFPU] in {

//===----------------------------------------------------------------------===//
// Intrinsic → Instruction lowering
//===----------------------------------------------------------------------===//

// --- 3-operand arithmetic ---
def : Pat<(int_riscv_tt_sfpmad i32:$a, i32:$b, i32:$c, timm:$mod1),
          (SFPMAD SFPUAllRegs:$a, SFPUAllRegs:$b,
                  SFPUAllRegs:$c, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpmul i32:$a, i32:$b, i32:$c, timm:$mod1),
          (SFPMUL SFPUAllRegs:$a, SFPUAllRegs:$b,
                  SFPUAllRegs:$c, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpadd i32:$a, i32:$b, i32:$c, timm:$mod1),
          (SFPADD SFPUAllRegs:$a, SFPUAllRegs:$b,
                  SFPUAllRegs:$c, uimm4_sfpu:$mod1)>;

// --- Standard unary ---
def : Pat<(int_riscv_tt_sfpmov i32:$src, timm:$imm, timm:$mod1),
          (SFPMOV uimm12_sfpu:$imm, SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpabs i32:$src, timm:$imm, timm:$mod1),
          (SFPABS uimm12_sfpu:$imm, SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpand i32:$src, timm:$imm, timm:$mod1),
          (SFPAND uimm12_sfpu:$imm, SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpor i32:$src, timm:$imm, timm:$mod1),
          (SFPOR uimm12_sfpu:$imm, SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpxor i32:$src, timm:$imm, timm:$mod1),
          (SFPXOR uimm12_sfpu:$imm, SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpnot i32:$src, timm:$imm, timm:$mod1),
          (SFPNOT uimm12_sfpu:$imm, SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpexexp i32:$src, timm:$imm, timm:$mod1),
          (SFPEXEXP uimm12_sfpu:$imm, SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpexman i32:$src, timm:$imm, timm:$mod1),
          (SFPEXMAN uimm12_sfpu:$imm, SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfplz i32:$src, timm:$imm, timm:$mod1),
          (SFPLZ uimm12_sfpu:$imm, SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpiadd i32:$src, timm:$imm, timm:$mod1),
          (SFPIADD uimm12_sfpu:$imm, SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpshft i32:$src, timm:$imm, timm:$mod1),
          (SFPSHFT uimm12_sfpu:$imm, SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpsetexp i32:$src, timm:$imm, timm:$mod1),
          (SFPSETEXP uimm12_sfpu:$imm, SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpsetman i32:$src, timm:$imm, timm:$mod1),
          (SFPSETMAN uimm12_sfpu:$imm, SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpsetsgn i32:$src, timm:$imm, timm:$mod1),
          (SFPSETSGN uimm12_sfpu:$imm, SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpdivp2 i32:$src, timm:$imm, timm:$mod1),
          (SFPDIVP2 uimm12_sfpu:$imm, SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpswap i32:$src, timm:$imm, timm:$mod1),
          (SFPSWAP uimm12_sfpu:$imm, SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpshft2 i32:$src, timm:$imm, timm:$mod1),
          (SFPSHFT2 uimm12_sfpu:$imm, SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpcast i32:$src, timm:$imm, timm:$mod1),
          (SFPCAST uimm12_sfpu:$imm, SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfptransp i32:$src, timm:$imm, timm:$mod1),
          (SFPTRANSP uimm12_sfpu:$imm, SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;

// --- CC stack and comparisons ---
// $lreg_dest is now uimm4 (immediate selector), not a register. Pass 0.
// $lreg_c for SFPENCC/SFPPUSHC/SFPPOPC is also uimm4 since SFPUUnaryCC
// uses SFPUAllRegs for it — but in these cases the actual source is unused,
// so we pass any allocatable register. For SFPSETCC, $lreg_c is the real source.
def : Pat<(int_riscv_tt_sfpsetcc i32:$src, timm:$imm, timm:$mod1),
          (SFPSETCC uimm12_sfpu:$imm, SFPUAllRegs:$src, 0, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpencc timm:$imm, timm:$mod1),
          (SFPENCC uimm12_sfpu:$imm, L9, 0, uimm4_sfpu:$mod1)>;

// --- No-operand ---
// Use L9 (reserved constant 0.0, always live) for the don't-care $lreg_c field.
def : Pat<(int_riscv_tt_sfpnop), (SFPNOP)>;
def : Pat<(int_riscv_tt_sfppushc), (SFPPUSHC 0, L9, 0, 0)>;
def : Pat<(int_riscv_tt_sfppopc), (SFPPOPC 0, L9, 0, 0)>;
def : Pat<(int_riscv_tt_sfpcompc), (SFPCOMPC 0, L9, 0, 0)>;

// --- SFPLUTFP32 ---
def : Pat<(int_riscv_tt_sfplutfp32 i32:$src, timm:$mod1),
          (SFPLUTFP32 SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;

// --- SFPLOADI ---
def : Pat<(int_riscv_tt_sfploadi timm:$mod0, timm:$imm16),
          (SFPLOADI uimm4_sfpu:$mod0, uimm16_sfpu:$imm16)>;

// SFPCONFIG: Needs custom C++ lowering (void + IntrHasSideEffects → INTRINSIC_VOID
// chain node doesn't match Pat<> directly).

// --- SFPMULI / SFPADDI (tied src/dest pseudos) ---
def : Pat<(int_riscv_tt_sfpmuli i32:$src, timm:$imm16, timm:$mod1),
          (PseudoSFPMULI SFPURegs:$src, uimm16_sfpu:$imm16, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpaddi i32:$src, timm:$imm16, timm:$mod1),
          (PseudoSFPADDI SFPURegs:$src, uimm16_sfpu:$imm16, uimm4_sfpu:$mod1)>;

// --- SFPLUT (2 immediates: mod0, packed dest_reg_addr) ---
def : Pat<(int_riscv_tt_sfplut timm:$mod0, timm:$dest_reg_addr),
          (SFPLUT uimm4_sfpu:$mod0, uimm16_sfpu:$dest_reg_addr)>;

// --- SFP_STOCH_RND (5 input operands: rnd_mode, imm5, src_b, src_c, mod1) ---
def : Pat<(int_riscv_tt_sfpstochrnd timm:$rnd_mode, timm:$imm5,
                                     i32:$src_b, i32:$src_c, timm:$mod1),
          (SFP_STOCH_RND uimm3_sfpu:$rnd_mode, uimm5_sfpu:$imm5,
                         SFPUAllRegs:$src_b, SFPUAllRegs:$src_c,
                         uimm4_sfpu:$mod1)>;
// or mayStore/mayLoad flag mismatches between intrinsic and instruction).

//===----------------------------------------------------------------------===//
// _lv variants
//===----------------------------------------------------------------------===//

def : Pat<(int_riscv_tt_sfpmov_lv i32:$live, i32:$src, timm:$imm, timm:$mod1),
          (SFPMOV_LV SFPURegs:$live, uimm12_sfpu:$imm, SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpmad_lv i32:$live, i32:$a, i32:$b, i32:$c, timm:$mod1),
          (SFPMAD_LV SFPURegs:$live, SFPUAllRegs:$a, SFPUAllRegs:$b,
                     SFPUAllRegs:$c, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpmul_lv i32:$live, i32:$a, i32:$b, i32:$c, timm:$mod1),
          (SFPMUL_LV SFPURegs:$live, SFPUAllRegs:$a, SFPUAllRegs:$b,
                     SFPUAllRegs:$c, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpadd_lv i32:$live, i32:$a, i32:$b, i32:$c, timm:$mod1),
          (SFPADD_LV SFPURegs:$live, SFPUAllRegs:$a, SFPUAllRegs:$b,
                     SFPUAllRegs:$c, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpcast_lv i32:$live, i32:$src, timm:$imm, timm:$mod1),
          (SFPCAST_LV SFPURegs:$live, uimm12_sfpu:$imm, SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpabs_lv i32:$live, i32:$src, timm:$imm, timm:$mod1),
          (SFPABS_LV SFPURegs:$live, uimm12_sfpu:$imm, SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpnot_lv i32:$live, i32:$src, timm:$imm, timm:$mod1),
          (SFPNOT_LV SFPURegs:$live, uimm12_sfpu:$imm, SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfplz_lv i32:$live, i32:$src, timm:$imm, timm:$mod1),
          (SFPLZ_LV SFPURegs:$live, uimm12_sfpu:$imm, SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpexexp_lv i32:$live, i32:$src, timm:$imm, timm:$mod1),
          (SFPEXEXP_LV SFPURegs:$live, uimm12_sfpu:$imm, SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpexman_lv i32:$live, i32:$src, timm:$imm, timm:$mod1),
          (SFPEXMAN_LV SFPURegs:$live, uimm12_sfpu:$imm, SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpdivp2_lv i32:$live, i32:$src, timm:$imm, timm:$mod1),
          (SFPDIVP2_LV SFPURegs:$live, uimm12_sfpu:$imm, SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;

} // Predicates = [HasXttSFPU]

//===----------------------------------------------------------------------===//
// BH-specific ISel patterns
//===----------------------------------------------------------------------===//

let Predicates = [HasXttSFPU, HasXttSFPUBH] in {

// BH Load/Store/LoadMacro: Custom ISel in RISCVISelDAGToDAG.cpp
// (addr/imm16 args can be runtime values — not TargetConstant)

// BH-only instructions
// SFPGT/SFPLE: $lreg_dest is now uimm4 (immediate), pass 0
def : Pat<(int_riscv_tt_sfpgt i32:$src, timm:$imm, timm:$mod1),
          (SFPGT uimm12_sfpu:$imm, SFPUAllRegs:$src, 0, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfple i32:$src, timm:$imm, timm:$mod1),
          (SFPLE uimm12_sfpu:$imm, SFPUAllRegs:$src, 0, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfparecip i32:$src, timm:$imm, timm:$mod1),
          (SFPARECIP uimm12_sfpu:$imm, SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpmul24 i32:$a, i32:$b, i32:$c, timm:$mod1),
          (SFPMUL24 SFPUAllRegs:$a, SFPUAllRegs:$b,
                    SFPUAllRegs:$c, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfparecip_lv i32:$live, i32:$src, timm:$imm, timm:$mod1),
          (SFPARECIP_LV SFPURegs:$live, uimm12_sfpu:$imm, SFPUAllRegs:$src, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpmul24_lv i32:$live, i32:$a, i32:$b, i32:$c, timm:$mod1),
          (SFPMUL24_LV SFPURegs:$live, SFPUAllRegs:$a, SFPUAllRegs:$b,
                       SFPUAllRegs:$c, uimm4_sfpu:$mod1)>;

} // Predicates = [HasXttSFPU, HasXttSFPUBH]

//===----------------------------------------------------------------------===//
// WH-specific ISel patterns
//===----------------------------------------------------------------------===//

let Predicates = [HasXttSFPU, HasXttSFPUWH] in {

// WH Load/Store/LoadMacro: Custom ISel in RISCVISelDAGToDAG.cpp

// WH C-010: 3-operand arithmetic with dst=src_a constraint.
// These override the generic SFPMAD/SFPMUL/SFPADD patterns from the base
// section. WH predicates are more specific so they take priority.
def : Pat<(int_riscv_tt_sfpmad i32:$a, i32:$b, i32:$c, timm:$mod1),
          (SFPMAD_WH SFPUAllRegs:$a, SFPUAllRegs:$b,
                     SFPUAllRegs:$c, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpmul i32:$a, i32:$b, i32:$c, timm:$mod1),
          (SFPMUL_WH SFPUAllRegs:$a, SFPUAllRegs:$b,
                     SFPUAllRegs:$c, uimm4_sfpu:$mod1)>;
def : Pat<(int_riscv_tt_sfpadd i32:$a, i32:$b, i32:$c, timm:$mod1),
          (SFPADD_WH SFPUAllRegs:$a, SFPUAllRegs:$b,
                     SFPUAllRegs:$c, uimm4_sfpu:$mod1)>;

} // Predicates = [HasXttSFPU, HasXttSFPUWH]
