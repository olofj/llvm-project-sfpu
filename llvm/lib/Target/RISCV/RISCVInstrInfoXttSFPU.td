//===-- RISCVInstrInfoXttSFPU.td - SFPU Instructions -------*- tablegen -*-===//
//
// Tenstorrent SFPU (Vector Unit) instruction definitions for the XttSFPU
// vendor extension. Implements all 38 base opcodes (0x70-0x95) plus BH-only
// extensions (SFPMUL24, SFPARECIP).
//
// Encoding: All SFPU instructions are 32-bit Tensix coprocessor words.
// bits[31:24] = Tensix opcode, bits[23:0] = payload per format family.
// Encoded in RISC-V ELF using custom instruction space (bits[1:0] = 0b11).
//
// Reference: ttsim-analysis/FUNCTIONAL_UNITS.md Section 3.4-3.5
//            ttsim-analysis/ERRATA.md C-006, C-009, C-020
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Feature predicates
//===----------------------------------------------------------------------===//

def HasXttSFPU   : Predicate<"Subtarget->hasVendorXttSFPU()">,
                   AssemblerPredicate<(all_of FeatureVendorXttSFPU),
                   "'XttSFPU' (Tenstorrent SFPU Vector Unit)">;

def HasXttSFPUBH : Predicate<"Subtarget->hasVendorXttSFPUBH()">,
                   AssemblerPredicate<(all_of FeatureVendorXttSFPUBH),
                   "'XttSFPU-BH' (Tenstorrent SFPU Blackhole extensions)">;

def HasXttSFPUWH : Predicate<"Subtarget->hasVendorXttSFPUWH()">,
                   AssemblerPredicate<(all_of FeatureVendorXttSFPUWH),
                   "'XttSFPU-WH' (Tenstorrent SFPU Wormhole extensions)">;

//===----------------------------------------------------------------------===//
// Operand types
//===----------------------------------------------------------------------===//

// 4-bit LReg index (0-16)
def SFPURegOp : RegisterOperand<SFPUAllRegs>;
def SFPUDstOp : RegisterOperand<SFPURegs>;
def SFPUStoreOp : RegisterOperand<SFPUStoreRegs>;

// Immediate operands
// Each inherits from Operand<i32> (for encoding/asm) AND TImmLeaf (for ISel
// pattern matching). TImmLeaf matches TargetConstant nodes produced by ImmArg
// on intrinsic definitions, with a range-checking predicate that enforces the
// correct bit width for each Tensix immediate field.

def simm12_sfpu : Operand<i32> {
  let ParserMatchClass = SImmAsmOperand<12>;
  let DecoderMethod = "decodeSImmOperand<12>";
}

def uimm2_sfpu : Operand<i32>, TImmLeaf<i32, [{return isUInt<2>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<2>;
  let DecoderMethod = "decodeUImmOperand<2>";
}

def uimm3_sfpu : Operand<i32>, TImmLeaf<i32, [{return isUInt<3>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<3>;
  let DecoderMethod = "decodeUImmOperand<3>";
}

def uimm4_sfpu : Operand<i32>, TImmLeaf<i32, [{return isUInt<4>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<4>;
  let DecoderMethod = "decodeUImmOperand<4>";
}

def uimm5_sfpu : Operand<i32>, TImmLeaf<i32, [{return isUInt<5>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<5>;
  let DecoderMethod = "decodeUImmOperand<5>";
}

def uimm8_sfpu : Operand<i32>, TImmLeaf<i32, [{return isUInt<8>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<8>;
  let DecoderMethod = "decodeUImmOperand<8>";
}

def uimm12_sfpu : Operand<i32>, TImmLeaf<i32, [{return isUInt<12>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<12>;
  let DecoderMethod = "decodeUImmOperand<12>";
}

def uimm13_sfpu : Operand<i32>, TImmLeaf<i32, [{return isUInt<13>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<13>;
  let DecoderMethod = "decodeUImmOperand<13>";
}

def uimm14_sfpu : Operand<i32>, TImmLeaf<i32, [{return isUInt<14>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<14>;
  let DecoderMethod = "decodeUImmOperand<14>";
}

def uimm16_sfpu : Operand<i32>, TImmLeaf<i32, [{return isUInt<16>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<16>;
  let DecoderMethod = "decodeUImmOperand<16>";
}

//===----------------------------------------------------------------------===//
// Encoding format base classes
//
// All Tensix instructions: bits[31:24] = opcode, bits[1:0] = 0b11 (32-bit).
// We use InstFormatOther to bypass standard RISC-V R/I/S/B format classes.
//===----------------------------------------------------------------------===//

// Base class for all Tensix SFPU instructions.
class RVInstTensix<bits<8> opcode, dag outs, dag ins, string opcodestr,
                   string argstr, list<dag> pattern>
    : RVInst<outs, ins, opcodestr, argstr, pattern, InstFormatOther> {
  let Inst{31-24} = opcode;
  let Inst{1-0} = 0b11;     // 32-bit instruction marker
  let DecoderNamespace = "XttSFPU";
  let Size = 4;
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 1;   // SFPU instructions have coprocessor side effects
}

//---
// Standard Unary Format (18 instructions):
//   [31:24] opcode | [23:12] imm12 | [11:8] lreg_c | [7:4] lreg_dest | [3:0] mod1
//---
class SFPUUnary<bits<8> opcode, string opcodestr, dag outs, dag ins,
                string argstr, list<dag> pattern = []>
    : RVInstTensix<opcode, outs, ins, opcodestr, argstr, pattern> {
  bits<12> imm12;
  bits<4> lreg_c;
  bits<4> lreg_dest;
  bits<4> mod1;

  let Inst{23-12} = imm12;
  let Inst{11-8}  = lreg_c;
  let Inst{7-4}   = lreg_dest;
  let Inst{3-0}   = mod1;
}

// Variant with explicit outs/ins for typical unary: dest = f(src_c, imm, mod)
class SFPUUnaryReg<bits<8> opcode, string opcodestr>
    : SFPUUnary<opcode, opcodestr,
                (outs SFPURegs:$lreg_dest),
                (ins uimm12_sfpu:$imm12, SFPUAllRegs:$lreg_c,
                     uimm4_sfpu:$mod1),
                "$lreg_dest, $lreg_c, $imm12, $mod1">;

// Variant for CC/control instructions that don't write a register.
// $lreg_dest is an immediate selector (bits[7:4]), NOT a register destination.
// For SFPSETCC it selects the CC mode, for SFPPUSHC/SFPPOPC it's unused (0).
class SFPUUnaryCC<bits<8> opcode, string opcodestr>
    : SFPUUnary<opcode, opcodestr,
                (outs),
                (ins uimm12_sfpu:$imm12, SFPUAllRegs:$lreg_c,
                     uimm4_sfpu:$lreg_dest, uimm4_sfpu:$mod1),
                "$lreg_dest, $lreg_c, $imm12, $mod1">;

//---
// 3-Operand Format (SFPMAD, SFPADD, SFPMUL):
//   [31:24] opcode | [23:20] EMPTY | [19:16] src_a | [15:12] src_b |
//   [11:8] src_c | [7:4] dest | [3:0] mod1
//
// VALIDATED against sfpu-ops-bh.h:
//   GCC: (lreg_src_a << 16) + (lreg_src_b << 12) + (lreg_src_c << 8)
//        + (lreg_dest << 4) + (instr_mod1 << 0)
//   Bits [23:20] are EMPTY (not src_a as FUNCTIONAL_UNITS.md suggested).
//---
class SFPU3Op<bits<8> opcode, string opcodestr>
    : RVInstTensix<opcode,
                   (outs SFPURegs:$lreg_dest),
                   (ins SFPUAllRegs:$lreg_src_a, SFPUAllRegs:$lreg_src_b,
                        SFPUAllRegs:$lreg_src_c, uimm4_sfpu:$mod1),
                   opcodestr,
                   "$lreg_dest, $lreg_src_a, $lreg_src_b, $lreg_src_c, $mod1",
                   []> {
  bits<4> lreg_src_a;
  bits<4> lreg_src_b;
  bits<4> lreg_src_c;
  bits<4> lreg_dest;
  bits<4> mod1;

  let Inst{23-20} = 0b0000;     // Empty (GCC: nothing at << 20)
  let Inst{19-16} = lreg_src_a; // GCC: lreg_src_a << 16
  let Inst{15-12} = lreg_src_b; // GCC: lreg_src_b << 12
  let Inst{11-8}  = lreg_src_c; // GCC: lreg_src_c << 8
  let Inst{7-4}   = lreg_dest;  // GCC: lreg_dest << 4
  let Inst{3-0}   = mod1;       // GCC: instr_mod1 << 0
}

//---
// Load/Store BH Format:
//   [31:24] opcode | [23:20] lreg | [19:16] mod0 | [15:13] addr_mode(3b) |
//   [12:0] addr(13b)
//---
class SFPULoadBH<bits<8> opcode, string opcodestr, dag outs, dag ins,
                 string argstr>
    : RVInstTensix<opcode, outs, ins, opcodestr, argstr, []> {
  bits<4> lreg_ind;
  bits<4> mod0;
  bits<3> addr_mode;
  bits<13> addr;

  let Inst{23-20} = lreg_ind;
  let Inst{19-16} = mod0;
  let Inst{15-13} = addr_mode;
  let Inst{12-0}  = addr;
}

//---
// Load/Store WH Format:
//   [31:24] opcode | [23:20] lreg | [19:16] mod0 | [15:14] addr_mode(2b) |
//   [13:0] addr(14b)
//---
class SFPULoadWH<bits<8> opcode, string opcodestr, dag outs, dag ins,
                 string argstr>
    : RVInstTensix<opcode, outs, ins, opcodestr, argstr, []> {
  bits<4> lreg_ind;
  bits<4> mod0;
  bits<2> addr_mode;
  bits<14> addr;

  let Inst{23-20} = lreg_ind;
  let Inst{19-16} = mod0;
  let Inst{15-14} = addr_mode;
  let Inst{13-0}  = addr;
}

//---
// Immediate-16 Format (SFPMULI, SFPADDI, SFPCONFIG):
//   [31:24] opcode | [23:8] imm16 | [7:4] dest | [3:0] mod1
//
// VALIDATED against sfpu-ops-bh.h:
//   GCC SFPMULI: (imm16_math << 8) + (lreg_dest << 4) + (instr_mod1 << 0)
//   GCC SFPCONFIG: (imm16_math << 8) + (config_dest << 4) + (instr_mod1 << 0)
//
// NOTE: SFPLOADI does NOT use this format! It uses SFPULoadI format below.
//---
class SFPUImm16<bits<8> opcode, string opcodestr, dag outs, dag ins,
                string argstr>
    : RVInstTensix<opcode, outs, ins, opcodestr, argstr, []> {
  bits<16> imm16;
  bits<4> lreg_dest;
  bits<4> mod1;

  let Inst{23-8} = imm16;
  let Inst{7-4}  = lreg_dest;
  let Inst{3-0}  = mod1;
}

//---
// SFPLOADI Format (0x71) — Load-style with imm16 at [15:0]:
//   [31:24] opcode | [23:20] lreg_ind | [19:16] instr_mod0 | [15:0] imm16
//
// VALIDATED against sfpu-ops-bh.h:
//   GCC: (lreg_ind << 20) + (instr_mod0 << 16) + (imm16 << 0)
//   SFPLOADI uses the Load format header (lreg + mod0) but with imm16
//   packed at [15:0] instead of addr_mode + addr. This is NOT Imm16 format.
//---
class SFPULoadI<bits<8> opcode, string opcodestr, dag outs, dag ins,
                string argstr>
    : RVInstTensix<opcode, outs, ins, opcodestr, argstr, []> {
  bits<4> lreg_ind;
  bits<4> mod0;
  bits<16> imm16;

  let Inst{23-20} = lreg_ind;   // GCC: lreg_ind << 20
  let Inst{19-16} = mod0;       // GCC: instr_mod0 << 16
  let Inst{15-0}  = imm16;      // GCC: imm16 << 0
}

//---
// SFPLUT Format (0x73) — Load-style with no addr_mode field:
//   [31:24] opcode | [23:20] lreg_ind | [19:16] instr_mod0 | [15:0] dest_reg_addr
//
// VALIDATED against sfpu-ops-bh.h:
//   GCC: (lreg_ind << 20) + (instr_mod0 << 16) + (dest_reg_addr << 0)
//   SFPLUT has NO addr_mode field. dest_reg_addr is a full 16-bit field.
//---
class SFPULut<bits<8> opcode, string opcodestr, dag outs, dag ins,
              string argstr>
    : RVInstTensix<opcode, outs, ins, opcodestr, argstr, []> {
  bits<4> lreg_ind;
  bits<4> mod0;
  bits<16> dest_reg_addr;

  let Inst{23-20} = lreg_ind;       // GCC: lreg_ind << 20
  let Inst{19-16} = mod0;           // GCC: instr_mod0 << 16
  let Inst{15-0}  = dest_reg_addr;  // GCC: dest_reg_addr << 0
}

//---
// SFPLOADMACRO BH Format (0x93) — addr_mode is 2-bit even on BH (C-012):
//   [31:24] opcode | [23:20] lreg | [19:16] mod0 | [15:14] addr_mode(2b) |
//   [13:0] addr(14b)
//
// VALIDATED against sfpu-ops-bh.h:
//   GCC: (lreg_ind << 20) + (instr_mod0 << 16) + (sfpu_addr_mode << 14)
//        + (dest_reg_addr << 0)
//   This is DIFFERENT from SFPLOAD BH which has 3-bit addr_mode!
//---
class SFPULoadMacroBH<bits<8> opcode, string opcodestr, dag outs, dag ins,
                      string argstr>
    : RVInstTensix<opcode, outs, ins, opcodestr, argstr, []> {
  bits<4> lreg_ind;
  bits<4> mod0;
  bits<2> addr_mode;
  bits<14> addr;

  let Inst{23-20} = lreg_ind;   // GCC: lreg_ind << 20
  let Inst{19-16} = mod0;       // GCC: instr_mod0 << 16
  let Inst{15-14} = addr_mode;  // GCC: sfpu_addr_mode << 14 (2-bit!)
  let Inst{13-0}  = addr;       // GCC: dest_reg_addr << 0
}

//---
// Stochastic Rounding Format (SFP_STOCH_RND, 0x8E):
//   [31:24] opcode | [23:21] rnd_mode(3b) | [20:16] imm(5b) |
//   [15:12] lreg_src_b | [11:8] lreg_src_c | [7:4] lreg_dest | [3:0] mod1
//
// VALIDATED against sfpu-ops-bh.h:
//   GCC: (rnd_mode << 21) + (imm8_math << 16) + (lreg_src_b << 12)
//        + (lreg_src_c << 8) + (lreg_dest << 4) + (instr_mod1 << 0)
//   Note: 6 operands including src_c! rnd_mode is 3 bits, "imm8" is 5 bits.
//   The name "imm8_math" in GCC is misleading — only 5 bits fit.
//---
class SFPUStochRnd<bits<8> opcode, string opcodestr>
    : RVInstTensix<opcode,
                   (outs SFPURegs:$lreg_dest),
                   (ins uimm3_sfpu:$rnd_mode, uimm5_sfpu:$imm5,
                        SFPUAllRegs:$lreg_src_b, SFPUAllRegs:$lreg_src_c,
                        uimm4_sfpu:$mod1),
                   opcodestr,
                   "$lreg_dest, $rnd_mode, $imm5, $lreg_src_b, $lreg_src_c, $mod1",
                   []> {
  bits<3> rnd_mode;
  bits<5> imm5;
  bits<4> lreg_src_b;
  bits<4> lreg_src_c;
  bits<4> lreg_dest;
  bits<4> mod1;

  let Inst{23-21} = rnd_mode;   // GCC: rnd_mode << 21
  let Inst{20-16} = imm5;       // GCC: imm8_math << 16 (only 5 bits fit)
  let Inst{15-12} = lreg_src_b; // GCC: lreg_src_b << 12
  let Inst{11-8}  = lreg_src_c; // GCC: lreg_src_c << 8
  let Inst{7-4}   = lreg_dest;  // GCC: lreg_dest << 4
  let Inst{3-0}   = mod1;       // GCC: instr_mod1 << 0
}

//---
// SFPLUTFP32 Special Format (0x95):
//   Uses the standard unary encoding but writes up to 3 registers (L0, L1, L7).
//   Modeled as a single instruction with implicit defs.
//---
class SFPULutFP32<bits<8> opcode, string opcodestr>
    : SFPUUnary<opcode, opcodestr,
                (outs SFPURegs:$lreg_dest),
                (ins SFPUAllRegs:$lreg_c, uimm4_sfpu:$mod1),
                "$lreg_dest, $lreg_c, $mod1"> {
  let imm12 = 0;
  // SFPLUTFP32 may write L0, L1, L7 depending on mode.
  let Defs = [L0, L1, L7];
}

//---
// SFPNOP Format (0x8F): No operands, all payload bits zero.
//---
class SFPUNop<bits<8> opcode, string opcodestr>
    : RVInstTensix<opcode, (outs), (ins), opcodestr, "", []> {
  let Inst{23-0} = 0;
}

//===----------------------------------------------------------------------===//
// Instruction Definitions — Base SFPU (0x70-0x95)
// All 38 opcodes present on both Wormhole and Blackhole.
//===----------------------------------------------------------------------===//

let Predicates = [HasXttSFPU] in {

//--- Load/Store Instructions (BH encoding — primary target) ---

def SFPLOAD_BH : SFPULoadBH<0x70, "sfpload",
    (outs SFPURegs:$lreg_ind),
    (ins uimm4_sfpu:$mod0, uimm3_sfpu:$addr_mode, uimm13_sfpu:$addr),
    "$lreg_ind, $mod0, $addr_mode, $addr"> {
  let mayLoad = 1;
  let Defs = [];
}

def SFPLOADI : SFPULoadI<0x71, "sfploadi",
    (outs SFPURegs:$lreg_ind),
    (ins uimm4_sfpu:$mod0, uimm16_sfpu:$imm16),
    "$lreg_ind, $mod0, $imm16"> {
  let mayLoad = 1;
}

def SFPSTORE_BH : SFPULoadBH<0x72, "sfpstore",
    (outs),
    (ins SFPUStoreRegs:$lreg_ind, uimm4_sfpu:$mod0,
         uimm3_sfpu:$addr_mode, uimm13_sfpu:$addr),
    "$lreg_ind, $mod0, $addr_mode, $addr"> {
  let mayStore = 1;
}

def SFPLUT : SFPULut<0x73, "sfplut",
    (outs SFPURegs:$lreg_ind),
    (ins uimm4_sfpu:$mod0, uimm16_sfpu:$dest_reg_addr),
    "$lreg_ind, $mod0, $dest_reg_addr"> {
  let mayLoad = 1;
}

def SFPLOADMACRO_BH : SFPULoadMacroBH<0x93, "sfploadmacro",
    (outs),
    (ins SFPUAllRegs:$lreg_ind, uimm4_sfpu:$mod0,
         uimm2_sfpu:$addr_mode, uimm14_sfpu:$addr),
    "$lreg_ind, $mod0, $addr_mode, $addr"> {
  let mayLoad = 1;
  // Writes L16 (macro staging register)
  let Defs = [L16];
}

//--- Immediate-16 Arithmetic Instructions ---

def SFPMULI : SFPUImm16<0x74, "sfpmuli",
    (outs SFPURegs:$lreg_dest),
    (ins uimm16_sfpu:$imm16, uimm4_sfpu:$mod1),
    "$lreg_dest, $imm16, $mod1"> {
}

def SFPADDI : SFPUImm16<0x75, "sfpaddi",
    (outs SFPURegs:$lreg_dest),
    (ins uimm16_sfpu:$imm16, uimm4_sfpu:$mod1),
    "$lreg_dest, $imm16, $mod1"> {
}

// Pseudo variants with tied src/dest for ISel (read-modify-write semantics).
// The intrinsic passes the dest register as an input because it's both read and
// written. The Constraints tie ensures regalloc keeps src and dest in the same
// physical register.
let Constraints = "$lreg_dest = $lreg_src" in {
  def PseudoSFPMULI : SFPUImm16<0x74, "sfpmuli",
      (outs SFPURegs:$lreg_dest),
      (ins SFPURegs:$lreg_src, uimm16_sfpu:$imm16, uimm4_sfpu:$mod1),
      "$lreg_dest, $imm16, $mod1">;
  def PseudoSFPADDI : SFPUImm16<0x75, "sfpaddi",
      (outs SFPURegs:$lreg_dest),
      (ins SFPURegs:$lreg_src, uimm16_sfpu:$imm16, uimm4_sfpu:$mod1),
      "$lreg_dest, $imm16, $mod1">;
}

//--- Standard Unary Instructions (simple unit, 1 cycle) ---

def SFPDIVP2  : SFPUUnaryReg<0x76, "sfpdivp2">;

def SFPEXEXP  : SFPUUnaryReg<0x77, "sfpexexp">;

def SFPEXMAN  : SFPUUnaryReg<0x78, "sfpexman">;

def SFPIADD   : SFPUUnaryReg<0x79, "sfpiadd">;

def SFPSHFT   : SFPUUnaryReg<0x7A, "sfpshft">;

def SFPSETCC  : SFPUUnaryCC<0x7B, "sfpsetcc">;

def SFPMOV    : SFPUUnaryReg<0x7C, "sfpmov">;

def SFPABS    : SFPUUnaryReg<0x7D, "sfpabs">;

def SFPAND    : SFPUUnaryReg<0x7E, "sfpand">;

def SFPOR     : SFPUUnaryReg<0x7F, "sfpor">;

def SFPNOT    : SFPUUnaryReg<0x80, "sfpnot">;

def SFPLZ     : SFPUUnaryReg<0x81, "sfplz">;

def SFPSETEXP : SFPUUnaryReg<0x82, "sfpsetexp">;

def SFPSETMAN : SFPUUnaryReg<0x83, "sfpsetman">;

//--- 3-Operand Arithmetic Instructions (MAD unit, 2 cycles) ---

def SFPMAD : SFPU3Op<0x84, "sfpmad">;

def SFPADD : SFPU3Op<0x85, "sfpadd">;

def SFPMUL : SFPU3Op<0x86, "sfpmul">;

//--- CC Stack / Predication Instructions (simple unit, 1 cycle) ---

def SFPPUSHC  : SFPUUnaryCC<0x87, "sfppushc">;

def SFPPOPC   : SFPUUnaryCC<0x88, "sfppopc">;

def SFPSETSGN : SFPUUnaryReg<0x89, "sfpsetsgn">;

def SFPENCC   : SFPUUnaryCC<0x8A, "sfpencc">;

def SFPCOMPC  : SFPUUnaryCC<0x8B, "sfpcompc">;

//--- Transpose / Cross-Lane (simple unit, 1 cycle) ---

def SFPTRANSP : SFPUUnaryReg<0x8C, "sfptransp">;

def SFPXOR    : SFPUUnaryReg<0x8D, "sfpxor">;

//--- Stochastic Rounding (round unit, 1 cycle) ---

def SFP_STOCH_RND : SFPUStochRnd<0x8E, "sfpstochrnd">;

//--- NOP ---

def SFPNOP : SFPUNop<0x8F, "sfpnop">;

//--- SFPCAST (unary format, round unit, 1 cycle) ---
// mod1 selects: FP32_TO_FP16A, FP32_TO_FP16B, FP16A_TO_FP32, FP16B_TO_FP32,
//               SM32_TO_INT32 (BH), INT32_TO_SM32 (BH)  -- per C-019

def SFPCAST : SFPUUnaryReg<0x90, "sfpcast">;

//--- SFPCONFIG (imm16 format, simple unit, 1 cycle) ---
// Writes immediate to config registers or L11-L14 (per C-023)

def SFPCONFIG : SFPUImm16<0x91, "sfpconfig",
    (outs),
    (ins uimm16_sfpu:$imm16, uimm4_sfpu:$lreg_dest, uimm4_sfpu:$mod1),
    "$lreg_dest, $imm16, $mod1"> {
}

//--- SFPSWAP (unary format, simple unit, 2 cycles) ---
// Exchanges VD <-> VC, or min/max/argmin/argmax per mod1 (C-025)

def SFPSWAP : SFPUUnaryReg<0x92, "sfpswap">;

//--- SFPSHFT2 (unary format, simple unit, 1-2 cycles) ---
// Cross-lane shift/rotate. WH SHFLSHR1 has zero-fill bug (E-002).

def SFPSHFT2 : SFPUUnaryReg<0x94, "sfpshft2">;

//--- SFPLUTFP32 (special unary format, MAD unit, 2 cycles) ---
// LUT lookup with FP32 interpolation. Writes L0, L1, L7.

def SFPLUTFP32 : SFPULutFP32<0x95, "sfplutfp32">;

//--- REPLAY (Tensix opcode 0x04, not SFPU but used by SFPU optimization) ---
// REPLAY: [31:24]0x04 | [23:14]start_idx | [13:4]len | [3:2]reserved |
//         [1]execute_while_loading | [0]load_mode
//
// Validated against sfpu-ops-bh.h:
//   TT_OP_BH(0x04, (start_idx << 14) + (len << 4) +
//            (execute_while_loading << 1) + (load_mode << 0))

def TTREPLAY : RVInstTensix<0x04,
    (outs),
    (ins uimm12_sfpu:$start_idx, uimm12_sfpu:$len,
         uimm4_sfpu:$exec_while_load, uimm4_sfpu:$load_mode),
    "replay",
    "$start_idx, $len, $exec_while_load, $load_mode",
    []> {
  bits<10> start_idx;
  bits<10> len;
  bits<1> exec_while_load;
  bits<1> load_mode;

  let Inst{23-14} = start_idx;
  let Inst{13-4}  = len;
  let Inst{3-2}   = 0b00;
  let Inst{1}     = exec_while_load;
  let Inst{0}     = load_mode;

  // Override the base class bits[1:0]=0b11 — REPLAY is a Tensix instruction,
  // not in the RISC-V custom space
  let Inst{1-0} = {exec_while_load, load_mode};

  let hasSideEffects = 1;
}

} // Predicates = [HasXttSFPU]

//===----------------------------------------------------------------------===//
// WH-specific Load/Store encodings (2-bit addr_mode, 14-bit addr)
// These are alternative encodings for the same opcodes, selected by subtarget.
//===----------------------------------------------------------------------===//

let Predicates = [HasXttSFPUWH] in {

def SFPLOAD_WH : SFPULoadWH<0x70, "sfpload",
    (outs SFPURegs:$lreg_ind),
    (ins uimm4_sfpu:$mod0, uimm2_sfpu:$addr_mode, uimm14_sfpu:$addr),
    "$lreg_ind, $mod0, $addr_mode, $addr"> {
  let mayLoad = 1;
}

def SFPSTORE_WH : SFPULoadWH<0x72, "sfpstore",
    (outs),
    (ins SFPUStoreRegs:$lreg_ind, uimm4_sfpu:$mod0,
         uimm2_sfpu:$addr_mode, uimm14_sfpu:$addr),
    "$lreg_ind, $mod0, $addr_mode, $addr"> {
  let mayStore = 1;
}

// SFPLUT uses the same format on WH and BH (no addr_mode field)
// So we don't need a separate WH definition — SFPLUT above is arch-independent.

def SFPLOADMACRO_WH : SFPULoadWH<0x93, "sfploadmacro",
    (outs),
    (ins SFPUAllRegs:$lreg_ind, uimm4_sfpu:$mod0,
         uimm2_sfpu:$addr_mode, uimm14_sfpu:$addr),
    "$lreg_ind, $mod0, $addr_mode, $addr"> {
  let mayLoad = 1;
  let Defs = [L16];
}

} // Predicates = [HasXttSFPUWH]

//===----------------------------------------------------------------------===//
// BH-Only Instructions (C-009)
//===----------------------------------------------------------------------===//

let Predicates = [HasXttSFPU, HasXttSFPUBH] in {

// SFPMUL24: 23b x 23b integer multiply (C-026)
// Note: src_c must be L9 (zero constant) due to hardware bug (GH-E-001)
def SFPMUL24 : SFPU3Op<0x96, "sfpmul24">;

// SFPARECIP: Approximate reciprocal (removed on QS, replaced by SFPNONLINEAR)
def SFPARECIP : SFPUUnaryReg<0x99, "sfparecip">;

// SFPGT: Greater-than comparison (sets CC) -- GH-Q-005
def SFPGT : SFPUUnaryCC<0x9A, "sfpgt">;

// SFPLE: Less-or-equal comparison (sets CC) -- GH-Q-005
def SFPLE : SFPUUnaryCC<0x9B, "sfple">;

// SFPMOV_CONFIG: Read configuration register via MOV
def SFPMOV_CONFIG : SFPUUnaryReg<0x9C, "sfpmov_config">;

} // Predicates = [HasXttSFPU, HasXttSFPUBH]

//===----------------------------------------------------------------------===//
// Pseudo-instructions for common patterns
//===----------------------------------------------------------------------===//

let Predicates = [HasXttSFPU] in {

// SFPMOV with no immediate (common case: sfpmov dst, src, 0, mod1)
let isAsCheapAsAMove = 1, isReMaterializable = 1 in
def SFPMOV_REG : SFPUUnary<0x7C, "sfpmov",
    (outs SFPURegs:$lreg_dest),
    (ins SFPUAllRegs:$lreg_c, uimm4_sfpu:$mod1),
    "$lreg_dest, $lreg_c, $mod1"> {
  let imm12 = 0;
}

// SFPNOP is trivially dead — safe to eliminate
def : InstAlias<"sfpnop", (SFPNOP)>;

} // Predicates = [HasXttSFPU]

//===----------------------------------------------------------------------===//
// Live-Value (_lv) Instruction Variants (Phase 5: CC Stack Predication)
//
// When a register is "live" across a predicated region (v_if/v_else/v_endif),
// the _lv variant must be used. It takes an extra operand: the previous value
// of the destination register. The hardware uses this to preserve per-lane
// values in lanes disabled by the condition code predicate.
//
// In GCC, _lv variants have flag 0x02 (bit 1) and an extra XTT_VEC operand
// prepended before the normal operands. The _lv variant emits the same opcode
// — the register allocator constrains dest to match the live-value operand.
//
// Validated against: sfpi-gcc/gcc/config/riscv/tt/rvtt-insn.def
//   Each instruction with _lv has: flags |= 0x02, extra XTT_VEC arg at front
//===----------------------------------------------------------------------===//

let Predicates = [HasXttSFPU] in {

// _lv Unary: dest must be tied to live_val (same register)
class SFPUUnaryRegLV<bits<8> opcode, string opcodestr>
    : SFPUUnary<opcode, opcodestr,
                (outs SFPURegs:$lreg_dest),
                (ins SFPURegs:$live_val, uimm12_sfpu:$imm12,
                     SFPUAllRegs:$lreg_c, uimm4_sfpu:$mod1),
                "$lreg_dest, $live_val, $lreg_c, $imm12, $mod1"> {
  let Constraints = "$lreg_dest = $live_val";
}

// _lv 3-Op: dest must be tied to live_val
class SFPU3OpLV<bits<8> opcode, string opcodestr>
    : RVInstTensix<opcode,
                   (outs SFPURegs:$lreg_dest),
                   (ins SFPURegs:$live_val, SFPUAllRegs:$lreg_src_a,
                        SFPUAllRegs:$lreg_src_b, SFPUAllRegs:$lreg_src_c,
                        uimm4_sfpu:$mod1),
                   opcodestr,
                   "$lreg_dest, $live_val, $lreg_src_a, $lreg_src_b, $lreg_src_c, $mod1",
                   []> {
  bits<4> lreg_src_a;
  bits<4> lreg_src_b;
  bits<4> lreg_src_c;
  bits<4> lreg_dest;
  bits<4> mod1;

  let Inst{23-20} = 0b0000;
  let Inst{19-16} = lreg_src_a;
  let Inst{15-12} = lreg_src_b;
  let Inst{11-8}  = lreg_src_c;
  let Inst{7-4}   = lreg_dest;
  let Inst{3-0}   = mod1;

  let Constraints = "$lreg_dest = $live_val";
}

// --- Standard Unary _lv variants ---

def SFPMOV_LV    : SFPUUnaryRegLV<0x7C, "sfpmov">;
def SFPABS_LV    : SFPUUnaryRegLV<0x7D, "sfpabs">;
def SFPEXEXP_LV  : SFPUUnaryRegLV<0x77, "sfpexexp">;
def SFPEXMAN_LV  : SFPUUnaryRegLV<0x78, "sfpexman">;
def SFPNOT_LV    : SFPUUnaryRegLV<0x80, "sfpnot">;
def SFPLZ_LV     : SFPUUnaryRegLV<0x81, "sfplz">;
def SFPDIVP2_LV  : SFPUUnaryRegLV<0x76, "sfpdivp2">;
def SFPSETEXP_LV : SFPUUnaryRegLV<0x82, "sfpsetexp">;
def SFPSETMAN_LV : SFPUUnaryRegLV<0x83, "sfpsetman">;
def SFPSETSGN_LV : SFPUUnaryRegLV<0x89, "sfpsetsgn">;
def SFPCAST_LV   : SFPUUnaryRegLV<0x90, "sfpcast">;
def SFPSHFT2_LV  : SFPUUnaryRegLV<0x94, "sfpshft2">;

// --- 3-Operand _lv variants ---

def SFPMAD_LV : SFPU3OpLV<0x84, "sfpmad">;
def SFPADD_LV : SFPU3OpLV<0x85, "sfpadd">;
def SFPMUL_LV : SFPU3OpLV<0x86, "sfpmul">;

// --- Load _lv variant ---

def SFPLOAD_BH_LV : SFPULoadBH<0x70, "sfpload",
    (outs SFPURegs:$lreg_ind),
    (ins SFPURegs:$live_val, uimm4_sfpu:$mod0,
         uimm3_sfpu:$addr_mode, uimm13_sfpu:$addr),
    "$lreg_ind, $live_val, $mod0, $addr_mode, $addr"> {
  let mayLoad = 1;
  let Constraints = "$lreg_ind = $live_val";
}

} // Predicates = [HasXttSFPU]

// --- BH-only _lv variants ---

let Predicates = [HasXttSFPU, HasXttSFPUBH] in {

def SFPMUL24_LV : SFPU3OpLV<0x96, "sfpmul24">;

def SFPARECIP_LV : SFPUUnaryRegLV<0x99, "sfparecip">;

def SFPMOV_CONFIG_LV : SFPUUnaryRegLV<0x9C, "sfpmov_config">;

} // Predicates = [HasXttSFPU, HasXttSFPUBH]
