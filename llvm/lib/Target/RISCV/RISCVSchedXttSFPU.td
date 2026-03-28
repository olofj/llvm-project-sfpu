//===-- RISCVSchedXttSFPU.td - Tensix Scheduling Model ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
// Scheduling model for the Tenstorrent Tensix tile's Baby RISC-V scalar
// pipeline and SFPU vector unit.
//
// The Baby RISC-V is a single-issue, in-order RV32IMAC core. It dispatches
// Tensix instructions (SFPU, FPU, matrix) by writing 32-bit instruction
// words to a memory-mapped FIFO. Better scalar scheduling = fewer pipeline
// bubbles in the dispatch path.
//
// Scalar pipeline (PIPELINE.md §2):
//   ALU:     1 cycle (EX1 stage, full forwarding)
//   MUL:     2 cycles (pipeline-stalling — blocks all issue)
//   DIV:     2-33 cycles (pipeline-stalling, iterative)
//   Load:    2 cycles (local data RAM) to 12 cycles (L1 SRAM)
//   Store:   1 cycle (store queue absorbs immediately)
//   Branch:  1 cycle + 5-cycle mispredict penalty
//
// SFPU pipeline (FUNCTIONAL_UNITS.md §3):
//   Simple:  1 cycle (MOV, logic, comparisons, CC stack)
//   MAD:     2 cycles (SFPMAD, SFPADD, SFPMUL, SFPMULI, SFPADDI)
//   Load:    1 cycle (SFPLOAD, SFPLOADI)
//   Store:   1 cycle (SFPSTORE)
//   Swap:    2 cycles (SFPSWAP)
//   LUTFP32: 2 cycles (SFPLUTFP32)
//
// Reference: ttsim-analysis/PIPELINE.md, FUNCTIONAL_UNITS.md, rvtt-tune.md
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Processor Resources
//===----------------------------------------------------------------------===//

// Scalar Baby RISC-V pipeline
def TensixUnitALU    : ProcResource<1> { let BufferSize = 0; }  // EX1: ALU, shifts, compares
def TensixUnitMem    : ProcResource<1> { let BufferSize = 0; }  // Load/store unit
def TensixUnitBranch : ProcResource<1> { let BufferSize = 0; }  // Branch resolution
def TensixUnitIMul   : ProcResource<1> { let BufferSize = 0; }  // MUL (stalls pipeline)
def TensixUnitIDiv   : ProcResource<1> { let BufferSize = 0; }  // DIV (stalls pipeline)

// SFPU functional units
def SFPULoadUnit   : ProcResource<1> { let BufferSize = 0; }
def SFPUSimpleUnit : ProcResource<1> { let BufferSize = 0; }
def SFPUMADUnit    : ProcResource<1> { let BufferSize = 0; }
def SFPURoundUnit  : ProcResource<1> { let BufferSize = 0; }
def SFPUStoreUnit  : ProcResource<1> { let BufferSize = 0; }

//===----------------------------------------------------------------------===//
// SchedWrite / SchedRead Classes (SFPU)
//===----------------------------------------------------------------------===//

def WriteSFPUSimple  : SchedWrite;  // MOV, logic, comparisons, CC stack, shifts
def WriteSFPULoad    : SchedWrite;  // SFPLOAD, SFPLOADI, SFPLUT, SFPLOADMACRO
def WriteSFPUStore   : SchedWrite;  // SFPSTORE
def WriteSFPURound   : SchedWrite;  // SFP_STOCH_RND, SFPCAST
def WriteSFPUMAD     : SchedWrite;  // SFPMAD, SFPADD, SFPMUL, SFPMULI, SFPADDI
def WriteSFPUSwap    : SchedWrite;  // SFPSWAP
def WriteSFPULUTFP32 : SchedWrite;  // SFPLUTFP32
def WriteSFPUShft2   : SchedWrite;  // SFPSHFT2
def WriteSFPUNop     : SchedWrite;  // SFPNOP
def ReadSFPUSrc      : SchedRead;

//===----------------------------------------------------------------------===//
// Blackhole Scheduling Model
//===----------------------------------------------------------------------===//

def TensixBHScalarModel : SchedMachineModel {
  let IssueWidth = 1;            // Single-issue in-order
  let MicroOpBufferSize = 0;     // No reorder buffer
  let LoadLatency = 2;           // Default: local data RAM (2 cycles)
  let MispredictPenalty = 5;     // 5-cycle frontend refill
  let CompleteModel = false;     // Allow instructions without sched info
  let PostRAScheduler = 1;       // Enable post-RA scheduling
  let UnsupportedFeatures = [HasStdExtZbkb, HasStdExtZbkc, HasStdExtZbkx,
                             HasStdExtZcmt, HasStdExtZknd, HasStdExtZkne,
                             HasStdExtZknh, HasStdExtZksed, HasStdExtZksh,
                             HasStdExtZkr, HasVInstructions, HasVInstructionsI64];
}

let SchedModel = TensixBHScalarModel in {

//--- Scalar WriteRes entries ---

// Branch / jump (1 cycle, resolved in EX1)
def : WriteRes<WriteJmp,  [TensixUnitBranch]>;
def : WriteRes<WriteJal,  [TensixUnitBranch]>;
def : WriteRes<WriteJalr, [TensixUnitBranch]>;

// Integer ALU (1 cycle)
def : WriteRes<WriteIALU,       [TensixUnitALU]>;
def : WriteRes<WriteIALU32,     [TensixUnitALU]>;
def : WriteRes<WriteShiftImm,   [TensixUnitALU]>;
def : WriteRes<WriteShiftImm32, [TensixUnitALU]>;
def : WriteRes<WriteShiftReg,   [TensixUnitALU]>;
def : WriteRes<WriteShiftReg32, [TensixUnitALU]>;

// Integer Multiply (2 cycles, stalls pipeline)
def : WriteRes<WriteIMul,   [TensixUnitIMul]> { let Latency = 2; let ReleaseAtCycles = [2]; }
def : WriteRes<WriteIMul32, [TensixUnitIMul]> { let Latency = 2; let ReleaseAtCycles = [2]; }

// Integer Divide (2-33 cycles, stalls pipeline)
def : WriteRes<WriteIDiv,   [TensixUnitIDiv]> { let Latency = 33; let ReleaseAtCycles = [33]; }
def : WriteRes<WriteIDiv32, [TensixUnitIDiv]> { let Latency = 33; let ReleaseAtCycles = [33]; }
def : WriteRes<WriteIRem,   [TensixUnitIDiv]> { let Latency = 33; let ReleaseAtCycles = [33]; }
def : WriteRes<WriteIRem32, [TensixUnitIDiv]> { let Latency = 33; let ReleaseAtCycles = [33]; }

// Load (2 cycles default — local data RAM; L1 is 8-12 but we use conservative default)
def : WriteRes<WriteLDB, [TensixUnitMem]> { let Latency = 2; }
def : WriteRes<WriteLDH, [TensixUnitMem]> { let Latency = 2; }
def : WriteRes<WriteLDW, [TensixUnitMem]> { let Latency = 2; }

// Store (1 cycle — store queue absorbs immediately)
def : WriteRes<WriteSTB, [TensixUnitMem]>;
def : WriteRes<WriteSTH, [TensixUnitMem]>;
def : WriteRes<WriteSTW, [TensixUnitMem]>;

// Atomics (12 cycles for L1 atomic operations)
def : WriteRes<WriteAtomicW,   [TensixUnitMem]> { let Latency = 12; let ReleaseAtCycles = [12]; }
def : WriteRes<WriteAtomicD,   [TensixUnitMem]> { let Latency = 12; let ReleaseAtCycles = [12]; } // rv32: unused
def : WriteRes<WriteAtomicLDW, [TensixUnitMem]> { let Latency = 12; let ReleaseAtCycles = [12]; }
def : WriteRes<WriteAtomicLDD, [TensixUnitMem]> { let Latency = 12; let ReleaseAtCycles = [12]; } // rv32: unused
def : WriteRes<WriteAtomicSTW, [TensixUnitMem]>;
def : WriteRes<WriteAtomicSTD, [TensixUnitMem]>; // rv32: unused

// 64-bit load/store (rv32: unused but required by scheduler model for A extension)
def : WriteRes<WriteLDD, [TensixUnitMem]> { let Latency = 2; }
def : WriteRes<WriteSTD, [TensixUnitMem]>;

// CSR / NOP (1 cycle, no functional unit)
def : WriteRes<WriteCSR, []>;
def : WriteRes<WriteNop, []>;

// COPY pseudo (modeled as ALU)
def : InstRW<[WriteIALU], (instrs COPY)>;

//--- Scalar ReadAdvance entries (all operands read at issue time) ---
def : ReadAdvance<ReadJmp, 0>;
def : ReadAdvance<ReadJalr, 0>;
def : ReadAdvance<ReadCSR, 0>;
def : ReadAdvance<ReadStoreData, 0>;
def : ReadAdvance<ReadMemBase, 0>;
def : ReadAdvance<ReadIALU, 0>;
def : ReadAdvance<ReadIALU32, 0>;
def : ReadAdvance<ReadShiftImm, 0>;
def : ReadAdvance<ReadShiftImm32, 0>;
def : ReadAdvance<ReadShiftReg, 0>;
def : ReadAdvance<ReadShiftReg32, 0>;
def : ReadAdvance<ReadIDiv, 0>;
def : ReadAdvance<ReadIDiv32, 0>;
def : ReadAdvance<ReadIRem, 0>;
def : ReadAdvance<ReadIRem32, 0>;
def : ReadAdvance<ReadIMul, 0>;
def : ReadAdvance<ReadIMul32, 0>;
def : ReadAdvance<ReadAtomicWA, 0>;
def : ReadAdvance<ReadAtomicWD, 0>;
def : ReadAdvance<ReadAtomicDA, 0>;
def : ReadAdvance<ReadAtomicDD, 0>;
def : ReadAdvance<ReadAtomicLDW, 0>;
def : ReadAdvance<ReadAtomicLDD, 0>;
def : ReadAdvance<ReadAtomicSTW, 0>;
def : ReadAdvance<ReadAtomicSTD, 0>;

//--- SFPU WriteRes entries ---
def : WriteRes<WriteSFPUSimple,  [SFPUSimpleUnit]>;
def : WriteRes<WriteSFPULoad,    [SFPULoadUnit]>;
def : WriteRes<WriteSFPUStore,   [SFPUStoreUnit]>;
def : WriteRes<WriteSFPURound,   [SFPURoundUnit]>;
def : WriteRes<WriteSFPUMAD,     [SFPUMADUnit]>    { let Latency = 2; }
def : WriteRes<WriteSFPUSwap,    [SFPUSimpleUnit]>  { let Latency = 2; }
def : WriteRes<WriteSFPULUTFP32, [SFPUMADUnit]>    { let Latency = 2; }
def : WriteRes<WriteSFPUShft2,   [SFPUSimpleUnit]>;
def : WriteRes<WriteSFPUNop,     []>;

def : ReadAdvance<ReadSFPUSrc, 0>;

//--- SFPU InstRW: map instructions to scheduling classes ---
// Simple unit (1 cycle): unary arithmetic, logic, CC stack, transposes
def : InstRW<[WriteSFPUSimple], (instrs
    SFPDIVP2, SFPEXEXP, SFPEXMAN, SFPIADD, SFPSHFT,
    SFPMOV, SFPABS, SFPAND, SFPOR, SFPNOT, SFPLZ,
    SFPSETEXP, SFPSETMAN, SFPSETSGN,
    SFPSETCC, SFPPUSHC, SFPPOPC, SFPENCC, SFPCOMPC,
    SFPTRANSP, SFPXOR, SFPCONFIG,
    // _lv variants
    SFPMOV_LV, SFPABS_LV, SFPEXEXP_LV, SFPEXMAN_LV,
    SFPDIVP2_LV, SFPLZ_LV, SFPNOT_LV,
    SFPSETEXP_LV, SFPSETMAN_LV, SFPSETSGN_LV, SFPSHFT2_LV
)>;

// MAD unit (2 cycles): 3-operand arithmetic + immediate arithmetic
def : InstRW<[WriteSFPUMAD], (instrs
    SFPMAD, SFPADD, SFPMUL, SFPMULI, SFPADDI,
    PseudoSFPMULI, PseudoSFPADDI,
    // WH constrained variants
    SFPMAD_WH, SFPADD_WH, SFPMUL_WH,
    // BH-only
    SFPMUL24,
    // _lv variants
    SFPMAD_LV, SFPADD_LV, SFPMUL_LV, SFPMUL24_LV
)>;

// Load unit (1 cycle)
def : InstRW<[WriteSFPULoad], (instrs
    SFPLOAD_BH, SFPLOADI, SFPLUT, SFPLOADMACRO_BH,
    SFPLOAD_BH_LV,
    // BH-only
    SFPARECIP, SFPARECIP_LV
)>;

// Store unit (1 cycle)
def : InstRW<[WriteSFPUStore], (instrs SFPSTORE_BH)>;

// Round unit (1 cycle): stochastic rounding, type cast
def : InstRW<[WriteSFPURound], (instrs SFP_STOCH_RND, SFPCAST, SFPCAST_LV)>;

// Swap (2 cycles)
def : InstRW<[WriteSFPUSwap], (instrs SFPSWAP)>;

// LUTFP32 (2 cycles)
def : InstRW<[WriteSFPULUTFP32], (instrs SFPLUTFP32)>;

// SHFT2 (1 cycle BH, 2 cycle WH — BH model uses 1)
def : InstRW<[WriteSFPUShft2], (instrs SFPSHFT2)>;

// NOP (0 cycles, no unit)
def : InstRW<[WriteSFPUNop], (instrs SFPNOP)>;

// CC-setting BH-only instructions
def : InstRW<[WriteSFPUSimple], (instrs SFPGT, SFPLE)>;

// REPLAY (Tensix scalar instruction, not SFPU — 1 cycle)
def : InstRW<[WriteIALU], (instrs TTREPLAY)>;

//--- Unsupported extensions (Tensix doesn't have these) ---
defm : UnsupportedSchedF;
defm : UnsupportedSchedD;
defm : UnsupportedSchedV;
defm : UnsupportedSchedZba;
defm : UnsupportedSchedZbb;
defm : UnsupportedSchedZbc;
defm : UnsupportedSchedZbs;
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedZfa;
defm : UnsupportedSchedZfh;
defm : UnsupportedSchedSFB;
defm : UnsupportedSchedXsfvcp;
defm : UnsupportedSchedZvk;
defm : UnsupportedSchedZabha;

} // SchedModel = TensixBHScalarModel

//===----------------------------------------------------------------------===//
// Wormhole Scheduling Model (same scalar pipeline, different SFPU latencies)
//===----------------------------------------------------------------------===//

def TensixWHScalarModel : SchedMachineModel {
  let IssueWidth = 1;
  let MicroOpBufferSize = 0;
  let LoadLatency = 2;
  let MispredictPenalty = 5;
  let CompleteModel = false;
  let PostRAScheduler = 1;
  let UnsupportedFeatures = [HasStdExtZbkb, HasStdExtZbkc, HasStdExtZbkx,
                             HasStdExtZcmt, HasStdExtZknd, HasStdExtZkne,
                             HasStdExtZknh, HasStdExtZksed, HasStdExtZksh,
                             HasStdExtZkr, HasVInstructions, HasVInstructionsI64];
}

let SchedModel = TensixWHScalarModel in {

// Scalar pipeline identical to BH
def : WriteRes<WriteJmp,  [TensixUnitBranch]>;
def : WriteRes<WriteJal,  [TensixUnitBranch]>;
def : WriteRes<WriteJalr, [TensixUnitBranch]>;
def : WriteRes<WriteIALU,       [TensixUnitALU]>;
def : WriteRes<WriteIALU32,     [TensixUnitALU]>;
def : WriteRes<WriteShiftImm,   [TensixUnitALU]>;
def : WriteRes<WriteShiftImm32, [TensixUnitALU]>;
def : WriteRes<WriteShiftReg,   [TensixUnitALU]>;
def : WriteRes<WriteShiftReg32, [TensixUnitALU]>;
def : WriteRes<WriteIMul,   [TensixUnitIMul]> { let Latency = 2; let ReleaseAtCycles = [2]; }
def : WriteRes<WriteIMul32, [TensixUnitIMul]> { let Latency = 2; let ReleaseAtCycles = [2]; }
def : WriteRes<WriteIDiv,   [TensixUnitIDiv]> { let Latency = 33; let ReleaseAtCycles = [33]; }
def : WriteRes<WriteIDiv32, [TensixUnitIDiv]> { let Latency = 33; let ReleaseAtCycles = [33]; }
def : WriteRes<WriteIRem,   [TensixUnitIDiv]> { let Latency = 33; let ReleaseAtCycles = [33]; }
def : WriteRes<WriteIRem32, [TensixUnitIDiv]> { let Latency = 33; let ReleaseAtCycles = [33]; }
def : WriteRes<WriteLDB, [TensixUnitMem]> { let Latency = 2; }
def : WriteRes<WriteLDH, [TensixUnitMem]> { let Latency = 2; }
def : WriteRes<WriteLDW, [TensixUnitMem]> { let Latency = 2; }
def : WriteRes<WriteSTB, [TensixUnitMem]>;
def : WriteRes<WriteSTH, [TensixUnitMem]>;
def : WriteRes<WriteSTW, [TensixUnitMem]>;
def : WriteRes<WriteAtomicW,   [TensixUnitMem]> { let Latency = 12; let ReleaseAtCycles = [12]; }
def : WriteRes<WriteAtomicD,   [TensixUnitMem]> { let Latency = 12; let ReleaseAtCycles = [12]; }
def : WriteRes<WriteAtomicLDW, [TensixUnitMem]> { let Latency = 12; let ReleaseAtCycles = [12]; }
def : WriteRes<WriteAtomicLDD, [TensixUnitMem]> { let Latency = 12; let ReleaseAtCycles = [12]; }
def : WriteRes<WriteAtomicSTW, [TensixUnitMem]>;
def : WriteRes<WriteAtomicSTD, [TensixUnitMem]>;
def : WriteRes<WriteLDD, [TensixUnitMem]> { let Latency = 2; }
def : WriteRes<WriteSTD, [TensixUnitMem]>;
def : WriteRes<WriteCSR, []>;
def : WriteRes<WriteNop, []>;
def : InstRW<[WriteIALU], (instrs COPY)>;

def : ReadAdvance<ReadJmp, 0>;
def : ReadAdvance<ReadJalr, 0>;
def : ReadAdvance<ReadCSR, 0>;
def : ReadAdvance<ReadStoreData, 0>;
def : ReadAdvance<ReadMemBase, 0>;
def : ReadAdvance<ReadIALU, 0>;
def : ReadAdvance<ReadIALU32, 0>;
def : ReadAdvance<ReadShiftImm, 0>;
def : ReadAdvance<ReadShiftImm32, 0>;
def : ReadAdvance<ReadShiftReg, 0>;
def : ReadAdvance<ReadShiftReg32, 0>;
def : ReadAdvance<ReadIDiv, 0>;
def : ReadAdvance<ReadIDiv32, 0>;
def : ReadAdvance<ReadIRem, 0>;
def : ReadAdvance<ReadIRem32, 0>;
def : ReadAdvance<ReadIMul, 0>;
def : ReadAdvance<ReadIMul32, 0>;
def : ReadAdvance<ReadAtomicWA, 0>;
def : ReadAdvance<ReadAtomicWD, 0>;
def : ReadAdvance<ReadAtomicDA, 0>;
def : ReadAdvance<ReadAtomicDD, 0>;
def : ReadAdvance<ReadAtomicLDW, 0>;
def : ReadAdvance<ReadAtomicLDD, 0>;
def : ReadAdvance<ReadAtomicSTW, 0>;
def : ReadAdvance<ReadAtomicSTD, 0>;

// SFPU (WH has same latencies as BH for most instructions)
def : WriteRes<WriteSFPUSimple,  [SFPUSimpleUnit]>;
def : WriteRes<WriteSFPULoad,    [SFPULoadUnit]>;
def : WriteRes<WriteSFPUStore,   [SFPUStoreUnit]>;
def : WriteRes<WriteSFPURound,   [SFPURoundUnit]>;
def : WriteRes<WriteSFPUMAD,     [SFPUMADUnit]>    { let Latency = 2; }
def : WriteRes<WriteSFPUSwap,    [SFPUSimpleUnit]>  { let Latency = 2; }
def : WriteRes<WriteSFPULUTFP32, [SFPUMADUnit]>    { let Latency = 2; }
def : WriteRes<WriteSFPUShft2,   [SFPUSimpleUnit]>  { let Latency = 2; } // WH: 2 cycles for shuffle
def : WriteRes<WriteSFPUNop,     []>;
def : ReadAdvance<ReadSFPUSrc, 0>;

//--- SFPU InstRW for WH ---
// Same mapping as BH except: WH load/store variants, no BH-only instructions,
// SFPSHFT2 is 2 cycles on WH (vs 1 on BH)
def : InstRW<[WriteSFPUSimple], (instrs
    SFPDIVP2, SFPEXEXP, SFPEXMAN, SFPIADD, SFPSHFT,
    SFPMOV, SFPABS, SFPAND, SFPOR, SFPNOT, SFPLZ,
    SFPSETEXP, SFPSETMAN, SFPSETSGN,
    SFPSETCC, SFPPUSHC, SFPPOPC, SFPENCC, SFPCOMPC,
    SFPTRANSP, SFPXOR, SFPCONFIG,
    SFPMOV_LV, SFPABS_LV, SFPEXEXP_LV, SFPEXMAN_LV,
    SFPDIVP2_LV, SFPLZ_LV, SFPNOT_LV,
    SFPSETEXP_LV, SFPSETMAN_LV, SFPSETSGN_LV, SFPSHFT2_LV
)>;

def : InstRW<[WriteSFPUMAD], (instrs
    SFPMAD, SFPADD, SFPMUL, SFPMULI, SFPADDI,
    PseudoSFPMULI, PseudoSFPADDI,
    SFPMAD_WH, SFPADD_WH, SFPMUL_WH,
    SFPMAD_LV, SFPADD_LV, SFPMUL_LV
)>;

def : InstRW<[WriteSFPULoad], (instrs
    SFPLOAD_WH, SFPLOADI, SFPLUT, SFPLOADMACRO_BH,
    SFPLOAD_WH_LV
)>;

def : InstRW<[WriteSFPUStore], (instrs SFPSTORE_WH)>;
def : InstRW<[WriteSFPURound], (instrs SFP_STOCH_RND, SFPCAST, SFPCAST_LV)>;
def : InstRW<[WriteSFPUSwap], (instrs SFPSWAP)>;
def : InstRW<[WriteSFPULUTFP32], (instrs SFPLUTFP32)>;
def : InstRW<[WriteSFPUShft2], (instrs SFPSHFT2)>;  // WH: 2 cycles (model above)
def : InstRW<[WriteSFPUNop], (instrs SFPNOP)>;
def : InstRW<[WriteIALU], (instrs TTREPLAY)>;

//--- Unsupported extensions ---
defm : UnsupportedSchedF;
defm : UnsupportedSchedD;
defm : UnsupportedSchedV;
defm : UnsupportedSchedZba;
defm : UnsupportedSchedZbb;
defm : UnsupportedSchedZbc;
defm : UnsupportedSchedZbs;
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedZfa;
defm : UnsupportedSchedZfh;
defm : UnsupportedSchedSFB;
defm : UnsupportedSchedXsfvcp;
defm : UnsupportedSchedZvk;
defm : UnsupportedSchedZabha;

} // SchedModel = TensixWHScalarModel
