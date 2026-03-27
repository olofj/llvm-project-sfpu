//===-- RISCVSchedXttSFPU.td - SFPU Scheduling Defs --------*- tablegen -*-===//
//
// Scheduling write/read class definitions for the Tenstorrent SFPU.
//
// The full scheduling model (TensixBHModel, TensixWHModel with ProcResource
// definitions and WriteRes bindings) is preserved in
// llvm-project/llvm/lib/Target/RISCV/RISCVSchedXttSFPU.td for future use.
//
// For now, we define only the SchedWrite and SchedRead classes so that
// SFPU instruction definitions can carry Sched annotations. The actual
// instruction scheduling is handled by our custom SFPU passes
// (RISCVXttSFPUErrata, RISCVXttSFPUReplay, etc.) which have accurate
// pipeline models for both BH and WH.
//
// To enable the full LLVM scheduling model in the future, the Tensix
// SchedMachineModel definitions need WriteRes entries for ALL standard
// RISC-V SchedWrite classes (WriteIALU, WriteILFL, etc.) — approximately
// 200+ entries — since LLVM validates completeness even with
// CompleteModel = 0 when a processor is bound to the model.
//
// Reference: ttsim-analysis/FUNCTIONAL_UNITS.md Section 3.1
//            ttsim-analysis/ERRATA.md E-004 (pipeline hazards)
//
//===----------------------------------------------------------------------===//

// --- SFPU SchedWrite classes ---
// These are used in Sched<[...]> annotations on SFPU instruction defs.
// Latency semantics (for reference — not enforced by scheduler currently):
//
//   1-cycle: WriteSFPUSimple, WriteSFPULoad, WriteSFPUStore, WriteSFPURound, WriteSFPUNop
//   2-cycle: WriteSFPUMAD, WriteSFPUSwap, WriteSFPULUTFP32
//   1-2 cycle: WriteSFPUShft2 (1 on BH, 2 on WH for SHFLSHR1)

def WriteSFPUSimple  : SchedWrite;  // MOV, logic, comparisons, CC stack, shifts
def WriteSFPULoad    : SchedWrite;  // SFPLOAD, SFPLOADI, SFPLUT, SFPLOADMACRO
def WriteSFPUStore   : SchedWrite;  // SFPSTORE
def WriteSFPURound   : SchedWrite;  // SFP_STOCH_RND, SFPCAST
def WriteSFPUMAD     : SchedWrite;  // SFPMAD, SFPADD, SFPMUL, SFPMULI, SFPADDI
def WriteSFPUSwap    : SchedWrite;  // SFPSWAP
def WriteSFPULUTFP32 : SchedWrite;  // SFPLUTFP32
def WriteSFPUShft2   : SchedWrite;  // SFPSHFT2
def WriteSFPUNop     : SchedWrite;  // SFPNOP

// --- SFPU SchedRead class ---
def ReadSFPUSrc : SchedRead;
