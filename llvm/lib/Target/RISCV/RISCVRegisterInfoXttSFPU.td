//===-- RISCVRegisterInfoXttSFPU.td - SFPU Register Defs ---*- tablegen -*-===//
//
// Tenstorrent SFPU (Vector Unit) register classes for the XttSFPU extension.
//
// The SFPU has 17 registers per lane (32 lanes), each 32 bits wide.
// L0-L7 are general-purpose allocatable registers.
// L8-L10 are read-only hardware constants.
// L11-L14 are writable only via SFPCONFIG (reserved by convention).
// L15 is read-only lane index (lane i = 2*i).
// L16 is a macro staging register (SFPLOADMACRO write / SFPSTORE read only).
//
// Reference: ttsim-analysis/FUNCTIONAL_UNITS.md Section 3.2
//            ttsim-analysis/ERRATA.md C-001, C-002, C-021, C-022, C-023
//
//===----------------------------------------------------------------------===//

let Namespace = "RISCV" in {

// --- SFPU LReg register definitions ---
// 32 lanes x 32 bits each; modeled as i32 for the scalar register proxy.
// The 4-bit encoding field maps directly to register number.

def L0  : RISCVReg<0,  "l0">;
def L1  : RISCVReg<1,  "l1">;
def L2  : RISCVReg<2,  "l2">;
def L3  : RISCVReg<3,  "l3">;
def L4  : RISCVReg<4,  "l4">;
def L5  : RISCVReg<5,  "l5">;
def L6  : RISCVReg<6,  "l6">;
def L7  : RISCVReg<7,  "l7">;

// Read-only hardware constants (C-002)
def L8  : RISCVReg<8,  "l8">;   // 0.8373 (ln(2) + 0.144 approx)
def L9  : RISCVReg<9,  "l9">;   // 0.0 (zero)
def L10 : RISCVReg<10, "l10">;  // 1.0 (one)

// SFPCONFIG-writable constants (C-023)
def L11 : RISCVReg<11, "l11">;  // Convention: -1.0
def L12 : RISCVReg<12, "l12">;  // Convention: 1/512.0 (0.001953125)
def L13 : RISCVReg<13, "l13">;  // Convention: -0.67487759
def L14 : RISCVReg<14, "l14">;  // Convention: -0.34484843

// Lane index register (C-021): lane i = integer 2*i
def L15 : RISCVReg<15, "l15">;

// Macro staging register (C-022): SFPLOADMACRO write / SFPSTORE read only
def L16 : RISCVReg<16, "l16">;

} // Namespace = "RISCV"

// --- Register Classes ---

// Allocatable general-purpose SFPU registers (L0-L7).
// The register allocator will only assign from this class.
// Note: L7 is used as scratch by SFPLUTFP32 (3-register destination encoding).
def SFPURegs : RegisterClass<"RISCV", [i32], 32,
    (add L0, L1, L2, L3, L4, L5, L6, L7)> {
  let Size = 32;
}

// All readable SFPU registers (L0-L16).
// Used for source operands that can reference constants or lane index.
// NOT allocatable — the register allocator must not assign virtual regs here.
def SFPUAllRegs : RegisterClass<"RISCV", [i32], 32,
    (add L0, L1, L2, L3, L4, L5, L6, L7,
     L8, L9, L10, L11, L12, L13, L14, L15, L16)> {
  let Size = 32;
  let isAllocatable = 0;
}

// Registers valid as SFPSTORE source (E-005: L12-L15 cannot be store source).
def SFPUStoreRegs : RegisterClass<"RISCV", [i32], 32,
    (add L0, L1, L2, L3, L4, L5, L6, L7,
     L8, L9, L10, L11, L16)> {
  let Size = 32;
  let isAllocatable = 0;
}

// Constant registers only (for operands that only accept CREGs).
def SFPUConstRegs : RegisterClass<"RISCV", [i32], 32,
    (add L8, L9, L10, L11, L12, L13, L14)> {
  let Size = 32;
  let isAllocatable = 0;
}

// SFPCONFIG destination registers (VD 11-14 for LReg writeback).
def SFPUConfigRegs : RegisterClass<"RISCV", [i32], 32,
    (add L11, L12, L13, L14)> {
  let Size = 32;
  let isAllocatable = 0;
}
