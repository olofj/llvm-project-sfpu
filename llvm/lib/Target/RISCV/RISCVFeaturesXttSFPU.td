//===-- RISCVFeaturesXttSFPU.td - XttSFPU Feature Defs -----*- tablegen -*-===//
//
// Feature flag definitions for the Tenstorrent SFPU vendor extension.
// These are added to RISCVFeatures.td via include.
//
// Three features:
//   XttSFPU    — Base SFPU ISA (38 opcodes, 0x70-0x95)
//   XttSFPUBH  — Blackhole extensions (SFPMUL24, SFPARECIP, SFPGT, SFPLE,
//                SFPMOV_CONFIG, 3-bit addr_mode, hardware scoreboarding)
//   XttSFPUWH  — Wormhole extensions (2-bit addr_mode, dst=src_a constraint,
//                software NOP insertion required)
//
// Usage: -march=rv32imac_xttsfpu_xttsfpubh for Blackhole targets
//        -march=rv32imac_xttsfpu_xttsfpuwh for Wormhole targets
//    or: -mattr=+xttsfpu,+xttsfpubh for Blackhole targets
//        -mattr=+xttsfpu,+xttsfpuwh for Wormhole targets
//
// Note: Extension names use no hyphens (RISC-V -march allows only [a-z0-9]).
//       The old names xttsfpu-bh/xttsfpu-wh still work via -mattr but not -march.
//
// Reference: ttsim-analysis/ERRATA.md C-009, C-010, C-011
//
//===----------------------------------------------------------------------===//

// Use RISCVExtension (not SubtargetFeature) so these extensions appear in the
// -march parser's SupportedExtensions table (RISCVTargetParserDef.inc).
// Version 1.0 for all three.

def FeatureVendorXttSFPU
    : RISCVExtension<"xttsfpu", 1, 0,
                     "'XttSFPU' (Tenstorrent SFPU Vector Unit base ISA)">;

def FeatureVendorXttSFPUBH
    : RISCVExtension<"xttsfpubh", 1, 0,
                     "'XttSFPU-BH' (Tenstorrent SFPU Blackhole extensions)",
                     [FeatureVendorXttSFPU]>;

def FeatureVendorXttSFPUWH
    : RISCVExtension<"xttsfpuwh", 1, 0,
                     "'XttSFPU-WH' (Tenstorrent SFPU Wormhole extensions)",
                     [FeatureVendorXttSFPU]>;

// Processor definitions are in RISCVProcessors.td (TENSIX_BH, TENSIX_WH).
