//===-- RISCVXttSFPUISelPatterns.td - ISel DAG Patterns -----*- tablegen -*-===//
//
// ISel patterns for the Tenstorrent SFPU extension.
//
// ALL SFPU intrinsics are handled by custom ISel in RISCVISelDAGToDAG.cpp.
// This is necessary because almost no intrinsic parameters are ImmArg —
// sfpi.h's C++ API computes mod1, imm12, addr, etc. at runtime. The custom
// ISel handles both constant and runtime operand values.
//
// The TImmLeaf predicates (uimmN_sfpu) on instruction operands still provide
// width validation when the values ARE constants at encoding time.
//
//===----------------------------------------------------------------------===//

// No Pat<> patterns — all ISel is in RISCVISelDAGToDAG.cpp Select() method.
// This file is kept for documentation and as a placeholder for future
// patterns (e.g., constant-folding optimizations).
