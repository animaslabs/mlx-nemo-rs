#!/usr/bin/env python3
"""Fresh comparison of encoder norms between Python and Rust."""

# Python norms (from compare_all_chunks.py output just now)
python_norms = [
    (0, 1.103194, 1.083664),
    (1, 1.020964, 1.058773),
    (2, 1.306042, 2.475712),
    (3, 2.440653, 2.913716),
    (4, 1.951846, 2.271862),
    (5, 2.568122, 3.149290),
    (6, 2.411710, 1.833048),
    (7, 2.179056, 2.047819),
    (8, 2.933265, 3.057082),
    (9, 2.793201, 2.370446),
    (10, 2.453074, 3.329585),
    (11, 3.236674, 2.417527),
    (12, 3.430049, 3.557004),
    (13, 3.354391, 2.632269),
    (14, 2.065395, 2.777093),
    (15, 2.986700, 2.709218),
    (16, 2.486865, 1.889336),
    (17, 2.418602, 1.870260),
    (18, 2.651788, 2.435911),
    (19, 1.819174, 1.577810),
    (20, 2.636665, 2.511245),
    (21, 2.297469, 2.431777),
    (22, 2.337762, 2.272261),
    (23, 1.882861, 2.870488),
    (24, 1.790708, 2.410094),
    (25, 2.437626, 2.796035),
    (26, 2.067289, 1.719791),
    (27, 2.173057, 2.198407),
    (28, 1.850479, 2.403894),
    (29, 2.688426, 1.751646),
]

# Rust norms (from trace_streaming output, with ANSI codes stripped)
rust_norms = [
    (0, 1.103194, 1.083664),
    (1, 1.023646, 1.071583),
    (2, 1.273362, 2.539051),
    (3, 2.485486, 3.116751),
    (4, 2.049166, 2.334786),
    (5, 2.708536, 3.217166),
    (6, 2.471940, 1.959179),
    (7, 2.231989, 2.074597),
    (8, 3.011189, 3.095266),
    (9, 2.819739, 2.473841),
    (10, 2.542907, 3.382078),
    (11, 3.214352, 2.458083),
    (12, 3.495951, 3.579803),
    (13, 3.427432, 2.729249),
    (14, 2.136614, 2.853317),
    (15, 2.963779, 2.423442),
    (16, 2.492465, 1.905877),
    (17, 2.407659, 2.261906),
    (18, 2.586461, 2.024302),
    (19, 2.479132, 1.687098),
    (20, 2.717803, 2.604892),
    (21, 2.417321, 2.576592),
    (22, 2.267519, 2.349991),
    (23, 1.941321, 2.903341),
    (24, 1.713339, 2.398343),
    (25, 2.163364, 2.683191),
    (26, 2.005139, 1.767412),
    (27, 2.278572, 2.267121),
    (28, 3.037537, 2.179622),
    (29, 1.876056, 2.299102),
]

print("Chunk | Python F0 | Rust F0   | Diff %  | Python F1 | Rust F1   | Diff %")
print("-" * 75)

for i in range(min(len(python_norms), len(rust_norms))):
    ci, py_f0, py_f1 = python_norms[i]
    _, ru_f0, ru_f1 = rust_norms[i]
    diff_f0 = abs(py_f0 - ru_f0) / py_f0 * 100
    diff_f1 = abs(py_f1 - ru_f1) / py_f1 * 100
    flag = "!!" if diff_f0 > 5 or diff_f1 > 5 else ""
    print(f"{ci:5} | {py_f0:9.6f} | {ru_f0:9.6f} | {diff_f0:6.2f}% | {py_f1:9.6f} | {ru_f1:9.6f} | {diff_f1:6.2f}% {flag}")

if len(python_norms) > len(rust_norms):
    print(f"\nPython has {len(python_norms) - len(rust_norms)} extra chunks:")
    for i in range(len(rust_norms), len(python_norms)):
        ci, py_f0, py_f1 = python_norms[i]
        print(f"  Chunk {ci}: F0={py_f0:.6f}, F1={py_f1:.6f}")

# Count problematic chunks
print("\n=== Summary ===")
issues = []
for i in range(min(len(python_norms), len(rust_norms))):
    ci, py_f0, py_f1 = python_norms[i]
    _, ru_f0, ru_f1 = rust_norms[i]
    diff_f0 = abs(py_f0 - ru_f0) / py_f0 * 100
    diff_f1 = abs(py_f1 - ru_f1) / py_f1 * 100
    if diff_f0 > 5 or diff_f1 > 5:
        issues.append((ci, diff_f0, diff_f1))

print(f"Total chunks compared: {min(len(python_norms), len(rust_norms))}")
print(f"Chunks with >5% difference: {len(issues)}")
if issues:
    print("Problematic chunks:")
    for ci, d0, d1 in issues:
        print(f"  Chunk {ci}: F0 diff={d0:.1f}%, F1 diff={d1:.1f}%")
