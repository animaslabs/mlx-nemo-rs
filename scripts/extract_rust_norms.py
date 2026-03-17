#!/usr/bin/env python3
"""Extract encoder norms from Rust trace output."""
import sys
import re

frame0_norms = []
frame1_norms = []

for line in sys.stdin:
    if 'encoder frame 0' in line:
        match = re.search(r'frame0_norm=([0-9.]+)', line)
        if match:
            frame0_norms.append(float(match.group(1)))
    elif 'encoder frame 1' in line:
        match = re.search(r'frame1_norm=([0-9.]+)', line)
        if match:
            frame1_norms.append(float(match.group(1)))

# Print as Python list format
print("# Rust encoder norms")
print("rust_norms = [")
for i, (f0, f1) in enumerate(zip(frame0_norms, frame1_norms)):
    print(f"    ({i}, {f0:.6f}, {f1:.6f}),")
print("]")
