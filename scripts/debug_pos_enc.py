#!/usr/bin/env python3
"""Debug script to examine NeMo's positional encoding in streaming mode."""

import torch
import numpy as np


# Simulate NeMo's RelPositionalEncoding behavior
def create_rel_pos_emb(max_len, d_model):
    """Create relative positional embedding like NeMo."""
    # positions from max_len-1 to -(max_len-1)
    positions = torch.arange(max_len - 1, -max_len, -1, dtype=torch.float32).unsqueeze(
        1
    )

    # Create sinusoidal encoding
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * -(np.log(10000.0) / d_model)
    )
    pe = torch.zeros(2 * max_len - 1, d_model)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe.unsqueeze(0)  # [1, 2*max_len-1, d_model]


def get_pos_emb_nemo(pe, seq_len, cache_len):
    """Extract positional embedding like NeMo does."""
    input_len = seq_len + cache_len
    center_pos = pe.size(1) // 2 + 1
    start_pos = center_pos - input_len
    end_pos = center_pos + input_len - 1
    return pe[:, start_pos:end_pos]


def get_pos_emb_ours(pe, total_len):
    """Extract positional embedding like our implementation."""
    # We generate for 2*total_len - 1 positions
    needed = 2 * total_len - 1
    center_pos = pe.size(1) // 2 + 1
    start_pos = center_pos - total_len
    end_pos = center_pos + total_len - 1
    return pe[:, start_pos:end_pos]


# Parameters matching the model
d_model = 1024
max_len = 5000  # Large enough

# Create PE table
pe = create_rel_pos_emb(max_len, d_model)
print(f"PE table shape: {pe.shape}")  # [1, 2*max_len-1, d_model]

# Chunk 2 parameters
seq_len = 14  # New frames
filled_cache_len = 12  # Actual cached frames
allocated_cache_len = 70  # NeMo's cache_len

print(f"\n=== Chunk 2 Comparison ===")
print(f"seq_len = {seq_len}")
print(f"filled_cache_len = {filled_cache_len}")
print(f"allocated_cache_len = {allocated_cache_len}")

# NeMo's approach: use allocated cache size
nemo_input_len = seq_len + allocated_cache_len  # 14 + 70 = 84
nemo_pos_emb = get_pos_emb_nemo(pe, seq_len, allocated_cache_len)
print(f"\nNeMo:")
print(f"  input_len = {nemo_input_len}")
print(
    f"  pos_emb shape = {nemo_pos_emb.shape}"
)  # [1, 2*84-1, d_model] = [1, 167, d_model]
print(f"  pos_emb covers positions {-nemo_input_len + 1} to {nemo_input_len - 1}")

# Our approach: use filled cache size
our_total_len = seq_len + filled_cache_len  # 14 + 12 = 26
our_pos_emb = get_pos_emb_ours(pe, our_total_len)
print(f"\nOurs:")
print(f"  total_len = {our_total_len}")
print(
    f"  pos_emb shape = {our_pos_emb.shape}"
)  # [1, 2*26-1, d_model] = [1, 51, d_model]
print(f"  pos_emb covers positions {-our_total_len + 1} to {our_total_len - 1}")

# Now let's see which positions correspond to the actual attention
# In relative attention, position index i corresponds to relative distance (center - i)

# For query at the first new frame position attending to:
# - First cached frame: relative distance = filled_cache_len = 12
# - Same position (first new frame): relative distance = 0

print("\n=== Position Encoding Values ===")

# NeMo's center position
nemo_center = nemo_pos_emb.size(1) // 2
print(f"\nNeMo center index: {nemo_center} (corresponds to relative pos 0)")
print(f"NeMo pos_emb[0, center, :5] = {nemo_pos_emb[0, nemo_center, :5].numpy()}")
print(
    f"NeMo pos_emb[0, center-12, :5] = {nemo_pos_emb[0, nemo_center - 12, :5].numpy()}  (rel pos +12)"
)
print(
    f"NeMo pos_emb[0, center+12, :5] = {nemo_pos_emb[0, nemo_center + 12, :5].numpy()}  (rel pos -12)"
)

# Our center position
our_center = our_pos_emb.size(1) // 2
print(f"\nOur center index: {our_center} (corresponds to relative pos 0)")
print(f"Our pos_emb[0, center, :5] = {our_pos_emb[0, our_center, :5].numpy()}")
print(
    f"Our pos_emb[0, center-12, :5] = {our_pos_emb[0, our_center - 12, :5].numpy()}  (rel pos +12)"
)
print(
    f"Our pos_emb[0, center+12, :5] = {our_pos_emb[0, our_center + 12, :5].numpy()}  (rel pos -12)"
)

# Compare the actual values for same relative positions
print("\n=== Comparing Values for Same Relative Positions ===")
for rel_pos in [0, 12, -12, 25, -25]:
    nemo_idx = nemo_center - rel_pos
    our_idx = our_center - rel_pos

    if 0 <= our_idx < our_pos_emb.size(1):
        nemo_val = nemo_pos_emb[0, nemo_idx, :3].numpy()
        our_val = our_pos_emb[0, our_idx, :3].numpy()
        match = np.allclose(nemo_val, our_val)
        print(
            f"Relative pos {rel_pos:3d}: NeMo[{nemo_idx}]={nemo_val}, Ours[{our_idx}]={our_val}, Match={match}"
        )
    else:
        print(f"Relative pos {rel_pos:3d}: Out of range for our implementation")
