#!/usr/bin/env python3
"""
Simple encoder debug - compare subsampling output directly.
"""

import torch
import numpy as np
import soundfile as sf
import warnings
import os

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import nemo.collections.asr as nemo_asr


def main():
    # Load model
    model_path = "../models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo"
    print(f"Loading model from: {model_path}")
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_path)
    model.eval()

    # Load audio and compute mel features
    audio_path = "../samples/Speaker26_004_16k.wav"
    audio, sr = sf.read(audio_path, dtype="float32")
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    audio_len = torch.tensor([len(audio)])

    print(f"Audio: {len(audio)} samples at {sr} Hz")

    # Compute mel features
    with torch.no_grad():
        mel, mel_len = model.preprocessor(input_signal=audio_tensor, length=audio_len)

    print(f"\nMel shape: {mel.shape}")  # [1, 128, time]

    # Extract chunks manually like streaming would
    # Chunk 0: frames 0-104 (105 frames)
    # Chunk 1: frames 96-216 (121 frames) with pre_encode_cache=9

    chunk0 = mel[:, :, :105]
    chunk1 = mel[:, :, 96:217]  # 96 to 216 inclusive = 121 frames

    print(f"\nChunk 0 shape: {chunk0.shape}")
    print(f"Chunk 1 shape: {chunk1.shape}")

    # Hook to capture subsampling output
    subsampling_output = [None]

    def hook_subsampling(module, input, output):
        if isinstance(output, tuple):
            subsampling_output[0] = output[0].detach().clone()
        else:
            subsampling_output[0] = output.detach().clone()

    hook_handle = model.encoder.pre_encode.register_forward_hook(hook_subsampling)

    try:
        # Process chunk 0 through subsampling only
        print("\n" + "=" * 60)
        print("CHUNK 0")
        print("=" * 60)

        print(f"Mel (first 5 frames, first 3 bins):")
        for t in range(min(5, chunk0.shape[2])):
            print(
                f"  frame {t}: [{chunk0[0, 0, t]:.4f}, {chunk0[0, 1, t]:.4f}, {chunk0[0, 2, t]:.4f}, ...]"
            )

        # Call subsampling directly
        with torch.no_grad():
            # Transpose mel to [B, T, F] for pre_encode
            chunk0_transposed = chunk0.transpose(1, 2)  # [1, 105, 128]
            sub_out, sub_len = model.encoder.pre_encode(
                audio_signal=chunk0_transposed, length=torch.tensor([105])
            )

        print(f"\nSubsampling output shape: {sub_out.shape}")
        print(f"Subsampling (first 3 frames, first 5 dims):")
        for t in range(min(3, sub_out.shape[1])):
            vals = sub_out[0, t, :5].tolist()
            print(f"  frame {t}: [{', '.join(f'{v:.4f}' for v in vals)}, ...]")

        # Process chunk 1
        print("\n" + "=" * 60)
        print("CHUNK 1 (frames 96-216, includes 9 frame overlap)")
        print("=" * 60)

        print(f"Mel (first 5 frames, first 3 bins):")
        for t in range(min(5, chunk1.shape[2])):
            print(
                f"  frame {t}: [{chunk1[0, 0, t]:.4f}, {chunk1[0, 1, t]:.4f}, {chunk1[0, 2, t]:.4f}, ...]"
            )

        with torch.no_grad():
            chunk1_transposed = chunk1.transpose(1, 2)  # [1, 121, 128]
            sub_out1, sub_len1 = model.encoder.pre_encode(
                audio_signal=chunk1_transposed, length=torch.tensor([121])
            )

        print(f"\nSubsampling output shape: {sub_out1.shape}")
        print(f"Subsampling (first 5 frames, first 5 dims):")
        for t in range(min(5, sub_out1.shape[1])):
            vals = sub_out1[0, t, :5].tolist()
            print(f"  frame {t}: [{', '.join(f'{v:.4f}' for v in vals)}, ...]")

        # After drop_extra_pre_encoded=2
        print(f"\nAfter drop_extra_pre_encoded=2:")
        print(
            f"Frame 0 (was frame 2): [{', '.join(f'{v:.4f}' for v in sub_out1[0, 2, :5].tolist())}, ...]"
        )
        print(
            f"Frame 1 (was frame 3): [{', '.join(f'{v:.4f}' for v in sub_out1[0, 3, :5].tolist())}, ...]"
        )

    finally:
        hook_handle.remove()

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
