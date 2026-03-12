#!/usr/bin/env python3
"""
Debug script to extract encoder intermediate values for comparison with Rust.
"""

import torch
import numpy as np
import soundfile as sf
import warnings
import os

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.streaming_utils import (
    CacheAwareStreamingAudioBuffer,
)


def main():
    # Load model
    model_path = "../models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo"
    print(f"Loading model from: {model_path}")
    model = nemo_asr.models.EncDecRNNTBPEModel.restore_from(model_path)
    model.eval()

    # Load audio
    audio_path = "../samples/Speaker26_004_16k.wav"
    audio, sr = sf.read(audio_path, dtype="float32")
    print(f"Audio: {len(audio)} samples at {sr} Hz")

    # Setup streaming mode 0
    mode_idx = 0
    if hasattr(model.encoder, "att_context_size_all"):
        att_context_size = model.encoder.att_context_size_all[mode_idx]
        if hasattr(model.encoder, "set_default_att_context_size"):
            model.encoder.set_default_att_context_size(att_context_size)
        else:
            model.encoder.att_context_size = att_context_size
            model.encoder.setup_streaming_params()
    else:
        model.encoder.setup_streaming_params()

    # Get streaming config
    streaming_cfg = model.encoder.streaming_cfg
    print(f"\nStreaming config:")
    print(f"  chunk_size: {streaming_cfg.chunk_size}")
    print(f"  shift_size: {streaming_cfg.shift_size}")
    print(f"  pre_encode_cache_size: {streaming_cfg.pre_encode_cache_size}")
    print(f"  drop_extra_pre_encoded: {streaming_cfg.drop_extra_pre_encoded}")
    print(f"  valid_out_len: {streaming_cfg.valid_out_len}")

    # Create streaming buffer
    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=model, online_normalization=False
    )
    processed_signal, processed_signal_length, stream_id = (
        streaming_buffer.append_audio_file(audio_path, stream_id=-1)
    )

    # Get initial cache
    batch_size = 1
    cache_last_channel, cache_last_time, cache_last_channel_len = (
        model.encoder.get_initial_cache_state(batch_size=batch_size)
    )

    # Process first two chunks
    print("\n" + "=" * 60)
    print("Processing chunks...")
    print("=" * 60)

    # Hook to capture subsampling output
    subsampling_output = [None]
    pos_emb_output = [None]

    def hook_subsampling(module, input, output):
        if isinstance(output, tuple):
            subsampling_output[0] = output[0].detach().clone()
        else:
            subsampling_output[0] = output.detach().clone()

    def hook_pos_enc(module, input, output):
        if isinstance(output, tuple):
            pos_emb_output[0] = (output[0].detach().clone(), output[1].detach().clone())
        else:
            pos_emb_output[0] = output.detach().clone()

    # Register hooks
    hook_handle_sub = model.encoder.pre_encode.register_forward_hook(hook_subsampling)
    hook_handle_pos = model.encoder.pos_enc.register_forward_hook(hook_pos_enc)

    try:
        for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer):
            if step_num > 1:
                break

            drop_extra = streaming_cfg.drop_extra_pre_encoded if step_num > 0 else 0

            print(f"\n{'=' * 60}")
            print(f"CHUNK {step_num}")
            print(f"{'=' * 60}")
            print(
                f"Mel input shape: {chunk_audio.shape}, length: {chunk_lengths.item()}"
            )
            print(f"drop_extra_pre_encoded: {drop_extra}")

            # Print first mel frame values
            print(f"\nMel input (first 5 frames, first 3 bins):")
            for t in range(min(5, chunk_audio.shape[2])):
                print(
                    f"  frame {t}: [{chunk_audio[0, 0, t]:.4f}, {chunk_audio[0, 1, t]:.4f}, {chunk_audio[0, 2, t]:.4f}, ...]"
                )

            with torch.inference_mode():
                # Process through model
                result = model.conformer_stream_step(
                    processed_signal=chunk_audio,
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                    cache_last_channel_len=cache_last_channel_len,
                    drop_extra_pre_encoded=drop_extra,
                    return_transcription=True,
                    previous_hypotheses=None,
                    previous_pred_out=None,
                )
                # Unpack based on actual return values
                if len(result) == 6:
                    (
                        best_hyp,
                        cache_last_channel,
                        cache_last_time,
                        cache_last_channel_len,
                        previous_hypotheses,
                        previous_pred_out,
                    ) = result
                else:
                    best_hyp = result[0]
                    cache_last_channel = (
                        result[1] if len(result) > 1 else cache_last_channel
                    )
                    cache_last_time = result[2] if len(result) > 2 else cache_last_time
                    cache_last_channel_len = (
                        result[3] if len(result) > 3 else cache_last_channel_len
                    )

                # Print subsampling output
                if subsampling_output[0] is not None:
                    sub_out = subsampling_output[0]
                    print(f"\nSubsampling output shape: {sub_out.shape}")
                    print(f"Subsampling output (first 3 frames, first 5 dims):")
                    for t in range(min(3, sub_out.shape[1])):
                        vals = sub_out[0, t, :5].tolist()
                        print(
                            f"  frame {t}: [{', '.join(f'{v:.4f}' for v in vals)}, ...]"
                        )

                # Print positional encoding info
                if pos_emb_output[0] is not None:
                    if isinstance(pos_emb_output[0], tuple):
                        x_out, pos_emb = pos_emb_output[0]
                        print(
                            f"\nAfter pos_enc: x shape {x_out.shape}, pos_emb shape {pos_emb.shape}"
                        )
                        print(f"x (first 3 frames, first 5 dims):")
                        for t in range(min(3, x_out.shape[1])):
                            vals = x_out[0, t, :5].tolist()
                            print(
                                f"  frame {t}: [{', '.join(f'{v:.4f}' for v in vals)}, ...]"
                            )
                        print(f"pos_emb (first 3 positions, first 5 dims):")
                        for t in range(min(3, pos_emb.shape[1])):
                            vals = pos_emb[0, t, :5].tolist()
                            print(
                                f"  pos {t}: [{', '.join(f'{v:.4f}' for v in vals)}, ...]"
                            )

                # Print transcription
                print(f"\nTranscription: '{best_hyp}'")
                print(f"cache_last_channel_len: {cache_last_channel_len}")

    finally:
        hook_handle_sub.remove()
        hook_handle_pos.remove()

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
