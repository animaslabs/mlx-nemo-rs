#!/usr/bin/env python3
"""
Debug script to compare encoder intermediate values with Rust.
Captures subsampling output and cache state for chunks 0 and 1.
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

    print(f"\nInitial cache:")
    print(f"  cache_last_channel shape: {cache_last_channel.shape}")
    print(f"  cache_last_time shape: {cache_last_time.shape}")
    print(f"  cache_last_channel_len: {cache_last_channel_len}")

    # Storage for captured values
    captured = {}

    def hook_pre_encode(module, input, output):
        """Capture subsampling output."""
        if isinstance(output, tuple):
            captured["subsampling"] = output[0].detach().clone()
        else:
            captured["subsampling"] = output.detach().clone()

    def hook_pos_enc(module, input, output):
        """Capture positional encoding output."""
        if isinstance(output, tuple):
            captured["pos_enc_x"] = output[0].detach().clone()
            captured["pos_emb"] = output[1].detach().clone()

    # Hook into the first conformer layer to capture its output
    def hook_layer0(module, input, output):
        """Capture conformer layer 0 output."""
        if isinstance(output, tuple):
            captured["layer0_out"] = output[0].detach().clone()
        else:
            captured["layer0_out"] = output.detach().clone()

    # Register hooks
    hook_handles = []
    hook_handles.append(model.encoder.pre_encode.register_forward_hook(hook_pre_encode))
    hook_handles.append(model.encoder.pos_enc.register_forward_hook(hook_pos_enc))
    if hasattr(model.encoder, "layers") and len(model.encoder.layers) > 0:
        hook_handles.append(model.encoder.layers[0].register_forward_hook(hook_layer0))

    try:
        previous_hypotheses = None
        previous_pred_out = None

        for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer):
            if step_num > 1:
                break

            captured.clear()
            drop_extra = streaming_cfg.drop_extra_pre_encoded if step_num > 0 else 0

            print(f"\n{'=' * 70}")
            print(f"CHUNK {step_num}")
            print(f"{'=' * 70}")
            print(
                f"Mel input shape: {chunk_audio.shape}, length: {chunk_lengths.item()}"
            )
            print(f"drop_extra_pre_encoded: {drop_extra}")
            print(f"cache_last_channel_len before: {cache_last_channel_len.item()}")

            # Print first 5 mel frames
            print(f"\nMel input (first 5 frames, first 3 bins):")
            for t in range(min(5, chunk_audio.shape[2])):
                print(
                    f"  frame {t}: [{chunk_audio[0, 0, t]:.4f}, {chunk_audio[0, 1, t]:.4f}, {chunk_audio[0, 2, t]:.4f}, ...]"
                )

            with torch.inference_mode():
                result = model.conformer_stream_step(
                    processed_signal=chunk_audio,
                    processed_signal_length=chunk_lengths,
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                    cache_last_channel_len=cache_last_channel_len,
                    keep_all_outputs=streaming_buffer.is_buffer_empty(),
                    previous_hypotheses=previous_hypotheses,
                    previous_pred_out=previous_pred_out,
                    drop_extra_pre_encoded=drop_extra,
                    return_transcription=True,
                    return_log_probs=False,
                )

                (
                    previous_pred_out,
                    transcribed_texts,
                    cache_last_channel,
                    cache_last_time,
                    cache_last_channel_len,
                    previous_hypotheses,
                ) = result

            # Print captured values
            if "subsampling" in captured:
                sub = captured["subsampling"]
                print(f"\nSubsampling output shape: {sub.shape}")
                print(f"Subsampling (first 5 frames, first 5 dims):")
                for t in range(min(5, sub.shape[1])):
                    vals = sub[0, t, :5].tolist()
                    print(f"  frame {t}: [{', '.join(f'{v:.4f}' for v in vals)}, ...]")

                # After drop_extra_pre_encoded
                if drop_extra > 0:
                    print(f"\nAfter drop_extra_pre_encoded={drop_extra}:")
                    for t in range(drop_extra, min(drop_extra + 3, sub.shape[1])):
                        vals = sub[0, t, :5].tolist()
                        print(
                            f"  frame {t - drop_extra} (was {t}): [{', '.join(f'{v:.4f}' for v in vals)}, ...]"
                        )

            if "pos_enc_x" in captured:
                x = captured["pos_enc_x"]
                pos = captured["pos_emb"]
                print(f"\nAfter pos_enc:")
                print(f"  x shape: {x.shape}")
                print(f"  pos_emb shape: {pos.shape}")
                print(f"  x (first 3 frames, first 5 dims):")
                for t in range(min(3, x.shape[1])):
                    vals = x[0, t, :5].tolist()
                    print(
                        f"    frame {t}: [{', '.join(f'{v:.4f}' for v in vals)}, ...]"
                    )

            if "layer0_out" in captured:
                layer0 = captured["layer0_out"]
                print(f"\nLayer 0 output shape: {layer0.shape}")
                print(f"Layer 0 (first 3 frames, first 5 dims):")
                for t in range(min(3, layer0.shape[1])):
                    vals = layer0[0, t, :5].tolist()
                    print(f"  frame {t}: [{', '.join(f'{v:.4f}' for v in vals)}, ...]")

            # Print cache state after
            print(f"\ncache_last_channel_len after: {cache_last_channel_len.item()}")

            # Print some values from the cache to compare
            # cache_last_channel is [num_layers, batch, cache_size, d_model]
            cache_len = int(cache_last_channel_len.item())
            print(
                f"cache_last_channel layer 0 (positions {cache_len - 3} to {cache_len - 1}, first 5 dims):"
            )
            for t in range(max(0, cache_len - 3), cache_len):
                vals = cache_last_channel[0, 0, t, :5].tolist()
                print(f"  pos {t}: [{', '.join(f'{v:.4f}' for v in vals)}, ...]")

            # Also show the raw cache content at positions 0-2 for initial chunks
            if step_num == 0:
                print(
                    f"cache_last_channel layer 0 (positions 0-2, first 5 dims) [raw]:"
                )
                for t in range(min(3, cache_last_channel.shape[2])):
                    vals = cache_last_channel[0, 0, t, :5].tolist()
                    print(f"  pos {t}: [{', '.join(f'{v:.4f}' for v in vals)}, ...]")

            # Print transcription
            if transcribed_texts:
                if hasattr(transcribed_texts[0], "text"):
                    text = transcribed_texts[0].text
                else:
                    text = str(transcribed_texts[0])
                print(f"\nTranscription: '{text}'")

    finally:
        for h in hook_handles:
            h.remove()

    print("\n" + "=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
