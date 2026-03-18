#!/usr/bin/env python3
"""Detailed streaming comparison between Python NeMo and Rust implementation."""

import os
import sys
import torch
import numpy as np

# Set the model path
MODEL_PATH = os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--nemotron-speech-streaming-en-0.6b/snapshots/e6a95c855dabb83223055f513d9f9fbce96d6970/nemotron-speech-streaming-en-0.6b.nemo")
AUDIO_PATH = "samples/christmas2025_nightbeforechristmas_kri_64kb.wav"

def main():
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareRNNTPipeline, CacheAwareStreamingBuffer
    import soundfile as sf

    # Load model
    print(f"Loading model from: {MODEL_PATH}", file=sys.stderr)
    model = nemo_asr.models.ASRModel.restore_from(MODEL_PATH)
    model.eval()

    # Create streaming pipeline - mode 0
    mode_idx = 0
    pipeline = CacheAwareRNNTPipeline(model=model, lookahead_size=mode_idx)

    # Get streaming configuration
    cfg = pipeline.streaming_cfg
    print(f"Streaming configuration (mode {mode_idx}):", file=sys.stderr)
    print(f"  Chunk size: ({cfg.chunk_size[0]}, {cfg.chunk_size[1]})", file=sys.stderr)
    print(f"  Shift size: ({cfg.shift_size[0]}, {cfg.shift_size[1]})", file=sys.stderr)
    print(f"  Valid out len: {cfg.valid_out_len}", file=sys.stderr)
    print(f"  Pre encode cache size: ({cfg.pre_encode_cache_size[0]}, {cfg.pre_encode_cache_size[1]})", file=sys.stderr)
    print(f"  Drop extra pre encoded: {cfg.drop_extra_pre_encoded}", file=sys.stderr)

    # Load audio
    print(f"\nLoading audio: {AUDIO_PATH}", file=sys.stderr)
    samples, sr = sf.read(AUDIO_PATH, dtype='float32')
    if len(samples.shape) > 1:
        samples = samples[:, 0]  # Take first channel
    print(f"Audio duration: {len(samples)/sr:.2f}s ({len(samples)} samples at {sr} Hz)", file=sys.stderr)

    # Process in chunks (same as Rust example - 100ms chunks = 1600 samples at 16kHz)
    chunk_ms = 100
    sample_rate = int(sr)
    chunk_samples = int(sample_rate * chunk_ms / 1000)
    print(f"\nProcessing in {chunk_ms}ms chunks ({chunk_samples} samples)", file=sys.stderr)

    # Reset pipeline
    pipeline.reset()

    print("\n--- Streaming Transcription ---", file=sys.stderr)
    chunks_fed = 0
    step = 0
    total_tokens = []

    for chunk_idx in range(0, len(samples), chunk_samples):
        chunk = samples[chunk_idx:chunk_idx + chunk_samples]
        chunks_fed += 1

        # Check if this is the last chunk
        is_last = (chunk_idx + chunk_samples) >= len(samples)

        # Feed audio chunk
        pipeline.add_audio_data(chunk)

        if is_last:
            pipeline.end_of_stream()

        # Process any available transcription steps
        while pipeline._buffer.is_ready_for_processing():
            # Process one step - manually to get more info
            buf = pipeline._buffer

            # Get mel features before transcription
            mel_buffer = buf.mel_buffer
            mel_len = buf.current_mel_len
            is_first_chunk = buf._is_first
            is_last_chunk = buf.stream_ended

            # Process one step
            output = pipeline.transcribe_step()

            if output is not None:
                new_tokens = output.get('tokens', [])
                total_tokens.extend(new_tokens)
                text = output.get('text', '')

                # Get encoder output from last transcription
                enc = pipeline._last_encoded if hasattr(pipeline, '_last_encoded') else None

                if enc is not None:
                    enc_np = enc.cpu().numpy()  # [B, D, T] or [B, T, D] depending on format
                    if len(enc_np.shape) == 3:
                        # Determine layout - check which dim is larger
                        if enc_np.shape[1] > enc_np.shape[2]:
                            # [B, D, T] format - transpose
                            enc_np = enc_np.transpose(0, 2, 1)  # -> [B, T, D]

                        # Now [B, T, D]
                        num_frames = enc_np.shape[1]
                        frame0_first10 = enc_np[0, 0, :10].tolist() if num_frames > 0 else []
                        frame0_norm = float(np.linalg.norm(enc_np[0, 0, :])) if num_frames > 0 else 0.0

                        # Output in format matching Rust trace output
                        print(f"STEP {step}: enc_frames={num_frames}, frame0_norm={frame0_norm:.6f}, new_tokens={new_tokens}, total_tokens={len(total_tokens)}")
                else:
                    print(f"STEP {step}: enc_frames=?, frame0_norm=?, new_tokens={new_tokens}, total_tokens={len(total_tokens)}")

                step += 1

    print(f"\n--- Final Result ---", file=sys.stderr)
    print(f"Text: {pipeline.get_text()}", file=sys.stderr)
    print(f"\n--- Statistics ---", file=sys.stderr)
    print(f"Audio chunks fed: {chunks_fed}", file=sys.stderr)
    print(f"Transcribe steps: {step}", file=sys.stderr)
    print(f"Total tokens: {len(total_tokens)}", file=sys.stderr)

if __name__ == "__main__":
    main()
