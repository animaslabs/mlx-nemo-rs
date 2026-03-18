#!/usr/bin/env python3
"""Compare streaming outputs between Python NeMo and Rust implementation."""

import os
import sys
import torch
import numpy as np

# Set the model path
MODEL_PATH = os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--nemotron-speech-streaming-en-0.6b/snapshots/e6a95c855dabb83223055f513d9f9fbce96d6970/nemotron-speech-streaming-en-0.6b.nemo")
AUDIO_PATH = "samples/christmas2025_nightbeforechristmas_kri_64kb.wav"

def main():
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareRNNTPipeline
    import soundfile as sf

    # Load model
    print(f"Loading model from: {MODEL_PATH}")
    model = nemo_asr.models.ASRModel.restore_from(MODEL_PATH)
    model.eval()

    # Create streaming pipeline - mode 0
    mode_idx = 0
    pipeline = CacheAwareRNNTPipeline(model=model, lookahead_size=mode_idx)

    # Get streaming configuration
    cfg = pipeline.streaming_cfg
    print(f"\nStreaming configuration (mode {mode_idx}):")
    print(f"  Chunk size: ({cfg.chunk_size[0]}, {cfg.chunk_size[1]})")
    print(f"  Shift size: ({cfg.shift_size[0]}, {cfg.shift_size[1]})")
    print(f"  Valid out len: {cfg.valid_out_len}")
    print(f"  Pre encode cache size: ({cfg.pre_encode_cache_size[0]}, {cfg.pre_encode_cache_size[1]})")
    print(f"  Drop extra pre encoded: {cfg.drop_extra_pre_encoded}")

    # Load audio
    print(f"\nLoading audio: {AUDIO_PATH}")
    samples, sr = sf.read(AUDIO_PATH, dtype='float32')
    if len(samples.shape) > 1:
        samples = samples[:, 0]  # Take first channel
    print(f"Audio duration: {len(samples)/sr:.2f}s ({len(samples)} samples at {sr} Hz)")

    # Process in chunks (same as Rust example - 100ms chunks = 1600 samples at 16kHz)
    chunk_ms = 100
    sample_rate = int(sr)
    chunk_samples = int(sample_rate * chunk_ms / 1000)
    print(f"\nProcessing in {chunk_ms}ms chunks ({chunk_samples} samples)")

    # Reset pipeline
    pipeline.reset()

    print("\n--- Streaming Transcription ---")
    chunks_fed = 0
    step = 0

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
            # Get the mel buffer state before processing
            mel_before = pipeline._buffer.mel_buffer.clone() if pipeline._buffer.mel_buffer is not None else None

            # Process one step
            output = pipeline.transcribe_step()

            if output is not None:
                # Log encoder output stats
                if hasattr(pipeline, '_last_encoded') and pipeline._last_encoded is not None:
                    enc = pipeline._last_encoded
                    enc_np = enc[0].cpu().numpy()  # [T, D]
                    if enc_np.shape[0] > 0:
                        frame0 = enc_np[0, :10]
                        frame0_norm = np.linalg.norm(enc_np[0])
                        print(f"CHUNK {step} raw_enc shape={list(enc.shape)} len={enc.shape[1]} frame0={[f'{x:.6f}' for x in frame0]} norm={frame0_norm:.6f}")

                new_tokens = output.get('tokens', [])
                text = output.get('text', '')
                print(f"STEP {step} new_tokens={new_tokens} text={repr(text[:50])}...")
                step += 1

    print("\n--- Final Result ---")
    print(f"Text: {pipeline.get_text()}")
    print(f"\n--- Statistics ---")
    print(f"Audio chunks fed: {chunks_fed}")
    print(f"Transcribe steps: {step}")

if __name__ == "__main__":
    main()
