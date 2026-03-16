#!/usr/bin/env python3
"""
Debug script to dump mel spectrogram and encoder values at specific chunks.
Used to compare Python NeMo vs Rust implementation.
"""

import argparse
import os
import sys
import warnings
import numpy as np
from pathlib import Path
import json

# Suppress noisy warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000


def main():
    parser = argparse.ArgumentParser(
        description="Dump mel/encoder values for debugging"
    )
    parser.add_argument("--model", "-m", type=str, required=True, help="Model path")
    parser.add_argument("--lookahead", type=int, default=1040, help="Lookahead ms")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument(
        "--dump-chunks",
        type=str,
        default="0,1,88,89,90,91,92,93,94,95,100",
        help="Comma-separated list of chunk indices to dump (0-indexed)",
    )
    parser.add_argument(
        "--output", type=str, default="python_mel_dump.json", help="Output file"
    )
    parser.add_argument("audio", type=str, help="Audio file")

    args = parser.parse_args()

    # Parse dump chunks
    dump_chunks = set(int(x.strip()) for x in args.dump_chunks.split(","))
    print(f"Will dump chunks: {sorted(dump_chunks)}")

    # Import NeMo after argument parsing to avoid slow startup if just --help
    import torch
    import soundfile as sf
    from omegaconf import OmegaConf

    # Suppress logging
    import logging

    for logger_name in ["nemo", "nemo_logger", "pytorch_lightning", "lightning"]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    from nemo.collections.asr.models import ASRModel
    from nemo.collections.asr.parts.utils.streaming_utils import (
        CacheAwareStreamingAudioBuffer,
    )

    # Load audio
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1

    audio_data, sr = sf.read(str(audio_path), dtype="float32")
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)

    audio_duration = len(audio_data) / SAMPLE_RATE
    print(f"Audio duration: {audio_duration:.2f}s ({len(audio_data)} samples)")

    # Load model
    print(f"Loading model: {args.model}")
    if args.model.endswith(".nemo"):
        model = ASRModel.restore_from(args.model, map_location=args.device)
    else:
        model = ASRModel.from_pretrained(args.model, map_location=args.device)

    model.eval()
    model.to(args.device)

    # Set streaming mode
    lookahead_frames = args.lookahead // 80  # 80ms per frame
    att_context_size = [70, lookahead_frames]
    print(f"Attention context size: {att_context_size}")

    # Get preprocessor config
    preproc_cfg = model.cfg.preprocessor
    print(
        f"Preprocessor config: sample_rate={preproc_cfg.sample_rate}, "
        f"window_size={preproc_cfg.window_size}, window_stride={preproc_cfg.window_stride}"
    )

    # Configure streaming
    model.encoder.set_default_att_context_size(att_context_size)
    model.encoder.setup_streaming_params()

    streaming_cfg = model.encoder.streaming_cfg
    print(f"\nStreaming config:")
    print(f"  chunk_size: {streaming_cfg.chunk_size}")
    print(f"  shift_size: {streaming_cfg.shift_size}")
    print(f"  pre_encode_cache_size: {streaming_cfg.pre_encode_cache_size}")
    print(f"  valid_out_len: {streaming_cfg.valid_out_len}")
    print(f"  drop_extra_pre_encoded: {streaming_cfg.drop_extra_pre_encoded}")

    # Initialize audio buffer - pad_and_drop_preencoded=True is the default streaming mode
    audio_buffer = CacheAwareStreamingAudioBuffer(
        model=model,
        online_normalization=False,
        pad_and_drop_preencoded=True,
    )

    # Initialize encoder cache
    cache_last_channel, cache_last_time, cache_last_channel_len = (
        model.encoder.get_initial_cache_state(batch_size=1)
    )
    if args.device != "cpu":
        cache_last_channel = cache_last_channel.to(args.device)
        cache_last_time = cache_last_time.to(args.device)
        cache_last_channel_len = cache_last_channel_len.to(args.device)

    # Initialize decoder state
    decoder_state = model.decoder.initialize_state(
        torch.zeros(1, 1024, 1).to(args.device)  # dummy encoder output
    )
    prev_token = None

    # Process audio - append the audio file to the buffer
    audio_buffer.append_audio_file(str(audio_path))
    # No signal_end_of_stream - just iterate until buffer is exhausted

    # Results storage
    results = {
        "model": args.model,
        "audio": str(audio_path),
        "audio_duration": audio_duration,
        "lookahead": args.lookahead,
        "att_context_size": att_context_size,
        "streaming_cfg": {
            "chunk_size": streaming_cfg.chunk_size,
            "shift_size": streaming_cfg.shift_size,
            "pre_encode_cache_size": streaming_cfg.pre_encode_cache_size,
            "valid_out_len": streaming_cfg.valid_out_len,
            "drop_extra_pre_encoded": streaming_cfg.drop_extra_pre_encoded,
        },
        "chunks": {},
    }

    # Process chunks
    chunk_idx = 0
    all_tokens = []
    all_text = ""

    print("\nProcessing chunks...")

    with torch.no_grad():
        for audio_chunk, chunk_lengths in audio_buffer:
            is_first = chunk_idx == 0
            step_num = 0 if is_first else 1

            # audio_chunk is the mel features from the buffer
            # Shape: [batch, n_mels, time]
            mel_chunk = audio_chunk
            if isinstance(mel_chunk, torch.Tensor):
                mel_np = mel_chunk.cpu().numpy()
            else:
                mel_np = np.array(mel_chunk)

            # Record mel stats
            mel_shape = mel_np.shape
            mel_min = float(mel_np.min())
            mel_max = float(mel_np.max())
            mel_mean = float(mel_np.mean())
            mel_std = float(mel_np.std())

            if chunk_idx in dump_chunks:
                print(
                    f"\nChunk {chunk_idx}: mel shape={mel_shape}, "
                    f"min={mel_min:.6f}, max={mel_max:.6f}, mean={mel_mean:.6f}, std={mel_std:.6f}"
                )

                # Dump first few mel frames (squeeze batch dim for comparison)
                mel_2d = mel_np.squeeze(0)  # [n_mels, time]
                first_frames = min(5, mel_2d.shape[1])
                mel_samples = []
                for t in range(first_frames):
                    frame_vals = mel_2d[:3, t].tolist()  # First 3 mel bins
                    mel_samples.append({"frame": t, "bins_0_2": frame_vals})
                    print(f"  frame {t}: bins[0:3] = {frame_vals}")

            # Prepare for encoder
            mel_tensor = (
                torch.tensor(mel_np).to(args.device)
                if not isinstance(mel_chunk, torch.Tensor)
                else mel_chunk.to(args.device)
            )
            mel_len = torch.tensor([mel_tensor.shape[2]]).to(args.device)

            # Get drop_extra for this step - applied to streaming_cfg
            drop_extra = 0 if is_first else streaming_cfg.drop_extra_pre_encoded

            # Run encoder using cache_aware_stream_step which handles drop_extra internally
            (
                encoded,
                encoded_len,
                cache_last_channel,
                cache_last_time,
                cache_last_channel_len,
            ) = model.encoder.cache_aware_stream_step(
                processed_signal=mel_tensor,
                processed_signal_length=mel_len,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                drop_extra_pre_encoded=drop_extra,
            )

            # Transpose encoder output from [B, D, T] to [B, T, D]
            encoded = encoded.transpose(1, 2)

            # Record encoder stats
            enc_np = encoded.cpu().numpy()
            enc_shape = enc_np.shape
            enc_min = float(enc_np.min())
            enc_max = float(enc_np.max())
            enc_mean = float(enc_np.mean())
            enc_std = float(enc_np.std())
            enc_len_val = int(encoded_len[0].item())

            if chunk_idx in dump_chunks:
                print(
                    f"  encoder: shape={enc_shape}, len={enc_len_val}, "
                    f"min={enc_min:.6f}, max={enc_max:.6f}, mean={enc_mean:.6f}, std={enc_std:.6f}"
                )

                # Dump first few encoder frames
                enc_3d = enc_np  # [batch, time, dim]
                enc_samples = []
                for t in range(min(3, enc_3d.shape[1])):
                    frame_vals = enc_3d[0, t, :5].tolist()  # First 5 dims
                    enc_samples.append({"frame": t, "dims_0_4": frame_vals})
                    print(
                        f"    enc frame {t}: dims[0:5] = {[f'{v:.6f}' for v in frame_vals]}"
                    )

            # Truncate to valid_out_len
            valid_len = min(enc_len_val, streaming_cfg.valid_out_len, encoded.shape[1])

            # Greedy decoding
            chunk_tokens = []
            for t in range(valid_len):
                enc_frame = encoded[:, t : t + 1, :]  # [1, 1, dim]

                symbols_emitted = 0
                max_symbols = 10

                while symbols_emitted < max_symbols:
                    # Get prediction from decoder
                    if prev_token is not None:
                        token_input = torch.tensor([[prev_token]]).to(args.device)
                    else:
                        token_input = None

                    pred_output, new_state = model.decoder.predict(
                        token_input,
                        state=decoder_state,
                        add_sos=False,
                        batch_size=1,
                    )

                    # Joint network
                    joint_out = model.joint.joint(
                        enc_frame, pred_output
                    )  # [1, 1, 1, vocab]
                    log_probs = joint_out.log_softmax(dim=-1).squeeze()  # [vocab]

                    pred_token = log_probs.argmax().item()

                    # Check for blank
                    blank_id = model.joint.num_classes_with_blank - 1
                    if pred_token == blank_id:
                        break

                    chunk_tokens.append(pred_token)
                    all_tokens.append(pred_token)
                    prev_token = pred_token
                    decoder_state = new_state
                    symbols_emitted += 1

            # Decode text
            if chunk_tokens:
                chunk_text = model.tokenizer.ids_to_text(chunk_tokens)
            else:
                chunk_text = ""

            if chunk_idx in dump_chunks:
                print(f"  tokens: {chunk_tokens}")
                print(f"  text: '{chunk_text}'")

                # Store in results
                results["chunks"][chunk_idx] = {
                    "mel_shape": list(mel_shape),
                    "mel_stats": {
                        "min": mel_min,
                        "max": mel_max,
                        "mean": mel_mean,
                        "std": mel_std,
                    },
                    "mel_samples": mel_samples if "mel_samples" in dir() else [],
                    "encoder_shape": list(enc_shape),
                    "encoder_len": enc_len_val,
                    "encoder_stats": {
                        "min": enc_min,
                        "max": enc_max,
                        "mean": enc_mean,
                        "std": enc_std,
                    },
                    "encoder_samples": enc_samples if "enc_samples" in dir() else [],
                    "valid_len": valid_len,
                    "tokens": chunk_tokens,
                    "text": chunk_text,
                }

            if chunk_text:
                all_text += chunk_text

            chunk_idx += 1

    print(f"\n\n=== FINAL RESULTS ===")
    print(f"Total chunks processed: {chunk_idx}")
    print(f"Total tokens: {len(all_tokens)}")
    print(f"Final text:\n{all_text}")

    results["total_chunks"] = chunk_idx
    results["total_tokens"] = len(all_tokens)
    results["all_tokens"] = all_tokens
    results["final_text"] = all_text

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
