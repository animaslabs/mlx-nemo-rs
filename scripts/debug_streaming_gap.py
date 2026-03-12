#!/usr/bin/env python3
"""
Debug script to analyze streaming gap issue.

Compares what happens when processing from start vs from middle of file.
"""

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

import torch
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .nemo model")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--mode", type=int, default=0, help="Streaming mode")
    parser.add_argument(
        "--chunk-ms", type=int, default=100, help="Chunk size in milliseconds"
    )
    args = parser.parse_args()

    # Load model
    print(f"Loading model from: {args.model}", file=sys.stderr)
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.restore_from(args.model, map_location="cpu")
    model.eval()
    print(f"Model type: {type(model).__name__}", file=sys.stderr)
    sample_rate = model.cfg.preprocessor.sample_rate
    print(f"Sample rate: {sample_rate}", file=sys.stderr)

    # Setup streaming params
    if hasattr(model.encoder, "att_context_size_all"):
        att_context_sizes = model.encoder.att_context_size_all
        att_context_size = att_context_sizes[args.mode]
        if hasattr(model.encoder, "set_default_att_context_size"):
            model.encoder.set_default_att_context_size(att_context_size)
        else:
            model.encoder.att_context_size = att_context_size
            model.encoder.setup_streaming_params()
    else:
        model.encoder.setup_streaming_params()

    streaming_cfg = model.encoder.streaming_cfg
    print(f"\nStreaming config (mode {args.mode}):", file=sys.stderr)
    print(f"  chunk_size: {streaming_cfg.chunk_size}", file=sys.stderr)
    print(f"  shift_size: {streaming_cfg.shift_size}", file=sys.stderr)
    print(f"  valid_out_len: {streaming_cfg.valid_out_len}", file=sys.stderr)
    if hasattr(streaming_cfg, "pre_encode_cache_size"):
        print(
            f"  pre_encode_cache_size: {streaming_cfg.pre_encode_cache_size}",
            file=sys.stderr,
        )

    # Load audio
    import soundfile as sf

    print(f"\nLoading audio: {args.audio}", file=sys.stderr)
    audio, sr = sf.read(args.audio, dtype="float32")
    if sr != sample_rate:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate
    print(
        f"Audio duration: {len(audio) / sr:.2f}s ({len(audio)} samples)",
        file=sys.stderr,
    )

    # Process using NeMo's streaming buffer
    from nemo.collections.asr.parts.utils.streaming_utils import (
        CacheAwareStreamingAudioBuffer,
    )

    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=model, online_normalization=False
    )

    # Add audio file to buffer
    processed_signal, processed_signal_length, stream_id = (
        streaming_buffer.append_audio_file(str(args.audio), stream_id=-1)
    )

    # Get initial cache state
    batch_size = 1
    cache_last_channel, cache_last_time, cache_last_channel_len = (
        model.encoder.get_initial_cache_state(batch_size=batch_size)
    )

    # Process chunks and track output
    print("\n--- Processing chunks ---", file=sys.stderr)
    previous_hypotheses = None
    previous_pred_out = None
    total_tokens = 0
    last_text = ""

    streaming_buffer_iter = iter(streaming_buffer)

    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
        drop_extra_pre_encoded = (
            streaming_cfg.drop_extra_pre_encoded if step_num != 0 else 0
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
                drop_extra_pre_encoded=drop_extra_pre_encoded,
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

        # Extract text
        current_text = ""
        if transcribed_texts:
            if hasattr(transcribed_texts[0], "text"):
                current_text = transcribed_texts[0].text
            else:
                current_text = str(transcribed_texts[0])

        # Count tokens in current hypothesis
        if previous_hypotheses and len(previous_hypotheses) > 0:
            hyp = previous_hypotheses[0]
            if hasattr(hyp, "y_sequence"):
                token_count = len(hyp.y_sequence)
            else:
                token_count = 0
        else:
            token_count = 0

        # Calculate audio position
        audio_pos_sec = step_num * args.chunk_ms / 1000

        # Print detailed info around problematic region (100-130s)
        if 100 <= audio_pos_sec <= 125:
            new_text = (
                current_text[len(last_text) :]
                if current_text.startswith(last_text)
                else current_text
            )
            # Check encoder output shape/stats
            prev_tokens = total_tokens
            new_tokens = token_count - prev_tokens if token_count > prev_tokens else 0
            print(
                f"Step {step_num}: {audio_pos_sec:.1f}s, tokens={token_count} (+{new_tokens}), text='{new_text[:60] if new_text else ''}'",
                file=sys.stderr,
            )

        last_text = current_text
        total_tokens = token_count

    print(f"\n--- Final ---", file=sys.stderr)
    print(f"Total tokens: {total_tokens}", file=sys.stderr)
    print(f"Text length: {len(last_text)}", file=sys.stderr)

    # Check for the problematic content
    if "While the food was cooking" in last_text:
        print("✓ Contains 'While the food was cooking'", file=sys.stderr)
    else:
        print("✗ MISSING 'While the food was cooking'", file=sys.stderr)

    if "at the bottom" in last_text and "That, you know" in last_text:
        # Check if they're adjacent (missing content in between)
        idx1 = last_text.find("at the bottom")
        idx2 = last_text.find("That, you know")
        if idx2 > idx1:
            between = last_text[idx1:idx2]
            if "While the food" not in between:
                print(
                    f"✗ GAP detected between 'at the bottom' and 'That, you know'",
                    file=sys.stderr,
                )
                print(f"  Text between: '{between}'", file=sys.stderr)

    print(f"\nFull text:\n{last_text}", file=sys.stderr)


if __name__ == "__main__":
    main()
