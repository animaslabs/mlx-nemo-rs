#!/usr/bin/env python3
"""Compare step-by-step tokens between Python NeMo and Rust implementation."""

import os
import sys
import warnings

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

import logging
for n in ['nemo', 'nemo_logger', 'pytorch_lightning', 'lightning']:
    logging.getLogger(n).setLevel(logging.ERROR)
os.environ['NEMO_LOG_LEVEL'] = 'ERROR'

import torch
import numpy as np
from pathlib import Path

MODEL_PATH = os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--nemotron-speech-streaming-en-0.6b/snapshots/e6a95c855dabb83223055f513d9f9fbce96d6970/nemotron-speech-streaming-en-0.6b.nemo")
AUDIO_PATH = "samples/christmas2025_nightbeforechristmas_kri_64kb.wav"


def main():
    import soundfile as sf
    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf

    # Load model
    print(f"Loading model from: {MODEL_PATH}", file=sys.stderr)
    asr_model = nemo_asr.models.ASRModel.restore_from(MODEL_PATH)
    asr_model.eval()

    # Get model components
    preprocessor = asr_model.preprocessor
    encoder = asr_model.encoder
    decoder = asr_model.decoder  # Prediction network
    joint = asr_model.joint

    # Get vocab size and blank ID
    vocab_size = asr_model.joint.num_classes_with_blank - 1
    blank_id = vocab_size
    print(f"Vocab size: {vocab_size}, Blank ID: {blank_id}", file=sys.stderr)

    # Setup streaming
    mode_idx = 0
    encoder.setup_streaming_params()
    cfg = encoder.streaming_cfg

    print(f"Streaming config (mode {mode_idx}):", file=sys.stderr)
    print(f"  Chunk size: {cfg.chunk_size}", file=sys.stderr)
    print(f"  Shift size: {cfg.shift_size}", file=sys.stderr)
    print(f"  Valid out len: {cfg.valid_out_len}", file=sys.stderr)
    print(f"  Pre encode cache size: {cfg.pre_encode_cache_size}", file=sys.stderr)
    print(f"  Drop extra: {cfg.drop_extra_pre_encoded}", file=sys.stderr)

    # Load audio
    audio, sr = sf.read(AUDIO_PATH, dtype='float32')
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    print(f"Audio: {len(audio)} samples, {sr} Hz, {len(audio)/sr:.2f}s", file=sys.stderr)

    # Get streaming parameters
    pre_encode_cache_size = cfg.pre_encode_cache_size[1]
    model_chunk_size = cfg.chunk_size[1]
    hop_length = int(sr * 0.01)  # 10ms hop

    # Audio chunk size for each encoder step
    buffer_mel_frames = model_chunk_size + pre_encode_cache_size
    audio_chunk_samples = buffer_mel_frames * hop_length
    print(f"Audio chunk size: {audio_chunk_samples} samples ({audio_chunk_samples/sr*1000:.0f}ms)", file=sys.stderr)

    # Initialize caches
    batch_size = 1
    dtype = torch.float32
    device = torch.device('cpu')

    cache_last_channel, cache_last_time, cache_last_channel_len = encoder.get_initial_cache_state(
        batch_size, dtype, device, max_dim=0
    )

    # Initialize decoder state (None = zeros in NeMo)
    decoder_state = None
    last_label = None

    # Process audio
    audio_pos = 0
    step = 0
    total_tokens = []

    with torch.no_grad():
        while audio_pos < len(audio):
            end_pos = min(audio_pos + audio_chunk_samples, len(audio))
            audio_chunk = audio[audio_pos:end_pos]

            # Pad if needed
            if len(audio_chunk) < audio_chunk_samples:
                audio_chunk = np.pad(audio_chunk, (0, audio_chunk_samples - len(audio_chunk)))

            is_first = (step == 0)
            drop_extra = 0 if is_first else cfg.drop_extra_pre_encoded

            # Preprocess
            chunk_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0)
            chunk_len = torch.tensor([len(audio_chunk)])
            mel, mel_len = preprocessor(input_signal=chunk_tensor, length=chunk_len)

            # Encode
            (encoded, encoded_len,
             cache_last_channel, cache_last_time,
             cache_last_channel_len) = encoder.cache_aware_stream_step(
                processed_signal=mel,
                processed_signal_length=mel_len,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                keep_all_outputs=True,
                drop_extra_pre_encoded=drop_extra,
            )

            # Transpose [B, D, T] -> [B, T, D]
            encoded = encoded.transpose(1, 2)
            enc_len = int(encoded_len.item())

            # Get encoder frame 0 info
            enc_np = encoded.numpy()
            num_frames = enc_np.shape[1]
            frame0_norm = float(np.linalg.norm(enc_np[0, 0, :])) if num_frames > 0 else 0.0

            # Decode each encoder frame (greedy RNNT decoding)
            new_tokens = []
            for t in range(enc_len):
                enc_frame = encoded[:, t:t+1, :]  # [1, 1, D]

                max_symbols = 10
                for _ in range(max_symbols):
                    # Get prediction
                    if last_label is None:
                        # First prediction - pass None
                        g, new_decoder_state, *_ = decoder.predict(
                            y=None,
                            state=decoder_state,
                            add_sos=False,
                            batch_size=1
                        )
                    else:
                        # Subsequent predictions
                        y = torch.tensor([[last_label]])
                        g, new_decoder_state, *_ = decoder.predict(
                            y=y,
                            state=decoder_state,
                            add_sos=False,
                            batch_size=1
                        )

                    # Get joint output
                    joint_out = joint.joint(enc_frame, g)  # [B, 1, 1, V]
                    log_probs = joint_out.log_softmax(dim=-1)
                    pred_token = log_probs[0, 0, 0, :].argmax().item()

                    if pred_token == blank_id:
                        # Blank - move to next encoder frame
                        break
                    else:
                        # Non-blank - emit token and update state
                        new_tokens.append(pred_token)
                        total_tokens.append(pred_token)
                        last_label = pred_token
                        decoder_state = new_decoder_state

            # Output step info
            print(f"STEP {step}: enc_frames={enc_len}, frame0_norm={frame0_norm:.6f}, new_tokens={new_tokens}, total_tokens={len(total_tokens)}")

            audio_pos = end_pos
            step += 1

    print(f"\n--- Final ---", file=sys.stderr)
    print(f"Total steps: {step}", file=sys.stderr)
    print(f"Total tokens: {len(total_tokens)}", file=sys.stderr)

    # Decode final text
    tokenizer = asr_model.tokenizer
    text = tokenizer.ids_to_text(total_tokens)
    print(f"Text: {text}", file=sys.stderr)


if __name__ == '__main__':
    main()
