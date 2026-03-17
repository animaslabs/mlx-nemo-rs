#!/usr/bin/env python3
"""Trace pre-encode outputs to understand overlap between chunks."""

import os
import warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

import logging
for n in ['nemo', 'nemo_logger', 'pytorch_lightning', 'lightning']:
    logging.getLogger(n).setLevel(logging.ERROR)
os.environ['NEMO_LOG_LEVEL'] = 'ERROR'

import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from omegaconf import OmegaConf


def main():
    audio_path = '../samples/2086-149220-0033.wav'

    config_path = Path('streaming_config.yaml')
    cfg = OmegaConf.load(config_path)
    cfg.asr.model_name = '../models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo'
    cfg.asr.device = 'cpu'
    cfg.asr.compute_dtype = 'float32'
    cfg.asr.use_amp = False
    cfg.streaming.att_context_size = [70, 1]
    cfg.asr_decoding_type = 'rnnt'
    cfg.log_level = 40

    from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder
    pipeline = PipelineBuilder.build_pipeline(cfg)

    asr_model = pipeline.asr_model.asr_model
    preprocessor = asr_model.preprocessor
    encoder = asr_model.encoder

    streaming_cfg = encoder.streaming_cfg
    if streaming_cfg is None:
        encoder.setup_streaming_params()
        streaming_cfg = encoder.streaming_cfg

    audio, sr = sf.read(audio_path, dtype='float32')

    pre_encode_cache_size = streaming_cfg.pre_encode_cache_size[1]
    model_chunk_size = streaming_cfg.chunk_size[1]

    if hasattr(preprocessor, 'featurizer'):
        fb = preprocessor.featurizer
        hop_length = int(getattr(fb, 'hop_length', getattr(fb, 'hop_size', 160)))
    else:
        hop_length = int(16000 * 0.01)

    buffer_mel_frames = model_chunk_size + pre_encode_cache_size
    audio_chunk_samples = buffer_mel_frames * hop_length

    print("=== Pre-encode Analysis ===")
    print(f"pre_encode_cache_size: {pre_encode_cache_size}")
    print(f"model_chunk_size: {model_chunk_size}")
    print(f"drop_extra_pre_encoded: {streaming_cfg.drop_extra_pre_encoded}")
    print()

    # Process chunks and track pre-encode outputs
    pre_encode_outputs = []

    audio_pos = 0
    max_chunks = 3

    with torch.no_grad():
        for chunk_idx in range(max_chunks):
            if audio_pos >= len(audio):
                break

            end_pos = min(audio_pos + audio_chunk_samples, len(audio))
            audio_chunk = audio[audio_pos:end_pos]
            if len(audio_chunk) < audio_chunk_samples:
                audio_chunk = np.pad(audio_chunk, (0, audio_chunk_samples - len(audio_chunk)))

            chunk_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0)
            chunk_len = torch.tensor([len(audio_chunk)])

            mel, mel_len = preprocessor(input_signal=chunk_tensor, length=chunk_len)

            print(f"=== Chunk {chunk_idx} ===")
            print(f"mel shape: {list(mel.shape)}, mel_len: {mel_len.item()}")

            # Run pre-encode manually
            mel_transposed = mel.transpose(1, 2)  # [B, T, F]
            pre_encoded, pre_len = encoder.pre_encode(x=mel_transposed, lengths=mel_len)

            print(f"pre_encode output shape: {list(pre_encoded.shape)}, len: {pre_len.item()}")
            print(f"  frame 0 [:5]: {pre_encoded[0, 0, :5].tolist()}")
            print(f"  frame -1 [:5]: {pre_encoded[0, -1, :5].tolist()}")
            print(f"  frame -1 norm: {torch.norm(pre_encoded[0, -1]).item():.6f}")

            pre_encode_outputs.append(pre_encoded[0].clone())

            # After drop_extra
            is_first = (chunk_idx == 0)
            drop_extra = 0 if is_first else streaming_cfg.drop_extra_pre_encoded
            if drop_extra > 0:
                dropped = pre_encoded[:, drop_extra:, :]
                print(f"  After drop_extra={drop_extra}:")
                print(f"    shape: {list(dropped.shape)}")
                print(f"    frame 0 [:5]: {dropped[0, 0, :5].tolist()}")
                print(f"    frame -1 [:5]: {dropped[0, -1, :5].tolist()}")
            print()

            audio_pos = end_pos

    # Compare overlapping frames between chunks
    print("=== Overlap Analysis ===")
    if len(pre_encode_outputs) >= 2:
        for i in range(len(pre_encode_outputs) - 1):
            curr = pre_encode_outputs[i]
            next_ = pre_encode_outputs[i + 1]

            print(f"\nComparing chunk {i} vs chunk {i+1}:")

            # Check if last N frames of curr match first N frames of next
            for offset in range(min(5, curr.shape[0], next_.shape[0])):
                curr_frame = curr[-(offset+1)]
                next_frame = next_[offset]
                diff = torch.norm(curr_frame - next_frame).item()
                match = "MATCH" if diff < 1e-5 else f"diff={diff:.6f}"
                print(f"  curr[-{offset+1}] vs next[{offset}]: {match}")


if __name__ == '__main__':
    main()
