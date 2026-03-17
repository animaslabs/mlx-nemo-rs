#!/usr/bin/env python3
"""Trace layer-by-layer outputs to find divergence source."""

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

    batch_size = 1
    dtype = torch.float32
    device = torch.device('cpu')
    cache_last_channel, cache_last_time, cache_last_channel_len = encoder.get_initial_cache_state(
        batch_size, dtype, device, max_dim=0
    )

    pre_encode_cache_size = streaming_cfg.pre_encode_cache_size[1]
    model_chunk_size = streaming_cfg.chunk_size[1]

    if hasattr(preprocessor, 'featurizer'):
        fb = preprocessor.featurizer
        hop_length = int(getattr(fb, 'hop_length', getattr(fb, 'hop_size', 160)))
    else:
        hop_length = int(16000 * 0.01)

    buffer_mel_frames = model_chunk_size + pre_encode_cache_size
    audio_chunk_samples = buffer_mel_frames * hop_length

    # Hook to capture layer outputs
    layer_outputs = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            layer_outputs[name] = out.detach().clone()
        return hook

    # Register hooks on conformer layers
    hooks = []
    for i, layer in enumerate(encoder.layers):
        h = layer.register_forward_hook(make_hook(f'layer_{i}'))
        hooks.append(h)

    # Also hook pre_encode (subsampling)
    h = encoder.pre_encode.register_forward_hook(make_hook('pre_encode'))
    hooks.append(h)

    audio_pos = 0
    chunk_idx = 0
    target_chunks = [0, 1]  # Trace chunks 0 and 1

    with torch.no_grad():
        while audio_pos < len(audio) and chunk_idx <= 1:
            end_pos = min(audio_pos + audio_chunk_samples, len(audio))
            audio_chunk = audio[audio_pos:end_pos]
            if len(audio_chunk) < audio_chunk_samples:
                audio_chunk = np.pad(audio_chunk, (0, audio_chunk_samples - len(audio_chunk)))

            chunk_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0)
            chunk_len = torch.tensor([len(audio_chunk)])

            mel, mel_len = preprocessor(input_signal=chunk_tensor, length=chunk_len)

            is_first = (chunk_idx == 0)
            drop_extra = 0 if is_first else streaming_cfg.drop_extra_pre_encoded

            layer_outputs.clear()

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

            if chunk_idx in target_chunks:
                print(f"\n{'='*60}")
                print(f"CHUNK {chunk_idx}")
                print(f"{'='*60}")
                print(f"mel shape: {list(mel.shape)}, mel_len: {mel_len.item()}")
                print(f"drop_extra: {drop_extra}")

                # Print pre_encode output
                if 'pre_encode' in layer_outputs:
                    pre_enc = layer_outputs['pre_encode']
                    if isinstance(pre_enc, tuple):
                        pre_enc = pre_enc[0]
                    print(f"\npre_encode output shape: {list(pre_enc.shape)}")
                    print(f"pre_encode frame 0 norm: {torch.norm(pre_enc[0, 0]).item():.6f}")
                    print(f"pre_encode frame 0 [:10]: {pre_enc[0, 0, :10].tolist()}")

                # Print first few layer outputs
                for i in range(min(3, len(encoder.layers))):
                    key = f'layer_{i}'
                    if key in layer_outputs:
                        out = layer_outputs[key]
                        if isinstance(out, tuple):
                            out = out[0]
                        print(f"\nlayer {i} output shape: {list(out.shape)}")
                        print(f"layer {i} frame 0 norm: {torch.norm(out[0, 0]).item():.6f}")
                        print(f"layer {i} frame 0 [:10]: {out[0, 0, :10].tolist()}")
                        if out.shape[1] > 1:
                            print(f"layer {i} frame 1 norm: {torch.norm(out[0, 1]).item():.6f}")
                            print(f"layer {i} frame 1 [:10]: {out[0, 1, :10].tolist()}")

                # Print final encoder output
                enc_np = encoded.numpy()
                if enc_np.shape[1] == 1024:  # [B, D, T]
                    frame0 = enc_np[0, :, 0]
                    frame1 = enc_np[0, :, 1] if enc_np.shape[2] > 1 else None
                else:  # [B, T, D]
                    frame0 = enc_np[0, 0, :]
                    frame1 = enc_np[0, 1, :] if enc_np.shape[1] > 1 else None

                print(f"\nFinal encoder output shape: {list(encoded.shape)}")
                print(f"frame 0 norm: {np.linalg.norm(frame0):.6f}")
                print(f"frame 0 [:10]: {frame0[:10].tolist()}")
                if frame1 is not None:
                    print(f"frame 1 norm: {np.linalg.norm(frame1):.6f}")
                    print(f"frame 1 [:10]: {frame1[:10].tolist()}")

            audio_pos = end_pos
            chunk_idx += 1

    # Remove hooks
    for h in hooks:
        h.remove()


if __name__ == '__main__':
    main()
