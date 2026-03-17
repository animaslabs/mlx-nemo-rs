#!/usr/bin/env python3
"""Trace layer 0 attention output for chunk 0."""

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

    # Get layer 0's self_attn
    layer0 = encoder.layers[0]
    mha = layer0.self_attn

    # Storage for captured data
    captured = {}

    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(out, tuple) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                captured[f'{name}_output'] = out[0].detach().clone()
            elif isinstance(out, torch.Tensor):
                captured[f'{name}_output'] = out.detach().clone()
        return hook

    # Register hook on layer 0's self_attn
    hooks = []
    hooks.append(mha.register_forward_hook(make_hook('layer0_mha')))

    # Also hook the full layer 0 to see its output
    hooks.append(layer0.register_forward_hook(make_hook('layer0')))

    # Print available submodules
    print(f"Layer 0 submodules: {[name for name, _ in layer0.named_children()]}")

    # Hook whatever conv/ff modules exist
    for name, module in layer0.named_children():
        if 'conv' in name.lower() or 'ff' in name.lower() or 'feed' in name.lower():
            hooks.append(module.register_forward_hook(make_hook(f'layer0_{name}')))

    audio_pos = 0
    chunk_idx = 0

    with torch.no_grad():
        while audio_pos < len(audio) and chunk_idx <= 0:  # Only chunk 0
            end_pos = min(audio_pos + audio_chunk_samples, len(audio))
            audio_chunk = audio[audio_pos:end_pos]
            if len(audio_chunk) < audio_chunk_samples:
                audio_chunk = np.pad(audio_chunk, (0, audio_chunk_samples - len(audio_chunk)))

            chunk_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0)
            chunk_len = torch.tensor([len(audio_chunk)])

            mel, mel_len = preprocessor(input_signal=chunk_tensor, length=chunk_len)

            is_first = (chunk_idx == 0)
            drop_extra = 0 if is_first else streaming_cfg.drop_extra_pre_encoded

            captured.clear()

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

            if chunk_idx == 0 and captured:
                print(f"\n{'='*70}")
                print(f"CHUNK 0 - Layer 0 Sublayer Outputs")
                print(f"{'='*70}")

                def print_tensor(name, t, frame_idx=4):
                    print(f"\n{name}:")
                    print(f"  shape: {list(t.shape)}")
                    t_np = t.numpy()
                    if len(t.shape) == 3:  # [B, T, D]
                        # Print all frames to see which are zero
                        for i in range(min(t.shape[1], 5)):
                            frame = t_np[0, i, :]
                            print(f"  [0,{i},:10]: {frame[:10].tolist()}")
                            print(f"  [0,{i}] norm: {np.linalg.norm(frame):.6f}")
                    elif len(t.shape) == 2:
                        print(f"  [:10]: {t_np.flatten()[:10].tolist()}")

                # Print captured tensors
                for key in sorted(captured.keys()):
                    if captured[key] is not None:
                        print_tensor(key, captured[key])

            audio_pos = end_pos
            chunk_idx += 1

    # Remove hooks
    for h in hooks:
        h.remove()


if __name__ == '__main__':
    main()
