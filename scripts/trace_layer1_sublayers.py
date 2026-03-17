#!/usr/bin/env python3
"""Trace layer 1 sublayer outputs for chunk 1."""

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

    # Get layer 1 and its submodules
    layer1 = encoder.layers[1]

    # Hook storage
    sublayer_outputs = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            # Handle various input types - some modules receive empty tuples with kwargs
            inp = None
            if isinstance(input, tuple) and len(input) > 0:
                inp = input[0].detach().clone()
            sublayer_outputs[name] = {
                'input': inp,
                'output': out.detach().clone()
            }
        return hook

    # Register hooks on layer 1 submodules
    hooks = []

    # First feedforward
    if hasattr(layer1, 'norm_feed_forward1'):
        h = layer1.norm_feed_forward1.register_forward_hook(make_hook('ff1_norm'))
        hooks.append(h)
    if hasattr(layer1, 'feed_forward1'):
        h = layer1.feed_forward1.register_forward_hook(make_hook('ff1'))
        hooks.append(h)

    # Self-attention
    if hasattr(layer1, 'norm_self_att'):
        h = layer1.norm_self_att.register_forward_hook(make_hook('mha_norm'))
        hooks.append(h)
    if hasattr(layer1, 'self_attn'):
        h = layer1.self_attn.register_forward_hook(make_hook('mha'))
        hooks.append(h)

    # Convolution
    if hasattr(layer1, 'norm_conv'):
        h = layer1.norm_conv.register_forward_hook(make_hook('conv_norm'))
        hooks.append(h)
    if hasattr(layer1, 'conv'):
        h = layer1.conv.register_forward_hook(make_hook('conv'))
        hooks.append(h)

    # Second feedforward
    if hasattr(layer1, 'norm_feed_forward2'):
        h = layer1.norm_feed_forward2.register_forward_hook(make_hook('ff2_norm'))
        hooks.append(h)
    if hasattr(layer1, 'feed_forward2'):
        h = layer1.feed_forward2.register_forward_hook(make_hook('ff2'))
        hooks.append(h)

    # Final norm
    if hasattr(layer1, 'norm_out'):
        h = layer1.norm_out.register_forward_hook(make_hook('final_norm'))
        hooks.append(h)

    audio_pos = 0
    chunk_idx = 0

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

            sublayer_outputs.clear()

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

            if chunk_idx == 1:
                print(f"\n{'='*60}")
                print(f"CHUNK 1 - Layer 1 Sublayer Outputs")
                print(f"{'='*60}")

                for name in ['ff1_norm', 'ff1', 'mha_norm', 'mha', 'conv_norm', 'conv', 'ff2_norm', 'ff2', 'final_norm']:
                    if name in sublayer_outputs:
                        data = sublayer_outputs[name]
                        inp = data['input']
                        out = data['output']
                        print(f"\n{name}:")
                        if inp is not None:
                            print(f"  input shape: {list(inp.shape)}")
                            print(f"  input frame0 norm: {torch.norm(inp[0, 0]).item():.6f}")
                            print(f"  input frame0 [:5]: {inp[0, 0, :5].tolist()}")
                        else:
                            print(f"  input: (kwargs-based, not captured)")
                        print(f"  output shape: {list(out.shape)}")
                        print(f"  output frame0 norm: {torch.norm(out[0, 0]).item():.6f}")
                        print(f"  output frame0 [:5]: {out[0, 0, :5].tolist()}")

            audio_pos = end_pos
            chunk_idx += 1

    # Remove hooks
    for h in hooks:
        h.remove()


if __name__ == '__main__':
    main()
