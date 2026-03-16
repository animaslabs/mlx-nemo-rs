#!/usr/bin/env python3
"""
Trace what NeMo's actual streaming pipeline produces.
"""

import os
import sys
import warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

import logging
for n in ['nemo', 'nemo_logger', 'pytorch_lightning', 'lightning']:
    logging.getLogger(n).setLevel(logging.ERROR)
os.environ['NEMO_LOG_LEVEL'] = 'ERROR'

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
    from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions

    print("Building pipeline...")
    pipeline = PipelineBuilder.build_pipeline(cfg)

    request_generator = pipeline.get_request_generator()
    options = [ASRRequestOptions()]
    request_generator.set_audio_filepaths([audio_path], options)

    print(f"\n=== Processing: {audio_path} ===")

    pipeline.open_session()

    chunk_idx = 0
    try:
        for requests in request_generator:
            step_outputs = pipeline.transcribe_step(requests)
            chunk_idx += 1

            for step_output in step_outputs:
                partial = step_output.partial_transcript.strip()
                final = step_output.final_transcript.strip() if step_output.final_transcript else ""

                if partial or final:
                    print(f"Chunk {chunk_idx:3d}: partial='{partial}', final='{final}'")

                if chunk_idx <= 5:
                    # Try to access internal state for debugging
                    if hasattr(step_output, 'labels'):
                        print(f"         labels: {step_output.labels}")

            if chunk_idx >= 20:
                print("(stopping after 20 chunks)")
                break

    finally:
        pipeline.close_session()

    print("\nDone")


if __name__ == '__main__':
    main()
