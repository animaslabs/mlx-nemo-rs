#!/usr/bin/env python3
"""
Dump token IDs from streaming inference for comparison with Rust implementation.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

# Suppress noisy warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000  # Hz

VALID_LOOKAHEADS = {
    0: 0,
    80: 1,
    480: 6,
    1040: 13,
}


def lookahead_to_att_context(lookahead_ms: int, left_context: int = 70) -> list:
    if lookahead_ms in VALID_LOOKAHEADS:
        right_context = VALID_LOOKAHEADS[lookahead_ms]
    else:
        right_context = lookahead_ms // 80
    return [left_context, right_context]


def main():
    parser = argparse.ArgumentParser(
        description="Dump token IDs from streaming inference"
    )
    parser.add_argument("--model", "-m", type=str, required=True, help="Model path")
    parser.add_argument("audio", type=str, help="Audio file path")
    parser.add_argument("--lookahead", type=int, default=1040, help="Lookahead in ms")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1

    # Import NeMo modules
    from omegaconf import OmegaConf
    from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder

    # Build config
    config_path = Path(__file__).parent / "streaming_config.yaml"
    cfg = OmegaConf.load(config_path)
    cfg.asr.model_name = args.model
    cfg.asr.device = args.device
    cfg.asr.compute_dtype = "float32"
    cfg.asr.use_amp = False
    cfg.streaming.att_context_size = lookahead_to_att_context(args.lookahead)
    cfg.asr_decoding_type = "rnnt"
    cfg.log_level = 40  # ERROR

    print(f"Building pipeline for {args.model}...", file=sys.stderr)
    pipeline = PipelineBuilder.build_pipeline(cfg)

    # Get the model to access tokenizer
    model = pipeline.model

    # Get request generator and process
    request_generator = pipeline.get_request_generator()
    from nemo.collections.asr.inference.streaming.framing.request_options import (
        ASRRequestOptions,
    )

    options = [ASRRequestOptions()]
    request_generator.set_audio_filepaths([str(audio_path)], options)

    all_tokens = []
    chunk_idx = 0

    pipeline.open_session()
    try:
        for requests in request_generator:
            step_outputs = pipeline.transcribe_step(requests)

            for step_output in step_outputs:
                # Try to get tokens from the step output
                if hasattr(step_output, "hypotheses") and step_output.hypotheses:
                    for hyp in step_output.hypotheses:
                        if hasattr(hyp, "y_sequence") and hyp.y_sequence is not None:
                            tokens = (
                                hyp.y_sequence.tolist()
                                if hasattr(hyp.y_sequence, "tolist")
                                else list(hyp.y_sequence)
                            )
                            if tokens:
                                print(
                                    f"CHUNK {chunk_idx}: tokens={tokens}",
                                    file=sys.stderr,
                                )
                                all_tokens.extend(tokens)

                # Also try to get from final_hypotheses
                if (
                    hasattr(step_output, "final_hypotheses")
                    and step_output.final_hypotheses
                ):
                    for hyp in step_output.final_hypotheses:
                        if hasattr(hyp, "y_sequence") and hyp.y_sequence is not None:
                            tokens = (
                                hyp.y_sequence.tolist()
                                if hasattr(hyp.y_sequence, "tolist")
                                else list(hyp.y_sequence)
                            )
                            if tokens:
                                print(
                                    f"CHUNK {chunk_idx} FINAL: tokens={tokens}",
                                    file=sys.stderr,
                                )
                                # Don't extend - final hypotheses might duplicate

            chunk_idx += 1
    finally:
        pipeline.close_session()

    print(f"\nTotal chunks: {chunk_idx}", file=sys.stderr)
    print(f"Total tokens: {len(all_tokens)}", file=sys.stderr)

    # Output tokens to stdout for comparison
    print(",".join(map(str, all_tokens)))

    # Decode with tokenizer
    if hasattr(model, "tokenizer") and model.tokenizer:
        text = model.tokenizer.ids_to_text(all_tokens)
        print(f"\nDecoded text: {text}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
