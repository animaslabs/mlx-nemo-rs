#!/usr/bin/env python3
"""
NeMo Multi-Speaker Streaming ASR with Diarization

This script demonstrates real-time streaming transcription with speaker diarization
using NVIDIA NeMo's SpeakerTaggedASR pipeline. It combines:
- ASR: nvidia/multitalker-parakeet-streaming-0.6b-v1
- Diarization: nvidia/diar_streaming_sortformer_4spk-v2.1

The output is a SegLST format with speaker-tagged segments showing who said what.

Usage:
    python scripts/streaming_multispeaker.py samples/Speaker26_004_16k.wav

    python scripts/streaming_multispeaker.py \
        --max-speakers 2 \
        --lookahead 1120 \
        --device cuda \
        -o output.json \
        samples/meeting.wav

Requirements:
    cd scripts && ./setup.sh

    Requires NeMo 2.2+ with multispeaker transcription support:
    - nemo.collections.asr.models.ASRModel
    - nemo.collections.asr.models.SortformerEncLabelModel
    - nemo.collections.asr.parts.utils.multispk_transcribe_utils.SpeakerTaggedASR
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

# Suppress noisy warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# Encoder step length is 80ms for FastConformer models
ENCODER_STEP_LENGTH = 80  # ms
SAMPLE_RATE = 16000  # Hz

# Valid lookahead sizes for multi-speaker streaming (in ms)
# Maps ms to frame count
VALID_LOOKAHEADS = {
    80: 1,    # 1 frame = 80ms (lowest latency)
    160: 2,   # 2 frames = 160ms
    560: 7,   # 7 frames = 560ms
    1120: 13, # 13 frames = 1120ms (recommended for best accuracy)
}

# Default models
DEFAULT_ASR_MODEL = "nvidia/multitalker-parakeet-streaming-0.6b-v1"
DEFAULT_DIAR_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2.1"


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    for module in ["numpy", "torch", "soundfile", "nemo", "omegaconf"]:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    if missing:
        print("Error: Missing required dependencies:", ", ".join(missing))
        print("\nInstall with:")
        print("  cd scripts && ./setup.sh")
        sys.exit(1)

    # Check for multispeaker transcription API
    try:
        from nemo.collections.asr.models import ASRModel
        from nemo.collections.asr.models import SortformerEncLabelModel
        from nemo.collections.asr.parts.utils.multispk_transcribe_utils import (
            SpeakerTaggedASR,
        )
    except ImportError as e:
        print("Error: NeMo's multispeaker transcription API not found.")
        print(f"Import error: {e}")
        print("\nThis script requires NeMo 2.2+ with:")
        print("  - nemo.collections.asr.models.ASRModel")
        print("  - nemo.collections.asr.models.SortformerEncLabelModel")
        print("  - nemo.collections.asr.parts.utils.multispk_transcribe_utils.SpeakerTaggedASR")
        print("\nPlease upgrade NeMo or check your installation.")
        sys.exit(1)


def lookahead_to_att_context(lookahead_ms: int, left_context: int = 70) -> list:
    """
    Convert lookahead in milliseconds to attention context size.

    Args:
        lookahead_ms: Lookahead size in milliseconds
        left_context: Left context size (default: 70 for FastConformer)

    Returns:
        [left_context, right_context] list for att_context_size config
    """
    if lookahead_ms in VALID_LOOKAHEADS:
        right_context = VALID_LOOKAHEADS[lookahead_ms]
    else:
        # For non-standard lookahead, compute frames (80ms per frame)
        right_context = lookahead_ms // ENCODER_STEP_LENGTH
        print(
            f"Warning: Non-standard lookahead {lookahead_ms}ms -> {right_context} frames"
        )
        print(
            f"Valid options for multispeaker models: {list(VALID_LOOKAHEADS.keys())} ms"
        )

    return [left_context, right_context]


def load_audio(audio_path: str, target_sample_rate: int = 16000):
    """Load and optionally resample audio file."""
    import soundfile as sf
    import numpy as np

    data, sample_rate = sf.read(audio_path, dtype="float32")

    # Convert stereo to mono if needed
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Resample if needed
    if sample_rate != target_sample_rate:
        try:
            import librosa

            print(f"Resampling from {sample_rate} Hz to {target_sample_rate} Hz...")
            data = librosa.resample(
                data, orig_sr=sample_rate, target_sr=target_sample_rate
            )
        except ImportError:
            print(
                f"Warning: librosa not installed, cannot resample from {sample_rate} Hz"
            )
            print("Audio must be 16kHz for accurate transcription.")

    return data, target_sample_rate


def build_config(args):
    """
    Build the pipeline configuration from command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        OmegaConf DictConfig for the pipeline
    """
    from omegaconf import OmegaConf

    # Load base config
    config_path = Path(__file__).parent / "streaming_multispeaker_config.yaml"
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    cfg = OmegaConf.load(config_path)

    # Apply CLI overrides
    cfg.asr.model_name = args.asr_model
    cfg.asr.device = args.device
    cfg.diarization.model_name = args.diar_model
    cfg.diarization.device = args.device

    # Set compute dtype based on device
    if args.device == "cuda":
        cfg.asr.compute_dtype = "bfloat16"
        cfg.asr.use_amp = True
    else:
        cfg.asr.compute_dtype = "float32"
        cfg.asr.use_amp = False

    # Set attention context size based on lookahead
    cfg.streaming.att_context_size = lookahead_to_att_context(args.lookahead)

    # Set max speakers
    cfg.multispeaker.max_num_of_spks = args.max_speakers

    # Set session ID from audio filename
    if args.audio:
        cfg.output.session_id = Path(args.audio).stem

    # Set log level
    if args.quiet:
        cfg.log_level = 40  # ERROR
    elif args.verbose:
        cfg.log_level = 10  # DEBUG
    else:
        cfg.log_level = 30  # WARNING

    return cfg


def load_models(cfg, verbose: bool = False):
    """
    Load ASR and diarization models.

    Args:
        cfg: Configuration object
        verbose: Print verbose output

    Returns:
        tuple: (asr_model, diar_model)
    """
    import torch
    from nemo.collections.asr.models import ASRModel
    from nemo.collections.asr.models import SortformerEncLabelModel

    device = cfg.asr.device
    if device == "mps":
        # MPS may not be fully supported, fall back to CPU
        print("Warning: MPS device may not be fully supported, using CPU")
        device = "cpu"

    print(f"Loading ASR model: {cfg.asr.model_name}")
    asr_model = ASRModel.from_pretrained(cfg.asr.model_name, map_location=device)
    asr_model.eval()

    print(f"Loading diarization model: {cfg.diarization.model_name}")
    diar_model = SortformerEncLabelModel.from_pretrained(
        cfg.diarization.model_name, map_location=device
    )
    diar_model.eval()

    if device == "cuda" and torch.cuda.is_available():
        asr_model = asr_model.cuda()
        diar_model = diar_model.cuda()

    return asr_model, diar_model


def format_seglst_output(seglsts, session_id: str) -> list:
    """
    Format SegLST output for JSON serialization.

    Args:
        seglsts: List of segment dictionaries from SpeakerTaggedASR
        session_id: Session identifier

    Returns:
        List of segment dictionaries
    """
    output = []
    for seg in seglsts:
        # seglsts are dictionaries with keys: speaker, start_time, end_time, words, session_id
        segment = {
            "session_id": session_id,
            "speaker": seg.get("speaker", "speaker_0"),
            "words": seg.get("words", ""),
            "start_time": float(seg.get("start_time", 0)),
            "end_time": float(seg.get("end_time", 0)),
        }
        if segment["words"].strip():
            output.append(segment)
    return output


def print_speaker_transcript(seglsts, last_printed_idx: int = 0) -> int:
    """
    Print new speaker-tagged transcript segments.

    Args:
        seglsts: List of segment dictionaries
        last_printed_idx: Index of last printed segment

    Returns:
        New last_printed_idx
    """
    for seg in seglsts[last_printed_idx:]:
        # seglsts are dictionaries with keys: speaker, start_time, end_time, words
        speaker = seg.get("speaker", "speaker_0")
        words = seg.get("words", "")
        start = seg.get("start_time", 0)
        end = seg.get("end_time", 0)

        if words.strip():
            print(f"[{start:.2f}-{end:.2f}] {speaker}: {words}")

    return len(seglsts)


def run_multispeaker_streaming(
    asr_model,
    diar_model,
    cfg,
    audio_path: str,
    verbose: bool = False,
):
    """
    Run multi-speaker streaming inference on an audio file.

    Args:
        asr_model: Loaded ASR model
        diar_model: Loaded diarization model
        cfg: Configuration object
        audio_path: Path to the audio file
        verbose: If True, print debug information

    Returns:
        tuple: (seglst_output, elapsed_time, audio_duration)
    """
    import torch
    import soundfile as sf
    from nemo.collections.asr.parts.utils.multispk_transcribe_utils import (
        SpeakerTaggedASR,
    )
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    # Get audio info
    info = sf.info(audio_path)
    audio_duration = info.duration
    sample_rate = info.samplerate

    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Sample rate: {sample_rate} Hz")

    # Create SpeakerTaggedASR configuration
    from omegaconf import OmegaConf

    # Get session ID for manifest
    session_id = cfg.output.session_id or Path(audio_path).stem

    speaker_cfg = OmegaConf.create({
        # Required: Either manifest_file or audio_file must be set
        "manifest_file": None,
        "audio_file": audio_path,  # Use audio_file mode for single file processing

        # Multi-speaker settings
        "max_num_of_spks": cfg.multispeaker.max_num_of_spks,
        "att_context_size": cfg.streaming.att_context_size,
        "binary_diar_preds": cfg.multispeaker.binary_diar_preds,
        "masked_asr": cfg.multispeaker.masked_asr,
        "sent_break_sec": cfg.multispeaker.sent_break_sec,
        "batch_size": cfg.streaming.batch_size,

        # ASR speaker tagging settings
        "fix_prev_words_count": 15,  # Number of previous words to keep fixed
        "update_prev_words_sentence": 10,  # Words to update in sentence
        "ignored_initial_frame_steps": 0,  # Initial frames to ignore
        "word_window": 10,  # Word window size for speaker assignment

        # Optional settings
        "cache_gating": False,
        "cache_gating_buffer_size": 2,
        "mask_preencode": False,
        "single_speaker_mode": False,
        "verbose": verbose,
        "generate_realtime_scripts": False,
        "deploy_mode": False,  # Using audio_file mode instead
        "online_normalization": True,  # Normalize audio online
        "pad_and_drop_preencoded": True,  # Handle pre-encoded frame padding
    })

    # Initialize SpeakerTaggedASR
    print("\nInitializing SpeakerTaggedASR pipeline...")
    speaker_tagger = SpeakerTaggedASR(
        cfg=speaker_cfg,
        asr_model=asr_model,
        diar_model=diar_model,
    )

    # Create CacheAwareStreamingAudioBuffer for proper audio chunking
    print("Setting up streaming audio buffer...")
    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=asr_model,
        online_normalization=speaker_cfg.get("online_normalization", True),
        pad_and_drop_preencoded=speaker_cfg.get("pad_and_drop_preencoded", True),
    )
    streaming_buffer.append_audio_file(audio_filepath=audio_path, stream_id=-1)

    # Get streaming parameters
    right_context = cfg.streaming.att_context_size[1]
    print(f"Lookahead: {right_context} frames ({right_context * ENCODER_STEP_LENGTH}ms)")

    print("\n--- Multi-Speaker Streaming Transcription ---")

    start_time = time.time()
    step_num = 0
    last_printed_idx = 0
    seglsts = []

    # Get device for inference
    device = next(asr_model.parameters()).device

    # Process audio through streaming buffer
    streaming_buffer_iter = iter(streaming_buffer)

    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
        # Move tensors to device
        chunk_audio = chunk_audio.to(device)
        chunk_lengths = chunk_lengths.to(device)

        # Determine drop_extra_pre_encoded
        if step_num == 0 and not speaker_cfg.get("pad_and_drop_preencoded", True):
            drop_extra_pre_encoded = 0
        else:
            drop_extra_pre_encoded = asr_model.encoder.streaming_cfg.drop_extra_pre_encoded

        # Process chunk through SpeakerTaggedASR
        try:
            with torch.inference_mode():
                speaker_tagger.perform_parallel_streaming_stt_spk(
                    step_num=step_num,
                    chunk_audio=chunk_audio,
                    chunk_lengths=chunk_lengths,
                    is_buffer_empty=streaming_buffer.is_buffer_empty(),
                    drop_extra_pre_encoded=drop_extra_pre_encoded,
                )

            # Get current transcripts and print new segments
            if hasattr(speaker_tagger, "instance_manager"):
                batch_states = speaker_tagger.instance_manager.batch_asr_states
                if batch_states and len(batch_states) > 0:
                    current_seglsts = batch_states[0].seglsts
                    if current_seglsts:
                        seglsts = current_seglsts
                        last_printed_idx = print_speaker_transcript(
                            seglsts, last_printed_idx
                        )

            if verbose:
                print(f"[Step {step_num}] Processed chunk, buffer_empty={streaming_buffer.is_buffer_empty()}")

        except Exception as e:
            if verbose:
                print(f"[Step {step_num}] Error: {e}")
                import traceback
                traceback.print_exc()

    elapsed = time.time() - start_time

    # Generate final SegLST output
    samples = [{'audio_filepath': audio_path}]
    try:
        speaker_tagger.generate_seglst_dicts_from_parallel_streaming(samples=samples)
        final_seglsts = speaker_tagger.instance_manager.seglst_dict_list
    except Exception as e:
        if verbose:
            print(f"Error generating final seglsts: {e}")
        final_seglsts = seglsts

    # Format output
    seglst_output = format_seglst_output(final_seglsts, session_id)

    return seglst_output, elapsed, audio_duration


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Speaker Streaming ASR with Diarization using NeMo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic multi-speaker transcription
    python scripts/streaming_multispeaker.py samples/Speaker26_004_16k.wav

    # With custom settings
    python scripts/streaming_multispeaker.py \\
        --max-speakers 2 \\
        --lookahead 1120 \\
        --device cuda \\
        samples/meeting.wav

    # Save output to JSON file
    python scripts/streaming_multispeaker.py \\
        -o output.json \\
        samples/conversation.wav

Lookahead options (latency vs accuracy tradeoff):
    80ms   - Lowest latency, lower accuracy (1 frame)
    160ms  - Low latency (2 frames)
    560ms  - Medium latency (7 frames)
    1120ms - Best accuracy, highest latency (13 frames, recommended)

Output format (SegLST):
    [
      {"session_id": "audio", "speaker": "speaker_0", "words": "Hello", "start_time": 0.08, "end_time": 1.52},
      {"session_id": "audio", "speaker": "speaker_1", "words": "Hi there", "start_time": 1.60, "end_time": 2.24}
    ]
        """,
    )
    parser.add_argument(
        "audio",
        type=str,
        nargs="?",
        help="Path to the audio file (.wav, 16kHz mono recommended)",
    )
    parser.add_argument(
        "--asr-model",
        type=str,
        default=DEFAULT_ASR_MODEL,
        help=f"ASR model name (default: {DEFAULT_ASR_MODEL})",
    )
    parser.add_argument(
        "--diar-model",
        type=str,
        default=DEFAULT_DIAR_MODEL,
        help=f"Diarization model name (default: {DEFAULT_DIAR_MODEL})",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=4,
        help="Maximum number of speakers to detect (default: 4, max: 4)",
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=1120,
        help="Lookahead size in milliseconds (default: 1120). "
        "Valid options: 80, 160, 560, or 1120.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run inference on (default: cpu)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output SegLST JSON file path",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress logging output"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose/debug output"
    )

    args = parser.parse_args()

    # Check for audio file
    if not args.audio:
        print("Error: No audio file specified.")
        parser.print_help()
        return 1

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1

    # Validate max speakers
    if args.max_speakers < 1 or args.max_speakers > 4:
        print(f"Error: max-speakers must be between 1 and 4, got {args.max_speakers}")
        return 1

    # Suppress logging if quiet mode
    if args.quiet:
        import logging

        for logger_name in ["nemo", "nemo_logger", "pytorch_lightning", "lightning"]:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        os.environ["NEMO_LOG_LEVEL"] = "ERROR"

    # Check dependencies
    check_dependencies()

    # Build configuration
    print(f"ASR Model: {args.asr_model}")
    print(f"Diarization Model: {args.diar_model}")
    print(f"Device: {args.device}")
    print(f"Max Speakers: {args.max_speakers}")
    print(f"Lookahead: {args.lookahead}ms")

    cfg = build_config(args)

    # Load models
    print("\nLoading models...")
    load_start = time.time()
    asr_model, diar_model = load_models(cfg, args.verbose)
    print(f"Models loaded in {time.time() - load_start:.2f}s")

    # Run inference
    print(f"\nProcessing: {audio_path}")

    seglst_output, elapsed, audio_duration = run_multispeaker_streaming(
        asr_model, diar_model, cfg, str(audio_path), args.verbose
    )

    # Print final results
    print("\n--- Final Result ---")
    if seglst_output:
        for seg in seglst_output:
            print(f"[{seg['start_time']:.2f}-{seg['end_time']:.2f}] {seg['speaker']}: {seg['words']}")
    else:
        print("No transcripts generated.")

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(seglst_output, f, indent=2)
        print(f"\nOutput saved to: {output_path}")

    # Statistics
    print("\n--- Statistics ---")
    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Processing time: {elapsed:.2f}s")
    print(f"Real-time factor: {elapsed / audio_duration:.2f}x")
    print(f"Lookahead: {args.lookahead}ms")
    print(f"Attention context: {cfg.streaming.att_context_size}")
    print(f"Segments: {len(seglst_output)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
