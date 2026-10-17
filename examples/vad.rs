//! Example: Voice Activity Detection using a NeMo classification model.
//!
//! Usage:
//!   cargo run --example vad -- --model path/to/vad_model.nemo audio.wav
//!
//! Options:
//!   --onset <f32>       Speech onset threshold (default: 0.5)
//!   --offset <f32>      Speech offset threshold (default: 0.5)
//!   --pad-onset <f32>   Padding before speech in seconds (default: 0.0)
//!   --pad-offset <f32>  Padding after speech in seconds (default: 0.0)
//!   --min-on <f32>      Minimum speech duration in seconds (default: 0.0)
//!   --min-off <f32>     Minimum silence duration in seconds (default: 0.0)

use std::time::Instant;

use mlx_voice::asr::{EncDecClassificationModel, VadPostProcessConfig, VadPostProcessor};

fn main() -> mlx_voice::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 4 || args[1] != "--model" {
        eprintln!(
            "Usage: {} --model <vad_model.nemo> [options] <audio.wav> [audio2.wav ...]",
            args[0]
        );
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --onset <f32>       Speech onset threshold (default: 0.5)");
        eprintln!("  --offset <f32>      Speech offset threshold (default: 0.5)");
        eprintln!("  --pad-onset <f32>   Padding before speech (seconds, default: 0.0)");
        eprintln!("  --pad-offset <f32>  Padding after speech (seconds, default: 0.0)");
        eprintln!("  --min-on <f32>      Minimum speech duration (seconds, default: 0.0)");
        eprintln!("  --min-off <f32>     Minimum silence duration (seconds, default: 0.0)");
        std::process::exit(1);
    }

    let model_path = &args[2];

    // Parse optional flags
    let mut vad_config = VadPostProcessConfig::default();
    let mut audio_paths = Vec::new();
    let mut i = 3;

    while i < args.len() {
        match args[i].as_str() {
            "--onset" if i + 1 < args.len() => {
                vad_config.onset = args[i + 1].parse().unwrap_or(vad_config.onset);
                i += 2;
            }
            "--offset" if i + 1 < args.len() => {
                vad_config.offset = args[i + 1].parse().unwrap_or(vad_config.offset);
                i += 2;
            }
            "--pad-onset" if i + 1 < args.len() => {
                vad_config.pad_onset = args[i + 1].parse().unwrap_or(vad_config.pad_onset);
                i += 2;
            }
            "--pad-offset" if i + 1 < args.len() => {
                vad_config.pad_offset = args[i + 1].parse().unwrap_or(vad_config.pad_offset);
                i += 2;
            }
            "--min-on" if i + 1 < args.len() => {
                vad_config.min_duration_on =
                    args[i + 1].parse().unwrap_or(vad_config.min_duration_on);
                i += 2;
            }
            "--min-off" if i + 1 < args.len() => {
                vad_config.min_duration_off =
                    args[i + 1].parse().unwrap_or(vad_config.min_duration_off);
                i += 2;
            }
            other => {
                audio_paths.push(other.to_string());
                i += 1;
            }
        }
    }

    if audio_paths.is_empty() {
        eprintln!("Error: No audio files provided");
        std::process::exit(1);
    }

    println!("Loading VAD model from: {}", model_path);
    let start = Instant::now();

    let mut model = EncDecClassificationModel::from_pretrained(model_path)?;

    println!("Model loaded in {:.2}s", start.elapsed().as_secs_f32());
    println!("Sample rate: {} Hz", model.sample_rate());
    println!("Frame duration: {:.1}ms", model.frame_duration() * 1000.0);
    println!("Num classes: {}", model.num_classes());
    println!();

    // Run inference
    println!("Processing {} file(s)...", audio_paths.len());
    let start = Instant::now();

    let all_probs = model.infer(&audio_paths)?;

    let elapsed = start.elapsed().as_secs_f32();
    println!("Inference completed in {:.2}s", elapsed);
    println!();

    // Post-process
    let processor = VadPostProcessor::new(vad_config);
    let frame_duration = model.frame_duration();

    for (path, probs) in audio_paths.iter().zip(all_probs.iter()) {
        println!("File: {}", path);
        println!("  Frames: {}", probs.len());

        let audio_duration = probs.len() as f32 * frame_duration;
        println!("  Audio duration: {:.2}s", audio_duration);

        let segments = processor.process(probs, frame_duration, audio_duration);

        if segments.is_empty() {
            println!("  No speech detected");
        } else {
            println!("  Speech segments:");
            for (i, seg) in segments.iter().enumerate() {
                println!(
                    "    [{:3}] {:.3}s - {:.3}s  ({:.3}s)",
                    i,
                    seg.start,
                    seg.end,
                    seg.duration()
                );
            }
        }
        println!();
    }

    Ok(())
}
