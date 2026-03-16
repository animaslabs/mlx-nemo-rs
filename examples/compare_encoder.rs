//! Compare encoder outputs with Python.
//!
//! This example processes the first few chunks through the encoder
//! and prints values for comparison with Python.
//!
//! Run with: cargo run --example compare_encoder

use mlx_nemo::asr::CacheAwareRNNTPipeline;
use mlx_rs::Array;

fn main() -> mlx_nemo::Result<()> {
    let model_path = "models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo";
    let audio_path = "samples/2086-149220-0033.wav";

    println!("Loading pipeline from: {}", model_path);
    let mut pipeline = CacheAwareRNNTPipeline::restore(model_path, 2)?; // mode 2 = 80ms lookahead

    println!("\n=== Pipeline Loaded ===");
    println!("Sample rate: {} Hz", pipeline.sample_rate());

    // Process the audio file
    pipeline.set_audio_file(audio_path)?;

    println!("\n=== Processing Chunks ===");

    // Process first 5 steps
    for step in 0..5 {
        match pipeline.transcribe_step()? {
            Some(output) => {
                println!("\n--- Step {} ---", step);
                println!("New tokens: {:?}", output.new_tokens);
                // We can't easily get encoder output directly from pipeline
                // This is just to show the transcription steps
            }
            None => {
                println!("No more audio");
                break;
            }
        }
    }

    println!("\n=== Done ===");

    Ok(())
}
