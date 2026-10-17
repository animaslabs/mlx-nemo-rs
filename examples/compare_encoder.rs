//! Compare encoder outputs with Python.
//!
//! This example processes the first few chunks through the encoder
//! and prints values for comparison with Python.
//!
//! Run with: cargo run --example compare_encoder

use mlx_voice::asr::{EncDecRNNTModel, StreamingOptions};

fn main() -> mlx_voice::Result<()> {
    let model_path =
        "models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo";
    let audio_path = "samples/2086-149220-0033.wav";

    println!("Loading model from: {}", model_path);
    let mut model = EncDecRNNTModel::from_pretrained(model_path)?;
    let mut stream = model.stream(StreamingOptions::default().mode(2))?; // mode 2 = 80ms lookahead

    println!("\n=== Model Loaded ===");
    println!("Sample rate: {} Hz", stream.sample_rate());

    // Process the audio file
    let (samples, sample_rate) = mlx_voice::audio::load_wav(audio_path)?;
    stream.push_audio(&samples, sample_rate)?;
    stream.end_audio()?;

    println!("\n=== Processing Chunks ===");

    // Process first 5 steps
    for step in 0..5 {
        match stream.next_update()? {
            Some(update) => {
                println!("\n--- Step {} ---", step);
                let ids: Vec<usize> = update.tokens.iter().map(|t| t.id).collect();
                println!("New tokens: {:?}", ids);
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
