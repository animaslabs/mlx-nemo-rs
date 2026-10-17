//! Debug decoder step by step to compare with Python.

use mlx_voice::asr::{EncDecRnntModel, StreamingOptions};

fn main() -> mlx_voice::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let model_path =
        "models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo";

    println!("Loading model from: {}", model_path);
    let mut model = EncDecRnntModel::from_pretrained(model_path)?;

    println!("\n=== Model Loaded ===");
    println!("Sample rate: {} Hz", model.sample_rate());
    println!("Blank ID: {}", model.blank_id());

    // Load audio and process
    let audio_path = "samples/2086-149220-0033.wav";
    println!("\n=== Processing audio: {} ===", audio_path);

    let mut stream = model.stream(StreamingOptions::default().mode(2))?; // mode 2 = 80ms lookahead
    let (samples, sample_rate) = mlx_voice::audio::load_wav(audio_path)?;
    stream.push_audio(&samples, sample_rate)?;
    stream.end_audio()?;

    // Process first few steps
    let mut step = 0;
    while let Some(update) = stream.next_update()? {
        step += 1;
        println!("\n--- Step {} ---", step);
        let ids: Vec<usize> = update.tokens.iter().map(|t| t.id).collect();
        println!("New tokens: {:?}", ids);
        let all: Vec<usize> = stream.transcript().tokens.iter().map(|t| t.id).collect();
        println!("All tokens: {:?}", all);
        println!("Text: '{}'", stream.text());
        println!("Is final: {}", update.is_final);

        if step >= 10 || update.is_final {
            break;
        }
    }

    println!("\n=== Done ===");

    Ok(())
}
