//! Debug decoder step by step to compare with Python.

use mlx_nemo::asr::CacheAwareRNNTPipeline;
use mlx_rs::Array;

fn main() -> mlx_nemo::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let model_path = "models/nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo";

    println!("Loading pipeline from: {}", model_path);
    let pipeline = CacheAwareRNNTPipeline::restore(model_path, 2)?;  // mode 2 = 80ms lookahead

    // Get the decoder component
    // Note: We'd need to add accessor methods to the pipeline to get decoder details

    println!("\n=== Pipeline Loaded ===");
    println!("Sample rate: {} Hz", pipeline.sample_rate());
    println!("Blank ID: {}", pipeline.blank_id());

    // Load audio and process
    let audio_path = "samples/2086-149220-0033.wav";
    println!("\n=== Processing audio: {} ===", audio_path);

    let mut pipeline = pipeline;
    pipeline.set_audio_file(audio_path)?;

    // Process first few steps
    let mut step = 0;
    while let Some(output) = pipeline.transcribe_step()? {
        step += 1;
        println!("\n--- Step {} ---", step);
        println!("New tokens: {:?}", output.new_tokens);
        println!("All tokens: {:?}", output.tokens);
        println!("Text: '{}'", output.text);
        println!("Is final: {}", output.is_final);

        if step >= 10 || output.is_final {
            break;
        }
    }

    println!("\n=== Done ===");

    Ok(())
}
