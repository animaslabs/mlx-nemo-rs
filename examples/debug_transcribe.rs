//! Debug example: Output intermediate values for comparison with Python/NeMo.
//!
//! Usage:
//!   cargo run --example debug_transcribe --release -- --model path/to/model.nemo audio.wav

use std::time::Instant;

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;

use mlx_nemo::asr::EncDecCTCModel;

fn print_tensor_stats(name: &str, tensor: &Array) {
    tensor.eval().unwrap();

    let shape: Vec<i32> = tensor.shape().to_vec();

    // Get statistics
    let min_val = mlx_rs::ops::min(tensor, false).unwrap();
    let max_val = mlx_rs::ops::max(tensor, false).unwrap();
    let mean_val = mlx_rs::ops::mean(tensor, false).unwrap();

    // Compute std: sqrt(mean((x - mean)^2))
    let centered = tensor - &mean_val;
    let squared = &centered * &centered;
    let var = mlx_rs::ops::mean(&squared, false).unwrap();
    let std_val = mlx_rs::ops::sqrt(&var).unwrap();

    // Compute median by sorting and taking middle value
    let flat = tensor.reshape(&[-1]).unwrap();
    let sorted = mlx_rs::ops::sort(&flat).unwrap();

    min_val.eval().unwrap();
    max_val.eval().unwrap();
    mean_val.eval().unwrap();
    std_val.eval().unwrap();
    sorted.eval().unwrap();

    let min_f: f32 = min_val.as_slice::<f32>()[0];
    let max_f: f32 = max_val.as_slice::<f32>()[0];
    let mean_f: f32 = mean_val.as_slice::<f32>()[0];
    let std_f: f32 = std_val.as_slice::<f32>()[0];

    let sorted_data: &[f32] = sorted.as_slice();
    let median_f = sorted_data[sorted_data.len() / 2];

    println!("\n{}:", name);
    println!("  Shape: {:?}", shape);
    println!("  Min: {:.6}", min_f);
    println!("  Max: {:.6}", max_f);
    println!("  Mean: {:.6}", mean_f);
    println!("  Std: {:.6}", std_f);
    println!("  Median: {:.6}", median_f);

    // Print sample values at specific positions
    if shape.len() == 3 {
        let c = shape[1];
        let t = shape[2];

        if c >= 1 && t >= 5 {
            let slice1 = tensor.index((0, 0, ..5));
            slice1.eval().unwrap();
            let data1: &[f32] = slice1.as_slice();
            println!("  [0,0,:5]: {:?}", data1);

            // Middle channel
            let mid_c = c / 2;
            let slice2 = tensor.index((0, mid_c, ..5));
            slice2.eval().unwrap();
            let data2: &[f32] = slice2.as_slice();
            println!("  [0,{},:5]: {:?}", mid_c, data2);
        }
    }
}

fn print_per_channel_stats(name: &str, tensor: &Array) {
    tensor.eval().unwrap();
    let shape = tensor.shape();

    if shape.len() != 3 {
        return;
    }

    // tensor is [batch, channels, time]
    // Compute mean and std for each channel (over time dimension)
    let means = mlx_rs::ops::mean_axis(tensor, 2, false).unwrap(); // [batch, channels]
    let centered = tensor - &mlx_rs::ops::mean_axis(tensor, 2, true).unwrap();
    let squared = &centered * &centered;
    let var = mlx_rs::ops::mean_axis(&squared, 2, false).unwrap();
    let stds = mlx_rs::ops::sqrt(&var).unwrap();

    means.eval().unwrap();
    stds.eval().unwrap();

    let means_data: &[f32] = means.as_slice();
    let stds_data: &[f32] = stds.as_slice();

    let mean_min = means_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let mean_max = means_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let std_min = stds_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let std_max = stds_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!("\n{} per-channel statistics:", name);
    println!("  Channel means range: [{:.4}, {:.4}]", mean_min, mean_max);
    println!("  Channel stds range: [{:.4}, {:.4}]", std_min, std_max);
}

fn main() -> mlx_nemo::Result<()> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 4 || args[1] != "--model" {
        eprintln!("Usage: {} --model <model.nemo> <audio.wav>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[2];
    let audio_path = &args[3];

    println!("============================================================");
    println!("  Rust Implementation Statistics");
    println!("============================================================");
    println!("Model: {}", model_path);
    println!("Audio: {}", audio_path);

    // Load model
    let start = Instant::now();
    let mut model = EncDecCTCModel::restore(model_path)?;
    println!("\nModel loaded in {:.2}s", start.elapsed().as_secs_f32());

    // Print encoder config
    let config = model.config();
    println!("\nEncoder config:");
    println!("  d_model: {}", config.encoder.d_model);
    println!("  n_layers: {}", config.encoder.n_layers);
    println!("  n_heads: {}", config.encoder.n_heads);
    println!("  conv_kernel_size: {}", config.encoder.conv_kernel_size);

    // Print filterbank info for debugging
    let filterbank = model.preprocessor().filterbank();
    filterbank.eval().unwrap();
    let fb_shape = filterbank.shape();
    println!("\nFilterbank shape: {:?}", fb_shape);
    let fb_max = mlx_rs::ops::max(filterbank, false).unwrap();
    fb_max.eval().unwrap();
    println!("Filterbank max: {:.6}", fb_max.as_slice::<f32>()[0]);

    // Sum per filter (row)
    let fb_sums = mlx_rs::ops::sum_axis(filterbank, 1, false).unwrap();
    fb_sums.eval().unwrap();
    let sums: &[f32] = fb_sums.as_slice();
    println!(
        "Filter sums: 0={:.4}, 40={:.4}, 79={:.4}",
        sums[0], sums[40], sums[79]
    );

    // First filter non-zero values
    let filter0 = filterbank.index((0, ..));
    filter0.eval().unwrap();
    let f0_data: &[f32] = filter0.as_slice();
    let nz0: Vec<(usize, f32)> = f0_data
        .iter()
        .enumerate()
        .filter(|(_, &v)| v > 0.0)
        .map(|(i, &v)| (i, v))
        .collect();
    println!(
        "Filter 0 non-zero: {:?}",
        nz0.iter().take(5).collect::<Vec<_>>()
    );

    let filter40 = filterbank.index((40, ..));
    filter40.eval().unwrap();
    let f40_data: &[f32] = filter40.as_slice();
    let nz40: Vec<(usize, f32)> = f40_data
        .iter()
        .enumerate()
        .filter(|(_, &v)| v > 0.0)
        .map(|(i, &v)| (i, v))
        .collect();
    println!(
        "Filter 40 non-zero: {:?}",
        nz40.iter().take(5).collect::<Vec<_>>()
    );

    // Load audio
    use hound::WavReader;
    let reader = WavReader::open(audio_path).expect("Failed to open WAV file");
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .expect("Failed to read samples"),
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1 << (bits - 1)) as f32;
            reader
                .into_samples::<i32>()
                .collect::<Result<Vec<_>, _>>()
                .expect("Failed to read samples")
                .iter()
                .map(|&s| s as f32 / max_val)
                .collect()
        }
    };

    let samples: Vec<f32> = if channels > 1 {
        samples
            .chunks(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples
    };

    println!(
        "\nAudio: {} samples ({:.3}s) at {} Hz",
        samples.len(),
        samples.len() as f32 / sample_rate as f32,
        sample_rate
    );

    // Create audio tensor
    let audio = Array::from_slice(&samples, &[1, samples.len() as i32]);
    let lengths = Array::from_slice(&[samples.len() as i32], &[1]);

    // Run forward with debug
    let start = Instant::now();
    let debug_output = model.forward_debug(&audio, Some(&lengths))?;
    println!(
        "Inference completed in {:.3}s",
        start.elapsed().as_secs_f32()
    );

    // Print statistics for each stage
    print_tensor_stats(
        "Mel Pre-Norm (log mel before normalization)",
        &debug_output.mel_pre_norm,
    );
    print_tensor_stats(
        "Mel Spectrogram (after normalization)",
        &debug_output.mel_spectrogram,
    );
    print_per_channel_stats("Mel", &debug_output.mel_spectrogram);

    print_tensor_stats("Encoder Output", &debug_output.encoder_output);
    print_per_channel_stats("Encoder", &debug_output.encoder_output);

    print_tensor_stats("Decoder Logits", &debug_output.log_probs);

    // Run transcription
    let transcriptions = model.transcribe(&[audio_path])?;
    println!("\nTranscription: \"{}\"", transcriptions[0]);

    Ok(())
}
