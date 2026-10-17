use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
use std::{fs, path::PathBuf};

/// Shared eval set (audio + references), built by mlx-models `samples/fetch_eval.py`.
const EVAL_REPO: &str = "animaslabs/asr-eval";
/// Bump together with `EVAL_REVISION` in mlx-models `samples/score.py`.
const EVAL_REVISION: &str = "b74107914b1f6530a7096fd6f6b1bcb047486237";

#[derive(Parser)]
#[command(name = "xtask", about = "Development tasks for mlx-voice-rs")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Download a model from Hugging Face
    Download {
        /// Repository ID (e.g., nvidia/parakeet-ctc-0.6b)
        repo: String,

        /// Output directory (default: models/<repo-name>)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Specific files to download (downloads all if not specified)
        #[arg(short, long)]
        file: Vec<String>,

        /// Hugging Face API token (for gated/private models)
        /// Can also be set via HF_TOKEN environment variable
        #[arg(long)]
        token: Option<String>,
    },

    /// Download the shared ASR eval set (private HF dataset) to samples/eval/
    DownloadEval {
        /// Output directory (default: samples/eval)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Dataset revision (default: the pinned revision)
        #[arg(long, default_value = EVAL_REVISION)]
        revision: String,

        /// Hugging Face API token; falls back to HF_TOKEN, then `hf auth login`
        #[arg(long)]
        token: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Download {
            repo,
            output,
            file,
            token,
        } => {
            // Check for token in environment if not provided
            let token = token.or_else(|| std::env::var("HF_TOKEN").ok());
            download_model(&repo, output, file, token)?;
        }
        Commands::DownloadEval {
            output,
            revision,
            token,
        } => {
            let token = token.or_else(|| std::env::var("HF_TOKEN").ok());
            download_eval(output, revision, token)?;
        }
    }
    Ok(())
}

fn download_model(
    repo_id: &str,
    output: Option<PathBuf>,
    files: Vec<String>,
    token: Option<String>,
) -> Result<()> {
    let api = build_api(token)?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

    let output_dir = output.unwrap_or_else(|| project_root().join("models").join(repo_id));
    fs::create_dir_all(&output_dir).context("Failed to create output directory")?;

    println!("Downloading {repo_id}...");

    if files.is_empty() {
        let repo_info = repo.info().with_context(|| {
            format!("Failed to get repo info for {repo_id}. If the model is gated/private, provide --token or set HF_TOKEN")
        })?;

        for file in repo_info.siblings {
            let filename = &file.rfilename;
            // Skip directories and hidden files
            if filename.contains('/') || filename.starts_with('.') {
                continue;
            }
            download_file(&repo, filename, &output_dir)?;
        }
    } else {
        for filename in &files {
            download_file(&repo, filename, &output_dir)?;
        }
    }

    println!("Done! Files saved to {}", output_dir.display());
    Ok(())
}

fn download_eval(output: Option<PathBuf>, revision: String, token: Option<String>) -> Result<()> {
    let api = build_api(token)?;
    let repo = api.repo(Repo::with_revision(
        EVAL_REPO.to_string(),
        RepoType::Dataset,
        revision.clone(),
    ));

    let output_dir = output.unwrap_or_else(|| project_root().join("samples").join("eval"));
    fs::create_dir_all(&output_dir).context("Failed to create output directory")?;

    println!(
        "Downloading {EVAL_REPO}@{}...",
        &revision[..revision.len().min(8)]
    );
    let info = repo.info().with_context(|| {
        format!("Failed to get repo info for {EVAL_REPO}. It's private: run `hf auth login`, or pass --token / set HF_TOKEN")
    })?;
    for file in info.siblings {
        let filename = &file.rfilename;
        let wanted = filename == "manifest.jsonl"
            || filename.ends_with(".wav")
            || filename.ends_with(".txt");
        if wanted && !filename.contains('/') {
            download_file(&repo, filename, &output_dir)?;
        }
    }

    println!("Done! Eval set saved to {}", output_dir.display());
    Ok(())
}

/// Only override the token when one is given; otherwise keep the one hf-hub
/// reads from the `hf auth login` cache.
fn build_api(token: Option<String>) -> Result<hf_hub::api::sync::Api> {
    let mut builder = ApiBuilder::new();
    if token.is_some() {
        builder = builder.with_token(token);
    }
    builder.build().context("Failed to create HF API client")
}

fn download_file(
    repo: &hf_hub::api::sync::ApiRepo,
    filename: &str,
    output_dir: &PathBuf,
) -> Result<()> {
    println!("  {filename}...");
    let src = repo
        .get(filename)
        .with_context(|| format!("Failed to download {filename}"))?;
    let dest = output_dir.join(filename);
    fs::copy(&src, &dest).with_context(|| format!("Failed to copy {filename}"))?;
    Ok(())
}

fn project_root() -> PathBuf {
    let dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(dir).parent().unwrap().to_path_buf()
}
