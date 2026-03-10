use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use std::{fs, path::PathBuf};

#[derive(Parser)]
#[command(name = "xtask", about = "Development tasks for mlx-nemo-rs")]
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
    }
    Ok(())
}

fn download_model(
    repo_id: &str,
    output: Option<PathBuf>,
    files: Vec<String>,
    token: Option<String>,
) -> Result<()> {
    let api = ApiBuilder::new()
        .with_token(token)
        .build()
        .context("Failed to create HF API client")?;

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
