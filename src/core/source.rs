//! Where to load a model from: a local `.nemo` archive or model directory, or
//! a Hugging Face Hub repository.

use std::fmt;
use std::path::{Path, PathBuf};

use crate::core::error::{Error, Result};

/// Where to load a model from.
///
/// Usually built from a string or path: an existing file is a `.nemo`
/// archive, an existing directory holds safetensors weights, a config and a
/// tokenizer, and anything else of the form `owner/name` is a Hugging Face
/// Hub repository (downloading needs the `hub` feature).
///
/// ```no_run
/// # fn main() -> mlx_voice::Result<()> {
/// use mlx_voice::asr::EncDecRnntModel;
/// use mlx_voice::core::source::{HubRepo, ModelSource};
///
/// let model = EncDecRnntModel::from_pretrained("models/parakeet-rnnt-0.6b.nemo")?;
/// let model = EncDecRnntModel::from_pretrained("nvidia/parakeet-rnnt-0.6b")?;
/// let model = EncDecRnntModel::from_pretrained(
///     HubRepo::new("nvidia/nemotron-speech-streaming-en-0.6b").revision("main"),
/// )?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ModelSource {
    /// A local `.nemo` archive or model directory.
    Local(PathBuf),
    /// A Hugging Face Hub repository.
    Hub(HubRepo),
}

/// A Hugging Face Hub model repository.
#[derive(Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct HubRepo {
    /// Repository id, e.g. `"nvidia/parakeet-rnnt-0.6b"`.
    pub id: String,
    /// Branch, tag or commit; the default branch if `None`.
    pub revision: Option<String>,
    /// Access token for gated or private repositories. Without one, the
    /// `HF_TOKEN` environment variable is used, then the token saved by
    /// `hf auth login`.
    token: Option<String>,
}

impl HubRepo {
    /// The repository `id`, e.g. `"nvidia/parakeet-rnnt-0.6b"`.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            revision: None,
            token: None,
        }
    }

    /// Use a branch, tag or commit other than the default branch.
    pub fn revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = Some(revision.into());
        self
    }

    /// Authenticate with `token`.
    pub fn token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }
}

impl fmt::Debug for HubRepo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HubRepo")
            .field("id", &self.id)
            .field("revision", &self.revision)
            .field("token", &self.token.as_ref().map(|_| "<redacted>"))
            .finish()
    }
}

impl From<HubRepo> for ModelSource {
    fn from(repo: HubRepo) -> Self {
        Self::Hub(repo)
    }
}

impl From<&Path> for ModelSource {
    fn from(path: &Path) -> Self {
        Self::Local(path.to_path_buf())
    }
}

impl From<PathBuf> for ModelSource {
    fn from(path: PathBuf) -> Self {
        Self::Local(path)
    }
}

impl From<&PathBuf> for ModelSource {
    fn from(path: &PathBuf) -> Self {
        Self::Local(path.clone())
    }
}

impl From<&str> for ModelSource {
    /// An existing path is local; otherwise an `owner/name` string is a Hub
    /// repository id.
    fn from(s: &str) -> Self {
        if !Path::new(s).exists() && is_repo_id(s) {
            Self::Hub(HubRepo::new(s))
        } else {
            Self::Local(PathBuf::from(s))
        }
    }
}

impl From<String> for ModelSource {
    fn from(s: String) -> Self {
        Self::from(s.as_str())
    }
}

impl From<&String> for ModelSource {
    fn from(s: &String) -> Self {
        Self::from(s.as_str())
    }
}

/// Whether `s` looks like a Hub repository id (`owner/name`).
fn is_repo_id(s: &str) -> bool {
    let valid = |part: &str| {
        !part.is_empty()
            && !part.starts_with('.')
            && part
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    };
    matches!(s.split_once('/'), Some((owner, name)) if valid(owner) && valid(name))
}

/// A source resolved to local files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LocalModel {
    /// A `.nemo` archive (uncompressed or gzip TAR)
    Nemo(PathBuf),
    /// A directory of safetensors weights, config and tokenizer
    Directory(PathBuf),
}

impl ModelSource {
    /// Find the model's files, downloading them from the Hub if needed.
    pub(crate) fn resolve(&self) -> Result<LocalModel> {
        match self {
            Self::Local(path) => resolve_local(path),
            Self::Hub(repo) => download(repo),
        }
    }
}

fn resolve_local(path: &Path) -> Result<LocalModel> {
    if path.is_dir() {
        Ok(LocalModel::Directory(path.to_path_buf()))
    } else if path.is_file() {
        Ok(LocalModel::Nemo(path.to_path_buf()))
    } else {
        Err(Error::checkpoint_at(
            format!("Model not found: {}", path.display()),
            path,
        ))
    }
}

/// File extensions downloaded for a safetensors model directory.
#[cfg(feature = "hub")]
const DIRECTORY_EXTENSIONS: &[&str] = &[
    "safetensors",
    "json",
    "yaml",
    "yml",
    "model",
    "vocab",
    "txt",
];

#[cfg(feature = "hub")]
fn download(repo: &HubRepo) -> Result<LocalModel> {
    use hf_hub::api::sync::ApiBuilder;
    use hf_hub::{Repo, RepoType};

    let hub_error = |message: String, e: hf_hub::api::sync::ApiError| {
        Error::download(
            format!(
                "{message} (if the repository is gated or private, pass a token or set HF_TOKEN; \
                 there is no local file or directory named `{}` either)",
                repo.id
            ),
            e,
        )
    };

    let token = repo
        .token
        .clone()
        .or_else(|| std::env::var("HF_TOKEN").ok());
    let mut builder = ApiBuilder::from_env().with_progress(false);
    if token.is_some() {
        builder = builder.with_token(token);
    }
    let api = builder
        .build()
        .map_err(|e| Error::download("Failed to create the Hugging Face API client", e))?;
    let api_repo = api.repo(match &repo.revision {
        Some(revision) => Repo::with_revision(repo.id.clone(), RepoType::Model, revision.clone()),
        None => Repo::new(repo.id.clone(), RepoType::Model),
    });

    let info = api_repo
        .info()
        .map_err(|e| hub_error(format!("Failed to list files of `{}`", repo.id), e))?;
    let files: Vec<String> = info.siblings.into_iter().map(|s| s.rfilename).collect();

    let get = |file: &str| {
        tracing::info!(repo = %repo.id, file, "downloading from Hugging Face Hub");
        api_repo
            .get(file)
            .map_err(|e| hub_error(format!("Failed to download `{file}` from `{}`", repo.id), e))
    };

    // NeMo repositories hold a single .nemo archive
    let nemo: Vec<&String> = files.iter().filter(|f| f.ends_with(".nemo")).collect();
    if !nemo.is_empty() {
        let name = repo.id.rsplit('/').next().unwrap_or_default();
        let preferred = format!("{name}.nemo");
        let file = match nemo.as_slice() {
            [only] => only.as_str(),
            _ => nemo
                .iter()
                .find(|f| **f == &preferred)
                .map(|f| f.as_str())
                .ok_or_else(|| {
                    Error::checkpoint(format!(
                        "`{}` has several .nemo files ({}); download one and load it by path",
                        repo.id,
                        nemo.iter()
                            .map(|f| f.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                })?,
        };
        return Ok(LocalModel::Nemo(get(file)?));
    }

    // Otherwise a directory of safetensors weights, config and tokenizer
    let mut weights_dir = None;
    for file in &files {
        let ext = Path::new(file).extension().and_then(|e| e.to_str());
        if !ext.is_some_and(|e| DIRECTORY_EXTENSIONS.contains(&e)) {
            continue;
        }
        let path = get(file)?;
        if file.ends_with(".safetensors") && weights_dir.is_none() {
            weights_dir = path.parent().map(Path::to_path_buf);
        }
    }
    weights_dir.map(LocalModel::Directory).ok_or_else(|| {
        Error::checkpoint(format!(
            "`{}` has neither a .nemo archive nor safetensors weights",
            repo.id
        ))
    })
}

#[cfg(not(feature = "hub"))]
fn download(repo: &HubRepo) -> Result<LocalModel> {
    Err(Error::not_implemented(format!(
        "downloading `{}` from the Hugging Face Hub needs the `hub` feature \
         (no local file or directory has that name)",
        repo.id
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repo_ids() {
        assert!(is_repo_id("nvidia/parakeet-rnnt-0.6b"));
        assert!(is_repo_id("mlx-community/parakeet_tdt.v2"));
        assert!(!is_repo_id("model.nemo"));
        assert!(!is_repo_id("./models/x"));
        assert!(!is_repo_id("/abs/path"));
        assert!(!is_repo_id("a/b/c"));
        assert!(!is_repo_id("owner/"));
    }

    #[test]
    fn test_string_sources() {
        assert_eq!(
            ModelSource::from("nvidia/parakeet-rnnt-0.6b"),
            ModelSource::Hub(HubRepo::new("nvidia/parakeet-rnnt-0.6b"))
        );
        // An existing path stays local even if it looks like a repo id
        assert_eq!(
            ModelSource::from("src/lib.rs"),
            ModelSource::Local(PathBuf::from("src/lib.rs"))
        );
        assert_eq!(
            ModelSource::from("missing.nemo"),
            ModelSource::Local(PathBuf::from("missing.nemo"))
        );
    }

    #[test]
    fn test_local_resolution() {
        assert_eq!(
            ModelSource::from("src").resolve().unwrap(),
            LocalModel::Directory(PathBuf::from("src"))
        );
        assert_eq!(
            ModelSource::from("src/lib.rs").resolve().unwrap(),
            LocalModel::Nemo(PathBuf::from("src/lib.rs"))
        );
        assert!(ModelSource::from("missing.nemo").resolve().is_err());
    }

    #[cfg(not(feature = "hub"))]
    #[test]
    fn test_hub_needs_feature() {
        let err = ModelSource::from("owner/missing-model")
            .resolve()
            .unwrap_err()
            .to_string();
        assert!(err.contains("`hub` feature"), "{err}");
    }

    #[test]
    fn test_token_is_not_printed() {
        let repo = HubRepo::new("a/b").token("secret");
        assert!(!format!("{repo:?}").contains("secret"));
    }
}
