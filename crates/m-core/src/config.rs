//! Config: TOML at ~/.config/m/config.toml, with a built-in zero-config
//! default profile pointing at the local llama-server.

use serde::Deserialize;

use crate::error::{Error, Result};

#[derive(Debug, Clone, Deserialize)]
pub struct Profile {
    pub base_url: String,
    #[serde(default)]
    pub api_key: String,
    pub model: String,
    /// Context window used for the overflow guard when the server doesn't
    /// expose one (llama-server does via /props).
    #[serde(default = "default_ctx")]
    pub ctx: usize,
    /// Omitted → server-side default sampling (right for the local server).
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

fn default_ctx() -> usize {
    131_072
}

impl Default for Profile {
    fn default() -> Self {
        Profile {
            base_url: "http://localhost:8080".into(),
            api_key: "none".into(),
            model: "local".into(),
            ctx: default_ctx(),
            temperature: None,
            max_tokens: None,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ConfigFile {
    #[serde(default)]
    pub default_profile: Option<String>,
    #[serde(default)]
    pub profiles: std::collections::BTreeMap<String, Profile>,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub profile_name: String,
    pub profile: Profile,
    /// Cap on agent iterations (model call + tools = 1 turn). 0 = unlimited.
    pub max_turns: usize,
    /// Ask before write/edit/bash (TUI only). Default: YOLO.
    pub confirm: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            profile_name: "local".into(),
            profile: Profile::default(),
            max_turns: 0,
            confirm: false,
        }
    }
}

pub fn config_dir() -> std::path::PathBuf {
    dirs::config_dir().unwrap_or_else(|| ".".into()).join("m")
}

pub fn data_dir() -> std::path::PathBuf {
    dirs::data_dir().unwrap_or_else(|| ".".into()).join("m")
}

/// Load config, selecting `profile` (CLI flag) > M_PROFILE env >
/// default_profile from file > built-in "local".
pub fn load(profile: Option<&str>) -> Result<Config> {
    let path = config_dir().join("config.toml");
    let file: ConfigFile = match std::fs::read_to_string(&path) {
        Ok(s) => toml::from_str(&s).map_err(|e| Error::msg(format!("{}: {e}", path.display())))?,
        Err(_) => ConfigFile::default(),
    };
    let env_profile = std::env::var("M_PROFILE").ok();
    let name = profile
        .map(str::to_string)
        .or(env_profile)
        .or(file.default_profile.clone())
        .unwrap_or_else(|| "local".into());
    let prof = match file.profiles.get(&name) {
        Some(p) => p.clone(),
        None if name == "local" => Profile::default(),
        None => {
            return Err(Error::msg(format!(
                "profile '{name}' not found in {} (available: {})",
                path.display(),
                if file.profiles.is_empty() {
                    "built-in 'local'".to_string()
                } else {
                    file.profiles.keys().cloned().collect::<Vec<_>>().join(", ")
                }
            )));
        }
    };
    Ok(Config {
        profile_name: name,
        profile: prof,
        max_turns: 0,
        confirm: false,
    })
}
