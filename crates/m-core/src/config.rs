//! Config: TOML at ~/.config/m/config.toml, with a built-in zero-config
//! default profile pointing at the local llama-server.
//!
//! A profile is a **provider** (base URL + credentials + optional model
//! catalog). The active selection is always `provider/model`, e.g.
//! `zai-coding-plan/glm-5.2` or `grok/grok-build`.

use std::collections::BTreeMap;
use std::process::Command;

use serde::Deserialize;

use crate::error::{Error, Result};

#[derive(Debug, Clone, Deserialize)]
pub struct Profile {
    pub base_url: String,
    #[serde(default)]
    pub api_key: String,
    /// If set, run this shell command and use stdout (trimmed) as the API key.
    /// Useful for reading a rotating token (e.g. `jq` on `~/.grok/auth.json`).
    /// Wins over `api_key` when non-empty.
    #[serde(default)]
    pub api_key_cmd: Option<String>,
    /// Default model for this provider.
    pub model: String,
    /// Optional catalog of models available on this provider. Used by
    /// `/model` listing and completion. Empty → only `model` is listed, but
    /// free-form model overrides are still allowed.
    #[serde(default)]
    pub models: Vec<String>,
    /// Context window used for the overflow guard when the server doesn't
    /// expose one (llama-server does via /props).
    #[serde(default = "default_ctx")]
    pub ctx: usize,
    /// Omitted → server-side default sampling (right for the local server).
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Extra HTTP headers sent on every chat request (e.g. Grok CLI proxy
    /// needs `X-XAI-Token-Auth` + `x-grok-client-version`). Values support
    /// `$ENV` / `${ENV}` expansion. `x-grok-model-override` is always set
    /// from the active `model` when present in this map.
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
}

fn default_ctx() -> usize {
    131_072
}

impl Default for Profile {
    fn default() -> Self {
        Profile {
            base_url: "http://localhost:8080".into(),
            api_key: "none".into(),
            api_key_cmd: None,
            model: "local".into(),
            models: Vec::new(),
            ctx: default_ctx(),
            temperature: None,
            max_tokens: None,
            headers: BTreeMap::new(),
        }
    }
}

impl Profile {
    /// Models shown in `/model` lists: the catalog if non-empty, otherwise
    /// just the default `model`. Always includes `model` even if omitted
    /// from the catalog.
    pub fn available_models(&self) -> Vec<String> {
        if self.models.is_empty() {
            return vec![self.model.clone()];
        }
        let mut out = self.models.clone();
        if !out.iter().any(|m| m == &self.model) {
            out.insert(0, self.model.clone());
        }
        out
    }

    /// Resolve the bearer token: `api_key_cmd` (if set) → env-expanded `api_key`.
    /// Returns empty string when the key is missing or the sentinel `"none"`.
    pub fn resolve_api_key(&self) -> Result<String> {
        if let Some(cmd) = self.api_key_cmd.as_deref().map(str::trim).filter(|c| !c.is_empty()) {
            let out = Command::new("sh")
                .arg("-c")
                .arg(cmd)
                .output()
                .map_err(|e| Error::msg(format!("api_key_cmd failed to spawn: {e}")))?;
            if !out.status.success() {
                let stderr = String::from_utf8_lossy(&out.stderr);
                return Err(Error::msg(format!(
                    "api_key_cmd exited {}: {}",
                    out.status.code().unwrap_or(-1),
                    stderr.trim()
                )));
            }
            let key = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if key.is_empty() {
                return Err(Error::msg("api_key_cmd produced an empty key"));
            }
            return Ok(key);
        }
        let key = expand_env(&self.api_key);
        if key.is_empty() || key == "none" {
            Ok(String::new())
        } else {
            Ok(key)
        }
    }

    /// Authorization + profile-configured extra headers, ready for `http::post_json`.
    pub fn request_headers(&self) -> Result<Vec<(String, String)>> {
        let mut out = Vec::new();
        let key = self.resolve_api_key()?;
        if !key.is_empty() {
            out.push(("Authorization".into(), format!("Bearer {key}")));
        }
        for (k, v) in &self.headers {
            // Keep the Grok CLI proxy's routing header in sync with the
            // active model (catalog switches would otherwise leave a stale
            // override pointing at the profile default).
            let val = if k.eq_ignore_ascii_case("x-grok-model-override") {
                self.model.clone()
            } else {
                expand_env(v)
            };
            out.push((k.clone(), val));
        }
        Ok(out)
    }
}

/// Expand a bare `$VAR` / `${VAR}` string; leave other values unchanged.
fn expand_env(s: &str) -> String {
    let s = s.trim();
    if let Some(name) = s.strip_prefix("${").and_then(|r| r.strip_suffix('}')) {
        return std::env::var(name).unwrap_or_default();
    }
    if let Some(name) = s.strip_prefix('$') {
        if !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return std::env::var(name).unwrap_or_default();
        }
    }
    s.to_string()
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ConfigFile {
    #[serde(default)]
    pub default_profile: Option<String>,
    #[serde(default)]
    pub profiles: std::collections::BTreeMap<String, Profile>,
}

impl ConfigFile {
    fn load_from_disk() -> Self {
        let path = config_dir().join("config.toml");
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| toml::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Profile names known to this file, always including built-in `local`.
    fn profile_names(&self) -> Vec<String> {
        let mut names: std::collections::BTreeSet<String> =
            self.profiles.keys().cloned().collect();
        names.insert("local".to_string());
        names.into_iter().collect()
    }

    fn get_profile(&self, name: &str) -> Result<Profile> {
        match self.profiles.get(name) {
            Some(p) => Ok(p.clone()),
            None if name == "local" => Ok(Profile::default()),
            None => Err(Error::msg(format!(
                "provider '{name}' not found (available: {})",
                self.profile_names().join(", ")
            ))),
        }
    }
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

impl Config {
    /// `provider/model` label used in the status bar and notices.
    pub fn selection_label(&self) -> String {
        format!("{}/{}", self.profile_name, self.profile.model)
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
///
/// The profile argument may be a bare provider (`zai-coding-plan`) or a
/// `provider/model` pair (`zai-coding-plan/glm-5-turbo`). Model ids that
/// themselves contain slashes (e.g. OpenRouter's `qwen/qwen3-coder`) are
/// handled by matching the longest known provider prefix.
pub fn load(profile: Option<&str>) -> Result<Config> {
    let path = config_dir().join("config.toml");
    let file: ConfigFile = match std::fs::read_to_string(&path) {
        Ok(s) => toml::from_str(&s).map_err(|e| Error::msg(format!("{}: {e}", path.display())))?,
        Err(_) => ConfigFile::default(),
    };
    let env_profile = std::env::var("M_PROFILE").ok();
    let raw = profile
        .map(str::to_string)
        .or(env_profile)
        .or(file.default_profile.clone())
        .unwrap_or_else(|| "local".into());
    let (name, model_override) = split_provider_model(&raw, &file.profile_names())?;
    let mut prof = file.get_profile(&name).map_err(|e| {
        // Include the config path in the not-found message.
        Error::msg(format!("{e} in {}", path.display()))
    })?;
    if let Some(m) = model_override {
        prof.model = m;
    }
    Ok(Config {
        profile_name: name,
        profile: prof,
        max_turns: 0,
        confirm: false,
    })
}

/// Split `provider` or `provider/model…` against the known provider names.
/// Longest-prefix wins so `or/qwen/qwen3-coder` → (`or`, `qwen/qwen3-coder`).
pub fn split_provider_model(
    raw: &str,
    providers: &[String],
) -> Result<(String, Option<String>)> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err(Error::msg("empty provider/model spec"));
    }
    let mut names = providers.to_vec();
    names.sort_by_key(|n| std::cmp::Reverse(n.len()));
    for n in &names {
        if raw == n {
            return Ok((n.clone(), None));
        }
        let prefix = format!("{n}/");
        if let Some(model) = raw.strip_prefix(&prefix) {
            if !model.is_empty() {
                return Ok((n.clone(), Some(model.to_string())));
            }
        }
    }
    Err(Error::msg(format!(
        "provider '{raw}' not found (available: {})",
        {
            let mut p = providers.to_vec();
            p.sort();
            p.join(", ")
        }
    )))
}

/// Names of every provider selectable via `load`: everything in
/// config.toml's `[profiles.*]`, plus the built-in "local".
pub fn profile_names() -> Vec<String> {
    ConfigFile::load_from_disk().profile_names()
}

/// Every selectable `provider/model` pair for `/model` listing.
pub fn list_selections() -> Vec<(String, String)> {
    let file = ConfigFile::load_from_disk();
    let mut out = Vec::new();
    for name in file.profile_names() {
        if let Ok(prof) = file.get_profile(&name) {
            for model in prof.available_models() {
                out.push((name.clone(), model));
            }
        }
    }
    out.sort();
    out
}

/// Resolve a `/model` argument into `(provider, model)`.
///
/// Accepts:
/// - `provider` — that provider's default model
/// - `provider/model` — explicit pair (model may contain `/`)
/// - `provider model` — space-separated form
/// - bare `model` — current provider if it offers it, else the unique
///   provider that does, else an error
pub fn resolve_model_spec(
    spec: &str,
    current_provider: &str,
    current_model: &str,
) -> Result<(String, String)> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Err(Error::msg("empty model spec"));
    }
    let file = ConfigFile::load_from_disk();
    let providers = file.profile_names();

    // Space form: `provider model` (model may contain further spaces? no —
    // take first token as provider if it matches, rest as model).
    if let Some((a, b)) = spec.split_once(char::is_whitespace) {
        let b = b.trim();
        if !b.is_empty() && providers.iter().any(|p| p == a) {
            let prof = file.get_profile(a)?;
            return Ok((a.to_string(), if b.is_empty() { prof.model } else { b.to_string() }));
        }
    }

    // Bare provider or provider/model…
    if let Ok((name, model_ov)) = split_provider_model(spec, &providers) {
        let prof = file.get_profile(&name)?;
        return Ok((name, model_ov.unwrap_or(prof.model)));
    }

    // Bare model id — prefer current provider if it lists the model, else the
    // unique provider that does. Free-form overrides only apply when the
    // current provider has an empty catalog *and* no other provider claims
    // the id (so `/model glm-5.2` jumps to zai even from local).
    let _ = current_model;
    let current = file.get_profile(current_provider)?;
    if current.available_models().iter().any(|m| m == spec) {
        return Ok((current_provider.to_string(), spec.to_string()));
    }

    let mut hits: Vec<(String, String)> = Vec::new();
    for name in &providers {
        if let Ok(prof) = file.get_profile(name) {
            if prof.available_models().iter().any(|m| m == spec) {
                hits.push((name.clone(), spec.to_string()));
            }
        }
    }
    match hits.len() {
        1 => Ok(hits.pop().unwrap()),
        0 if current.models.is_empty() => {
            Ok((current_provider.to_string(), spec.to_string()))
        }
        0 => Err(Error::msg(format!(
            "unknown model '{spec}' — /model for the catalog, or /model <provider>/<model>"
        ))),
        _ => {
            let opts = hits
                .iter()
                .map(|(p, m)| format!("{p}/{m}"))
                .collect::<Vec<_>>()
                .join(", ");
            Err(Error::msg(format!(
                "model '{spec}' is on multiple providers ({opts}); use /model <provider>/{spec}"
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_longest_provider_prefix() {
        let providers = vec![
            "or".into(),
            "zai-coding-plan".into(),
            "local".into(),
            "grok".into(),
        ];
        assert_eq!(
            split_provider_model("zai-coding-plan", &providers).unwrap(),
            ("zai-coding-plan".into(), None)
        );
        assert_eq!(
            split_provider_model("zai-coding-plan/glm-5-turbo", &providers).unwrap(),
            ("zai-coding-plan".into(), Some("glm-5-turbo".into()))
        );
        // OpenRouter-style model ids keep their slash.
        assert_eq!(
            split_provider_model("or/qwen/qwen3-coder", &providers).unwrap(),
            ("or".into(), Some("qwen/qwen3-coder".into()))
        );
        assert!(split_provider_model("nope/x", &providers).is_err());
    }

    #[test]
    fn available_models_includes_default() {
        let mut p = Profile::default();
        p.model = "glm-5.2".into();
        p.models = vec!["glm-5-turbo".into(), "glm-4.5-air".into()];
        assert_eq!(
            p.available_models(),
            vec!["glm-5.2".to_string(), "glm-5-turbo".into(), "glm-4.5-air".into()]
        );
    }
}
