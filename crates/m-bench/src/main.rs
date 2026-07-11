//! m-bench — SWE-bench Lite runner for the m agent.
//!
//! Flow per instance: start the official per-instance docker image with
//! host networking, copy in the static `m` binary, run it headless against
//! the local llama-server, collect `git diff` as the prediction, and write
//! the official predictions.jsonl for scoring with the swebench harness.
//!
//!   m-bench fetch                      # cache dataset rows from HF
//!   m-bench pick -n 30                 # print a stratified instance list
//!   m-bench run --instances FILE --out bench/runs/NAME
//!   m-bench report --run bench/runs/NAME [--eval REPORT.json]

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use m_core::http::{Url, get_json};
use m_core::tools::run_bash;

const DATASET: &str = "SWE-bench/SWE-bench_Lite";
const MODEL_NAME: &str = "m-gemma4-12b-mtp";
const DEFAULT_MAX_TURNS: u32 = 40;
const DEFAULT_TIMEOUT_S: u64 = 1500;

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Instance {
    instance_id: String,
    repo: String,
    base_commit: String,
    problem_statement: String,
    #[serde(default)]
    version: String,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(String::as_str) {
        Some("fetch") => cmd_fetch(&args[1..]),
        Some("pick") => cmd_pick(&args[1..]),
        Some("run") => cmd_run(&args[1..]),
        Some("report") => cmd_report(&args[1..]),
        _ => {
            eprintln!(
                "usage: m-bench <fetch|pick|run|report> [options]\n\
                 \n\
                 fetch  [--out bench/dataset.json]\n\
                 pick   [-n 30] [--dataset bench/dataset.json]\n\
                 run    --instances FILE --out DIR [--dataset F] [--bin PATH]\n\
                 \x20      [--max-turns N] [--timeout SECONDS] [--keep]\n\
                 report --run DIR [--eval swebench-report.json]"
            );
            2
        }
    };
    std::process::exit(code);
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

// ------------------------------------------------------------------ fetch

fn cmd_fetch(args: &[String]) -> i32 {
    let out = flag(args, "--out").unwrap_or_else(|| "bench/dataset.json".into());
    let cancel = Arc::new(AtomicBool::new(false));
    let mut rows: Vec<Instance> = Vec::new();
    let mut offset = 0usize;
    loop {
        let url = format!(
            "https://datasets-server.huggingface.co/rows?dataset={}&config=default&split=test&offset={offset}&length=100",
            DATASET.replace('/', "%2F")
        );
        eprintln!("fetching rows {offset}..{}", offset + 100);
        let url = match Url::parse(&url) {
            Ok(u) => u,
            Err(e) => {
                eprintln!("m-bench: {e}");
                return 1;
            }
        };
        let body = match get_json(&url, &[], cancel.clone()) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("m-bench: fetch: {e}");
                return 1;
            }
        };
        let v: Value = match serde_json::from_str(&body) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("m-bench: bad JSON at offset {offset}: {e}");
                return 1;
            }
        };
        let total = v.get("num_rows_total").and_then(Value::as_u64).unwrap_or(0) as usize;
        let page = v.get("rows").and_then(Value::as_array).cloned().unwrap_or_default();
        for r in &page {
            if let Some(row) = r.get("row")
                && let Ok(inst) = serde_json::from_value::<Instance>(row.clone())
            {
                rows.push(inst);
            }
        }
        offset += page.len();
        if page.is_empty() || offset >= total {
            break;
        }
    }
    if let Some(dir) = Path::new(&out).parent() {
        std::fs::create_dir_all(dir).ok();
    }
    std::fs::write(&out, serde_json::to_string(&rows).unwrap()).expect("write dataset");
    eprintln!("wrote {} instances to {out}", rows.len());
    if rows.is_empty() { 1 } else { 0 }
}

fn load_dataset(path: &str) -> Result<Vec<Instance>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("{path}: {e} (run `m-bench fetch` first)"))?;
    serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))
}

// ------------------------------------------------------------------ pick

/// Deterministic stratified slice: sort all instance ids, take every
/// (300/n)-th. Reproducible by anyone from the public dataset.
fn cmd_pick(args: &[String]) -> i32 {
    let n: usize = flag(args, "-n").and_then(|s| s.parse().ok()).unwrap_or(30);
    let dataset = flag(args, "--dataset").unwrap_or_else(|| "bench/dataset.json".into());
    let rows = match load_dataset(&dataset) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("m-bench: {e}");
            return 1;
        }
    };
    let mut ids: Vec<&str> = rows.iter().map(|r| r.instance_id.as_str()).collect();
    ids.sort();
    let step = (ids.len().max(1)) as f64 / n as f64;
    let mut picked = Vec::new();
    let mut x = 0.0f64;
    while picked.len() < n && (x as usize) < ids.len() {
        picked.push(ids[x as usize]);
        x += step;
    }
    for id in picked {
        println!("{id}");
    }
    0
}

// ------------------------------------------------------------------ run

struct RunCfg {
    dataset: String,
    instances: String,
    out: PathBuf,
    bin: PathBuf,
    max_turns: u32,
    timeout: u64,
    keep: bool,
}

fn cmd_run(args: &[String]) -> i32 {
    let cfg = RunCfg {
        dataset: flag(args, "--dataset").unwrap_or_else(|| "bench/dataset.json".into()),
        instances: flag(args, "--instances").unwrap_or_else(|| "bench/instances.txt".into()),
        out: PathBuf::from(flag(args, "--out").unwrap_or_else(|| "bench/runs/run".into())),
        bin: PathBuf::from(flag(args, "--bin").unwrap_or_else(|| {
            "target/x86_64-unknown-linux-musl/release/m".into()
        })),
        max_turns: flag(args, "--max-turns").and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_MAX_TURNS),
        timeout: flag(args, "--timeout").and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_TIMEOUT_S),
        keep: has_flag(args, "--keep"),
    };
    if !cfg.bin.exists() {
        eprintln!(
            "m-bench: agent binary not found: {} \
             (build with: cargo build -p m-tui --release --target x86_64-unknown-linux-musl --no-default-features)",
            cfg.bin.display()
        );
        return 2;
    }
    let rows = match load_dataset(&cfg.dataset) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("m-bench: {e}");
            return 1;
        }
    };
    let want = match std::fs::read_to_string(&cfg.instances) {
        Ok(s) => s.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty() && !l.starts_with('#')).collect::<Vec<_>>(),
        Err(e) => {
            eprintln!("m-bench: {}: {e}", cfg.instances);
            return 1;
        }
    };
    std::fs::create_dir_all(&cfg.out).expect("create out dir");

    // Refuse to run when the local server is down.
    let cancel = Arc::new(AtomicBool::new(false));
    let health = Url::parse("http://localhost:8080/health").unwrap();
    if get_json(&health, &[], cancel).is_err() {
        eprintln!("m-bench: llama-server not reachable at localhost:8080 (start it with ./run.sh)");
        return 2;
    }

    let total = want.len();
    let mut n_patch = 0usize;
    for (i, id) in want.iter().enumerate() {
        let Some(inst) = rows.iter().find(|r| &r.instance_id == id) else {
            eprintln!("[{}/{total}] {id}: not in dataset, skipping", i + 1);
            continue;
        };
        eprintln!("[{}/{total}] {id}", i + 1);
        let started = Instant::now();
        let meta = run_instance(&cfg, inst);
        let secs = started.elapsed().as_secs();
        match &meta {
            Ok(m) => {
                let patched = m.get("patch_bytes").and_then(Value::as_u64).unwrap_or(0) > 0;
                if patched {
                    n_patch += 1;
                }
                eprintln!(
                    "    {}s · turns {} · patch {} bytes",
                    secs,
                    m.get("turns").and_then(Value::as_u64).unwrap_or(0),
                    m.get("patch_bytes").and_then(Value::as_u64).unwrap_or(0),
                );
            }
            Err(e) => eprintln!("    FAILED after {secs}s: {e}"),
        }
        let meta = meta.unwrap_or_else(|e| json!({ "instance_id": id, "error": e }));
        let mp = cfg.out.join(format!("{id}.meta.json"));
        std::fs::write(mp, serde_json::to_string_pretty(&meta).unwrap()).ok();
    }
    eprintln!(
        "done: {n_patch}/{total} instances produced a patch → {}",
        cfg.out.join("predictions.jsonl").display()
    );
    0
}

fn sh(cmd: &str, timeout: Duration) -> Result<String, String> {
    let cancel = Arc::new(AtomicBool::new(false));
    let cwd = std::env::current_dir().unwrap();
    match run_bash(cmd, &cwd, timeout, &cancel) {
        Ok(out) if !out.is_error => Ok(out.content),
        Ok(out) => Err(out.content),
        Err(e) => Err(e.to_string()),
    }
}

fn image_of(instance_id: &str) -> String {
    // Docker Hub forbids "__"; the official images encode it as "_1776_".
    format!("swebench/sweb.eval.x86_64.{}:latest", instance_id.replace("__", "_1776_"))
}

fn prompt_of(inst: &Instance) -> String {
    format!(
        "Fix the following GitHub issue in the repository checked out at /testbed \
         ({repo}, a git repo with its environment already installed).\n\
         \n\
         <issue>\n{issue}\n</issue>\n\
         \n\
         Approach:\n\
         1. Explore the relevant code (grep/find/read) and write a small script to reproduce the issue.\n\
         2. Fix the root cause in the library source with minimal, surgical edits.\n\
         3. Rerun your reproduction script to verify, and consider edge cases.\n\
         \n\
         Rules:\n\
         - Do NOT modify any test files or add tests; fix only the library code.\n\
         - `python` already has the project installed (editable): reuse it.\n\
         - Every bash command must be one concrete command — never a comment or a plan. \
         Put reproduction code in a file with the write tool instead of long inline heredocs.\n\
         - If an approach fails twice, stop repeating it: re-read the relevant source and change tactics.\n\
         - When the fix is verified, reply with a one-paragraph summary and stop.",
        repo = inst.repo,
        issue = inst.problem_statement.trim(),
    )
}

fn run_instance(cfg: &RunCfg, inst: &Instance) -> Result<Value, String> {
    let id = &inst.instance_id;
    let image = image_of(id);
    let cname = format!("m-bench-{}", id.replace("__", "-"));

    // Pull if missing (long timeout: multi-GB images).
    sh(&format!("docker image inspect {image} >/dev/null 2>&1 || docker pull -q {image}"),
        Duration::from_secs(1800))
        .map_err(|e| format!("pull {image}: {e}"))?;

    let _ = sh(&format!("docker rm -f {cname} 2>/dev/null"), Duration::from_secs(60));
    sh(&format!("docker run -d --name {cname} --network host {image} tail -f /dev/null"),
        Duration::from_secs(120))
        .map_err(|e| format!("start container: {e}"))?;

    let result = (|| -> Result<Value, String> {
        sh(&format!("docker cp {} {cname}:/usr/local/bin/m", cfg.bin.display()),
            Duration::from_secs(60))
            .map_err(|e| format!("copy agent: {e}"))?;

        // Prompt via file to avoid any quoting pitfalls.
        let prompt_path = cfg.out.join(format!("{id}.prompt.txt"));
        std::fs::write(&prompt_path, prompt_of(inst)).map_err(|e| e.to_string())?;
        sh(&format!("docker cp {} {cname}:/tmp/m-prompt.txt", prompt_path.display()),
            Duration::from_secs(60))
            .map_err(|e| format!("copy prompt: {e}"))?;

        // Conda env python first on PATH, as the swebench images lay it out.
        // The trajectory goes to a file in the container (docker cp'd out
        // afterwards) so it is never clipped by output limits.
        let exec = format!(
            "docker exec -e PATH=/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
             -e HOME=/root -w /testbed {cname} \
             sh -c 'm -p --json --max-turns {mt} --max-tokens 4096 --temp 0 \"$(cat /tmp/m-prompt.txt)\" >/tmp/m-traj.jsonl 2>&1; echo M_EXIT:$?'",
            mt = cfg.max_turns
        );
        let started = Instant::now();
        let exec_out = match sh(&exec, Duration::from_secs(cfg.timeout)) {
            Ok(out) => out,
            Err(out) => out,
        };
        let secs = started.elapsed().as_secs();
        let traj_path = cfg.out.join(format!("{id}.trajectory.jsonl"));
        let _ = sh(
            &format!("docker cp {cname}:/tmp/m-traj.jsonl {}", traj_path.display()),
            Duration::from_secs(60),
        );
        let trajectory = std::fs::read_to_string(&traj_path).unwrap_or_default();

        let exit: i64 = exec_out
            .lines()
            .rev()
            .find_map(|l| l.strip_prefix("M_EXIT:").and_then(|c| c.parse().ok()))
            .unwrap_or(-1);
        let mut turns = 0u64;
        let mut completion_tokens = 0u64;
        let mut tok_per_sec = Vec::new();
        for line in trajectory.lines() {
            let Ok(v) = serde_json::from_str::<Value>(line) else { continue };
            if let Some("telemetry") = v.get("type").and_then(Value::as_str) {
                turns += 1;
                completion_tokens +=
                    v.get("completion_tokens").and_then(Value::as_u64).unwrap_or(0);
                if let Some(t) = v.get("tok_per_sec").and_then(Value::as_f64)
                    && t > 0.0
                {
                    tok_per_sec.push(t);
                }
            }
        }

        let patch = sh(
            &format!("docker exec -w /testbed {cname} git diff"),
            Duration::from_secs(120),
        )
        .unwrap_or_default();
        let patch = if patch.starts_with("(no output") { String::new() } else { patch };

        // predictions.jsonl (official schema), appended atomically per instance.
        let pred = json!({
            "instance_id": id,
            "model_name_or_path": MODEL_NAME,
            "model_patch": if patch.is_empty() { Value::Null } else { json!(format!("{patch}\n")) },
        });
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(cfg.out.join("predictions.jsonl"))
            .map_err(|e| e.to_string())?;
        writeln!(f, "{pred}").map_err(|e| e.to_string())?;

        let mean_tps = if tok_per_sec.is_empty() {
            0.0
        } else {
            tok_per_sec.iter().sum::<f64>() / tok_per_sec.len() as f64
        };
        Ok(json!({
            "instance_id": id,
            "seconds": secs,
            "turns": turns,
            "completion_tokens": completion_tokens,
            "mean_tok_per_sec": (mean_tps * 10.0).round() / 10.0,
            "agent_exit": exit,
            "patch_bytes": patch.len(),
        }))
    })();

    if !cfg.keep {
        let _ = sh(&format!("docker rm -f {cname}"), Duration::from_secs(60));
    }
    result
}

// ------------------------------------------------------------------ report

fn cmd_report(args: &[String]) -> i32 {
    let run_dir = PathBuf::from(flag(args, "--run").unwrap_or_else(|| "bench/runs/run".into()));
    let eval: Option<Value> = flag(args, "--eval")
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str(&s).ok());

    let resolved: Vec<String> = eval
        .as_ref()
        .and_then(|v| v.get("resolved_ids"))
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        .unwrap_or_default();

    let mut metas: Vec<Value> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&run_dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.file_name().is_some_and(|n| n.to_string_lossy().ends_with(".meta.json"))
                && let Ok(s) = std::fs::read_to_string(&p)
                && let Ok(v) = serde_json::from_str::<Value>(&s)
            {
                metas.push(v);
            }
        }
    }
    metas.sort_by_key(|m| m.get("instance_id").and_then(Value::as_str).unwrap_or("").to_string());
    if metas.is_empty() {
        eprintln!("m-bench: no *.meta.json in {}", run_dir.display());
        return 1;
    }

    let n = metas.len();
    let n_patch = metas
        .iter()
        .filter(|m| m.get("patch_bytes").and_then(Value::as_u64).unwrap_or(0) > 0)
        .count();
    let n_res = resolved.len();
    let tot_secs: u64 = metas.iter().filter_map(|m| m.get("seconds").and_then(Value::as_u64)).sum();
    let tot_turns: u64 = metas.iter().filter_map(|m| m.get("turns").and_then(Value::as_u64)).sum();
    let mean_tps: f64 = {
        let v: Vec<f64> = metas
            .iter()
            .filter_map(|m| m.get("mean_tok_per_sec").and_then(Value::as_f64))
            .filter(|t| *t > 0.0)
            .collect();
        if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
    };

    let mut md = String::new();
    md.push_str("# m — SWE-bench Lite results\n\n");
    md.push_str(&format!(
        "Model: **Gemma 4 12B (Q5_K_XL) + MTP drafter** via llama.cpp on an RTX 4070 Ti SUPER, \
         agent: **m** (headless `-p` mode, temp 0, max {DEFAULT_MAX_TURNS} turns).\n\n"
    ));
    md.push_str("| metric | value |\n|---|---|\n");
    if eval.is_some() {
        md.push_str(&format!("| **resolved** | **{n_res}/{n}** ({:.1}%) |\n", n_res as f64 * 100.0 / n as f64));
    }
    md.push_str(&format!("| patch generated | {n_patch}/{n} |\n"));
    md.push_str(&format!("| total wall time | {}h{:02}m |\n", tot_secs / 3600, (tot_secs % 3600) / 60));
    md.push_str(&format!("| mean turns | {:.1} |\n", tot_turns as f64 / n as f64));
    md.push_str(&format!("| mean generation speed | {mean_tps:.0} tok/s |\n\n"));
    md.push_str("| instance | outcome | turns | time | patch |\n|---|---|---|---|---|\n");
    for m in &metas {
        let id = m.get("instance_id").and_then(Value::as_str).unwrap_or("?");
        let patch = m.get("patch_bytes").and_then(Value::as_u64).unwrap_or(0);
        let outcome = if resolved.iter().any(|r| r == id) {
            "✅ resolved"
        } else if m.get("error").is_some() {
            "💥 error"
        } else if patch > 0 {
            if eval.is_some() { "❌ not resolved" } else { "patch" }
        } else {
            "— no patch"
        };
        md.push_str(&format!(
            "| {id} | {outcome} | {} | {}m{:02}s | {} B |\n",
            m.get("turns").and_then(Value::as_u64).unwrap_or(0),
            m.get("seconds").and_then(Value::as_u64).unwrap_or(0) / 60,
            m.get("seconds").and_then(Value::as_u64).unwrap_or(0) % 60,
            patch,
        ));
    }
    md.push_str(&format!(
        "\nScoring: official harness — `python -m swebench.harness.run_evaluation \
         --dataset_name {DATASET} --predictions_path {}/predictions.jsonl \
         --run_id m-bench --max_workers 4`\n",
        run_dir.display()
    ));

    let out = run_dir.join("RESULTS.md");
    std::fs::write(&out, &md).expect("write RESULTS.md");
    println!("{md}");
    eprintln!("wrote {}", out.display());
    0
}
