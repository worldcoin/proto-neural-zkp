use eyre::{bail, eyre, Result, WrapErr as _};
use std::{
    env::{var, VarError},
    fs,
    path::Path,
    process::Command,
};
use time::{format_description::well_known::Rfc3339, OffsetDateTime, UtcOffset};

fn main() -> Result<()> {
    let commit = rerun_if_git_changes().unwrap_or_else(|e| {
        eprintln!("Warning: {}", e);
        None
    });

    println!(
        "cargo:rustc-env=COMMIT_SHA={}",
        env_or_cmd("COMMIT_SHA", &["git", "rev-parse", "HEAD"]).unwrap_or_else(|e| {
            eprintln!("Warning: {}", e);
            commit.unwrap_or_else(|| "0000000000000000000000000000000000000000".to_string())
        })
    );
    let build_date = OffsetDateTime::now_utc();
    let commit_date = env_or_cmd("COMMIT_DATE", &[
        "git",
        "log",
        "-n1",
        "--pretty=format:'%aI'",
    ])
    .and_then(|str| Ok(OffsetDateTime::parse(str.trim_matches('\''), &Rfc3339)?))
    .unwrap_or_else(|e| {
        eprintln!("Warning: {}", e);
        OffsetDateTime::UNIX_EPOCH
    });
    println!(
        "cargo:rustc-env=COMMIT_DATE={}",
        commit_date.to_offset(UtcOffset::UTC).date()
    );
    println!(
        "cargo:rustc-env=BUILD_DATE={}",
        build_date.to_offset(UtcOffset::UTC).date()
    );
    println!(
        "cargo:rustc-env=TARGET={}",
        var("TARGET").wrap_err("Fetching environment variable TARGET")?
    );

    Ok(())
}

fn env_or_cmd(env: &str, cmd: &[&str]) -> Result<String> {
    // Try env first
    match var(env) {
        Ok(s) => return Ok(s),
        Err(VarError::NotPresent) => (),
        Err(e) => bail!(e),
    };

    // Try command
    let err = || {
        format!(
            "Variable {} is unset and command \"{}\" failed",
            env,
            cmd.join(" ")
        )
    };
    let output = Command::new(cmd[0])
        .args(&cmd[1..])
        .output()
        .with_context(err)?;
    if output.status.success() {
        Ok(String::from_utf8(output.stdout)?.trim().to_string())
    } else {
        bail!(err())
    }
}

fn rerun_if_git_changes() -> Result<Option<String>> {
    let git_head = Path::new(".git/HEAD");

    // Skip if not in a git repo
    if !git_head.exists() {
        eprintln!("No .git/HEAD found, not rerunning on git change");
        return Ok(None);
    }

    // TODO: Worktree support where `.git` is a file
    println!("cargo:rerun-if-changed=.git/HEAD");

    // If HEAD contains a ref, then echo that path also.
    let contents = fs::read_to_string(git_head).wrap_err("Error reading .git/HEAD")?;
    let head_ref = contents.split(": ").collect::<Vec<_>>();
    let commit = if head_ref.len() == 2 && head_ref[0] == "ref" {
        let ref_path = Path::new(".git").join(head_ref[1].trim());
        let ref_path_str = ref_path
            .to_str()
            .ok_or_else(|| eyre!("Could not convert ref path {:?} to string", ref_path))?;
        println!("cargo:rerun-if-changed={}", ref_path_str);
        fs::read_to_string(&ref_path).with_context(|| format!("Error reading {}", ref_path_str))?
    } else {
        contents
    };
    Ok(Some(commit))
}
