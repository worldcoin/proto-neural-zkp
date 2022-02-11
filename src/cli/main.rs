#![warn(clippy::all, clippy::pedantic, clippy::cargo, clippy::nursery)]

use neural_zkp as lib;

mod logging;
mod random;

use eyre::{Result as EyreResult, WrapErr as _};
use structopt::StructOpt;
use tokio::runtime::{self, Runtime};
use tracing::info;

const VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    "\n",
    env!("COMMIT_SHA"),
    " ",
    env!("COMMIT_DATE"),
    "\n",
    env!("TARGET"),
    " ",
    env!("BUILD_DATE"),
    "\n",
    env!("CARGO_PKG_AUTHORS"),
    "\n",
    env!("CARGO_PKG_HOMEPAGE"),
    "\n",
    env!("CARGO_PKG_DESCRIPTION"),
);

#[derive(StructOpt)]
struct Options {
    #[structopt(flatten)]
    log: logging::Options,

    #[structopt(flatten)]
    app: lib::Options,

    #[structopt(flatten)]
    random: random::Options,

    /// Number of compute threads to use (defaults to number of cores)
    #[structopt(long)]
    threads: Option<usize>,
}

fn main() -> EyreResult<()> {
    // Install error handler
    color_eyre::install()?;

    // Parse CLI and handle help and version (which will stop the application).
    let matches = Options::clap().long_version(VERSION).get_matches();
    let options = Options::from_clap(&matches);

    // Start subsystems
    options.log.init()?;
    let rng = options.random.init();
    init_rayon(options.threads)?;
    let runtime = init_tokio()?;

    // Run main
    let main_future = lib::main(rng, options.app);
    runtime.block_on(main_future)?;

    // Terminate successfully
    info!("Program terminating normally");
    Ok(())
}

fn init_rayon(threads: Option<usize>) -> EyreResult<()> {
    if let Some(threads) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .context("Failed to build thread pool.")?;
    }
    info!(
        "Using {} compute threads on {} cores",
        rayon::current_num_threads(),
        num_cpus::get()
    );
    Ok(())
}

fn init_tokio() -> EyreResult<Runtime> {
    runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .wrap_err("Error creating Tokio runtime")
    // TODO: Log num_workers once RuntimeMetrics are stable
}

#[cfg(test)]
pub mod test {
    use super::*;
    use tracing::{error, warn};
    use tracing_test::traced_test;

    #[test]
    #[traced_test]
    fn test_with_log_output() {
        error!("logged on the error level");
        assert!(logs_contain("logged on the error level"));
    }

    #[tokio::test]
    #[traced_test]
    #[allow(clippy::semicolon_if_nothing_returned)] // False positive
    async fn async_test_with_log() {
        // Local log
        info!("This is being logged on the info level");

        // Log from a spawned task (which runs in a separate thread)
        tokio::spawn(async {
            warn!("This is being logged on the warn level from a spawned task");
        })
        .await
        .unwrap();

        // Ensure that `logs_contain` works as intended
        assert!(logs_contain("logged on the info level"));
        assert!(logs_contain("logged on the warn level"));
        assert!(!logs_contain("logged on the error level"));
    }
}
