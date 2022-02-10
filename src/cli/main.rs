#![warn(clippy::all, clippy::pedantic, clippy::cargo, clippy::nursery)]

use neural_zkp as lib;

mod allocator;
mod logging;
mod prometheus;
mod shutdown;
mod tokio_console;

use self::allocator::Allocator;
use eyre::{Result as EyreResult, WrapErr as _};
use structopt::StructOpt;
use tokio::{runtime, sync::broadcast};
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

#[cfg(not(feature = "mimalloc"))]
#[global_allocator]
pub static ALLOCATOR: Allocator<allocator::StdAlloc> = allocator::new_std();

#[cfg(feature = "mimalloc")]
#[global_allocator]
pub static ALLOCATOR: Allocator<allocator::MiMalloc> = allocator::new_mimalloc();

#[derive(StructOpt)]
struct Options {
    #[structopt(flatten)]
    log:            logging::Options,
    #[structopt(flatten)]
    pub prometheus: prometheus::Options,
    #[structopt(flatten)]
    app:            lib::Options,
}

fn main() -> EyreResult<()> {
    // Install error handler
    color_eyre::install()?;

    // Parse CLI and handle help and version (which will stop the application).
    let matches = Options::clap().long_version(VERSION).get_matches();
    let options = Options::from_clap(&matches);

    // Meter memory consumption
    ALLOCATOR.start_metering();

    // Start log system
    options.log.init()?;

    // Launch Tokio runtime
    runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .wrap_err("Error creating Tokio runtime")?
        .block_on(async {
            // Create shutdown signal
            // TODO: Fix minor race conditions
            let (shutdown, _) = broadcast::channel(1);
            tokio::spawn({
                let shutdown = shutdown.clone();
                async move {
                    shutdown::signal_shutdown().await.unwrap();
                    let _ = shutdown.send(());
                }
            });

            // Start prometheus
            let prometheus = tokio::spawn(prometheus::main(options.prometheus, shutdown.clone()));

            // Start main
            lib::main(options.app, shutdown.clone()).await?;

            // Send (potentially redundant) shutdown in case `lib::main` returned by itself
            let _ = shutdown.send(());

            // Stop prometheus
            info!("Stopping metrics server");
            prometheus.await?
        })?;

    // Terminate successfully
    info!("Program terminating normally");
    Ok(())
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
