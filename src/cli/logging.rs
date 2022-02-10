#![warn(clippy::all, clippy::pedantic, clippy::cargo, clippy::nursery)]

use core::str::FromStr;
use eyre::{bail, Error as EyreError, Result as EyreResult, WrapErr as _};
use std::process::id as pid;
use structopt::StructOpt;
use tracing::{info, Level, Subscriber};
use tracing_log::LogTracer;
use tracing_subscriber::{filter::Targets, fmt, layer::SubscriberExt, Layer, Registry};

#[derive(Debug, PartialEq)]
enum LogFormat {
    Compact,
    Pretty,
    Json,
}

impl LogFormat {
    fn to_layer<S>(&self) -> impl Layer<S>
    where
        S: Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a> + Send + Sync,
    {
        match self {
            LogFormat::Compact => Box::new(fmt::Layer::new().event_format(fmt::format().compact()))
                as Box<dyn Layer<S> + Send + Sync>,
            LogFormat::Pretty => Box::new(fmt::Layer::new().event_format(fmt::format().pretty())),
            LogFormat::Json => Box::new(fmt::Layer::new().event_format(fmt::format().json())),
        }
    }
}

impl FromStr for LogFormat {
    type Err = EyreError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "compact" => Self::Compact,
            "pretty" => Self::Pretty,
            "json" => Self::Json,
            _ => bail!("Invalid log format: {}", s),
        })
    }
}

#[derive(Debug, PartialEq, StructOpt)]
pub struct Options {
    /// Verbose mode (-v, -vv, -vvv, etc.)
    #[structopt(short, long, parse(from_occurrences))]
    verbose: usize,

    /// Apply an env_filter compatible log filter
    #[structopt(long, env, default_value)]
    log_filter: String,

    /// Log format, one of 'compact', 'pretty' or 'json'
    #[structopt(long, env, default_value = "compact")]
    log_format: LogFormat,
}

impl Options {
    #[allow(dead_code)]
    pub fn init(&self) -> EyreResult<()> {
        // Log filtering is a combination of `--log-filter` and `--verbose` arguments.
        let verbosity = {
            let (all, app) = match self.verbose {
                0 => (Level::INFO, Level::INFO),
                1 => (Level::INFO, Level::DEBUG),
                2 => (Level::INFO, Level::TRACE),
                3 => (Level::DEBUG, Level::TRACE),
                _ => (Level::TRACE, Level::TRACE),
            };
            Targets::new()
                .with_default(all)
                .with_target("lib", app)
                .with_target(env!("CARGO_CRATE_NAME"), app)
        };
        let log_filter = if self.log_filter.is_empty() {
            Targets::new()
        } else {
            self.log_filter
                .parse()
                .wrap_err("Error parsing log-filter")?
        };
        let targets = log_filter.with_targets(verbosity);

        // Route events to stdout
        let subscriber = Registry::default().with(self.log_format.to_layer().with_filter(targets));
        tracing::subscriber::set_global_default(subscriber)?;

        // Enable `log` crate compatibility
        LogTracer::init()?;

        // Log version information
        info!(
            host = env!("TARGET"),
            pid = pid(),
            main = &crate::main as *const _ as usize,
            commit = &env!("COMMIT_SHA")[..8],
            "{name} {version}",
            name = env!("CARGO_CRATE_NAME"),
            version = env!("CARGO_PKG_VERSION"),
        );

        Ok(())
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_parse_args() {
        let cmd = "arg0 -v --log-filter foo -vvv";
        let options = Options::from_iter_safe(cmd.split(' ')).unwrap();
        assert_eq!(options, Options {
            verbose:    4,
            log_filter: "foo".to_owned(),
            log_format: LogFormat::Pretty,
        });
    }
}
