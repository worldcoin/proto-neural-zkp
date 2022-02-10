use eyre::Result as EyreResult;
use tracing::info;

#[cfg(unix)]
use tokio::signal::unix::{signal, SignalKind};

#[cfg(not(unix))]
use tokio::signal::ctrl_c;

#[cfg(unix)]
#[allow(clippy::module_name_repetitions)]
pub async fn signal_shutdown() -> EyreResult<()> {
    let sigint = signal(SignalKind::interrupt())?;
    let sigterm = signal(SignalKind::terminate())?;
    tokio::pin!(sigint);
    tokio::pin!(sigterm);
    tokio::select! {
        _ = sigint.recv() => { info!("SIGINT received, shutting down"); }
        _ = sigterm.recv() => { info!("SIGTERM received, shutting down"); }
    };
    Ok(())
}

#[cfg(not(unix))]
#[allow(clippy::module_name_repetitions)]
pub async fn signal_shutdown() -> EyreResult<()> {
    ctrl_c().await?;
    info!("Ctrl-C received");
    Ok(())
}
