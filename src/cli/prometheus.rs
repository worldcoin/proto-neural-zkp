use ::prometheus::{
    opts, register_counter, register_gauge, register_histogram, Counter, Encoder as _, Gauge,
    Histogram,
};
use eyre::{bail, ensure, Result as EyreResult, WrapErr as _};
use hyper::{
    body::HttpBody,
    header::CONTENT_TYPE,
    service::{make_service_fn, service_fn},
    Body, Method, Request, Response, Server,
};
use once_cell::sync::Lazy;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use structopt::StructOpt;
use tokio::sync::broadcast;
use tracing::{error, info, trace};
use url::{Host, Url};

#[derive(Debug, PartialEq, StructOpt)]
pub struct Options {
    /// Prometheus scrape endpoint
    // See <https://github.com/prometheus/prometheus/wiki/Default-port-allocations>
    #[structopt(long, env, default_value = "http://127.0.0.1:9998/metrics")]
    pub prometheus: Url,
}

static REQ_COUNTER: Lazy<Counter> = Lazy::new(|| {
    register_counter!(opts!(
        "prometheus_requests_total",
        "Number of Prometheus scrape requests made."
    ))
    .unwrap()
});
static REQ_BODY_GAUGE: Lazy<Gauge> = Lazy::new(|| {
    register_gauge!(opts!(
        "prometheus_response_size_bytes",
        "The Prometheus response sizes in bytes."
    ))
    .unwrap()
});
static REQ_HISTOGRAM: Lazy<Histogram> = Lazy::new(|| {
    register_histogram!(
        "prometheus_request_duration_seconds",
        "The Prometheus request latencies in seconds."
    )
    .unwrap()
});

#[allow(clippy::unnecessary_wraps)]
#[allow(clippy::needless_pass_by_value)]
#[allow(clippy::unused_async)]
async fn serve_req(_req: Request<Body>) -> Result<Response<Body>, hyper::Error> {
    // let encoder = prometheus::ProtobufEncoder;
    let encoder = prometheus::TextEncoder;
    let metric_families = prometheus::gather();
    let mut buffer = vec![];
    let result = encoder.encode(&metric_families, &mut buffer);
    if let Err(e) = result {
        error!("Internal server error: {}", e);
        let response = Response::builder()
            .status(500)
            .body(Body::from(e.to_string()))
            .unwrap(); // TODO
        return Ok(response);
    }

    let response = Response::builder()
        .status(200)
        .header(CONTENT_TYPE, encoder.format_type())
        .body(Body::from(buffer))
        .unwrap(); // TODO

    Ok(response)
}

#[allow(clippy::unused_async)] // We are implementing an interface
async fn route(req: Request<Body>) -> Result<Response<Body>, hyper::Error> {
    trace!("Receiving request at path {}", req.uri());
    REQ_COUNTER.inc();
    let timer = REQ_HISTOGRAM.start_timer();

    let response = match (req.method(), req.uri().path()) {
        (&Method::GET, "/metrics") => serve_req(req).await?,
        _ => Response::builder()
            .status(404)
            .body(Body::from("404"))
            .unwrap(),
    };

    #[allow(clippy::cast_precision_loss)]
    REQ_BODY_GAUGE.set(response.body().size_hint().lower() as f64);
    timer.observe_duration();
    Ok(response)
}

pub async fn main(options: Options, shutdown: broadcast::Sender<()>) -> EyreResult<()> {
    ensure!(
        options.prometheus.scheme() == "http",
        "Only http:// is supported in {}",
        options.prometheus
    );
    ensure!(
        options.prometheus.path() == "/metrics",
        "Only /metrics is supported in {}",
        options.prometheus
    );
    let ip: IpAddr = match options.prometheus.host() {
        Some(Host::Ipv4(ip)) => ip.into(),
        Some(Host::Ipv6(ip)) => ip.into(),
        Some(_) => bail!("Cannot bind {}", options.prometheus),
        None => Ipv4Addr::LOCALHOST.into(),
    };
    let port = options.prometheus.port().unwrap_or(9998);
    let addr = SocketAddr::new(ip, port);

    let server = Server::try_bind(&addr)
        .wrap_err("Could not bind Prometheus server port")?
        .serve(make_service_fn(|_| async {
            Ok::<_, hyper::Error>(service_fn(route))
        }))
        .with_graceful_shutdown(async move {
            shutdown.subscribe().recv().await.ok();
        });
    info!(url = %options.prometheus, "Metrics server listening");

    server.await?;
    Ok(())
}
