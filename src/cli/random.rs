use rand::{rngs::OsRng, RngCore, SeedableRng};
use rand_pcg::Mcg128Xsl64;
use structopt::StructOpt;
use tracing::info;

pub type Generator = Mcg128Xsl64;

#[derive(StructOpt)]
pub struct Options {
    /// Random seed for deterministic random number generation.
    /// If not specified a seed is periodically generated from OS entropy.
    #[structopt(long, parse(try_from_str = parse_hex_u64))]
    seed: Option<u64>,
}

impl Options {
    #[must_use]
    pub fn init(self) -> Generator {
        let rng_seed = self.seed.unwrap_or_else(random_seed);
        info!("Using random seed {rng_seed:16x}");
        Mcg128Xsl64::seed_from_u64(rng_seed)
    }
}

#[must_use]
fn random_seed() -> u64 {
    OsRng::default().next_u64()
}

#[must_use]
fn parse_hex_u64(src: &str) -> Result<u64, std::num::ParseIntError> {
    u64::from_str_radix(src, 16)
}
