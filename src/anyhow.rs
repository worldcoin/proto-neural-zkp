use eyre::eyre;

pub trait MapAny<T> {
    fn map_any(self) -> eyre::Result<T>;
}

impl<T> MapAny<T> for anyhow::Result<T> {
    fn map_any(self) -> eyre::Result<T> {
        self.map_err(|e| eyre!(e.to_string()))
    }
}
