use core::sync::atomic::{AtomicBool, Ordering};
use once_cell::sync::Lazy;
use prometheus::{
    exponential_buckets, register_histogram, register_int_counter, Histogram, IntCounter,
};
use std::alloc::{GlobalAlloc, Layout};

pub use std::alloc::System as StdAlloc;

#[cfg(feature = "mimalloc")]
pub use mimalloc::MiMalloc;

static ALLOCATED: Lazy<IntCounter> =
    Lazy::new(|| register_int_counter!("mem_alloc", "Cumulative memory allocated.").unwrap());
static FREED: Lazy<IntCounter> =
    Lazy::new(|| register_int_counter!("mem_free", "Cumulative memory freed.").unwrap());
static SIZE: Lazy<Histogram> = Lazy::new(|| {
    register_histogram!(
        "mem_alloc_size",
        "Distribution of allocation sizes.",
        exponential_buckets(16.0, 4.0, 10).unwrap()
    )
    .unwrap()
});

#[cfg_attr(feature = "std", derive(Debug))]
pub struct Allocator<T: GlobalAlloc> {
    inner:    T,
    metering: AtomicBool,
}

#[cfg(not(feature = "mimalloc"))]
// TODO: Turn this into a generic constructor taking an `inner: T` once
// #![feature(const_fn_trait_bound)] is stable.
pub const fn new_std() -> Allocator<StdAlloc> {
    Allocator {
        inner:    StdAlloc,
        metering: AtomicBool::new(false),
    }
}

#[cfg(feature = "mimalloc")]
pub const fn new_mimalloc() -> Allocator<MiMalloc> {
    Allocator {
        inner:    MiMalloc,
        metering: AtomicBool::new(false),
    }
}

impl<T: GlobalAlloc> Allocator<T> {
    pub fn start_metering(&self) {
        if self.metering.load(Ordering::Acquire) {
            return;
        }
        Lazy::force(&ALLOCATED);
        Lazy::force(&SIZE);
        Lazy::force(&FREED);
        self.metering.store(true, Ordering::Release);
    }

    fn count_alloc(&self, size: usize) {
        // Avoid re-entrancy here when metrics are first initialized.
        if self.metering.load(Ordering::Acquire) {
            ALLOCATED.inc_by(size as u64);
            #[allow(clippy::cast_precision_loss)]
            SIZE.observe(size as f64);
        }
    }

    fn count_dealloc(&self, size: usize) {
        if self.metering.load(Ordering::Acquire) {
            FREED.inc_by(size as u64);
        }
    }
}

// GlobalAlloc is an unsafe trait for allocators
#[allow(unsafe_code)]
unsafe impl<T: GlobalAlloc> GlobalAlloc for Allocator<T> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.count_alloc(layout.size());
        self.inner.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.count_dealloc(layout.size());
        self.inner.dealloc(ptr, layout);
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        self.count_alloc(layout.size());
        self.inner.alloc_zeroed(layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let old_size = layout.size();
        if new_size >= old_size {
            self.count_alloc(new_size - old_size);
        } else {
            self.count_dealloc(old_size - new_size);
        }
        self.inner.realloc(ptr, layout, new_size)
    }
}
