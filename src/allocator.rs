use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};
pub use std::alloc::System as StdAlloc;
use std::alloc::{GlobalAlloc, Layout};

#[cfg(feature = "mimalloc")]
pub use mimalloc::MiMalloc;

#[cfg_attr(feature = "std", derive(Debug))]
pub struct Allocator<T: GlobalAlloc> {
    inner:                 T,
    pub allocated:         AtomicUsize,
    pub peak_allocated:    AtomicUsize,
    pub total_allocated:   AtomicUsize,
    pub largest_allocated: AtomicUsize,
    pub num_allocations:   AtomicUsize,
}

#[cfg(not(feature = "mimalloc"))]
// TODO: Turn this into a generic constructor taking an `inner: T` once
// #![feature(const_fn_trait_bound)] is stable.
pub const fn new_std() -> Allocator<StdAlloc> {
    Allocator::new(StdAlloc)
}

#[cfg(feature = "mimalloc")]
pub const fn new_mimalloc() -> Allocator<MiMalloc> {
    Allocator::new(MiMalloc)
}

impl<T: GlobalAlloc> Allocator<T> {
    pub const fn new(alloc: T) -> Self {
        Self {
            inner:             alloc,
            allocated:         AtomicUsize::new(0),
            peak_allocated:    AtomicUsize::new(0),
            total_allocated:   AtomicUsize::new(0),
            largest_allocated: AtomicUsize::new(0),
            num_allocations:   AtomicUsize::new(0),
        }
    }

    fn count_alloc(&self, size: usize) {
        // TODO: We are doing a lot of atomic operations here, what is
        // the performance impact?
        let allocated = self.allocated.fetch_add(size, Relaxed);
        self.total_allocated.fetch_add(size, Relaxed);
        self.num_allocations.fetch_add(1, Relaxed);
        // HACK: Using `allocated` here again is not completely fool proof
        self.peak_allocated.fetch_max(allocated, Relaxed);
        self.largest_allocated.fetch_max(size, Relaxed);
    }

    fn count_dealloc(&self, size: usize) {
        self.allocated.fetch_sub(size, Relaxed);
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
