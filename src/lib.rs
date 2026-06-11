use std::{ffi::c_int, os::raw::c_void, sync::OnceLock};

use pyo3::prelude::*;

#[allow(non_camel_case_types)]
pub type openblas_dojob_callback =
    Option<extern "C" fn(thread_num: c_int, jobdata: *mut c_void, dojob_data: c_int)>;

#[allow(non_camel_case_types)]
pub type openblas_threads_callback = Option<
    extern "C" fn(
        sync: c_int,
        dojob: openblas_dojob_callback,
        numjobs: c_int,
        jobdata_elsize: usize,
        jobdata: *mut c_void,
        dojob_data: c_int,
    ),
>;

/// API to get number of threads:
#[allow(non_upper_case_globals)]
static openblas_get_num_threads: OnceLock<unsafe extern "C" fn() -> c_int> = OnceLock::new();

fn install(
    get_num_threads: unsafe extern "C" fn() -> c_int,
    install_threads_callback: unsafe extern "C" fn(callback: openblas_threads_callback),
) -> Result<(), Box<dyn std::error::Error>> {
    openblas_get_num_threads.set(get_num_threads).unwrap();
    unsafe { install_threads_callback(Some(run_in_threads_callback)) };
    Ok(())
}

extern "C" fn run_in_threads_callback(
    sync: c_int,
    dojob: openblas_dojob_callback,
    numjobs: c_int,
    jobdata_elsize: usize,
    jobdata: *mut c_void,
    dojob_data: c_int,
) {
    println!("RUN IN THREAD!");
    let numjobs = numjobs as isize;
    let jobdata_elsize = jobdata_elsize as isize;
    // TODO no thread pool yet
    for i in 0..numjobs {
        let element_addr = unsafe { jobdata.byte_offset(i * jobdata_elsize) };
        dojob.unwrap()(i as c_int, element_addr, dojob_data);
    }
}

/// Switch out OpenBLAS pthreads global thread pool module with an on-demand
/// thread pool, per thread. This matches the behavior of OpenBLAS compiled with
/// OpenMP.
#[pymodule]
mod _openblas_tod {
    use std::{ffi::c_void, mem::transmute};

    use super::install;
    use pyo3::prelude::*;

    /// Install a new thread pool model into the given OpenBLAS shared library.
    #[pyfunction]
    fn _install(get_num_threads_addr: usize, install_threads_callback_addr: usize) -> PyResult<()> {
        install(
            unsafe { transmute(get_num_threads_addr as *const c_void) },
            unsafe { transmute(install_threads_callback_addr as *const c_void) },
        )
        .unwrap();
        Ok(())
    }
}
