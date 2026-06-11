use std::{
    ffi::c_int,
    os::raw::c_void,
    sync::{OnceLock, mpsc::channel},
    thread,
};

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

struct Job {
    thread_num: c_int,
    data: *mut c_void,
}

/// These pointers are designed to be run in other threads:
unsafe impl Send for Job {}

extern "C" fn run_in_threads_callback(
    _sync: c_int, // TODO what does this mean?
    dojob: openblas_dojob_callback,
    numjobs: c_int,
    jobdata_elsize: usize,
    jobdata: *mut c_void,
    dojob_data: c_int,
) {
    let numjobs = numjobs as usize;
    let jobdata_elsize = jobdata_elsize as isize;
    let dojob = dojob.unwrap();

    // Create threads:
    let num_threads = unsafe { openblas_get_num_threads.get().unwrap()() } as usize;
    println!(
        "STARTING CUSTOM THREAD POOL WITH {num_threads} threads to run {numjobs} jobs from OpenBLAS"
    );
    thread::scope(|scope| {
        let mut threads = vec![];
        let mut txs = vec![];

        for _ in 0..num_threads {
            let (tx, rx) = channel::<Job>();
            threads.push(scope.spawn(|| {
                for job in rx {
                    dojob(job.thread_num, job.data, dojob_data);
                }
            }));
            txs.push(tx);
        }

        for i in 0..numjobs {
            let data = unsafe { jobdata.byte_offset((i as isize) * jobdata_elsize) };
            txs[i.rem_euclid(num_threads)]
                .send(Job {
                    thread_num: i as c_int,
                    data,
                })
                .unwrap();
        }
        drop(txs);

        for t in threads {
            t.join().unwrap();
        }
    });
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
