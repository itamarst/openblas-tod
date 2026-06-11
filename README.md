# OpenBLAS-TOD: Replace the global OpenBLAS thread pool with threads on demand

The goal:

* When using OpenBLAS pthreads (e.g. `pip install numpy`), setting the number of threads in OpenBLAS sets it globally for a process-wide thread pool.
* When using OpenBLAS with OpenMP, setting the the number of threads is _thread-local_.

Imagine you set the number of OpenBLAS threads to 4, and you start a Python thread pool with 5 threads; each one starts an OpenBLAS computation.
Depending on which version of OpenBLAS you get, you might 4 threads running OpenBLAS code (mixing work from 5 Python thread), or a total of 4×5 == 20 threads running OpenBLAS code.

This inconsistency is a problem!

This project (currently a proof of concept) installs new logic into OpenBLAS on pthreads, so that it starts a thread pool for each parallel operation, more like OpenMP.
(Unlike OpenMP there is still only one single process-wide setting for number of OpenBLAS threads.)
