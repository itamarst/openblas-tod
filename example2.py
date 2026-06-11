"""
Demonstrate that number of threads used by OpenBLAS is per-thread calling into
OpenBLAS, instead of global.
"""

import threading
import psutil
import numpy as np
import threadpoolctl
threadpoolctl.threadpool_limits(limits=30, user_api="blas")

import sys
if sys._is_gil_enabled():
    print("You'll want to use free-threaded Python for this to give reasonable noutput")
    sys.exit(1)

if sys.argv[1:] == ["tod"]:
    import openblas_tod
    openblas_tod.install()

A = np.random.random((1, 10_000_000))
B = np.random.random((10_000_000, 20))

max_num_threads = 0

process = psutil.Process()

def max_threads():
    global max_num_threads
    while threads:
        max_num_threads = max(process.num_threads(), max_num_threads)

threads = []
t = threading.Thread(target=max_threads)
threads.append(t)
t.start()

for _ in range(10):
    t = threading.Thread(target=lambda: A @ B)
    threads.append(t)
    t.start()



while threads:
    t = threads.pop()
    t.join()

print("Max number of threads counted:", max_num_threads)
