import numpy as np
from numpy.testing import assert_array_almost_equal

A = np.random.random((1, 10_000_000))
B = np.random.random((10_000_000, 2))

print("Running with normal OpenBLAS thread pool:")
result1 = A @ B
print(result1[:1000])

print("Installing openblas-tod thread pool:")
import openblas_tod
openblas_tod.install()

print("Running with new thread pool:")
result2 = A @ B

print("Checking if it's the same...")
assert_array_almost_equal(result1, result2)
print("IT IS!")
