"""
Install a custom threading model into OpenBLAS.
"""

import threadpoolctl

from ._openblas_tod import _install


def install() -> None:
    """Install the custom threading model."""
    # TODO single-threaded for now
    threadpoolctl.threadpool_limits(limits=1, user_api="blas")
    for library in threadpoolctl.threadpool_info():
        if (
            library["user_api"] == "blas"
            and library["internal_api"] == "openblas"
            and library["threading_layer"] == "pthreads"
        ):
            _install(library["filepath"])
            return

    raise RuntimeError("No suitable OpenBLAS found")


__all__ = ["install"]
