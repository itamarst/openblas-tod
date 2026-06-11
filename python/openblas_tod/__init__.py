"""
Install a custom threading model into OpenBLAS.
"""

from ctypes import cast, c_void_p

import threadpoolctl

from ._openblas_tod import _install


def _address(f: any) -> int:
    return cast(f, c_void_p).value


def install() -> None:
    """Install the custom threading model."""
    for library in threadpoolctl.threadpool_info():
        if (
            library["user_api"] == "blas"
            and library["internal_api"] == "openblas"
            and library["threading_layer"] == "pthreads"
        ):
            controller = threadpoolctl.OpenBLASController(
                filepath=library["filepath"], prefix=library["prefix"]
            )
            _install(
                _address(controller._get_symbol("openblas_get_num_threads")),
                _address(
                    controller._get_symbol("openblas_set_threads_callback_function")
                ),
            )
            return

    raise RuntimeError("No suitable OpenBLAS found")


__all__ = ["install"]
