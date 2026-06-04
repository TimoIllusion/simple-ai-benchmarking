# Cap BLAS/OpenMP thread pools BEFORE numpy/torch are imported anywhere in the
# package. RunPod-style containers are a slice of a much larger host, but
# numpy/OpenBLAS/MKL/torch read the HOST core count and otherwise spawn one
# thread per core (often 100+) on a few allocated vCPUs -> severe oversubscription
# that can stall a benchmark for many minutes. This module is imported before any
# submodule's numpy/torch import (including in spawned worker processes), so the
# limits take effect. setdefault lets callers override (e.g. OMP_NUM_THREADS=16).
import os as _os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    _os.environ.setdefault(_var, "8")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
