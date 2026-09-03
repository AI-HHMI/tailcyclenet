"""tailcyclenet -- posetail finetuned into an animal pose estimator.

The pinned posetail release still needs one scoped PyTorch SDPA compatibility shim; it is applied
only during pose-model scene encoding in ``tailcyclenet.model``.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("tailcyclenet")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
