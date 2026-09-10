"""tailcyclenet -- posetail finetuned into an animal pose estimator.

The pinned posetail release's projection is full-K, but its inverse normalization and projection
sensitivity historically ignored intrinsic skew. A scoped compatibility layer is installed before
pose, detector, or scorer modules import those shared geometry helpers.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("tailcyclenet")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

# Install before any posetail submodule imports its geometry helpers. The pinned release's
# projection is full-K, but its inverse normalization is scalar-focal for nonzero skew.
from .geometry import install_full_intrinsics as _install_full_intrinsics

_install_full_intrinsics()
