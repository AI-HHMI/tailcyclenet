"""tailcyclenet -- posetail finetuned into an animal pose estimator.

No monkeypatching: posetail >= 0.3.5 ships every behaviour this repo once had to patch in.
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("tailcyclenet")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
