"""The inference program: the window loop, the driver over a dataset, and the command line.

Private names are re-exported on purpose -- `detector/evaluate.py` imports `_window_starts` --
so dropping one breaks callers silently at import time.  The re-exports are lazy so importing a
small inference utility (for example, ``tailcyclenet.infer.bridge``) does not load the complete
inference stack.
"""
from importlib import import_module as _import_module


# Keep the package import cheap.  In particular, bridge and other table-only utilities should not
# import the CLI, dataset, or posetail merely because they live below this package.
_LAZY_EXPORTS = {
    'build_parser': ('.cli', 'build_parser'),
    'main': ('.cli', 'main'),
    'run_dataset': ('.driver', 'run_dataset'),
    'FrameStore': ('.store', 'FrameStore'),
    'ANCHORS': ('.window', 'ANCHORS'),
    'CARRY_SOURCES': ('.window', 'CARRY_SOURCES'),
    'ORACLE_CORRUPTIONS': ('.window', 'ORACLE_CORRUPTIONS'),
    'OUTCOMES': ('.window', 'OUTCOMES'),
    'InferConfig': ('.window', 'InferConfig'),
    'boxes_from_points': ('.window', 'boxes_from_points'),
    'merge_blocks': ('.window', 'merge_blocks'),
    'run_blocks': ('.window', 'run_blocks'),
    'run_group': ('.window', 'run_group'),
    'self_prompt': ('.window', 'self_prompt'),
    '_build_prior': ('.window', '_build_prior'),
    '_corrupt_prior': ('.window', '_corrupt_prior'),
    '_crop_views': ('.window', '_crop_views'),
    '_deploy_box_prompt': ('.window', '_deploy_box_prompt'),
    '_window_starts': ('.window', '_window_starts'),
}

# Keep the historical order: callers may use __all__ as part of the public surface.
__all__ = [
    'ANCHORS', 'CARRY_SOURCES', 'ORACLE_CORRUPTIONS', 'OUTCOMES', 'FrameStore',
    'InferConfig', 'boxes_from_points', 'merge_blocks', 'run_blocks', 'run_group', 'self_prompt',
    'run_dataset', 'build_parser', 'main', '_build_prior', '_corrupt_prior', '_crop_views',
    '_deploy_box_prompt', '_window_starts',
]

# These package submodules were exposed as attributes as a side effect of the old eager imports.
# Resolve them lazily too, retaining that compatibility without reopening the import closure.
_LAZY_MODULES = {
    name: f'.{name}' for name in ('cli', 'driver', 'store', 'window', 'predictions')
}


def __getattr__(name: str):
    """Resolve a package re-export or submodule only when that attribute is requested."""
    target = _LAZY_EXPORTS.get(name)
    if target is not None:
        module_name, attribute = target
        value = getattr(_import_module(module_name, __name__), attribute)
    else:
        module_name = _LAZY_MODULES.get(name)
        if module_name is None:
            raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
        value = _import_module(module_name, __name__)
    globals()[name] = value
    return value


def __dir__():
    """Include lazy exports and submodules in package introspection."""
    return sorted(set(globals()) | set(__all__) | set(_LAZY_MODULES))
