"""Mapped into the wheel as `tailcyclenet.configs` (see pyproject.toml `[tool.setuptools.package-dir]`).

This file exists ONLY so the `package_dir` remap survives a `pip install -e .` (setuptools'
PEP 660 editable finder resolves a mapped dotted name by looking for `<mapped-dir>/__init__.py`
directly, with no namespace-package fallback) -- confirmed empirically while writing
`dev/plans/pip_training_cli.md`. `configs/` itself is still the repo-root, human-edited, hand-
written directory `CLAUDE.md` documents; nothing here is generated or moved.
"""
