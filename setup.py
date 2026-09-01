from setuptools import setup

# All metadata lives in pyproject.toml [project]/[tool.setuptools] -- that static config is
# authoritative for `packages`, so this list is kept in sync for readability, not because
# setuptools reads it (pyproject.toml wins; confirmed while writing dev/plans/pip_training_cli.md).
# Keep discovery explicit: the repository also contains large data and environment directories
# that make recursive discovery slow.
setup(packages=["tailcyclenet", "tailcyclenet.detector", "tailcyclenet.infer",
                "tailcyclenet.configs", "tailcyclenet.configs.datasets"])
