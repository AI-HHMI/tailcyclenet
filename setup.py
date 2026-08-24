from setuptools import setup

# All metadata lives in pyproject.toml [project]. Keep discovery explicit: the repository
# also contains large data and environment directories that make recursive discovery slow.
setup(packages=["tailcyclenet", "tailcyclenet.detector", "tailcyclenet.infer"])
