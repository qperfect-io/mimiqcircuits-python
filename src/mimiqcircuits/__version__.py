from importlib.metadata import version

# Sourced from the installed package metadata (the `version` field in
# pyproject.toml), so the constant stays in sync with the package
# itself — no manual bump and no risk of drift like the previous
# hard-coded literal had.
__version__ = version("mimiqcircuits")
