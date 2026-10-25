# The version is compiled into the native extension from Cargo.toml, which a
# release bumps together with pyproject.toml. Reading it from there avoids
# importlib.metadata, which is slow to import.
from mimiqcircuits._core import __version__  # noqa: F401
