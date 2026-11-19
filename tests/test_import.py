#
# Copyright © 2026 QPerfect. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import ast
import importlib.metadata
import importlib.util
import pathlib
import subprocess
import sys
import sysconfig

import pytest

import mimiqcircuits as mc

# Heavy dependencies that only some features need. Importing any of them
# from module scope makes `import mimiqcircuits` several times slower.
DEFERRED = ["sympy", "scipy", "mimiqlink", "matplotlib", "importlib.metadata"]


def test_import_does_not_load_heavy_dependencies():
    code = (
        "import sys, mimiqcircuits\n"
        f"print(','.join(m for m in {DEFERRED!r} if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == ""


def test_import_keeps_gil_disabled():
    # An extension that does not declare free-threading support makes the
    # interpreter turn the GIL back on for the whole process when it loads.
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        pytest.skip("not a free-threaded interpreter")
    code = "import sys, mimiqcircuits\nprint(sys._is_gil_enabled())"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


def test_version_matches_package_metadata():
    assert mc.__version__ == importlib.metadata.version("mimiqcircuits")


def test_mimiqlink_constants():
    import mimiqlink

    assert mc.QPERFECT_CLOUD == mimiqlink.QPERFECT_CLOUD
    assert mc.QPERFECT_DEV == mimiqlink.QPERFECT_DEV
    assert mc.PLANQK_API == mimiqlink.PLANQK_API


def _root_imports(path):
    """Imports in `path` that go through the `mimiqcircuits` package itself."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "mimiqcircuits":
                    yield node.lineno, f"import {alias.name}"
        elif isinstance(node, ast.ImportFrom) and node.module == "mimiqcircuits":
            for alias in node.names:
                # Submodules are fine: importing one does not depend on what
                # the package has initialized so far.
                if importlib.util.find_spec(f"mimiqcircuits.{alias.name}") is None:
                    yield node.lineno, f"from mimiqcircuits import {alias.name}"


def test_modules_import_from_defining_module():
    # Inside the package, a name read from `mimiqcircuits` itself only exists
    # once `__init__.py` has reached the line that imports it, so the package
    # would load only in one particular order. Import from the module that
    # defines the name instead.
    src = pathlib.Path(mc.__file__).parent
    found = [
        f"{path.relative_to(src)}:{lineno}: {stmt}"
        for path in sorted(src.rglob("*.py"))
        if path != src / "__init__.py"
        for lineno, stmt in _root_imports(path)
    ]
    assert found == []
