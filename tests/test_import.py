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
import importlib.metadata
import subprocess
import sys

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


def test_version_matches_package_metadata():
    assert mc.__version__ == importlib.metadata.version("mimiqcircuits")


def test_mimiqlink_constants():
    import mimiqlink

    assert mc.QPERFECT_CLOUD == mimiqlink.QPERFECT_CLOUD
    assert mc.QPERFECT_DEV == mimiqlink.QPERFECT_DEV
    assert mc.PLANQK_API == mimiqlink.PLANQK_API
