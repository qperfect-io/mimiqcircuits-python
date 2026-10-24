#
# Copyright © 2022-2024 University of Strasbourg. All Rights Reserved.
# Copyright © 2023-2025 QPerfect. All Rights Reserved.
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
import importlib
import pkgutil

import pytest

import mimiqcircuits

MODULES = [
    m.name
    for m in pkgutil.walk_packages(mimiqcircuits.__path__, "mimiqcircuits.")
    # optional extra
    if not m.name.startswith("mimiqcircuits.visualization")
]


@pytest.mark.parametrize("name", ["mimiqcircuits"] + MODULES)
def test_all_names_are_defined(name):
    mod = importlib.import_module(name)
    missing = [n for n in getattr(mod, "__all__", []) if not hasattr(mod, n)]
    assert missing == []
