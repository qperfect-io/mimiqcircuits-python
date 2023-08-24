#
# Copyright © 2022-2023 University of Strasbourg. All Rights Reserved.
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
from mimiqcircuits.gates import Gate
from mimiqcircuits.barrier import Barrier
from mimiqcircuits.measure import Measure
from mimiqcircuits.reset import Reset
from mimiqcircuits.parallel import Parallel
from mimiqcircuits.control import Control
from mimiqcircuits.ifstatement import IfStatement


def operation_from_json(d):
    name = d['name']

    if name == "Barrier":
        return Barrier.from_json(d)

    elif name == "Measure":
        return Measure.from_json(d)

    elif name == "Reset":
        return Reset.from_json(d)

    elif name == "Parallel":
        return Parallel.from_json(d)

    elif name == "Control":
        return Control.from_json(d)

    elif name == "If":
        return IfStatement.from_json(d)

    return Gate.from_json(d)
