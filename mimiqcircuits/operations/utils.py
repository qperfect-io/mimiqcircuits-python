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

from mimiqcircuits.operations.power import Power
from mimiqcircuits.operations.gates.standard.id import GateID


def power_nhilpotent(op, pow):
    pmod = pow % 2

    if pmod == 0:
        if op.num_qubits < 1:
            raise ValueError("Operation defined on zero qubits.")
        if op.num_qubits == 1:
            return GateID()
        else:
            return GateID().control(op.num_qubits - 1)
    else:
        return Power(op, pmod)
