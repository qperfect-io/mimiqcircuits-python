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

# References
# [1] Barenco, A. et al. Elementary gates for quantum computation. Phys. Rev. A 52, 3457–3467 (1995).
#

import mimiqcircuits as mc


def control_decompose(circ, operation, controls, target):
    if len(controls) == 0:
        circ.push(operation, target)
    elif len(controls) == 1:
        circ.push(mc.Control(1, operation), controls[0], target)
    else:
        control_recursive_decompose(circ, operation, controls, target)

    return circ


def cx_decompose(circ, controls, target, ancillas):
    nctrl = len(controls)
    nq = nctrl + 1 + len(ancillas)

    if nctrl == 0:
        circ.push(mc.GateX(), target)

    elif nctrl <= 2:
        circ.push(mc.Control(nctrl, mc.GateX()), *controls, target)

    elif nq >= 2 * nctrl - 1 and nctrl >= 3:
        # Decomposition according to Lemma 7.2 of [1]
        circ1 = mc.Circuit()

        for i in range(nctrl - 3):
            circ1.push(
                mc.GateCCX(), controls[-(i + 2)], ancillas[-(i + 2)], ancillas[-(i + 1)]
            )

        circ2 = mc.Circuit()
        circ2.push(mc.GateCCX(), controls[0], controls[1], ancillas[-(nctrl - 3)])

        circ3 = mc.Circuit()
        circ3 = circ3.push(mc.GateCCX(), controls[-1], ancillas[-1], target)

        circ.append(circ3)
        circ.append(circ1)
        circ.append(circ2)
        circ.append(circ1.inverse())
        circ.append(circ3.inverse())
        circ.append(circ1)
        circ.append(circ2)
        circ.append(circ1.inverse())

    elif not len(ancillas) == 0:
        # Decomposition according to Lemma 7.3 of [1]
        m = nq // 2

        free1 = list(controls[m:]) + [target] + list(ancillas[1:])
        ctrl1 = controls[:m]
        trgt1 = ancillas[0]
        circ1 = mc.Circuit()
        cx_decompose(circ1, ctrl1, trgt1, free1)

        free2 = list(controls[:m]) + list(ancillas[1:])
        ctrl2 = list(controls[m:]) + [ancillas[0]]
        trgt2 = target
        circ2 = mc.Circuit()
        cx_decompose(circ2, ctrl2, trgt2, free2)

        circ.append(circ1)
        circ.append(circ2)
        circ.append(circ1)
        circ.append(circ2)
    else:
        control_decompose(circ, mc.GateX(), controls, target)

    return circ


def control_recursive_decompose(circ, operation, controls, target):
    N = len(controls)

    V = operation.power(1 / 2)
    Vdag = V.inverse()

    circ.push(mc.Control(1, V), controls[-1], target)

    # here target can be used as a free qubit
    cx_decompose(circ, controls[:-1], controls[-1], [target])
    # circ.push(mc.Control(N - 1, GateX()), *controls)

    circ.push(mc.Control(1, Vdag), controls[-1], target)

    # here target can be used as a free qubit
    cx_decompose(circ, controls[:-1], controls[-1], [target])
    # circ.push(mc.Control(N - 1, GateX()), *controls)

    if N == 2:
        circ.push(mc.Control(N - 1, V), controls[:-1], target)
    else:
        control_recursive_decompose(circ, V, controls[:-1], target)

    return circ
