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



def control_decompose(circ, operation, controls, target):
    """Push ``operation`` on ``target`` controlled by ``controls`` to ``circ``.

    With zero or one control the gate is pushed directly; with more, it is
    decomposed recursively into singly-controlled gates and multi-controlled
    X gates, following Barenco et al. (1995).

    Args:
        circ: The circuit to push to.
        operation: A single-qubit gate.
        controls: The control qubits.
        target: The target qubit.

    Returns:
        The circuit ``circ``.
    """
    from mimiqcircuits.operations.control import Control
    if len(controls) == 0:
        circ.push(operation, target)
    elif len(controls) == 1:
        circ.push(Control(1, operation), controls[0], target)
    else:
        control_recursive_decompose(circ, operation, controls, target)

    return circ


def cx_decompose(circ, controls, target, ancillas):
    """Push an X gate on ``target`` controlled by ``controls`` to ``circ``.

    Uses the free qubits in ``ancillas`` to decompose the gate into Toffoli
    gates (Lemmas 7.2 and 7.3 of Barenco et al., 1995). The ancillas are
    returned to their initial state, whatever it is. Without ancillas it
    falls back to :func:`control_decompose`.

    Args:
        circ: The circuit to push to.
        controls: The control qubits.
        target: The target qubit.
        ancillas: Qubits not involved in the gate, usable as workspace.

    Returns:
        The circuit ``circ``.
    """
    from mimiqcircuits.operations.gates.standard.pauli import GateX
    from mimiqcircuits.operations.gates.standard.cnx import GateCCX
    from mimiqcircuits.operations.control import Control
    from mimiqcircuits.circuit import Circuit
    nctrl = len(controls)
    nq = nctrl + 1 + len(ancillas)

    if nctrl == 0:
        circ.push(GateX(), target)

    elif nctrl <= 2:
        circ.push(Control(nctrl, GateX()), *controls, target)

    elif nq >= 2 * nctrl - 1 and nctrl >= 3:
        # Decomposition according to Lemma 7.2 of [1]
        circ1 = Circuit()

        for i in range(nctrl - 3):
            circ1.push(
                GateCCX(), controls[-(i + 2)], ancillas[-(i + 2)], ancillas[-(i + 1)]
            )

        circ2 = Circuit()
        circ2.push(GateCCX(), controls[0], controls[1], ancillas[-(nctrl - 2)])

        circ3 = Circuit()
        circ3 = circ3.push(GateCCX(), controls[-1], ancillas[-1], target)

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
        circ1 = Circuit()
        cx_decompose(circ1, ctrl1, trgt1, free1)

        free2 = list(controls[:m]) + list(ancillas[1:])
        ctrl2 = list(controls[m:]) + [ancillas[0]]
        trgt2 = target
        circ2 = Circuit()
        cx_decompose(circ2, ctrl2, trgt2, free2)

        circ.append(circ1)
        circ.append(circ2)
        circ.append(circ1)
        circ.append(circ2)
    else:
        control_decompose(circ, GateX(), controls, target)

    return circ


def control_recursive_decompose(circ, operation, controls, target):
    """Push a multi-controlled ``operation`` to ``circ`` by recursion on its square root.

    Follows Barenco et al. (1995): the gate with ``n``
    controls is built from singly-controlled square roots of
    ``operation``, multi-controlled X gates, and the same construction
    with ``n - 1`` controls.

    Args:
        circ: The circuit to push to.
        operation: A single-qubit gate.
        controls: The control qubits, at least two.
        target: The target qubit.

    Returns:
        The circuit ``circ``.
    """
    from mimiqcircuits.operations.control import Control
    N = len(controls)

    V = operation.power(1 / 2)
    Vdag = V.inverse()

    circ.push(Control(1, V), controls[-1], target)

    # here target can be used as a free qubit
    cx_decompose(circ, controls[:-1], controls[-1], [target])

    circ.push(Control(1, Vdag), controls[-1], target)

    # here target can be used as a free qubit
    cx_decompose(circ, controls[:-1], controls[-1], [target])

    if N == 2:
        circ.push(Control(N - 1, V), controls[:-1], target)
    else:
        control_recursive_decompose(circ, V, controls[:-1], target)

    return circ


def rccx(circ, a, b, t):
    """Relative-phase Toffoli.

    A Toffoli up to a diagonal phase on its three qubits, with 4 T gates. It
    is its own inverse.
    """
    from mimiqcircuits.operations.gates.standard.cpauli import GateCX
    from mimiqcircuits.operations.gates.standard.hadamard import GateH
    from mimiqcircuits.operations.gates.standard.t import GateT, GateTDG

    circ.push(GateH(), t)
    circ.push(GateT(), t)
    circ.push(GateCX(), b, t)
    circ.push(GateTDG(), t)
    circ.push(GateCX(), a, t)
    circ.push(GateT(), t)
    circ.push(GateCX(), b, t)
    circ.push(GateTDG(), t)
    circ.push(GateH(), t)
    return circ


def cnx_ladder(circ, controls, target, ancillas):
    """C_n(X) with m clean ancillas, m <= n - 2.

    The ladder stores in ancilla i the AND of the first i+1 controls. The
    remaining controls and the last ancilla control the target, then the
    ladder is undone. The relative phases of the ladder cancel: they are
    diagonal, and the middle gate only reads the ladder qubits. With
    m < n - 2 the middle gate still has 3 or more controls; it borrows the
    qubits the ladder has already read, which sit idle meanwhile.
    """
    from mimiqcircuits.circuit import Circuit
    from mimiqcircuits.operations.annotations import AncillaAlloc, AncillaFree

    m = len(ancillas)
    for a in ancillas:
        circ.push(AncillaAlloc(), a)
    ladder = rccx(Circuit(), controls[0], controls[1], ancillas[0])
    for i in range(1, m):
        rccx(ladder, ancillas[i - 1], controls[i + 1], ancillas[i])
    rest = list(controls[m + 1 :])
    circ.append(ladder)
    cx_decompose(
        circ,
        [ancillas[m - 1]] + rest,
        target,
        list(controls[: m + 1]) + list(ancillas[: m - 1]),
    )
    circ.append(ladder.inverse())
    for a in ancillas:
        circ.push(AncillaFree(), a)
    return circ


def and_compute(circ, a, b, t):
    """Logical AND of ``a`` and ``b`` into ``t``, which must be in ``|0>``.

    The relative-phase Toffoli leaves a phase i on ``|111>`` only, among the
    states it reaches from t = 0; the S-dagger removes it.
    """
    from mimiqcircuits.operations.gates.standard.s import GateSDG

    rccx(circ, a, b, t)
    circ.push(GateSDG(), t)
    return circ


def and_uncompute(circ, a, b, t, bit):
    """Undo ``and_compute(circ, a, b, t)`` by measurement, without T gates.

    Measured in the X basis, ``t`` gives a random bit; when it is 1 the state
    has picked up a phase -1 where a and b are both 1, which a CZ conditioned
    on the bit undoes. The reset returns ``t`` to ``|0>``.
    """
    from mimiqcircuits.bitstrings import BitString
    from mimiqcircuits.operations.gates.standard.cpauli import GateCZ
    from mimiqcircuits.operations.gates.standard.hadamard import GateH
    from mimiqcircuits.operations.ifstatement import IfStatement
    from mimiqcircuits.operations.measurereset import MeasureReset

    circ.push(GateH(), t)
    circ.push(MeasureReset(), t, bit)
    circ.push(IfStatement(GateCZ(), BitString("1")), a, b, bit)
    return circ


def cnx_measured_ladder(circ, controls, target, ancillas, bit):
    """C_n(X) with m clean ancillas, m <= n - 1, and a bit to measure into.

    As :func:`cnx_ladder`, but each step is an exact AND, undone by
    measurement.
    """
    from mimiqcircuits.operations.annotations import AncillaAlloc, AncillaFree

    m = len(ancillas)
    steps = [
        (controls[0] if i == 0 else ancillas[i - 1], controls[i + 1], ancillas[i])
        for i in range(m)
    ]
    for a in ancillas:
        circ.push(AncillaAlloc(), a)
    for a, b, t in steps:
        and_compute(circ, a, b, t)
    rest = list(controls[m + 1 :])
    cx_decompose(
        circ,
        [ancillas[m - 1]] + rest,
        target,
        list(controls[: m + 1]) + list(ancillas[: m - 1]),
    )
    for a, b, t in reversed(steps):
        and_uncompute(circ, a, b, t, bit)
    for a in ancillas:
        circ.push(AncillaFree(), a)
    return circ
