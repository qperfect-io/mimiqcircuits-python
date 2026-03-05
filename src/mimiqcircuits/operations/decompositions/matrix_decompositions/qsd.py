#
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

import numpy as np

import mimiqcircuits as mc
from .zyz import _zyz_decomposition
from .csd import _csd_decomposition
from .utils import _as_numpy_numeric


def _qsd_decomposition(U):
    """
    Perform Quantum Shannon Decomposition (QSD) on an N-qubit unitary U.
    Returns (Circuit, Phase) where Circuit implements U * exp(-i * Phase).
    """

    # normalize
    U_np = _as_numpy_numeric(U)
    n = U_np.shape[0]
    nb_qubits = int(round(np.log2(n)))

    if nb_qubits == 1:
        return _decompose_1q(U, target=0)

    #  CSD on MSB 
    L0, L1, R0, R1, theta = _csd_decomposition(U)

    circ = mc.Circuit()

    sub_targets = list(range(1, nb_qubits))

    # Right term: diag(R0, R1)
    pR = _append_multiplexed_unitary(
        circ, R0, R1, control=0, targets=sub_targets
    )

    # Middle term: multiplexed Ry
    angles = [2.0 * float(theta[i, 0]) for i in range(theta.nrows())]
    _append_multiplexed_ry(
        circ, angles, target=0, controls=sub_targets
    )

    #  Left term: diag(L0, L1) 
    pL = _append_multiplexed_unitary(
        circ, L0, L1, control=0, targets=sub_targets
    )

    return circ, (pR + pL)


def _decompose_1q(U, target):
    theta, phi, lam, gamma = _zyz_decomposition(U)
    c = mc.Circuit()
    c.push(mc.GateU(theta, phi, lam, gamma), target)
    return c, 0.0


def _append_multiplexed_unitary(circ, U0, U1, control, targets):
    # Base case
    if len(targets) == 1:
        target = targets[0]

        t0, p0, l0, g0 = _zyz_decomposition(U0)
        t1, p1, l1, g1 = _zyz_decomposition(U1)

        _append_multiplexed_rz(circ, [l0, l1], target, [control])
        _append_multiplexed_ry(circ, [t0, t1], target, [control])
        _append_multiplexed_rz(circ, [p0, p1], target, [control])

        phase0 = g0 + (p0 + l0) / 2.0
        phase1 = g1 + (p1 + l1) / 2.0

        circ.push(mc.GateP(phase1 - phase0), control)

        return phase0

    # Recursive case
    return _append_multiplexed_unitary_recursive(
        circ, U0, U1, control, targets
    )


def _append_multiplexed_unitary_recursive(circ, U0, U1, control, targets):
    c0, p0 = _qsd_decomposition(U0)
    c1, p1 = _qsd_decomposition(U1)

    mapping = {i: targets[i] for i in range(len(targets))}

    circ.push(mc.GateX(), control)
    for inst in c0:
        circ.push(*_add_control_remapped(inst, control, mapping))
    circ.push(mc.GateX(), control)

    for inst in c1:
        circ.push(*_add_control_remapped(inst, control, mapping))

    circ.push(mc.GateP(p1 - p0), control)

    return p0


def _add_control_remapped(inst, control_qubit, mapping):
    op = inst.get_operation()
    q = list(inst.get_qubits())
    new_q = [mapping[i] for i in q]
    return (mc.Control(1, op), control_qubit, *new_q)


def _append_multiplexed_ry(circ, angles, target, controls, eps=1e-10):
    k = len(controls)
    n = len(angles)

    if k == 0:
        circ.push(mc.GateRY(angles[0]), target)
        return

    for i in range(n):
        angle = angles[i]
        if abs(angle) < eps:
            continue

        state_idx = i

        active_x = []
        for bit_pos in range(k):
            bit = (state_idx >> (k - 1 - bit_pos)) & 1
            if bit == 0:
                active_x.append(bit_pos)

        for idx in active_x:
            circ.push(mc.GateX(), controls[idx])

        circ.push(mc.Control(k, mc.GateRY(angle)), *controls, target)

        for idx in active_x:
            circ.push(mc.GateX(), controls[idx])


def _append_multiplexed_rz(circ, angles, target, controls, eps=1e-10):
    k = len(controls)
    n = len(angles)

    if k == 0:
        circ.push(mc.GateRZ(angles[0]), target)
        return

    for i in range(n):
        angle = angles[i]
        if abs(angle) < eps:
            continue

        state_idx = i

        active_x = []
        for bit_pos in range(k):
            bit = (state_idx >> (k - 1 - bit_pos)) & 1
            if bit == 0:
                active_x.append(bit_pos)

        for idx in active_x:
            circ.push(mc.GateX(), controls[idx])

        circ.push(mc.Control(k, mc.GateRZ(angle)), *controls, target)

        for idx in active_x:
            circ.push(mc.GateX(), controls[idx])
