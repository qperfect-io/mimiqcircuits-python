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
"""Pauli-based computation pass."""
"""Tools built on the AncillaAlloc / AncillaFree markers."""

from mimiqcircuits.circuit import Circuit
from mimiqcircuits.operations.annotations import AncillaAlloc, AncillaFree


def ancillas(c):
    """The qubits of ``c`` that carry an :class:`AncillaAlloc`, in increasing order.

    These are the clean ancillas that :func:`decompose` added or reused;
    every other qubit holds data.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(Control(4, GateX()), 0, 1, 2, 3, 4)
        5-qubit circuit with 1 instruction:
        └── C₄X @ q[0:3], q[4]
        <BLANKLINE>
        >>> ancillas(decompose(c, CanonicalBasis(max_ancillas=2)))
        [5, 6]
    """
    return sorted({i.qubits[0] for i in c if isinstance(i.operation, AncillaAlloc)})


def merge_ancilla_windows(c):
    """Merge the ancilla windows that meet on a qubit.

    Returns a copy of ``c`` where an :class:`AncillaFree` followed on the
    same qubit, with nothing in between, by an :class:`AncillaAlloc` is
    removed together with it. The two windows become one. This makes two
    operations that use the same ancilla adjacent on its wire again, for
    example so that a pass can cancel a call against its inverse.
    """
    keep = [True] * len(c)
    # Index of the last instruction on each qubit, if it is an AncillaFree.
    lastfree = {}
    for i, inst in enumerate(c):
        op = inst.operation
        if isinstance(op, AncillaAlloc):
            j = lastfree.pop(inst.qubits[0], None)
            if j is not None:
                keep[i] = keep[j] = False
        elif isinstance(op, AncillaFree):
            lastfree[inst.qubits[0]] = i
        else:
            for q in inst.qubits:
                lastfree.pop(q, None)
    return Circuit([inst for inst, k in zip(c, keep) if k])


def check_ancillas(c):
    """Check that the ancilla markers of ``c`` are consistent.

    It checks that

    - on each qubit, every :class:`AncillaFree` closes an open
      :class:`AncillaAlloc`, and every window is closed by the end of ``c``;
    - the last ``num_ancillas`` qubit targets of every operation with
      ``num_ancillas`` above zero are inside open windows.

    It does not simulate ``c``: whether an ancilla really is in ``|0>`` where
    the markers say so is up to whoever placed them.

    Raises:
        ValueError: Naming the first problem found.
    """
    depth = {}
    for i, inst in enumerate(c):
        op = inst.operation
        if isinstance(op, AncillaAlloc):
            q = inst.qubits[0]
            depth[q] = depth.get(q, 0) + 1
        elif isinstance(op, AncillaFree):
            q = inst.qubits[0]
            if depth.get(q, 0) == 0:
                raise ValueError(
                    f"instruction {i}: AncillaFree on qubit {q}, which has no open window"
                )
            depth[q] -= 1
        else:
            nanc = op.num_ancillas
            for q in inst.qubits[len(inst.qubits) - nanc :]:
                if depth.get(q, 0) == 0:
                    raise ValueError(
                        f"instruction {i}: clean ancilla qubit {q} of {op.name} is "
                        "not inside an AncillaAlloc/AncillaFree window"
                    )
    for q, d in depth.items():
        if d != 0:
            raise ValueError(f"qubit {q}: AncillaAlloc without a matching AncillaFree")


__all__ = ["ancillas", "check_ancillas", "merge_ancilla_windows"]
