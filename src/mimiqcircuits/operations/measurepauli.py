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
"""Pauli product measurement (MeasurePauli)."""

from typing import Union

from mimiqcircuits.operations.operation import Operation
from mimiqcircuits.operations.measure import AbstractMeasurement, Measure
from mimiqcircuits.operations.gates.generalized.paulistring import PauliString
import mimiqcircuits as mc


class MeasurePauli(AbstractMeasurement):
    r"""Pauli product measurement.

    Projective measurement of the :math:`N`-qubit observable
    :math:`P = P_1 \otimes P_2 \otimes \dots \otimes P_N`, with each
    :math:`P_i \in \{I, X, Y, Z\}`.

    The instruction targets :math:`N` qubits and one classical bit. The state
    is projected onto the :math:`+1` or :math:`-1` eigenspace of :math:`P` and
    the bit records the outcome: ``0`` for :math:`+1`, ``1`` for :math:`-1`.
    Qubits marked ``I`` are targets of the instruction but are not measured.

    To measure :math:`-P`, measure :math:`P` and flip the bit with
    :class:`Not`.

    Args:
        pauli (PauliString or str): the observable, as a :class:`PauliString`
            or a string of ``I``, ``X``, ``Y``, ``Z``.

    Raises:
        TypeError: if ``pauli`` is neither a :class:`PauliString` nor a
            ``str``.
        ValueError: if the string is empty, contains a character other than
            ``I``, ``X``, ``Y``, ``Z``, or is the identity.

    See Also:
        :class:`PauliString`, :class:`RPauli`, :class:`MeasureZZ`,
        :class:`MeasureXX`, :class:`MeasureYY`,
        :func:`pauli_based_computation`

    Examples:

        Measure :math:`X_0 Z_2` on qubits 0 to 2 and store the outcome in
        bit 0. Qubit 1 is a target, but is left untouched:

        >>> from mimiqcircuits import *
        >>> MeasurePauli("XIZ")
        MP(XIZ)
        >>> c = Circuit()
        >>> c.push(MeasurePauli("XIZ"), 0, 1, 2, 0)
        3-qubit, 1-bit circuit with 1 instruction:
        └── MP(XIZ) @ q[0:2], c[0]
        <BLANKLINE>

    Decomposition:

        Each measured qubit is rotated to the Z basis (``H`` for ``X``,
        ``SDG`` then ``H`` for ``Y``), CX gates collect the parity of the
        measured qubits on the last one, that qubit is measured, and the CX
        gates and rotations are undone:

        >>> decompose_step(MeasurePauli("XZ"))
        2-qubit, 1-bit circuit with 5 instructions:
        ├── H @ q[0]
        ├── CX @ q[0], q[1]
        ├── M @ q[1], c[0]
        ├── CX @ q[0], q[1]
        └── H @ q[0]
        <BLANKLINE>

        Only ``H``, ``S``, ``SDG``, ``CX`` and ``Measure`` appear, so
        :class:`QASMBasis` decomposes it. :class:`StimBasis` keeps it, as Stim
        measures Pauli products natively (``MPP``), and so does
        :class:`CliffordTBasis`.
    """

    _name = "MeasurePauli"
    _num_bits = 1
    _num_cregs = 1
    _cregsizes = [1]
    _num_zvars = 0

    def __init__(self, pauli: Union[PauliString, str]):
        if isinstance(pauli, str):
            pauli = PauliString(pauli)
        elif not isinstance(pauli, PauliString):
            raise TypeError(
                f"MeasurePauli takes a PauliString or a str, got {type(pauli).__name__}."
            )
        if pauli.is_identity():
            raise ValueError("Cannot measure the identity Pauli string.")
        self.pauli = pauli
        self._num_qubits = pauli.num_qubits
        self._qregsizes = [pauli.num_qubits]

    def pstring(self):
        """Return the Pauli string as a plain ``str``."""
        return self.pauli.pauli

    def inverse(self):
        raise TypeError("MeasurePauli is not inversible")

    def power(self, p):
        raise TypeError("MeasurePauli^p is not defined.")

    def control(self, num_qubits):
        raise TypeError("Controlled MeasurePauli is not defined.")

    def iswrapper(self):
        return False

    def get_operation(self):
        return self

    def __str__(self):
        return f"MP({self.pstring()})"

    def asciiwidth(self, qubits, bits, zvars):
        # Leave room for the per-qubit index labels drawn inside the box.
        return Operation.asciiwidth(self, qubits, bits, zvars)

    def _decompose(self, circ, qubits, bits, zvars):
        active = [(p, q) for p, q in zip(self.pstring(), qubits) if p != "I"]
        target = active[-1][1]

        # Keep the decomposed circuit as wide as the instruction.
        max_targ = max(qubits)
        if all(q != max_targ for _, q in active):
            circ.push(mc.GateID(), max_targ)

        # H·S† maps Y to Z using only gates that the Stim and QASM bases keep.
        for p, q in active:
            if p == "Y":
                circ.push(mc.GateSDG(), q)
            if p != "Z":
                circ.push(mc.GateH(), q)
        for _, q in active:
            if q != target:
                circ.push(mc.GateCX(), q, target)

        circ.push(Measure(), target, bits[0])

        for _, q in reversed(active):
            if q != target:
                circ.push(mc.GateCX(), q, target)
        for p, q in reversed(active):
            if p != "Z":
                circ.push(mc.GateH(), q)
            if p == "Y":
                circ.push(mc.GateS(), q)

        return circ


__all__ = ["MeasurePauli"]
