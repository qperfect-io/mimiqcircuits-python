#
# Copyright © 2022-2023 University of Strasbourg. All Rights Reserved.
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
import symengine as se
from mimiqcircuits._sympycheck import _is_basic, _is_sympy_matrix
from mimiqcircuits.operations.krauschannel import krauschannel
from mimiqcircuits.operations.rescaledgates import RescaledGate
from mimiqcircuits.operations.gates.custom import GateCustom
from mimiqcircuits.operations.gates.gate import Gate


def _is_valid_power_of_2(n):
    return n > 0 and (n & (n - 1)) == 0


def _convert_to_symengine(matrix):
    if isinstance(matrix, np.ndarray):
        return se.Matrix(matrix.tolist())
    elif _is_sympy_matrix(matrix):
        return se.Matrix(matrix.tolist())
    elif isinstance(matrix, se.Matrix):
        return matrix
    else:
        raise ValueError(f"Unsupported matrix type: {type(matrix)}")


class MixedUnitary(krauschannel):
    r"""Custom N-qubit mixed unitary channel.

    The channel applies the unitary :math:`U_k` with probability :math:`p_k`:

    .. math::
        \mathcal{E}(\rho) = \sum_k p_k U_k \rho U_k^\dagger,

    with :math:`0 \leq p_k \leq 1` and :math:`\sum_k p_k = 1`. If some Kraus
    operators are not proportional to a unitary, use :class:`Kraus` instead.

    There are two ways to build it:

    - ``MixedUnitary(p, U)`` with a list of probabilities ``p`` and a list
      ``U`` of the same length of gates or :math:`2^N \times 2^N` matrices
      (symengine, sympy or numpy). Matrices are wrapped in :class:`GateCustom`.
    - ``MixedUnitary(gates)`` with a list of :class:`RescaledGate`, where the
      gate ``RescaledGate(U_k, s_k)`` stands for :math:`U_k` applied with
      probability :math:`p_k = s_k^2`.

    Matrices are written in the computational basis in the usual textbook order,
    with the first qubit as the left-most one: :math:`|0\rangle, |1\rangle` for
    one qubit, :math:`|00\rangle, |01\rangle, |10\rangle, |11\rangle` for two.

    Args:
        *args: Either ``(p, U)`` or ``(gates,)`` as described above.
        lossy (list, optional): One entry per branch listing the qubits that are
            lost when that branch is drawn, as 1-based positions within the
            channel's targets (``1`` is the first target, up to :math:`N`).
            For example ``[[], [1]]`` makes the second branch lose the first
            target qubit. Branches are lossless by default.
            :meth:`Circuit.sample_mixedunitaries` emits a certain :class:`Loss`
            on these qubits when it draws a lossy branch.

    Raises:
        ValueError: If the arguments do not match one of the two forms, the
            lists have different lengths, a numeric probability is outside
            :math:`[0, 1]`, numeric probabilities do not add up to 1, a matrix size is not a power of 2, :math:`N` is not 1 or 2,
            or a ``lossy`` entry is out of range.

    Note:
        Only 1- and 2-qubit custom mixed unitary channels are supported.

    See Also:
        :class:`Kraus`, :meth:`ismixedunitary`, :class:`RescaledGate`

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> # identity with probability 0.9, X with probability 0.1
        >>> c = Circuit()
        >>> c.push(MixedUnitary([0.9, 0.1], [Matrix([[1, 0], [0, 1]]), Matrix([[0, 1], [1, 0]])]), 0)
        1-qubit circuit with 1 instruction:
        └── MixedUnitary((0.9, "Custom([1 0; 0 1])"),(0.1, "Custom([0 1; 1 0])")) @ q[0]
        <BLANKLINE>

        >>> # matrices can come from existing gates
        >>> c.push(MixedUnitary([0.8, 0.2], [Matrix(GateID().matrix()), Matrix(GateRX(0.2).matrix())]), 1)
        2-qubit circuit with 2 instructions:
        ├── MixedUnitary((0.9, "Custom([1 0; 0 1])"),(0.1, "Custom([0 1; 1 0])")) @ q[0]
        └── MixedUnitary((0.8, "Custom([1.0 0; 0 1.0])"),(0.2, "Custom([0.995004165278026 0.0 - 0.0998334166468282*I; 0.0 - 0.0998334166468282*I 0.995004165278026])")) @ q[1]
        <BLANKLINE>

        >>> # the same kind of channel from rescaled gates: the scale is sqrt(p)
        >>> rescaled_gate1 = RescaledGate(GateCustom(Matrix([[1, 0], [0, 1]])), sqrt(0.2))
        >>> rescaled_gate2 = RescaledGate(GateCustom(Matrix([[0, 1], [1, 0]])), sqrt(0.8))
        >>> op = MixedUnitary([rescaled_gate1, rescaled_gate2])
        >>> op
        MixedUnitary((0.2, "Custom([1 0; 0 1])"), (0.8, "Custom([0 1; 1 0])"))
        >>> c.push(op, 1)
        2-qubit circuit with 3 instructions:
        ├── MixedUnitary((0.9, "Custom([1 0; 0 1])"),(0.1, "Custom([0 1; 1 0])")) @ q[0]
        ├── MixedUnitary((0.8, "Custom([1.0 0; 0 1.0])"),(0.2, "Custom([0.995004165278026 0.0 - 0.0998334166468282*I; 0.0 - 0.0998334166468282*I 0.995004165278026])")) @ q[1]
        └── MixedUnitary((0.2, "Custom([1 0; 0 1])"),(0.8, "Custom([0 1; 1 0])")) @ q[1]
        <BLANKLINE>

        >>> # gates work too; here the X branch also loses the qubit
        >>> MixedUnitary([0.5, 0.5], [GateID(), GateX()], lossy=[[], [1]])
        MixedUnitary((0.5, "ID"), (0.5, "X"); lossy=((), (1,)))
    """

    _name = "MixedUnitary"
    _parnames = ()

    def __init__(self, *args, lossy=None):
        if len(args) == 1:
            rescaledgates = args[0]
            if not isinstance(rescaledgates, list) or not all(
                isinstance(g, RescaledGate) for g in rescaledgates
            ):
                raise ValueError("Expected a list of RescaledGate objects.")
            self.gates = rescaledgates
            self.p = [g.get_scale() ** 2 for g in self.gates]
            self.U = [g.get_operation() for g in self.gates]

        elif len(args) == 2:
            probabilities, gates_or_matrices = args
            if len(probabilities) != len(gates_or_matrices):
                raise ValueError(
                    "Lists of probabilities and gates/matrices must have the same length."
                )
            self.p = probabilities
            self.gates = []
            self.U = []

            for U, p in zip(gates_or_matrices, probabilities):
                if isinstance(U, (np.ndarray, se.Matrix)) or _is_sympy_matrix(U):
                    # Convert numpy or sympy matrices to symengine and wrap in GateCustom
                    U = _convert_to_symengine(U)
                    if not _is_valid_power_of_2(U.rows):
                        raise ValueError(
                            f"Matrix row count {U.rows} is not a valid power of 2."
                        )
                    gate = GateCustom(U)
                elif isinstance(U, Gate):
                    gate = U
                else:
                    raise ValueError(f"Unsupported type for U: {type(U)}")

                scale = se.sqrt(p)
                self.gates.append(RescaledGate(gate, scale))
                self.U.append(gate)

        else:
            raise ValueError("Invalid number of arguments. Expected 1 or 2.")

        numeric = [
            float(p) for p in self.p if not (_is_basic(p) and p.free_symbols)
        ]
        if any(p < 0 or p > 1 for p in numeric):
            raise ValueError("Probabilities must be between 0 and 1.")
        if len(numeric) == len(self.p) and not np.isclose(sum(numeric), 1, rtol=1e-13):
            raise ValueError("Probabilities must sum to 1.")

        # Use num_qubits() from the Gate class, or calculate from matrix dimensions
        if isinstance(self.gates[0].get_operation(), Gate):
            self.N = self.gates[0].get_operation().num_qubits
        else:
            self.N = int(se.log(self.gates[0].get_operation().rows, 2))

        if self.N < 1:
            raise ValueError("Cannot define a 0-qubit custom mixed unitary channel")
        if self.N > 2:
            raise ValueError(
                "Custom mixed unitary channels larger than 2 qubits are not supported"
            )

        # Per-branch lossy qubit masks (local 1:N indices), empty when lossless.
        if lossy is None:
            self.lossy = [tuple() for _ in self.gates]
        else:
            if len(lossy) != len(self.gates):
                raise ValueError(
                    "Lists of unitaries and lossy masks must have the same length."
                )
            self.lossy = []
            for m in lossy:
                qs = tuple(sorted({int(q) for q in m}))
                if any(q < 1 or q > self.N for q in qs):
                    raise ValueError(
                        f"Lossy qubit indices must be in 1:{self.N}, got {qs}."
                    )
                self.lossy.append(qs)

        super().__init__()
        self._num_qubits = self.N

    def haslossybranch(self):
        """Whether any branch loses qubits when drawn, as set by ``lossy``."""
        return any(self.lossy)

    def evaluate(self, values: dict):
        """Return a copy with symbols substituted by values.

        Substitutes into the probabilities and into the gates. The ``lossy``
        masks are kept.

        Args:
            values: Map from symbols to the values to substitute.

        Returns:
            MixedUnitary: A new channel with substituted parameters.

        Raises:
            ValueError: If the substituted probabilities are all numeric and do
                not add up to 1.
        """
        evaluated_p = [p.subs(values) if hasattr(p, "subs") else p for p in self.p]

        if all(
            isinstance(prob, (float, int)) or prob.is_number for prob in evaluated_p
        ):
            numeric_probs = [
                float(prob) if isinstance(prob, se.Basic) else prob
                for prob in evaluated_p
            ]
            if not np.allclose(np.sum(numeric_probs), 1, rtol=1e-13):
                raise ValueError("Evaluated probabilities must sum to 1.")

        evaluated_U = [
            u.evaluate(values) if hasattr(u, "evaluate") else u for u in self.U
        ]

        return MixedUnitary(evaluated_p, evaluated_U, lossy=self.lossy)

    def krausmatrices(self):
        probabilities = [se.sqrt(p) for p in self.probabilities()]
        unitary_gates = self.unitarygates()

        # Convert gates to matrices before multiplying
        unitary_matrices = [
            (
                gate.matrix()
                if hasattr(gate, "matrix") and callable(gate.matrix)
                else gate.matrix
            )
            for gate in unitary_gates
        ]

        # Make sure calling matrix() or using matrices
        return [prob * matrix for prob, matrix in zip(probabilities, unitary_matrices)]

    def probabilities(self):
        return self.p

    def unitarygates(self):
        return [g.get_operation() for g in self.gates]

    def unwrappedkrausmatrices(self):
        return self.krausmatrices()

    def krausoperators(self):
        # Get the unitary gates
        gates = self.unitarygates()
        scales = [se.sqrt(p) for p in self.probabilities()]
        return [RescaledGate(gate, scale) for gate, scale in zip(gates, scales)]

    @classmethod
    def ismixedunitary(self):
        return True

    def __str__(self):
        """Compact string representation for the MixedUnitary."""
        return self._format_mixedunitary(compact=True)

    def __repr__(self):
        """Full string representation for the MixedUnitary."""
        return self._format_mixedunitary(compact=False)

    def _format_mixedunitary(self, compact):
        """Helper function to format MixedUnitary based on compactness."""
        ps = self.probabilities()
        Us = self.unitarygates()

        repr_string = f"{self._name}("
        sep = "," if compact else ", "

        for i, (prob, gate) in enumerate(zip(ps, Us)):
            # Check if prob is numeric and can be rounded
            if isinstance(prob, (float, int)):
                prob = round(prob, 10)
            # Otherwise, keep the symbolic representation as is

            if isinstance(gate, GateCustom):
                matrix_str = f"Custom([{'; '.join([' '.join(map(str, row)) for row in gate._matrix().tolist()])}])"
                repr_string += f'({prob}, "{matrix_str}")'
            else:
                repr_string += f'({prob}, "{str(gate)}")'

            if i < len(ps) - 1:
                repr_string += sep

        if self.haslossybranch():
            repr_string += f"; lossy={tuple(self.lossy)}"

        repr_string += ")"
        return repr_string


__all__ = [
    "MixedUnitary",
]
