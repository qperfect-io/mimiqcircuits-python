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
from typing import List, Union
import symengine as se
import mimiqcircuits as mc
import sympy as sp
from mimiqcircuits.operations.krauschannel import krauschannel
from mimiqcircuits.operations.rescaledgates import RescaledGate
from mimiqcircuits.operations.gates.custom import GateCustom
from mimiqcircuits.operations.gates.gate import Gate


def _is_valid_power_of_2(n):
    return n > 0 and (n & (n - 1)) == 0


def _convert_to_symengine(matrix):
    if isinstance(matrix, np.ndarray):
        return se.Matrix(matrix.tolist())
    elif isinstance(matrix, sp.Matrix):
        return se.Matrix(matrix.tolist())
    elif isinstance(matrix, se.Matrix):
        return matrix
    else:
        raise ValueError(f"Unsupported matrix type: {type(matrix)}")


class MixedUnitary(krauschannel):
    r"""MixedUnitary(p,umatrices).

    Custom N-qubit mixed unitary channel specified by a list of :math:`2^N \times 2^N`
    unitary matrices and a list of probabilities that add up to 1.

    A mixed unitary noise channel is defined by:

    .. math::
        \mathcal{E}(\rho) = \sum_k p_k U_k \rho U_k^\dagger,

    where :math:`0 \leq p_k \leq 1` and :math:`U_k` are unitary matrices.
    The probabilities must fulfill :math:`\sum_k p_k = 1`.

    If your Kraus matrices are not all proportional to unitaries, use :func:`KrausChannel` instead.

    The unitary matrices are defined in the computational basis in the usual textbook order
    (the first qubit corresponds to the left-most qubit).
    For 1 qubit, we have :math:`|0\rangle`, :math:`|1\rangle`.
    For 2 qubits, we have :math:`|00\rangle`, :math:`|01\rangle`, :math:`|10\rangle`, :math:`|11\rangle`.

    **Note:** Currently, only 1 and 2-qubit custom MixedUnitary channels are supported.

    See Also:
        :class:`Kraus`
        :func:`ismixedunitary`
        :class:`krauschannel`

    Parameters:
        p (list): List of probabilities, must be positive real numbers and add up to 1.
        umatrices (list): List of complex-valued :math:`2^N \times 2^N` matrices. The number of qubits is equal to :math:`N`.

    The length of the lists `p` and `umatrices` must be equal.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import *
        >>> c = Circuit()
        >>> c.push(MixedUnitary([0.9, 0.1], [Matrix([[1, 0], [0, 1]]), Matrix([[0, 1], [1, 0]])]), 0)
        1-qubit circuit with 1 instructions:
        └── MixedUnitary((0.9, "Custom([1 0; 0 1])"),(0.1, "Custom([0 1; 1 0])")) @ q[0]
        <BLANKLINE>

        >>> c.push(MixedUnitary([0.8, 0.2], [Matrix(GateID().matrix()), Matrix(GateRX(0.2).matrix())]), 1)
        2-qubit circuit with 2 instructions:
        ├── MixedUnitary((0.9, "Custom([1 0; 0 1])"),(0.1, "Custom([0 1; 1 0])")) @ q[0]
        └── MixedUnitary((0.8, "Custom([1.0 0; 0 1.0])"),(0.2, "Custom([0.995004165278026 -0.0 - 0.0998334166468282*I; -0.0 - 0.0998334166468282*I 0.995004165278026])")) @ q[1]
        <BLANKLINE>

        RescaleGate

        >>> p1= 0.2
        >>> p2 = 0.8
        >>> U1 = Matrix([[1, 0], [0, 1]])  # Identity matrix
        >>> U2 = Matrix([[0, 1], [1, 0]])  # Pauli-X matrix
        >>> gate1 = GateCustom(U1)
        >>> gate2 = GateCustom(U2)
        >>> rescaled_gate1 = RescaledGate(gate1, sqrt(p1))
        >>> rescaled_gate2 = RescaledGate(gate2, sqrt(p2))
        >>> op = MixedUnitary([rescaled_gate1, rescaled_gate2])
        >>> op
        MixedUnitary((0.2, "Custom([1 0; 0 1])"), (0.8, "Custom([0 1; 1 0])"))
        >>> c.push(op,1)
        2-qubit circuit with 3 instructions:
        ├── MixedUnitary((0.9, "Custom([1 0; 0 1])"),(0.1, "Custom([0 1; 1 0])")) @ q[0]
        ├── MixedUnitary((0.8, "Custom([1.0 0; 0 1.0])"),(0.2, "Custom([0.995004165278026 -0.0 - 0.0998334166468282*I; -0.0 - 0.0998334166468282*I 0.995004165278026])")) @ q[1]
        └── MixedUnitary((0.2, "Custom([1 0; 0 1])"),(0.8, "Custom([0 1; 1 0])")) @ q[1]
        <BLANKLINE>
    """

    _name = "MixedUnitary"
    _parnames = ()

    def __init__(self, *args):
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
            # Check if there are any symbolic probabilities
            contains_symbolic_probabilities = any(
                isinstance(p, (se.Basic, sp.Basic)) for p in probabilities
            )

            if not contains_symbolic_probabilities:
                if not np.allclose(np.sum(probabilities), 1, rtol=1e-13):
                    raise ValueError("Probabilities must sum to 1.")

            self.p = probabilities
            self.gates = []
            self.U = []

            for U, p in zip(gates_or_matrices, probabilities):
                if isinstance(U, (np.ndarray, sp.Matrix, se.Matrix)):
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

        super().__init__()
        self._num_qubits = self.N

    def evaluate(self, values: dict):
        """Evaluates symbolic parameters in the MixedUnitary using a dictionary of values.

        Args:
            values (dict): A dictionary where keys are symbolic variables and values are the corresponding numerical values.

        Returns:
            MixedUnitary: A new MixedUnitary instance with evaluated parameters.
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

        return MixedUnitary(evaluated_p, evaluated_U)

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

    def unitarymatrices(self):
        return self.U

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

            if isinstance(gate, mc.GateCustom):
                matrix_str = f"Custom([{'; '.join([' '.join(map(str, row)) for row in gate.matrix.tolist()])}])"
                repr_string += f'({prob}, "{matrix_str}")'
            else:
                repr_string += f'({prob}, "{str(gate)}")'

            if i < len(ps) - 1:
                repr_string += sep

        repr_string += ")"
        return repr_string


__all__ = [
    "MixedUnitary",
]
