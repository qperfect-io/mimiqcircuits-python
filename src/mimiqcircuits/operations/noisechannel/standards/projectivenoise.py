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

import symengine as se
from mimiqcircuits.operations.krauschannel import krauschannel
from mimiqcircuits.operations.operators.projectors import (
    Projector0,
    Projector1,
    ProjectorX0,
    ProjectorX1,
    ProjectorY0,
    ProjectorY1,
)


class ProjectiveNoiseX(krauschannel):
    r"""One-qubit projective noise in the X basis.

    The Kraus operators are

    .. math::
        E_1 = |+\rangle\langle+|, \quad E_2 = |-\rangle\langle-|,

    the projectors onto the eigenstates :math:`|+\rangle` and :math:`|-\rangle` of Pauli X. The channel removes the coherences between
    the two basis states, like a measurement in the X basis whose outcome is
    discarded. ``ProjectiveNoise("X")`` returns an instance of this class.

    See Also:
        :class:`ProjectiveNoise`, :class:`ProjectiveNoiseY`,
        :class:`ProjectiveNoiseZ`

    Examples:
        >>> from mimiqcircuits import *
        >>> ProjectiveNoiseX().krausoperators()
        [PX₀(1), PX₁(1)]
    """

    _name = "ProjectiveNoiseX"
    _num_qubits = 1
    _num_bits = 0
    _num_zvars = 0
    _qregsizes = [1]

    def opname(self):
        return "ProjectiveNoiseX"

    def krausoperators(self):
        return [ProjectorX0(), ProjectorX1()]

    def __str__(self):
        return f"{self._name}"


class ProjectiveNoiseY(krauschannel):
    r"""One-qubit projective noise in the Y basis.

    The Kraus operators are

    .. math::
        E_1 = |y+\rangle\langle y+|, \quad E_2 = |y-\rangle\langle y-|,

    the projectors onto the eigenstates :math:`|y+\rangle` and :math:`|y-\rangle` of Pauli Y. The channel removes the coherences between
    the two basis states, like a measurement in the Y basis whose outcome is
    discarded. ``ProjectiveNoise("Y")`` returns an instance of this class.

    See Also:
        :class:`ProjectiveNoise`, :class:`ProjectiveNoiseX`,
        :class:`ProjectiveNoiseZ`

    Examples:
        >>> from mimiqcircuits import *
        >>> ProjectiveNoiseY().krausoperators()
        [PY₀(1), PY₁(1)]
    """

    _name = "ProjectiveNoiseY"
    _num_qubits = 1
    _num_bits = 0
    _num_zvars = 0
    _qregsizes = [1]

    def opname(self):
        return "ProjectiveNoiseY"

    def krausoperators(self):
        return [ProjectorY0(), ProjectorY1()]

    def __str__(self):
        return f"{self._name}"


class ProjectiveNoiseZ(krauschannel):
    r"""One-qubit projective noise in the Z basis.

    The Kraus operators are

    .. math::
        E_1 = |0\rangle\langle 0|, \quad E_2 = |1\rangle\langle 1|,

    the projectors onto the computational basis states, the eigenstates of Pauli Z. The channel removes the coherences between
    the two basis states, like a measurement in the Z basis whose outcome is
    discarded. ``ProjectiveNoise("Z")`` returns an instance of this class.

    See Also:
        :class:`ProjectiveNoise`, :class:`ProjectiveNoiseX`,
        :class:`ProjectiveNoiseY`

    Examples:
        >>> from mimiqcircuits import *
        >>> ProjectiveNoiseZ().krausoperators()
        [P₀(1), P₁(1)]
    """

    _name = "ProjectiveNoiseZ"
    _num_qubits = 1
    _num_bits = 0
    _num_zvars = 0
    _qregsizes = [1]

    def opname(self):
        return "ProjectiveNoiseZ"

    def __str__(self):
        return f"{self._name}"

    def krausoperators(self):
        return [Projector0(), Projector1()]


class ProjectiveNoise(krauschannel):
    r"""One-qubit projective noise in a Pauli basis.

    The Kraus operators are

    .. math::
        E_1 = |\alpha\rangle \langle\alpha|, \quad E_2 = |\beta\rangle \langle\beta|,

    where :math:`|\alpha\rangle` and :math:`|\beta\rangle` are the +1 and -1
    eigenstates of a Pauli operator: :math:`\{ |0\rangle, |1\rangle \}` for Z,
    :math:`\{ |+\rangle, |-\rangle \}` for X and
    :math:`\{ |y+\rangle, |y-\rangle \}` for Y.

    The channel acts like a measurement in that basis whose outcome is not
    stored: the coherences between the two basis states are lost.

    Calling ``ProjectiveNoise(basis)`` returns an instance of
    :class:`ProjectiveNoiseX`, :class:`ProjectiveNoiseY` or
    :class:`ProjectiveNoiseZ`, not of ``ProjectiveNoise``.

    Args:
        basis (str): ``"X"``, ``"Y"`` or ``"Z"`` (default ``"Z"``).

    Raises:
        ValueError: If ``basis`` is not one of ``"X"``, ``"Y"``, ``"Z"``.

    Examples:
        >>> from mimiqcircuits import *
        >>> # dephase qubit 1 in the computational basis
        >>> c = Circuit()
        >>> c.push(ProjectiveNoise("Z"), 1)
        2-qubit circuit with 1 instruction:
        └── ProjectiveNoiseZ @ q[1]
        <BLANKLINE>

        >>> # the Kraus matrices are the projectors onto the basis states
        >>> ProjectiveNoise("X").krausmatrices()
        [[0.5, 0.5]
        [0.5, 0.5]
        , [0.5, -0.5]
        [-0.5, 0.5]
        ]

        >>> ProjectiveNoise("Y").krausmatrices()
        [[0.5, 0.0 - 0.5*I]
        [0.0 + 0.5*I, 0.5]
        , [0.5, 0.0 + 0.5*I]
        [0.0 - 0.5*I, 0.5]
        ]

        >>> ProjectiveNoise("Z").krausmatrices()
        [[1.0, 0]
        [0, 0]
        , [0, 0]
        [0, 1.0]
        ]
    """

    _name = "ProjectiveNoise"
    _num_qubits = 1
    _qregsizes = [1]
    _cregsizes = ()

    def __new__(self, basis="Z"):
        if basis == "X":
            return ProjectiveNoiseX()
        elif basis == "Y":
            return ProjectiveNoiseY()
        elif basis == "Z":
            return ProjectiveNoiseZ()
        else:
            raise ValueError(
                "Invalid basis for Projective noise. Must be 'X', 'Y', or 'Z'."
            )

    def krausmatrices(self):
        return [se.Matrix(kraus.matrix().tolist()) for kraus in self.krausoperators()]

    def __str__(self):
        return f"{self._name}"


__all__ = [
    "ProjectiveNoise",
    "ProjectiveNoiseX",
    "ProjectiveNoiseY",
    "ProjectiveNoiseZ",
]
