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
import mimiqcircuits as mc


def checknormalizlength(normaliz, length):
    if len(normaliz) != length:
        raise ValueError(f"normaliz must be a list or tuple of length {length}.")


class ProjectiveNoiseX(krauschannel):
    r"""
    ProjectiveNoiseX()

    Single qubit projection noise onto an X Pauli basis.

    This channel is defined by the Kraus operators:

    .. math::
        E_1 = |-\rangle \langle-|, \quad E_2 = |+\rangle \langle+|,

    where :math:`\ket{+}` and :math:`\ket{-}` are the eigenstates of Pauli `X`.

    See also:
        - :class:`ProjectiveNoise`
        - :class:`ProjectiveNoiseY`
        - :class:`ProjectiveNoiseZ`
    """

    _name = "ProjectiveNoiseX"
    _num_qubits = 1
    _num_bits = 0
    _num_zvars = 0
    _qregsizes = [1]

    def opname(self):
        return "ProjectiveNoiseX"

    def krausoperators(self):
        return [mc.ProjectorX0(), mc.ProjectorX1()]

    def __str__(self):
        return f"{self._name}"


class ProjectiveNoiseY(krauschannel):
    r"""
    ProjectiveNoiseY()

    Single qubit projection noise onto a Y Pauli basis.

    This channel is defined by the Kraus operators:

    .. math::
        E_1 = |Y0\rangle \langle Y0|, \quad E_2 = |Y1\rangle \langle Y1|,

    where :math:`\ket{Y0}` and :math:`\ket{Y1}` are the eigenstates of Pauli `Y`.

    See also:
        - :class:`ProjectiveNoise`
        - :class:`ProjectiveNoiseX`
        - :class:`ProjectiveNoiseZ`
    """

    _name = "ProjectiveNoiseY"
    _num_qubits = 1
    _num_bits = 0
    _num_zvars = 0
    _qregsizes = [1]

    def opname(self):
        return "ProjectiveNoiseY"

    def krausoperators(self):
        return [mc.ProjectorY0(), mc.ProjectorY1()]

    def __str__(self):
        return f"{self._name}"


class ProjectiveNoiseZ(krauschannel):
    r"""
    ProjectiveNoiseZ()

    Single qubit projection noise onto a Z Pauli basis.

    This channel is defined by the Kraus operators:

    .. math::
        E_1 = |0\rangle \langle Z0|, \quad E_2 = |1\rangle \langle Z1|,

    where :math:`\ket{0}` and :math:`\ket{1}` are the eigenstates of Pauli `Z`.

    See also:
        - :class:`ProjectiveNoise`
        - :class:`ProjectiveNoiseX`
        - :class:`ProjectiveNoiseY`
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
        return [mc.Projector0(), mc.Projector1()]


class ProjectiveNoise(krauschannel):
    r"""
    ProjectiveNoise(basis)

    Single qubit projection noise onto a Pauli basis.

    This channel is defined by the Kraus operators:

    .. math::
        E_1 = |\alpha\rangle \langle\alpha|, \quad E_2 = |\beta\rangle \langle\beta|,

    where the states :math:`|\alpha\rangle` and :math:`|\beta\rangle` are the +1 and -1
    eigenstates of a Pauli operator. Specifically, they correspond to
    :math:`\{ |0\rangle, |1\rangle \}` (Z basis),
    :math:`\{ |+\rangle, |-\rangle \}` (X basis),
    or :math:`\{ |y+\rangle, |y-\rangle \}` (Y basis).

    This operation is similar to measuring in the corresponding basis (X, Y, or Z),
    except that the outcome of the measurement is not stored, i.e., there is a loss of information.

    Parameters:
        basis (str): String or character that selects the Pauli basis, `"X"`, `"Y"`, or `"Z"`.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(ProjectiveNoise("Z"), 1)
        2-qubit circuit with 1 instructions:
        └── ProjectiveNoiseZ @ q[1]
        <BLANKLINE>

    The Kraus matrices are given by:

        >>> ProjectiveNoise("X").krausmatrices()
        [[0.5, 0.5]
        [0.5, 0.5]
        , [0.5, -0.5]
        [-0.5, 0.5]
        ]

        >>> ProjectiveNoise("Y").krausmatrices()
        [[0.5, -0.0 - 0.5*I]
        [0.0 + 0.5*I, 0.5]
        , [0.5, 0.0 + 0.5*I]
        [-0.0 - 0.5*I, 0.5]
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
