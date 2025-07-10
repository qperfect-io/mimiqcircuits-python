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

import symengine as se
import mimiqcircuits as mc


def to_subscript(number):
    subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    return str(number).translate(subscript_map)


class Projector0(mc.AbstractOperator):
    r"""One-qubit operator corresponding to a projection onto :math:`|0\rangle`.

    **Matrix Representation**

    .. math::
        \begin{pmatrix}
            a & 0 \\
            0 & 0
        \end{pmatrix}

    This matrix is parametrized by `a` to allow for phases/rescaling.
    Equivalent to :class:`DiagonalOp(a, 0)`.

    The parameter `a` is optional and is set to 1 by default.

    See Also:
        :class:`Projector1`, :class:`DiagonalOp`

    Parameters:
        a (complex, optional): The top-left entry of the matrix. Defaults to 1.

    Examples:
        >>> from mimiqcircuits import *
        >>> Projector0()
        P₀(1)

        >>> Projector0(0.5)
        P₀(0.5)

        >>> c = Circuit()
        >>> c.push(ExpectationValue(Projector0()), 1, 1)
        2-qubit, 2-zvar circuit with 1 instructions:
        └── ⟨P₀(1)⟩ @ q[1], z[1]
        <BLANKLINE>
    """

    _name = "Projector0"
    _num_qubits = 1
    _parnames = ()
    _qregsizes = [1]

    def __init__(self, a=1):
        self.a = a
        super().__init__()
        self._parnames = ("a",)

    def _matrix(self):
        return se.Matrix([[self.a, 0], [0, 0]])

    @property
    def parnames(self):
        return self._parnames

    def __str__(self):
        return f"P{to_subscript(0)}({self.a})"

    def rescale(self, scale):
        return Projector0(self.a * scale)

    def rescale_inplace(self, scale):
        self.a *= scale
        return self

    def opsquared(self):
        return Projector0(abs(self.a) ** 2)


class Projector1(mc.AbstractOperator):
    r"""One-qubit operator corresponding to a projection onto :math:`|1\rangle`.

    **Matrix Representation**

    .. math::
        \begin{pmatrix}
            0 & 0 \\
            0 & a
        \end{pmatrix}

    This matrix is parametrized by `a` to allow for phases/rescaling.
    Equivalent to :class:`DiagonalOp(0, a)`.

    The parameter `a` is optional and is set to 1 by default.

    See Also:
        :class:`Projector0`, :class:`DiagonalOp`

    Parameters:
        a (complex, optional): The bottom-right entry of the matrix. Defaults to 1.

    Examples:
        >>> from mimiqcircuits import *
        >>> Projector1()
        P₁(1)

        >>> Projector1(0.5)
        P₁(0.5)

        >>> c = Circuit()
        >>> c.push(ExpectationValue(Projector1()), 1, 1)
        2-qubit, 2-zvar circuit with 1 instructions:
        └── ⟨P₁(1)⟩ @ q[1], z[1]
        <BLANKLINE>
    """

    _name = "Projector1"
    _num_qubits = 1
    _parnames = ()
    _qregsizes = [1]

    def __init__(self, a=1):
        self.a = a
        super().__init__()
        self._parnames = ("a",)

    def _matrix(self):
        return se.Matrix([[0, 0], [0, self.a]])

    @property
    def parnames(self):
        return self._parnames

    def __str__(self):
        return f"P{to_subscript(1)}({self.a})"

    def rescale(self, scale):
        return Projector1(self.a * scale)

    def rescale_inplace(self, scale):
        self.a *= scale
        return self

    def opsquared(self):
        return Projector1(abs(self.a) ** 2)


class ProjectorX0(mc.AbstractOperator):
    r"""One-qubit operator corresponding to a projection onto :math:`|+\rangle`.

    **Matrix Representation**

    .. math::
        \frac{a}{2}
        \begin{pmatrix}
            1 & 1 \\
            1 & 1
        \end{pmatrix}

    This matrix is parametrized by `a` to allow for phases/rescaling.

    The parameter `a` is optional and is set to 1 by default.

    See Also:
        :class:`ProjectorX1`

    Parameters:
        a (complex, optional): Scaling factor for the matrix. Defaults to 1.

    Examples:
        >>> from mimiqcircuits import *
        >>> ProjectorX0()
        PX₀(1)

        >>> ProjectorX0(0.5)
        PX₀(0.5)

        >>> c = Circuit()
        >>> c.push(ExpectationValue(ProjectorX0()), 1, 1)
        2-qubit, 2-zvar circuit with 1 instructions:
        └── ⟨PX₀(1)⟩ @ q[1], z[1]
        <BLANKLINE>
    """

    _name = "PX0"
    _num_qubits = 1
    _parnames = ()
    _qregsizes = [1]

    def __init__(self, a=1):
        self.a = a
        super().__init__()
        self._parnames = ("a",)

    def _matrix(self):
        return self.a / 2 * se.Matrix([[1, 1], [1, 1]])

    @property
    def parnames(self):
        return self._parnames

    def __str__(self):
        return f"PX{to_subscript(0)}({self.a})"

    def rescale(self, scale):
        return ProjectorX0(self.a * scale)

    def rescale_inplace(self, scale):
        self.a *= scale
        return self

    def opsquared(self):
        return ProjectorX0(abs(self.a) ** 2)


class ProjectorX1(mc.AbstractOperator):
    r"""One-qubit operator corresponding to a projection onto :math:`|-\rangle`.

    **Matrix Representation**

    .. math::
        \frac{a}{2}
        \begin{pmatrix}
            1 & -1 \\
            -1 & 1
        \end{pmatrix}

    This matrix is parametrized by `a` to allow for phases/rescaling.

    The parameter `a` is optional and is set to 1 by default.

    See Also:
        :class:`ProjectorX0`

    Parameters:
        a (complex, optional): Scaling factor for the matrix. Defaults to 1.

    Examples:
        >>> from mimiqcircuits import *
        >>> ProjectorX1()
        PX₁(1)

        >>> ProjectorX1(0.5)
        PX₁(0.5)

        >>> c = Circuit()
        >>> c.push(ExpectationValue(ProjectorX1()), 1, 1)
        2-qubit, 2-zvar circuit with 1 instructions:
        └── ⟨PX₁(1)⟩ @ q[1], z[1]
        <BLANKLINE>
        
    """

    _name = "PX1"
    _num_qubits = 1
    _parnames = ()
    _qregsizes = [1]

    def __init__(self, a=1):
        self.a = a
        super().__init__()
        self._parnames = ("a",)

    def _matrix(self):
        return self.a / 2 * se.Matrix([[1, -1], [-1, 1]])

    @property
    def parnames(self):
        return self._parnames

    def __str__(self):
        return f"PX{to_subscript(1)}({self.a})"

    def rescale(self, scale):
        return ProjectorX1(self.a * scale)

    def rescale_inplace(self, scale):
        self.a *= scale
        return self

    def opsquared(self):
        return ProjectorX1(abs(self.a) ** 2)


class ProjectorY0(mc.AbstractOperator):
    r"""One-qubit operator corresponding to a projection onto :math:`|y+\rangle`.

    **Matrix Representation**

    .. math::
        \frac{a}{2}
        \begin{pmatrix}
            1 & -i \\
            i & 1
        \end{pmatrix}

    This matrix is parametrized by `a` to allow for phases/rescaling.

    The parameter `a` is optional and is set to 1 by default.

    See Also:
        :class:`ProjectorY1`

    Parameters:
        a (complex, optional): Scaling factor for the matrix. Defaults to 1.

    Examples:
        >>> from mimiqcircuits import *
        >>> ProjectorY0()
        PY₀(1)

        >>> ProjectorY0(0.5)
        PY₀(0.5)

        >>> c = Circuit()
        >>> c.push(ExpectationValue(ProjectorY0()), 1, 1)
        2-qubit, 2-zvar circuit with 1 instructions:
        └── ⟨PY₀(1)⟩ @ q[1], z[1]
        <BLANKLINE>
    """

    _name = "PY0"
    _num_qubits = 1
    _parnames = ()
    _qregsizes = [1]

    def __init__(self, a=1):
        self.a = a
        super().__init__()
        self._parnames = ("a",)

    def _matrix(self):
        return self.a / 2 * se.Matrix([[1, -1j], [1j, 1]])

    @property
    def parnames(self):
        return self._parnames

    def __str__(self):
        return f"PY{to_subscript(0)}({self.a})"

    def rescale(self, scale):
        return ProjectorY0(self.a * scale)

    def rescale_inplace(self, scale):
        self.a *= scale
        return self

    def opsquared(self):
        return ProjectorY0(abs(self.a) ** 2)


class ProjectorY1(mc.AbstractOperator):
    r"""One-qubit operator corresponding to a projection onto :math:`|y-\rangle`.

    **Matrix Representation**

    .. math::
        \frac{a}{2}
        \begin{pmatrix}
            1 & i \\
            -i & 1
        \end{pmatrix}

    This matrix is parametrized by `a` to allow for phases/rescaling.

    The parameter `a` is optional and is set to 1 by default.

    See Also:
        :class:`ProjectorY0`

    Parameters:
        a (complex, optional): Scaling factor for the matrix. Defaults to 1.

    Examples:
        >>> from mimiqcircuits import *
        >>> ProjectorY1()
        PY₁(1)

        >>> ProjectorY1(0.5)
        PY₁(0.5)

        >>> c = Circuit()
        >>> c.push(ExpectationValue(ProjectorY1()), 1, 1)
        2-qubit, 2-zvar circuit with 1 instructions:
        └── ⟨PY₁(1)⟩ @ q[1], z[1]
        <BLANKLINE>
    """

    _name = "PY1"
    _num_qubits = 1
    _parnames = ()
    _qregsizes = [1]

    def __init__(self, a=1):
        self.a = a
        super().__init__()
        self._parnames = ("a",)

    def _matrix(self):
        return self.a / 2 * se.Matrix([[1, 1j], [-1j, 1]])

    @property
    def parnames(self):
        return self._parnames

    def __str__(self):
        return f"PY{to_subscript(1)}({self.a})"

    def rescale(self, scale):
        return ProjectorY1(self.a * scale)

    def rescale_inplace(self, scale):
        self.a *= scale
        return self

    def opsquared(self):
        return ProjectorY1(abs(self.a) ** 2)


class Projector00(mc.AbstractOperator):
    r"""Two-qubit operator corresponding to a projection onto :math:`|00\rangle`.

    **Matrix Representation**

    .. math::
        \begin{pmatrix}
            a & 0 & 0 & 0\\
            0 & 0 & 0 & 0\\
            0 & 0 & 0 & 0\\
            0 & 0 & 0 & 0
        \end{pmatrix}

    This matrix is parametrized by `a` to allow for phases/rescaling.

    The parameter `a` is optional and is set to 1 by default.

    See Also:
        :class:`Projector01`, :class:`Projector10`, :class:`Projector11`

    Parameters:
        a (complex, optional): Scaling factor for the matrix. Defaults to 1.

    Examples:
        >>> from mimiqcircuits import *
        >>> Projector00()
        P₀₀(1)

        >>> Projector00(0.5)
        P₀₀(0.5)

        >>> c = Circuit()
        >>> c.push(ExpectationValue(Projector00()), 1, 2, 1)
        3-qubit, 2-zvar circuit with 1 instructions:
        └── ⟨P₀₀(1)⟩ @ q[1,2], z[1]
        <BLANKLINE>
    """

    _name = "Projector00"
    _num_qubits = 2
    _parnames = ()
    _qregsizes = [2]

    def __init__(self, a=1):
        self.a = a
        super().__init__()
        self._parnames = ("a",)

    def _matrix(self):
        return se.Matrix([[self.a, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    @property
    def parnames(self):
        return self._parnames

    def __str__(self):
        return f"P{to_subscript(0)}{to_subscript(0)}({self.a})"

    def rescale(self, scale):
        return Projector00(self.a * scale)

    def rescale_inplace(self, scale):
        self.a *= scale
        return self

    def opsquared(self):
        return Projector00(abs(self.a) ** 2)


class Projector01(mc.AbstractOperator):
    r"""Two-qubit operator corresponding to a projection onto :math:`|01\rangle`.

    **Matrix Representation**

    .. math::
        \begin{pmatrix}
            0 & 0 & 0 & 0\\
            0 & a & 0 & 0\\
            0 & 0 & 0 & 0\\
            0 & 0 & 0 & 0
        \end{pmatrix}

    This matrix is parametrized by `a` to allow for phases/rescaling.

    The parameter `a` is optional and is set to 1 by default.

    See Also:
        :class:`Projector00`, :class:`Projector10`, :class:`Projector11`

    Parameters:
        a (complex, optional): Scaling factor for the matrix. Defaults to 1.

    Examples:
        >>> from mimiqcircuits import *
        >>> Projector01()
        P₀₁(1)

        >>> Projector01(0.5)
        P₀₁(0.5)

        >>> c = Circuit()
        >>> c.push(ExpectationValue(Projector01()), 1, 2, 1)
        3-qubit, 2-zvar circuit with 1 instructions:
        └── ⟨P₀₁(1)⟩ @ q[1,2], z[1]
        <BLANKLINE>
    """

    _name = "Projector01"
    _num_qubits = 2
    _parnames = ()
    _qregsizes = [2]

    def __init__(self, a=1):
        self.a = a
        super().__init__()
        self._parnames = ("a",)

    def _matrix(self):
        return se.Matrix([[0, 0, 0, 0], [0, self.a, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    @property
    def parnames(self):
        return self._parnames

    def __str__(self):
        return f"P{to_subscript(0)}{to_subscript(1)}({self.a})"

    def rescale(self, scale):
        return Projector01(self.a * scale)

    def rescale_inplace(self, scale):
        self.a *= scale
        return self

    def opsquared(self):
        return Projector01(abs(self.a) ** 2)


class Projector10(mc.AbstractOperator):
    r"""Two-qubit operator corresponding to a projection onto :math:`|10\rangle`.

    **Matrix Representation**

    .. math::
        \begin{pmatrix}
            0 & 0 & 0 & 0\\
            0 & 0 & 0 & 0\\
            0 & 0 & a & 0\\
            0 & 0 & 0 & 0
        \end{pmatrix}

    This matrix is parametrized by `a` to allow for phases/rescaling.

    The parameter `a` is optional and is set to 1 by default.

    See Also:
        :class:`Projector00`, :class:`Projector01`, :class:`Projector11`

    Parameters:
        a (complex, optional): Scaling factor for the matrix. Defaults to 1.

    Examples:
        >>> from mimiqcircuits import *
        >>> Projector10()
        P₁₀(1)

        >>> Projector10(0.5)
        P₁₀(0.5)

        >>> c = Circuit()
        >>> c.push(ExpectationValue(Projector10()), 1, 2, 1)
        3-qubit, 2-zvar circuit with 1 instructions:
        └── ⟨P₁₀(1)⟩ @ q[1,2], z[1]
        <BLANKLINE>
    """

    _name = "Projector10"
    _num_qubits = 2
    _parnames = ()
    _qregsizes = [2]

    def __init__(self, a=1):
        self.a = a
        super().__init__()
        self._parnames = ("a",)

    def _matrix(self):
        return se.Matrix([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, self.a, 0], [0, 0, 0, 0]])

    @property
    def parnames(self):
        return self._parnames

    def __str__(self):
        return f"P{to_subscript(1)}{to_subscript(0)}({self.a})"

    def rescale(self, scale):
        return Projector10(self.a * scale)

    def rescale_inplace(self, scale):
        self.a *= scale
        return self

    def opsquared(self):
        return Projector10(abs(self.a) ** 2)


class Projector11(mc.AbstractOperator):
    r"""Two-qubit operator corresponding to a projection onto :math:`|11\rangle`.

    **Matrix Representation**

    .. math::
        \begin{pmatrix}
            0 & 0 & 0 & 0\\
            0 & 0 & 0 & 0\\
            0 & 0 & 0 & 0\\
            0 & 0 & 0 & a
        \end{pmatrix}

    This matrix is parametrized by `a` to allow for phases/rescaling.

    The parameter `a` is optional and is set to 1 by default.

    See Also:
        :class:`Projector00`, :class:`Projector01`, :class:`Projector10`

    Parameters:
        a (complex, optional): Scaling factor for the matrix. Defaults to 1.

    Examples:
        >>> from mimiqcircuits import *
        >>> Projector11()
        P₁₁(1)

        >>> Projector11(0.5)
        P₁₁(0.5)

        >>> c = Circuit()
        >>> c.push(ExpectationValue(Projector11()), 1, 2, 1)
        3-qubit, 2-zvar circuit with 1 instructions:
        └── ⟨P₁₁(1)⟩ @ q[1,2], z[1]
        <BLANKLINE>
    """

    _name = "Projector11"
    _num_qubits = 2
    _parnames = ()
    _qregsizes = [2]

    def __init__(self, a=1):
        self.a = a
        super().__init__()
        self._parnames = ("a",)

    def _matrix(self):
        return se.Matrix([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, self.a]])

    @property
    def parnames(self):
        return self._parnames

    def __str__(self):
        return f"P{to_subscript(1)}{to_subscript(1)}({self.a})"

    def rescale(self, scale):
        return Projector11(self.a * scale)

    def rescale_inplace(self, scale):
        self.a *= scale
        return self

    def opsquared(self):
        return Projector11(abs(self.a) ** 2)


class ProjectorZ0(mc.AbstractOperator):
    r"""ProjectorZ0()
    Alias for :class:`Projector0`
    """

    def __new__(self):
        return Projector0()


class ProjectorZ1(mc.AbstractOperator):
    r"""ProjectorZ1()
    Alias for :class:`Projector1`
    """

    def __new__(self):
        return Projector1()


__all__ = [
    "Projector0",
    "Projector00",
    "Projector01",
    "Projector1",
    "Projector10",
    "Projector11",
    "ProjectorY1",
    "ProjectorY0",
    "ProjectorX1",
    "ProjectorX0",
    "ProjectorZ1",
    "ProjectorZ0",
]
