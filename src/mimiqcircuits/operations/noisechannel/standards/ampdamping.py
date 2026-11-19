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


from mimiqcircuits.operations.krauschannel import krauschannel
from mimiqcircuits.operations.operators.diagonals import DiagonalOp
from mimiqcircuits.operations.operators.sigmas import SigmaMinus, SigmaPlus
import symengine as se
from mimiqcircuits._sympycheck import _is_basic


class AmplitudeDamping(krauschannel):
    r"""One-qubit amplitude damping noise channel.

    Models energy relaxation: the excited state :math:`|1\rangle` decays to
    :math:`|0\rangle` with probability :math:`\gamma`. The Kraus operators are

    .. math::
        E_1 =
        \begin{pmatrix}
            1 & 0 \\
            0 & \sqrt{1-\gamma}
        \end{pmatrix}
        ,\quad
        E_2 =
        \begin{pmatrix}
            0 & \sqrt{\gamma} \\
            0 & 0
        \end{pmatrix}.

    Args:
        gamma (float): Damping probability, in :math:`[0, 1]`; may be symbolic.

    Raises:
        ValueError: If a numeric ``gamma`` is outside :math:`[0, 1]`.

    See Also:
        :class:`GeneralizedAmplitudeDamping`, :class:`PhaseAmplitudeDamping`

    Examples:
        >>> from mimiqcircuits import *
        >>> # |1> decays to |0> with probability 0.1
        >>> c = Circuit()
        >>> c.push(AmplitudeDamping(0.1), 0)
        1-qubit circuit with 1 instruction:
        └── AmplitudeDamping(0.1) @ q[0]
        <BLANKLINE>
    """

    _num_qubits = 1
    _name = "AmplitudeDamping"
    _parnames = ()

    def __init__(self, gamma):
        if not _is_basic(gamma) and (gamma < 0 or gamma > 1):
            raise ValueError("Value of gamma must be between 0 and 1.")
        self.gamma = gamma
        self._parnames = ("gamma",)

    def evaluate(self, d={}):
        """Return a copy with symbols in ``gamma`` substituted by values.

        Args:
            d (dict): Map from symbols to the values to substitute.

        Returns:
            AmplitudeDamping: A new channel; ``gamma`` becomes a float when it
            is fully numeric.

        Raises:
            ValueError: If the substituted ``gamma`` is outside :math:`[0, 1]`.
        """
        evaluated_gamma = self.gamma
        if hasattr(evaluated_gamma, "subs"):
            evaluated_gamma = evaluated_gamma.subs(d)
            if evaluated_gamma.is_number:
                evaluated_gamma = float(evaluated_gamma.evalf())

        if isinstance(evaluated_gamma, (float, int)) and not (0 <= evaluated_gamma <= 1):
            raise ValueError("Probability after evaluation must be between 0 and 1.")

        return AmplitudeDamping(evaluated_gamma)

    def krausoperators(self):
        E1 = DiagonalOp(1, se.sqrt(1 - self.gamma))
        E2 = SigmaMinus(se.sqrt(self.gamma))
        return [E1, E2]

    def krausmatrices(self):
        return [operator.matrix() for operator in self.krausoperators()]

    def iswrapper(self):
        pass

    @property
    def opname(self):
        return self._name

    @property
    def parnames(self):
        return self._parnames

    def __str__(self):
        return f"{self._name}({self.gamma})"


class GeneralizedAmplitudeDamping(krauschannel):
    r"""One-qubit generalized amplitude damping noise channel.

    Amplitude damping towards a thermal state instead of :math:`|0\rangle`:
    with weight :math:`p` the qubit decays from :math:`|1\rangle` to
    :math:`|0\rangle`, with weight :math:`1-p` it is excited from
    :math:`|0\rangle` to :math:`|1\rangle`, in both cases with probability
    :math:`\gamma`. The Kraus operators are

    .. math::
        E_1 =
        \sqrt{p}
        \begin{pmatrix}
            1 & 0 \\
            0 & \sqrt{1-\gamma}
        \end{pmatrix}
        ,\quad
        E_2 =
        \sqrt{1-p}
        \begin{pmatrix}
            \sqrt{1-\gamma} & 0 \\
            0 & 1
        \end{pmatrix}
        ,\quad
        E_3 =
        \sqrt{p}
        \begin{pmatrix}
            0 & \sqrt{\gamma} \\
            0 & 0
        \end{pmatrix}
        ,\quad
        E_4 =
        \sqrt{1-p}
        \begin{pmatrix}
            0 & 0 \\
            \sqrt{\gamma} & 0
        \end{pmatrix}.

    For :math:`p = 1` it reduces to :class:`AmplitudeDamping`.

    Args:
        p (float): Weight of the decay towards :math:`|0\rangle`, in
            :math:`[0, 1]`; may be symbolic.
        gamma (float): Damping probability, in :math:`[0, 1]`; may be symbolic.

    Raises:
        ValueError: If a numeric ``p`` or ``gamma`` is outside :math:`[0, 1]`.

    See Also:
        :class:`AmplitudeDamping`, :class:`PhaseAmplitudeDamping`

    Examples:
        >>> from mimiqcircuits import *
        >>> # p = 1: pure decay to |0>, same as AmplitudeDamping(0.1)
        >>> c = Circuit()
        >>> c.push(GeneralizedAmplitudeDamping(1, 0.1), 0)
        1-qubit circuit with 1 instruction:
        └── GeneralizedAmplitudeDamping((1, 0.1)) @ q[0]
        <BLANKLINE>
    """

    _num_qubits = 1
    _name = "GeneralizedAmplitudeDamping"
    _parnames = ()

    def __init__(self, p, gamma):
        if not _is_basic(p) and (p < 0 or p > 1):
            raise ValueError("Value of p must be between 0 and 1.")

        if not _is_basic(gamma) and (gamma < 0 or gamma > 1):
            raise ValueError("Value of gamma must be between 0 and 1.")

        self.p = p
        self.gamma = gamma
        self._parnames = ("p", "gamma")

    def evaluate(self, d={}):
        """Return a copy with symbols in ``p`` and ``gamma`` substituted by values.

        Args:
            d (dict): Map from symbols to the values to substitute.

        Returns:
            GeneralizedAmplitudeDamping: A new channel; each parameter becomes
            a float when it is fully numeric.

        Raises:
            ValueError: If a substituted parameter is outside :math:`[0, 1]`.
        """
        # Helper function to evaluate a parameter
        def evaluate_param(param):
            if hasattr(param, "subs"):
                substituted_value = param.subs(d)
                return (
                    float(substituted_value.evalf())
                    if substituted_value.is_number
                    else substituted_value
                )
            return param

        evaluated_p = evaluate_param(self.p)
        evaluated_gamma = evaluate_param(self.gamma)

        if isinstance(evaluated_p, (float, int)) and not (0 <= evaluated_p <= 1):
            raise ValueError("Probability p after evaluation must be between 0 and 1.")
        if isinstance(evaluated_gamma, (float, int)) and not (
            0 <= evaluated_gamma <= 1
        ):
            raise ValueError(
                "Probability gamma after evaluation must be between 0 and 1."
            )

        # Return a new instance of GeneralizedAmplitudeDamping with the evaluated values
        return GeneralizedAmplitudeDamping(evaluated_p, evaluated_gamma)

    def krausoperators(self):
        p = self.p
        gamma = self.gamma
        E1 = DiagonalOp(se.sqrt(p), se.sqrt(p) * se.sqrt(1 - gamma))
        E2 = DiagonalOp(se.sqrt(1 - p) * se.sqrt(1 - gamma), se.sqrt(1 - p))
        E3 = SigmaMinus(se.sqrt(p) * se.sqrt(gamma))
        E4 = SigmaPlus(se.sqrt(1 - p) * se.sqrt(gamma))
        return [E1, E2, E3, E4]

    def krausmatrices(self):
        return [operator.matrix() for operator in self.krausoperators()]

    def iswrapper(self):
        pass

    @property
    def num_qubits(self):
        return self._num_qubits

    @property
    def parnames(self):
        return self._parnames

    def __str__(self):
        return f"{self._name}({self.p, self.gamma})"


__all__ = ["GeneralizedAmplitudeDamping", "AmplitudeDamping"]
