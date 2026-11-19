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

from typing import Union
from mimiqcircuits.operations.operators.diagonals import DiagonalOp
from mimiqcircuits.operations.operators.projectors import Projector0
from mimiqcircuits.operations.operators.sigmas import SigmaMinus, SigmaPlus
import symengine as se
from mimiqcircuits._sympycheck import _is_basic
from mimiqcircuits.operations.krauschannel import krauschannel


class PhaseAmplitudeDamping(krauschannel):
    r"""One-qubit phase amplitude damping noise channel.

    The channel maps a density matrix :math:`\rho` to

    .. math::
        \mathcal{E}(\rho) =
        \begin{pmatrix}
            (1-\gamma)\rho_{00} + \gamma p & (1-2\beta)\sqrt{1-\gamma}\rho_{01} \\
            (1-2\beta)\sqrt{1-\gamma}\rho_{10} & (1-\gamma)\rho_{11} + (1-p)\gamma
        \end{pmatrix}

    with :math:`p, \gamma, \beta \in [0,1]`. It is the same channel as
    ``GeneralizedAmplitudeDamping(p, gamma)`` followed by ``PauliZ(beta)``.
    :meth:`krausmatrices` returns a Kraus representation.

    Args:
        p: Weight of the decay towards :math:`|0\rangle`, in :math:`[0, 1]`.
        gamma: Damping probability, in :math:`[0, 1]`.
        beta: Phase flip probability, in :math:`[0, 1]`.

    Any of the parameters may be symbolic; the range checks then apply only to
    the numeric ones.

    Raises:
        ValueError: If a numeric parameter is outside :math:`[0, 1]`.

    See Also:
        :class:`AmplitudeDamping`, :class:`GeneralizedAmplitudeDamping`,
        :class:`ThermalNoise`, :class:`PauliZ`

    Examples:
        >>> from mimiqcircuits import *
        >>> # p = 0.1, gamma = 0.2, beta = 0.3 on qubit 1
        >>> c = Circuit()
        >>> c.push(PhaseAmplitudeDamping(0.1, 0.2, 0.3), 1)
        2-qubit circuit with 1 instruction:
        └── PhaseAmplitudeDamping(0.1, 0.2, 0.3) @ q[1]
        <BLANKLINE>
    """

    _name = "PhaseAmplitudeDamping"
    _num_qubits = 1
    _parnames = ()

    def __init__(
        self, p: Union[float, int], gamma: Union[float, int], beta: Union[float, int]
    ):
        if not _is_basic(p) and not (0 <= p <= 1):
            raise ValueError("Value of p must be between 0 and 1.")
        if not _is_basic(gamma) and not (0 <= gamma <= 1):
            raise ValueError("Value of gamma must be between 0 and 1.")
        if not _is_basic(beta) and not (0 <= beta <= 1):
            raise ValueError("Value of beta must be between 0 and 1.")

        self.p = p
        self.gamma = gamma
        self.beta = beta
        super().__init__()
        self._parnames = ("p", "gamma", "beta")

    def evaluate(self, d={}):
        """Return a copy with symbols in the parameters substituted by values.

        Args:
            d (dict): Map from symbols to the values to substitute.

        Returns:
            PhaseAmplitudeDamping: A new channel; each parameter becomes a
            float when it is fully numeric.

        Raises:
            ValueError: If a substituted parameter is outside :math:`[0, 1]`.
        """
        attributes = ["p", "gamma", "beta"]
        evaluated_values = {}

        for attr in attributes:
            value = getattr(self, attr)
            if hasattr(value, "subs"):
                substituted_value = value.subs(d)
                evaluated_values[attr] = (
                    float(substituted_value.evalf())
                    if substituted_value.is_number
                    else substituted_value
                )
            else:
                evaluated_values[attr] = value

        # Extract evaluated values
        evaluated_p = evaluated_values["p"]
        evaluated_gamma = evaluated_values["gamma"]
        evaluated_beta = evaluated_values["beta"]

        # Perform checks only if values are numeric
        if isinstance(evaluated_p, (float, int)) and not (0 <= evaluated_p <= 1):
            raise ValueError("Value of `p` must be between 0 and 1 after evaluation.")
        if isinstance(evaluated_gamma, (float, int)) and not (
            0 <= evaluated_gamma <= 1
        ):
            raise ValueError(
                "Value of `gamma` must be between 0 and 1 after evaluation."
            )
        if isinstance(evaluated_beta, (float, int)) and not (0 <= evaluated_beta <= 1):
            raise ValueError(
                "Value of `beta` must be between 0 and 1 after evaluation."
            )

        # Return a new instance with evaluated parameters
        return PhaseAmplitudeDamping(evaluated_p, evaluated_gamma, evaluated_beta)

    def krausmatrices(self):
        return [se.Matrix(kraus.matrix().tolist()) for kraus in self.krausoperators()]

    def krausoperators(self):
        p = self.p
        gamma = self.gamma
        beta = self.beta
        # (1 - gamma) / (1 - gamma * p) is 1 at p == 1; dividing would give 0/0
        # once gamma rounds to 1, e.g. ThermalNoise with time >> T1.
        ratio = 1 if p == 1 else (1 - gamma) / (1 - gamma * p)
        pref1 = se.sqrt(1 - gamma * p)
        pref1K = (1 - 2 * beta) * se.sqrt(ratio)
        pref2 = se.sqrt(1 - gamma * (1 - p) - ratio * (1 - 2 * beta) ** 2)
        pref3 = se.sqrt(gamma * p)
        pref4 = se.sqrt(gamma * (1 - p))

        return [
            DiagonalOp(pref1K, pref1),
            Projector0(pref2),
            SigmaMinus(pref3),
            SigmaPlus(pref4),
        ]

    @property
    def parnames(self):
        return self._parnames

    def __str__(self):
        return f"{self._name}{self.p, self.gamma, self.beta}"


class ThermalNoise(krauschannel):
    r"""One-qubit thermal noise channel.

    The thermal noise channel is equivalent to the :class:`PhaseAmplitudeDamping` channel,
    but it is parametrized instead as:

    .. math::
        \mathcal{E}(\rho) =
        \begin{pmatrix}
            e^{-\Gamma_1 t}\rho_{00} + (1-n_e)(1-e^{-\Gamma_1 t}) & e^{-\Gamma_2 t}\rho_{01} \\
            e^{-\Gamma_2 t}\rho_{10} & e^{-\Gamma_1 t}\rho_{11} + n_e(1-e^{-\Gamma_1 t})
        \end{pmatrix}

    where :math:`\Gamma_1 = 1/T_1` and :math:`\Gamma_2 = 1/T_2`, and the parameters must fulfill
    :math:`T_1 > 0`, :math:`0 < T_2 \leq 2 T_1`, :math:`t \geq 0`, and :math:`0 \leq n_e \leq 1`.

    The parameters of the equivalent :class:`PhaseAmplitudeDamping` channel are
    :math:`p = 1-n_e`, :math:`\gamma = 1-e^{-\Gamma_1 t}`, and
    :math:`\beta = \frac{1}{2}(1-e^{-(\Gamma_2-\Gamma_1/2)t})`.

    Args:
        T1: Longitudinal relaxation time :math:`T_1`, greater than 0.
        T2: Transverse relaxation (dephasing) time :math:`T_2`, greater
            than 0 and at most :math:`2 T_1`.
        time: Duration :math:`t` over which the noise acts, at least 0.
        ne: Excited-state population :math:`n_e` at thermal equilibrium with
            the environment, in :math:`[0, 1]`.

    Any of the parameters may be symbolic; the checks then apply only to the
    numeric ones.

    Raises:
        ValueError: If a numeric parameter breaks one of the constraints above.

    See Also:
        :class:`PhaseAmplitudeDamping`, :class:`AmplitudeDamping`,
        :class:`GeneralizedAmplitudeDamping`

    Examples:
        >>> from mimiqcircuits import *
        >>> # T1 = 0.5, T2 = 0.6, t = 1.2, ne = 0.3 on qubit 1
        >>> c = Circuit()
        >>> c.push(ThermalNoise(0.5, 0.6, 1.2, 0.3), 1)
        2-qubit circuit with 1 instruction:
        └── ThermalNoise(0.5, 0.6, 1.2, 0.3) @ q[1]
        <BLANKLINE>
    """

    _name = "ThermalNoise"
    _num_qubits = 1
    _parnames = ()

    def __init__(
        self,
        T1: Union[float, int],
        T2: Union[float, int],
        time: Union[float, int],
        ne: Union[float, int],
    ):
        if not _is_basic(T1) and (T1 <= 0):
            raise ValueError("Value of T1 must be > 0.")
        if not _is_basic(T2) and (T2 <= 0):
            raise ValueError("Value of T2 must be > 0.")
        if not (_is_basic(T1) or _is_basic(T2)) and (T2 > 2 * T1):
            raise ValueError("Value of T2 must be <= 2 * T1.")
        if not _is_basic(time) and (time < 0):
            raise ValueError("Value of time must be >= 0.")
        if not _is_basic(ne) and not (0 <= ne <= 1):
            raise ValueError("Value of ne must be between 0 and 1.")

        self.T1 = T1
        self.T2 = T2
        self.time = time
        self.ne = ne
        super().__init__()
        self._parnames = ("T1", "T2", "time", "ne")

    def evaluate(self, d={}):
        """Return a copy with symbols in the parameters substituted by values.

        Args:
            d (dict): Map from symbols to the values to substitute.

        Returns:
            ThermalNoise: A new channel; each parameter becomes a float when it
            is fully numeric.

        Raises:
            ValueError: If a substituted parameter breaks the constraints of
                the class.
        """
        # List of attributes to evaluate
        attributes = ["T1", "T2", "time", "ne"]
        evaluated_values = {}

        # Evaluate each attribute
        for attr in attributes:
            value = getattr(self, attr)
            if hasattr(value, "subs"):
                # Substitute values using the dictionary and ensure substitution is applied
                substituted_value = value.subs(d)
                # Evaluate to a float if numeric, otherwise keep symbolic
                evaluated_values[attr] = (
                    float(substituted_value.evalf())
                    if substituted_value.is_number
                    else substituted_value
                )
            else:
                evaluated_values[attr] = value

        # Extract evaluated values
        evaluated_T1 = evaluated_values["T1"]
        evaluated_T2 = evaluated_values["T2"]
        evaluated_time = evaluated_values["time"]
        evaluated_ne = evaluated_values["ne"]

        # Perform checks only if values are numeric
        # Return a new instance with evaluated parameters
        return ThermalNoise(evaluated_T1, evaluated_T2, evaluated_time, evaluated_ne)

    def krausmatrices(self):
        return [se.Matrix(kraus.matrix().tolist()) for kraus in self.krausoperators()]

    def krausoperators(self):
        Gamma1 = 1 / self.T1
        Gamma2 = 1 / self.T2

        p = 1 - self.ne
        gamma = 1 - se.exp(-Gamma1 * self.time)
        beta = 0.5 * (1 - se.exp(-(Gamma2 - Gamma1 / 2) * self.time))

        pad = PhaseAmplitudeDamping(p, gamma, beta)
        return pad.krausoperators()

    @staticmethod
    def matrix(gate):
        return gate.matrix()

    @property
    def parnames(self):
        return self._parnames

    def __str__(self):
        return f"{self._name}{self.T1, self.T2, self.time, self.ne}"


__all__ = ["ThermalNoise", "PhaseAmplitudeDamping"]
