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
from typing import Union
import mimiqcircuits as mc
import symengine as se
from mimiqcircuits.operations.krauschannel import krauschannel
import sympy as sp


class PhaseAmplitudeDamping(krauschannel):
    r"""One-qubit phase amplitude damping noise channel.

    This channel is defined by:

    .. math::
        \mathcal{E}(\rho) =
        \begin{pmatrix}
            (1-\gamma)\rho_{00} + \gamma p & (1-2\beta)\sqrt{1-\gamma}\rho_{01} \\
            (1-2\beta)\sqrt{1-\gamma}\rho_{10} & (1-\gamma)\rho_{11} + (1-p)\gamma
        \end{pmatrix}

    Here, :math:`p, \gamma, \beta \in [0,1]`.

    This channel is equivalent to a `GeneralizedAmplitudeDamping(p, \gamma)` channel
    (see :class:`GeneralizedAmplitudeDamping`), followed by a `PauliZ(\beta)`
    channel (see :class:`PauliZ`).

    Use :func:`krausmatrices` to see a Kraus matrix representation of the channel.

    See Also:
        :class:`AmplitudeDamping`, :class:`GeneralizedAmplitudeDamping`,
        :class:`ThermalNoise`.

    Parameters:
        p (float): Probability parameter, must be in the range [0, 1].
        γ (float): Damping parameter, must be in the range [0, 1].
        β (float): Phase flip parameter, must be in the range [0, 1].

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(PhaseAmplitudeDamping(0.1, 0.2, 0.3), 1)
        2-qubit circuit with 1 instructions:
        └── PhaseAmplitudeDamping(0.1, 0.2, 0.3) @ q[1]
        <BLANKLINE>
    """

    _name = "PhaseAmplitudeDamping"
    _num_qubits = 1
    _parnames = ()

    def __init__(
        self, p: Union[float, int], gamma: Union[float, int], beta: Union[float, int]
    ):
        if not isinstance(p, (se.Basic, sp.Basic)) and not (0 <= p <= 1):
            raise ValueError("Value of p must be between 0 and 1.")
        if not isinstance(gamma, (se.Basic, sp.Basic)) and not (0 <= gamma <= 1):
            raise ValueError("Value of gamma must be between 0 and 1.")
        if not isinstance(beta, (se.Basic, sp.Basic)) and not (0 <= beta <= 1):
            raise ValueError("Value of beta must be between 0 and 1.")

        self.p = p
        self.gamma = gamma
        self.beta = beta
        super().__init__()
        self._parnames = ("p", "gamma", "beta")

    def evaluate(self, d={}):
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
        K = np.sqrt(1 - gamma) * (1 - 2 * beta) / (1 - gamma * p)
        pref1 = np.sqrt(1 - gamma * p)
        pref2 = np.sqrt(1 - gamma * (1 - p) - (1 - gamma * p) * K**2)
        pref3 = np.sqrt(gamma * p)
        pref4 = np.sqrt(gamma * (1 - p))

        return [
            mc.DiagonalOp(pref1 * K, pref1),
            mc.Projector0(pref2),
            mc.SigmaMinus(pref3),
            mc.SigmaPlus(pref4),
        ]

    @property
    def parnames(self):
        return self._parnames

    def __str__(self):
        return f"{self._name}{self.p,self.gamma,self.beta}"


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
    :math:`T_1 \geq 0`, :math:`T_2 \leq 2 T_1`, :math:`t \geq 0`, and :math:`0 \leq n_e \leq 1`.

    These parameters can be related to the ones used to define the :class:`PhaseAmplitudeDamping`
    channel through :math:`p = 1-n_e`, :math:`\gamma = 1-e^{-\Gamma_1 t}`, and
    :math:`\beta = \frac{1}{2}(1-e^{-(\Gamma_2-\Gamma_1/2)t})`.

    See Also:
        :class:`PhaseAmplitudeDamping`, :class:`AmplitudeDamping`,
        :class:`GeneralizedAmplitudeDamping`.

    Parameters:
        T₁ (float): Longitudinal relaxation rate, must be greater than or equal to 0.
        T₂ (float): Transversal relaxation rate, must be less than or equal to `2 * T₁`.
        t (float): Time duration of the gate, must be greater than or equal to 0.
        nₑ (float): Excitation fraction when in thermal equilibrium with the environment, must be in the range [0, 1].

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(ThermalNoise(0.5, 0.6, 1.2, 0.3), 1)
        2-qubit circuit with 1 instructions:
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
        if not isinstance(T1, (se.Basic, sp.Basic)) and (T1 < 0):
            raise ValueError("Value of T1 must be >= 0.")
        if not isinstance(T2, (se.Basic, sp.Basic)) and (T2 > 2 * T1):
            raise ValueError("Value of T2 must be <= 2 * T1.")
        if not isinstance(time, (se.Basic, sp.Basic)) and (time < 0):
            raise ValueError("Value of time must be >= 0.")
        if not isinstance(ne, (se.Basic, sp.Basic)) and not (0 <= ne <= 1):
            raise ValueError("Value of ne must be between 0 and 1.")

        self.T1 = T1
        self.T2 = T2
        self.time = time
        self.ne = ne
        super().__init__()
        self._parnames = ("T1", "T2", "time", "ne")

    def evaluate(self, d={}):
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
        if isinstance(evaluated_T1, (float, int)) and evaluated_T1 < 0:
            raise ValueError("Value of `T1` must be >= 0 after evaluation.")
        if isinstance(evaluated_T2, (float, int)) and evaluated_T2 > 2 * evaluated_T1:
            raise ValueError("Value of `T2` must be <= 2 * T1 after evaluation.")
        if isinstance(evaluated_time, (float, int)) and evaluated_time < 0:
            raise ValueError("Value of `time` must be >= 0 after evaluation.")
        if isinstance(evaluated_ne, (float, int)) and not (0 <= evaluated_ne <= 1):
            raise ValueError("Value of `ne` must be between 0 and 1 after evaluation.")

        # Return a new instance with evaluated parameters
        return ThermalNoise(evaluated_T1, evaluated_T2, evaluated_time, evaluated_ne)

    def krausmatrices(self):
        return [se.Matrix(kraus.matrix().tolist()) for kraus in self.krausoperators()]

    def krausoperators(self):
        Gamma1 = 1 / self.T1
        Gamma2 = 1 / self.T2

        p = 1 - self.ne
        gamma = 1 - np.exp(-Gamma1 * self.time)
        beta = 0.5 * (1 - np.exp(-(Gamma2 - Gamma1 / 2) * self.time))

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
