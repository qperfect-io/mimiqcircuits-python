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
import mimiqcircuits as mc
import numpy as np
import symengine as se
import sympy as sp


class AmplitudeDamping(krauschannel):
    r"""One-qubit amplitude damping noise channel.

    The amplitude damping channel is defined by the Kraus operators
    See Also:
        :func:`GeneralizedAmplitudeDamping`

    **Kraus Matrices representation:**

    .. math::
        \operatorname E_1 =
        \begin{pmatrix}
            1 & 0 \\
            0 & \sqrt{1-\gamma}
        \end{pmatrix}
        ,\quad
        \operatorname E_2 =
        \begin{pmatrix}
            0 & \sqrt{\gamma} \\
            0 & 0
        \end{pmatrix},

    Parameters:
        gamma: ``gamma in [0,1]``.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(AmplitudeDamping(0.1), 0)
        1-qubit circuit with 1 instructions:
        └── AmplitudeDamping(0.1) @ q[0]
        <BLANKLINE>
        
    """

    _num_qubits = 1
    _name = "AmplitudeDamping"
    _parnames = ()

    def __init__(self, gamma):
        if not isinstance(gamma, (sp.Basic, se.Basic)) and (gamma < 0 or gamma > 1):
            raise ValueError("Value of gamma must be between 0 and 1.")
        self.gamma = gamma
        self._parnames = ("gamma",)

    def evaluate(self, d={}):
        evaluated_gamma = (
            float(self.gamma.subs(d).evalf())
            if hasattr(self.gamma, "subs")
            else self.gamma
        )

        if not (0 <= evaluated_gamma <= 1):
            raise ValueError("Probability after evaluation must be between 0 and 1.")

        return AmplitudeDamping(evaluated_gamma)

    def krausoperators(self):
        E1 = mc.DiagonalOp(1, np.sqrt(1 - self.gamma))
        E2 = mc.SigmaMinus(np.sqrt(self.gamma))
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

    The amplitude generalized damping channel is defined by the Kraus operators.
    
    See Also:
        :func:`AmplitudeDamping`

    **Kraus Matrices representation:**

    .. math::
        \operatorname E_1 =
        \sqrt{p}
        \begin{pmatrix}
            1 & 0 \\
            0 & \sqrt{1-\gamma}
        \end{pmatrix}
        ,\quad
        \operatorname E_2 =
        \sqrt{p}
        \begin{pmatrix}
            0 & \sqrt{\gamma} \\
            0 & 0
        \end{pmatrix}
        ,\quad
        \operatorname E_3 =
        \sqrt{1-p}
        \begin{pmatrix}
            \sqrt{1-\gamma} & 0 \\
            0 & 1
        \end{pmatrix}
        ,\quad
        \operatorname E_4 =
        \sqrt{1-p}
        \begin{pmatrix}
            0 & 0 \\
            \sqrt{\gamma} & 0
        \end{pmatrix},

    Parameters:
        gamma: ``gamma in [0,1]``.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(GeneralizedAmplitudeDamping(0.1,1), 0)
        1-qubit circuit with 1 instructions:
        └── GeneralizedAmplitudeDamping((0.1, 1)) @ q[0]
        <BLANKLINE>
        
    """

    _num_qubits = 1
    _name = "GeneralizedAmplitudeDamping"
    _parnames = ()

    def __init__(self, p, gamma):
        if not isinstance(p, (sp.Basic, se.Basic)) and (p < 0 or p > 1):
            raise ValueError("Value of p must be between 0 and 1.")

        if not isinstance(gamma, (sp.Basic, se.Basic)) and (gamma < 0 or gamma > 1):
            raise ValueError("Value of gamma must be between 0 and 1.")

        self.p = p
        self.gamma = gamma
        self._parnames = ("p", "gamma")

    def evaluate(self, d={}):
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
        E1 = mc.DiagonalOp(np.sqrt(p), np.sqrt(p) * np.sqrt(1 - gamma))
        E2 = mc.DiagonalOp(np.sqrt(1 - p) * np.sqrt(1 - gamma), np.sqrt(1 - p))
        E3 = mc.SigmaMinus(np.sqrt(p) * np.sqrt(gamma))
        E4 = mc.SigmaPlus(np.sqrt(1 - p) * np.sqrt(gamma))
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
        return f"{self._name}({self.p,self.gamma})"


__all__ = ["GeneralizedAmplitudeDamping", "AmplitudeDamping"]
