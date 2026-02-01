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

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Union
from mimiqcircuits import Circuit, QCSResults
from symengine import Symbol
from collections import OrderedDict


DEFAULT_OPTIMIZER = "COBYLA"
DEFAULT_LABEL = "optexp_python"
DEFAULT_MAXITERS = None
DEFAULT_ZREGISTER = 0


def check_optimization_args(circuit: Circuit, initparams: Dict, zregister: int):
    if len(circuit) == 0:
        raise ValueError("The circuit must not be empty.")

    if not circuit.is_symbolic():
        raise ValueError("The circuit must be symbolic.")

    cvars = set(map(str, circuit.listvars()))
    pvars = set(map(str, initparams.keys()))

    if cvars != pvars:
        raise ValueError(
            "Initial values must be given for each symbolic variable in the circuit.\n"
            f"  expected: {sorted(cvars)}\n"
            f"  provided: {sorted(pvars)}"
        )

    if zregister < 0:
        raise ValueError("zregister must be ≥ 0")

    return True


@dataclass(repr=False)
class OptimizationExperiment:
    r"""
    Represents a variational quantum optimization experiment configuration.

    An `OptimizationExperiment` defines the symbolic quantum circuit, initial parameter
    values, classical optimization method, and metadata required to perform a variational
    parameter optimization run.

    .. math::
        \text{Optimize } \langle Z \rangle = \langle \psi(\vec{\theta}) | H | \psi(\vec{\theta}) \rangle

    where :math:`\vec{\theta}` are the symbolic parameters used in the circuit.

    Args:
        circuit (Circuit): The symbolic quantum circuit used to evaluate the cost function.
        initparams (Dict[str, float]): Dictionary of initial values for all symbolic parameters.
        optimizer (str): Optimization method to use. Must be one of:
            ``"BFGS"``, ``"LBFGS"``, ``"CG"``, ``"NELDERMEAD"``, ``"NEWTON"``,
            ``"COBYLA"``, ``"CMAES"``.
        label (str): Optional name or label for the experiment.
        maxiters (Optional[int]): Maximum number of optimization iterations (or `None`).
        zregister (int): Index of Z-register where cost values are accumulated.

    Raises:
        ValueError: If the circuit is not symbolic, is empty, the method is unsupported,
                    or if the parameter dictionary does not match circuit variables.

    Example:
        >>> from mimiqcircuits import *
        >>> from symengine import symbols
        >>> x, y = symbols("x y")
        >>> c = Circuit().push(GateRX(x), 0).push(GateRY(y),1)
        >>> exp = OptimizationExperiment(circuit = c, initparams = {x: 0.1, y: 2})
        >>> exp
        OptimizationExperiment:
        ├── optimizer: COBYLA
        ├── label: optexp_python
        ├── maxiters: None
        ├── zregister: 0
        └── initparams:
            ├── x => 0.1
            └── y => 2
    """

    circuit: Circuit
    initparams: OrderedDict[Union[str, Symbol], float] = field(
        default_factory=OrderedDict
    )
    optimizer: str = DEFAULT_OPTIMIZER
    label: str = DEFAULT_LABEL
    maxiters: Optional[int] = DEFAULT_MAXITERS
    zregister: int = DEFAULT_ZREGISTER

    def __post_init__(self):
        check_optimization_args(self.circuit, self.initparams, self.zregister)

    def is_valid(self):
        try:
            check_optimization_args(self.circuit, self.initparams, self.zregister)
            return True
        except Exception:
            return False

    def num_params(self):
        return len(self.initparams)

    def num_qubits(self):
        return self.circuit.num_qubits()

    def num_bits(self):
        return self.circuit.num_bits()

    def num_zvars(self):
        return self.circuit.num_zvars()

    def get_param(self, n):
        return self.initparams[n]

    def get_params(self):
        return self.initparams

    def change_parameters(self, newvals: Dict[Symbol, float]):
        updated = {k: newvals.get(k, v) for k, v in self.initparams.items()}
        return OptimizationExperiment(
            circuit=self.circuit,
            initparams=updated,
            optimizer=self.optimizer,
            label=self.label,
            maxiters=self.maxiters,
            zregister=self.zregister,
        )

    def change_list_of_parameters(self, theta: List[float]):
        vars_ = sorted(self.circuit.listvars(), key=str)
        if len(vars_) != len(theta):
            raise ValueError(
                f"Length of value list does not match listvars. Expected {len(vars_)}, got {len(theta)}"
            )
        parammap = {v: theta[i] for i, v in enumerate(vars_)}
        return self.change_parameters(parammap)

    def __str__(self):
        lines = [
            "OptimizationExperiment:",
            f"├── optimizer: {self.optimizer}",
            f"├── label: {self.label}",
            f"├── maxiters: {self.maxiters}",
            f"├── zregister: {self.zregister}",
            "└── initparams:",
        ]
        items = list(self.initparams.items())
        for i, (k, v) in enumerate(items):
            prefix = "    └── " if i == len(items) - 1 else "    ├── "
            lines.append(f"{prefix}{k} => {v}")
        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()

    def saveproto(self, file):
        from mimiqcircuits.proto.protoio import saveproto

        return saveproto(self, file)

    @staticmethod
    def loadproto(file):
        from mimiqcircuits.proto.protoio import loadproto

        return loadproto(file, OptimizationExperiment)


@dataclass(repr=False)
class OptimizationRun:
    r"""
    Represents a single evaluation of the cost function during a variational optimization run.

    An `OptimizationRun` stores the final cost value for a set of parameters,
    the corresponding parameter values, and the raw execution results from the quantum circuit simulation.

    Args:
        cost (float): Final cost (objective) value for the evaluated parameters.
        parameters (Dict[str, float]): Parameter values at this optimization step.
        results (QCSResults): Raw execution results from the quantum simulator.

    Example:
        >>> from mimiqcircuits import QCSResults
        >>> OptimizationRun(cost=0.153, parameters={"x": 0.2, "y": 1.5}, results=QCSResults("",""))
        OptimizationRun:
        ├── cost: 0.153
        ├── parameters:
        │   x => 0.2
        │   y => 1.5
        └── results: QCSResults(...)
    """

    cost: float = 0.0
    parameters: Dict[str, float] = field(default_factory=dict)
    results: QCSResults = field(default_factory=QCSResults)

    def __str__(self):
        lines = ["OptimizationRun:", f"├── cost: {self.cost}", "├── parameters:"]
        for k, v in self.parameters.items():
            lines.append(f"│   {k} => {v}")

        # Avoid broken __repr__ of QCSResults
        lines.append(f"└── results: QCSResults(...)")

        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()

    def get_cost(self):
        return self.cost

    def get_params(self):
        return self.parameters

    def get_param(self, n):
        return self.parameters[n]

    def get_resultofbest(self):
        return self.results

    def saveproto(self, file):
        from mimiqcircuits.proto.protoio import saveproto

        return saveproto(self, file)

    @staticmethod
    def loadproto(file):
        from mimiqcircuits.proto.protoio import loadproto

        return loadproto(file, OptimizationRun)


@dataclass(repr=False)
class OptimizationResults:
    r"""
    Container for storing the best optimization result and the full history of optimization runs.

    An `OptimizationResults` object tracks the full sequence of cost evaluations and
    identifies the best-performing parameter set from the optimization process.

    Args:
        best (OptimizationRun): The best optimization run found.
        history (List[OptimizationRun]): All optimization runs evaluated during the process.

    Example:
        >>> from mimiqcircuits import OptimizationResults, OptimizationRun, QCSResults
        >>> run1 = OptimizationRun(cost=0.5, parameters={"x": 0.1}, results=QCSResults("", ""))
        >>> run1
        OptimizationRun:
        ├── cost: 0.5
        ├── parameters:
        │   x => 0.1
        └── results: QCSResults(...)
        >>> run2 = OptimizationRun(cost=0.3, parameters={"x": 0.2}, results=QCSResults())
        >>> run2
        OptimizationRun:
        ├── cost: 0.3
        ├── parameters:
        │   x => 0.2
        └── results: QCSResults(...)
        >>> results = OptimizationResults(best=run2, history=[run1, run2])
        >>> results
        OptimizationResults:
        ├── Best Run:
        │   OptimizationRun:
        │   ├── cost: 0.3
        │   ├── parameters:
        │   │   x => 0.2
        │   └── results: QCSResults(...)
        └── History (2 runs):
            ├── Run 0:
            │   OptimizationRun:
            │   ├── cost: 0.5
            │   ├── parameters:
            │   │   x => 0.1
            │   └── results: QCSResults(...)
            └── Run 1:
                OptimizationRun:
                ├── cost: 0.3
                ├── parameters:
                │   x => 0.2
                └── results: QCSResults(...)
    """

    best: OptimizationRun = field(default_factory=OptimizationRun)
    history: List[OptimizationRun] = field(default_factory=list)

    def __str__(self):
        lines = ["OptimizationResults:"]
        lines.append("├── Best Run:")

        best_lines = str(self.best).split("\n")
        for line in best_lines:
            lines.append("│   " + line)

        lines.append(f"└── History ({len(self.history)} runs):")
        for i, run in enumerate(self.history):
            is_last = i == len(self.history) - 1
            connector = "    └── " if is_last else "    ├── "
            lines.append(f"{connector}Run {i}:")
            run_lines = str(run).split("\n")
            for rline in run_lines:
                prefix = "        " if is_last else "    │   "
                lines.append(prefix + rline)

        return "\n".join(lines)

    def get_best(self):
        return self.best

    def get_resultofbest(self):
        return self.best.results

    def get_resultsofhistory(self):
        return [h.results for h in self.history]

    def __repr__(self):
        return self.__str__()

    def saveproto(self, file):
        from mimiqcircuits.proto.protoio import saveproto

        return saveproto(self, file)

    @staticmethod
    def loadproto(file):
        from mimiqcircuits.proto.protoio import loadproto

        return loadproto(file, OptimizationResults)


__all__ = ["OptimizationExperiment", "OptimizationResults", "OptimizationRun"]
