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
from mimiqcircuits.circuit import Circuit
from mimiqcircuits.qcsresults import QCSResults
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
    r"""A variational optimization problem: a symbolic circuit, initial
    values for its symbols, and the optimizer settings.

    The cost of a set of values is the real part of the mean of z-register
    ``zregister`` over the samples of the circuit evaluated with them. Push
    the quantity to minimize into that register, for example with
    :meth:`~mimiqcircuits.Circuit.push_expval`. Run the experiment with
    :meth:`LocalBackend.optimize <mimiqcircuits.backends.LocalBackend.optimize>`,
    :meth:`MimiqRemoteBackend.optimize
    <mimiqcircuits.backends.MimiqRemoteBackend.optimize>` or
    ``MimiqConnection.optimize``.

    Args:
        circuit (Circuit): The symbolic circuit.
        initparams (dict): Initial value of every symbol of the circuit,
            keyed by symbol or by symbol name.
        optimizer (str): ``"BFGS"``, ``"LBFGS"``, ``"CG"``, ``"NELDERMEAD"``,
            ``"NEWTON"``, ``"COBYLA"`` or ``"CMAES"``. Checked when the
            experiment runs, not here. Defaults to ``"COBYLA"``.
        label (str): Name of the experiment. Defaults to
            ``"optexp_python"``.
        maxiters (int or None): Iteration limit, or ``None`` for the
            optimizer's own.
        zregister (int): Index of the z-register that holds the cost.
            Defaults to 0.

    Raises:
        ValueError: If the circuit is empty or has no symbols,
            ``initparams`` does not give a value for exactly the symbols of
            the circuit, or ``zregister`` is negative.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import symbols

        The cost is ``⟨Z⟩`` of qubit 0 after two rotations, written to
        z-register 0.

        >>> x, y = symbols("x y")
        >>> c = Circuit().push(GateRX(x), 0).push(GateRY(y), 0)
        >>> c.push(ExpectationValue(GateZ()), 0, 0)
        1-qubit, 1-zvar circuit with 3 instructions:
        ├── RX(x) @ q[0]
        ├── RY(y) @ q[0]
        └── ⟨Z⟩ @ q[0], z[0]
        <BLANKLINE>

        Every symbol needs an initial value.

        >>> ex = OptimizationExperiment(c, {x: 0.1, y: 2.0}, maxiters=100)
        >>> ex
        OptimizationExperiment:
        ├── optimizer: COBYLA
        ├── label: optexp_python
        ├── maxiters: 100
        ├── zregister: 0
        └── initparams:
            ├── x => 0.1
            └── y => 2.0

        Changing the initial values gives a new experiment and leaves this
        one unchanged.

        >>> ex.change_parameters({x: 0.5}).get_params()
        {x: 0.5, y: 2.0}

    See Also:
        :meth:`LocalBackend.optimize
        <mimiqcircuits.backends.LocalBackend.optimize>`, whose example runs
        an experiment.
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
        """Return whether the experiment passes the checks of the
        constructor."""
        try:
            check_optimization_args(self.circuit, self.initparams, self.zregister)
            return True
        except Exception:
            return False

    def num_params(self):
        """Return the number of symbols with an initial value."""
        return len(self.initparams)

    def num_qubits(self):
        """Return the number of qubits of the circuit."""
        return self.circuit.num_qubits()

    def num_bits(self):
        """Return the number of classical bits of the circuit."""
        return self.circuit.num_bits()

    def num_zvars(self):
        """Return the number of z-variables of the circuit."""
        return self.circuit.num_zvars()

    def get_param(self, n):
        """Return the initial value of ``n``, a key of ``initparams``."""
        return self.initparams[n]

    def get_params(self):
        """Return the ``initparams`` dictionary itself, not a copy."""
        return self.initparams

    def change_parameters(self, newvals: Dict[Symbol, float]):
        """Return a copy of the experiment with some initial values replaced.

        Args:
            newvals: New values, keyed like ``initparams``. Keys that are not
                in ``initparams`` are ignored.

        Returns:
            A new :class:`OptimizationExperiment`; this one is unchanged.
        """
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
        """Return a copy of the experiment with all initial values replaced.

        Args:
            theta: One value per symbol of the circuit, in the order of the
                symbol names sorted alphabetically.

        Returns:
            A new :class:`OptimizationExperiment`.

        Raises:
            ValueError: If ``theta`` does not have one value per symbol.
        """
        vars_ = sorted(self.circuit.listvars(), key=str)
        if len(vars_) != len(theta):
            raise ValueError(
                f"Length of value list does not match listvars. Expected {len(vars_)}, got {len(theta)}"
            )
        # Match the key type of initparams, which may be symbols or names.
        parammap = {
            (v if v in self.initparams else str(v)): t for v, t in zip(vars_, theta)
        }
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
        """Save the experiment to a Protobuf file or file-like object.

        Returns the number of bytes written. See
        :func:`~mimiqcircuits.proto.protoio.saveproto`.
        """
        from mimiqcircuits.proto.protoio import saveproto

        return saveproto(self, file)

    @staticmethod
    def loadproto(file):
        """Load an :class:`OptimizationExperiment` saved with
        :meth:`saveproto`."""
        from mimiqcircuits.proto.protoio import loadproto

        return loadproto(file, OptimizationExperiment)


@dataclass(repr=False)
class OptimizationRun:
    r"""One evaluation of the cost during an optimization: the parameter
    values, the cost they gave, and the results of the execution.

    Args:
        cost (float): The cost at these values.
        parameters (dict): The value of each symbol, keyed like the
            ``initparams`` of the experiment.
        results (QCSResults): The results of the execution that gave the
            cost.

    Examples:
        >>> from mimiqcircuits import *
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
        """Return the cost."""
        return self.cost

    def get_params(self):
        """Return the ``parameters`` dictionary itself, not a copy."""
        return self.parameters

    def get_param(self, n):
        """Return the value of ``n``, a key of ``parameters``."""
        return self.parameters[n]

    def get_resultofbest(self):
        """Return the :class:`~mimiqcircuits.QCSResults` of this run."""
        return self.results

    def saveproto(self, file):
        """Save the run to a Protobuf file or file-like object.

        Returns the number of bytes written. See
        :func:`~mimiqcircuits.proto.protoio.saveproto`.
        """
        from mimiqcircuits.proto.protoio import saveproto

        return saveproto(self, file)

    @staticmethod
    def loadproto(file):
        """Load an :class:`OptimizationRun` saved with :meth:`saveproto`."""
        from mimiqcircuits.proto.protoio import loadproto

        return loadproto(file, OptimizationRun)


@dataclass(repr=False)
class OptimizationResults:
    r"""The result of an optimization: the lowest-cost run and, when asked
    for, every evaluated run.

    Returned by the ``optimize`` method of a backend, and by ``get_result``
    or ``get_results`` for a job started with a connection's ``optimize``.
    ``history`` is filled only with ``return_history=True`` (local) or
    ``history=True`` (connection).

    Args:
        best (OptimizationRun): The run with the lowest cost.
        history (list[OptimizationRun]): Every evaluated run, in order.

    Examples:
        >>> from mimiqcircuits import *
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
        """Return the lowest-cost :class:`OptimizationRun`."""
        return self.best

    def get_resultofbest(self):
        """Return the :class:`~mimiqcircuits.QCSResults` of the best run."""
        return self.best.results

    def get_resultsofhistory(self):
        """Return the :class:`~mimiqcircuits.QCSResults` of every run in the
        history, in order."""
        return [h.results for h in self.history]

    def __repr__(self):
        return self.__str__()

    def saveproto(self, file):
        """Save the results to a Protobuf file or file-like object.

        Returns the number of bytes written. See
        :func:`~mimiqcircuits.proto.protoio.saveproto`.
        """
        from mimiqcircuits.proto.protoio import saveproto

        return saveproto(self, file)

    @staticmethod
    def loadproto(file):
        """Load an :class:`OptimizationResults` saved with
        :meth:`saveproto`."""
        from mimiqcircuits.proto.protoio import loadproto

        return loadproto(file, OptimizationResults)


__all__ = ["OptimizationExperiment", "OptimizationResults", "OptimizationRun"]
