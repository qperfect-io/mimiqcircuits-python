#
# Copyright © 2023-2026 QPerfect. All Rights Reserved.
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

"""Compiled-circuit types and the metadata they carry.

:meth:`LocalBackend.compile` turns a :class:`Circuit` into a
backend-specific :class:`CompiledCircuit`. The compile step must be
pure: no RNG, no sampling. Stochastic per-trajectory work belongs in
:meth:`LocalBackend.prepare_trajectory`.

Every compiled circuit carries a :class:`CompileMetadata` (read via
its ``metadata`` attribute) so generic drivers can read measurement
analysis, qubit permutations, and other compile-time side effects
without unpacking a tuple.

Concrete types:

- :class:`DefaultCompiledCircuit` — generic passthrough wrap when
  the backend's lowering already returns its final compiled form.
- :class:`CompiledParametricCircuit` — compiled artifact that still
  holds free symbolic parameters; resolve them via
  :meth:`LocalBackend.bind`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CompileMetadata:
    """Compile-time side effects carried alongside a :class:`CompiledCircuit`.

    Attributes
    ----------
    measures : list
        Measurement-analysis entries (classifies each classical bit
        as Unused / Direct / Flip / Const0 / Const1). Empty when
        analysis has not run.
    active_qubits : list[int]
        Qubit indices that must be simulated; the others can be
        dropped because they are provably ``|0⟩`` at measurement.
    qubit_flips : list[bool]
        Per-qubit final NOT flag absorbed from trailing
        single-qubit X gates.
    qubit_permutation : list[int] or None
        Composed permutation applied by the pass pipeline.
        ``None`` when the pipeline is identity on qubit indices.
    measure_action_counts : dict[str, int]
        Frequency of each measurement-action branch (debug aid).
    prefix_endpoint : int
        Index of the last instruction folded into the deterministic
        prefix when the backend splits prefix from stochastic
        suffix; ``0`` when no split applies.
    """

    measures: list = field(default_factory=list)
    active_qubits: list[int] = field(default_factory=list)
    qubit_flips: list[bool] = field(default_factory=list)
    qubit_permutation: Optional[list[int]] = None
    measure_action_counts: dict[str, int] = field(default_factory=dict)
    prefix_endpoint: int = 0


class CompiledCircuit:
    """Backend-specific deterministic representation of a circuit.

    Returned by :meth:`LocalBackend.compile`. Concrete backends
    subclass to carry whatever payload they need — an MPO for an MPS
    simulator, a plain :class:`Circuit` for a state-vector backend,
    a streaming thunk for an out-of-core simulator, …

    Every subclass exposes two attributes:

    - :attr:`metadata` — a :class:`CompileMetadata` describing
      compile-time side effects;
    - :attr:`source` — the wrapped, backend-specific artifact.
    """

    @property
    def metadata(self) -> CompileMetadata:  # pragma: no cover — abstract
        raise NotImplementedError

    @property
    def source(self):  # pragma: no cover — abstract
        """The backend-specific artifact wrapped by this compiled
        circuit (e.g. a :class:`Circuit`, an MPO, a streaming
        thunk)."""
        raise NotImplementedError


@dataclass
class DefaultCompiledCircuit(CompiledCircuit):
    """Generic passthrough wrapper.

    Use this when your backend's ``compile`` step is effectively the
    identity — for example a state-vector simulator that consumes
    the input :class:`Circuit` directly.
    """

    _source: Any
    _metadata: CompileMetadata = field(default_factory=CompileMetadata)

    @property
    def metadata(self) -> CompileMetadata:
        return self._metadata

    @property
    def source(self):
        return self._source


@dataclass
class CompiledParametricCircuit(CompiledCircuit):
    """Compiled artifact that still carries free symbolic parameters.

    Produced by :meth:`LocalBackend.compile` when the input is
    symbolic and the backend advertises the ``"parametric"``
    capability. Resolve the parameters by calling
    :meth:`LocalBackend.bind`.
    """

    _source: Any  # the original symbolic Circuit, retained for re-evaluation
    _metadata: CompileMetadata = field(default_factory=CompileMetadata)

    @property
    def metadata(self) -> CompileMetadata:
        return self._metadata

    @property
    def source(self):
        return self._source


class UnboundSymbolicError(Exception):
    """Raised when :meth:`LocalBackend.compile` is handed a circuit
    that still has free symbolic parameters but the backend does
    not advertise the ``"parametric"`` capability.
    """

    def __init__(self, backend_name: str):
        self.backend_name = backend_name
        super().__init__(
            f"UnboundSymbolicError: backend {backend_name} does not declare "
            "'parametric' — circuit contains free symbolic parameters"
        )
