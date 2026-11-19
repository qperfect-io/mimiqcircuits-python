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
pure: no randomness, no sampling. Per-trajectory random work belongs in
:meth:`LocalBackend.prepare_trajectory`.

Every compiled circuit carries a :class:`CompileMetadata`, read through
its ``metadata`` attribute, for compile-time information such as
measurement analysis and qubit permutations.

Concrete types:

- :class:`DefaultCompiledCircuit` wraps an artifact as it is, for
  backends whose compile step returns the final form directly.
- :class:`CompiledParametricCircuit` wraps a circuit that still has free
  symbolic parameters; :meth:`LocalBackend.bind` resolves them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CompileMetadata:
    """Compile-time information carried by a :class:`CompiledCircuit`.

    Backends fill in the fields that apply to them and leave the others at
    their defaults. The execute driver of :class:`LocalBackend` does not read
    them.

    Attributes:
        measures (list): Measurement-analysis entries, one per classical bit,
            classifying it as unused, read directly, read flipped, or
            constant 0 or 1. Empty when no analysis ran.
        active_qubits (list[int]): Qubits that must be simulated; the others
            are known to be in ``|0⟩`` when measured.
        qubit_flips (list[bool]): Per qubit, whether a trailing ``X`` gate was
            absorbed into the read-out.
        qubit_permutation (list[int] or None): Permutation applied by the
            pass pipeline, or ``None`` when the qubits keep their indices.
        measure_action_counts (dict[str, int]): Number of bits in each
            measurement-analysis class, for debugging.
        prefix_endpoint (int): Index of the last instruction in the
            deterministic prefix, when the backend splits the circuit into a
            deterministic prefix and a random suffix; ``0`` otherwise.

    Examples:
        >>> from mimiqcircuits import *
        >>> from mimiqcircuits.backends import CompileMetadata
        >>> meta = CompileMetadata(active_qubits=[0, 1])
        >>> meta.active_qubits, meta.qubit_permutation
        ([0, 1], None)
    """

    measures: list = field(default_factory=list)
    active_qubits: list[int] = field(default_factory=list)
    qubit_flips: list[bool] = field(default_factory=list)
    qubit_permutation: Optional[list[int]] = None
    measure_action_counts: dict[str, int] = field(default_factory=dict)
    prefix_endpoint: int = 0


class CompiledCircuit:
    """Base class for the backend-specific form of a circuit.

    Returned by :meth:`LocalBackend.compile` and run by
    :meth:`LocalBackend.evolve`. Backends subclass it to carry whatever they
    need: an MPO for an MPS simulator, a plain :class:`Circuit` for a
    state-vector simulator, and so on. Subclasses implement :attr:`metadata`
    and :attr:`source`; :class:`DefaultCompiledCircuit` does both and fits
    most backends.
    """

    @property
    def metadata(self) -> CompileMetadata:  # pragma: no cover — abstract
        """The :class:`CompileMetadata` of this compiled circuit."""
        raise NotImplementedError

    @property
    def source(self):  # pragma: no cover — abstract
        """The backend-specific artifact wrapped by this compiled circuit
        (for example a :class:`Circuit` or an MPO)."""
        raise NotImplementedError


@dataclass
class DefaultCompiledCircuit(CompiledCircuit):
    """Compiled circuit that wraps an artifact as it is.

    Use it when the backend's compile step has nothing to lower, for example
    a state-vector simulator that runs the input :class:`Circuit` directly.

    Args:
        _source: The artifact, returned by :attr:`source`.
        _metadata: The :class:`CompileMetadata`, returned by
            :attr:`metadata`. Defaults to an empty one.

    Examples:
        >>> from mimiqcircuits import *
        >>> from mimiqcircuits.backends import DefaultCompiledCircuit
        >>> c = Circuit().push(GateH(), 0)
        >>> compiled = DefaultCompiledCircuit(c)
        >>> compiled.source is c
        True
        >>> compiled.metadata.active_qubits
        []
    """

    _source: Any
    _metadata: CompileMetadata = field(default_factory=CompileMetadata)

    @property
    def metadata(self) -> CompileMetadata:
        """The :class:`CompileMetadata` given at construction."""
        return self._metadata

    @property
    def source(self):
        """The wrapped artifact."""
        return self._source


@dataclass
class CompiledParametricCircuit(CompiledCircuit):
    """Compiled circuit that still has free symbolic parameters.

    A backend that declares the ``"parametric"`` capability can return it
    from :meth:`LocalBackend.compile` for a symbolic circuit.
    :meth:`LocalBackend.bind` resolves the parameters.

    Args:
        _source: The symbolic :class:`Circuit`, kept so that
            :meth:`LocalBackend.bind` can evaluate it.
        _metadata: The :class:`CompileMetadata`. Defaults to an empty one.
    """

    _source: Any
    _metadata: CompileMetadata = field(default_factory=CompileMetadata)

    @property
    def metadata(self) -> CompileMetadata:
        """The :class:`CompileMetadata` given at construction."""
        return self._metadata

    @property
    def source(self):
        """The symbolic circuit."""
        return self._source


class UnboundSymbolicError(Exception):
    """Error for a circuit with free symbolic parameters on a backend that
    does not declare the ``"parametric"`` capability.

    Backends raise it from :meth:`LocalBackend.compile`.

    Args:
        backend_name: Name of the backend, kept as :attr:`backend_name`.
    """

    def __init__(self, backend_name: str):
        self.backend_name = backend_name
        super().__init__(
            f"UnboundSymbolicError: backend {backend_name} does not declare "
            "'parametric' — circuit contains free symbolic parameters"
        )
