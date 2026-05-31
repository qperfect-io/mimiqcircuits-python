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

"""
mimiqcircuits.backends — abstract Backend interface for quantum simulators.

Mirrors the Julia `AbstractQCSs.jl` surface so simulators plug in
uniformly across both languages.

Module map:

    capabilities, limits, topology  -> .capabilities
    Backend, LocalBackend, RemoteBackend  -> .backend
    Fidelity ADT  -> .fidelity
    CompiledCircuit subclasses + bind  -> .compiled
    AbstractPass + PassPipeline  -> .passes
    Concrete passes (RemoveSwapsPass, …)  -> .concrete_passes

Recommended import:

    from mimiqcircuits.backends import Backend, LocalBackend, ...
"""

from mimiqcircuits.backends.capabilities import (
    Capability,
    CAPABILITY_VOCABULARY,
    Limits,
    Topology,
    AllToAll,
    CouplingMap,
    LinearChain,
    AdmissionResult,
    Admissible,
    Marginal,
    Inadmissible,
    is_admissible,
    UnsupportedCapabilityError,
    CAP_PROBES,
    register_cap_probe,
)
# Side-effect import: registers the default probe-circuit constructors
# into CAP_PROBES so the anti-capability conformance harness has
# probes out of the box. Imported after capabilities so the registry
# is available; consumed by `tests/conftest.py` helpers downstream.
from mimiqcircuits.backends import _default_cap_probes  # noqa: F401
from mimiqcircuits.backends.fidelity import (
    Fidelity,
    ExactFidelity,
    UnknownFidelity,
    TruncationLowerBound,
    LowerBoundPerStep,
    EstimatedFidelity,
    as_lower_bound,
    as_expected,
    as_interval,
)
from mimiqcircuits.backends.compiled import (
    CompiledCircuit,
    DefaultCompiledCircuit,
    CompiledParametricCircuit,
    CompileMetadata,
    UnboundSymbolicError,
)
from mimiqcircuits.backends.passes import (
    PassParam,
    PStr,
    PInt,
    PFloat,
    PBool,
    PSym,
    PList,
    PDict,
    to_pass_param,
    PassSpec,
    PassResult,
    PassContext,
    AbstractPass,
    PassPipeline,
    apply_passes,
    invert_perm,
    UnacceptedPassError,
    RemotePassOrderError,
)
from mimiqcircuits.backends.measure_analysis import (
    extract_projection,
    evaluate_projection,
    needs_trajectories,
    needs_loss_sampling,
    any_mixed_unitary,
    remap_projection_qubits,
)
from mimiqcircuits.backends.stochastic_kind import (
    StochasticKind,
    default_stochastic_kind,
    stochastic_kind,
    is_deterministic,
    is_trajectory_sampleable,
    is_runtime_only,
    is_stochastic,
    first_stochastic,
    last_stochastic,
    first_runtime_only,
    last_runtime_only,
)
from mimiqcircuits.backends.backend import (
    Backend,
    LocalBackend,
    RemoteBackend,
    State,
    RNGs,
)
from mimiqcircuits.backends.progress import (
    Progress,
    Stage,
    NoProgress,
    TqdmProgress,
    to_progress,
)
from mimiqcircuits.backends.remote import (
    MimiqRemoteBackend,
)
from mimiqcircuits.backends._rng_utils import (
    normalize_seed,
    derive_grid_seeds,
)

__all__ = [
    # capabilities
    "Capability",
    "CAPABILITY_VOCABULARY",
    "Limits",
    "Topology",
    "AllToAll",
    "CouplingMap",
    "LinearChain",
    "AdmissionResult",
    "Admissible",
    "Marginal",
    "Inadmissible",
    "is_admissible",
    "UnsupportedCapabilityError",
    "CAP_PROBES",
    "register_cap_probe",
    # fidelity
    "Fidelity",
    "ExactFidelity",
    "UnknownFidelity",
    "TruncationLowerBound",
    "LowerBoundPerStep",
    "EstimatedFidelity",
    "as_lower_bound",
    "as_expected",
    "as_interval",
    # compiled
    "CompiledCircuit",
    "DefaultCompiledCircuit",
    "CompiledParametricCircuit",
    "CompileMetadata",
    "UnboundSymbolicError",
    # passes
    "PassParam",
    "PStr",
    "PInt",
    "PFloat",
    "PBool",
    "PSym",
    "PList",
    "PDict",
    "to_pass_param",
    "PassSpec",
    "PassResult",
    "PassContext",
    "AbstractPass",
    "PassPipeline",
    "apply_passes",
    "invert_perm",
    "UnacceptedPassError",
    "RemotePassOrderError",
    # backend
    "Backend",
    "LocalBackend",
    "RemoteBackend",
    "State",
    "RNGs",
    # progress
    "Progress",
    "Stage",
    "NoProgress",
    "TqdmProgress",
    "to_progress",
    # remote
    "MimiqRemoteBackend",
    # rng helpers
    "normalize_seed",
    "derive_grid_seeds",
    # final-block analysis
    "extract_projection",
    "evaluate_projection",
    "needs_trajectories",
    "needs_loss_sampling",
    "any_mixed_unitary",
    "remap_projection_qubits",
    # stochastic kind
    "StochasticKind",
    "default_stochastic_kind",
    "stochastic_kind",
    "is_deterministic",
    "is_trajectory_sampleable",
    "is_runtime_only",
    "is_stochastic",
    "first_stochastic",
    "last_stochastic",
    "first_runtime_only",
    "last_runtime_only",
]
