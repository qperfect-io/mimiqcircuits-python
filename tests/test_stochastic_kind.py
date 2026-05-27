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
"""Tests for the resolution-time taxonomy.

Mirrors the Julia surface in AbstractQCSs.jl/test/test_stochastic_kind.jl.
The classification is backend-dependent: a stub backend exercises the
defaults, and a second stub overrides for one op type to verify the
override actually bites.
"""
from __future__ import annotations

import pytest

import mimiqcircuits as mc
from mimiqcircuits.backends import (
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
    Backend,
    LocalBackend,
    Limits,
)


# ── Stub backends used only in these tests ────────────────────────────────


class _BareBackend(Backend):
    """A minimal backend that overrides nothing — exercises the
    default `stochastic_kind` dispatch."""

    @property
    def name(self) -> str:
        return "BareBackend"

    @property
    def version(self) -> str:
        return "0.0.0"

    def capabilities(self):
        return set()

    def execute(self, *_args, **_kwargs):  # pragma: no cover
        raise NotImplementedError


class _AllKrausIsRTBackend(_BareBackend):
    """Overrides `stochastic_kind` so every krauschannel — mix-unitary
    or not — is classified as RuntimeOnly. The load-bearing test for
    the "backend-dependent" claim."""

    @property
    def name(self) -> str:
        return "AllKrausIsRT"

    def stochastic_kind(self, op_or_inst):
        from mimiqcircuits.backends.stochastic_kind import _resolve_op
        op = _resolve_op(op_or_inst)
        if isinstance(op, mc.krauschannel):
            return StochasticKind.RuntimeOnly
        return super().stochastic_kind(op_or_inst)


# ── Enum surface ──────────────────────────────────────────────────────────


class TestStochasticKindEnum:
    def test_ordering(self):
        assert StochasticKind.Deterministic < StochasticKind.TrajectorySampleable
        assert StochasticKind.TrajectorySampleable < StochasticKind.RuntimeOnly

    def test_int_values(self):
        assert int(StochasticKind.Deterministic) == 0
        assert int(StochasticKind.TrajectorySampleable) == 1
        assert int(StochasticKind.RuntimeOnly) == 2


# ── Defaults ──────────────────────────────────────────────────────────────


class TestDefaultClassification:
    def setup_method(self):
        self.backend = _BareBackend()

    def test_plain_gates_are_deterministic(self):
        assert self.backend.stochastic_kind(mc.GateH()) == StochasticKind.Deterministic
        assert self.backend.stochastic_kind(mc.GateX()) == StochasticKind.Deterministic
        assert self.backend.stochastic_kind(mc.GateCX()) == StochasticKind.Deterministic
        assert is_deterministic(self.backend, mc.GateH())
        assert not is_stochastic(self.backend, mc.GateH())

    def test_barrier_is_deterministic(self):
        assert self.backend.stochastic_kind(mc.Barrier(1)) == StochasticKind.Deterministic

    def test_mix_unitary_kraus_is_TS(self):
        # PauliX is a mix-unitary Kraus channel.
        op = mc.PauliX(0.1)
        assert op.ismixedunitary()
        assert self.backend.stochastic_kind(op) == StochasticKind.TrajectorySampleable
        assert is_trajectory_sampleable(self.backend, op)
        assert is_stochastic(self.backend, op)

    def test_depolarizing_is_TS_not_RT(self):
        # Depolarizing's Kraus ops are ∝ I, X, Y, Z — mix-unitary.
        # Regression guard against the "all Kraus is RT" footgun.
        op = mc.Depolarizing1(0.1)
        assert self.backend.stochastic_kind(op) == StochasticKind.TrajectorySampleable

    def test_non_mix_unitary_kraus_is_RT(self):
        op = mc.AmplitudeDamping(0.1)
        assert not op.ismixedunitary()
        assert self.backend.stochastic_kind(op) == StochasticKind.RuntimeOnly
        assert is_runtime_only(self.backend, op)

    def test_loss_err_is_RT_in_v1(self):
        # F-S2 will promote to TS once sample_losses moves into
        # prepare_trajectory; today there is no such resolver.
        assert self.backend.stochastic_kind(mc.LossErr(0.1)) == StochasticKind.RuntimeOnly

    def test_measure_is_RT(self):
        assert self.backend.stochastic_kind(mc.Measure()) == StochasticKind.RuntimeOnly

    def test_ifstatement_inherits_from_inner(self):
        # IfStatement wrapping a deterministic gate → Deterministic.
        ifs_det = mc.IfStatement(mc.GateX(), mc.BitString("1"))
        assert self.backend.stochastic_kind(ifs_det) == StochasticKind.Deterministic

        # IfStatement wrapping an RT op → RuntimeOnly.
        ifs_rt = mc.IfStatement(mc.AmplitudeDamping(0.1), mc.BitString("1"))
        assert self.backend.stochastic_kind(ifs_rt) == StochasticKind.RuntimeOnly


# ── Instruction dispatch ──────────────────────────────────────────────────


class TestInstructionDispatch:
    def test_instruction_unwraps_to_op(self):
        backend = _BareBackend()
        c = mc.Circuit()
        c.push(mc.GateH(), 1)
        c.push(mc.PauliX(0.1), 1)
        c.push(mc.AmplitudeDamping(0.1), 1)
        c.push(mc.GateX(), 1)

        assert backend.stochastic_kind(c[0]) == StochasticKind.Deterministic
        assert backend.stochastic_kind(c[1]) == StochasticKind.TrajectorySampleable
        assert backend.stochastic_kind(c[2]) == StochasticKind.RuntimeOnly
        assert backend.stochastic_kind(c[3]) == StochasticKind.Deterministic


# ── Backend override is the load-bearing test ─────────────────────────────


class TestBackendOverride:
    def test_override_changes_classification(self):
        """The load-bearing test for the backend-dependent claim.

        Bare backend → mix-unitary Kraus is TS. Override backend →
        every Kraus (mix-unitary or not) is RT. If `stochastic_kind`
        were truly op-only the override would not bite.
        """
        op = mc.PauliX(0.1)  # mix-unitary Kraus
        assert _BareBackend().stochastic_kind(op) == StochasticKind.TrajectorySampleable
        assert _AllKrausIsRTBackend().stochastic_kind(op) == StochasticKind.RuntimeOnly

    def test_override_does_not_change_non_kraus(self):
        # Non-Kraus ops fall through to the super() call → bare default.
        assert _AllKrausIsRTBackend().stochastic_kind(mc.GateH()) == StochasticKind.Deterministic


# ── Free-function helpers ─────────────────────────────────────────────────


class TestFreeFunctions:
    def test_stochastic_kind_free_function(self):
        backend = _BareBackend()
        assert stochastic_kind(backend, mc.GateH()) == StochasticKind.Deterministic
        assert stochastic_kind(backend, mc.AmplitudeDamping(0.1)) == StochasticKind.RuntimeOnly

    def test_first_last_stochastic(self):
        backend = _BareBackend()

        c_det = mc.Circuit()
        c_det.push(mc.GateH(), 1)
        c_det.push(mc.GateCX(), 1, 2)
        assert first_stochastic(backend, c_det) is None
        assert last_stochastic(backend, c_det) is None

        c_mixed = mc.Circuit()
        c_mixed.push(mc.GateH(), 1)                       # 0: det
        c_mixed.push(mc.PauliX(0.1), 1)                   # 1: TS
        c_mixed.push(mc.GateCX(), 1, 2)                   # 2: det
        c_mixed.push(mc.AmplitudeDamping(0.1), 2)         # 3: RT
        c_mixed.push(mc.GateX(), 2)                       # 4: det
        assert first_stochastic(backend, c_mixed) == 1
        assert last_stochastic(backend, c_mixed) == 3

    def test_first_last_runtime_only(self):
        backend = _BareBackend()
        c = mc.Circuit()
        c.push(mc.GateH(), 1)                              # 0
        c.push(mc.PauliX(0.1), 1)                          # 1: TS
        c.push(mc.AmplitudeDamping(0.1), 1)                # 2: RT
        c.push(mc.PauliX(0.1), 1)                          # 3: TS
        assert first_runtime_only(backend, c) == 2
        assert last_runtime_only(backend, c) == 2

        c_no_rt = mc.Circuit()
        c_no_rt.push(mc.GateH(), 1)
        c_no_rt.push(mc.PauliX(0.1), 1)
        assert first_runtime_only(backend, c_no_rt) is None
        assert last_runtime_only(backend, c_no_rt) is None
