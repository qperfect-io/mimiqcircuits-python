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

"""Final-block analysis. Reverse-scan a `Circuit`, absorb its trailing
"projection block" (Measures, Resets, post-measure X gates) into a
small `projection_circuit` of classical-bit instructions, and return
the remaining `quantum_circuit` plus that projection.

The two-circuit decomposition is the post-evolution analog of the
old ``MeasureInfo`` enum:

    quantum_circuit, projection_circuit = extract_projection(c)
    # evolve through quantum_circuit once
    # for each shot:
    #     sample = quantum_state.sample()
    #     cstate = evaluate_projection(projection_circuit, sample)

`projection_circuit` only contains:

- ``Measure(q, b)``               — ``cstate[b] = sample[q]``
- ``Not(b)``                      — ``cstate[b] = !cstate[b]``
- ``SetBit0(b)`` / ``SetBit1(b)`` — bit is classically known

Mirrors the Julia `AbstractQCSs.extract_projection`. The two ports
must stay behavioural-parity — if one is fixed, fix the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import mimiqcircuits as mc


__all__ = [
    "extract_projection",
    "evaluate_projection",
    "needs_trajectories",
    "needs_loss_sampling",
    "any_mixed_unitary",
    "remap_projection_qubits",
]


def needs_trajectories(circuit: "mc.Circuit") -> bool:
    """Return True if `circuit` still contains any non-unitary op
    that requires per-shot evolution. Operations that don't touch
    qubits (Amplitude on a z-register, Tick, …) or that declare
    themselves unitary (Gates, AbstractAnnotation, ExpectationValue, …)
    are ignored.

    Mirrors the Julia `AbstractQCSs.needs_trajectories`:
    ``num_qubits(op) != 0 && !isunitary(op)``.
    """
    for inst in circuit.instructions:
        op = inst.operation
        if op.num_qubits == 0:
            continue
        if op.isunitary():
            continue
        return True
    return False


def needs_loss_sampling(circuit: "mc.Circuit") -> bool:
    """Return True if `circuit` contains a `LossErr` or `QubitLoss`
    operation that must be sampled (Method-1 pre-evolve sampling)
    before the simulator runs."""
    try:
        from mimiqcircuits.operations.losschannel import LossErr, QubitLoss
    except ImportError:
        return False
    for inst in circuit.instructions:
        if isinstance(inst.operation, (LossErr, QubitLoss)):
            return True
    return False


def any_mixed_unitary(circuit: "mc.Circuit") -> bool:
    """Return True if `circuit` contains a mixed-unitary
    :class:`krauschannel` whose `ismixedunitary()` is true. Used as
    the default predicate for the per-trajectory recompile decision
    in :meth:`LocalBackend.recompile_per_trajectory`.

    Mirrors the Julia `AbstractQCSs.any_mixed_unitary`.
    """
    for inst in circuit.instructions:
        op = inst.operation
        if isinstance(op, mc.krauschannel) and op.ismixedunitary():
            return True
    return False


def remap_projection_qubits(projection: "mc.Circuit",
                            qubit_order: list[int],
                            do_remap: bool) -> "mc.Circuit":
    """Rewrite every `Measure(q, b)` instruction in `projection` so
    that ``q → qubit_order[q]``. Used by the driver when the pass
    pipeline reordered the qubits and the projection was synthesised
    in the reordered frame.

    Returns a new `Circuit`; the input is not mutated.
    """
    if not do_remap:
        return projection
    out = mc.Circuit()
    for inst in projection.instructions:
        op = inst.operation
        if isinstance(op, mc.Measure):
            q_old = inst.qubits[0]
            b = inst.bits[0]
            out.push(op, qubit_order[q_old], b)
        else:
            out.push(op, *inst.qubits, *inst.bits, *inst.zvars)
    return out


# ── public API ──────────────────────────────────────────────────────────────


def extract_projection(circuit: "mc.Circuit") -> Tuple["mc.Circuit", "mc.Circuit"]:
    """Reverse-scan `circuit`, absorbing every trailing operation that
    does not affect the qubit observables of the post-evolution state
    into a `projection_circuit` of classical-bit instructions. The
    remaining operations are returned as `quantum_circuit` — the part
    the simulator must evolve.

    Absorbed operations:

    - Trailing ``Measure(q, b)`` / ``MeasureReset(q, b)`` → ``Measure(q, b)``.
    - Trailing ``GateID`` — dropped as a no-op. ``GateX`` / ``GateY``
      / ``GateZ`` are NOT absorbed: ``Y`` and ``Z`` carry phases that
      would corrupt amplitude lookups, and ``X`` absorbed alone would
      require XOR-ing the qubit-flip pattern into every user-supplied
      ``bitstrings`` entry for amplitudes to stay consistent. Keeping
      all Paulis in ``quantum_circuit`` means the projection only
      contains phase-free transformations and amplitude lookups need
      no compensation.
    - ``Reset(q)`` whose qubit has captured pending bits → those bits
      become classical constants (``SetBit0`` / ``SetBit1`` based on
      absorbed X parity). A ``Reset`` whose qubit has no captured
      bits is harmless and dropped.

    Anything that cannot be absorbed (other gates, Kraus channels,
    ``IfStatement``, ``Amplitude``, ``ExpectationValue``, …) blocks
    its qubits and survives into ``quantum_circuit``.

    If the source has no classical register, the projection is
    synthesised over an identity bit↔qubit mapping (bit ``i`` mirrors
    qubit ``i``, length = ``circuit.num_qubits()``).
    """
    nq = circuit.num_qubits()
    nm = circuit.num_bits()
    nb_eff = nq if nm == 0 else nm

    qstates: List[_QubitTailState] = [_QubitTailState() for _ in range(nq)]
    if nm == 0:
        for q in range(nq):
            qstates[q].pending.append(q)

    bit_done = [False] * nb_eff
    insts = list(circuit.instructions)
    last_kept_idx = -1

    for i in range(len(insts) - 1, -1, -1):
        inst = insts[i]
        op = inst.operation
        qs = inst.qubits
        bs = inst.bits

        # Case 1: writing op (Measure, MeasureReset, IfStatement, ...).
        if _is_writing_op(op):
            if isinstance(op, (mc.Measure, mc.MeasureReset)) and \
               _try_absorb_measurement(op, qs[0], bs[0], qstates, bit_done):
                continue

            last_kept_idx = max(last_kept_idx, i)
            for q in qs:
                qstates[q].blocked = True
            for b in bs:
                if 0 <= b < nb_eff:
                    bit_done[b] = True
            continue

        # Case 2: trailing X (absorbable, phase-free) or ID (no-op).
        if qs and _try_absorb_gate(op, qs[0], qstates):
            continue

        # Case 3: Reset on a qubit with captured pending bits → promote.
        if isinstance(op, mc.Reset) and qs:
            q = qs[0]
            if not qstates[q].blocked:
                _const_promote(qstates[q])
                continue

        # Case 4: zero-qubit op or idle qubits → drop, but only when
        # the op is unitary. A Kraus channel on an idle qubit can
        # still mix probabilities on entangled measured qubits, so
        # non-unitary ops must survive into `quantum_circuit`.
        all_idle = True
        for q in qs:
            if qstates[q].pending or qstates[q].blocked:
                all_idle = False
                break
        if all_idle and op.isunitary():
            continue

        # Case 5: non-absorbable — block its qubits and keep it.
        last_kept_idx = max(last_kept_idx, i)
        _block_qubits(qstates, qs)

    for q in range(nq):
        _finalise_pending(qstates[q], q)

    # Build quantum_circuit from the surviving prefix.
    quantum_circuit = mc.Circuit()
    for inst in insts[: last_kept_idx + 1] if last_kept_idx >= 0 else []:
        quantum_circuit.push(inst.operation, *inst.qubits, *inst.bits, *inst.zvars)

    # Any captured measurement whose qubit never appears in
    # `quantum_circuit` is provably reading |0⟩ — const-promote.
    circuit_qubits: set[int] = set()
    for inst in quantum_circuit.instructions:
        for q in inst.qubits:
            circuit_qubits.add(q)
    for q in range(nq):
        if q not in circuit_qubits:
            _force_const_promote(qstates[q])

    projection_circuit = mc.Circuit()
    for q in range(nq):
        for pd in qstates[q].done:
            _emit_projection(projection_circuit, pd)

    return quantum_circuit, projection_circuit


def evaluate_projection(projection: "mc.Circuit",
                        sample: "mc.BitString") -> "mc.BitString":
    """Run the projection circuit on a single raw quantum-state
    `sample`, returning the resulting classical bitstring."""
    nb = projection.num_bits()
    cstate = mc.BitString(nb)
    nq_sample = len(sample)
    for inst in projection.instructions:
        op = inst.operation
        if isinstance(op, mc.Measure):
            q = inst.qubits[0]
            b = inst.bits[0]
            if 0 <= q < nq_sample:
                cstate[b] = bool(sample[q])
        elif isinstance(op, mc.Not):
            b = inst.bits[0]
            cstate[b] = not bool(cstate[b])
        elif isinstance(op, mc.SetBit0):
            cstate[inst.bits[0]] = False
        elif isinstance(op, mc.SetBit1):
            cstate[inst.bits[0]] = True
        else:
            raise ValueError(
                f"evaluate_projection: unsupported instruction {type(op).__name__}"
            )
    return cstate


# ── internal types and helpers ───────────────────────────────────────────────


@dataclass
class _PendingDone:
    bit: int
    direct_qubit: int          # -1 sentinel for "classical constant"
    const_value: int           # 0 / 1, only consulted when direct_qubit == -1


@dataclass
class _QubitTailState:
    blocked: bool = False
    pending: List[int] = field(default_factory=list)   # bit indices
    done: List[_PendingDone] = field(default_factory=list)


def _try_absorb_measurement(op, q: int, b: int,
                            qstates: List[_QubitTailState],
                            bit_done: List[bool]) -> bool:
    qs = qstates[q]
    if not bit_done[b] and not qs.blocked:
        if isinstance(op, mc.MeasureReset):
            # Any bits captured *later* in forward time read a |0⟩
            # register; const-promote them.
            _const_promote(qs)
        bit_done[b] = True
        qs.pending.append(b)
        return True
    return False


def _try_absorb_gate(op, q: int, qstates: List[_QubitTailState]) -> bool:
    """Absorb only `GateID` (true no-op). All non-trivial Paulis
    (X, Y, Z) stay in `quantum_circuit` so amplitude lookups need
    no compensation; the projection circuit is then guaranteed
    phase-free.
    """
    qs = qstates[q]
    if qs.blocked:
        return False
    if isinstance(op, mc.GateID):
        return True
    return False


def _const_promote(qs: _QubitTailState) -> None:
    for b in qs.pending:
        qs.done.append(_PendingDone(bit=b, direct_qubit=-1, const_value=0))
    qs.pending.clear()


def _finalise_pending(qs: _QubitTailState, q: int) -> None:
    for b in qs.pending:
        qs.done.append(_PendingDone(bit=b, direct_qubit=q, const_value=0))
    qs.pending.clear()


def _block_qubits(qstates: List[_QubitTailState], qubits) -> None:
    for q in qubits:
        qs = qstates[q]
        if not qs.blocked:
            _finalise_pending(qs, q)
            qs.blocked = True


def _force_const_promote(qs: _QubitTailState) -> None:
    for i, pd in enumerate(qs.done):
        if pd.direct_qubit >= 0:
            qs.done[i] = _PendingDone(bit=pd.bit, direct_qubit=-1, const_value=0)


def _is_writing_op(op) -> bool:
    if op.iswrapper():
        return _is_writing_op(op.op)
    return op.num_bits > 0 or op.num_zvars > 0


def _emit_projection(projection: "mc.Circuit", pd: _PendingDone) -> None:
    if pd.direct_qubit >= 0:
        projection.push(mc.Measure(), pd.direct_qubit, pd.bit)
    else:
        if pd.const_value == 0:
            projection.push(mc.SetBit0(), pd.bit)
        else:
            projection.push(mc.SetBit1(), pd.bit)
