#
# Copyright © 2026 QPerfect. All Rights Reserved.
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
"""Circuit simplification passes.

Every pass takes a circuit and returns a new one with the same unitary, global
phase included, and with measurements, resets and other non-unitary operations
where they were. :func:`simplify` chains them.

The passes run in the native extension. This module describes each distinct
operation of the circuit once by a few facts (its numeric matrix, whether it is
a multi-controlled X, which rotations it merges with, ...), hands the circuit
over as arrays, and builds the result. Instructions a pass leaves alone come
back as the same objects. Arithmetic on symbolic parameters stays here: the
native code calls back for it.
"""

import math

import numpy as np

from mimiqcircuits import _core
from mimiqcircuits.backends.passes import (
    AbstractPass,
    PassResult,
    PassSpec,
    PInt,
    PList,
    PSym,
)
from mimiqcircuits.circuit import Circuit
from mimiqcircuits.instruction import Instruction
from mimiqcircuits.operations.amplitude import Amplitude
from mimiqcircuits.operations.control import Control
from mimiqcircuits.operations.entanglement import (
    BondDim,
    SchmidtRank,
    VonNeumannEntropy,
)
from mimiqcircuits.operations.gates.customdiagonal import GateCustomDiagonal
from mimiqcircuits.operations.gates.delay import Delay
from mimiqcircuits.operations.gates.gate import Gate
from mimiqcircuits.operations.gates.generalized.rpauli import RPauli
from mimiqcircuits.operations.gates.standard.hadamard import GateH
from mimiqcircuits.operations.gates.standard.interactions import (
    GateRXX,
    GateRYY,
    GateRZX,
    GateRZZ,
    GateXXminusYY,
    GateXXplusYY,
)
from mimiqcircuits.operations.gates.standard.pauli import GateX, GateY, GateZ
from mimiqcircuits.operations.gates.standard.phase import GateP
from mimiqcircuits.operations.gates.standard.rotations import (
    GateR,
    GateRX,
    GateRY,
    GateRZ,
)
from mimiqcircuits.operations.gates.standard.cpauli import GateCX
from mimiqcircuits.operations.gates.standard.s import GateS, GateSDG
from mimiqcircuits.operations.gates.standard.swap import GateSWAP
from mimiqcircuits.operations.gates.standard.sx import GateSX, GateSXDG
from mimiqcircuits.operations.gates.standard.sy import GateSY, GateSYDG
from mimiqcircuits.operations.gates.standard.t import GateT, GateTDG
from mimiqcircuits.operations.gates.standard.deprecated import GateU1
from mimiqcircuits.operations.gates.standard.u import GateU

__all__ = [
    "remove_identities",
    "cancel_inverses",
    "merge_rotations",
    "resynthesize_1q",
    "resynthesize_2q",
    "resynthesize_linear",
    "simplify",
    "DEFAULT_SIMPLIFY_PASSES",
    "SimplifyPass",
]

# Gates wider than this never get a dense matrix: it has 4**n entries.
_MAX_MATRIX_QUBITS = 6

# Gates that compose by adding their first parameter, provided the others agree.
_ADDITIVE_GATES = (
    GateRX,
    GateRY,
    GateRZ,
    GateP,
    GateU1,
    GateR,
    GateRXX,
    GateRYY,
    GateRZZ,
    GateRZX,
    GateXXplusYY,
    GateXXminusYY,
)

# Operations whose value depends on the whole register, not on their targets.
_GLOBAL = (Amplitude, BondDim, SchmidtRank, VonNeumannEntropy)

# The gates the native code creates, by code.
_CODES = (
    GateU,
    GateP,
    GateRZ,
    GateRX,
    GateRY,
    GateH,
    GateCX,
    GateX,
    GateT,
    GateS,
    GateZ,
    GateSDG,
    GateTDG,
    GateSX,
    GateSXDG,
    GateY,
    GateSY,
    GateSYDG,
)

_STANDARD_MODULE = "mimiqcircuits.operations.gates.standard"


def _opkey(op):
    """A key equal for two operations only if they are the same operation.

    Standard gates are fully described by their class and parameters, and a
    controlled gate by its number of controls and the gate it controls. Other
    gates (custom matrices, declared gates, ...) are only equal to themselves.
    Every other operation is keyed by class and size, since the passes never
    rebuild one.
    """
    t = type(op)
    if not isinstance(op, Gate):
        return (t, op.num_qubits)
    if t.__module__.startswith(_STANDARD_MODULE):
        return (t, *op.getparams())
    if t is Control:
        return (Control, op.num_controls, _opkey(op.op))
    return ("id", id(op))


def _isdiagonal(op):
    """Diagonal by structure, for gates without a numeric matrix."""
    if not (isinstance(op, Gate) and not isinstance(op, Delay) and op.isunitary()):
        return False
    if isinstance(op, Control):
        return _isdiagonal(op.op)
    if isinstance(op, GateCustomDiagonal):
        return True
    if op.num_qubits > _MAX_MATRIX_QUBITS:
        return False
    try:
        m = op.matrix()
    except Exception:
        return False
    n = m.rows
    return all(m[i, j] == 0 for i in range(n) for j in range(n) if i != j)


def _mergekey(op):
    """What two gates must share to merge by adding their first parameter."""
    if type(op) in _ADDITIVE_GATES:
        return (type(op), *op.getparams()[1:])
    if isinstance(op, RPauli):
        return (RPauli, str(op.pauli))
    if isinstance(op, Control):
        inner = _mergekey(op.op)
        return None if inner is None else (Control, op.num_controls, inner)
    return None


def _firstparam(op):
    if isinstance(op, Control):
        return _firstparam(op.op)
    if isinstance(op, RPauli):
        return op.theta
    return op.getparams()[0]


def _withfirstparam(op, value):
    """`op` with its first parameter replaced by `value`."""
    if isinstance(op, Control):
        return Control(op.num_controls, _withfirstparam(op.op, value))
    if isinstance(op, RPauli):
        return RPauli(op.pauli, value)
    params = op.getparams()
    return type(op)(value, *params[1:])


def _isnumber(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


class _Host:
    """The operations of one native call, and the callbacks it needs."""

    def __init__(self):
        self.ops = []
        self.facts = []
        self.ids = {}
        self.classes = {}

    def opid(self, op):
        key = _opkey(op)
        i = self.ids.get(key)
        if i is None:
            i = self.add(op)
            self.ids[key] = i
        return i

    def add(self, op):
        self.ops.append(op)
        self.facts.append(self._facts(op))
        return len(self.ops) - 1

    def mergeclass(self, key):
        if key is None:
            return -1
        return self.classes.setdefault(key, len(self.classes))

    def _facts(self, op):
        nq = op.num_qubits
        if not isinstance(op, Gate):
            return (False, False, nq, None, False, 0, False, -1, math.nan, 0, isinstance(op, _GLOBAL))
        fusable = op.isunitary()
        plain = fusable and not isinstance(op, Delay)
        matrix = None
        if fusable and nq <= _MAX_MATRIX_QUBITS:
            try:
                matrix = np.ascontiguousarray(op.unwrappedmatrix(), dtype=np.complex128).tobytes()
            except Exception:
                matrix = None
        diag = matrix is None and _isdiagonal(op)
        if isinstance(op, GateX):
            xtarget = 1
        elif isinstance(op, Control) and isinstance(op.op, GateX):
            xtarget = nq
        else:
            xtarget = 0
        if isinstance(op, GateX):
            linear = 1
        elif isinstance(op, Control) and op.num_controls == 1 and isinstance(op.op, GateX):
            linear = 2
        elif isinstance(op, GateSWAP):
            linear = 3
        else:
            linear = 0
        mclass = self.mergeclass(_mergekey(op))
        first = math.nan
        if mclass >= 0:
            p = _firstparam(op)
            if _isnumber(p):
                first = float(p)
        return (
            plain,
            fusable,
            nq,
            matrix,
            diag,
            xtarget,
            isinstance(op, GateRX),
            mclass,
            first,
            linear,
            False,
        )

    def build(self, desc):
        kind = desc[0]
        if kind == 0:
            return self.ops[desc[1]]
        if kind == 1:
            return _CODES[desc[1]](*desc[2])
        return _withfirstparam(self.build(desc[1]), desc[2])

    # Called by the native code for gates with symbolic parameters.

    def merge(self, da, db):
        a, b = self.build(da), self.build(db)
        merged = _withfirstparam(a, _firstparam(a) + _firstparam(b))
        k = self.add(merged)
        return k, self.facts[k]

    def inverse(self, da, db):
        try:
            return self.build(da).inverse() == self.build(db)
        except Exception:
            return False


def _run(circuit, codes, maxiter):
    host = _Host()
    ops, qubits, nqubits, bits, nbits, zvars, nzvars = [], [], [], [], [], [], []
    for inst in circuit:
        ops.append(host.opid(inst.operation))
        qubits.extend(inst.qubits)
        nqubits.append(len(inst.qubits))
        bits.extend(inst.bits)
        nbits.append(len(inst.bits))
        zvars.extend(inst.zvars)
        nzvars.append(len(inst.zvars))
    classes = [host.mergeclass(_mergekey(g(0.0))) for g in (GateP, GateRZ, GateRX, GateRY)]

    descs, out = _core.simplify_circuit(
        host.facts, ops, qubits, nqubits, bits, nbits, zvars, nzvars,
        classes, list(codes), maxiter, host,
    )

    n0 = len(host.facts)
    built = {}

    def op(entry):
        if entry < n0:
            return host.ops[entry]
        o = built.get(entry)
        if o is None:
            o = built[entry] = host.build(descs[entry - n0])
        return o

    insts = circuit.instructions
    result = []
    for i, entry, qs in out:
        result.append(insts[i] if i >= 0 else Instruction(op(entry), tuple(qs)))
    return Circuit(result)


def remove_identities(circuit):
    """Remove gates whose matrix is the identity.

    The test is on the exact matrix, so a gate that is the identity only up to
    a global phase, like ``GateRX(2π)``, is kept. Gates with symbolic
    parameters, ``Delay`` and non-unitary operations are always kept.

    Examples:
        >>> import mimiqcircuits as mc
        >>> c = mc.Circuit()
        >>> _ = c.push(mc.GateRZ(0), 0)
        >>> _ = c.push(mc.GateH(), 0)
        >>> mc.remove_identities(c)
        1-qubit circuit with 1 instruction:
        └── H @ q[0]
        <BLANKLINE>
    """
    return _run(circuit, [0], 0)


def cancel_inverses(circuit):
    """Delete pairs of gates that multiply to the identity.

    This covers pairs such as ``CX CX``, ``H H`` or ``S SDG``, also when other
    gates sit between them, as long as the second gate commutes with all of
    them. Two gates cancel when one is the inverse of the other on the same
    qubits, or when the product of their matrices is the identity, which also
    catches ``SWAP(0,1) SWAP(1,0)``.

    Commutation is checked on the gates' matrices, so for example ``Z`` passes
    a ``CX`` control and ``X`` passes a ``CX`` target; gates with symbolic
    parameters only pass through each other when both are diagonal.
    Multi-controlled X gates (``GateCX``, ``GateCCX``, ``Control(n, GateX())``)
    follow structural rules at any width: two of them commute unless the target
    of one is a control of the other, they commute with diagonal gates that
    avoid their target and with X rotations that avoid their controls, and two
    on the same qubits with the same target cancel, whatever the order of the
    controls. Other gates wider than six qubits are never moved or cancelled.

    Non-unitary operations, barriers and classically controlled gates are never
    moved across, and the search looks back a bounded number of instructions.

    Examples:
        >>> import mimiqcircuits as mc
        >>> c = mc.Circuit()
        >>> _ = c.push(mc.GateCX(), 0, 1)
        >>> _ = c.push(mc.GateZ(), 0)      # Z commutes with the CX control...
        >>> _ = c.push(mc.GateCX(), 0, 1)  # ... so the two CX gates cancel
        >>> mc.cancel_inverses(c)
        1-qubit circuit with 1 instruction:
        └── Z @ q[0]
        <BLANKLINE>
    """
    return _run(circuit, [1], 0)


def merge_rotations(circuit):
    """Merge pairs of rotations on the same qubits into one.

    ``RZ(a) RZ(b)`` becomes ``RZ(a + b)``, and a pair whose sum is the identity
    is removed. The two rotations may be separated by gates that commute with
    the second one, as in :func:`cancel_inverses`.

    Merged gates are ``GateRX``, ``GateRY``, ``GateRZ``, ``GateP``, ``GateU1``,
    ``GateR``, ``GateRXX``, ``GateRYY``, ``GateRZZ``, ``GateRZX``,
    ``GateXXplusYY``, ``GateXXminusYY`` and ``RPauli``, any of them under the
    same ``Control`` (``GateCRZ``, ``GateCP``, ...). Gates with more than one
    parameter merge only when the others are equal. Symbolic angles are added
    symbolically.

    Single-qubit gates with numeric parameters also merge across kinds, within
    families of rotations about the same axis that compose without a global
    phase:

    - ``GateZ``, ``GateS``, ``GateSDG``, ``GateT``, ``GateTDG``, ``GateP``,
      ``GateU1``
    - ``GateX``, ``GateSX``, ``GateSXDG``
    - ``GateY``, ``GateSY``, ``GateSYDG``

    so ``T T`` becomes ``S`` and ``S Z`` becomes ``SDG``. The result is the
    named gate when there is one and ``GateP`` or ``GateU`` otherwise.
    ``GateRZ``, ``GateRX`` and ``GateRY`` merge only with themselves, since they
    differ from the other members of their family by a global phase.

    Examples:
        >>> import mimiqcircuits as mc
        >>> c = mc.Circuit()
        >>> _ = c.push(mc.GateT(), 0)
        >>> _ = c.push(mc.GateCX(), 0, 1)  # T on the control commutes with CX
        >>> _ = c.push(mc.GateT(), 0)
        >>> mc.merge_rotations(c)          # T T = S
        2-qubit circuit with 2 instructions:
        ├── S @ q[0]
        └── CX @ q[0], q[1]
        <BLANKLINE>
    """
    return _run(circuit, [2], 0)


def resynthesize_1q(circuit):
    """Replace every run of two or more single-qubit gates on one qubit by one
    ``GateU``, or remove it when the run multiplies to the identity.

    A run ends at any instruction touching the qubit that is not a single-qubit
    gate with numeric parameters. Single gates are left as they are.

    Examples:
        >>> import mimiqcircuits as mc
        >>> c = mc.Circuit()
        >>> _ = c.push(mc.GateH(), 0)
        >>> _ = c.push(mc.GateS(), 0)
        >>> _ = c.push(mc.GateH(), 0)
        >>> len(mc.resynthesize_1q(c))
        1
    """
    return _run(circuit, [3], 0)


def resynthesize_2q(circuit):
    """Rewrite each two-qubit block with the fewest CX gates it needs.

    The gates are grouped into blocks acting on at most two qubits, as
    :func:`fuse_circuit` does, and each block is rebuilt from its KAK
    decomposition with between zero and three ``GateCX``. A block is rewritten
    only when that lowers its number of two-qubit gates: a SWAP written as
    three CX gates stays as it is, while any longer sequence on the same pair
    comes down to three or fewer. The rewritten block uses ``GateCX``,
    ``GateU``, ``GateRX``, ``GateRY``, ``GateRZ`` and ``GateH``; follow with
    :func:`resynthesize_1q` to merge the single-qubit gates it leaves behind.

    Gates with symbolic parameters and non-unitary operations are left alone
    and end the blocks around them.

    Examples:
        >>> import mimiqcircuits as mc
        >>> c = mc.Circuit()
        >>> for _ in range(2):              # three CX with rotations in between...
        ...     _ = c.push(mc.GateCX(), 0, 1)
        ...     _ = c.push(mc.GateRX(0.3), 0)
        >>> _ = c.push(mc.GateCX(), 0, 1)
        >>> sum(len(i.qubits) == 2 for i in mc.resynthesize_2q(c))  # ...need one
        1
    """
    return _run(circuit, [4], 0)


def resynthesize_linear(circuit):
    """Rewrite every block of ``GateX``, ``GateCX`` and ``GateSWAP`` gates.

    Such a block, of any width, sends the basis state ``x`` to ``A x ⊕ b`` for
    an invertible Boolean matrix ``A``, so it computes a classical linear
    (affine) function of the bits. The block is rebuilt from ``A`` with
    Patel-Markov-Hayes elimination, which needs ``O(n²/log n)`` CX gates for
    ``n`` qubits, followed by one ``GateX`` for each set bit of ``b``. The
    rewrite is kept only when it has fewer two-qubit gates than the block, or as
    many and fewer gates in total. A block that computes the identity is
    removed.

    It is not one of the :data:`DEFAULT_SIMPLIFY_PASSES`: circuits where CX
    networks are a small part gain little from it. Pass it to :func:`simplify`
    explicitly when they are not.

    Examples:
        >>> import mimiqcircuits as mc
        >>> c = mc.Circuit()
        >>> for _ in range(2):              # a CX ladder, applied twice...
        ...     for q in range(3):
        ...         _ = c.push(mc.GateCX(), q, q + 1)
        >>> sum(len(i.qubits) == 2 for i in mc.resynthesize_linear(c))  # ...needs 2
        2
    """
    return _run(circuit, [5], 0)


_PASS_CODES = {
    remove_identities: 0,
    cancel_inverses: 1,
    merge_rotations: 2,
    resynthesize_1q: 3,
    resynthesize_2q: 4,
    resynthesize_linear: 5,
}

#: The passes :func:`simplify` runs by default, in order.
DEFAULT_SIMPLIFY_PASSES = (
    remove_identities,
    cancel_inverses,
    merge_rotations,
    resynthesize_2q,
    resynthesize_1q,
)


def _cost(circuit):
    return (sum(len(inst.qubits) == 2 for inst in circuit), len(circuit))


def simplify(circuit, passes=DEFAULT_SIMPLIFY_PASSES, maxiter=10):
    """Simplify a circuit by running ``passes`` in order, repeatedly.

    Rounds stop when one no longer lowers the number of two-qubit gates or,
    failing that, the total number of instructions, or after ``maxiter``
    rounds. A pass that already ran on the current circuit is skipped until
    another pass changes it. The result has the same unitary as ``circuit``,
    global phase included, and the same measurements, resets and other
    non-unitary operations in the same order.

    ``passes`` is any sequence of functions from ``Circuit`` to ``Circuit``.
    The default, :data:`DEFAULT_SIMPLIFY_PASSES`, is :func:`remove_identities`,
    :func:`cancel_inverses`, :func:`merge_rotations`, :func:`resynthesize_2q`
    and :func:`resynthesize_1q`. When every pass is one of this module's, the
    whole loop runs natively.

    Gates are rewritten numerically, so single-qubit runs come out as ``GateU``
    and named gates such as ``GateH`` or ``GateS`` may be replaced by it. Pass a
    subset to keep them, for example
    ``passes=(remove_identities, cancel_inverses, merge_rotations)``.
    :func:`resynthesize_linear` is not a default pass: add it for circuits with
    large CX networks.

    Examples:
        >>> import mimiqcircuits as mc
        >>> c = mc.Circuit()
        >>> _ = c.push(mc.GateCX(), 0, 1)
        >>> _ = c.push(mc.GateRZ(0.2), 0)  # RZ commutes with the CX control...
        >>> _ = c.push(mc.GateCX(), 0, 1)  # ...so the two CX gates cancel
        >>> _ = c.push(mc.GateRZ(0.3), 0)  # and the rotations merge
        >>> mc.simplify(c)
        1-qubit circuit with 1 instruction:
        └── RZ(0.5) @ q[0]
        <BLANKLINE>
    """
    codes = [_PASS_CODES.get(p) for p in passes]
    if all(code is not None for code in codes):
        return _run(circuit, codes, maxiter)

    cost = _cost(circuit)
    version = 0
    seen = [-1] * len(passes)
    for _ in range(maxiter):
        nxt = circuit
        for k, p in enumerate(passes):
            if seen[k] == version:
                continue
            d = p(nxt)
            if d != nxt:
                version += 1
            seen[k] = version
            nxt = d
        nextcost = _cost(nxt)
        if not nextcost < cost:
            break
        circuit, cost = nxt, nextcost
    return circuit


class SimplifyPass(AbstractPass):
    """Pass that simplifies the circuit with :func:`simplify`.

    It removes identities, cancels inverse pairs, merges rotations and
    rewrites one- and two-qubit blocks with fewer gates, keeping the unitary
    exactly and every non-unitary operation in place. Qubit indices are never
    changed, so :attr:`PassResult.qubit_permutation` is ``None``.

    ``passes`` names the simplification passes to run, in order:
    ``"remove_identities"``, ``"cancel_inverses"``, ``"merge_rotations"``,
    ``"resynthesize_1q"``, ``"resynthesize_2q"`` and ``"resynthesize_linear"``.
    ``None`` (the default) runs :data:`DEFAULT_SIMPLIFY_PASSES`. ``maxiter``
    bounds the number of rounds, as in :func:`simplify`.

    Only these named passes are allowed, so that the pass can be described to
    a remote executor; to run other functions, call :func:`simplify` on the
    circuit before executing it.

    Examples:
        >>> import mimiqcircuits as mc
        >>> from mimiqcircuits.backends.passes import PassContext
        >>> c = mc.Circuit()
        >>> _ = c.push(mc.GateH(), 0)
        >>> _ = c.push(mc.GateH(), 0)
        >>> out, result = mc.SimplifyPass().apply(PassContext(), c)
        >>> len(out), result.qubit_permutation
        (0, None)
    """

    def __init__(self, passes=None, maxiter=10):
        if passes is not None:
            passes = tuple(passes)
            for name in passes:
                if name not in _PASSES_BY_NAME:
                    known = ", ".join(repr(n) for n in _PASSES_BY_NAME)
                    raise ValueError(
                        f"unknown simplification pass {name!r}; use one of {known}"
                    )
        if isinstance(maxiter, bool) or not isinstance(maxiter, int) or maxiter < 1:
            raise ValueError(f"maxiter must be an integer of at least 1, not {maxiter!r}")
        self.passes = passes
        self.maxiter = maxiter

    def spec(self):
        params = [("maxiter", PInt(self.maxiter))]
        if self.passes is not None:
            params.append(("passes", PList(tuple(PSym(n) for n in self.passes))))
        return PassSpec("simplify", tuple(params))

    def apply(self, ctx, circuit):
        if self.passes is None:
            passes = DEFAULT_SIMPLIFY_PASSES
        else:
            passes = tuple(_PASSES_BY_NAME[n] for n in self.passes)
        return simplify(circuit, passes=passes, maxiter=self.maxiter), PassResult()


# The simplification passes a `SimplifyPass` may run, by name.
_PASSES_BY_NAME = {
    "remove_identities": remove_identities,
    "cancel_inverses": cancel_inverses,
    "merge_rotations": merge_rotations,
    "resynthesize_1q": resynthesize_1q,
    "resynthesize_2q": resynthesize_2q,
    "resynthesize_linear": resynthesize_linear,
}
