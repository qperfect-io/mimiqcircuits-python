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
"""Circuit class and utilities."""

from __future__ import (
    annotations,
)  # Required for postponed evaluation of type hints in Python 3.7+
from mimiqcircuits.bitstrings import BitString
from mimiqcircuits.lazy import LazyExpr
from mimiqcircuits.operations.amplitude import Amplitude
from mimiqcircuits.operations.barrier import Barrier
from mimiqcircuits.operations.block import Block
from mimiqcircuits.operations.control import Control
from mimiqcircuits.operations.gates.gate import Gate
from mimiqcircuits.operations.krauschannel import krauschannel
from mimiqcircuits.operations.losschannel import Loss
from mimiqcircuits.operations.operation import Operation
from mimiqcircuits.operations.operator import AbstractOperator
from mimiqcircuits.canvas import AsciiCircuit
from mimiqcircuits.instruction import Instruction
from mimiqcircuits.operations.gates.generalized.paulistring import PauliString
from mimiqcircuits.operations.ifstatement import IfStatement
from mimiqcircuits.operations.parallel import Parallel
from mimiqcircuits.operations.reset import Reset
from mimiqcircuits.operations.repeat import Repeat
from mimiqcircuits.push import push_instruction_container
import symengine as se
import copy
import operator
from collections.abc import Iterable
import shutil
from typing import List, Union
import random
from mimiqcircuits.dag import (
    build_dag,
    traverse_by_bfs as _traverse_by_bfs,
    traverse_by_dfs as _traverse_by_dfs,
    to_networkx as _to_networkx,
)


def _reorder_op_internals(op, perm):
    """Return a copy of ``op`` with any embedded qubit-index data
    rewritten through ``perm``. Most operations don't embed qubit
    indices outside ``inst.qubits``, so they are returned unchanged.

    :class:`Amplitude` is the leaf case: its :attr:`bs` encodes a
    computational-basis state keyed by qubit index, so after a qubit
    permutation the bit at position ``perm[q]`` in the new ``bs``
    must equal the bit at position ``q`` in the old one. That keeps
    the rewritten ``Amplitude`` reading the same amplitude of the
    same quantum-state component as the original.

    The general-operation wrappers (:class:`IfStatement`,
    :class:`Repeat`, :class:`Block`) forward the permutation to their
    inner op(s) so a wrapped :class:`Amplitude` is rewritten in the
    new qubit frame. Classical-bit conditions on :class:`IfStatement`
    are qubit-independent and stay unchanged. :class:`Block` is
    treated as transparent — its inner instructions' qubit / bit /
    zvar targets stay in the block's local frame, only embedded
    payload (e.g. ``Amplitude.bs``) is rewritten. Gate-only wrappers
    (:class:`Control`, :class:`Inverse`, :class:`Power`,
    :class:`Parallel`) constrain the inner type to :class:`Gate`, so
    they cannot wrap :class:`Amplitude` and no override is needed.
    """
    if isinstance(op, Amplitude):
        old_bs = op.bs
        nq = len(old_bs)
        new_bits = [0] * nq
        for q in range(nq):
            new_bits[perm[q]] = int(old_bs[q])
        return Amplitude(BitString(new_bits))
    if isinstance(op, IfStatement):
        new_inner = _reorder_op_internals(op.get_operation(), perm)
        return IfStatement(new_inner, op.bitstring)
    if isinstance(op, Repeat):
        new_inner = _reorder_op_internals(op.op, perm)
        return Repeat(op.repeats, new_inner)
    if isinstance(op, Block):
        new_insts = []
        for inst in op.instructions:
            new_op = _reorder_op_internals(inst.operation, perm)
            new_insts.append(Instruction(new_op, inst.qubits, inst.bits, inst.zvars))
        return Block(op.num_qubits, op.num_bits, op.num_zvars, new_insts)
    return op


def _build_instruction(operation, args):
    """Build the instruction :meth:`Circuit.insert` adds for `operation` on `args`.

    `operation` is an :class:`Instruction` (then `args` must be empty), an
    :class:`Operation`, or the :class:`Barrier` class (then it spans all of
    `args`). Targets in `args` are split into qubits, bits and z-variables by
    the operation's counts.

    Raises:
        TypeError: If `operation` is not an Operation.
        ValueError: If `args` does not match the operation's number of targets,
            or `operation` is an abstract operator that is not a Gate.
    """
    L = len(args)

    if isinstance(operation, Instruction):
        if L != 0:
            raise ValueError(
                "No extra arguments allowed when inserting an instruction."
            )
        return operation

    if operation == Barrier:
        return Instruction(Barrier(L), (*args,), ())

    if not isinstance(operation, Operation):
        raise TypeError("Non Operation object passed to push.")

    # Check if the operation is an abstract operator but not a gate
    if isinstance(operation, AbstractOperator) and not isinstance(
        operation, Gate
    ):
        raise ValueError(
            "Cannot add an abstract operator that is not a Gate to the circuit."
        )

    N = operation.num_qubits
    M = operation.num_bits
    Z = operation.num_zvars

    if L != N + M + Z:
        raise ValueError(
            f"Wrong number of target qubits and bits, given {L} for a {N} qubits + {M} bits operation + {Z} Z-variables."
        )

    return Instruction(
        operation, (*args[:N],), (*args[N : N + M],), (*args[N + M :],)
    )


class Circuit:
    """An ordered list of instructions acting on qubits, classical bits and z-variables.

    Operations are added one at a time with ``c.push(operation, targets...)``,
    which returns the circuit so calls can be chained. The number of qubits,
    bits and z-variables is not declared up front: it is one more than the
    largest index used on each register.

    Args:
        instructions (list of Instruction): Instructions to start with. The
            list is used as the circuit's storage, not copied. Defaults to an
            empty circuit.

    Raises:
        TypeError: If ``instructions`` is not a list, or holds anything other
            than :class:`Instruction` objects.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import pi

        Create a new circuit object

        >>> c = Circuit()

        Add a GateX (Pauli-X) gate on qubit 0

        >>> c.push(GateX(), 0)
        1-qubit circuit with 1 instruction:
        └── X @ q[0]
        <BLANKLINE>

        Add a Controlled-NOT (CX) gate with control qubit 0 and target qubit 1

        >>> c.push(GateCX(), 0, 1)
        2-qubit circuit with 2 instructions:
        ├── X @ q[0]
        └── CX @ q[0], q[1]
        <BLANKLINE>

        Add a Parametric GateRX gate with parameters pi/4

        >>> c.push(GateRX(pi / 4),0)
        2-qubit circuit with 3 instructions:
        ├── X @ q[0]
        ├── CX @ q[0], q[1]
        └── RX((1/4)*pi) @ q[0]
        <BLANKLINE>

        Add a Reset gate on qubit 0

        >>> c.push(Reset(), 0)
        2-qubit circuit with 4 instructions:
        ├── X @ q[0]
        ├── CX @ q[0], q[1]
        ├── RX((1/4)*pi) @ q[0]
        └── Reset @ q[0]
        <BLANKLINE>

        Add a Barrier gate on qubits 0 and 1

        >>> c.push(Barrier(2), 0, 1)
        2-qubit circuit with 5 instructions:
        ├── X @ q[0]
        ├── CX @ q[0], q[1]
        ├── RX((1/4)*pi) @ q[0]
        ├── Reset @ q[0]
        └── Barrier @ q[0:1]
        <BLANKLINE>

        Add a Measurement gate on qubit 0, storing the result in bit 0.

        >>> c.push(Measure(), 0, 0)
        2-qubit, 1-bit circuit with 6 instructions:
        ├── X @ q[0]
        ├── CX @ q[0], q[1]
        ├── RX((1/4)*pi) @ q[0]
        ├── Reset @ q[0]
        ├── Barrier @ q[0:1]
        └── M @ q[0], c[0]
        <BLANKLINE>

        Add a Control gate with GateX as the target gate. The first 3
        qubits are the control qubits.

        >>> c.push(Control(3, GateX()), 0, 1, 2, 3)
        4-qubit, 1-bit circuit with 7 instructions:
        ├── X @ q[0]
        ├── CX @ q[0], q[1]
        ├── RX((1/4)*pi) @ q[0]
        ├── Reset @ q[0]
        ├── Barrier @ q[0:1]
        ├── M @ q[0], c[0]
        └── C₃X @ q[0:2], q[3]
        <BLANKLINE>

        Add a 3-qubit Parallel gate with GateX

        >>> c.push(Parallel(3,GateX()),0, 1, 2)
        4-qubit, 1-bit circuit with 8 instructions:
        ├── X @ q[0]
        ├── CX @ q[0], q[1]
        ├── RX((1/4)*pi) @ q[0]
        ├── Reset @ q[0]
        ├── Barrier @ q[0:1]
        ├── M @ q[0], c[0]
        ├── C₃X @ q[0:2], q[3]
        └── ⨷ ³ X @ q[0], q[1], q[2]
        <BLANKLINE>

        To add operations whose size depends on the targets, such as
        :class:`QFT`, give the targets as registers with :meth:`emplace`.

    Note:
        Commonly used operations include the single-qubit gates
        :class:`GateX`, :class:`GateY`, :class:`GateZ`, :class:`GateH`,
        :class:`GateS`, :class:`GateT`, :class:`GateSX`, :class:`GateID`,
        :class:`GateU`, :class:`GateP`, :class:`GateRX`, :class:`GateRY` and
        :class:`GateRZ`; the two-qubit gates :class:`GateCX`,
        :class:`GateCY`, :class:`GateCZ`, :class:`GateCH`, :class:`GateSWAP`,
        :class:`GateISWAP`, :class:`GateCS`, :class:`GateCSX`,
        :class:`GateECR`, :class:`GateDCX`, :class:`GateCU`, :class:`GateCP`,
        :class:`GateCRX`, :class:`GateCRY`, :class:`GateCRZ`,
        :class:`GateRXX`, :class:`GateRYY`, :class:`GateRZZ`,
        :class:`GateXXplusYY` and :class:`GateXXminusYY`; custom gates
        (:class:`GateCustom`); the wrappers :class:`Control`,
        :class:`Parallel`, :class:`Power` and :class:`Inverse`; the
        generalized gates :class:`QFT` and :class:`PhaseGradient`;
        :class:`Measure` and :class:`Reset`; and :class:`Barrier`.
    """

    def __init__(self, instructions=None):
        if instructions is None:
            instructions = []

        if not isinstance(instructions, list):
            raise TypeError("Circuit should be initialized with a list of Instruction")

        for instruction in instructions:
            if not isinstance(instruction, Instruction):
                raise TypeError("Non Gate object passed to constructor.")

        self._instructions = instructions
        self._graph = None
        self._graph_valid = False
        self._nq = 0
        self._nb = 0
        self._nz = 0
        self._resources_valid = False

    @property
    def instructions(self):
        """Read-only view of the circuit's instructions.

        The list is the circuit's own storage, exposed for reading and
        iteration. Add or remove instructions through :meth:`push`,
        :meth:`insert`, :meth:`append`, or :meth:`remove`: these keep the
        cached dependency graph and resource counts consistent, whereas
        mutating the returned list in place would silently bypass that
        bookkeeping.
        """
        return self._instructions

    def _invalidate_cache(self):
        self._graph_valid = False
        self._resources_valid = False

    def _append_raw(self, instruction):
        """Append an already-built instruction and invalidate the caches.

        Internal entry point shared with the push helper so that every append
        flows through a single cache-invalidating path.
        """
        self._instructions.append(instruction)
        # Same as _invalidate_cache, inlined: this runs once per pushed
        # instruction.
        self._graph_valid = False
        self._resources_valid = False

    def _replace_instructions(self, instructions):
        """Replace all instructions with already-built ones and invalidate the caches.

        The list is replaced in place, so the Circuit object and any reference
        to its ``instructions`` list stay valid. Used by passes that rebuild
        the whole list in one go, such as ``add_noise``. Returns the circuit.
        """
        self._instructions[:] = instructions
        self._invalidate_cache()
        return self

    def _ensure_resources(self):
        # Qubit/bit/zvar counts are the largest index used on each register
        # plus one, computed in a single pass for all three and cached.
        if self._resources_valid:
            return
        nq = nb = nz = -1
        for inst in self._instructions:
            for q in inst.qubits:
                if q > nq:
                    nq = q
            for b in inst.bits:
                if b > nb:
                    nb = b
            for z in inst.zvars:
                if z > nz:
                    nz = z
        self._nq, self._nb, self._nz = nq + 1, nb + 1, nz + 1
        self._resources_valid = True

    def dag(self):
        """Return the circuit's dependency graph as a :class:`CircuitDAG`.

        The graph is built on first use and cached. Changing the circuit
        through :meth:`push`, :meth:`insert`, :meth:`append` or :meth:`remove`
        drops the cache, so the next call rebuilds it.

        Returns:
            CircuitDAG: The graph whose vertex ``i`` is instruction ``i``.

        Examples:
            >>> from mimiqcircuits import *
            >>> c = Circuit()
            >>> _ = c.push(GateH(), 0)
            >>> _ = c.push(GateCX(), 0, 1)
            >>> _ = c.push(GateX(), 1)
            >>> # each instruction depends on the previous one on a shared qubit
            >>> c.dag().edges()
            [(0, 1), (1, 2)]
        """
        if not self._graph_valid:
            self._graph = build_dag(self)
            self._graph_valid = True
        return self._graph

    def traverse_by_bfs(self):
        """Iterate the instructions in breadth-first topological order.

        See :func:`mimiqcircuits.traverse_by_bfs`.
        """
        return _traverse_by_bfs(self)

    def traverse_by_dfs(self):
        """Iterate the instructions in depth-first topological order.

        See :func:`mimiqcircuits.traverse_by_dfs`.
        """
        return _traverse_by_dfs(self)

    def to_networkx(self):
        """Return the dependency graph as a ``networkx.DiGraph``.

        See :func:`mimiqcircuits.to_networkx`. Requires the optional
        ``networkx`` dependency (``pip install mimiqcircuits[graph]``).
        """
        return _to_networkx(self)

    def num_qubits(self):
        """Return the number of qubits of the circuit.

        This is one more than the largest qubit index used, so unused qubits
        below it are counted. An empty circuit has 0 qubits.

        Examples:
            >>> from mimiqcircuits import *
            >>> Circuit().push(GateX(), 3).num_qubits()
            4
        """
        self._ensure_resources()
        return self._nq

    def num_bits(self):
        """Return the number of classical bits of the circuit.

        This is one more than the largest bit index used, or 0 when no
        instruction targets a bit.
        """
        self._ensure_resources()
        return self._nb

    def getparams(self):
        """Return the parameters of every instruction, in circuit order.

        Numeric and symbolic parameters are both returned, one entry per
        parameter of each operation, so a value can appear more than once.

        Returns:
            list: The parameters of all instructions, concatenated.

        Examples:
            >>> import mimiqcircuits as mc
            >>> from symengine import symbols
            >>> theta = symbols("theta")
            >>> c = mc.Circuit().push(mc.GateRX(theta), 0).push(mc.GateP(0.5), 1)
            >>> c.getparams()
            [theta, 0.5]
        """
        params = []
        for inst in self.instructions:
            if hasattr(inst, "getparams"):
                params.extend(inst.getparams())
        return params

    def listvars(self):
        """List the symbolic variables of all the instructions of the circuit.

        Operations inside a ``Block`` or a wrapper such as ``Control`` are
        included. Variables already substituted by :meth:`evaluate` are not
        listed.

        Returns:
            list: Each ``symengine.Symbol`` once, in no particular order.
            The list is empty when the circuit has no symbolic parameters.

        Examples:
            >>> from mimiqcircuits import *
            >>> from symengine import Symbol
            >>> theta, phi = Symbol("theta"), Symbol("phi")
            >>> c = Circuit()
            >>> c.push(GateRX(theta), 0)
            1-qubit circuit with 1 instruction:
            └── RX(theta) @ q[0]
            <BLANKLINE>
            >>> c.push(GateRZ(phi), 1)
            2-qubit circuit with 2 instructions:
            ├── RX(theta) @ q[0]
            └── RZ(phi) @ q[1]
            <BLANKLINE>
            >>> sorted(c.listvars(), key=str)
            [phi, theta]
            >>> c.evaluate({theta: 0.1}).listvars()
            [phi]
        """
        return list({v for inst in self.instructions for v in inst.listvars()})

    def num_zvars(self):
        """Return the number of z-variables of the circuit.

        This is one more than the largest z-variable index used, or 0 when no
        instruction targets a z-variable.
        """
        self._ensure_resources()
        return self._nz

    def empty(self):
        """Return ``True`` if the circuit has no instructions."""
        return len(self.instructions) == 0

    def push(self, operation, *args):
        """Add an operation or an instruction at the end of the circuit.

        The targets are given in order: first the qubits, then the classical
        bits, then the z-variables, as many of each as the operation acts on.
        A target can be an integer or an iterable of integers. When some
        targets are iterables, one instruction is added for each position,
        zipping the iterables together and repeating the integer targets; the
        shortest iterable sets the number of instructions.

        Args:
            operation (Operation or Instruction): The operation to add, or an
                already-built instruction (then no targets are allowed).
            *args (int or iterable of int): The targets.

        Returns:
            Circuit: The circuit itself, so calls can be chained.

        Raises:
            TypeError: If ``operation`` is not an :class:`Operation` or an
                :class:`Instruction`, or a target is neither an integer nor
                an iterable.
            ValueError: If the number of targets does not match the
                operation, a target index is negative or repeated, or
                ``operation`` is an operator that is not a gate (such as a
                projector), which can only be used inside other operations.

        Examples:
            Targets can be integers or iterables of integers. An iterable
            adds one instruction per element.

            >>> from mimiqcircuits import *
            >>> from symengine import pi
            >>> c = Circuit()
            >>> c.push(GateH(), 0)
            1-qubit circuit with 1 instruction:
            └── H @ q[0]
            <BLANKLINE>
            >>> c.push(GateT(), 0)
            1-qubit circuit with 2 instructions:
            ├── H @ q[0]
            └── T @ q[0]
            <BLANKLINE>
            >>> c.push(GateH(), [0,2])  # one H on each of qubits 0 and 2
            3-qubit circuit with 4 instructions:
            ├── H @ q[0]
            ├── T @ q[0]
            ├── H @ q[0]
            └── H @ q[2]
            <BLANKLINE>
            >>> c.push(GateS(), 0)
            3-qubit circuit with 5 instructions:
            ├── H @ q[0]
            ├── T @ q[0]
            ├── H @ q[0]
            ├── H @ q[2]
            └── S @ q[0]
            <BLANKLINE>
            >>> c.push(GateCX(), [2, 0], 1)  # controls 2 and 0, target 1 each time
            3-qubit circuit with 7 instructions:
            ├── H @ q[0]
            ├── T @ q[0]
            ├── H @ q[0]
            ├── H @ q[2]
            ├── S @ q[0]
            ├── CX @ q[2], q[1]
            └── CX @ q[0], q[1]
            <BLANKLINE>
            >>> c.push(GateH(), 0)
            3-qubit circuit with 8 instructions:
            ├── H @ q[0]
            ├── T @ q[0]
            ├── H @ q[0]
            ├── H @ q[2]
            ├── S @ q[0]
            ├── CX @ q[2], q[1]
            ├── CX @ q[0], q[1]
            └── H @ q[0]
            <BLANKLINE>
            >>> c.push(Barrier(3), *range(3))  # same as c.push(Barrier(3), 0, 1, 2)
            3-qubit circuit with 9 instructions:
            ├── H @ q[0]
            ├── T @ q[0]
            ├── H @ q[0]
            ├── H @ q[2]
            ├── S @ q[0]
            ├── CX @ q[2], q[1]
            ├── CX @ q[0], q[1]
            ├── H @ q[0]
            └── Barrier @ q[0:2]
            <BLANKLINE>
            >>> c.push(Measure(), range(3), range(3))  # measure qubit i into bit i
            3-qubit, 3-bit circuit with 12 instructions:
            ├── H @ q[0]
            ├── T @ q[0]
            ├── H @ q[0]
            ├── H @ q[2]
            ├── S @ q[0]
            ├── CX @ q[2], q[1]
            ├── CX @ q[0], q[1]
            ├── H @ q[0]
            ├── Barrier @ q[0:2]
            ├── M @ q[0], c[0]
            ├── M @ q[1], c[1]
            └── M @ q[2], c[2]
            <BLANKLINE>
        """

        return push_instruction_container(self, operation, *args, check_fn=None)

    def _emplace_operation(self, op, regs):
        lr = len(regs)
        lq = op.num_qregs
        lc = op.num_cregs
        lz = op._num_zregs

        if lr != lq + lc + lz:
            raise ValueError(
                f"Wrong number of registers. Expected {lq} quantum + {lc} classical + {lz} z, got {lr}"
            )

        qr = op.qregsizes
        for i in range(lq):
            if len(regs[i]) != qr[i]:
                raise ValueError(
                    f"Wrong size for {i}th quantum register. Expected {qr[i]} but got {len(regs[i])}."
                )

        cr = op.cregsizes
        for i in range(lc):
            if len(regs[lq + i]) != cr[i]:
                raise ValueError(
                    f"Wrong size for {i}th classical register. Expected {cr[i]} but got {len(regs[lq + i])}."
                )

        zr = op.zregsizes
        for i in range(lz):
            if len(regs[lq + lc + i]) != zr[i]:
                raise ValueError(
                    f"Wrong size for {i}th z-register. Expected {zr[i]} but got {len(regs[lq + lc + i])}."
                )

        targets = []
        for reg in regs:
            targets.extend(reg)

        self.push(op, *targets)

        return self

    def emplace(self, op, *regs):
        """Add an operation at the end of the circuit, with targets given by register.

        Each register is a list of indices. An operation has one register for
        each group of qubits, bits or z-variables it acts on (for example a
        :class:`Control` has one register for the controls and one for the
        targets), given in the order quantum, classical, z-variable. The
        registers are joined and passed to :meth:`push`.

        Operations whose size is still open, such as ``QFT()`` or
        ``control(GateX())``, take their size from the length of the
        registers. A :class:`Parallel` takes one single-element register for
        each qubit.

        Args:
            op (Operation): The operation to add, or a lazy expression that
                builds it from the register sizes.
            *regs (list of int): One list of indices per register.

        Returns:
            Circuit: The circuit itself, so calls can be chained.

        Raises:
            ValueError: If the number of registers or the size of one does not
                match the operation, or ``op`` is an operator that is not a
                gate.
            TypeError: If ``op`` is not an operation or a lazy expression.

        Examples:
            >>> from mimiqcircuits import *
            >>> c = Circuit()
            >>> # a fixed-size gate takes one register per group of qubits
            >>> c.emplace(GateRX(0.2), [0])
            1-qubit circuit with 1 instruction:
            └── RX(0.2) @ q[0]
            <BLANKLINE>
            >>> # QFT() has no size yet: the register of 3 qubits sets it
            >>> c.emplace(QFT(), range(3))
            3-qubit circuit with 2 instructions:
            ├── RX(0.2) @ q[0]
            └── QFT @ q[0:2]
            <BLANKLINE>
            >>> # the number of controls comes from the first register
            >>> c.emplace(control(GateX()), [0, 1], [2])
            3-qubit circuit with 3 instructions:
            ├── RX(0.2) @ q[0]
            ├── QFT @ q[0:2]
            └── C₂X @ q[0:1], q[2]
            <BLANKLINE>
        """

        if isinstance(op, LazyExpr):
            # The lazy expression takes register sizes from the front; any left
            # over are checked against the built operation's registers below.
            sizes = [len(reg) for reg in reversed(regs)]
            self._emplace_operation(op._lazy_recursive_evaluate(sizes), regs)
        elif isinstance(op, Parallel):
            if any((isinstance(reg, Iterable) and len(reg) != 1) for reg in regs):
                raise ValueError("Each iterable should contain exactly one qubit.")
            self.push(op, *regs)
        elif isinstance(op, Operation):
            if isinstance(op, AbstractOperator) and not isinstance(op, Gate):
                raise ValueError(
                    "Cannot emplace an abstract operator that is not a Gate into the circuit."
                )
            self._emplace_operation(op, regs)
        elif isinstance(op, type) and issubclass(op, Operation):
            self._emplace_operation(op, regs)
        else:
            raise TypeError("Invalid type passed to emplace")

        return self

    def insert(self, index: int, operation, *args):
        """Insert an operation, an instruction or a whole circuit at a position.

        The new instruction is placed before the one currently at ``index``,
        as with ``list.insert``. Unlike :meth:`push`, targets must be plain
        integers, so exactly one instruction is inserted. When ``operation``
        is a :class:`Circuit`, all its instructions are inserted in order and
        no targets are given.

        Args:
            index: The position to insert at.
            operation (Operation, Instruction or Circuit): What to insert.
            *args (int): The target qubits, then bits, then z-variables of
                ``operation``. Not allowed for an instruction or a circuit.

        Returns:
            Circuit: The circuit itself, so calls can be chained.

        Raises:
            TypeError: If ``operation`` is not an operation, an instruction or
                a circuit, or ``index`` is not an integer when inserting a
                circuit.
            ValueError: If the number of targets is wrong, a target is
                invalid, or ``operation`` is an operator that is not a gate.

        Examples:
            >>> from mimiqcircuits import *
            >>> c= Circuit()
            >>> c.push(GateX(), 0)
            1-qubit circuit with 1 instruction:
            └── X @ q[0]
            <BLANKLINE>
            >>> c.push(GateCX(),0,1)
            2-qubit circuit with 2 instructions:
            ├── X @ q[0]
            └── CX @ q[0], q[1]
            <BLANKLINE>
            >>> # put an H between the X and the CX
            >>> c.insert(1, GateH(), 0)
            2-qubit circuit with 3 instructions:
            ├── X @ q[0]
            ├── H @ q[0]
            └── CX @ q[0], q[1]
            <BLANKLINE>
            >>> # insert a whole circuit at the start
            >>> c.insert(0, Circuit().push(GateZ(), 1))
            2-qubit circuit with 4 instructions:
            ├── Z @ q[1]
            ├── X @ q[0]
            ├── H @ q[0]
            └── CX @ q[0], q[1]
            <BLANKLINE>
        """
        if isinstance(operation, Circuit):
            # operator.index rejects None and other non-integers, which a
            # slice would silently accept as "the whole list".
            index = operator.index(index)
            # One slice assignment shifts the tail once, rather than once per
            # inserted instruction.
            self._instructions[index:index] = operation.instructions
        else:
            self._instructions.insert(index, _build_instruction(operation, args))
        self._invalidate_cache()

        return self

    def append(self, other):
        """Add all the instructions of another circuit at the end of this one.

        The instructions are added as they are, without copying or checking
        them. The circuit is changed in place and nothing is returned.

        Args:
            other (Circuit or list of Instruction): The instructions to add.

        Raises:
            TypeError: If ``other`` is neither a circuit nor a list.

        Examples:
            >>> from mimiqcircuits import *
            >>> c = Circuit().push(GateH(), 0)
            >>> c.append(Circuit().push(GateCX(), 0, 1))
            >>> c
            2-qubit circuit with 2 instructions:
            ├── H @ q[0]
            └── CX @ q[0], q[1]
            <BLANKLINE>
        """
        instructions = None
        if isinstance(other, Circuit):
            instructions = other.instructions
        elif isinstance(other, list):
            instructions = other
        else:
            raise TypeError(
                "Only allowed to append a circuit or a list of instructions"
            )

        for inst in instructions:
            self._instructions.append(inst)
        self._invalidate_cache()

    def remove(self, index: int):
        """Remove the instruction at ``index``.

        Args:
            index: The position of the instruction to remove. Negative
                indices count from the end, as for a list.

        Returns:
            Circuit: The circuit itself, so calls can be chained.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        del self._instructions[index]
        self._invalidate_cache()
        return self

    def inverse(self):
        """Return a new circuit that undoes this one.

        The instructions are reversed and each operation is replaced by its
        inverse. The circuit itself is not changed.

        Returns:
            Circuit: The inverse circuit.

        Raises:
            TypeError: If an operation has no inverse, such as a measurement
                or a reset.
            NotImplementedError: If an operation is a Kraus channel.

        Examples:
            >>> from mimiqcircuits import *
            >>> c = Circuit().push(GateH(), 0).push(GateS(), 0)
            >>> c.inverse()
            1-qubit circuit with 2 instructions:
            ├── S† @ q[0]
            └── H @ q[0]
            <BLANKLINE>
        """
        invgates = [x.inverse() for x in self.instructions]
        invgates.reverse()
        return Circuit(invgates)

    def _decompose(self, circ):
        for instruction in self.instructions:
            instruction._decompose(circ)
        return circ

    def decompose(self, basis=None):
        """Decompose all operations in the circuit to a target basis.

        Operations are rewritten recursively until every one is terminal in
        the basis. The circuit itself is not changed.

        Args:
            basis: The target basis, a :class:`DecompositionBasis`, or a
                :class:`RewriteRule` to apply until it no longer matches.
                Defaults to :class:`CanonicalBasis` (``GateU`` and
                ``GateCX``).

        Returns:
            Circuit: A new circuit with every operation in the target basis.

        Examples:
            Decompose to the canonical basis (the default):

            >>> from mimiqcircuits import *
            >>> c = Circuit()
            >>> c.push(GateH(), 0)
            1-qubit circuit with 1 instruction:
            └── H @ q[0]
            <BLANKLINE>
            >>> c.push(GateCCX(), 0, 1, 2)
            3-qubit circuit with 2 instructions:
            ├── H @ q[0]
            └── C₂X @ q[0:1], q[2]
            <BLANKLINE>
            >>> decomposed = c.decompose()
            >>> # every gate is now a GateU or a GateCX
            >>> sorted({type(i.operation).__name__ for i in decomposed})
            ['GateCX', 'GateU']

            Decompose to Clifford+T:

            >>> from mimiqcircuits import CliffordTBasis
            >>> decomposed = c.decompose(CliffordTBasis())

            Use a specific rewrite rule:

            >>> from mimiqcircuits import ZYZRewrite
            >>> decomposed = c.decompose(ZYZRewrite())

        See Also:
            :func:`mimiqcircuits.decompose`: Module-level decompose function
            :class:`mimiqcircuits.CanonicalBasis`: Default decomposition basis
            :class:`mimiqcircuits.CliffordTBasis`: Clifford+T decomposition
        """
        from mimiqcircuits.decomposition import decompose

        return decompose(self, basis)

    def decompose_step(self, rule=None):
        """Apply a single decomposition step to all matching operations.

        Unlike :meth:`decompose`, this method is non-recursive. Matching
        operations are rewritten once and the resulting instructions are kept
        as-is, even if they are still non-terminal in the canonical basis.

        Args:
            rule: The rewrite rule to apply. Defaults to
                  :class:`CanonicalRewrite`.

        Returns:
            Circuit: A new circuit with one decomposition step applied.

        Examples:
            Apply the default canonical rewrite once:

            >>> from mimiqcircuits import *
            >>> c = Circuit()
            >>> c.push(GateH(), 0)
            1-qubit circuit with 1 instruction:
            └── H @ q[0]
            <BLANKLINE>
            >>> c.decompose_step()
            1-qubit circuit with 1 instruction:
            └── U((1/2)*pi, 0, pi, 0.0) @ q[0]
            <BLANKLINE>

            Apply a specific rewrite rule once:

            >>> from symengine import pi
            >>> from mimiqcircuits import ZYZRewrite
            >>> c = Circuit().push(GateU(pi / 2, pi / 4, pi / 3), 0)
            >>> c.decompose_step(ZYZRewrite())
            1-qubit circuit with 3 instructions:
            ├── RZ((1/3)*pi) @ q[0]
            ├── RY((1/2)*pi) @ q[0]
            └── RZ((1/4)*pi) @ q[0]
            <BLANKLINE>

        See Also:
            :func:`mimiqcircuits.decompose_step`: Module-level single-step
            decomposition function
            :meth:`decompose`: Recursive decomposition to a target basis
        """
        from mimiqcircuits.decomposition import decompose_step

        return decompose_step(self, rule)

    def evaluate(self, d):
        """Return a copy of the circuit with symbols replaced by values.

        Each instruction is evaluated with ``d``. Symbols that are not in
        ``d`` stay symbolic. The circuit itself is not modified.

        Args:
            d: Dictionary from symbols to values.

        Returns:
            Circuit: A new circuit with the substituted parameters.

        Examples:
            >>> import mimiqcircuits as mc
            >>> from symengine import symbols
            >>> theta, phi = symbols("theta phi")
            >>> c = mc.Circuit().push(mc.GateRX(theta), 0).push(mc.GateRY(phi), 0)
            >>> c.evaluate({theta: 0.5})
            1-qubit circuit with 2 instructions:
            ├── RX(0.5) @ q[0]
            └── RY(phi) @ q[0]
            <BLANKLINE>
        """
        # symengine rebuilds its internal map from a Python dict on every
        # `subs` call, so convert once here instead of once per parameter.
        if isinstance(d, dict):
            d = se.DictBasic(d)

        c = Circuit()

        for inst in self.instructions:
            c.push(inst.evaluate(d))

        return c

    def __len__(self):
        return len(self.instructions)

    def __iter__(self):
        return iter(self.instructions)

    def __getitem__(self, index):
        if type(index) is slice:
            return Circuit(self.instructions[index])
        return self.instructions[index]

    def _header_numbers(self):
        nq = self.num_qubits()
        nb = self.num_bits()
        nz = self.num_zvars()

        parts = []
        if nq > 0:
            parts.append(f"{nq}-qubit")
        if nb > 0:
            parts.append(f"{nb}-bit")
        if nz > 0:
            parts.append(f"{nz}-zvar")

        return ", ".join(parts)

    def _header(self):
        if self.empty():
            return "empty circuit"

        n = len(self)
        numbers = self._header_numbers()

        if numbers:
            return f"{numbers} circuit with {n} instruction{'s' if n != 1 else ''}"
        else:
            return f"circuit with {n} instruction{'s' if n != 1 else ''}"

    def __str__(self, compact=False):
        lines = shutil.get_terminal_size().lines
        n = len(self)

        output = ""

        if compact and n > 0:
            output += f"{self._header()}:\n"

            if lines - 4 <= 0:
                output += "└── ...\n"
            else:
                max_display = min(n, lines - 4)

                for g in self.instructions[: max_display - 1]:
                    output += f"├── {g}\n"

                if n > max_display:
                    output += "⋮   ⋮\n"

                g = self.instructions[-1]
                output += f"└── {g}\n"

        elif not compact and n > 0:
            output = f"{self._header()}:"

            # iterate from the second gate
            for g in self.instructions[:-1]:
                output += f"\n├── {g}"

            g = self.instructions[-1]
            output += f"\n└── {g}"

        elif n == 0:
            output += "empty circuit"

        return output

    def __repr__(self):
        return self.__str__(compact=True)

    def __eq__(self, other):
        if not isinstance(other, Circuit):
            return False

        return self.instructions == other.instructions

    def __ne__(self, other):
        return not self.__eq__(other)

    def depth(self):
        """Return the depth of the circuit.

        The depth is the length of the longest chain of instructions that
        share a qubit, a classical bit or a z-variable. Barriers are not
        counted. A circuit without qubits has depth 0.

        Returns:
            int: The depth.

        Examples:
            >>> from mimiqcircuits import *
            >>> c = Circuit()
            >>> _ = c.push(GateH(), [0, 1])     # one layer: H on both qubits
            >>> _ = c.push(GateCX(), 0, 1)      # a second layer
            >>> _ = c.push(Measure(), 0, 0)     # a third layer, on qubit 0 only
            >>> c.depth()
            3
        """
        if self.empty() or self.num_qubits() == 0:
            return 0

        d = [0 for _ in range(self.num_qubits() + self.num_bits() + self.num_zvars())]

        for g in self:
            if isinstance(g.operation, Barrier):
                continue
            nq = self.num_qubits()
            nb = self.num_bits()
            optargets = (
                g.qubits
                + tuple(map(lambda x: x + nq, g.bits))
                + tuple(map(lambda x: x + nq + nb, g.zvars))
            )
            dm = max([d[t] for t in optargets])
            for t in g.qubits:
                d[t] = dm + 1
            for t in g.bits:
                d[t + nq] = dm + 1
            for t in g.zvars:
                d[t + nq + nb] = dm + 1

        return max(d)

    def copy(self):
        """Return a shallow copy of the circuit.

        The copy has its own list of instructions, so adding an instruction
        to one circuit does not change the other. The instructions themselves
        are shared. Use :meth:`deepcopy` to copy them too.

        Returns:
            Circuit: The shallow copy.
        """
        return copy.copy(self)

    def __copy__(self):
        new = type(self).__new__(type(self))
        new.__dict__.update(self.__dict__)
        new._instructions = list(self._instructions)
        return new

    def reorder_qubits(self, perm):
        """Return a new circuit with every qubit index sent through ``perm``.

        An instruction acting on qubit ``q`` acts on qubit ``perm[q]`` in the
        result. Operations that refer to qubits outside their targets are
        rewritten too: the bitstring of an :class:`Amplitude` (also inside
        an :class:`IfStatement`, :class:`Repeat` or :class:`Block`) is
        reordered so it reads the same amplitude as before.

        Classical bits and z-variables are not renumbered, so the result
        produces the same classical and z-variable outcomes as the original.

        Args:
            perm: A permutation of ``range(self.num_qubits())``, as a
                sequence where ``perm[q]`` is the new index of qubit ``q``.

        Returns:
            Circuit: The relabeled circuit.

        Raises:
            ValueError: If ``perm`` is not a permutation of
                ``range(self.num_qubits())``.

        Examples:
            >>> from mimiqcircuits import *
            >>> c = Circuit().push(GateH(), 0).push(GateCX(), 0, 1)
            >>> # swap the roles of qubits 0 and 1
            >>> c.reorder_qubits([1, 0])
            2-qubit circuit with 2 instructions:
            ├── H @ q[1]
            └── CX @ q[1], q[0]
            <BLANKLINE>
        """
        nq = self.num_qubits()
        if sorted(perm) != list(range(nq)):
            raise ValueError(
                f"perm must be a permutation of range({nq}); "
                f"got {perm!r}"
            )
        out = Circuit()
        for inst in self.instructions:
            op = inst.operation
            new_qubits = tuple(perm[q] for q in inst.qubits)
            new_op = _reorder_op_internals(op, perm)
            out.push(new_op, *new_qubits, *inst.bits, *inst.zvars)
        return out

    def deepcopy(self):
        """Return a deep copy of the circuit.

        The instructions and their operations are copied too, so the result
        shares nothing with the original.

        Returns:
            Circuit: The independent copy.
        """
        return copy.deepcopy(self)

    def get_on_qubits(self, target_qubits):
        """Return the instructions that act on any of the given qubits.

        Args:
            target_qubits (int or list of int): The qubits to look for.

        Returns:
            Circuit: A new circuit with the matching instructions, in their
            original order and with their original targets.

        Examples:
            >>> from mimiqcircuits import *
            >>> c = Circuit().push(GateH(), 0).push(GateX(), 2).push(GateCX(), 0, 1)
            >>> c.get_on_qubits(1)
            2-qubit circuit with 1 instruction:
            └── CX @ q[0], q[1]
            <BLANKLINE>
        """
        if isinstance(target_qubits, int):
            target_qubits = [target_qubits]

        selected_instructions = []

        for instruction in self.instructions:
            if any(qubit in instruction.qubits for qubit in target_qubits):
                selected_instructions.append(instruction)

        return Circuit(instructions=selected_instructions)

    def saveproto(self, file):
        """Save the circuit to a protobuf (binary) file.

        Symbolic parameters are saved as expressions. Read the file back with
        :meth:`loadproto`.

        Args:
            file: A file path or a binary file-like object to write to.

        Returns:
            int: The number of bytes written.

        Examples:
            >>> from mimiqcircuits import *
            >>> from symengine import *
            >>> import tempfile
            >>> x, y = symbols("x y")
            >>> c = Circuit()
            >>> c.push(GateH(), 0)
            1-qubit circuit with 1 instruction:
            └── H @ q[0]
            <BLANKLINE>
            >>> c.push(GateXXplusYY(x**2, y),0,1)
            2-qubit circuit with 2 instructions:
            ├── H @ q[0]
            └── XXplusYY(x**2, y) @ q[0:1]
            <BLANKLINE>
            >>> c.push(Measure(),0,0)
            2-qubit, 1-bit circuit with 3 instructions:
            ├── H @ q[0]
            ├── XXplusYY(x**2, y) @ q[0:1]
            └── M @ q[0], c[0]
            <BLANKLINE>
            >>> # a temporary file keeps the example self-contained; any path works
            >>> tmpfile = tempfile.NamedTemporaryFile(suffix=".pb", delete=True)
            >>> c.saveproto(tmpfile.name)
            64
            >>> Circuit.loadproto(tmpfile.name)
            2-qubit, 1-bit circuit with 3 instructions:
            ├── H @ q[0]
            ├── XXplusYY(x**2, y) @ q[0:1]
            └── M @ q[0], c[0]
            <BLANKLINE>
        """
        from mimiqcircuits.proto.protoio import saveproto

        return saveproto(self, file)

    @staticmethod
    def loadproto(file):
        """Load a circuit from a protobuf (binary) file written by :meth:`saveproto`.

        Args:
            file: A file path or a binary file-like object to read from.

        Returns:
            Circuit: The loaded circuit.

        Raises:
            ValueError: If the file cannot be parsed.

        See Also:
            :meth:`saveproto`, which has an example of saving and loading.
        """
        from mimiqcircuits.proto.protoio import loadproto

        return loadproto(file, Circuit)

    def draw(self, compact=True):
        """Print an ASCII drawing of the circuit.

        Each qubit gets a wire; the classical bits share one double wire, and
        so do the z-variables. When the drawing
        reaches the width of the canvas, the part drawn so far is printed,
        followed by ``...``, and drawing continues on a fresh canvas.

        Args:
            compact (bool): If ``True`` (the default), instructions on
                disjoint wires share a column, and consecutive instructions of
                the same operation on disjoint wires, such as those from one
                broadcast :meth:`push`, start in the same column. Instructions
                that touch the classical or z-variable register are always
                drawn one after the other. If ``False``, every instruction gets
                its own column, in circuit order.

        Returns:
            None: The drawing is printed, not returned.

        Raises:
            ValueError: If a single instruction is wider than the canvas.

        Examples:
            >>> from mimiqcircuits import *
            >>> c = Circuit().push(GateH(), 0).push(GateCX(), 0, 1)
            >>> c.draw()  # doctest: +SKIP
        """
        num_qubits = self.num_qubits()
        num_bits = self.num_bits()
        num_zvars = self.num_zvars()

        canvas = AsciiCircuit()
        canvas.draw_wires(range(num_qubits), range(num_bits), range(num_zvars))
        lastcol = canvas.get_current_col()

        # Each instruction goes in the first column free on all the rows it
        # blocks, so independent instructions share columns. Instructions that
        # block a common row keep their circuit order.
        if compact:
            groups = canvas.groups(self.instructions)
        else:
            groups = [[instruction] for instruction in self.instructions]

        for group in groups:
            rows = [canvas.row_span(instruction) for instruction in group]
            if compact:
                col = max(canvas.frontier[r] for span in rows for r in span)
            else:
                col = lastcol
            required_width = max(
                instruction.get_operation().asciiwidth(
                    instruction.qubits, instruction.bits, instruction.zvars
                )
                for instruction in group
            )

            if required_width >= canvas.canvas.get_cols() - col:
                print(canvas.canvas)
                print("...")
                canvas.reset()
                canvas.draw_wires(range(num_qubits), range(num_bits), range(num_zvars))
                lastcol = col = canvas.get_current_col()

            if required_width >= canvas.canvas.get_cols() - col:
                raise ValueError(
                    "Cannot draw instruction. Insufficient space on screen."
                )

            for instruction, span in zip(group, rows):
                canvas.currentcol = col
                operation = instruction.get_operation()

                # Handle drawing based on operation type
                if isinstance(operation, Control):
                    canvas.draw_control(
                        operation,
                        instruction.qubits,
                        instruction.bits if hasattr(instruction, "bits") else [],
                    )

                elif isinstance(operation, PauliString):
                    canvas.draw_paulistring(
                        operation,
                        instruction.qubits,
                        instruction.bits if hasattr(instruction, "bits") else [],
                        instruction.zvars if hasattr(instruction, "zvars") else [],
                    )

                elif isinstance(operation, Reset):
                    canvas.draw_reset(
                        operation,
                        instruction.qubits,
                        instruction.bits if hasattr(instruction, "bits") else [],
                        instruction.zvars if hasattr(instruction, "zvars") else [],
                    )

                elif isinstance(operation, Barrier):
                    canvas.draw_barrier(
                        operation,
                        instruction.qubits,
                        instruction.bits if hasattr(instruction, "bits") else [],
                        instruction.zvars if hasattr(instruction, "zvars") else [],
                    )

                elif isinstance(operation, Parallel):
                    canvas.draw_parallel(
                        operation,
                        instruction.qubits,
                        instruction.bits if hasattr(instruction, "bits") else [],
                        instruction.zvars if hasattr(instruction, "zvars") else [],
                    )

                elif isinstance(operation, IfStatement):
                    canvas.draw_ifstatement(
                        operation,
                        instruction.qubits,
                        instruction.bits if hasattr(instruction, "bits") else [],
                        instruction.zvars if hasattr(instruction, "zvars") else [],
                    )

                elif isinstance(operation, Operation):
                    canvas.draw_operation(
                        operation,
                        instruction.qubits,
                        instruction.bits if hasattr(instruction, "bits") else [],
                        instruction.zvars if hasattr(instruction, "zvars") else [],
                    )
                else:
                    # Default drawing method for general operations
                    canvas.draw_instruction(instruction)

                for r in span:
                    canvas.frontier[r] = canvas.get_current_col()
                lastcol = max(lastcol, canvas.get_current_col())

        print(canvas.canvas)
        return None

    def specify_operations(self):
        """Print how many instructions the circuit has, grouped by their number of targets.

        Instructions are grouped by how many qubits, bits and z-variables
        they act on, so an ``H`` and an ``X`` fall in the same group. Groups
        are listed in the order they first appear in the circuit.

        Returns:
            None: The summary is printed, not returned.

        Examples:
            >>> from mimiqcircuits import *
            >>> c = Circuit()

            Add a Pauli-X (GateX) gate on qubit 0

            >>> c.push(GateX(), 0)
            1-qubit circuit with 1 instruction:
            └── X @ q[0]
            <BLANKLINE>

            Add a Controlled-NOT (CX) gate with control qubit 0 and target qubit 1

            >>> c.push(GateCX(), 0, 1)
            2-qubit circuit with 2 instructions:
            ├── X @ q[0]
            └── CX @ q[0], q[1]
            <BLANKLINE>

            Add a Measurement operation on qubit 0, storing the result in bit 0

            >>> c.push(Measure(), 0, 0)
            2-qubit, 1-bit circuit with 3 instructions:
            ├── X @ q[0]
            ├── CX @ q[0], q[1]
            └── M @ q[0], c[0]
            <BLANKLINE>

            Add an ExpectationValue operation with GateX on qubit 1, storing the result in z-variable 2.

            >>> c.push(ExpectationValue(GateX()), 1, 2)
            2-qubit, 1-bit, 3-zvar circuit with 4 instructions:
            ├── X @ q[0]
            ├── CX @ q[0], q[1]
            ├── M @ q[0], c[0]
            └── ⟨X⟩ @ q[1], z[2]
            <BLANKLINE>

            Print a summary of the types and numbers of operations

            >>> c.specify_operations()
            Total number of operations: 4
            ├── 1 x 1_qubits
            ├── 1 x 2_qubits
            ├── 1 x 1_qubits & 1_bits
            └── 1 x 1_qubits & 1_zvars
        """
        counts = {}
        for i in self.instructions:
            nq = len(i._qubits)
            nb = len(i._bits)
            nz = len(i._zvars)

            key = ""

            if nq > 0:
                key += f"{nq}_qubits"

            if nb > 0:
                if key:
                    key += " & "
                key += f"{nb}_bits"
            if nz > 0:
                if key:
                    key += " & "
                key += f"{nz}_zvars"

            counts[key] = counts.get(key, 0) + 1

        total_operations = sum(counts.values())
        print(f"Total number of operations: {total_operations}")

        count_items = list(counts.items())
        for idx, (key, count) in enumerate(count_items):
            if idx == len(count_items) - 1:
                print(f"└── {count} x {key}")
            else:
                print(f"├── {count} x {key}")

    def is_symbolic(self):
        """Return ``True`` if any operation of the circuit has a symbolic parameter.

        Operations nested inside wrappers such as :class:`Control` or
        :class:`Block` are checked too. Use :meth:`evaluate` to substitute
        values and :meth:`listvars` to see which symbols remain.

        Returns:
            bool: ``True`` if at least one parameter is still symbolic.

        Examples:
            >>> from mimiqcircuits import *
            >>> from symengine import *
            >>> x, y = symbols("x y")
            >>> c = Circuit()
            >>> c.push(GateH(), 0)
            1-qubit circuit with 1 instruction:
            └── H @ q[0]
            <BLANKLINE>
            >>> c.is_symbolic()
            False
            >>> c.push(GateP(x), 0)
            1-qubit circuit with 2 instructions:
            ├── H @ q[0]
            └── P(x) @ q[0]
            <BLANKLINE>
            >>> c.is_symbolic()
            True
            >>> c = c.evaluate({x: 1, y: 2})
            >>> c
            1-qubit circuit with 2 instructions:
            ├── H @ q[0]
            └── P(1) @ q[0]
            <BLANKLINE>
            >>> c.is_symbolic()
            False
        """
        return any(
            instruction.operation.is_symbolic() for instruction in self.instructions
        )

    def decorate_on_match_single(
        self,
        g: Operation,
        decoration: Union[krauschannel, Gate],
        before: bool = False,
    ):
        """Return a copy of the circuit with ``decoration`` added next to each ``g``.

        Every instruction whose operation equals ``g`` gets ``decoration``
        right after it (or right before it), on the same qubits. The circuit
        itself is not changed; see :meth:`add_noise` for the in-place
        version.

        Args:
            g: The operation to match, compared with ``==``.
            decoration: The gate or Kraus channel to insert. It must act on
                as many qubits as ``g``.
            before: If ``True``, insert the decoration before each match
                instead of after it.

        Returns:
            Circuit: A new, decorated circuit.

        Raises:
            ValueError: If ``decoration`` equals ``g``, or acts on a different
                number of qubits.

        Examples:
            >>> from mimiqcircuits import *
            >>> c = Circuit()
            >>> c.push(GateH(), 0)
            1-qubit circuit with 1 instruction:
            └── H @ q[0]
            <BLANKLINE>
            >>> c.decorate_on_match_single(GateH(), AmplitudeDamping(0.2))
            1-qubit circuit with 2 instructions:
            ├── H @ q[0]
            └── AmplitudeDamping(0.2) @ q[0]
            <BLANKLINE>
            >>> # the original circuit is unchanged
            >>> c
            1-qubit circuit with 1 instruction:
            └── H @ q[0]
            <BLANKLINE>
        """
        from mimiqcircuits.addnoise import decorate_on_match_single

        if g == decoration:
            raise ValueError(
                "Decoration operation cannot be the same as the target operation."
            )

        def matcher(inst):
            return inst.operation == g

        def generator(inst):
            # Returns (operation, targets) tuple for insertion helper
            return (decoration, inst.qubits)

        return decorate_on_match_single(self, matcher, generator, before=before)

    def decorate_on_match_parallel(
        self,
        g: Operation,
        decoration: Union[krauschannel, Gate],
        before: bool = False,
    ):
        """Add ``decoration`` after (or before) each block of parallel ``g`` instructions.

        A block is a run of consecutive instructions equal to ``g`` that act
        on disjoint qubits, such as the instructions from one broadcast
        :meth:`push`. The decorations for the whole block are inserted
        together, after the block or before it, each on the qubits of one
        matched instruction. The circuit is changed in place.

        Args:
            g: The operation to match, compared with ``==``.
            decoration: The gate or Kraus channel to insert. It must act on
                as many qubits as ``g``.
            before: If ``True``, insert the decorations before each block
                instead of after it.

        Returns:
            Circuit: The circuit itself, now decorated.

        Raises:
            ValueError: If ``decoration`` equals ``g``, or acts on a different
                number of qubits.

        Examples:
            >>> from mimiqcircuits import *
            >>> c = Circuit()
            >>> c.push(GateH(), range(3))
            3-qubit circuit with 3 instructions:
            ├── H @ q[0]
            ├── H @ q[1]
            └── H @ q[2]
            <BLANKLINE>
            >>> c.decorate_on_match_parallel(GateH(), AmplitudeDamping(0.2))
            3-qubit circuit with 6 instructions:
            ├── H @ q[0]
            ├── H @ q[1]
            ├── H @ q[2]
            ├── AmplitudeDamping(0.2) @ q[0]
            ├── AmplitudeDamping(0.2) @ q[1]
            └── AmplitudeDamping(0.2) @ q[2]
            <BLANKLINE>
        """
        from mimiqcircuits.addnoise import decorate_on_match_parallel_inplace

        if g == decoration:
            raise ValueError(
                "Decoration operation cannot be the same as the target operation."
            )

        def matcher(inst):
            return inst.operation == g

        def generator(inst):
            # Returns (operation, targets) tuple for insertion helper
            return (decoration, inst.qubits)

        return decorate_on_match_parallel_inplace(
            self, matcher, generator, before=before
        )

    def add_noise(
        self,
        g: Union[Operation, List[Operation]],
        kraus: Union[krauschannel, Gate, List[Gate], List[krauschannel]],
        before: Union[bool, List[bool]] = False,
        parallel: Union[bool, List[bool]] = False,
    ):
        """Add the noise ``kraus`` next to every instance of ``g``, in place.

        ``kraus`` acts on the same qubits as each matched instruction, so it
        must act on as many qubits as ``g``. By default it goes right after
        each match. With ``parallel=True``, runs of consecutive matches on
        disjoint qubits (as produced by one broadcast :meth:`push`) are kept
        together and their noise is added after the whole run, as in
        :meth:`decorate_on_match_parallel`.

        For rules that depend on more than the operation, or for many rules at
        once, use a :class:`NoiseModel`.

        Args:
            g: The operation to match. An operation instance matches
                instructions whose operation is equal to it; an operation
                class, such as ``GateH``, matches every instance of it.
            kraus: The Kraus channel or gate to add.
            before: If ``True``, add the noise before each match instead of
                after it. Mostly used with :class:`Measure`.
            parallel: If ``True``, add the noise after (or before) each run of
                parallel matches instead of after each match.

        Returns:
            Circuit: The circuit itself, with the noise added.

        Raises:
            TypeError: If ``kraus`` is not a gate or a Kraus channel, or ``g``
                is not an operation or an operation class.
            ValueError: If ``kraus`` acts on a different number of qubits
                than the matched instructions.

        Examples:
            Adding noise after each gate:

            >>> from mimiqcircuits import *
            >>> c = Circuit()
            >>> c.push(GateH(), [1,2,3])
            4-qubit circuit with 3 instructions:
            ├── H @ q[1]
            ├── H @ q[2]
            └── H @ q[3]
            <BLANKLINE>
            >>> c.add_noise(GateH(), AmplitudeDamping(0.2))
            4-qubit circuit with 6 instructions:
            ├── H @ q[1]
            ├── AmplitudeDamping(0.2) @ q[1]
            ├── H @ q[2]
            ├── AmplitudeDamping(0.2) @ q[2]
            ├── H @ q[3]
            └── AmplitudeDamping(0.2) @ q[3]
            <BLANKLINE>

            Adding the noise after each run of parallel gates:

            >>> c = Circuit()
            >>> c.push(GateH(), [1, 2, 3])
            4-qubit circuit with 3 instructions:
            ├── H @ q[1]
            ├── H @ q[2]
            └── H @ q[3]
            <BLANKLINE>
            >>> c.add_noise(GateH(), AmplitudeDamping(0.2), parallel=True)
            4-qubit circuit with 6 instructions:
            ├── H @ q[1]
            ├── H @ q[2]
            ├── H @ q[3]
            ├── AmplitudeDamping(0.2) @ q[1]
            ├── AmplitudeDamping(0.2) @ q[2]
            └── AmplitudeDamping(0.2) @ q[3]
            <BLANKLINE>

            Gates that share a qubit are not parallel, so each one is followed by
            its own noise:

            >>> c = Circuit()
            >>> c.push(GateCZ(), 1, range(2,5))
            5-qubit circuit with 3 instructions:
            ├── CZ @ q[1], q[2]
            ├── CZ @ q[1], q[3]
            └── CZ @ q[1], q[4]
            <BLANKLINE>
            >>> c.add_noise(GateCZ(), Depolarizing2(0.1), parallel=True)
            5-qubit circuit with 6 instructions:
            ├── CZ @ q[1], q[2]
            ├── Depolarizing(0.1) @ q[1:2]
            ├── CZ @ q[1], q[3]
            ├── Depolarizing(0.1) @ q[1,3]
            ├── CZ @ q[1], q[4]
            └── Depolarizing(0.1) @ q[1,4]
            <BLANKLINE>

            Adding a bit-flip before each measurement, to model readout errors:

            >>> c = Circuit()
            >>> c.push(Measure(), [1, 2, 3], [1, 2, 3])
            4-qubit, 4-bit circuit with 3 instructions:
            ├── M @ q[1], c[1]
            ├── M @ q[2], c[2]
            └── M @ q[3], c[3]
            <BLANKLINE>
            >>> c.add_noise(Measure(), PauliX(0.1), before=True)
            4-qubit, 4-bit circuit with 6 instructions:
            ├── PauliX(0.1) @ q[1]
            ├── M @ q[1], c[1]
            ├── PauliX(0.1) @ q[2]
            ├── M @ q[2], c[2]
            ├── PauliX(0.1) @ q[3]
            └── M @ q[3], c[3]
            <BLANKLINE>

            A unitary gate can be used as noise too, for example a small
            over-rotation:

            >>> c = Circuit()
            >>> c.push(GateH(), [1, 2, 3])
            4-qubit circuit with 3 instructions:
            ├── H @ q[1]
            ├── H @ q[2]
            └── H @ q[3]
            <BLANKLINE>
            >>> c.add_noise(GateH(), GateRX(0.01))
            4-qubit circuit with 6 instructions:
            ├── H @ q[1]
            ├── RX(0.01) @ q[1]
            ├── H @ q[2]
            ├── RX(0.01) @ q[2]
            ├── H @ q[3]
            └── RX(0.01) @ q[3]
            <BLANKLINE>

        """
        from mimiqcircuits.addnoise import add_noise_inplace

        return add_noise_inplace(self, g, kraus, before, parallel)

    def sample_mixedunitaries(self, rng=None, ids=False):
        """Return a copy of the circuit with each mixed-unitary channel replaced by one sampled gate.

        A mixed-unitary channel applies one of a fixed set of unitaries with
        fixed probabilities that do not depend on the state, so the choice can
        be made before running the circuit. Each such channel is replaced by
        the gate drawn with its probability. Other channels and operations are
        kept as they are. This is how noisy circuits are sampled when
        simulating with trajectories, and it can be used to draw noisy
        circuits without running them.

        If the drawn branch is marked lossy (see the ``lossy`` argument of
        :class:`MixedUnitary`), a ``Loss(1.0)`` is also added on each of its
        lossy qubits, to be resolved together with any other loss in the
        circuit.

        Args:
            rng (random.Random): Random number generator with a ``random()``
                method. Defaults to a new ``random.Random()``.
            ids (bool): If ``True``, a drawn identity gate is kept in the
                circuit. By default it is left out; most draws of a weak
                noise channel are the identity.

        Returns:
            Circuit: The sampled circuit. The circuit itself is not changed.

        See Also:
            :meth:`krauschannel.ismixedunitary`, :class:`MixedUnitary`

        Examples:
            Gates and non-mixed-unitary Kraus channels remain unchanged:

            >>> from mimiqcircuits import *
            >>> c = Circuit()
            >>> c.push(GateH(), [1, 2, 3])
            4-qubit circuit with 3 instructions:
            ├── H @ q[1]
            ├── H @ q[2]
            └── H @ q[3]
            <BLANKLINE>
            >>> c.push(Depolarizing1(0.5), [1, 2, 3])
            4-qubit circuit with 6 instructions:
            ├── H @ q[1]
            ├── H @ q[2]
            ├── H @ q[3]
            ├── Depolarizing(0.5) @ q[1]
            ├── Depolarizing(0.5) @ q[2]
            └── Depolarizing(0.5) @ q[3]
            <BLANKLINE>
            >>> c.push(AmplitudeDamping(0.5), [1, 2, 3])
            4-qubit circuit with 9 instructions:
            ├── H @ q[1]
            ├── H @ q[2]
            ├── H @ q[3]
            ├── Depolarizing(0.5) @ q[1]
            ├── Depolarizing(0.5) @ q[2]
            ├── Depolarizing(0.5) @ q[3]
            ├── AmplitudeDamping(0.5) @ q[1]
            ├── AmplitudeDamping(0.5) @ q[2]
            └── AmplitudeDamping(0.5) @ q[3]
            <BLANKLINE>

            Each depolarizing channel becomes one Pauli gate; amplitude
            damping is not mixed-unitary and stays. A seeded generator makes
            the draws reproducible:

            >>> import random
            >>> rng = random.Random(42)
            >>> new_circuit = c.sample_mixedunitaries(rng=rng, ids=True)
            >>> print(new_circuit)
            4-qubit circuit with 9 instructions:
            ├── H @ q[1]
            ├── H @ q[2]
            ├── H @ q[3]
            ├── X @ q[1]
            ├── I @ q[2]
            ├── I @ q[3]
            ├── AmplitudeDamping(0.5) @ q[1]
            ├── AmplitudeDamping(0.5) @ q[2]
            └── AmplitudeDamping(0.5) @ q[3]

            By default, identities are not included:

            >>> new_circuit = c.sample_mixedunitaries(rng=rng)
            >>> print(new_circuit)
            4-qubit circuit with 8 instructions:
            ├── H @ q[1]
            ├── H @ q[2]
            ├── H @ q[3]
            ├── Y @ q[2]
            ├── Y @ q[3]
            ├── AmplitudeDamping(0.5) @ q[1]
            ├── AmplitudeDamping(0.5) @ q[2]
            └── AmplitudeDamping(0.5) @ q[3]

            Different calls to the function generate different results:

            >>> new_circuit = c.sample_mixedunitaries(rng=rng)
            >>> print(new_circuit)
            4-qubit circuit with 7 instructions:
            ├── H @ q[1]
            ├── H @ q[2]
            ├── H @ q[3]
            ├── Z @ q[1]
            ├── AmplitudeDamping(0.5) @ q[1]
            ├── AmplitudeDamping(0.5) @ q[2]
            └── AmplitudeDamping(0.5) @ q[3]

            >>> new_circuit = c.sample_mixedunitaries(rng=rng)
            >>> print(new_circuit)
            4-qubit circuit with 7 instructions:
            ├── H @ q[1]
            ├── H @ q[2]
            ├── H @ q[3]
            ├── X @ q[3]
            ├── AmplitudeDamping(0.5) @ q[1]
            ├── AmplitudeDamping(0.5) @ q[2]
            └── AmplitudeDamping(0.5) @ q[3]
        """

        if rng is None:
            rng = random.Random()

        scirc = Circuit()

        for inst in self.instructions:
            op = inst.get_operation()

            if isinstance(op, krauschannel) and op.ismixedunitary():
                cumulative_probs = op.unwrappedcumprobabilities()

                r = rng.random()

                # Use `next` to find the index of the first cumulative probability greater than `r`
                index = next(i for i, p in enumerate(cumulative_probs) if p > r)

                gate = op.unitarygates()[index]

                if ids or not gate.isidentity():
                    scirc.push(gate, *inst.qubits)

                # A lossy branch loses its marked qubits: emit a certain Loss, to be
                # resolved by the loss-lowering pass. The loss fires regardless of
                # whether the (possibly identity) gate was added.
                lossy = getattr(op, "lossy", None)
                if lossy is not None:
                    for q in lossy[index]:
                        scirc.push(Loss(1.0), inst.qubits[q - 1])
            else:
                scirc.push(inst)

        return scirc

    def sample_losses(self, rng=None):
        """Resolve the random qubit-loss events in this circuit.

        Same as :func:`mimiqcircuits.sample_losses` applied to this circuit.
        Each ``Loss(p)`` becomes ``Loss(1.0)`` with probability ``p`` and is
        dropped otherwise; everything else is kept.

        Args:
            rng (random.Random): Random number generator with a ``random()``
                method. Defaults to a new ``random.Random()``.

        Returns:
            Circuit: A new circuit. The circuit itself is not changed.

        Examples:
            >>> from mimiqcircuits import *
            >>> c = Circuit()
            >>> _ = c.push(Loss(), 1)
            >>> _ = c.push(GateH(), 1)
            >>> c.sample_losses()
            2-qubit circuit with 2 instructions:
            ├── Loss(1.0) @ q[1]
            └── H @ q[1]
            <BLANKLINE>
        """
        from mimiqcircuits.lossmodel import sample_losses

        return sample_losses(self, rng=rng)

    def lower_losses(self, rng=None, lossmodel=None):
        """Lower loss bookkeeping into primitives.

        Same as :func:`mimiqcircuits.lower_losses` applied to this circuit.
        It expects the randomness to be resolved already (see
        :meth:`sample_losses`) and treats any remaining ``Loss`` as a certain
        loss.

        Args:
            rng (random.Random): Random number generator for stochastic
                rules of ``lossmodel``.
            lossmodel (LossModel): How gates touching a lost qubit behave.

        Returns:
            Circuit: A new circuit with no ``Loss`` left.
        """
        from mimiqcircuits.lossmodel import lower_losses

        return lower_losses(self, rng=rng, lossmodel=lossmodel)

    def resolve_losses(self, rng=None, lossmodel=None):
        """Fully resolve loss into a runnable, loss-free circuit.

        Same as :func:`mimiqcircuits.resolve_losses` applied to this circuit:
        it runs :meth:`sample_losses`, then :meth:`lower_losses`.

        Args:
            rng (random.Random): Random number generator.
            lossmodel (LossModel): How gates touching a lost qubit behave.

        Returns:
            Circuit: A new circuit with no ``Loss`` left.

        Examples:
            In this example the ``CX`` touches the lost qubit 1, so it is
            dropped and qubit 1 is marked as lost:


            >>> from mimiqcircuits import *
            >>> c = Circuit()
            >>> _ = c.push(Loss(), 1)
            >>> _ = c.push(GateCX(), 0, 1)
            >>> c.resolve_losses()
            2-qubit circuit with 1 instruction:
            └── Lost @ q[1]
            <BLANKLINE>
        """
        from mimiqcircuits.lossmodel import resolve_losses

        return resolve_losses(self, rng=rng, lossmodel=lossmodel)

    def sample_loss_scenario(self, loss_indices, *, p=1.0, rng=None, lossmodel=None):
        """Build a deterministic "what-if" loss scenario from this circuit.

        Same as :func:`mimiqcircuits.sample_loss_scenario` applied to this
        circuit: the selected ``Loss`` instructions become ``Loss(p)``, the
        others ``Loss(0.0)``, and the result is resolved as by
        :meth:`resolve_losses`.

        Args:
            loss_indices (int or iterable of int): Which ``Loss``
                instructions fire, counted from 1 in circuit order among the
                ``Loss`` instructions only.
            p (float): The loss probability given to the selected sites.
            rng (random.Random): Random number generator.
            lossmodel (LossModel): How gates touching a lost qubit behave.

        Returns:
            Circuit: A new circuit with no ``Loss`` left.

        Raises:
            ValueError: If ``p`` is not between 0 and 1.

        Examples:
            Only the second ``Loss`` (on qubit 1) fires:


            >>> from mimiqcircuits import *
            >>> c = Circuit()
            >>> _ = c.push(Loss(0.2), 0)
            >>> _ = c.push(GateCX(), 0, 1)
            >>> _ = c.push(Loss(0.4), 1)
            >>> c.sample_loss_scenario(2)
            2-qubit circuit with 2 instructions:
            ├── CX @ q[0], q[1]
            └── Lost @ q[1]
            <BLANKLINE>
        """
        from mimiqcircuits.lossmodel import sample_loss_scenario

        return sample_loss_scenario(
            self, loss_indices, p=p, rng=rng, lossmodel=lossmodel
        )

    def pauli_based_computation(self, basis=None, magic_states=False):
        """Rewrite this circuit in Pauli-based computation form.

        Equivalent to ``pauli_based_computation(self, basis, magic_states)``.
        See :func:`mimiqcircuits.pauli_based_computation` for the arguments,
        the form of the result and the errors raised.

        Here an X gate followed by a measurement of qubit 0 becomes a
        measurement of :math:`Z_0` whose bit is then flipped, since
        :math:`X Z X = -Z`:

        Examples:
            >>> from mimiqcircuits import *
            >>> c = Circuit()
            >>> _ = c.push(GateX(), 0)
            >>> _ = c.push(Measure(), 0, 0)
            >>> c.pauli_based_computation()
            1-qubit, 1-bit circuit with 2 instructions:
            ├── MP(Z) @ q[0], c[0]
            └── c[0] = !c[0]
            <BLANKLINE>
        """
        from mimiqcircuits.paulibased import pauli_based_computation

        return pauli_based_computation(self, basis=basis, magic_states=magic_states)

    def remove_unused(self):
        """Renumber qubits, bits and z-variables so that none is left unused.

        Same as :func:`mimiqcircuits.remove_unused` applied to this circuit.
        The used indices of each register are renumbered ``0, 1, 2, ...`` in
        increasing order. The circuit itself is not changed.

        Returns:
            tuple: ``(new_circuit, qubit_map, bit_map, zvar_map)``, where each
            map is a dict from an old index to its new index.

        Examples:
            >>> from mimiqcircuits import *
            >>> c = Circuit()
            >>> c.push(GateX(), 10)
            11-qubit circuit with 1 instruction:
            └── X @ q[10]
            <BLANKLINE>
            >>> c.push(GateSWAP(), 1, 20)
            21-qubit circuit with 2 instructions:
            ├── X @ q[10]
            └── SWAP @ q[1,20]
            <BLANKLINE>
            >>> c.push(GateH(), 4)
            21-qubit circuit with 3 instructions:
            ├── X @ q[10]
            ├── SWAP @ q[1,20]
            └── H @ q[4]
            <BLANKLINE>
            >>> c.remove_unused()
            (4-qubit circuit with 3 instructions:
            ├── X @ q[2]
            ├── SWAP @ q[0,3]
            └── H @ q[1]
            , {1: 0, 4: 1, 10: 2, 20: 3}, {}, {})
        """
        from mimiqcircuits.circuit_extras import remove_unused

        return remove_unused(self)

    def remove_swaps(self, recursive=False):
        """Remove the SWAP gates by relabeling the qubits of the gates after them.

        Same as :func:`mimiqcircuits.remove_swaps` applied to this circuit;
        see there for the details. The circuit itself is not changed.

        Args:
            recursive: If ``True``, also remove SWAP gates inside declared
                gates (:class:`GateCall`), their inverses and
                :class:`Block` operations.

        Returns:
            tuple: ``(new_circuit, qubit_permutation)``. Running
            ``new_circuit`` leaves the state that the original circuit leaves
            on qubit ``i`` on qubit ``qubit_permutation[i]``.

        Examples:
            >>> from mimiqcircuits import *
            >>> c = Circuit()
            >>> c.push(GateH(), 1)
            2-qubit circuit with 1 instruction:
            └── H @ q[1]
            <BLANKLINE>
            >>> c.push(GateSWAP(), 1, 2)
            3-qubit circuit with 2 instructions:
            ├── H @ q[1]
            └── SWAP @ q[1:2]
            <BLANKLINE>
            >>> c.push(GateCX(), 2, 3)
            4-qubit circuit with 3 instructions:
            ├── H @ q[1]
            ├── SWAP @ q[1:2]
            └── CX @ q[2], q[3]
            <BLANKLINE>
            >>> new_c, perm = c.remove_swaps()
            >>> new_c
            4-qubit circuit with 2 instructions:
            ├── H @ q[1]
            └── CX @ q[1], q[3]
            <BLANKLINE>
            >>> # the state of qubit 1 ends on qubit 2, and that of qubit 2 on qubit 1
            >>> perm
            [0, 2, 1, 3]

            Multiple swaps example:

            >>> c2 = Circuit()
            >>> c2.push(GateSWAP(), 1, 2)
            3-qubit circuit with 1 instruction:
            └── SWAP @ q[1:2]
            <BLANKLINE>
            >>> c2.push(GateSWAP(), 2, 3)
            4-qubit circuit with 2 instructions:
            ├── SWAP @ q[1:2]
            └── SWAP @ q[2:3]
            <BLANKLINE>
            >>> c2.push(GateCX(), 1, 3)
            4-qubit circuit with 3 instructions:
            ├── SWAP @ q[1:2]
            ├── SWAP @ q[2:3]
            └── CX @ q[1], q[3]
            <BLANKLINE>
            >>> new_c2, perm2 = c2.remove_swaps()
            >>> # an identity on qubit 3 keeps the number of qubits unchanged
            >>> new_c2
            4-qubit circuit with 2 instructions:
            ├── CX @ q[2], q[1]
            └── ID @ q[3]
            <BLANKLINE>
            >>> # what the original circuit leaves on qubit 1 is on qubit 2 of
            >>> # new_c2, qubit 2 on qubit 3 and qubit 3 on qubit 1
            >>> perm2
            [0, 2, 3, 1]

        See Also:
            :func:`mimiqcircuits.remove_swaps`
        """
        from mimiqcircuits.circuit_extras import remove_swaps

        return remove_swaps(self, recursive=recursive)

    def isunitary(self):
        """Return ``True`` if every instruction of the circuit is unitary.

        An empty circuit is unitary. Measurements, resets and noise channels
        are not unitary.

        Examples:
            >>> from mimiqcircuits import *
            >>> c = Circuit().push(GateH(), 0)
            >>> c.isunitary()
            True
            >>> c.push(Measure(), 0, 0).isunitary()
            False
        """
        return all(inst.isunitary() for inst in self.instructions)


# export the cirucit classes
__all__ = ["Circuit"]
