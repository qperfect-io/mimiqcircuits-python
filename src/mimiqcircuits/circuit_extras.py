#
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

from mimiqcircuits.instruction import Instruction
from mimiqcircuits.operations.gates.standard.swap import GateSWAP
from mimiqcircuits.circuit import Circuit
from mimiqcircuits.operations.inverse import Inverse
from mimiqcircuits.operations.block import Block
from mimiqcircuits.gatedecl import GateDecl, GateCall
import mimiqcircuits as mc


def remove_unused(self):
    """
    Removes unused qubits, bits, and zvars from the given circuit.
    Returns (new_circuit, qubit_map, bit_map, zvar_map).

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(GateH(), 0)
        1-qubit circuit with 1 instruction:
        └── H @ q[0]
        <BLANKLINE>
        >>> c.push(GateCX(), 0, 2)  # qubit 1 is unused
        3-qubit circuit with 2 instructions:
        ├── H @ q[0]
        └── CX @ q[0], q[2]
        <BLANKLINE>
        >>> c.push(Measure(), 2, 2)  # bits 0 & 1 are unused
        3-qubit, 3-bit circuit with 3 instructions:
        ├── H @ q[0]
        ├── CX @ q[0], q[2]
        └── M @ q[2], c[2]
        <BLANKLINE>
        >>> c.push(ExpectationValue(GateX()), 0, 1)  # zvar 0 is unused
        3-qubit, 3-bit, 2-zvar circuit with 4 instructions:
        ├── H @ q[0]
        ├── CX @ q[0], q[2]
        ├── M @ q[2], c[2]
        └── ⟨X⟩ @ q[0], z[1]
        <BLANKLINE>
        >>> new_c, qmap, bmap, zmap = remove_unused(c)
        >>> new_c
        2-qubit, 1-bit, 1-zvar circuit with 4 instructions:
        ├── H @ q[0]
        ├── CX @ q[0], q[1]
        ├── M @ q[1], c[0]
        └── ⟨X⟩ @ q[0], z[0]
        <BLANKLINE>

        >>> qmap
        {0: 0, 2: 1}
        >>> bmap
        {2: 0}
        >>> zmap
        {1: 0}
    """
    used_qubits = set()
    used_bits = set()
    used_zvars = set()

    for g in self:
        used_qubits.update(g.get_qubits())
        used_bits.update(g.get_bits())
        used_zvars.update(g.get_zvars())

    qubit_map = {q: i for i, q in enumerate(sorted(used_qubits))}
    bit_map = {b: i for i, b in enumerate(sorted(used_bits))}
    zvar_map = {z: i for i, z in enumerate(sorted(used_zvars))}

    new_circuit = Circuit()

    for g in self:
        new_qubits = tuple(qubit_map[q] for q in g.get_qubits() if q in qubit_map)
        new_bits = tuple(bit_map[b] for b in g.get_bits() if b in bit_map)
        new_ztargets = tuple(zvar_map[z] for z in g.get_zvars() if z in zvar_map)

        new_instr = Instruction(g.get_operation(), new_qubits, new_bits, new_ztargets)
        new_circuit.append([new_instr])

    return new_circuit, qubit_map, bit_map, zvar_map


def remove_swaps(circuit, recursive=False, cache=None):
    """
    Remove all SWAP gates from a quantum circuit by tracking qubit permutations and
    remapping subsequent operations to their correct physical qubits.

    Returns:
        tuple: A tuple containing:
            - new_circuit: Circuit with SWAP gates removed and operations remapped
            - qubit_permutation: Final permutation where qubit_permutation[i] gives the
              physical qubit location of logical qubit i

    Args:
        circuit: Input quantum circuit
        recursive: If True, recursively remove swaps from nested blocks/subcircuits

    Details:
        When a SWAP gate is encountered on qubits (i, j), instead of keeping the gate:

        1. The qubit mapping is updated to track that logical qubits i and j have exchanged physical positions
        2. All subsequent gates are automatically remapped to operate on the correct physical qubits

        This transformation preserves circuit semantics while eliminating SWAP overhead.

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
        >>> new_c, perm = remove_swaps(c)
        >>> new_c
        4-qubit circuit with 2 instructions:
        ├── H @ q[1]
        └── CX @ q[1], q[3]
        <BLANKLINE>
        >>> perm
        [0, 2, 1, 3]

        >>> c = Circuit()
        >>> c.push(GateSWAP(), 1, 2)
        3-qubit circuit with 1 instruction:
        └── SWAP @ q[1:2]
        <BLANKLINE>
        >>> c.push(GateSWAP(), 2, 3)
        4-qubit circuit with 2 instructions:
        ├── SWAP @ q[1:2]
        └── SWAP @ q[2:3]
        <BLANKLINE>
        >>> c.push(GateCX(), 1, 3)
        4-qubit circuit with 3 instructions:
        ├── SWAP @ q[1:2]
        ├── SWAP @ q[2:3]
        └── CX @ q[1], q[3]
        <BLANKLINE>
        >>> new_c, perm = remove_swaps(c)
        >>> new_c
        3-qubit circuit with 1 instruction:
        └── CX @ q[2], q[1]
        <BLANKLINE>
        >>> perm
        [0, 2, 3, 1]

        # Example of checking recursive removel of swaps

        >>> inner = Circuit()
        >>> _ = inner.push(GateSWAP(), 0, 1)
        >>> _ = inner.push(GateCX(), 1, 0)
        >>> Inner = GateDecl("Inner", [], inner)

        >>> mid = Circuit()
        >>> _ = mid.push(GateSWAP(), 0, 1)
        >>> _ = mid.push(GateCall(Inner, ()), 0, 1)
        >>> Mid = GateDecl("Mid", [], mid)

        >>> outer = Circuit()
        >>> _ = outer.push(GateSWAP(), 1, 2)
        >>> _ = outer.push(GateCall(Mid, ()), 1, 2)
        >>> Outer = GateDecl("Outer", [], outer)

        >>> c = Circuit()
        >>> _ = c.push(GateSWAP(), 1, 2)
        >>> _ = c.push(GateCall(Outer, ()), 0, 1, 2)

        >>> c
        3-qubit circuit with 2 instructions:
        ├── SWAP @ q[1:2]
        └── Outer() @ q[0:2]
        <BLANKLINE>

        >>> c.decompose()
        3-qubit circuit with 13 instructions:
        ├── CX @ q[1], q[2]
        ├── CX @ q[2], q[1]
        ├── CX @ q[1], q[2]
        ├── CX @ q[1], q[2]
        ├── CX @ q[2], q[1]
        ├── CX @ q[1], q[2]
        ├── CX @ q[1], q[2]
        ├── CX @ q[2], q[1]
        ├── CX @ q[1], q[2]
        ├── CX @ q[1], q[2]
        ├── CX @ q[2], q[1]
        ├── CX @ q[1], q[2]
        └── CX @ q[2], q[1]
        <BLANKLINE>

        >>> c.decompose().decompose()
        3-qubit circuit with 13 instructions:
        ├── CX @ q[1], q[2]
        ├── CX @ q[2], q[1]
        ├── CX @ q[1], q[2]
        ├── CX @ q[1], q[2]
        ├── CX @ q[2], q[1]
        ├── CX @ q[1], q[2]
        ├── CX @ q[1], q[2]
        ├── CX @ q[2], q[1]
        ├── CX @ q[1], q[2]
        ├── CX @ q[1], q[2]
        ├── CX @ q[2], q[1]
        ├── CX @ q[1], q[2]
        └── CX @ q[2], q[1]
        <BLANKLINE>

        # Test to see that all swaps are removed recursively

        >>> new_c, _ = remove_swaps(c, recursive=True)

        >>> any(isinstance(i.get_operation(), GateSWAP) for i in new_c)
        False

        >>> Outercheck = new_c.instructions[0].get_operation()._decl
        >>> any(isinstance(i.get_operation(), GateSWAP) for i in Outercheck.circuit)
        False

        >>> Midcheck = Outercheck.circuit.instructions[0].get_operation()._decl
        >>> any(isinstance(i.get_operation(), GateSWAP) for i in Midcheck.circuit)
        False

        >>> Innercheck = Midcheck.circuit.instructions[0].get_operation()._decl
        >>> any(isinstance(i.get_operation(), GateSWAP) for i in Innercheck.circuit)
        False

    """
    if cache is None:
        cache = {}

    perm = list(range(circuit.num_qubits()))
    new_circuit = Circuit()

    for instr in circuit:
        op = instr.get_operation()
        qubits = list(instr.get_qubits())

        if isinstance(op, GateSWAP):
            q1, q2 = qubits
            perm[q1], perm[q2] = perm[q2], perm[q1]

        elif recursive and _is_composite_operation(op):
            new_op, block_map = _remove_swaps_from_operation(op, recursive, cache)
            new_qubits = tuple(perm[q] for q in qubits)
            new_instr = Instruction(
                new_op, new_qubits, instr.get_bits(), instr.get_zvars()
            )
            new_circuit.append([new_instr])

            # Apply block's permutation to our permutation
            # perm[qubits] = perm[qubits[block_map]]
            old_perm_values = [perm[qubits[i]] for i in block_map]
            for i, q in enumerate(qubits):
                perm[q] = old_perm_values[i]

        else:
            new_qubits = tuple(perm[q] for q in qubits)
            new_instr = Instruction(op, new_qubits, instr.get_bits(), instr.get_zvars())
            new_circuit.append([new_instr])

    _pad_to_arity(new_circuit.instructions, circuit.num_qubits())

    return new_circuit, perm


def _pad_to_arity(instructions, target_nq):
    """Pad instruction list with GateID to preserve qubit arity."""
    nq = _get_num_qubits(instructions)
    for q in range(nq, target_nq):
        instructions.append(mc.Instruction(mc.GateID(), (q,), (), ()))


_PYOBJ_IDENTITY_CACHE_PREFIX = "__remove_swaps_identity__"

def _get_identity_cached(cache, op):
    entry = cache.get((_PYOBJ_IDENTITY_CACHE_PREFIX, id(op)))
    if entry is None:
        return None
    cached_op, cached_value = entry
    if cached_op is op:
        return cached_value
    return None


def _set_identity_cached(cache, op, value):
    cache[(_PYOBJ_IDENTITY_CACHE_PREFIX, id(op))] = (op, value)


def _is_composite_operation(op):
    """Check if operation is a composite that may contain swaps.

    Note: Control(GateCall) and IfStatement(Block) are intentionally excluded.
    Their internal SWAPs are conditional (only execute when control is |1> or
    classical condition is met), so they cannot be tracked as static permutations.
    """
    return (
        isinstance(op, GateCall)
        or isinstance(op, Block)
        or (isinstance(op, Inverse) and isinstance(op.op, GateCall))
    )


def _remove_swaps_from_operation(op, recursive, cache):
    if isinstance(op, GateCall):
        new_decl, qubit_map = _remove_swaps_from_gate_decl(op._decl, recursive, cache)
        return GateCall(new_decl, op._args), qubit_map

    elif isinstance(op, Block):
        cached = _get_identity_cached(cache, op)
        if cached is not None:
            return cached

        new_instrs, qubit_map = _remove_swaps_from_instructions(
            op.instructions, recursive, cache
        )
        # Preserve original block dimensions (don't let Block shrink)
        nq = op._num_qubits
        nb = op._num_bits
        nz = op._num_zvars
        # Pad qubit_map with identity for qubits not used in instructions
        if len(qubit_map) < nq:
            qubit_map.extend(range(len(qubit_map), nq))

        _pad_to_arity(new_instrs, nq)

        res = (Block(nq, nb, nz, new_instrs), qubit_map)
        _set_identity_cached(cache, op, res)
        return res

    elif isinstance(op, Inverse) and isinstance(op.op, GateCall):
        cached = _get_identity_cached(cache, op)
        if cached is not None:
            return cached

        gcall = op.op
        decl = gcall._decl

        # Expand the inverse body, remove swaps, then rebuild the forward body
        # to keep the Inverse(GateCall(...)) wrapper.
        inv_circuit = decl.circuit.inverse()
        inv_instrs = inv_circuit.instructions

        # Remove swaps from the expanded inverse
        new_inv_instrs, qubit_map = _remove_swaps_from_instructions(
            inv_instrs, recursive, cache
        )

        # Preserve original arity
        orig_nq = decl.num_qubits() if callable(decl.num_qubits) else decl.num_qubits
        _pad_to_arity(new_inv_instrs, orig_nq)

        # Rebuild the forward body
        new_fwd_circuit = Circuit(new_inv_instrs).inverse()
        new_fwd_instrs = new_fwd_circuit.instructions

        fwd_decl = GateDecl(decl.name, decl.arguments, mc.Circuit(new_fwd_instrs))
        res = (mc.Inverse(GateCall(fwd_decl, gcall._args)), qubit_map)
        _set_identity_cached(cache, op, res)
        return res

    else:
        raise ValueError(f"Unsupported composite operation type: {type(op)}")


def _remove_swaps_from_instructions(instructions, recursive, cache):
    perm = list(range(_get_num_qubits(instructions)))
    new_instrs = []

    for instr in instructions:
        op = instr.get_operation()
        qubits = list(instr.get_qubits())

        if isinstance(op, GateSWAP):
            q1, q2 = qubits
            perm[q1], perm[q2] = perm[q2], perm[q1]

        elif recursive and _is_composite_operation(op):
            new_op, block_map = _remove_swaps_from_operation(op, recursive, cache)
            new_qubits = tuple(perm[q] for q in qubits)
            new_instrs.append(
                Instruction(new_op, new_qubits, instr.get_bits(), instr.get_zvars())
            )

            old_perm_values = [perm[qubits[i]] for i in block_map]
            for i, q in enumerate(qubits):
                perm[q] = old_perm_values[i]

        else:
            new_qubits = tuple(perm[q] for q in qubits)
            new_instrs.append(
                Instruction(op, new_qubits, instr.get_bits(), instr.get_zvars())
            )

    return new_instrs, perm


def _remove_swaps_from_gate_decl(decl, recursive, cache):
    if decl in cache:
        return cache[decl]

    new_instrs, qubit_map = _remove_swaps_from_instructions(
        decl.circuit.instructions, recursive, cache
    )

    # Preserve original arity (do NOT let GateDecl shrink)
    orig_nq = decl.num_qubits() if callable(decl.num_qubits) else decl.num_qubits
    _pad_to_arity(new_instrs, orig_nq)

    res = (GateDecl(decl.name, decl.arguments, mc.Circuit(new_instrs)), qubit_map)
    cache[decl] = res
    return res


def _get_num_qubits(instructions):
    """Helper to determine number of qubits in instruction list."""
    if not instructions:
        return 0
    return max(max(instr.get_qubits(), default=-1) for instr in instructions) + 1
