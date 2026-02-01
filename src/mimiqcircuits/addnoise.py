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

import mimiqcircuits as mc
from typing import Callable, Union, Tuple, Sequence, Any


def _insert_generated(circuit, index: int, generated: Any):
    """
    Internal helper for inserting generated operations into a circuit.

    This function standardizes how new instructions (produced by generator functions)
    are inserted into the circuit. It accepts either:

    1. An `mc.Instruction` object — inserted directly.
    2. A tuple specifying what to insert:
       - `(operation, targets)` => calls `circuit.insert(index, operation, *targets)`
       - `(operation, qubits, bits?, zvars?)` => concatenates all targets and inserts.

    Examples:
        >>> from mimiqcircuits import *
        >>> from mimiqcircuits.addnoise import _insert_generated

        >>> c = Circuit()
        >>> c.push(GateX(), 0)
        1-qubit circuit with 1 instruction:
        └── X @ q[0]
        <BLANKLINE>
        >>> c.push(GateZ(), 1)
        2-qubit circuit with 2 instructions:
        ├── X @ q[0]
        └── Z @ q[1]
        <BLANKLINE>

        # Insert a Hadamard *before* the first gate (index = 0)
        >>> _insert_generated(c, 0, mc.Instruction(GateH(), (0,)))

        # Insert another Hadamard *after* the first gate (index = 2)
        >>> _insert_generated(c, 2, (GateH(), [1]))

        # Insert with multiple register types: qubits, bits, and zvars
        >>> _insert_generated(c, 1, (Measure(), [0], [0]))

        >>> c
        2-qubit, 1-bit circuit with 5 instructions:
        ├── H @ q[0]
        ├── M @ q[0], c[0]
        ├── X @ q[0]
        ├── H @ q[1]
        └── Z @ q[1]
        <BLANKLINE>

    This design keeps generator functions lightweight and independent
    of the `mc.Instruction` constructor interface.

    Args:
        circuit: The circuit in which to insert.
        index (int): Position at which to insert the new instruction (0-based).
        generated (Any): Either an `mc.Instruction` or a tuple describing the operation.

    Raises:
        TypeError: If `generated` is neither an `mc.Instruction` nor a valid tuple form.
    """
    if isinstance(generated, mc.Instruction):
        circuit.insert(index, generated)
        return

    # allow (operation, targets)
    if isinstance(generated, tuple) and generated:
        op = generated[0]
        # flatten remaining into a single sequence of targets
        if len(generated) == 2 and isinstance(generated[1], (list, tuple)):
            targets = tuple(generated[1])
        else:
            rest = []
            for part in generated[1:]:
                if isinstance(part, (list, tuple)):
                    rest.extend(part)
                else:
                    rest.append(part)
            targets = tuple(rest)

        circuit.insert(index, op, *targets)
        return

    raise TypeError(
        "Generator must return an mc.Instruction or a tuple like (operation, targets)."
    )


def decorate_on_match_single_inplace(
    circuit,
    matcher: Callable[[mc.Instruction], bool],
    generator: Callable[[mc.Instruction], Any],
    before: bool = False,
):
    """
    In-place: add a generated instruction before/after each matched instruction.

    matcher(inst) -> bool
    generator(inst) -> either:
        - mc.Instruction
        - (operation, targets)  e.g. (noise_op, inst.qubits)
    """
    rel = 0 if before else 1
    i = 0
    while i < len(circuit.instructions):
        inst = circuit.instructions[i]
        if matcher(inst):
            gen = generator(inst)
            _insert_generated(circuit, i + rel, gen)
            # skip original + inserted
            i += 2
        else:
            i += 1
    return circuit


def decorate_on_match_single(
    circuit,
    matcher: Callable[[mc.Instruction], bool],
    generator: Callable[[mc.Instruction], Any],
    before: bool = False,
):
    """Non-mutating wrapper for decorate_on_match_single_inplace."""
    newc = circuit.deepcopy()
    return decorate_on_match_single_inplace(newc, matcher, generator, before=before)


def _find_transversal_block(
    circuit,
    i: int,
    matcher,
    generator,
):
    """
    Internal helper to find a transversal block (disjoint qubits, bits, and zvars).

    Starting from instruction index `i`, this scans forward to collect consecutive
    instructions that both satisfy the `matcher` predicate and act on disjoint
    qubits, bits, and zvars from the existing block. It also ensures that the
    generated decorations do not overlap on the same registers.

    Returns a tuple of two lists: `(original_block, generated_block)`.

    Examples:
        >>> from mimiqcircuits import *
        >>> from mimiqcircuits.addnoise import _find_transversal_block
        >>> c = Circuit()
        >>> for q in range(3):
        ...     _ =c.push(GateH(), q)
        >>> matcher = lambda inst: inst.operation == GateH()
        >>> generator = lambda inst: Instruction(GateS(), inst.qubits)
        >>> ob, gb = _find_transversal_block(c, 0, matcher, generator)
        >>> [str(x) for x in ob]
        ['H @ q[0]', 'H @ q[1]', 'H @ q[2]']
        >>> [str(x) for x in gb]
        ['S @ q[0]', 'S @ q[1]', 'S @ q[2]']
    """

    inst = circuit[i]
    original_block = [inst]
    generated = generator(inst)
    generated_block = [generated]

    # Track used registers in both original and generated blocks
    qubits_in_block = set(getattr(inst, "qubits", []))
    bits_in_block = set(getattr(inst, "bits", []))
    zvars_in_block = set(getattr(inst, "zvars", []))

    qubits_in_generated = set(getattr(generated, "qubits", []))
    bits_in_generated = set(getattr(generated, "bits", []))
    zvars_in_generated = set(getattr(generated, "zvars", []))

    j = i + 1
    while j < len(circuit):
        nxt = circuit[j]

        next_qubits = set(getattr(nxt, "qubits", []))
        next_bits = set(getattr(nxt, "bits", []))
        next_zvars = set(getattr(nxt, "zvars", []))

        if (
            matcher(nxt)
            and qubits_in_block.isdisjoint(next_qubits)
            and qubits_in_generated.isdisjoint(next_qubits)
            and bits_in_block.isdisjoint(next_bits)
            and bits_in_generated.isdisjoint(next_bits)
            and zvars_in_block.isdisjoint(next_zvars)
            and zvars_in_generated.isdisjoint(next_zvars)
        ):
            generated = generator(nxt)
            original_block.append(nxt)
            generated_block.append(generated)

            # Update resource sets for both original and generated sides
            qubits_in_block |= next_qubits
            bits_in_block |= next_bits
            zvars_in_block |= next_zvars

            qubits_in_generated |= set(getattr(generated, "qubits", []))
            bits_in_generated |= set(getattr(generated, "bits", []))
            zvars_in_generated |= set(getattr(generated, "zvars", []))

            j += 1
        else:
            break

    return original_block, generated_block


def decorate_on_match_parallel_inplace(
    circuit,
    matcher: Callable[[mc.Instruction], bool],
    generator: Callable[[mc.Instruction], Any],
    before: bool = False,
):
    """
    In-place: add blocks of generated instructions next to transversal blocks
    of matched instructions.
    """
    i = 0
    while i < len(circuit.instructions):
        inst = circuit.instructions[i]
        if matcher(inst):
            orig_block, gen_block = _find_transversal_block(
                circuit, i, matcher, generator
            )
            n = len(orig_block)

            if before:
                for k in range(n):
                    _insert_generated(circuit, i + k, gen_block[k])
            else:
                for k in range(n):
                    _insert_generated(circuit, i + n + k, gen_block[k])

            i += 2 * n
        else:
            i += 1
    return circuit


def decorate_on_match_parallel(
    circuit,
    matcher: Callable[[mc.Instruction], bool],
    generator: Callable[[mc.Instruction], Any],
    *,
    before: bool = False,
):
    """Non-mutating wrapper for decorate_on_match_parallel_inplace."""
    newc = circuit.deepcopy()
    return decorate_on_match_parallel_inplace(newc, matcher, generator, before=before)


def decorate_inplace(
    circuit,
    matcher: Callable[[mc.Instruction], bool],
    generator: Callable[[mc.Instruction], Any],
    before: bool = False,
    parallel: bool = False,
):
    """In-place unified decoration (single vs parallel)."""
    if parallel:
        return decorate_on_match_parallel_inplace(
            circuit, matcher, generator, before=before
        )
    return decorate_on_match_single_inplace(circuit, matcher, generator, before=before)


def decorate(
    circuit,
    matcher: Callable[[mc.Instruction], bool],
    generator: Callable[[mc.Instruction], Any],
    *,
    before: bool = False,
    parallel: bool = False,
):
    """Non-mutating unified decoration (single vs parallel)."""
    if parallel:
        return decorate_on_match_parallel(circuit, matcher, generator, before=before)
    return decorate_on_match_single(circuit, matcher, generator, before=before)


# inplace noise addition helpers
def add_noise_inplace(
    circuit,
    g: Union[mc.Operation, type],
    noise: Union[mc.krauschannel, mc.Gate],
    before: bool = False,
    parallel: bool = False,
):
    """
    In-place: add `noise` next to every instance (or type) of operation `g`.

    - `g` can be an Operation instance or an Operation subclass (type).
    - `noise` must be a Gate or Kraus channel.
    """
    if not isinstance(noise, (mc.krauschannel, mc.Gate)):
        raise TypeError(f"{noise} must be a Gate or Kraus channel")

    if isinstance(g, mc.Operation):
        matcher = lambda inst: inst.operation == g
    elif isinstance(g, type) and issubclass(g, mc.Operation):
        matcher = lambda inst: isinstance(inst.operation, g)
    else:
        raise TypeError(f"{g} must be an Operation instance or Operation subclass")

    # Generator returns a (operation, targets) tuple -> robust insertion
    generator = lambda inst: (noise, inst.qubits)

    return decorate_inplace(
        circuit, matcher, generator, before=before, parallel=parallel
    )


# Non-mutating wrapper for add_noise_inplace.
def add_noise(
    circuit,
    g: Union[mc.Operation, type],
    noise: Union[mc.krauschannel, mc.Gate],
    before: bool = False,
    parallel: bool = False,
):
    """Non-mutating wrapper for add_noise_inplace."""
    newc = circuit.deepcopy()
    return add_noise_inplace(newc, g, noise, before=before, parallel=parallel)


__all__ = [
    # (single)
    "decorate_on_match_single_inplace",
    "decorate_on_match_single",
    # (parallel)
    "decorate_on_match_parallel_inplace",
    "decorate_on_match_parallel",
    # unified
    "decorate_inplace",
    "decorate",
    # noise helpers
    "add_noise_inplace",
    "add_noise",
]
