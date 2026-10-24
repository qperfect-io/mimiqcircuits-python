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
from mimiqcircuits.circuit import _build_instruction
from typing import Callable, Union, Any


def _generated_insert_args(generated: Any) -> tuple:
    """Turn a generator's output into the arguments of :meth:`Circuit.insert`.

    A generator passed to the ``decorate_on_match_*`` functions returns either
    an `mc.Instruction`, turned into ``(instruction,)``, or a tuple
    ``(operation, targets)`` or ``(operation, qubits, bits, zvars)``, turned
    into ``(operation, *targets)`` with all targets concatenated. This keeps
    generators independent of the `mc.Instruction` constructor.

    Raises:
        TypeError: If `generated` is neither an `mc.Instruction` nor a
            non-empty tuple.
    """
    if isinstance(generated, mc.Instruction):
        return (generated,)

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

        return (op, *targets)

    raise TypeError(
        "Generator must return an mc.Instruction or a tuple like (operation, targets)."
    )


def _generated_instructions(generated: Any) -> list:
    """The instructions a generator's output stands for.

    Accepts the forms described in :func:`_generated_insert_args`, plus a
    ``(circuit,)`` tuple whose instructions are inserted as they are.

    Examples:
        >>> from mimiqcircuits import *
        >>> from mimiqcircuits.addnoise import _generated_instructions

        A tuple gives the operation followed by its qubits, bits and zvars:

        >>> _generated_instructions((Measure(), [0], [0]))
        [M @ q[0], c[0]]

        An instruction is used unchanged:

        >>> _generated_instructions(Instruction(GateH(), (1,)))
        [H @ q[1]]

    Raises:
        TypeError: If `generated` is neither an `mc.Instruction` nor a
            non-empty tuple.
    """
    op, *targets = _generated_insert_args(generated)
    if isinstance(op, mc.Circuit):
        # Circuit.insert splices a circuit in whole and ignores any targets.
        return list(op.instructions)
    return [_build_instruction(op, targets)]


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
    # Build the decorated list in one pass. Inserting into the circuit at
    # each match would shift the whole tail every time, which is quadratic.
    new_instructions = []
    for inst in circuit.instructions:
        if not matcher(inst):
            new_instructions.append(inst)
            continue
        generated = _generated_instructions(generator(inst))
        if before:
            new_instructions.extend(generated)
            new_instructions.append(inst)
        else:
            new_instructions.append(inst)
            new_instructions.extend(generated)
    return circuit._replace_instructions(new_instructions)


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
    # Blocks are found on the untouched circuit and the result is built in a
    # new list, so each instruction is copied once rather than shifted at
    # every insertion.
    instructions = circuit.instructions
    new_instructions = []
    i = 0
    while i < len(instructions):
        inst = instructions[i]
        if not matcher(inst):
            new_instructions.append(inst)
            i += 1
            continue

        orig_block, gen_block = _find_transversal_block(
            circuit, i, matcher, generator
        )
        generated = []
        for g in gen_block:
            generated.extend(_generated_instructions(g))

        if before:
            new_instructions.extend(generated)
            new_instructions.extend(orig_block)
        else:
            new_instructions.extend(orig_block)
            new_instructions.extend(generated)
        i += len(orig_block)
    return circuit._replace_instructions(new_instructions)


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
