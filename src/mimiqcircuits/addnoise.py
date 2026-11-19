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

from mimiqcircuits.circuit import Circuit
from mimiqcircuits.instruction import Instruction
from mimiqcircuits.operations.gates.gate import Gate
from mimiqcircuits.operations.krauschannel import krauschannel
from mimiqcircuits.operations.operation import Operation
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
    if isinstance(generated, Instruction):
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
    if isinstance(op, Circuit):
        # Circuit.insert splices a circuit in whole and ignores any targets.
        return list(op.instructions)
    return [_build_instruction(op, targets)]


def decorate_on_match_single_inplace(
    circuit,
    matcher: Callable[[Instruction], bool],
    generator: Callable[[Instruction], Any],
    before: bool = False,
):
    """Add generated instructions next to every matching instruction, in place.

    For each instruction ``inst`` of ``circuit`` with ``matcher(inst)`` true,
    the output of ``generator(inst)`` is inserted right after ``inst`` (or
    right before it). The generator returns one of:

    - an :class:`Instruction`;
    - a tuple ``(operation, targets)``, such as ``(noise, inst.qubits)``, or
      ``(operation, qubits, bits, zvars)``;
    - a tuple ``(circuit,)``, whose instructions are inserted as they are.

    This is the engine behind :meth:`Circuit.decorate_on_match_single`.

    Args:
        circuit (Circuit): Circuit to change.
        matcher: Function ``inst -> bool`` selecting the instructions.
        generator: Function ``inst -> instruction or tuple`` giving what to
            insert.
        before: Insert before each match instead of after it.

    Returns:
        Circuit: ``circuit`` itself, changed.

    Raises:
        TypeError: If the generator returns an unsupported value.

    Examples:
        >>> from mimiqcircuits import *
        >>> from mimiqcircuits.addnoise import decorate_on_match_single_inplace
        >>> c = Circuit()
        >>> _ = c.push(GateH(), 0)
        >>> _ = c.push(GateCX(), 0, 1)
        >>> # add bit-flip noise after every one-qubit gate
        >>> decorate_on_match_single_inplace(
        ...     c,
        ...     lambda inst: inst.operation.num_qubits == 1,
        ...     lambda inst: (PauliX(0.01), inst.qubits),
        ... )
        2-qubit circuit with 3 instructions:
        ├── H @ q[0]
        ├── PauliX(0.01) @ q[0]
        └── CX @ q[0], q[1]
        <BLANKLINE>
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
    matcher: Callable[[Instruction], bool],
    generator: Callable[[Instruction], Any],
    before: bool = False,
):
    """Return a decorated copy of ``circuit``.

    Same as :func:`decorate_on_match_single_inplace`, applied to a deep copy;
    ``circuit`` is not changed.

    Returns:
        Circuit: The decorated copy.
    """
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
    matcher: Callable[[Instruction], bool],
    generator: Callable[[Instruction], Any],
    before: bool = False,
):
    """Add generated instructions next to blocks of parallel matches, in place.

    A block is a run of consecutive matching instructions that act on disjoint
    qubits, bits and z-variables, such as the instructions produced by one
    broadcast :meth:`Circuit.push`. The generated instructions of a whole
    block are inserted together, after the block (or before it). The
    ``matcher`` and ``generator`` follow the same rules as in
    :func:`decorate_on_match_single_inplace`.

    This is the engine behind :meth:`Circuit.decorate_on_match_parallel`.

    Args:
        circuit (Circuit): Circuit to change.
        matcher: Function ``inst -> bool`` selecting the instructions.
        generator: Function ``inst -> instruction or tuple`` giving what to
            insert.
        before: Insert before each block instead of after it.

    Returns:
        Circuit: ``circuit`` itself, changed.

    Raises:
        TypeError: If the generator returns an unsupported value.

    Examples:
        >>> from mimiqcircuits import *
        >>> from mimiqcircuits.addnoise import decorate_on_match_parallel_inplace
        >>> c = Circuit()
        >>> _ = c.push(GateH(), range(3))
        >>> # the three H gates form one block: the noise comes after all of them
        >>> decorate_on_match_parallel_inplace(
        ...     c,
        ...     lambda inst: inst.operation == GateH(),
        ...     lambda inst: (AmplitudeDamping(0.1), inst.qubits),
        ... )
        3-qubit circuit with 6 instructions:
        ├── H @ q[0]
        ├── H @ q[1]
        ├── H @ q[2]
        ├── AmplitudeDamping(0.1) @ q[0]
        ├── AmplitudeDamping(0.1) @ q[1]
        └── AmplitudeDamping(0.1) @ q[2]
        <BLANKLINE>
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
    matcher: Callable[[Instruction], bool],
    generator: Callable[[Instruction], Any],
    *,
    before: bool = False,
):
    """Return a copy of ``circuit`` decorated block by block.

    Same as :func:`decorate_on_match_parallel_inplace`, applied to a deep
    copy; ``circuit`` is not changed. ``before`` is keyword-only.

    Returns:
        Circuit: The decorated copy.
    """
    newc = circuit.deepcopy()
    return decorate_on_match_parallel_inplace(newc, matcher, generator, before=before)


def decorate_inplace(
    circuit,
    matcher: Callable[[Instruction], bool],
    generator: Callable[[Instruction], Any],
    before: bool = False,
    parallel: bool = False,
):
    """Decorate matching instructions in place, one by one or by blocks.

    Calls :func:`decorate_on_match_parallel_inplace` when ``parallel`` is
    ``True`` and :func:`decorate_on_match_single_inplace` otherwise.

    Returns:
        Circuit: ``circuit`` itself, changed.
    """
    if parallel:
        return decorate_on_match_parallel_inplace(
            circuit, matcher, generator, before=before
        )
    return decorate_on_match_single_inplace(circuit, matcher, generator, before=before)


def decorate(
    circuit,
    matcher: Callable[[Instruction], bool],
    generator: Callable[[Instruction], Any],
    *,
    before: bool = False,
    parallel: bool = False,
):
    """Return a decorated copy, decorating one by one or by blocks.

    Calls :func:`decorate_on_match_parallel` when ``parallel`` is ``True``
    and :func:`decorate_on_match_single` otherwise. ``circuit`` is not
    changed.

    Returns:
        Circuit: The decorated copy.
    """
    if parallel:
        return decorate_on_match_parallel(circuit, matcher, generator, before=before)
    return decorate_on_match_single(circuit, matcher, generator, before=before)


# inplace noise addition helpers
def add_noise_inplace(
    circuit,
    g: Union[Operation, type],
    noise: Union[krauschannel, Gate],
    before: bool = False,
    parallel: bool = False,
):
    """Add ``noise`` next to every instance of ``g``, in place.

    ``noise`` acts on the same qubits as each matched instruction, so it must
    act on as many qubits as ``g``. This is the engine behind
    :meth:`Circuit.add_noise`.

    Args:
        circuit (Circuit): Circuit to change.
        g: Operation to match. An instance matches equal operations; an
            :class:`Operation` subclass, such as ``GateH``, matches all its
            instances.
        noise: Gate or Kraus channel to add.
        before: Add the noise before each match instead of after it.
        parallel: Add the noise after (or before) each block of parallel
            matches, as in :func:`decorate_on_match_parallel_inplace`.

    Returns:
        Circuit: ``circuit`` itself, changed.

    Raises:
        TypeError: If ``noise`` is not a gate or Kraus channel, or ``g`` is
            neither an operation nor an operation class.
        ValueError: If ``noise`` acts on a different number of qubits than a
            matched instruction.

    Examples:
        >>> from mimiqcircuits import *
        >>> from mimiqcircuits.addnoise import add_noise_inplace
        >>> c = Circuit()
        >>> _ = c.push(GateH(), 0)
        >>> _ = c.push(GateX(), 1)
        >>> # match every Pauli X gate by its class
        >>> add_noise_inplace(c, GateX, PauliZ(0.05))
        2-qubit circuit with 3 instructions:
        ├── H @ q[0]
        ├── X @ q[1]
        └── PauliZ(0.05) @ q[1]
        <BLANKLINE>
    """
    if not isinstance(noise, (krauschannel, Gate)):
        raise TypeError(f"{noise} must be a Gate or Kraus channel")

    if isinstance(g, Operation):
        matcher = lambda inst: inst.operation == g
    elif isinstance(g, type) and issubclass(g, Operation):
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
    g: Union[Operation, type],
    noise: Union[krauschannel, Gate],
    before: bool = False,
    parallel: bool = False,
):
    """Return a copy of ``circuit`` with ``noise`` added next to every ``g``.

    Same as :func:`add_noise_inplace`, applied to a deep copy; ``circuit`` is
    not changed.

    Returns:
        Circuit: The noisy copy.
    """
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
