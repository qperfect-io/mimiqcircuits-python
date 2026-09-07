Qubit Loss on MIMIQ
===================

.. currentmodule:: mimiqcircuits

MIMIQ supports explicit qubit loss for simulations where a physical qubit may
leave the computational space instead of only accumulating ordinary gate or
measurement noise. This is useful when testing quantum algorithms against more
realistic near-term hardware behavior, especially in workflows where missing
qubits should change the later circuit execution.

In the Python API, loss can be represented directly in a circuit. You can insert
stochastic loss events, reload a lost qubit, write its loss status into classical
bits, and rewrite partially affected instructions with a
:class:`~mimiqcircuits.lossmodel.LossModel`.

Loss can also be part of a custom Kraus channel. A
:class:`~mimiqcircuits.operations.noisechannel.kraus.Kraus` channel becomes
loss-aware when one or more branches are tagged with
:class:`~mimiqcircuits.operations.operators.lossyoperator.LossyOperator`, which
lets the channel separate survival branches from branches that lose a qubit.

All examples on this page assume the following imports:

.. doctest:: loss

    >>> from mimiqcircuits import *
    >>> import random
    >>> from symengine import Matrix, sqrt

Summary of Loss Functionality
-----------------------------

**Loss operations:** :class:`~mimiqcircuits.operations.losschannel.Loss`,
:class:`~mimiqcircuits.operations.losschannel.Reload`,
:class:`~mimiqcircuits.operations.losschannel.Check`,
:class:`~mimiqcircuits.operations.losschannel.MeasureCheck`.

**Loss processing:** :func:`~mimiqcircuits.lossmodel.sample_losses` draws the
random loss events, :func:`~mimiqcircuits.lossmodel.lower_losses` rewrites loss
into primitives, and :func:`~mimiqcircuits.lossmodel.resolve_losses` does both
and is the usual entry point. Each is also available as a method on
:class:`~mimiqcircuits.circuit.Circuit`. Use
:func:`~mimiqcircuits.lossmodel.sample_loss_scenario` to force a specific set of
loss sites.

**Loss-model rules:** :class:`~mimiqcircuits.lossmodel.DropRule`,
:class:`~mimiqcircuits.lossmodel.ReplaceRule`,
:class:`~mimiqcircuits.lossmodel.DecorateRule`,
:class:`~mimiqcircuits.lossmodel.CustomRule`.

**Loss markers:** :class:`~mimiqcircuits.operations.annotations.Lost` and
:class:`~mimiqcircuits.operations.annotations.Reloaded` are passive annotations
that record where loss landed in a resolved circuit.

**Loss-aware Kraus:**
:class:`~mimiqcircuits.operations.operators.lossyoperator.LossyOperator`
branches inside :class:`~mimiqcircuits.operations.noisechannel.kraus.Kraus`,
inspected with ``hasloss``, ``lossoperators``, ``survivaloperators``, and
``losseffect``.

Loss Operations
---------------

Loss in MIMIQ is represented explicitly in the circuit. You can add operations
that mark a qubit as lost, reload a lost qubit, or query whether a qubit is
still present.

Loss
~~~~

:class:`~mimiqcircuits.operations.losschannel.Loss` represents a loss event: at
that point in the circuit the qubit is lost with probability ``p``. ``Loss()``
is ``Loss(1.0)``, a certain loss.

.. doctest:: loss

    >>> circuit = Circuit()
    >>> circuit.push(Loss(0.1), 0)
    1-qubit circuit with 1 instruction:
    └── Loss(0.1) @ q[0]
    <BLANKLINE>

The probability may also be symbolic, but it must be numeric before the loss is
resolved.

Once a qubit is lost, later instructions touching that qubit are dropped while
resolving until the qubit is reloaded.

Reloading a Lost Qubit
~~~~~~~~~~~~~~~~~~~~~~

:class:`~mimiqcircuits.operations.losschannel.Reload` makes a qubit available
again. It always re-initialises the qubit to ``|0>``, whether or not it was
lost, so later operations can use it.

.. doctest:: loss

    >>> circuit = Circuit()
    >>> circuit.push(Loss(), 0)
    1-qubit circuit with 1 instruction:
    └── Loss(1.0) @ q[0]
    <BLANKLINE>
    >>> circuit.push(Reload(), 0)
    1-qubit circuit with 2 instructions:
    ├── Loss(1.0) @ q[0]
    └── Reload @ q[0]
    <BLANKLINE>
    >>> circuit.push(GateX(), 0)
    1-qubit circuit with 3 instructions:
    ├── Loss(1.0) @ q[0]
    ├── Reload @ q[0]
    └── X @ q[0]
    <BLANKLINE>

Checking for Loss
~~~~~~~~~~~~~~~~~

MIMIQ provides two operations to query the loss status of a qubit.

:class:`~mimiqcircuits.operations.losschannel.Check` writes one classical bit:

* ``1`` if the qubit is present
* ``0`` if the qubit is lost

It does not touch the quantum state.

.. doctest:: loss

    >>> circuit = Circuit()
    >>> circuit.push(Check(), 0, 0)
    1-qubit, 1-bit circuit with 1 instruction:
    └── Check @ q[0], c[0]
    <BLANKLINE>

:class:`~mimiqcircuits.operations.losschannel.MeasureCheck` measures the qubit
if it is present and reports whether it is still present.

.. doctest:: loss

    >>> circuit = Circuit()
    >>> circuit.push(MeasureCheck(), 0, 0, 1)
    1-qubit, 2-bit circuit with 1 instruction:
    └── MeasureCheck @ q[0], c[0:1]
    <BLANKLINE>

The first classical bit stores the measurement result (``0`` if the qubit is
lost), and the second classical bit stores the loss status.

Resolving Loss
--------------

A circuit that contains loss operations is not yet runnable: the random events
must be drawn and every loss operation rewritten into ordinary primitives.
:func:`~mimiqcircuits.lossmodel.resolve_losses`, also available as
:meth:`~mimiqcircuits.circuit.Circuit.resolve_losses`, does this in one step. It
first draws each :class:`~mimiqcircuits.operations.losschannel.Loss`, then lowers
the result into ``Reset`` / ``Measure`` / ``SetBit0`` / ``SetBit1`` plus passive
:class:`~mimiqcircuits.operations.annotations.Lost` and
:class:`~mimiqcircuits.operations.annotations.Reloaded` markers. The returned
circuit contains no loss operations.

The ``rng`` argument is a random number generator, used only to make the random
loss samples reproducible. Omit it if you do not need the same result every time.

.. doctest:: loss

    >>> rng = random.Random(42)
    >>> circuit = Circuit()
    >>> circuit.push(Loss(0.2), 0)
    1-qubit circuit with 1 instruction:
    └── Loss(0.2) @ q[0]
    <BLANKLINE>
    >>> circuit.push(GateH(), 0)
    1-qubit circuit with 2 instructions:
    ├── Loss(0.2) @ q[0]
    └── H @ q[0]
    <BLANKLINE>
    >>> circuit.push(Check(), 0, 0)
    1-qubit, 1-bit circuit with 3 instructions:
    ├── Loss(0.2) @ q[0]
    ├── H @ q[0]
    └── Check @ q[0], c[0]
    <BLANKLINE>
    >>> circuit.resolve_losses(rng=rng)
    1-qubit, 1-bit circuit with 2 instructions:
    ├── H @ q[0]
    └── c[0] = 1
    <BLANKLINE>

The basic behavior while resolving is:

* :class:`~mimiqcircuits.operations.losschannel.Loss` is drawn; a fired loss
  emits a :class:`~mimiqcircuits.operations.annotations.Lost` marker
* :class:`~mimiqcircuits.operations.losschannel.Reload` becomes a ``Reset`` plus
  a :class:`~mimiqcircuits.operations.annotations.Reloaded` marker
* :class:`~mimiqcircuits.operations.losschannel.Check` and
  :class:`~mimiqcircuits.operations.losschannel.MeasureCheck` become
  ``SetBit0`` / ``SetBit1`` (and a ``Measure`` for a present ``MeasureCheck``)
* Instructions acting only on lost qubits are dropped

If an instruction touches some lost qubits but not all of them, resolving
consults a :class:`~mimiqcircuits.lossmodel.LossModel`.

If you need the two halves separately,
:func:`~mimiqcircuits.lossmodel.sample_losses` returns a circuit with the random
events drawn but the loss bookkeeping still present, and
:func:`~mimiqcircuits.lossmodel.lower_losses` lowers an already-sampled circuit
into primitives.

Forcing a Loss Scenario
~~~~~~~~~~~~~~~~~~~~~~~

To study a specific outcome,
:func:`~mimiqcircuits.lossmodel.sample_loss_scenario` forces the chosen
:class:`~mimiqcircuits.operations.losschannel.Loss` sites (counted 1-based in
circuit order) to fire and resolves the rest as not lost.

.. doctest:: loss

    >>> circuit = Circuit()
    >>> _ = circuit.push(Loss(0.2), 0)
    >>> _ = circuit.push(GateCX(), 0, 1)
    >>> _ = circuit.push(Loss(0.4), 1)
    >>> circuit.sample_loss_scenario(2)
    2-qubit circuit with 2 instructions:
    ├── CX @ q[0], q[1]
    └── Lost @ q[1]
    <BLANKLINE>

Loss Models
-----------

Why Loss Models Exist
~~~~~~~~~~~~~~~~~~~~~

A :class:`~mimiqcircuits.lossmodel.LossModel` is the user-defined policy used
while resolving loss when an instruction is only partially affected. This
happens, for example, when a two-qubit gate is scheduled but one of its qubits
has already been lost while the other is still present.

MIMIQ can detect this situation, but it should not guess the physics for the
remaining qubits. Different hardware models and approximations can lead to
different choices: drop the instruction entirely, apply a one-qubit error
channel to each surviving qubit, keep a side-effect before or after the
attempted operation, or generate custom replacement instructions. A
``LossModel`` is where you specify that choice explicitly.

If no rule is provided, MIMIQ uses the conservative behavior and drops
instructions that touch lost qubits. Add rules when your hardware model or
simulation workflow has a more specific response to partial loss.

When Rules Are Used
~~~~~~~~~~~~~~~~~~~

While resolving, MIMIQ tracks which qubits are currently lost and rewrites the
circuit as follows:

* If an instruction touches no lost qubits, it is kept unchanged.
* If an instruction touches only lost qubits, it is dropped.
* If an instruction touches both lost and surviving qubits, the
  :class:`~mimiqcircuits.lossmodel.LossModel` is consulted.
* If no rule in the model matches, the instruction is dropped.

Rules are evaluated by priority and then by insertion order. A
:class:`~mimiqcircuits.lossmodel.DropRule` has higher priority than replacement or
decoration rules, so it can be used to exclude specific operations before a
broader salvage rule is applied. Once a rule matches, MIMIQ builds the rule's
output and filters out any generated instruction that still touches a lost
qubit.

This last filtering step is important. A one-qubit replacement such as
``Depolarizing1(0.2)`` is broadcast to the targets of the matched gate, and the
copies on lost qubits are removed. A multi-qubit replacement that still touches
a lost qubit is removed entirely.

You can create an empty model and add rules incrementally:

.. doctest:: loss

    >>> model = LossModel(name="My Loss Model")
    >>> model
    LossModel (My Loss Model, 0 rules)

The main helpers are:

* :meth:`~mimiqcircuits.lossmodel.LossModel.add_drop`
* :meth:`~mimiqcircuits.lossmodel.LossModel.add_replace`
* :meth:`~mimiqcircuits.lossmodel.LossModel.add_decorate`

Replacing a Partially Lost Gate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :class:`~mimiqcircuits.lossmodel.ReplaceRule` when the original instruction
should be removed and replaced by another operation on the surviving qubits. In
this example, a ``CX`` whose target qubit has been lost is replaced by a
one-qubit depolarizing channel on the remaining control qubit.

.. doctest:: loss

    >>> circuit = Circuit()
    >>> _ = circuit.push(Loss(), 1)
    >>> _ = circuit.push(GateCX(), 0, 1)
    >>> model = LossModel().add_replace(GateCX(), Depolarizing1(0.2))
    >>> circuit.resolve_losses(lossmodel=model)
    2-qubit circuit with 2 instructions:
    ├── Lost @ q[1]
    └── Depolarizing(0.2) @ q[0]
    <BLANKLINE>

If the lost qubit is the control instead, the same rule keeps the replacement
on the surviving target qubit.

.. doctest:: loss

    >>> circuit = Circuit()
    >>> _ = circuit.push(Loss(), 0)
    >>> _ = circuit.push(GateCX(), 0, 1)
    >>> circuit.resolve_losses(lossmodel=model)
    2-qubit circuit with 2 instructions:
    ├── Lost @ q[0]
    └── Depolarizing(0.2) @ q[1]
    <BLANKLINE>

Drop Rules
~~~~~~~~~~

:class:`~mimiqcircuits.lossmodel.DropRule` removes matching instructions when
they touch lost qubits. Use this when a partially affected operation should not
be salvaged. A ``DropRule`` without an operation is a catch-all rule.

.. doctest:: loss

    >>> model = LossModel().add_rule(DropRule(GateSWAP()))
    >>> model
    LossModel (unnamed, 1 rules)
    └── DropRule(SWAP)

You can also use the convenience form:

.. doctest:: loss

    >>> model = LossModel().add_drop(GateSWAP())
    >>> model
    LossModel (unnamed, 1 rules)
    └── DropRule(SWAP)

Because drop rules have higher priority, they can override broader replacement
rules:

.. doctest:: loss

    >>> model = LossModel([
    ...     ReplaceRule(GateSWAP(), GateX()),
    ...     DropRule(GateSWAP()),
    ... ])
    >>> model
    LossModel (unnamed, 2 rules)
    ├── DropRule(SWAP)
    └── ReplaceRule(SWAP => X)

Decorating a Partially Lost Gate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~mimiqcircuits.lossmodel.DecorateRule` adds another operation before or
after the matched instruction. While resolving, generated instructions touching
lost qubits are filtered out, so if the original gate still touches a lost
qubit it is removed and only surviving decorations remain.

.. doctest:: loss

    >>> model = LossModel().add_decorate(GateCZ(), Depolarizing1(0.01), before=True)
    >>> model
    LossModel (unnamed, 1 rules)
    └── DecorateRule(CZ, Depolarizing(0.01), before)

Use decoration when your model says that the attempted operation still causes a
side effect, such as a local error channel on the qubits that were present.

Custom Rules
~~~~~~~~~~~~

Use :class:`~mimiqcircuits.lossmodel.CustomRule` when the rewrite depends on
more than the operation type. The generator receives the matched instruction and
the current loss map. It may return ``None`` to drop the instruction, one
:class:`~mimiqcircuits.instruction.Instruction`, or a sequence of instructions.

If your custom rule needs randomness, define the generator with an ``rng``
argument or keyword. Otherwise, a two-argument generator ``(inst, lost)`` is
enough.

.. doctest:: loss

    >>> circuit = Circuit()
    >>> _ = circuit.push(Loss(), 1)
    >>> _ = circuit.push(GateCX(), 0, 1)
    >>> model = LossModel([
    ...     CustomRule(
    ...         lambda inst: isinstance(inst.get_operation(), GateCX),
    ...         lambda inst, lost: [
    ...             Instruction(GateZ(), (q,))
    ...             for q in inst.get_qubits()
    ...             if not lost.get(q, False)
    ...         ],
    ...     )
    ... ])
    >>> model
    LossModel (unnamed, 1 rules)
    └── CustomRule(<callable>)
    >>> circuit.resolve_losses(lossmodel=model)
    2-qubit circuit with 2 instructions:
    ├── Lost @ q[1]
    └── Z @ q[0]
    <BLANKLINE>

For most workflows, prefer ``DropRule``, ``ReplaceRule``, or ``DecorateRule``
because those rules are simpler to inspect and serialize. ``CustomRule`` is the
escape hatch for policies that cannot be expressed with the built-in rule
types.

.. _loss-aware-kraus-channels:

Loss-Aware Kraus Channels
-------------------------

Custom :class:`~mimiqcircuits.operations.noisechannel.kraus.Kraus` channels can
also model loss. A Kraus channel becomes loss-aware when one or more of its
branches are tagged with
:class:`~mimiqcircuits.operations.operators.lossyoperator.LossyOperator`.

Use this when the loss event is part of the physical noise channel itself,
rather than a separate
:class:`~mimiqcircuits.operations.losschannel.Loss` instruction. The
``LossyOperator`` matrix contains amplitudes, not probabilities, and the
remaining Kraus branches describe the no-loss evolution.

.. doctest:: loss

    >>> k = Kraus([
    ...     Matrix([[1, 0], [0, sqrt(0.9)]]),
    ...     LossyOperator(Matrix([[0, sqrt(0.1)], [0, 0]])),
    ... ])
    >>> k
    Kraus(Operator([[1, 0], [0, 0.948683298050514]]), LossyOperator([[0, 0.316227766016838], [0, 0]]; lossy=(1,)))

The helper methods separate the loss branches from the survival branches and
compute the total loss effect carried by the channel.

.. doctest:: loss

    >>> k.hasloss()
    True
    >>> k.lossoperators()
    [1-qubit LossyOperator (lossy=(1,)):
    ├── 0 0.316227766016838
    └── 0 0]
    >>> k.survivaloperators()
    [1-qubit Operator:
    ├── 1 0
    └── 0 0.948683298050514]
    >>> k.losseffect()
    1-qubit Operator:
    ├── 0.0 + 0.0*I 0.0 + 0.0*I
    └── 0.0 + 0.0*I 0.1 + 0.0*I

If you only need the general Kraus formalism, see
:ref:`kraus-operators`.
