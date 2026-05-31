Qubit Loss on MIMIQ
===================

.. currentmodule:: mimiqcircuits

MIMIQ supports explicit qubit loss for simulations where a physical qubit may
leave the computational space instead of only accumulating ordinary gate or
measurement noise. This is useful when testing quantum algorithms against more
realistic near-term hardware behavior, especially in workflows where missing
qubits should change the later circuit execution.

In the Python API, loss can be represented directly in a circuit. You can insert
stochastic loss events, mark a qubit as lost deterministically, reload it later,
write its loss status into classical bits, and rewrite partially affected
instructions with a :class:`~mimiqcircuits.lossmodel.LossModel`.

Loss can also be part of a custom Kraus channel. A
:class:`~mimiqcircuits.operations.noisechannel.kraus.Kraus` channel becomes
loss-aware when one or more branches are tagged with
:class:`~mimiqcircuits.operations.operators.lossyoperator.LossyOperator`, which
lets the channel separate survival branches from branches that lose a qubit.

.. doctest:: loss
    :hide:

    >>> from mimiqcircuits import *
    >>> import random
    >>> from symengine import Matrix, sqrt

Summary of Loss Functionality
-----------------------------

**Loss operations:** :class:`~mimiqcircuits.operations.losschannel.LossErr`,
:class:`~mimiqcircuits.operations.losschannel.QubitLoss`,
:class:`~mimiqcircuits.operations.losschannel.QubitReload`,
:class:`~mimiqcircuits.operations.losschannel.CheckLoss`,
:class:`~mimiqcircuits.operations.losschannel.MeasureCheckLoss`.

**Loss processing:** :func:`~mimiqcircuits.lossmodel.sample_losses` with
:class:`~mimiqcircuits.lossmodel.LossModel`. The same sampling functionality is
also available as :meth:`~mimiqcircuits.circuit.Circuit.sample_losses`.

**Loss-model rules:** :class:`~mimiqcircuits.lossmodel.DropRule`,
:class:`~mimiqcircuits.lossmodel.ReplaceRule`,
:class:`~mimiqcircuits.lossmodel.DecorateRule`,
:class:`~mimiqcircuits.lossmodel.CustomRule`.

**Loss-aware Kraus:**
:class:`~mimiqcircuits.operations.operators.lossyoperator.LossyOperator`
branches inside :class:`~mimiqcircuits.operations.noisechannel.kraus.Kraus`,
inspected with ``hasloss``, ``lossoperators``, ``survivaloperators``, and
``losseffect``.

Loss Operations
---------------

Loss in MIMIQ is represented explicitly in the circuit. You can add operations that mark a qubit as lost, sample stochastic loss events, reload a lost qubit,
or query whether a qubit is still present.

Loss Error
~~~~~~~~~~

:class:`~mimiqcircuits.operations.losschannel.LossErr` represents a
probabilistic loss event. At that point in the circuit, the qubit is lost with
probability ``p``.

.. doctest:: loss

    >>> circuit = Circuit()
    >>> circuit.push(LossErr(0.1), 0)
    1-qubit circuit with 1 instruction:
    └── LossErr(0.1) @ q[0]
    <BLANKLINE>

The probability may also be symbolic, but it must be numeric before calling
:func:`~mimiqcircuits.lossmodel.sample_losses`.

Deterministic Loss
~~~~~~~~~~~~~~~~~~

:class:`~mimiqcircuits.operations.losschannel.QubitLoss` marks a qubit as lost
unconditionally.

.. doctest:: loss

    >>> circuit = Circuit()
    >>> circuit.push(QubitLoss(), 1)
    2-qubit circuit with 1 instruction:
    └── QubitLoss @ q[1]
    <BLANKLINE>

Once a qubit is lost, later instructions touching that qubit are removed by
:func:`~mimiqcircuits.lossmodel.sample_losses` until the qubit is reloaded.

Reloading a Lost Qubit
~~~~~~~~~~~~~~~~~~~~~~

:class:`~mimiqcircuits.operations.losschannel.QubitReload` makes a previously
lost qubit available again. After reloading, the qubit is reset to ``|0>`` and
can be used by later operations.

.. doctest:: loss

    >>> circuit = Circuit()
    >>> circuit.push(QubitLoss(), 0)
    1-qubit circuit with 1 instruction:
    └── QubitLoss @ q[0]
    <BLANKLINE>
    >>> circuit.push(QubitReload(), 0)
    1-qubit circuit with 2 instructions:
    ├── QubitLoss @ q[0]
    └── QubitReload @ q[0]
    <BLANKLINE>
    >>> circuit.push(GateX(), 0)
    1-qubit circuit with 3 instructions:
    ├── QubitLoss @ q[0]
    ├── QubitReload @ q[0]
    └── X @ q[0]
    <BLANKLINE>

Checking for Loss
~~~~~~~~~~~~~~~~~

MIMIQ provides two operations to query the loss status of a qubit.

:class:`~mimiqcircuits.operations.losschannel.CheckLoss` writes one classical
bit:

* ``1`` if the qubit is present
* ``0`` if the qubit is lost

It does not measure the quantum state.

.. doctest:: loss

    >>> circuit = Circuit()
    >>> circuit.push(CheckLoss(), 0, 0)
    1-qubit, 1-bit circuit with 1 instruction:
    └── CL @ q[0], c[0]
    <BLANKLINE>

:class:`~mimiqcircuits.operations.losschannel.MeasureCheckLoss` both measures
the qubit and reports whether it is still present.

.. doctest:: loss

    >>> circuit = Circuit()
    >>> circuit.push(MeasureCheckLoss(), 0, 0, 1)
    1-qubit, 2-bit circuit with 1 instruction:
    └── MCL @ q[0], c[0:1]
    <BLANKLINE>

The first classical bit stores the measurement result, and the second classical
bit stores the loss status.

Sampling Loss Events
--------------------

To process loss operations in a circuit, use
:func:`~mimiqcircuits.lossmodel.sample_losses` or the convenience wrapper
:meth:`~mimiqcircuits.circuit.Circuit.sample_losses`. These functions walk the
circuit, sample :class:`~mimiqcircuits.operations.losschannel.LossErr` events,
keep track of which qubits are currently lost, and return a rewritten circuit.

The ``rng`` argument is a random number generator. It is only used to make the
random loss samples reproducible. You can omit it if you do not need the same
random result every time.

.. doctest:: loss

    >>> rng = random.Random(42)
    >>> circuit = Circuit()
    >>> circuit.push(LossErr(0.2), 0)
    1-qubit circuit with 1 instruction:
    └── LossErr(0.2) @ q[0]
    <BLANKLINE>
    >>> circuit.push(GateH(), 0)
    1-qubit circuit with 2 instructions:
    ├── LossErr(0.2) @ q[0]
    └── H @ q[0]
    <BLANKLINE>
    >>> circuit.push(CheckLoss(), 0, 0)
    1-qubit, 1-bit circuit with 3 instructions:
    ├── LossErr(0.2) @ q[0]
    ├── H @ q[0]
    └── CL @ q[0], c[0]
    <BLANKLINE>
    >>> circuit.sample_losses(rng=rng)
    1-qubit, 1-bit circuit with 2 instructions:
    ├── H @ q[0]
    └── CL @ q[0], c[0]
    <BLANKLINE>

The basic behavior is:

* :class:`~mimiqcircuits.operations.losschannel.LossErr` may emit a
  :class:`~mimiqcircuits.operations.losschannel.QubitLoss`
* :class:`~mimiqcircuits.operations.losschannel.QubitLoss` marks a qubit as lost
* :class:`~mimiqcircuits.operations.losschannel.QubitReload` makes the qubit
  available again
* :class:`~mimiqcircuits.operations.losschannel.CheckLoss` and
  :class:`~mimiqcircuits.operations.losschannel.MeasureCheckLoss` are always
  kept
* Instructions acting only on lost qubits are dropped

If an instruction touches some lost qubits but not all of them,
:func:`~mimiqcircuits.lossmodel.sample_losses` consults a
:class:`~mimiqcircuits.lossmodel.LossModel`.

Loss Models
-----------

Why Loss Models Exist
~~~~~~~~~~~~~~~~~~~~~

A :class:`~mimiqcircuits.lossmodel.LossModel` is the user-defined policy used by
:func:`~mimiqcircuits.lossmodel.sample_losses` when an instruction is only
partially affected by loss. This happens, for example, when a two-qubit gate is
scheduled but one of its qubits has already been lost while the other one is
still present.

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

During :func:`~mimiqcircuits.lossmodel.sample_losses`, MIMIQ tracks which
qubits are currently lost and rewrites the circuit as follows:

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
    >>> circuit.push(QubitLoss(), 1)
    2-qubit circuit with 1 instruction:
    └── QubitLoss @ q[1]
    <BLANKLINE>
    >>> circuit.push(GateCX(), 0, 1)
    2-qubit circuit with 2 instructions:
    ├── QubitLoss @ q[1]
    └── CX @ q[0], q[1]
    <BLANKLINE>
    >>> model = LossModel().add_replace(GateCX(), Depolarizing1(0.2))
    >>> circuit.sample_losses(lossmodel=model)
    2-qubit circuit with 2 instructions:
    ├── QubitLoss @ q[1]
    └── Depolarizing(0.2) @ q[0]
    <BLANKLINE>

If the lost qubit is the control instead, the same rule keeps the replacement
on the surviving target qubit.

.. doctest:: loss

    >>> circuit = Circuit()
    >>> circuit.push(QubitLoss(), 0)
    1-qubit circuit with 1 instruction:
    └── QubitLoss @ q[0]
    <BLANKLINE>
    >>> circuit.push(GateCX(), 0, 1)
    2-qubit circuit with 2 instructions:
    ├── QubitLoss @ q[0]
    └── CX @ q[0], q[1]
    <BLANKLINE>
    >>> circuit.sample_losses(lossmodel=model)
    2-qubit circuit with 2 instructions:
    ├── QubitLoss @ q[0]
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
after the matched instruction. In a loss model, generated instructions touching
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
    >>> circuit.push(QubitLoss(), 1)
    2-qubit circuit with 1 instruction:
    └── QubitLoss @ q[1]
    <BLANKLINE>
    >>> circuit.push(GateCX(), 0, 1)
    2-qubit circuit with 2 instructions:
    ├── QubitLoss @ q[1]
    └── CX @ q[0], q[1]
    <BLANKLINE>
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
    >>> circuit.sample_losses(lossmodel=model)
    2-qubit circuit with 2 instructions:
    ├── QubitLoss @ q[1]
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
:class:`~mimiqcircuits.operations.losschannel.LossErr` instruction. The
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
