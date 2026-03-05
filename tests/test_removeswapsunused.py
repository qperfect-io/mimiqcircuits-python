import pytest
import numpy as np
import symengine as se
import sympy as sp
from mimiqcircuits import *
from mimiqcircuits.circuit_extras import remove_swaps, remove_unused

import mimiqcircuits as mc


def test_remove_swaps_recursive_nested_gatedecl():
    # Inner gate (has SWAP)
    inner = Circuit()
    inner.push(GateSWAP(), 0, 1)
    inner.push(GateCX(), 1, 0)
    Inner = GateDecl("Inner", (), inner)

    # Mid gate (has SWAP + calls Inner)
    mid = Circuit()
    mid.push(GateSWAP(), 0, 1)
    mid.push(GateCall(Inner, ()), 0, 1)
    Mid = GateDecl("Mid", (), mid)

    # Outer gate (has SWAP + calls Mid)
    outer = Circuit()
    outer.push(GateSWAP(), 0, 1)
    outer.push(GateCall(Mid, ()), 0, 1)
    Outer = GateDecl("Outer", (), outer)

    # Top-level circuit (has SWAP + calls Outer)
    c = Circuit()
    c.push(GateSWAP(), 0, 1)
    c.push(GateCall(Outer, ()), 0, 1)

    # Apply recursive swap removal
    new_c, _ = remove_swaps(c, recursive=True)

    # Assertions: no SWAP anywhere

    # Top level
    assert not any(isinstance(inst.get_operation(), GateSWAP) for inst in new_c)

    # Outer
    Outer2 = new_c.instructions[0].get_operation()._decl
    assert not any(
        isinstance(inst.get_operation(), GateSWAP) for inst in Outer2.circuit
    )

    # Mid
    Mid2 = Outer2.circuit.instructions[0].get_operation()._decl
    assert not any(isinstance(inst.get_operation(), GateSWAP) for inst in Mid2.circuit)

    # Inner
    Inner2 = Mid2.circuit.instructions[0].get_operation()._decl
    assert not any(
        isinstance(inst.get_operation(), GateSWAP) for inst in Inner2.circuit
    )


def test_remove_unused_simple():
    c = Circuit()
    c.push(GateH(), 0)
    c.push(GateCX(), 0, 2)  # qubit 1 is unused
    c.push(Measure(), 2, 2)  # bits 0 & 1 are unused
    c.push(ExpectationValue(GateX()), 0, 1)  # zvar 0 is unused

    assert c.num_qubits() == 3
    assert c.num_bits() == 3
    assert c.num_zvars() == 2

    new_c, qubit_map, bit_map, zvar_map = remove_unused(c)

    # Only two qubits should remain
    assert new_c.num_qubits() == 2
    assert new_c.num_bits() == 1
    assert new_c.num_zvars() == 1

    # No operation should reference qubit 2 anymore
    assert all(q < new_c.num_qubits() for inst in new_c for q in inst.get_qubits())
    assert qubit_map[0] == 0
    assert qubit_map[2] == 1

    assert bit_map[2] == 0

    assert zvar_map[1] == 0
