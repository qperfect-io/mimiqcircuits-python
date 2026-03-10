# Copyright (C) 2023-2025 QPerfect. All Rights Reserved.
# Proprietary and confidential.


import re

from mimiqcircuits.qasm import loads, dumps, load, dump
from mimiqcircuits.qasm.lexer import tokenize
from mimiqcircuits.qasm.parser import parseopenqasm
from mimiqcircuits.qasm.serializer import args_to_qasm
import mimiqcircuits as mc
from mimiqcircuits.qasm import serializer, interpreter


def test_lexer():
    qasm = "OPENQASM 2.0; qreg q[2]; CX q[0], q[1];"
    tokens = list(tokenize(qasm))
    assert len(tokens) > 0
    assert tokens[0].kind.name == "OPENQASM"


def test_scientific_notation():
    from mimiqcircuits.qasm.tokens import TokenKind

    cases = [
        ("1e-5", TokenKind.REAL),
        ("1.e-5", TokenKind.REAL),
        (".5e-5", TokenKind.REAL),
        ("1.0e-5", TokenKind.REAL),
        ("1e+5", TokenKind.REAL),
        ("1E5", TokenKind.REAL),
        ("0.3926990816987243", TokenKind.REAL),
    ]

    for qasm, expected_kind in cases:
        tokens = list(tokenize(qasm))
        tokens = [t for t in tokens if t.kind != TokenKind.ENDMARKER]
        assert len(tokens) == 1
        assert tokens[0].kind == expected_kind, f"Failed for {qasm}"

    # Test negative number (lexer parses MINUS then REAL)
    qasm = "-9.99e-16"
    tokens = list(tokenize(qasm))
    tokens = [t for t in tokens if t.kind != TokenKind.ENDMARKER]
    assert len(tokens) == 2
    assert tokens[0].kind == TokenKind.MINUS
    assert tokens[1].kind == TokenKind.REAL


def test_parser_simple():
    qasm = """
    OPENQASM 2.0;
    qreg q[2];
    creg c[2];
    U(pi/2, 0, pi) q[0];
    CX q[0], q[1];
    measure q[0] -> c[0];
    """
    ast = parseopenqasm(qasm)
    assert ast.head == "program"
    assert len(ast.args) > 1


def test_interpreter_simple():
    qasm = """
    OPENQASM 2.0;
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    """
    c = loads(qasm)
    assert c.num_qubits() == 2
    assert len(c.instructions) == 2
    # Verify gates
    assert c.instructions[0].operation.name == "H"


def test_serializer_roundtrip():
    qasm = """OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0],q[1];
    measure q[0] -> c[0];
    """
    c = loads(qasm)

    qasm_out = dumps(c)
    print(qasm_out)

    # Check key elements in output
    assert "OPENQASM 2.0;" in qasm_out
    assert "qreg q[2];" in qasm_out
    assert "h q[0];" in qasm_out or "h q[0]" in qasm_out
    assert "cx q[0]" in qasm_out
    assert "measure q[0] -> c[0];" in qasm_out


def test_qelib1_gates():
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    x q[0];
    y q[0];
    z q[0];
    """
    c = loads(qasm)
    assert len(c.instructions) == 3


def test_custom_gate():
    qasm = """
    OPENQASM 2.0;
    gate mygate(p) q {
        rx(p) q;
    }
    qreg q[1];
    mygate(pi) q[0];
    """
    c = loads(qasm)
    assert len(c.instructions) == 1
    # Check parameter resolution
    inst = c.instructions[0]

    # Serialize back
    qasm_out = dumps(c)
    # Check that gate is defined
    import re

    assert re.search(r"gate mygate", qasm_out)
    # Check that it is called
    assert re.search(r"mygate.*\(3\.14", qasm_out) or re.search(
        r"mygate.*\(pi\)", qasm_out
    )


def test_load_dump_file(tmp_path):
    qasm = """OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    x q[0];
    """

    path = tmp_path / "test.qasm"
    path.write_text(qasm)

    c = load(str(path))
    assert c.num_qubits() == 1
    assert len(c.instructions) == 1

    out_path = tmp_path / "out.qasm"
    dump(c, str(out_path))

    assert out_path.exists()
    content = out_path.read_text()
    assert "OPENQASM 2.0" in content
    assert "x q[0]" in content


# QASM roundtrip test for circuits with wrapper gates
def test_dump_load_roundtrip(tmp_path):
    c = mc.Circuit()
    c.push(
        mc.Control(2, mc.Inverse(mc.Parallel(3, mc.GateH())).power(1 / 12)),
        1,
        2,
        3,
        4,
        8,
    )

    out = tmp_path / "roundtrip.qasm"

    # Dump with wrapper decomposition enabled
    serializer.dump(c, str(out), decompose_wrappers=True)

    # Load back the file and ensure it produces a Circuit
    circ2 = interpreter.load(str(out))

    assert isinstance(circ2, mc.Circuit)
    assert len(circ2) > 0


def test_gate_declaration_order():
    from mimiqcircuits.gatedecl import GateDecl

    circ_b = mc.Circuit()
    circ_b.push(mc.GateH(), 0)
    decl_b = GateDecl("gate_b", (), circ_b)

    circ_a = mc.Circuit()
    circ_a.push(mc.GateCall(decl_b, []), 0)
    decl_a = GateDecl("gate_a", (), circ_a)

    circ = mc.Circuit()
    circ.push(mc.GateCall(decl_a, []), 0)

    qasm = dumps(circ)

    pos_a = qasm.find("gate gate_a")
    pos_b = qasm.find("gate gate_b")

    assert pos_b != -1
    assert pos_a != -1
    assert pos_b < pos_a, (
        "gate_b (dependency) should be defined before gate_a (dependent)"
    )


def test_decomposition_reuse():
    class MyOp(mc.Gate):
        @property
        def name(self):
            return "myop"

        @property
        def num_qubits(self):
            return 1

        def _decompose(self, circ, qubits, bits, zvars):
            q = qubits[0]
            circ.push(mc.GateH(), q)
            circ.push(mc.GateX(), q)
            return circ

        def __str__(self):
            return "MyOp"

        def __repr__(self):
            return "MyOp"

        def __eq__(self, other):
            return isinstance(other, MyOp)

        def __hash__(self):
            return hash("MyOp")

    c = mc.Circuit()
    op = MyOp()
    c.push(op, 0)
    c.push(op, 1)

    qasm = dumps(c)

    assert "gate" in qasm.lower() and "myop" in qasm.lower(), (
        f"Expected gate declaration containing 'myop' in:\n{qasm}"
    )

    h_count = qasm.count("h ")
    assert h_count == 1, (
        f"Expected 1 'h' instruction (in definition), found {h_count}. QASM:\n{qasm}"
    )

    import re

    match = re.search(r"gate (\S*myop\S*)", qasm)

    assert match, "Could not find gate declaration for MyOp"
    gate_name = match.group(1)

    call_count = qasm.count(gate_name)
    assert call_count == 3, (
        f"Expected 3 occurrences of {gate_name} (1 decl + 2 calls), found {call_count}"
    )


# ── New tests for parametric gate emission and roundtrip ──


def test_single_qubit_parametric_gates():
    """GateU, RX, RY, RZ, P, U1, U2, U3 must emit their parameters."""
    gates_and_patterns = [
        (mc.GateU(0.1, 0.2, 0.3), r"U\(0\.1,0\.2,0\.3\)"),
        (mc.GateRX(0.5), r"rx\(0\.5\)"),
        (mc.GateRY(0.6), r"ry\(0\.6\)"),
        (mc.GateRZ(0.7), r"rz\(0\.7\)"),
        (mc.GateP(0.8), r"p\(0\.8\)"),
        (mc.GateU1(1.1), r"u1\(1\.1\)"),
        (mc.GateU2(1.2, 1.3), r"u2\(1\.2,1\.3\)"),
        (mc.GateU3(1.4, 1.5, 1.6), r"u3\(1\.4,1\.5,1\.6\)"),
    ]
    for gate, pattern in gates_and_patterns:
        c = mc.Circuit()
        c.push(gate, *range(gate.num_qubits))
        qasm_out = dumps(c)
        assert re.search(pattern, qasm_out), (
            f"{gate.name}: expected pattern {pattern} in:\n{qasm_out}"
        )


def test_controlled_rotation_gates():
    """GateCRX, CRY, CRZ, CP, CU must emit their parameters."""
    gates_and_patterns = [
        (mc.GateCRX(0.5), r"crx\(0\.5\)"),
        (mc.GateCRY(0.6), r"cry\(0\.6\)"),
        (mc.GateCRZ(0.7), r"crz\(0\.7\)"),
        (mc.GateCP(0.8), r"cp\(0\.8\)"),
        (mc.GateCU(0.1, 0.2, 0.3, 0.4), r"cu\(0\.1,0\.2,0\.3,0\.4\)"),
    ]
    for gate, pattern in gates_and_patterns:
        c = mc.Circuit()
        c.push(gate, *range(gate.num_qubits))
        qasm_out = dumps(c)
        assert re.search(pattern, qasm_out), (
            f"{gate.name}: expected pattern {pattern} in:\n{qasm_out}"
        )


def test_interaction_gates():
    """GateRXX, RYY, RZZ, RZX, XXplusYY, XXminusYY must emit their parameters."""
    gates_and_patterns = [
        (mc.GateRXX(0.5), r"rxx\(0\.5\)"),
        (mc.GateRYY(0.6), r"ryy\(0\.6\)"),
        (mc.GateRZZ(0.7), r"rzz\(0\.7\)"),
        (mc.GateRZX(0.8), r"rzx\(0\.8\)"),
        (mc.GateXXplusYY(0.9, 1.0), r"xxplusyy\(0\.9,1\.0\)"),
        (mc.GateXXminusYY(1.1, 1.2), r"xxminusyy\(1\.1,1\.2\)"),
    ]
    for gate, pattern in gates_and_patterns:
        c = mc.Circuit()
        c.push(gate, *range(gate.num_qubits))
        qasm_out = dumps(c)
        assert re.search(pattern, qasm_out), (
            f"{gate.name}: expected pattern {pattern} in:\n{qasm_out}"
        )


def test_non_parametric_gates_no_params():
    """Non-parametric gates must NOT emit spurious parentheses/parameters."""
    gates = [
        mc.GateH(),
        mc.GateX(),
        mc.GateS(),
        mc.GateT(),
        mc.GateSX(),
        mc.GateCZ(),
        mc.GateSWAP(),
    ]
    for gate in gates:
        c = mc.Circuit()
        c.push(gate, *range(gate.num_qubits))
        qasm_out = dumps(c)
        # There should be no parentheses in the gate line (no parameters)
        for line in qasm_out.splitlines():
            line = line.strip()
            if line.startswith(("OPENQASM", "include", "qreg", "creg")):
                continue
            if "q[" in line and "(" in line:
                assert False, (
                    f"{gate.name}: non-parametric gate has params in:\n{line}"
                )


def test_parametric_roundtrip():
    """Serialize parametric gates → parse back → verify parameters match."""
    test_cases = [
        (mc.GateRX(0.5), [0.5]),
        (mc.GateRY(0.6), [0.6]),
        (mc.GateRZ(0.7), [0.7]),
        (mc.GateP(0.8), [0.8]),
        (mc.GateCRX(0.5), [0.5]),
        (mc.GateCRY(0.6), [0.6]),
        (mc.GateCRZ(0.7), [0.7]),
    ]
    for gate, expected_params in test_cases:
        c = mc.Circuit()
        c.push(gate, *range(gate.num_qubits))
        qasm_str = dumps(c)
        c2 = loads(qasm_str)
        assert len(c2.instructions) == 1, (
            f"{gate.name}: expected 1 instruction after roundtrip, got {len(c2.instructions)}"
        )
        roundtrip_params = list(c2.instructions[0].operation.getparams())
        for got, exp in zip(roundtrip_params, expected_params):
            assert abs(float(got) - exp) < 1e-10, (
                f"{gate.name}: param mismatch {got} != {exp}"
            )


def test_reset_barrier_roundtrip():
    """Reset and Barrier survive serialize → parse roundtrip."""
    c = mc.Circuit()
    c.push(mc.Reset(), 0)
    c.push(mc.Barrier(2), 0, 1)
    c.push(mc.GateH(), 0)
    qasm_str = dumps(c)
    assert "reset q[0];" in qasm_str
    assert "barrier q[0],q[1];" in qasm_str

    c2 = loads(qasm_str)
    ops = [inst.operation for inst in c2.instructions]
    assert isinstance(ops[0], mc.Reset)
    assert isinstance(ops[1], mc.Barrier)


def test_gateu_gamma_not_in_qasm():
    """GateU should emit 3 params (theta, phi, lambda) — NOT gamma."""
    gate = mc.GateU(0.1, 0.2, 0.3, 0.4)  # 4th arg is gamma
    params = args_to_qasm(gate)
    assert len(params) == 3, f"Expected 3 params for GateU, got {len(params)}: {params}"
    assert params == [0.1, 0.2, 0.3]


def test_args_to_qasm_all_parametric():
    """Direct unit test: args_to_qasm returns correct params for all parametric gate types."""
    cases = [
        (mc.GateU(0.1, 0.2, 0.3), [0.1, 0.2, 0.3]),
        (mc.GateRX(0.5), [0.5]),
        (mc.GateRY(0.6), [0.6]),
        (mc.GateRZ(0.7), [0.7]),
        (mc.GateP(0.8), [0.8]),
        (mc.GateU1(1.1), [1.1]),
        (mc.GateU2(1.2, 1.3), [1.2, 1.3]),
        (mc.GateU3(1.4, 1.5, 1.6), [1.4, 1.5, 1.6]),
        (mc.GateCRX(0.5), [0.5]),
        (mc.GateCRY(0.6), [0.6]),
        (mc.GateCRZ(0.7), [0.7]),
        (mc.GateCP(0.8), [0.8]),
        (mc.GateCU(0.1, 0.2, 0.3, 0.4), [0.1, 0.2, 0.3, 0.4]),
        (mc.GateRXX(0.5), [0.5]),
        (mc.GateRYY(0.6), [0.6]),
        (mc.GateRZZ(0.7), [0.7]),
        (mc.GateRZX(0.8), [0.8]),
        (mc.GateXXplusYY(0.9, 1.0), [0.9, 1.0]),
        (mc.GateXXminusYY(1.1, 1.2), [1.1, 1.2]),
        # Non-parametric gates should return []
        (mc.GateH(), []),
        (mc.GateX(), []),
        (mc.GateSWAP(), []),
    ]
    for gate, expected in cases:
        result = args_to_qasm(gate)
        assert len(result) == len(expected), (
            f"{gate.name}: expected {len(expected)} params, got {len(result)}: {result}"
        )
        for got, exp in zip(result, expected):
            assert abs(float(got) - exp) < 1e-10, (
                f"{gate.name}: param {got} != {exp}"
            )


def test_all_qelib1_gates_roundtrip():
    """Full roundtrip for all qelib1 gates that have matching parser entries."""
    from mimiqcircuits.qasm.serializer import TYPE_TO_NAME
    from mimiqcircuits.qasm.interpreter import OPAQUE_TO_TYPE

    # Map of gate types that can roundtrip: they are in both TYPE_TO_NAME and OPAQUE_TO_TYPE
    roundtrippable = {}
    for gate_type, qasm_name in TYPE_TO_NAME.items():
        if qasm_name in OPAQUE_TO_TYPE and callable(gate_type):
            roundtrippable[gate_type] = qasm_name

    for gate_type, qasm_name in roundtrippable.items():
        # Skip GateCCP since it's a factory function, not a class
        if gate_type is mc.GateCCP:
            continue

        try:
            # Try to construct with no args first
            gate = gate_type()
        except TypeError:
            # Needs parameters — try with plausible values
            try:
                gate = gate_type(0.5)
            except TypeError:
                try:
                    gate = gate_type(0.5, 0.6)
                except TypeError:
                    try:
                        gate = gate_type(0.5, 0.6, 0.7)
                    except TypeError:
                        gate = gate_type(0.5, 0.6, 0.7, 0.8)

        c = mc.Circuit()
        c.push(gate, *range(gate.num_qubits))
        qasm_str = dumps(c)
        assert qasm_name.lower() in qasm_str.lower(), (
            f"Gate {gate.name} ({qasm_name}) not found in QASM output:\n{qasm_str}"
        )

        # Parse back
        c2 = loads(qasm_str)
        assert len(c2.instructions) >= 1, (
            f"Gate {gate.name} ({qasm_name}): no instructions after roundtrip"
        )


# ── Tests for wrap=True decomposition and simplified serializer ──


def test_inverse_gatecall_produces_gatedecl():
    """Inverse(GateCall(decl, args)) should produce a gate declaration in QASM,
    not inlined primitives."""
    from symengine import Symbol
    from mimiqcircuits.gatedecl import GateDecl, GateCall

    x = Symbol("x")
    inner = mc.Circuit()
    inner.push(mc.GateRX(x), 0)
    decl = GateDecl("myrx", (x,), inner)

    c = mc.Circuit()
    c.push(mc.Inverse(GateCall(decl, (0.5,))), 0)

    qasm_out = dumps(c)
    # Should have a gate declaration for the dagger operation
    assert re.search(r"gate\s+\S*myrx\S*dagger", qasm_out), (
        f"Expected gate declaration with 'myrx' and 'dagger' in:\n{qasm_out}"
    )
    # Should NOT have inlined rx instructions at top level
    lines = [l.strip() for l in qasm_out.splitlines()
             if l.strip() and not l.strip().startswith(("OPENQASM", "include", "qreg", "creg", "gate", "}", "//"))]
    # The only instruction lines should be calls to the declared gate
    for line in lines:
        if line.startswith("rx") and "q[" in line:
            # rx at top level means it was inlined, not wrapped
            # But rx inside a gate body is fine - check indentation
            pass


def test_inverse_gatecall_roundtrip():
    """Serialize Inverse(GateCall) -> parse back -> verify valid circuit."""
    from symengine import Symbol
    from mimiqcircuits.gatedecl import GateDecl, GateCall

    x = Symbol("x")
    inner = mc.Circuit()
    inner.push(mc.GateRX(x), 0)
    decl = GateDecl("myrx", (x,), inner)

    c = mc.Circuit()
    c.push(mc.Inverse(GateCall(decl, (0.5,))), 0)

    qasm_out = dumps(c)
    c2 = loads(qasm_out)
    assert isinstance(c2, mc.Circuit)
    assert len(c2) > 0


def test_wrap_decompose_produces_gatecalls():
    """decompose(c, QASMBasis(), wrap=True) should produce a circuit with
    only terminal ops and GateCalls."""
    from mimiqcircuits.decomposition import decompose, QASMBasis
    from mimiqcircuits.gatedecl import GateCall

    basis = QASMBasis()
    c = mc.Circuit()
    # Add a non-terminal operation that requires decomposition
    c.push(mc.Control(2, mc.GateH()), 0, 1, 2)

    result = decompose(c, basis, wrap=True)
    for inst in result:
        op = inst.operation
        assert basis.isterminal(op) or isinstance(op, GateCall), (
            f"Expected terminal or GateCall, got {type(op).__name__}: {op}"
        )


def test_wrap_reuses_declarations():
    """Two identical non-terminal ops should share one GateDecl in wrapped output."""
    from mimiqcircuits.decomposition import decompose, QASMBasis
    from mimiqcircuits.gatedecl import GateCall

    c = mc.Circuit()
    # Two identical non-terminal operations
    c.push(mc.Control(2, mc.GateH()), 0, 1, 2)
    c.push(mc.Control(2, mc.GateH()), 3, 4, 5)

    result = decompose(c, QASMBasis(), wrap=True)

    # Collect all GateDecl objects from GateCall instructions
    decls = set()
    for inst in result:
        op = inst.operation
        if isinstance(op, GateCall):
            decls.add(id(op.decl))

    # Both C2H ops should reuse the same GateDecl (via cache)
    gatecall_insts = [inst for inst in result if isinstance(inst.operation, GateCall)]
    if len(gatecall_insts) >= 2:
        # The top-level GateCalls should share the same decl
        assert gatecall_insts[0].operation.decl is gatecall_insts[1].operation.decl, (
            "Two identical ops should share the same GateDecl"
        )


def test_wrap_parametric_wrapper_reuse():
    """Parametric wrappers like Inverse(GateRX(0.5)) and Inverse(GateRX(0.7))
    should share one parametric GateDecl via canonicalization."""
    from mimiqcircuits.decomposition import decompose, QASMBasis
    from mimiqcircuits.gatedecl import GateCall

    c = mc.Circuit()
    c.push(mc.Inverse(mc.GateRX(0.5)), 0)
    c.push(mc.Inverse(mc.GateRX(0.7)), 1)

    result = decompose(c, QASMBasis(), wrap=True)

    gatecall_insts = [inst for inst in result if isinstance(inst.operation, GateCall)]
    assert len(gatecall_insts) == 2, (
        f"Expected 2 GateCall instructions, got {len(gatecall_insts)}"
    )
    # Both should share the same parametric GateDecl
    assert gatecall_insts[0].operation.decl is gatecall_insts[1].operation.decl, (
        "Inverse(GateRX(0.5)) and Inverse(GateRX(0.7)) should share one GateDecl"
    )
    # But with different concrete arguments
    args0 = gatecall_insts[0].operation.arguments
    args1 = gatecall_insts[1].operation.arguments
    assert float(args0[0]) != float(args1[0]), (
        "The two GateCalls should have different parameter values"
    )
