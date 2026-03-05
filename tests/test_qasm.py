# Copyright (C) 2023-2025 QPerfect. All Rights Reserved.
# Proprietary and confidential.


import os
from mimiqcircuits.qasm import loads, dumps, load, dump
from mimiqcircuits.qasm.lexer import tokenize
from mimiqcircuits.qasm.parser import parseopenqasm
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

    assert "gate myop" in qasm.lower() or "gate g_myop" in qasm.lower()

    h_count = qasm.count("h ")
    assert h_count == 1, (
        f"Expected 1 'h' instruction (in definition), found {h_count}. QASM:\n{qasm}"
    )

    import re

    match = re.search(r"gate (g_myop_[0-9a-f]+)", qasm)
    if not match:
        match = re.search(r"gate (myop_[0-9a-f]+)", qasm)

    assert match, "Could not find gate declaration for MyOp"
    gate_name = match.group(1)

    call_count = qasm.count(gate_name)
    assert call_count == 3, (
        f"Expected 3 occurrences of {gate_name} (1 decl + 2 calls), found {call_count}"
    )
