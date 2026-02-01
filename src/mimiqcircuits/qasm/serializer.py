# Copyright (C) 2023 QPerfect. All Rights Reserved.
# Proprietary and confidential.
#
import io
import re
from typing import Any, Dict, List, Union

import mimiqcircuits as mc
from mimiqcircuits.gatedecl import GateCall, GateDecl
from mimiqcircuits.qasm.exceptions import QASMSerializationError
from mimiqcircuits.qasm.parser import QASMExpr

# Map mimiqcircuits types to QASM names
TYPE_TO_NAME = {
    mc.GateU: "U",
    mc.GateCX: "CX",
    mc.GateU3: "u3",
    mc.GateU2: "u2",
    mc.GateU1: "u1",
    mc.GateID: "id",
    mc.GateP: "p",
    mc.GateX: "x",
    mc.GateY: "y",
    mc.GateZ: "z",
    mc.GateH: "h",
    mc.GateS: "s",
    mc.GateSDG: "sdg",
    mc.GateT: "t",
    mc.GateTDG: "tdg",
    mc.GateRX: "rx",
    mc.GateRY: "ry",
    mc.GateRZ: "rz",
    mc.GateSX: "sx",
    mc.GateSXDG: "sxdg",
    mc.GateCZ: "cz",
    mc.GateCY: "cy",
    mc.GateSWAP: "swap",
    mc.GateCH: "ch",
    mc.GateCCX: "ccx",
    mc.GateCSWAP: "cswap",
    mc.GateCRX: "crx",
    mc.GateCRY: "cry",
    mc.GateCRZ: "crz",
    mc.GateCP: "cp",
    mc.GateCS: "cs",
    mc.GateCSDG: "csdg",
    mc.GateCSX: "csx",
    mc.GateCSXDG: "csxdg",
    mc.GateCU: "cu",
    mc.GateRXX: "rxx",
    mc.GateRZZ: "rzz",
    mc.GateRYY: "ryy",
    mc.GateRZX: "rzx",
    mc.GateC3X: "c3x",
    mc.GateXXminusYY: "xxminusyy",
    mc.GateXXplusYY: "xxplusyy",
    mc.GateDCX: "dcx",
    mc.GateECR: "ecr",
    mc.GateISWAP: "iswap",
    mc.GateCCP: "ccp",
}

CHAR_REPLACEMENTS = {
    "†": "dg",
    "^": "pow",
    "+": "plus",
    "-": "minus",
    "/": "div",
    "*": "mul",
    "(": "_",
    ")": "",
    "[": "_",
    "]": "",
    "=": "eq",
    ",": "_",
}

GREEK_REPLACEMENTS = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ι": "iota",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "ο": "omicron",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
    # Uppercase
    "Α": "Alpha",
    "Β": "Beta",
    "Γ": "Gamma",
    "Δ": "Delta",
    "Ε": "Epsilon",
    "Ζ": "Zeta",
    "Η": "Eta",
    "Θ": "Theta",
    "Ι": "Iota",
    "Κ": "Kappa",
    "Λ": "Lambda",
    "Μ": "Mu",
    "Ν": "Nu",
    "Ξ": "Xi",
    "Ο": "Omicron",
    "Π": "Pi",
    "Ρ": "Rho",
    "Σ": "Sigma",
    "Τ": "Tau",
    "Υ": "Upsilon",
    "Φ": "Phi",
    "Χ": "Chi",
    "Ψ": "Psi",
    "Ω": "Omega",
}

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_valid_identifier(name: str) -> bool:
    return bool(_IDENT_RE.match(name))


def _require_valid_identifier(name: str, context: str):
    if not _is_valid_identifier(name):
        raise QASMSerializationError(
            f"Invalid OpenQASM identifier '{name}' for {context}"
        )


def _sanitize_identifier(name: str) -> str:
    if name is None:
        return "gate"

    out = []
    for c in name:
        if c in CHAR_REPLACEMENTS:
            out.append(CHAR_REPLACEMENTS[c])
        elif c in GREEK_REPLACEMENTS:
            out.append(GREEK_REPLACEMENTS[c])
        elif c.isalnum() or c == "_":
            out.append(c)
        else:
            out.append("_")

    sanitized = "".join(out)

    # Remove multiple underscores
    sanitized = re.sub(r"__+", "_", sanitized)

    # Trim trailing underscore
    if sanitized.endswith("_") and len(sanitized) > 1:
        sanitized = sanitized[:-1]

    if not sanitized:
        return "gate"

    # Prefix with 'g_' if not starting with lowercase letter
    # OpenQASM identifiers must start with lowercase letter
    first = sanitized[0]
    if not first.islower() or not first.isalpha():
        sanitized = "g_" + sanitized

    sanitized = re.sub(r"__+", "_", sanitized)
    return sanitized


def uses_std_gates(instructions) -> bool:
    for inst in instructions:
        op = inst.operation
        # Built-in gates: U, Measure, Reset, Barrier. CX is also built-in.
        if isinstance(op, (mc.GateU, mc.Measure, mc.Reset, mc.Barrier)):
            continue

        # Check for CX (Control(1, X))
        if isinstance(op, mc.Control):
            if op.num_controls == 1 and isinstance(op.get_operation(), mc.GateX):
                continue  # CX is built-in
            # Other controls might need include (e.g. cy, cz, ch are in qelib1)
            pass

        elif isinstance(op, GateCall):
            if uses_std_gates(op.decl.circuit.instructions):
                return True
        elif type(op) in TYPE_TO_NAME:
            return True

    return False


def gate_to_qasm(gate: mc.Operation) -> str:
    # Special handling for Control gates (CX, CY, CZ, CH, etc)
    if isinstance(gate, mc.Control):
        if gate.num_controls == 1:
            base = gate.get_operation()
            if isinstance(base, mc.GateX):
                return "cx"
            if isinstance(base, mc.GateY):
                return "cy"
            if isinstance(base, mc.GateZ):
                return "cz"
            if isinstance(base, mc.GateH):
                return "ch"
            if isinstance(base, mc.GateSWAP):
                return "cswap"
            # Add others if needed

    if isinstance(gate, GateCall):
        _require_valid_identifier(gate.decl.name, "gate declaration name")
        return gate.decl.name

    # Map common Power aliases (e.g. X**0.5 -> SX) to QASM names.
    if isinstance(gate, mc.Power):
        base = gate.op
        exp = gate.exponent
        try:
            fexp = float(exp)
        except Exception:
            fexp = None

        if fexp == 0.5 and isinstance(base, mc.GateX):
            return TYPE_TO_NAME.get(mc.GateSX, "sx")
        if fexp == -0.5 and isinstance(base, mc.GateX):
            return TYPE_TO_NAME.get(mc.GateSXDG, "sxdg")

    t = type(gate)
    if t in TYPE_TO_NAME:
        return TYPE_TO_NAME[t]

    raise QASMSerializationError(f"Unsupported gate type: {t}")


def args_to_qasm(gate: mc.Operation) -> List[Any]:
    if isinstance(gate, GateCall):
        return list(gate.arguments)

    # Standard gates
    if isinstance(gate, mc.GateU):
        return [gate.theta, gate.phi, gate.lmbda]
    if isinstance(gate, mc.GateRX):
        return [gate.theta]
    if isinstance(gate, mc.GateRY):
        return [gate.theta]
    if isinstance(gate, mc.GateRZ):
        return [gate.lmbda]
    if isinstance(gate, mc.GateP):
        return [gate.lmbda]

    return []


def qubits_to_qasm(qubits: Union[int, List[int]], size: int) -> List[QASMExpr]:
    if isinstance(qubits, int):
        qubits = [qubits]
    return [QASMExpr("ref", ["q", q]) for q in qubits]


def instruction_to_qasm(
    inst: mc.Instruction,
    num_qubits: int,
    decl_names: Dict[GateDecl, str] = None,
    sanitize_names: bool = False,
    op_cache: Dict = None,
) -> QASMExpr:
    op = inst.operation
    qs = inst.qubits

    if isinstance(op, mc.Power) and isinstance(op.op, mc.Inverse):
        inner = op.op.op
        if hasattr(inner, "power"):
            try:
                new_inner = inner.power(op.exponent)
                new_op = mc.Inverse(new_inner)
                return instruction_to_qasm(
                    mc.Instruction(new_op, qs, inst.bits),
                    num_qubits,
                    decl_names,
                    sanitize_names=sanitize_names,
                    op_cache=op_cache,
                )
            except Exception:
                pass

    if isinstance(op, (mc.Inverse, mc.Power)):
        try:
            subcirc = inst.decompose()
        except Exception:
            subcirc = None

        if subcirc is not None and len(subcirc.instructions) > 0:
            stmts = []
            for subinst in subcirc.instructions:
                sub_q = instruction_to_qasm(
                    subinst,
                    num_qubits,
                    decl_names,
                    sanitize_names=sanitize_names,
                    op_cache=op_cache,
                )
                if isinstance(sub_q, list):
                    stmts.extend(sub_q)
                else:
                    stmts.append(sub_q)
            return stmts

    if isinstance(op, mc.Measure):
        q_expr = qubits_to_qasm(qs[0], num_qubits)[0]
        cs = inst.bits
        c_expr = QASMExpr("ref", ["c", cs[0]]) if cs else QASMExpr("ref", ["c", qs[0]])
        return QASMExpr("measure", [q_expr, c_expr])

    if isinstance(op, mc.Reset):
        return QASMExpr("reset", qubits_to_qasm(qs, num_qubits))

    if isinstance(op, mc.Barrier):
        return QASMExpr("barrier", qubits_to_qasm(qs, num_qubits))

    if isinstance(op, GateCall):
        if decl_names is None:
            raise QASMSerializationError(
                "GateCall found but no declaration mapping provided", op
            )

        if op.decl not in decl_names:
            base = op.decl.name
            if sanitize_names:
                base = _sanitize_identifier(base)
            else:
                _require_valid_identifier(base, "gate declaration name")
            unique_name = f"{base}_{hex(id(op.decl))[2:]}"
            decl_names[op.decl] = unique_name

        name = decl_names[op.decl]
        gate_args = list(op.arguments)
        q_exprs = qubits_to_qasm(qs, num_qubits)

        gate_call = QASMExpr("call", [name] + gate_args)
        return QASMExpr("unitary", [gate_call] + q_exprs)

    try:
        name = gate_to_qasm(op)
    except QASMSerializationError as e:
        if op_cache is not None and op in op_cache:
            decl = op_cache[op]
            if decl not in decl_names:
                base = decl.name
                if sanitize_names:
                    base = _sanitize_identifier(base)
                else:
                    _require_valid_identifier(base, "gate declaration name")
                unique_name = f"{base}_{hex(id(decl))[2:]}"
                decl_names[decl] = unique_name

            name = decl_names[decl]
            gate_call = QASMExpr("call", [name])
            targets = qubits_to_qasm(qs, num_qubits)
            return QASMExpr("unitary", [gate_call] + targets)

        try:
            decomp_circ = op.decompose()
            is_trivial = (
                len(decomp_circ.instructions) == 1
                and decomp_circ.instructions[0].operation == op
            )

            if (
                decomp_circ is not None
                and len(decomp_circ.instructions) > 0
                and not is_trivial
            ):
                op_name = getattr(op, "name", "gate") or "gate"
                clean_name = _sanitize_identifier(op_name)

                decl = GateDecl(clean_name, (), decomp_circ)

                if op_cache is not None:
                    op_cache[op] = decl

                if decl_names is not None:
                    if decl not in decl_names:
                        base = clean_name
                        if sanitize_names:
                            base = _sanitize_identifier(base)
                        else:
                            _require_valid_identifier(base, "gate declaration name")
                        unique_name = f"{base}_{hex(id(decl))[2:]}"
                        decl_names[decl] = unique_name

                name = decl_names[decl]
                gate_call = QASMExpr("call", [name])
                targets = qubits_to_qasm(qs, num_qubits)
                return QASMExpr("unitary", [gate_call] + targets)
        except Exception:
            pass

        try:
            subcirc = inst.decompose()
        except Exception:
            subcirc = None

        if subcirc is not None and len(subcirc.instructions) > 0:
            stmts = []
            for subinst in subcirc.instructions:
                sub_q = instruction_to_qasm(
                    subinst,
                    num_qubits,
                    decl_names,
                    sanitize_names=sanitize_names,
                    op_cache=op_cache,
                )
                if isinstance(sub_q, list):
                    stmts.extend(sub_q)
                else:
                    stmts.append(sub_q)
            return stmts
        raise

    gate_args = args_to_qasm(op)
    targets = qubits_to_qasm(qs, num_qubits)

    gate_call = QASMExpr("call", [name] + gate_args)
    return QASMExpr("unitary", [gate_call] + targets)


def collect_gatedecls(
    c: mc.Circuit, sanitize_names: bool = False
) -> Dict[GateDecl, str]:
    decls = {}
    _collect_gatedecls(decls, c.instructions, sanitize_names=sanitize_names)
    return decls


def _collect_gatedecls(decls: Dict, instructions, sanitize_names: bool = False):
    for inst in instructions:
        op = inst.operation
        if isinstance(op, GateCall):
            # If already collected, skip to avoid infinite recursion or re-ordering
            if op.decl in decls:
                continue

            # First recurse on dependencies
            _collect_gatedecls(
                decls, op.decl.circuit.instructions, sanitize_names=sanitize_names
            )

            # Then add the current gate declaration
            base = op.decl.name
            if sanitize_names:
                base = _sanitize_identifier(base)
            else:
                _require_valid_identifier(base, "gate declaration name")

            for arg in op.decl.arguments:
                _require_valid_identifier(str(arg), "gate declaration argument")

            unique_name = f"{base}_{hex(id(op.decl))[2:]}"
            decls[op.decl] = unique_name


def gatedecl_to_qasm(
    decl: GateDecl,
    decl_names: Dict[GateDecl, str],
    sanitize_names: bool = False,
    op_cache: Dict = None,
    emit_cb=None,
) -> QASMExpr:
    if not sanitize_names:
        _require_valid_identifier(decl.name, "gate declaration name")
    args = [str(arg) for arg in decl.arguments]
    nq = decl.circuit.num_qubits()
    qubit_names = [f"q_{i}" for i in range(nq)]

    body_statements = []
    for inst in decl.circuit.instructions:
        res = _instruction_to_qasm_gatedecl(
            inst,
            qubit_names,
            decl_names,
            sanitize_names=sanitize_names,
            op_cache=op_cache,
            emit_cb=emit_cb,
        )
        if isinstance(res, list):
            body_statements.extend(res)
        else:
            body_statements.append(res)

    decl_name = decl_names[decl]
    decl_call = QASMExpr("call", [decl_name] + args)
    decl_sig = QASMExpr("unitary", [decl_call] + qubit_names)

    return QASMExpr("gate", [decl_sig, QASMExpr("block", body_statements)])


def _instruction_to_qasm_gatedecl(
    inst,
    qubit_names,
    decl_names,
    sanitize_names: bool = False,
    op_cache: Dict = None,
    emit_cb=None,
):
    op = inst.operation
    qs = inst.qubits
    target_exprs = [qubit_names[q] for q in qs]

    if isinstance(op, GateCall):
        if op.decl not in decl_names:
            base = op.decl.name
            if sanitize_names:
                base = _sanitize_identifier(base)
            else:
                _require_valid_identifier(base, "gate declaration name")
            unique_name = f"{base}_{hex(id(op.decl))[2:]}"
            decl_names[op.decl] = unique_name

        if emit_cb:
            emit_cb(op.decl)

        name = decl_names[op.decl]
        gate_args = list(op.arguments)
        gate_call = QASMExpr("call", [name] + gate_args)
        return QASMExpr("unitary", [gate_call] + target_exprs)

    if isinstance(op, (mc.Inverse, mc.Power)):
        try:
            subcirc = inst.decompose()
        except Exception:
            subcirc = None

        if subcirc is not None and len(subcirc.instructions) > 0:
            stmts = []
            for subinst in subcirc.instructions:
                sub_q = _instruction_to_qasm_gatedecl(
                    subinst,
                    qubit_names,
                    decl_names,
                    sanitize_names=sanitize_names,
                    op_cache=op_cache,
                    emit_cb=emit_cb,
                )
                if isinstance(sub_q, list):
                    stmts.extend(sub_q)
                else:
                    stmts.append(sub_q)
            return stmts

    try:
        name = gate_to_qasm(op)
    except QASMSerializationError as e:
        if op_cache is not None and op in op_cache:
            decl = op_cache[op]
            if decl not in decl_names:
                base = decl.name
                if sanitize_names:
                    base = _sanitize_identifier(base)
                else:
                    _require_valid_identifier(base, "gate declaration name")
                unique_name = f"{base}_{hex(id(decl))[2:]}"
                decl_names[decl] = unique_name

            if emit_cb:
                emit_cb(decl)

            name = decl_names[decl]
            gate_call = QASMExpr("call", [name])
            return QASMExpr("unitary", [gate_call] + target_exprs)

        try:
            decomp_circ = op.decompose()
            is_trivial = (
                len(decomp_circ.instructions) == 1
                and decomp_circ.instructions[0].operation == op
            )

            if (
                decomp_circ is not None
                and len(decomp_circ.instructions) > 0
                and not is_trivial
            ):
                op_name = getattr(op, "name", "gate") or "gate"
                clean_name = _sanitize_identifier(op_name)

                decl = GateDecl(clean_name, (), decomp_circ)

                if op_cache is not None:
                    op_cache[op] = decl

                if decl not in decl_names:
                    base = clean_name
                    if sanitize_names:
                        base = _sanitize_identifier(base)
                    else:
                        _require_valid_identifier(base, "gate declaration name")
                    unique_name = f"{base}_{hex(id(decl))[2:]}"
                    decl_names[decl] = unique_name

                if emit_cb:
                    emit_cb(decl)

                name = decl_names[decl]
                gate_call = QASMExpr("call", [name])
                return QASMExpr("unitary", [gate_call] + target_exprs)
        except Exception:
            pass

        try:
            subcirc = inst.decompose()
        except Exception:
            subcirc = None

        if subcirc is not None and len(subcirc.instructions) > 0:
            stmts = []
            for subinst in subcirc.instructions:
                sub_q = _instruction_to_qasm_gatedecl(
                    subinst,
                    qubit_names,
                    decl_names,
                    sanitize_names=sanitize_names,
                    op_cache=op_cache,
                    emit_cb=emit_cb,
                )
                if isinstance(sub_q, list):
                    stmts.extend(sub_q)
                else:
                    stmts.append(sub_q)
            return stmts
        raise

    gate_args = args_to_qasm(op)
    gate_call = QASMExpr("call", [name] + gate_args)
    return QASMExpr("unitary", [gate_call] + target_exprs)


def circuit_to_qasm(c: mc.Circuit, sanitize_names: bool = True) -> QASMExpr:
    n_qubits = c.num_qubits()
    n_cbits = c.num_bits()

    if c.num_zvars() > 0:
        raise QASMSerializationError(
            "Circuit with Z-register variables cannot be converted to QASM"
        )

    statements = []
    if uses_std_gates(c.instructions):
        statements.append(QASMExpr("include", ["qelib1.inc"]))

    decl_names = collect_gatedecls(c, sanitize_names=sanitize_names)
    op_cache = {}
    gate_statements = []
    emitted = set()
    processing = set()

    def emit_new_decls():
        def emit_recursive(d):
            if d in emitted:
                return
            if d in processing:
                return

            processing.add(d)

            expr = gatedecl_to_qasm(
                d,
                decl_names,
                sanitize_names=sanitize_names,
                op_cache=op_cache,
                emit_cb=emit_recursive,
            )

            gate_statements.append(expr)
            emitted.add(d)
            processing.remove(d)

        while True:
            candidates = [k for k in decl_names if k not in emitted]
            if not candidates:
                break
            emit_recursive(candidates[0])

    emit_new_decls()

    instruction_statements = []
    for inst in c.instructions:
        res = instruction_to_qasm(
            inst, n_qubits, decl_names, sanitize_names=sanitize_names, op_cache=op_cache
        )
        if isinstance(res, list):
            instruction_statements.extend(res)
        else:
            instruction_statements.append(res)

    emit_new_decls()

    statements.extend(gate_statements)
    statements.append(QASMExpr("qreg", ["q", n_qubits]))
    statements.append(QASMExpr("creg", ["c", n_cbits]))
    statements.extend(instruction_statements)

    return QASMExpr("program", [2.0] + statements)


def print_qasm_expr(expr: QASMExpr, io_stream, level=0):
    indent = "    " * level

    if expr.head == "program":
        io_stream.write(f"OPENQASM {expr.args[0]};\n")
        for arg in expr.args[1:]:
            print_qasm_expr(arg, io_stream, level)
            io_stream.write("\n")

    elif expr.head == "include":
        io_stream.write(f'include "{expr.args[0]}";')

    elif expr.head in ("qreg", "creg"):
        io_stream.write(f"{expr.head} {expr.args[0]}[{expr.args[1]}];")

    elif expr.head == "gate":
        sig = expr.args[0]
        body = expr.args[1]

        io_stream.write("gate ")
        call = sig.args[0]
        targets = sig.args[1:]

        io_stream.write(f"{call.args[0]}")
        if len(call.args) > 1:
            io_stream.write("(")
            io_stream.write(",".join(map(str, call.args[1:])))
            io_stream.write(")")

        io_stream.write(" ")
        io_stream.write(
            ",".join(
                map(
                    lambda t: str(t)
                    if not isinstance(t, QASMExpr)
                    else f"{t.args[0]}[{t.args[1]}]"
                    if t.head == "ref"
                    else str(t),
                    targets,
                )
            )
        )

        io_stream.write(" {\n")
        for stmt in body.args:
            print_qasm_expr(stmt, io_stream, level + 1)
            io_stream.write("\n")
        io_stream.write("}")

    elif expr.head == "unitary":
        io_stream.write(indent)
        call = expr.args[0]
        targets = expr.args[1:]

        io_stream.write(f"{call.args[0]}")
        if len(call.args) > 1:
            io_stream.write("(")
            io_stream.write(",".join([_expr_to_str(a) for a in call.args[1:]]))
            io_stream.write(")")

        io_stream.write(" ")
        t_strs = []
        for t in targets:
            if isinstance(t, QASMExpr) and t.head == "ref":
                t_strs.append(f"{t.args[0]}[{t.args[1]}]")
            elif isinstance(t, str):
                t_strs.append(t)
            else:
                t_strs.append(str(t))

        io_stream.write(",".join(t_strs))
        io_stream.write(";")

    elif expr.head == "measure":
        io_stream.write(indent)
        q = expr.args[0]
        c = expr.args[1]
        q_str = f"{q.args[0]}[{q.args[1]}]" if isinstance(q, QASMExpr) else str(q)
        c_str = f"{c.args[0]}[{c.args[1]}]" if isinstance(c, QASMExpr) else str(c)
        io_stream.write(f"measure {q_str} -> {c_str};")

    elif expr.head in ("barrier", "reset"):
        io_stream.write(indent)
        io_stream.write(f"{expr.head} ")
        t_strs = []
        for t in expr.args:
            if isinstance(t, QASMExpr) and t.head == "ref":
                t_strs.append(f"{t.args[0]}[{t.args[1]}]")
            else:
                t_strs.append(str(t))
        io_stream.write(",".join(t_strs))
        io_stream.write(";")

    elif expr.head == "if":
        io_stream.write(indent)
        cond = expr.args[0]
        qop = expr.args[1]
        reg_name = cond.args[1]
        val = cond.args[2]
        io_stream.write(f"if({reg_name}=={val}) ")
        print_qasm_expr(qop, io_stream, 0)


def _expr_to_str(expr):
    if isinstance(expr, QASMExpr) and expr.head == "call":
        op = expr.args[0]
        if op in ("+", "-", "*", "/", "^"):
            lhs = _expr_to_str(expr.args[1])
            rhs = _expr_to_str(expr.args[2]) if len(expr.args) > 2 else ""
            if len(expr.args) == 2:
                return f"{op}{lhs}"
            return f"({lhs}{op}{rhs})"
        else:
            args = ",".join([_expr_to_str(a) for a in expr.args[1:]])
            return f"{op}({args})"
    return str(expr)


def dumps(
    c: mc.Circuit, decompose_wrappers: bool = False, sanitize_names: bool = True
) -> str:
    """Serialize a circuit to an OpenQASM 2.0 string.

    Args:
        c (mc.Circuit): The circuit to serialize.

    Returns:
        str: The OpenQASM 2.0 string representation.
    """
    # Optionally decompose wrapper operations (Inverse/Power/etc.) into
    # primitive gates before converting to QASM. This is useful when the
    # circuit contains high-level wrappers that cannot be directly mapped
    # to OpenQASM names.
    if decompose_wrappers:
        try:
            c = c.decompose()
        except Exception:
            pass

    expr = circuit_to_qasm(c, sanitize_names=sanitize_names)
    s = io.StringIO()
    print_qasm_expr(expr, s, level=0)
    return s.getvalue()


def dump(
    c: mc.Circuit,
    filename: str,
    decompose_wrappers: bool = False,
    sanitize_names: bool = True,
):
    """Serialize a circuit to an OpenQASM 2.0 file.

    Args:
        c (mc.Circuit): The circuit to serialize.
        filename (str): The file path to write to.
    """
    with open(filename, "w") as f:
        f.write(
            dumps(
                c,
                decompose_wrappers=decompose_wrappers,
                sanitize_names=sanitize_names,
            )
        )


# Deprecated/Legacy aliases
qasmstring = dumps
qasmsave = dump
