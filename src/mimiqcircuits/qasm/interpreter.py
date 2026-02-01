# Copyright (C) 2023 QPerfect. All Rights Reserved.
# Proprietary and confidential.
#
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Union

from symengine import Symbol

import mimiqcircuits as mc
from mimiqcircuits.gatedecl import GateDecl
from mimiqcircuits.qasm.exceptions import QASMError
from mimiqcircuits.qasm.parser import QASMExpr, parseopenqasm


class QASMStructureError(QASMError):
    pass

from mimiqcircuits.qasm.exceptions import (
    QASMError,
    QASMStructureError,
    UndefinedGateError,
    UndefinedRegisterError,
    DuplicateRegisterError,
    QASMArgumentError,
    QASMVersionError,
)

@dataclass
class InterpreterState:
    regs: Dict[str, range] = field(default_factory=dict)
    lastqreg: int = 0
    lastcreg: int = 0
    gates: Dict[str, Any] = field(default_factory=dict)  # GateDecl
    pending_gates: Dict[str, Any] = field(default_factory=dict)
    compiling_gates: Set[str] = field(default_factory=set)
    vars_map: Dict[str, Any] = field(default_factory=dict)
    circuit: mc.Circuit = field(default_factory=mc.Circuit)


OPAQUE_TO_TYPE = {
    "U": mc.GateU,
    "CX": mc.GateCX,
    "u3": mc.GateU3,
    "u2": mc.GateU2,
    "u1": mc.GateU1,
    "u": mc.GateU,
    "cx": mc.GateCX,
    "id": mc.GateID,
    "p": mc.GateP,
    "x": mc.GateX,
    "y": mc.GateY,
    "z": mc.GateZ,
    "h": mc.GateH,
    "s": mc.GateS,
    "sdg": mc.GateSDG,
    "t": mc.GateT,
    "tdg": mc.GateTDG,
    "rx": mc.GateRX,
    "ry": mc.GateRY,
    "rz": mc.GateRZ,
    "sx": mc.GateSX,
    "sxdg": mc.GateSXDG,
    "cz": mc.GateCZ,
    "cy": mc.GateCY,
    "swap": mc.GateSWAP,
    "ch": mc.GateCH,
    "ccx": mc.GateCCX,
    "cswap": mc.GateCSWAP,
    "crx": mc.GateCRX,
    "cry": mc.GateCRY,
    "crz": mc.GateCRZ,
    # "cu1": mc.Control(1, mc.GateU1) - Needs special handling as factory
    "cp": mc.GateCP,
    # "cu3": mc.Control(1, mc.GateU3)
    "cs": mc.GateCS,
    "csdg": mc.GateCSDG,
    "csx": mc.GateCSX,
    "csxdg": mc.GateCSXDG,
    "cu": mc.GateCU,
    "rxx": mc.GateRXX,
    "rzz": mc.GateRZZ,
    "ryy": mc.GateRYY,
    "rzx": mc.GateRZX,
    "c3x": mc.GateC3X,
    # "c4x": Control(4, GateX)
    # "c3sqrtx": Control(3, GateSX)
    "xxminusyy": mc.GateXXminusYY,
    "xxplusyy": mc.GateXXplusYY,
    "dcx": mc.GateDCX,
    "ecr": mc.GateECR,
    "iswap": mc.GateISWAP,
    "uphase": mc.GateU,  # Alias?
    "ccp": mc.GateCCP,
}


# Special gate factories for composed gates
def create_cu1(*args):
    return mc.Control(1, mc.GateU1(*args))


def create_cu3(*args):
    return mc.Control(1, mc.GateU3(*args))


def create_c4x(*args):
    return mc.Control(4, mc.GateX())


def create_c3sqrtx(*args):
    return mc.Control(3, mc.GateSX())


SPECIAL_FACTORIES = {
    "cu1": create_cu1,
    "cu3": create_cu3,
    "c4x": create_c4x,
    "c3sqrtx": create_c3sqrtx,
}

OPERATIONS = {
    "+": lambda x, y: x + y,
    "-": lambda x, y: x - y,  # Unary handled differently? Julia uses same dict.
    "*": lambda x, y: x * y,
    "/": lambda x, y: x / y,
    "^": lambda x, y: x**y,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
}


def interpret_expression(istate: InterpreterState, expr: Any):
    if isinstance(expr, (int, float)):
        return expr
    if isinstance(expr, str):
        if expr == "pi":
            return math.pi
        if expr in istate.vars_map:
            return istate.vars_map[expr]
        return Symbol(expr)

    if isinstance(expr, QASMExpr):
        if expr.head == "call":
            op_name = expr.args[0]
            args = [interpret_expression(istate, arg) for arg in expr.args[1:]]
            if op_name in OPERATIONS:
                # Handle unary minus if 1 arg
                if op_name == "-" and len(args) == 1:
                    return -args[0]
                return OPERATIONS[op_name](*args)
            else:
                raise QASMStructureError(f"Unknown operation {op_name}")

    return expr


def interpret_regs(istate: InterpreterState, expr: QASMExpr):
    name = expr.args[0]
    size = expr.args[1]

    if name in istate.regs:
        raise DuplicateRegisterError(f"Register {name} already defined")

    if expr.head == "qreg":
        istate.regs[name] = range(istate.lastqreg, istate.lastqreg + size)
        istate.lastqreg += size
    elif expr.head == "creg":
        istate.regs[name] = range(istate.lastcreg, istate.lastcreg + size)
        istate.lastcreg += size


def _is_ref_expr(expr: Any) -> bool:
    return (
        hasattr(expr, "head")
        and getattr(expr, "head") == "ref"
        and hasattr(expr, "args")
        and len(getattr(expr, "args")) >= 2
    )


def interpret_target(istate: InterpreterState, expr: Any) -> int:
    if isinstance(expr, str):  # Whole register reference?
        pass

    if _is_ref_expr(expr):
        name = expr.args[0]
        idx = expr.args[1]
        if name not in istate.regs:
            raise UndefinedRegisterError(f"Undefined register {name}")
        reg = istate.regs[name]
        return reg[idx]  # 0-indexed in Python range

    raise QASMStructureError(f"Expected register reference, got {expr}")


def interpret_targets(istate: InterpreterState, expr: Any) -> Union[int, List[int]]:
    if isinstance(expr, str):
        if expr in istate.regs:
            return list(istate.regs[expr])
        if expr in istate.vars_map:
            return istate.vars_map[expr]  # variable?
        raise UndefinedRegisterError(f"Undefined register {expr}")

    if _is_ref_expr(expr):
        return interpret_target(istate, expr)

    raise QASMStructureError(f"Invalid target {expr}")


def interpret_gatecall(istate: InterpreterState, expr: QASMExpr):
    name = expr.args[0]
    args = [interpret_expression(istate, arg) for arg in expr.args[1:]]

    if name not in istate.gates:
        compile_pending_gate(istate, name)

    if name in istate.gates:
        return istate.gates[name](*args)
    elif name in SPECIAL_FACTORIES:
        return SPECIAL_FACTORIES[name](*args)
    elif name in OPAQUE_TO_TYPE:
        return OPAQUE_TO_TYPE[name](*args)
    else:
        raise UndefinedGateError(f"Undefined gate {name}")


def interpret_unitary(istate: InterpreterState, expr: QASMExpr):
    gate_op = interpret_gatecall(istate, expr.args[0])

    targets = []
    for arg in expr.args[1:]:
        t = interpret_targets(istate, arg)
        if isinstance(t, list):
            targets.extend(t)
        else:
            targets.append(t)

    istate.circuit.push(gate_op, *targets)


def interpret_measure(istate: InterpreterState, expr: QASMExpr):
    qtarget = interpret_targets(istate, expr.args[0])
    ctarget = interpret_targets(istate, expr.args[1])

    istate.circuit.push(mc.Measure(), qtarget, ctarget)


def interpret_reset(istate: InterpreterState, expr: QASMExpr):
    target = interpret_targets(istate, expr.args[0])
    istate.circuit.push(mc.Reset(), target)


def interpret_barrier(istate: InterpreterState, expr: QASMExpr):
    targets = []
    for arg in expr.args:
        t = interpret_targets(istate, arg)
        if isinstance(t, list):
            targets.extend(t)
        else:
            targets.append(t)

    istate.circuit.push(mc.Barrier(len(targets)), *targets)


def interpret_if(istate: InterpreterState, expr: QASMExpr):
    # expr.args[0] is condition: call("==", creg_ref, val)
    # expr.args[1] is qop (unitary)

    condition = expr.args[0]
    op_name = condition.args[0]  # "=="
    if op_name != "==":
        raise QASMStructureError("Only equality checks supported in if")

    creg = interpret_targets(istate, condition.args[1])
    # creg should be list of bits for the register
    if isinstance(creg, int):
        creg = [creg]

    val = condition.args[2]

    # We need to build a sub-circuit for the body

    sub_istate = InterpreterState(
        regs=istate.regs,
        lastqreg=istate.lastqreg,
        lastcreg=istate.lastcreg,
        gates=istate.gates,
        vars_map=istate.vars_map,
        circuit=mc.Circuit(),
    )

    interpret_qop(sub_istate, expr.args[1])

    if len(sub_istate.circuit.instructions) != 1:
        raise QASMStructureError(
            "If statement body must result in exactly one instruction"
        )

    inst = sub_istate.circuit.instructions[0]

    from mimiqcircuits import BitString
    from mimiqcircuits import IfStatement

    # BitString signature: length, val
    bs = BitString(len(creg), val)

    if_op = IfStatement(inst.operation, bs)

    istate.circuit.push(if_op, *inst.qubits, *creg)


def interpret_gatedecl(istate: InterpreterState, expr: QASMExpr):
    # args[0]: gate signature (unitary -> call -> name, args...; targets...)
    # args[1]: body (block -> goplist)

    unitary = expr.args[0]
    gatecall = unitary.args[0]
    decl_name = gatecall.args[0]
    decl_params = [token_val_to_str(a) for a in gatecall.args[1:]]

    decl_targets = [token_val_to_str(t) for t in unitary.args[1:]]

    # Create new context
    new_vars = {p: Symbol(p) for p in decl_params}

    # Map targets to unique indices
    target_mapping = {name: range(i, i + 1) for i, name in enumerate(decl_targets)}

    sub_istate = InterpreterState(
        regs=target_mapping,
        lastqreg=len(decl_targets),  # Just counting
        lastcreg=0,
        gates=istate.gates,
        pending_gates=istate.pending_gates,
        compiling_gates=istate.compiling_gates,
        vars_map=new_vars,
        circuit=mc.Circuit(),
    )

    block = expr.args[1]
    for stmt in block.args:
        interpret_gop(sub_istate, stmt)

    istate.gates[decl_name] = GateDecl(
        decl_name, tuple(Symbol(p) for p in decl_params), sub_istate.circuit
    )


def compile_pending_gate(istate: InterpreterState, name: str):
    if name in istate.gates:
        return
    if name in istate.compiling_gates:
        raise UndefinedGateError(f"Recursive gate definition detected for {name}")
    if name not in istate.pending_gates:
        return

    istate.compiling_gates.add(name)
    expr = istate.pending_gates.pop(name)
    try:
        interpret_gatedecl(istate, expr)
    finally:
        istate.compiling_gates.remove(name)


def token_val_to_str(val):
    if isinstance(val, QASMExpr):
        return str(val)  # Should be string if parser returns strings for IDs
    return str(val)


def interpret_gop(istate: InterpreterState, expr: QASMExpr):
    if expr.head == "unitary":
        interpret_unitary(istate, expr)
    elif expr.head == "barrier":
        interpret_barrier(istate, expr)
    elif expr.head == "measure":
        interpret_measure(istate, expr)
    elif expr.head == "reset":
        interpret_reset(istate, expr)
    else:
        raise QASMStructureError(f"Invalid statement in gate body: {expr.head}")


def interpret_qop(istate: InterpreterState, expr: QASMExpr):
    if expr.head == "unitary":
        interpret_unitary(istate, expr)
    elif expr.head == "barrier":
        interpret_barrier(istate, expr)
    elif expr.head == "reset":
        interpret_reset(istate, expr)
    elif expr.head == "measure":
        interpret_measure(istate, expr)
    else:
        raise QASMStructureError(f"Unrecognized qop: {expr.head}")


def interpret_opaque(istate: InterpreterState, expr: QASMExpr):
    # Opaque declaration. We verify it matches standard or fail?
    # Julia verifies it's in OPAQUE_TO_TYPE.
    unitary = expr.args[0]
    gatecall = unitary.args[0]
    name = gatecall.args[0]

    if name not in OPAQUE_TO_TYPE and name not in SPECIAL_FACTORIES:
        raise QASMArgumentError(f"Unsupported opaque gate {name}")


def interpret(expr: QASMExpr) -> mc.Circuit:
    istate = InterpreterState()

    if expr.head != "program":
        raise QASMArgumentError("Not a valid OpenQASM program")

    version = expr.args[0]
    if version != 2.0:
        raise QASMVersionError(f"Unsupported version {version}")

    for stmt in expr.args[1:]:
        if stmt.head == "gate":
            unitary = stmt.args[0]
            gatecall = unitary.args[0]
            decl_name = gatecall.args[0]
            istate.pending_gates[decl_name] = stmt

    for stmt in expr.args[1:]:
        if stmt.head in {"qreg", "creg"}:
            interpret_regs(istate, stmt)
        elif stmt.head == "gate":
            compile_pending_gate(istate, stmt.args[0].args[0].args[0])
        elif stmt.head == "opaque":
            interpret_opaque(istate, stmt)
        elif stmt.head == "if":
            interpret_if(istate, stmt)
        else:
            interpret_qop(istate, stmt)

    return istate.circuit


def loads(s: str, includedirs: List[str] = None) -> mc.Circuit:
    """Parse and interpret an OpenQASM 2.0 string.

    Args:
        s (str): The OpenQASM 2.0 string.
        includedirs (List[str], optional): List of directories to search for included files.

    Returns:
        mc.Circuit: The interpreted circuit.
    """
    expr = parseopenqasm(s, includedirs)
    return interpret(expr)


def load(filename: str, includedirs: List[str] = None) -> mc.Circuit:
    """Parse and interpret an OpenQASM 2.0 file.

    Args:
        filename (str): Path to the OpenQASM 2.0 file.
        includedirs (List[str], optional): List of directories to search for included files.

    Returns:
        mc.Circuit: The interpreted circuit.
    """
    import os

    if includedirs is None:
        includedirs = []

    # Add file's directory to includedirs
    file_dir = os.path.dirname(os.path.abspath(filename))
    if file_dir not in includedirs:
        includedirs = [file_dir] + includedirs

    with open(filename, "r") as f:
        content = f.read()

    return loads(content, includedirs=includedirs)
