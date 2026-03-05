# Copyright (C) 2023 QPerfect. All Rights Reserved.
# Proprietary and confidential.
#
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Union

from mimiqcircuits.qasm.exceptions import ParseError
from mimiqcircuits.qasm.lexer import Lexer
from mimiqcircuits.qasm.tokens import EMPTY_TOKEN, Token, TokenKind

# Directory containing standard includes
QASMINCLUDEDIR = os.path.join(os.path.dirname(__file__), "qasmincs")


@dataclass
class QASMExpr:
    head: str
    args: List[Any] = field(default_factory=list)

    def __repr__(self):
        return f"QASMExpr({self.head}, {self.args})"


class ParserState:
    def __init__(self, lexer: Lexer, includedirs: List[str] = None):
        self.l = lexer
        self.isdone = False
        self.t = EMPTY_TOKEN
        self.nt = EMPTY_TOKEN
        self.nnt = EMPTY_TOKEN
        self.errored = False
        self.includedirs = [os.getcwd(), QASMINCLUDEDIR]
        if includedirs:
            self.includedirs.extend(includedirs)

        # Initialize lookahead
        self.readtoken()
        self.readtoken()

    def readtoken(self) -> Token:
        if self.isdone:
            return self.t

        try:
            nnt = self.l.next_token(start=True)
            # Skip comments and whitespace
            while nnt.kind == TokenKind.COMMENT or nnt.kind == TokenKind.WHITESPACE:
                nnt = self.l.next_token(start=True)
        except StopIteration:
            nnt = Token(TokenKind.ENDMARKER)

        self.t = self.nt
        self.nt = self.nnt
        self.nnt = nnt

        if self.t.kind == TokenKind.ENDMARKER:
            self.isdone = True

        return self.t

    def token(self):
        return self.t

    def peektoken(self):
        return self.nt

    def dpeektoken(self):
        return self.nt, self.nnt

    def expect(self, k: TokenKind) -> Token:
        t = self.token()
        if t.kind != k:
            self.errored = True
            raise ParseError(f"Found {t}, expected {k}", t, self)
        return self.t

    def expectnext(self, f: Union[TokenKind, Set[TokenKind]]) -> Token:
        t = self.peektoken()
        ok = False
        if isinstance(f, TokenKind):
            ok = t.kind == f
        else:  # Set/List
            ok = t.kind in f

        if not ok:
            self.errored = True
            expected = f.name if isinstance(f, TokenKind) else str(f)
            raise ParseError(f"Unexpected {t.kind.name}. Expected {expected}", t, self)

        return self.readtoken()

    def accept(self, f: Union[TokenKind, Set[TokenKind]]) -> bool:
        t = self.peektoken()
        ok = False
        if isinstance(f, TokenKind):
            ok = t.kind == f
        else:
            ok = t.kind in f

        if ok:
            self.readtoken()
        return ok


def parse_id(ps: ParserState) -> str:
    t = ps.expectnext(TokenKind.IDENTIFIER)
    return t.val


def parse_integer_value(ps: ParserState) -> int:
    t = ps.expectnext(TokenKind.INTEGER)
    return int(t.val)


def parse_real_value(ps: ParserState) -> float:
    t = ps.expectnext(TokenKind.REAL)
    return float(t.val)


def parse_unsigned_value(ps: ParserState) -> int:
    t = ps.expectnext(TokenKind.INTEGER)
    return int(t.val)


UNARYOP_TO_STR = {
    TokenKind.SIN: "sin",
    TokenKind.COS: "cos",
    TokenKind.TAN: "tan",
    TokenKind.EXP: "exp",
    TokenKind.LN: "log",
    TokenKind.SQRT: "sqrt",
}


def parse_unaryop(ps: ParserState) -> QASMExpr:
    # expectnext checks kind. Using Set for unary ops
    t = ps.expectnext(
        {
            TokenKind.SIN,
            TokenKind.COS,
            TokenKind.TAN,
            TokenKind.EXP,
            TokenKind.LN,
            TokenKind.SQRT,
        }
    )

    op_str = UNARYOP_TO_STR[t.kind]

    ps.expectnext(TokenKind.LPAREN)
    expr = parse_expr(ps)
    ps.expectnext(TokenKind.RPAREN)

    return QASMExpr("call", [op_str, expr])


def parse_expr_atom(ps: ParserState):
    nk = ps.peektoken().kind

    if nk == TokenKind.REAL:
        return parse_real_value(ps)
    elif nk == TokenKind.INTEGER:
        return parse_integer_value(ps)
    elif nk == TokenKind.IDENTIFIER:
        return parse_id(ps)
    elif nk in UNARYOP_TO_STR:  # UnaryOp
        return parse_unaryop(ps)
    elif nk == TokenKind.LPAREN:
        ps.readtoken()  # consume (
        expr = parse_expr(ps)
        ps.expectnext(TokenKind.RPAREN)
        return expr
    else:
        raise ParseError(f"Unexpected token {nk} in expression", ps.peektoken(), ps)


def parse_expr_power(ps: ParserState):
    first = parse_expr_atom(ps)
    if ps.accept(TokenKind.CIRCUMFLEX_ACCENT):
        second = parse_expr_atom(ps)
        return QASMExpr("call", ["^", first, second])
    return first


def parse_expr_factor(ps: ParserState):
    if ps.accept(TokenKind.MINUS):
        return QASMExpr("call", ["-", parse_expr_factor(ps)])
    elif ps.accept(TokenKind.PLUS):
        return parse_expr_factor(ps)

    return parse_expr_power(ps)


def parse_expr_product(ps: ParserState):
    lhs = parse_expr_factor(ps)

    if ps.accept(TokenKind.STAR):
        rhs = parse_expr_product(ps)
        return QASMExpr("call", ["*", lhs, rhs])
    elif ps.accept(TokenKind.FWD_SLASH):
        rhs = parse_expr_product(ps)
        return QASMExpr("call", ["/", lhs, rhs])

    return lhs


def parse_expr_sum(ps: ParserState):
    lhs = parse_expr_product(ps)  # Precedence

    if ps.accept(TokenKind.PLUS):
        rhs = parse_expr_sum(ps)
        return QASMExpr("call", ["+", lhs, rhs])
    elif ps.accept(TokenKind.MINUS):
        rhs = parse_expr_sum(ps)
        return QASMExpr("call", ["-", lhs, rhs])

    return lhs


def parse_expr(ps: ParserState):
    return parse_expr_sum(ps)


def parse_arg(ps: ParserState) -> Union[str, QASMExpr]:
    id_str = parse_id(ps)
    if ps.accept(TokenKind.LSQUARE):
        idx = parse_integer_value(ps)
        ps.expectnext(TokenKind.RSQUARE)
        return QASMExpr("ref", [id_str, idx])
    return id_str


def parse_arglist(ps: ParserState) -> List[Any]:
    args = []
    while True:
        args.append(parse_arg(ps))
        if not ps.accept(TokenKind.COMMA):
            break
    return args


def parse_idlist(ps: ParserState) -> List[str]:
    args = []
    while True:
        args.append(parse_id(ps))
        if not ps.accept(TokenKind.COMMA):
            break
    return args


def parse_idlist_paren(ps: ParserState) -> List[str]:
    args = []
    if ps.accept(TokenKind.LPAREN):
        if not ps.accept(TokenKind.RPAREN):
            args = parse_idlist(ps)
            ps.expectnext(TokenKind.RPAREN)
    return args


def parse_exprlist(ps: ParserState) -> List[Any]:
    exprs = []
    while True:
        exprs.append(parse_expr(ps))
        if not ps.accept(TokenKind.COMMA):
            break
    return exprs


def parse_exprlist_paren(ps: ParserState) -> List[Any]:
    exprs = []
    if ps.accept(TokenKind.LPAREN):
        if not ps.accept(TokenKind.RPAREN):
            exprs = parse_exprlist(ps)
            ps.expectnext(TokenKind.RPAREN)
    return exprs


def parse_gatecall(ps: ParserState):
    # expectnext(ps, [U, CX, IDENTIFIER])
    t = ps.expectnext({TokenKind.U, TokenKind.CX, TokenKind.IDENTIFIER})
    if t.kind == TokenKind.IDENTIFIER:
        name = t.val
    else:
        name = t.kind.name  # U or CX

    # Check if we have parameters
    params = []
    if ps.peektoken().kind == TokenKind.LPAREN:
        params = parse_exprlist_paren(ps)

    return QASMExpr("call", [name] + params)


def parse_uop(ps: ParserState):
    # Gate call (name + params)
    gcall = parse_gatecall(ps)
    # Targets
    targets = parse_arglist(ps)
    ps.expectnext(TokenKind.SEMICOLON)

    return QASMExpr("unitary", [gcall] + targets)


def parse_measure(ps: ParserState):
    ps.expectnext(TokenKind.MEASURE)
    qtarget = parse_arg(ps)
    ps.expectnext(TokenKind.ARROW)
    ctarget = parse_arg(ps)
    ps.expectnext(TokenKind.SEMICOLON)
    return QASMExpr("measure", [qtarget, ctarget])


def parse_reset(ps: ParserState):
    ps.expectnext(TokenKind.RESET)
    args = parse_arglist(ps)
    ps.expectnext(TokenKind.SEMICOLON)
    return QASMExpr("reset", args)


def parse_barrier(ps: ParserState):
    ps.expectnext(TokenKind.BARRIER)
    args = parse_arglist(ps)
    ps.expectnext(TokenKind.SEMICOLON)
    return QASMExpr("barrier", args)


def parse_if(ps: ParserState):
    ps.expectnext(TokenKind.IF)
    ps.expectnext(TokenKind.LPAREN)
    id_str = parse_id(ps)
    ps.expectnext(TokenKind.EQEQ)
    val = parse_unsigned_value(ps)
    ps.expectnext(TokenKind.RPAREN)

    qop = parse_qop(ps)

    condition = QASMExpr("call", ["==", id_str, val])
    return QASMExpr("if", [condition, qop])


def parse_qop(ps: ParserState):
    nk = ps.peektoken().kind
    if nk in {TokenKind.U, TokenKind.CX, TokenKind.IDENTIFIER}:
        return parse_uop(ps)
    elif nk == TokenKind.BARRIER:
        return parse_barrier(ps)
    elif nk == TokenKind.MEASURE:
        return parse_measure(ps)
    elif nk == TokenKind.RESET:
        return parse_reset(ps)
    else:
        raise ParseError("Unexpected token in qop", ps.peektoken(), ps)


def parse_goplist(ps: ParserState) -> List[QASMExpr]:
    gops = []
    while True:
        nk = ps.peektoken().kind
        if nk in {TokenKind.IDENTIFIER, TokenKind.U, TokenKind.CX}:
            gops.append(parse_uop(ps))
        elif nk == TokenKind.BARRIER:
            gops.append(parse_barrier(ps))
        elif nk == TokenKind.MEASURE:
            gops.append(parse_measure(ps))
        elif nk == TokenKind.RESET:
            gops.append(parse_reset(ps))
        else:
            break
    return gops


def parse_gatedecl(ps: ParserState):
    ps.expectnext(TokenKind.GATE)

    # Decl name and args
    t = ps.expectnext({TokenKind.U, TokenKind.CX, TokenKind.IDENTIFIER})
    if t.kind == TokenKind.IDENTIFIER:
        decl_name = t.val
    else:
        decl_name = t.kind.name

    args = parse_idlist_paren(ps)
    targets = parse_idlist(ps)

    decl_sig = QASMExpr("unitary", [QASMExpr("call", [decl_name] + args)] + targets)

    ps.expectnext(TokenKind.LBRACE)
    body = QASMExpr("block", parse_goplist(ps))
    ps.expectnext(TokenKind.RBRACE)

    return QASMExpr("gate", [decl_sig, body])


def parse_opaquedecl(ps: ParserState):
    ps.expectnext(TokenKind.OPAQUE)

    t = ps.expectnext({TokenKind.U, TokenKind.CX, TokenKind.IDENTIFIER})
    if t.kind == TokenKind.IDENTIFIER:
        decl_name = t.val
    else:
        decl_name = t.kind.name

    args = parse_idlist_paren(ps)
    targets = parse_idlist(ps)

    decl_sig = QASMExpr("unitary", [QASMExpr("call", [decl_name] + args)] + targets)

    ps.expectnext(TokenKind.SEMICOLON)
    return QASMExpr("opaque", [decl_sig])


def parse_decl(ps: ParserState) -> QASMExpr:
    t = ps.expectnext({TokenKind.QREG, TokenKind.CREG})
    type_kind = t.kind.name.lower()

    id_str = parse_id(ps)
    ps.expectnext(TokenKind.LSQUARE)
    size = parse_integer_value(ps)
    ps.expectnext(TokenKind.RSQUARE)
    ps.expectnext(TokenKind.SEMICOLON)

    return QASMExpr(type_kind, [id_str, size])


def parse_statement(ps: ParserState):
    nk = ps.peektoken().kind
    if nk in {TokenKind.QREG, TokenKind.CREG}:
        return parse_decl(ps)
    elif nk == TokenKind.GATE:
        return parse_gatedecl(ps)
    elif nk == TokenKind.OPAQUE:
        return parse_opaquedecl(ps)
    elif nk == TokenKind.IF:
        return parse_if(ps)
    elif nk == TokenKind.BARRIER:
        return parse_barrier(ps)
    elif nk in {
        TokenKind.MEASURE,
        TokenKind.RESET,
        TokenKind.U,
        TokenKind.CX,
        TokenKind.IDENTIFIER,
    }:
        return parse_qop(ps)
    else:
        raise ParseError("Unexpected statement", ps.peektoken(), ps)


PARSED_INCLUDES: Dict[str, List[QASMExpr]] = {}


def parse_include(ps: ParserState):
    ps.expectnext(TokenKind.INCLUDE)
    t = ps.expectnext(TokenKind.STRING)
    ps.expectnext(TokenKind.SEMICOLON)
    fname = t.val

    # Check cache
    if fname in PARSED_INCLUDES:
        return PARSED_INCLUDES[fname]

    # Resolve file
    found_path = None
    for d in ps.includedirs:
        p = os.path.join(d, fname)
        if os.path.isfile(p):
            found_path = p
            break

    if not found_path:
        raise ParseError(f"Include file {fname} not found", t, ps)

    # Parse include
    with open(found_path, "r") as f:
        content = f.read()

    sub_lexer = Lexer(content)
    sub_ps = ParserState(sub_lexer, ps.includedirs)

    stmts = []
    while sub_ps.peektoken().kind != TokenKind.ENDMARKER:
        stmts.append(parse_statement(sub_ps))

    # Cache
    PARSED_INCLUDES[fname] = stmts
    return stmts


def parse_version(ps: ParserState):
    ps.expectnext(TokenKind.OPENQASM)
    version = parse_real_value(ps)
    ps.expectnext(TokenKind.SEMICOLON)

    if version != 2.0:
        raise ParseError(f"OpenQASM version {version} not supported", ps.t, ps)

    return version


def parse_program(ps: ParserState):
    # Check for empty or comment only start?

    # Peek to see if OpenQASM header exists
    if ps.peektoken().kind == TokenKind.OPENQASM:
        version = parse_version(ps)
    else:
        version = 2.0  # Default if missing

    statements = []
    while ps.peektoken().kind != TokenKind.ENDMARKER:
        if ps.peektoken().kind == TokenKind.INCLUDE:
            statements.extend(parse_include(ps))
        else:
            statements.append(parse_statement(ps))

    return QASMExpr("program", [version] + statements)


def parseopenqasm(source: str, includedirs: List[str] = None):
    lexer = Lexer(source)
    ps = ParserState(lexer, includedirs)
    return parse_program(ps)
