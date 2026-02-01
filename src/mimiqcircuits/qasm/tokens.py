# Copyright (C) 2023-2025 QPerfect. All Rights Reserved.
# Proprietary and confidential.
#
from enum import Enum, auto
from dataclasses import dataclass
from typing import Tuple, Dict, Optional


class TokenKind(Enum):
    ENDMARKER = auto()  # EOF
    ERROR = auto()
    COMMENT = auto()
    WHITESPACE = auto()
    IDENTIFIER = auto()
    COMMA = auto()  # ,
    SEMICOLON = auto()  # ;
    ARROW = auto()  # ->

    # Keywords
    OPENQASM = auto()
    OPAQUE = auto()
    QREG = auto()
    CREG = auto()
    GATE = auto()
    IF = auto()
    BARRIER = auto()
    MEASURE = auto()
    RESET = auto()
    U = auto()
    CX = auto()
    INCLUDE = auto()
    KEYWORD = auto()  # general

    # Literals
    LITERAL = auto()  # general
    INTEGER = auto()  # 4
    STRING = auto()  # "foo"
    REAL = auto()  # 3.5

    # Delimiters
    LSQUARE = auto()  # [
    RSQUARE = auto()  # ]
    LBRACE = auto()  # {
    RBRACE = auto()  # }
    LPAREN = auto()  # (
    RPAREN = auto()  # )

    # Operators
    OP = auto()
    EQEQ = auto()  # ==
    PLUS = auto()  # +
    MINUS = auto()  # -
    STAR = auto()  # *
    FWD_SLASH = auto()  # /
    CIRCUMFLEX_ACCENT = auto()  # ^

    # Unary Ops
    UNARYOP = auto()  # general
    SIN = auto()
    COS = auto()
    TAN = auto()
    EXP = auto()
    LN = auto()
    SQRT = auto()


def is_keyword(k: TokenKind) -> bool:
    return k in {
        TokenKind.OPENQASM,
        TokenKind.OPAQUE,
        TokenKind.QREG,
        TokenKind.CREG,
        TokenKind.GATE,
        TokenKind.IF,
        TokenKind.BARRIER,
        TokenKind.MEASURE,
        TokenKind.RESET,
        TokenKind.U,
        TokenKind.CX,
        TokenKind.INCLUDE,
        TokenKind.KEYWORD,
    }


def is_literal(k: TokenKind) -> bool:
    return k in {TokenKind.INTEGER, TokenKind.STRING, TokenKind.REAL}


def is_operator(k: TokenKind) -> bool:
    return k in {
        TokenKind.OP,
        TokenKind.EQEQ,
        TokenKind.PLUS,
        TokenKind.MINUS,
        TokenKind.STAR,
        TokenKind.FWD_SLASH,
        TokenKind.CIRCUMFLEX_ACCENT,
    }


def is_unaryop(k: TokenKind) -> bool:
    return k in {
        TokenKind.UNARYOP,
        TokenKind.SIN,
        TokenKind.COS,
        TokenKind.TAN,
        TokenKind.EXP,
        TokenKind.LN,
        TokenKind.SQRT,
    }


def keyword_string(k: TokenKind) -> str:
    if k in {TokenKind.U, TokenKind.CX, TokenKind.OPENQASM}:
        return k.name
    return k.name.lower()


KEYWORDS: Dict[str, TokenKind] = {
    keyword_string(k): k for k in TokenKind if is_keyword(k)
}


@dataclass
class Token:
    kind: TokenKind = TokenKind.ERROR
    startpos: Tuple[int, int] = (0, 0)  # (line, column)
    endpos: Tuple[int, int] = (0, 0)  # (line, column)
    startbyte: int = 0
    endbyte: int = 0
    val: str = ""
    token_error: Optional[str] = None

    def exactkind(self) -> TokenKind:
        return self.kind

    def get_kind(self) -> TokenKind:
        if is_unaryop(self.kind):
            return TokenKind.UNARYOP
        if is_operator(self.kind):
            return TokenKind.OP
        if is_keyword(self.kind):
            return TokenKind.KEYWORD
        return self.kind

    def untokenize(self) -> str:
        if (
            self.kind == TokenKind.IDENTIFIER
            or is_literal(self.kind)
            or self.kind == TokenKind.COMMENT
            or self.kind == TokenKind.WHITESPACE
            or self.kind == TokenKind.ERROR
        ):
            return self.val
        elif is_keyword(self.kind) or is_unaryop(self.kind):
            return keyword_string(self.kind)
        elif self.kind == TokenKind.EQEQ:
            return "=="
        elif self.kind == TokenKind.PLUS:
            return "+"
        elif self.kind == TokenKind.MINUS:
            return "-"
        elif self.kind == TokenKind.STAR:
            return "*"
        elif self.kind == TokenKind.FWD_SLASH:
            return "/"
        elif self.kind == TokenKind.CIRCUMFLEX_ACCENT:
            return "^"
        elif self.kind == TokenKind.LPAREN:
            return "("
        elif self.kind == TokenKind.LSQUARE:
            return "["
        elif self.kind == TokenKind.LBRACE:
            return "{"
        elif self.kind == TokenKind.RPAREN:
            return ")"
        elif self.kind == TokenKind.RSQUARE:
            return "]"
        elif self.kind == TokenKind.RBRACE:
            return "}"
        elif self.kind == TokenKind.COMMA:
            return ","
        elif self.kind == TokenKind.SEMICOLON:
            return ";"
        elif self.kind == TokenKind.ARROW:
            return "->"
        return ""

    def __str__(self):
        return self.untokenize()

    def __repr__(self):
        start_r, start_c = self.startpos
        end_r, end_c = self.endpos
        content = self.val if self.kind != TokenKind.ENDMARKER else ""
        return f'{start_r},{start_c}-{end_r},{end_c} {self.kind.name} "{content}"'


EMPTY_TOKEN = Token()
