# Copyright (C) 2023-2025 QPerfect. All Rights Reserved.
# Proprietary and confidential.
#
import io
from typing import Callable, Iterator, Optional, Union

from mimiqcircuits.qasm.tokens import KEYWORDS, Token, TokenKind

UNICODE_OPS = {
    "+": TokenKind.PLUS,
    "−": TokenKind.MINUS,
    "*": TokenKind.STAR,
    "/": TokenKind.FWD_SLASH,
    "^": TokenKind.CIRCUMFLEX_ACCENT,
}


class Lexer:
    def __init__(self, input_source: Union[str, io.TextIOBase]):
        if isinstance(input_source, str):
            self.io = io.StringIO(input_source)
        else:
            self.io = input_source

        self.io_startpos = self.io.tell()

        # Current token start position
        self.token_start_row = 1
        self.token_start_col = 1
        self.token_startpos = self.io_startpos

        # Current scanner position
        self.current_row = 1
        self.current_col = 1
        self.current_pos = self.io_startpos

        self.last_token_kind = TokenKind.ERROR
        self.charstore = io.StringIO()

        # Lookahead buffer (c1, c2, c3)
        self.chars = ["", "", ""]

        # Initialize buffer
        c1 = self.io.read(1)
        c2 = self.io.read(1)
        c3 = self.io.read(1)

        self.chars = [c1, c2, c3]

        # Reset position to start for logic consistency

        self.doread = False

    def __iter__(self) -> Iterator[Token]:
        self.seekstart()
        return self

    def __next__(self) -> Token:
        t = self.next_token()
        if t.kind == TokenKind.ENDMARKER:
            raise StopIteration
        return t

    def seekstart(self):
        self.io.seek(self.io_startpos)
        self.current_row = 1
        self.current_col = 1
        self.token_start_row = 1
        self.token_start_col = 1
        # Refill buffer
        self.io.seek(self.io_startpos)
        c1 = self.io.read(1)
        c2 = self.io.read(1)
        c3 = self.io.read(1)

        self.chars = [c1, c2, c3]
        self.current_pos = self.io.tell()

    def start_token(self):
        self.token_startpos = self.current_pos  # Roughly
        self.token_start_row = self.current_row
        self.token_start_col = self.current_col

    def readchar(self) -> str:
        # Shift buffer
        c = self.io.read(1)

        current_char = self.chars[0]
        self.chars[0] = self.chars[1]
        self.chars[1] = self.chars[2]
        self.chars[2] = c

        if self.doread:
            self.charstore.write(current_char)

        if current_char == "\n":
            self.current_row += 1
            self.current_col = 1
        elif current_char != "":
            self.current_col += 1

        return current_char

    def readon(self, c: Optional[str] = None):
        # Clear charstore and start accumulating
        self.charstore = io.StringIO()
        if c is not None:
            self.charstore.write(c)
        self.doread = True

    def readoff(self):
        self.doread = False
        return self.chars[0]

    def peekchar(self):
        return self.chars[0]

    def dpeekchar(self):
        return self.chars[0], self.chars[1]

    def accept(self, f: Union[Callable[[str], bool], str, set]) -> bool:
        c = self.peekchar()
        ok = False
        if callable(f):
            ok = f(c)
        elif isinstance(f, str):
            ok = c == f
        else:  # set/list
            ok = c in f

        if ok:
            self.readchar()
        return ok

    def accept_batch(self, f) -> bool:
        ok = False
        while self.accept(f):
            ok = True
        return ok

    def emit(self, kind: TokenKind, err: Optional[str] = None) -> Token:
        if kind == TokenKind.ERROR:
            s = self.charstore.getvalue()
        elif (
            kind == TokenKind.IDENTIFIER
            or is_literal(kind)
            or kind == TokenKind.WHITESPACE
            or kind == TokenKind.COMMENT
        ):
            s = self.charstore.getvalue()
        else:
            s = ""

        # Reset charstore
        self.charstore = io.StringIO()

        tok = Token(
            kind=kind,
            startpos=(self.token_start_row, self.token_start_col),
            endpos=(self.current_row, self.current_col - 1),
            val=s,
            token_error=err,
        )
        self.last_token_kind = kind
        self.readoff()
        return tok

    def emit_error(self, err: str = "unknown"):
        return self.emit(TokenKind.ERROR, err)

    def next_token(self, start=True) -> Token:
        if start:
            self.start_token()

        c = self.readchar()

        if c == "":
            return self.emit(TokenKind.ENDMARKER)
        elif c.isspace():
            return self.lex_whitespace(c)
        elif c == "[":
            return self.emit(TokenKind.LSQUARE)
        elif c == "]":
            return self.emit(TokenKind.RSQUARE)
        elif c == "{":
            return self.emit(TokenKind.LBRACE)
        elif c == ";":
            return self.emit(TokenKind.SEMICOLON)
        elif c == "}":
            return self.emit(TokenKind.RBRACE)
        elif c == "(":
            return self.emit(TokenKind.LPAREN)
        elif c == ",":
            return self.emit(TokenKind.COMMA)
        elif c == ")":
            return self.emit(TokenKind.RPAREN)
        elif c == "*":
            return self.lex_star(c)
        elif c == "^":
            return self.emit(TokenKind.CIRCUMFLEX_ACCENT)
        elif c == "/":
            return self.lex_fwdslash(c)
        elif c == "+":
            return self.emit(TokenKind.PLUS)
        elif c == "-":
            return self.lex_minus(c)
        elif c == "=":
            return self.lex_equal(c)
        elif c == '"':
            return self.lex_quote()
        elif c.isidentifier() or c == "_" or c.isalpha():
            return self.lex_identifier(c)
        elif c.isdigit():
            return self.lex_digit(c, TokenKind.INTEGER)
        elif c == ".":
            # Check if it's a real number starting with .
            pc = self.peekchar()
            if pc.isdigit():
                return self.lex_digit(c, TokenKind.REAL)
            else:
                return self.emit_error()
        elif c in UNICODE_OPS:
            return self.emit(UNICODE_OPS[c])
        else:
            return self.emit_error()

    def lex_whitespace(self, c):
        self.readon(c)
        self.accept_batch(lambda ch: ch.isspace())
        return self.emit(TokenKind.WHITESPACE)

    def lex_star(self, c):
        if self.accept("*"):
            return self.emit_error("invalid operator")
        return self.emit(TokenKind.STAR)

    def lex_minus(self, c):
        if self.accept(">"):
            return self.emit(TokenKind.ARROW)
        return self.emit(TokenKind.MINUS)

    def lex_equal(self, c):
        if not self.accept("="):
            return self.emit_error("invalid operator")
        return self.emit(TokenKind.EQEQ)

    def lex_fwdslash(self, c):
        self.readon(c)
        if self.accept("/"):
            # It's a comment //
            while True:
                pc = self.peekchar()
                if pc == "\n" or pc == "":
                    return self.emit(TokenKind.COMMENT)
                self.readchar()
        return self.emit(TokenKind.FWD_SLASH)

    def lex_quote(self):
        self.readon()
        if self.accept('"'):
            return self.emit(TokenKind.STRING)

        self.charstore = io.StringIO()  # Clean
        self.doread = True  # Start reading

        while True:
            c = self.peekchar()
            if c == "\\":
                self.readchar()  # consume backslash
                if self.peekchar() == "":
                    return self.emit_error("unterminated string literal")
                self.readchar()  # consume escaped char
                continue

            if c == '"':
                # End of string
                tok = self.emit(TokenKind.STRING)
                self.readchar()  # consume closing quote
                return tok
            elif c == "":
                return self.emit_error("unterminated string literal")
            else:
                self.readchar()

    def lex_identifier(self, c):
        self.readon(c)
        while True:
            pc = self.peekchar()
            if pc.isidentifier() or pc.isdigit() or pc == "_" or pc == "/" or pc == "!":
                self.readchar()
            else:
                break

        # Extract string
        s = self.charstore.getvalue()
        # Check keywords
        kw_kind = KEYWORDS.get(s, TokenKind.IDENTIFIER)

        return self.emit(kw_kind)

    def lex_digit(self, c, kind):
        self.readon(c)

        def accept_number():
            while True:
                pc = self.peekchar()
                if pc == "_":  # 1_000
                    self.readchar()  # consume _
                    continue
                if pc.isdigit():
                    self.readchar()
                else:
                    return

        accept_number()

        pc, ppc = self.dpeekchar()

        if pc == ".":
            if kind == TokenKind.REAL:
                self.readchar()
                return self.emit_error("invalid numeric constant")

            self.readchar()  # consume .
            kind = TokenKind.REAL
            accept_number()

            pc, ppc = self.dpeekchar()
            if (pc == "e" or pc == "E") and (
                ppc.isdigit() or ppc == "+" or ppc == "-"
            ):
                # Exponent
                self.readchar()  # consume e
                self.accept({"+", "-"})
                if self.accept_batch(lambda c: c.isdigit()):
                    pc, _ = self.dpeekchar()
                    if pc == ".":
                        self.readchar()
                        return self.emit_error("invalid numeric constant")
                else:
                    return self.emit_error()

        # Check for start of exponent without dot
        elif (pc == "e" or pc == "E") and (ppc.isdigit() or ppc == "+" or ppc == "-"):
            kind = TokenKind.REAL
            self.readchar()  # consume e
            self.accept({"+", "-"})
            if not self.accept_batch(lambda c: c.isdigit()):
                return self.emit_error()
            pc, _ = self.dpeekchar()
            if pc == ".":
                self.readchar()
                return self.emit_error()

        return self.emit(kind)


def tokenize(text: str) -> Iterator[Token]:
    return Lexer(text)


def is_literal(kind: TokenKind) -> bool:
    from mimiqcircuits.qasm.tokens import is_literal

    return is_literal(kind)
