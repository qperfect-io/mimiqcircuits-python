class QASMError(Exception):
    """Base class for the errors raised when reading or writing OpenQASM."""

    pass


class TokenError(QASMError):
    """Raised when the OpenQASM source contains text that is not a valid token."""

    def __init__(self, message, token=None, position=None):
        self.message = message
        self.token = token
        self.position = position
        super().__init__(message)


class ParseError(QASMError):
    """Raised when the OpenQASM source is not valid syntax, uses an unsupported
    version, or includes a file that cannot be found."""

    def __init__(self, message, token=None, parser_state=None):
        self.message = message
        self.token = token
        self.parser_state = parser_state
        super().__init__(message)


class QASMSerializationError(QASMError):
    """Raised when a circuit cannot be written as OpenQASM 2.0."""

    pass


class QASMArgumentError(QASMError):
    """Raised when the parsed input is not a program or declares an unsupported opaque gate."""


class UndefinedGateError(QASMError):
    """Raised when the OpenQASM source calls a gate that is not declared."""


class DuplicateRegisterError(QASMError):
    """Raised when the OpenQASM source declares the same register name twice."""


class UndefinedRegisterError(QASMError):
    """Raised when the OpenQASM source uses a register that is not declared."""


class QASMStructureError(QASMError):
    """Raised when an OpenQASM statement is well formed but cannot be
    interpreted, for example an unknown operation, an invalid target, or an
    ``if`` condition other than equality."""


class QASMVersionError(QASMError):
    """Raised when the OpenQASM version is not 2.0."""
