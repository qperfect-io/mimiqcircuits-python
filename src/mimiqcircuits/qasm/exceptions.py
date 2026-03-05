class QASMError(Exception):
    """Base class for QASM errors."""

    pass


class TokenError(QASMError):
    """Raised when a tokenization error occurs."""

    def __init__(self, message, token=None, position=None):
        self.message = message
        self.token = token
        self.position = position
        super().__init__(message)


class ParseError(QASMError):
    """Raised when a parsing error occurs."""

    def __init__(self, message, token=None, parser_state=None):
        self.message = message
        self.token = token
        self.parser_state = parser_state
        super().__init__(message)


class QASMSerializationError(QASMError):
    """Raised when a serialization error occurs."""

    pass


class QASMArgumentError(QASMError):
    pass


class UndefinedGateError(QASMError):
    pass


class DuplicateRegisterError(QASMError):
    pass


class UndefinedRegisterError(QASMError):
    pass


class QASMStructureError(QASMError):
    pass


class QASMVersionError(QASMError):
    pass
