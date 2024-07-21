import mimiqcircuits as mc


class MeasureReset(mc.Operation):
    """MeasureReset operation.

    This operation measures a qubit q, stores the value in a classical bit c,
    then applies a X operation to the qubit if the measured value is 1, effectively
    resetting the qubit to the |0> state.

    Examples:

        >>> from mimiqcircuits import *
        >>> c = Circuit()
        >>> c.push(MeasureReset(), 1, 0)
        2-qubit circuit with 1 instructions:
        └── MeasureReset @ q[1], c[0]
        <BLANKLINE>
        >>> c.decompose()
        init if -> operation= X
        2-qubit circuit with 2 instructions:
        ├── Measure @ q[1], c[0]
        └── IF(c == 1) X @ q[1], c[0]
        <BLANKLINE>
    """

    _name = "MeasureReset"
    _num_bits = 1
    _num_qubits = 1
    _num_cregs = 1
    _qregsizes = [1]
    _cregsizes = [1]

    def inverse(self):
        raise TypeError("MeasureReset is not inversible")

    def power(self, p):
        raise TypeError("MeasureReset^p is not defined.")

    def control(self, num_qubits):
        raise TypeError("Controlled MeasureReset is not defined.")

    def iswrapper(self):
        return False

    def _decompose(self, circ, qubits, bits):
        circ.push(mc.Measure(), qubits, bits)
        circ.push(mc.IfStatement(mc.GateX(), 1, 1), qubits, bits)

        return circ

    def asciiwidth(self, qubits, bits):
        return max(2, len(str(self)))  # Adjust this value as needed

    def get_operation(self):
        return self

    def __str__(self):
        return f"{self._name}"


# export operations
__all__ = ["MeasureReset"]
