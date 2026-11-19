#
# Copyright © 2023-2025 QPerfect. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Experiment to check equivalence between two circuits.
"""

import warnings

from mimiqcircuits.circuit import Circuit
from mimiqcircuits.lazy import inverse
from mimiqcircuits.operations.gates.standard.cpauli import GateCX
from mimiqcircuits.operations.gates.standard.hadamard import GateH
from mimiqcircuits.operations.measure import Measure
from mimiqcircuits.bitstrings import BitString


class EquivalenceExperiment:
    r"""An experiment that checks whether two unitary circuits are equivalent.

    The test uses the Choi-Jamiolkowski isomorphism. Each of the :math:`n`
    input qubits is entangled with an ancilla in a Bell pair, then ``c1``
    and the inverse of ``c2`` are applied to the input qubits, and the Bell
    pairs are undone. The :math:`2n` qubits end in the all-zero state with
    probability

    .. math::

        P = \left| \frac{\operatorname{Tr}(U_2^\dagger U_1)}{2^n} \right|^2,

    which is 1 exactly when the two circuits implement the same unitary up
    to a global phase. :meth:`build_circuit` returns this test circuit and
    :meth:`interpret_results` reads :math:`P` from its results.

    Both circuits must be invertible, so they cannot contain measurements,
    resets or noise.

    Args:
        c1 (Circuit): The first circuit.
        c2 (Circuit): The second circuit.
        method (str): How :math:`P` is read. ``"samples"`` (the default)
            measures all qubits and counts the all-zero outcomes;
            ``"amplitudes"`` computes the amplitude of the all-zero state
            exactly.

    Raises:
        ValueError: If ``method`` is not ``"samples"`` or ``"amplitudes"``.

    Examples:
        >>> from mimiqcircuits import *
        >>> # HZH and X are the same gate
        >>> c1 = Circuit().push(GateH(), 0).push(GateZ(), 0).push(GateH(), 0)
        >>> c2 = Circuit().push(GateX(), 0)
        >>> exp = EquivalenceExperiment(c1, c2, method="amplitudes")
        >>> exp
        Choi-isomorphism circuit equivalence test:
        ├── a == b
        │   ├── a: 1-qubit circuit with 3 instructions
        │   └── b: 1-qubit circuit with 1 instruction
        └── compare amplitudes
        >>> # run the test circuit remotely and read the probability
        >>> conn = MimiqConnection()  # doctest: +SKIP
        >>> conn.connect()  # doctest: +SKIP
        >>> job = conn.submit(exp)  # doctest: +SKIP
        >>> exp.interpret_results(conn.get_result(job))  # doctest: +SKIP
        1.0
    """

    def __init__(self, c1: Circuit, c2: Circuit, method="samples"):
        nqinput = max(c1.num_qubits(), c2.num_qubits())

        if method not in ["samples", "amplitudes"]:
            raise ValueError("Method must be one of 'samples' or 'amplitudes'")

        self.c1 = c1
        self.c2 = c2
        self.nqinput = nqinput
        self.method = method

    def build_circuit(self):
        """Return the test circuit of the experiment.

        It uses twice as many qubits as the larger of the two circuits: the
        input qubits come first, the ancillas after them. With
        ``method="samples"`` it ends by measuring every qubit into the bit of
        the same index; with ``method="amplitudes"`` it ends with an
        :class:`Amplitude` of the all-zero state stored in z-variable 0. When
        both circuits are empty the test circuit is empty too.

        Returns:
            Circuit: The test circuit.

        Raises:
            TypeError: If an operation of ``c2`` cannot be inverted.
            NotImplementedError: If an operation of ``c2`` is a Kraus channel.

        Examples:
            >>> from mimiqcircuits import *
            >>> c = Circuit().push(GateX(), 0)
            >>> EquivalenceExperiment(c, c, method="amplitudes").build_circuit()
            2-qubit, 1-zvar circuit with 7 instructions:
            ├── H @ q[0]
            ├── CX @ q[0], q[1]
            ├── X @ q[0]
            ├── X @ q[0]
            ├── CX @ q[0], q[1]
            ├── H @ q[0]
            └── Amplitude(bs"00") @ z[0]
            <BLANKLINE>
        """
        c1 = self.c1
        c2 = self.c2
        nqinput = self.nqinput

        if nqinput == 0:
            return Circuit()

        total_qubits = nqinput * 2

        input_qubits = list(range(0, nqinput))
        test_ancilla = list(range(nqinput, nqinput * 2))

        # Prepare Bell state for Choi-Jamiolkowski isomorphism
        c = Circuit()
        bell_circuit = Circuit()
        bell_circuit.push(GateH(), input_qubits)
        bell_circuit.push(GateCX(), input_qubits, test_ancilla)
        c.append(bell_circuit)

        # Apply channel and inverse channel
        c.append(c1)
        c.append(inverse(c2))

        # Uncompute Bell state to map identity to computational basis zero state
        c.append(inverse(bell_circuit))

        # Measure in computational basis
        if self.method == "samples":
            c.push(Measure(), list(range(total_qubits)), list(range(total_qubits)))
        elif self.method == "amplitudes":
            from mimiqcircuits.operations.amplitude import Amplitude

            target_state = BitString("0" * total_qubits)
            c.push(Amplitude(target_state), 0)

        return c

    def interpret_results(self, results):
        """Return the probability of the all-zero state from the results of the test circuit.

        With ``method="samples"`` this is the fraction of samples that
        measured all zeros. With ``method="amplitudes"`` it is the squared
        modulus of the computed amplitude, averaged over the results that
        have one. A value of 1 means the circuits are equivalent up to a
        global phase. When both circuits are empty the result is 1.0; when
        the results hold no samples or amplitudes it is 0.0.

        Args:
            results (QCSResults): The results of running :meth:`build_circuit`.

        Returns:
            float: The probability, between 0 and 1.
        """
        if self.nqinput == 0:
            return 1.0

        if self.method == "amplitudes":
            # Calculate average probability from amplitudes
            if not results.zstates:
                return 0.0

            total_prob = 0.0
            count = 0
            for zstate in results.zstates:
                if zstate:
                    amp = zstate[0]
                    prob = abs(amp) ** 2
                    total_prob += prob
                    count += 1

            if count == 0:
                return 0.0
            return total_prob / count

        total_qubits = self.nqinput * 2

        # Equivalence implies probability of |0...0> is 1.0

        total_samples = len(results.cstates)
        if total_samples == 0:
            return 0.0

        target_state = BitString("0" * total_qubits)
        counts = results.histogram()

        # count occurrences of "0...0"
        zeros_count = counts.get(target_state, 0)

        # return probability of measuring all-zero state
        return zeros_count / total_samples

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = "Choi-isomorphism circuit equivalence test:\n"
        s += "├── a == b\n"
        s += f"│   ├── a: {self.c1._header()}\n"
        s += f"│   └── b: {self.c2._header()}\n"
        s += f"└── compare {self.method}"
        return s

    def saveproto(self, file):
        """Save the experiment to a protobuf (binary) file.

        Args:
            file: A file path or a binary file-like object to write to.

        Returns:
            int: The number of bytes written.
        """
        from mimiqcircuits.proto.protoio import saveproto

        return saveproto(self, file)

    @staticmethod
    def loadproto(file):
        """Load an experiment saved with :meth:`saveproto`.

        Args:
            file: A file path or a binary file-like object to read from.

        Returns:
            EquivalenceExperiment: The loaded experiment.
        """
        from mimiqcircuits.proto.protoio import loadproto

        return loadproto(file, EquivalenceExperiment)


class CircuitTesterExperiment(EquivalenceExperiment):
    """Deprecated alias of :class:`EquivalenceExperiment`.

    Creating one emits a ``DeprecationWarning``.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "CircuitTesterExperiment is deprecated, use EquivalenceExperiment",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
