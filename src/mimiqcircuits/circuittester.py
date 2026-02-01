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
Circuit tester experiment to check equivalence between two circuits.
"""

from bitarray import frozenbitarray

from mimiqcircuits import Circuit, GateH, GateCX, Measure, inverse
from mimiqcircuits import BitString


class CircuitTesterExperiment:
    r"""
    Represents a circuit equivalence checking experiment.

    Constructs a circuit to check equivalence between `c1` and `c2`.
    We implement the Choi-Jamiolkowski isomorphism to compare the channels.
    If the channels are equivalent, the final state must be |0...0>.

    Args:
        c1 (Circuit): First circuit.
        c2 (Circuit): Second circuit.
        method (str, optional): Verification method. Can be "samples" (default) or "amplitudes".
                                "samples" measures the final state. "amplitudes" computes the amplitude of the all-zero state.

    Example:
        >>> from mimiqcircuits import *
        >>> c1 = Circuit().push(GateX(), 0)
        >>> c2 = Circuit().push(GateX(), 0)
        >>> exp = CircuitTesterExperiment(c1, c2, method="samples")
        >>> # To execute: conn.execute(exp)
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
        """
        Builds the circuit to test equivalence between `c1` and `c2`.
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
        c.push(GateH(), input_qubits)
        c.push(GateCX(), input_qubits, test_ancilla)

        # Apply channel and inverse channel
        c.append(c1)
        c.append(inverse(c2))

        # Uncompute Bell state to map identity to computational basis zero state
        c.push(GateCX(), input_qubits, test_ancilla)
        c.push(GateH(), input_qubits)

        # Measure in computational basis
        if self.method == "samples":
            c.push(Measure(), list(range(total_qubits)), list(range(total_qubits)))
        elif self.method == "amplitudes":
            from mimiqcircuits.operations.amplitude import Amplitude

            target_state = BitString("0" * total_qubits)
            c.push(Amplitude(target_state), 0)

        return c

    def interpret_results(self, results):
        """
        Verifies the results of the circuit tester experiment. Returns the probability of the all-zero state.

        Args:
            results (QCSResults): Results from the execution.
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

        target_state = frozenbitarray("0" * total_qubits)
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
