from mimiqcircuits import *
import pytest
from bitarray import frozenbitarray


def test_circuittester_basic_repr():
    c1 = Circuit()
    c1.push(Control(3, GateX()), *range(0, 4))

    assert c1.num_qubits() == 4
    assert len(c1.instructions) == 1

    c2 = Circuit()
    c2.push(GateCCX(), 0, 1, 4)
    c2.push(Control(2, GateX()), 4, 2, 3)
    c2.push(Control(2, GateX()), 0, 1, 4)
    assert c2.num_qubits() == 5

    # Create the experiment
    ex = CircuitTesterExperiment(c1, c2)

    # Build the circuit
    tester = ex.build_circuit()

    assert tester.num_qubits() == 10
    assert tester.num_zvars() == 0


def test_interpret_results():
    c1 = Circuit()
    c2 = Circuit()
    ex = CircuitTesterExperiment(c1, c2)

    # Mock results with histogram method
    class MockResults:
        def __init__(self, counts=None, zstates=None, cstates=None):
            self.counts = counts if counts is not None else {}
            self.zstates = zstates if zstates is not None else []
            self.cstates = cstates if cstates is not None else []
            if cstates is None and counts is not None:
                # Naive reconstruction of cstates from counts for length check
                self.cstates = []
                for k, v in self.counts.items():
                    self.cstates.extend([k] * v)

        def histogram(self):
            return self.counts

    # Edge case: empty circuits

    # Non-empty circuits
    c1 = Circuit().push(GateX(), 0)
    c2 = Circuit().push(GateX(), 0)
    ex = CircuitTesterExperiment(c1, c2)

    # Perfect
    res_perfect = MockResults({frozenbitarray("00"): 100})
    assert ex.interpret_results(res_perfect) == 1.0

    # Imperfect
    res_bad = MockResults({frozenbitarray("00"): 50, frozenbitarray("11"): 50})
    assert ex.interpret_results(res_bad) == 0.5

    # Zero samples
    res_empty = MockResults({})
    assert ex.interpret_results(res_empty) == 0.0


def test_interpret_results_amplitudes():
    c1 = Circuit().push(GateX(), 0)
    c2 = Circuit().push(GateX(), 0)

    ex = CircuitTesterExperiment(c1, c2, method="amplitudes")

    # Verify build_circuit uses Amplitude
    built_c = ex.build_circuit()
    assert any("Amplitude" in str(inst) for inst in built_c.instructions)

    class MockResults:
        def __init__(self, zstates):
            self.zstates = zstates
            self.cstates = []

    # Perfect equivalence
    res_perfect = MockResults([[1.0], [1.0]])
    assert ex.interpret_results(res_perfect) == 1.0

    res_perfect_phase = MockResults([[1j], [-1j]])
    assert ex.interpret_results(res_perfect_phase) == 1.0

    # Imperfect
    val = 1 / (2**0.5)
    res_bad = MockResults([[val], [val]])
    assert abs(ex.interpret_results(res_bad) - 0.5) < 1e-6

    # Empty
    res_empty = MockResults([])
    assert ex.interpret_results(res_empty) == 0.0
