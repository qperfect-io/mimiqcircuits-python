#
# Copyright © 2022-2024 University of Strasbourg. All Rights Reserved.
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

from fractions import Fraction

import numpy as np
import pytest
import symengine as se

import mimiqcircuits as mc
from mimiqcircuits.proto.circuitproto import (
    fromproto_arg,
    fromproto_circuit,
    fromproto_gate,
    fromproto_gatedecl,
    fromproto_krauschannel,
    fromproto_operation,
    fromproto_operator,
    toproto_arg,
    toproto_circuit,
    toproto_gate,
    toproto_gatedecl,
    toproto_krauschannel,
    toproto_operation,
    toproto_operator,
)


class TestArgConversion:
    """Test conversion of basic argument types."""

    @pytest.mark.parametrize(
        "value",
        [
            42,  # integer
            3.14,  # float
            "param",  # string/symbol
            se.Symbol("theta"),  # symengine symbol
            complex(3.213, 0),  # complex number
            complex(3, 0),  # complex number
            se.pi,  # irrational number
            se.E,  # irrational number
        ],
    )
    def test_arg_conversion_roundtrip(self, value):
        """Test that values convert to proto and back correctly."""
        proto_value = toproto_arg(value)
        restored_value = fromproto_arg(proto_value)

        # Handle different types of comparison
        if isinstance(value, (int, float)):
            assert restored_value == value
        elif isinstance(value, complex):
            assert restored_value is not complex
            assert restored_value == value.real
        elif isinstance(value, se.Symbol):
            assert str(restored_value) == str(value)
        else:  # For irrational numbers like pi and e
            assert str(restored_value) == str(value)

    def test_expression_conversion(self):
        """Test conversion of symengine expressions."""
        x = se.Symbol("x")
        y = se.Symbol("y")

        expressions = [
            x + y,  # Addition
            x * y,  # Multiplication
            x**2,  # Power
            se.sin(x),  # Sin function
            se.cos(x),  # Cos function
            se.log(x),  # Log function
        ]

        for expr in expressions:
            proto_expr = toproto_arg(expr)
            restored_expr = fromproto_arg(proto_expr)
            # Compare string representation
            assert str(restored_expr) == str(expr)


class TestGateConversion:
    """Test conversion of gate types."""

    @pytest.mark.parametrize(
        "gate",
        [
            mc.GateID(),
            mc.GateX(),
            mc.GateY(),
            mc.GateZ(),
            mc.GateH(),
            mc.GateRX(0.5),
            mc.GateRY(np.pi / 4),
            mc.GateRZ(se.Symbol("theta")),
            mc.GateSWAP(),
            mc.GateU(0.1, 0.2, 0.3),
            mc.GateRZZ(0.3),
        ],
    )
    def test_simple_gate_conversion(self, gate):
        """Test conversion of simple gates."""
        proto_gate = toproto_gate(gate)
        restored_gate = fromproto_gate(proto_gate)

        # Check type equality
        assert type(restored_gate) == type(gate)

        # For parametric gates, check parameters
        if hasattr(gate, "getparams") and callable(gate.getparams):
            params = gate.getparams()
            restored_params = restored_gate.getparams()

            for p1, p2 in zip(params, restored_params):
                if isinstance(p1, (int, float)):
                    assert p1 == p2
                else:
                    assert str(p1) == str(p2)

    def test_custom_gate_conversion(self):
        """Test conversion of custom gates."""
        matrix = np.array([[1, 0], [0, -1]])
        gate = mc.GateCustom(matrix=matrix)

        proto_gate = toproto_gate(gate)
        restored_gate = fromproto_gate(proto_gate)

        assert np.array_equal(gate.matrix, restored_gate.matrix)

    def test_control_gate_conversion(self):
        """Test conversion of control gates."""
        base_gate = mc.GateX()
        control_gate = mc.Control(2, base_gate)

        proto_gate = toproto_gate(control_gate)
        restored_gate = fromproto_gate(proto_gate)

        assert isinstance(restored_gate, mc.Control)
        assert restored_gate.num_controls == control_gate.num_controls
        assert type(restored_gate.op) == type(control_gate.op)

    def test_power_gate_conversion(self):
        """Test conversion of power gates with different exponent types."""
        # Test with integer exponent
        power_int = mc.Power(mc.GateX(), 2)
        proto_gate = toproto_gate(power_int)
        restored_gate = fromproto_gate(proto_gate)
        assert isinstance(restored_gate, mc.Power)
        assert type(restored_gate.op) == type(power_int.op)
        assert restored_gate.exponent == power_int.exponent

        # Test with Fraction exponent
        power_frac = mc.Power(mc.GateX(), Fraction(1, 2))
        proto_gate = toproto_gate(power_frac)
        restored_gate = fromproto_gate(proto_gate)
        assert isinstance(restored_gate, mc.Power)
        assert restored_gate.exponent == power_frac.exponent

        # Test with float exponent
        power_float = mc.Power(mc.GateX(), 0.5)
        proto_gate = toproto_gate(power_float)
        restored_gate = fromproto_gate(proto_gate)
        assert isinstance(restored_gate, mc.Power)
        assert restored_gate.exponent == power_float.exponent

    def test_inverse_gate_conversion(self):
        """Test conversion of inverse gates."""
        inverse_gate = mc.Inverse(mc.GateH())

        proto_gate = toproto_gate(inverse_gate)
        restored_gate = fromproto_gate(proto_gate)

        assert isinstance(restored_gate, mc.Inverse)
        assert type(restored_gate.op) == type(inverse_gate.op)

    def test_parallel_gate_conversion(self):
        """Test conversion of parallel gates."""
        parallel_gate = mc.Parallel(3, mc.GateX())

        proto_gate = toproto_gate(parallel_gate)
        restored_gate = fromproto_gate(proto_gate)

        assert isinstance(restored_gate, mc.Parallel)
        assert restored_gate.num_repeats == parallel_gate.num_repeats
        assert type(restored_gate.op) == type(parallel_gate.op)

    def test_pauli_string_gate_conversion(self):
        """Test conversion of PauliString gates."""
        gate = mc.PauliString("XYZ")

        proto_gate = toproto_gate(gate)
        restored_gate = fromproto_gate(proto_gate)

        assert isinstance(restored_gate, mc.PauliString)
        assert restored_gate.pauli == gate.pauli
        assert restored_gate.num_qubits == gate.num_qubits

    def test_gatedecl_gatecall_conversion(self):
        """Test conversion of GateDecl and GateCall."""
        # Create a GateDecl with a parameterized circuit
        theta = se.Symbol("theta")
        circ = mc.Circuit()
        circ.push(mc.GateRZ(theta), 0)
        circ.push(mc.GateX(), 0)
        decl = mc.GateDecl(name="MyGate", arguments=(theta,), circuit=circ)

        # Create a GateCall that calls the declaration with a specific value
        call = decl(np.pi / 2)

        # Test GateDecl conversion
        proto_decl = toproto_gatedecl(decl)
        restored_decl = fromproto_gatedecl(proto_decl)

        assert isinstance(restored_decl, mc.GateDecl)
        assert restored_decl.name == decl.name
        assert restored_decl.arguments == decl.arguments
        assert len(restored_decl.circuit.instructions) == len(decl.circuit.instructions)

        # Test GateCall conversion
        proto_call = toproto_gate(call)
        restored_call = fromproto_gate(proto_call)

        assert isinstance(restored_call, mc.GateCall)
        assert restored_call._args[0] == call._args[0]
        assert restored_call._decl.name == call._decl.name


class TestOperatorConversion:
    """Test conversion of operator types."""

    @pytest.mark.parametrize(
        "operator",
        [
            mc.SigmaMinus(),
            mc.SigmaPlus(),
            mc.Projector0(),
            mc.Projector1(),
            mc.Projector00(),
            mc.Projector11(),
            mc.ProjectorX0(),
            mc.ProjectorY1(),
        ],
    )
    def test_simple_operator_conversion(self, operator):
        """Test conversion of simple operators."""
        proto_op = toproto_operator(operator)
        restored_op = fromproto_operator(proto_op)

        assert type(restored_op) == type(operator)

    def test_rescaled_gate_conversion(self):
        """Test conversion of RescaledGate operator."""
        gate = mc.GateX()
        scale = 0.5
        rescaled = mc.RescaledGate(gate, scale)

        proto_op = toproto_operator(rescaled)
        restored_op = fromproto_operator(proto_op)

        assert isinstance(restored_op, mc.RescaledGate)
        assert restored_op.get_scale() == rescaled.get_scale()
        assert type(restored_op.get_operation()) == type(rescaled.get_operation())

    def test_custom_operator_conversion(self):
        """Test conversion of custom Operator."""
        matrix = np.array([[1, 0], [0, -1]])
        operator = mc.Operator(mat=matrix)

        proto_op = toproto_operator(operator)
        restored_op = fromproto_operator(proto_op)

        # Compare matrices
        assert np.array_equal(
            operator.matrix if not callable(operator.matrix) else operator.matrix(),
            (
                restored_op.matrix
                if not callable(restored_op.matrix)
                else restored_op.matrix()
            ),
        )


class TestKrausChannelConversion:
    """Test conversion of Kraus channels."""

    @pytest.mark.parametrize(
        "channel",
        [
            mc.Reset(),
            mc.ResetX(),
            mc.ResetY(),
            mc.ResetZ(),
            mc.AmplitudeDamping(0.1),
            mc.GeneralizedAmplitudeDamping(0.1, 0.2),
            mc.PhaseAmplitudeDamping(0.1, 0.2, 0.3),
            mc.PauliX(0.1),
            mc.PauliY(0.2),
            mc.PauliZ(0.3),
        ],
    )
    def test_simple_kraus_channel_conversion(self, channel):
        """Test conversion of simple Kraus channels."""
        proto_channel = toproto_krauschannel(channel)
        restored_channel = fromproto_krauschannel(proto_channel)

        assert type(restored_channel) == type(channel)

        # For parametric channels, check parameters
        if hasattr(channel, "getparams") and callable(channel.getparams):
            params = channel.getparams()
            restored_params = restored_channel.getparams()

            for p1, p2 in zip(params, restored_params):
                if isinstance(p1, (int, float)):
                    assert p1 == p2
                else:
                    assert str(p1) == str(p2)

    def test_depolarizing_conversion(self):
        """Test conversion of Depolarizing channel."""
        channel = mc.Depolarizing(1, 0.1)

        proto_channel = toproto_krauschannel(channel)
        restored_channel = fromproto_krauschannel(proto_channel)

        assert isinstance(restored_channel, mc.Depolarizing)
        assert restored_channel.N == channel.N
        assert restored_channel.p == channel.p

    def test_pauli_noise_conversion(self):
        """Test conversion of PauliNoise channel."""
        probs = [0.7, 0.1, 0.1, 0.1]
        pauli_strs = [
            mc.PauliString("I"),
            mc.PauliString("X"),
            mc.PauliString("Y"),
            mc.PauliString("Z"),
        ]
        channel = mc.PauliNoise(probs, pauli_strs)

        proto_channel = toproto_krauschannel(channel)
        restored_channel = fromproto_krauschannel(proto_channel)

        assert isinstance(restored_channel, mc.PauliNoise)

        # Compare probabilities and Pauli strings
        for p1, p2 in zip(channel.probabilities(), restored_channel.probabilities()):
            assert p1 == p2

        for ps1, ps2 in zip(channel.paulistr, restored_channel.paulistr):
            assert ps1.pauli == ps2.pauli

    def test_mixed_unitary_conversion(self):
        """Test conversion of MixedUnitary channel."""
        probs = [0.8, 0.2]
        gates = [mc.GateID(), mc.GateX()]
        channel = mc.MixedUnitary(probs, gates)

        proto_channel = toproto_krauschannel(channel)
        restored_channel = fromproto_krauschannel(proto_channel)

        assert isinstance(restored_channel, mc.MixedUnitary)

        # Compare probabilities
        for p1, p2 in zip(channel.probabilities(), restored_channel.probabilities()):
            assert abs(p1 - p2) < 1e-10

        # Compare gates
        gates1 = (
            channel.unitarygates()
            if hasattr(channel, "unitarygates")
            else channel.unitarymatrices()
        )
        gates2 = (
            restored_channel.unitarygates()
            if hasattr(restored_channel, "unitarygates")
            else restored_channel.unitarymatrices()
        )

        for g1, g2 in zip(gates1, gates2):
            assert type(g1) == type(g2)


class TestOperationConversion:
    """Test conversion of operations."""

    @pytest.mark.parametrize(
        "operation",
        [
            mc.MeasureX(),
            mc.MeasureY(),
            mc.MeasureZ(),
            mc.Measure(),
            mc.MeasureXX(),
            mc.MeasureYY(),
            mc.MeasureZZ(),
            mc.MeasureResetX(),
            mc.MeasureResetY(),
            mc.MeasureReset(),
            mc.Not(),
        ],
    )
    def test_simple_operation_conversion(self, operation):
        """Test conversion of simple operations."""
        proto_op = toproto_operation(operation)
        restored_op = fromproto_operation(proto_op)

        assert type(restored_op) == type(operation)

    def test_barrier_conversion(self):
        """Test conversion of Barrier operation."""
        operation = mc.Barrier(3)

        proto_op = toproto_operation(operation)
        restored_op = fromproto_operation(proto_op)

        assert isinstance(restored_op, mc.Barrier)
        assert restored_op.num_qubits == operation.num_qubits

    def test_amplitude_conversion(self):
        """Test conversion of Amplitude operation."""
        bitstring = mc.BitString("101")
        operation = mc.Amplitude(bitstring)

        proto_op = toproto_operation(operation)
        restored_op = fromproto_operation(proto_op)

        assert isinstance(restored_op, mc.Amplitude)
        assert restored_op.bs == operation.bs

    def test_expectation_value_conversion(self):
        """Test conversion of ExpectationValue operation."""
        operator = mc.Projector0()
        operation = mc.ExpectationValue(operator)

        proto_op = toproto_operation(operation)
        restored_op = fromproto_operation(proto_op)

        assert isinstance(restored_op, mc.ExpectationValue)
        assert type(restored_op.op) == type(operation.op)

    def test_if_statement_conversion(self):
        """Test conversion of IfStatement operation."""
        bitstring = mc.BitString("101")
        gate = mc.GateX()
        operation = mc.IfStatement(gate, bitstring)

        proto_op = toproto_operation(operation)
        restored_op = fromproto_operation(proto_op)

        assert isinstance(restored_op, mc.IfStatement)
        assert restored_op.bitstring == operation.bitstring
        assert type(restored_op.op) == type(operation.op)

    def test_detector_conversion(self):
        """Test conversion of Detector operation."""
        operation = mc.Detector(3, [1.0, 2.0, 3.0])

        proto_op = toproto_operation(operation)
        restored_op = fromproto_operation(proto_op)

        assert isinstance(restored_op, mc.Detector)
        assert restored_op._num_bits == operation._num_bits
        assert restored_op.get_notes() == operation.get_notes()

    def test_observable_include_conversion(self):
        """Test conversion of ObservableInclude operation."""
        operation = mc.ObservableInclude(2, [1.0, 2.0])

        proto_op = toproto_operation(operation)
        restored_op = fromproto_operation(proto_op)

        assert isinstance(restored_op, mc.ObservableInclude)
        assert restored_op._num_bits == operation._num_bits
        assert restored_op.get_notes() == operation.get_notes()

    def test_annotation_conversion(self):
        """Test conversion of annotation operations."""
        operations = [
            mc.Tick(),
            mc.QubitCoordinates([1.0, 2.0, 3.0]),
            mc.ShiftCoordinates([0.5, 0.5, 0.0]),
        ]

        for operation in operations:
            proto_op = toproto_operation(operation)
            restored_op = fromproto_operation(proto_op)

            assert type(restored_op) == type(operation)
            if hasattr(operation, "get_notes") and callable(operation.get_notes):
                assert restored_op.get_notes() == operation.get_notes()

    def test_custom_math_operations_conversion(self):
        """Test conversion of custom math operations: Pow, Add, Multiply."""
        # Pow
        pow_op = mc.Pow(2.0)
        proto_pow = toproto_operation(pow_op)
        restored_pow = fromproto_operation(proto_pow)
        assert isinstance(restored_pow, mc.Pow)
        assert restored_pow.exponent == pow_op.exponent

        # Add
        add_op = mc.Add(3, c=1.5)
        proto_add = toproto_operation(add_op)
        restored_add = fromproto_operation(proto_add)
        assert isinstance(restored_add, mc.Add)
        assert restored_add.term == add_op.term
        assert restored_add.num_zvars == add_op.num_zvars

        # Multiply
        mult_op = mc.Multiply(3, c=2.0)
        proto_mult = toproto_operation(mult_op)
        restored_mult = fromproto_operation(proto_mult)
        assert isinstance(restored_mult, mc.Multiply)
        assert restored_mult.factor == mult_op.factor
        assert restored_mult.num_zvars == mult_op.num_zvars


class TestGateDeclCaching:
    """Test that GateDecl objects are properly cached and not duplicated."""

    def test_reused_gatedecl(self):
        """Test that the same GateDecl used multiple times is only stored once."""
        # Create a gate declaration
        decl_circuit = mc.Circuit().push(mc.GateX(), 0)
        decl = mc.GateDecl(name="ReusedGate", arguments=(), circuit=decl_circuit)

        # Create a circuit that uses the same GateDecl multiple times
        circuit = mc.Circuit()
        circuit.push(decl(), 0)
        circuit.push(decl(), 1)
        circuit.push(decl(), 2)

        # Convert to protobuf
        proto_circuit = toproto_circuit(circuit)

        # Only one declaration should be in the decls dictionary
        assert len(proto_circuit.decls) == 1

        # Verify all instructions refer to the same declaration
        declid = None
        for inst in proto_circuit.instructions:
            if inst.operation.HasField("cachedgatecall"):
                if declid is None:
                    declid = inst.operation.cachedgatecall.id
                else:
                    assert inst.operation.cachedgatecall.id == declid

    def test_nested_gatedecl(self):
        """Test that nested GateDecl objects are properly cached."""
        # Create a base gate declaration
        base_circuit = mc.Circuit().push(mc.GateH(), 0)
        base_decl = mc.GateDecl(name="BaseGate", arguments=(), circuit=base_circuit)

        # Create a gate declaration that uses the base declaration
        mid_circuit = mc.Circuit().push(base_decl(), 0).push(base_decl(), 0)
        mid_decl = mc.GateDecl(name="MidGate", arguments=(), circuit=mid_circuit)

        # Create a top-level gate declaration that uses the mid-level declaration
        top_circuit = mc.Circuit().push(mid_decl(), 0).push(mid_decl(), 0)
        top_decl = mc.GateDecl(name="TopGate", arguments=(), circuit=top_circuit)

        # Create a circuit that uses the top-level declaration
        circuit = mc.Circuit().push(top_decl(), 0).push(top_decl(), 1)

        # Convert to protobuf
        proto_circuit = toproto_circuit(circuit)

        # There should be three declarations in total (BaseGate, MidGate, TopGate)
        assert len(proto_circuit.decls) == 3

        # Verify the declorder has the correct order (BaseGate should be processed first)
        assert len(proto_circuit.declorder) == 3

    def test_large_repeated_gatedecl(self):
        """Test that large repeated GateDecl objects don't explode in size."""
        # Create a gate declaration with many operations
        large_circuit = mc.Circuit()
        for _ in range(100):
            large_circuit.push(mc.GateX(), 0)
        large_decl = mc.GateDecl(name="LargeGate", arguments=(), circuit=large_circuit)

        # Create a circuit that uses the large declaration many times
        small_circuit = mc.Circuit().push(large_decl(), 0)

        large_circuit = mc.Circuit()
        for _ in range(100):
            large_circuit.push(mc.GateCall(large_decl, ()), 0)

        # Convert both to protobuf and compare sizes
        small_proto = toproto_circuit(small_circuit)
        large_proto = toproto_circuit(large_circuit)

        # Convert to bytes to get the size
        small_bytes = small_proto.SerializeToString()
        large_bytes = large_proto.SerializeToString()

        # The large circuit should be significantly smaller than 100x the small circuit
        # because the declaration is only stored once
        assert len(large_bytes) < 10 * len(small_bytes)

    def test_different_args_same_decl(self):
        """Test that the same GateDecl with different arguments is cached properly."""
        # Create a parameterized gate declaration
        theta = se.Symbol("theta")
        decl_circuit = mc.Circuit().push(mc.GateRZ(theta), 0)
        decl = mc.GateDecl(name="RotGate", arguments=(theta,), circuit=decl_circuit)

        # Create a circuit that uses the same GateDecl with different parameters
        circuit = mc.Circuit()
        circuit.push(decl(0.1), 0)
        circuit.push(decl(0.2), 0)
        circuit.push(decl(0.3), 0)

        # Convert to protobuf
        proto_circuit = toproto_circuit(circuit)

        # Only one declaration should be in the decls dictionary
        assert len(proto_circuit.decls) == 1

        # Check that each call has the right parameter but references the same decl
        declid = None
        params = []
        for inst in proto_circuit.instructions:
            if inst.operation.HasField("cachedgatecall"):
                if declid is None:
                    declid = inst.operation.cachedgatecall.id
                else:
                    assert inst.operation.cachedgatecall.id == declid

                # Extract and store the parameter
                param = inst.operation.cachedgatecall.args[
                    0
                ].argvalue_value.double_value
                params.append(param)

        # Check that all parameters were preserved correctly
        assert params == [0.1, 0.2, 0.3]


class TestCircuitConversion:
    """Test conversion of circuits."""

    def test_empty_circuit_conversion(self):
        """Test conversion of an empty circuit."""
        circuit = mc.Circuit([])

        proto_circuit = toproto_circuit(circuit)
        restored_circuit = fromproto_circuit(proto_circuit)

        assert len(restored_circuit.instructions) == 0

    def test_simple_circuit_conversion(self):
        """Test conversion of a simple circuit."""
        circuit = mc.Circuit()
        circuit.push(mc.GateH(), 0)
        circuit.push(mc.GateX(), 1)
        circuit.push(mc.GateCX(), 0, 1)
        circuit.push(mc.Measure(), 1, 0)

        proto_circuit = toproto_circuit(circuit)
        restored_circuit = fromproto_circuit(proto_circuit)

        assert len(restored_circuit.instructions) == len(circuit.instructions)

        for i, (orig_inst, restored_inst) in enumerate(
            zip(circuit.instructions, restored_circuit.instructions)
        ):
            assert type(orig_inst.operation) == type(restored_inst.operation)
            assert orig_inst.qubits == restored_inst.qubits
            assert orig_inst.bits == restored_inst.bits
            assert orig_inst.zvars == restored_inst.zvars

    def test_complex_circuit_conversion(self):
        """Test conversion of a more complex circuit with parametric gates and control."""
        theta = se.Symbol("theta")
        circuit = mc.Circuit()
        circuit.push(mc.GateH(), 0)
        circuit.push(mc.GateRZ(theta), 1)
        circuit.push(mc.GateX().control(1), 0, 1)
        circuit.push(mc.GateX().power(0.5), 2)
        circuit.push(mc.Barrier(3), 0, 1, 2)
        circuit.push(mc.MeasureZ(), 0, 0)

        proto_circuit = toproto_circuit(circuit)
        restored_circuit = fromproto_circuit(proto_circuit)

        assert len(restored_circuit.instructions) == len(circuit.instructions)

        for i, (orig_inst, restored_inst) in enumerate(
            zip(circuit.instructions, restored_circuit.instructions)
        ):
            assert type(orig_inst.operation) == type(restored_inst.operation)
            assert orig_inst.qubits == restored_inst.qubits
            assert orig_inst.bits == restored_inst.bits
            assert orig_inst.zvars == restored_inst.zvars

            # Check parameters for RZ gate
            if isinstance(orig_inst.operation, mc.GateRZ):
                param1 = orig_inst.operation.getparams()[0]
                param2 = restored_inst.operation.getparams()[0]
                assert str(param1) == str(param2)

    def test_circuit_with_declarations(self):
        """Test conversion of a circuit with gate declarations and calls."""
        # Create a GateDecl
        theta = se.Symbol("theta")
        decl_circuit = mc.Circuit()
        decl_circuit.push(mc.GateRZ(theta), 0)
        decl_circuit.push(mc.GateX(), 0)
        decl = mc.GateDecl(name="MyGate", arguments=(theta,), circuit=decl_circuit)

        # Use the GateDecl in a circuit
        main_circuit = mc.Circuit()
        main_circuit.push(mc.GateH(), 0)
        main_circuit.push(decl(np.pi / 2), 1)
        main_circuit.push(decl(np.pi / 4), 2)

        proto_circuit = toproto_circuit(main_circuit)
        restored_circuit = fromproto_circuit(proto_circuit)

        assert len(restored_circuit.instructions) == len(main_circuit.instructions)

        # Check first instruction (H gate)
        assert isinstance(restored_circuit.instructions[0].operation, mc.GateH)

        # Check second and third instructions (GateCall)
        assert isinstance(restored_circuit.instructions[1].operation, mc.GateCall)
        assert isinstance(restored_circuit.instructions[2].operation, mc.GateCall)

        # Check that the declarations refer to the same gate type
        assert restored_circuit.instructions[1].operation._decl.name == "MyGate"
        assert restored_circuit.instructions[2].operation._decl.name == "MyGate"

        # Check that the parameter values were preserved
        assert (
            abs(restored_circuit.instructions[1].operation._args[0] - np.pi / 2) < 1e-10
        )
        assert (
            abs(restored_circuit.instructions[2].operation._args[0] - np.pi / 4) < 1e-10
        )


if __name__ == "__main__":
    pytest.main(["-v", "test_proto_conversion.py"])
