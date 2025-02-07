#
# Copyright © 2022-2024 University of Strasbourg. All Rights Reserved.
# Copyright © 2032-2024 QPerfect. All Rights Reserved.
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

import mimiqcircuits as mc
import symengine as se
from fractions import Fraction
from mimiqcircuits.proto import circuit_pb2
import numpy as np

# Forward mapping: mimiqcircuits -> protobuf
GATEMAP = {
    mc.GateID: circuit_pb2.GateType.GateID,
    mc.GateX: circuit_pb2.GateType.GateX,
    mc.GateY: circuit_pb2.GateType.GateY,
    mc.GateZ: circuit_pb2.GateType.GateZ,
    mc.GateH: circuit_pb2.GateType.GateH,
    mc.GateHXY: circuit_pb2.GateType.GateHXY,
    mc.GateHYZ: circuit_pb2.GateType.GateHYZ,
    mc.GateS: circuit_pb2.GateType.GateS,
    mc.GateT: circuit_pb2.GateType.GateT,
    mc.Delay: circuit_pb2.GateType.Delay,
    mc.GateU: circuit_pb2.GateType.GateU,
    mc.GateP: circuit_pb2.GateType.GateP,
    mc.GateRX: circuit_pb2.GateType.GateRX,
    mc.GateRY: circuit_pb2.GateType.GateRY,
    mc.GateRZ: circuit_pb2.GateType.GateRZ,
    mc.GateR: circuit_pb2.GateType.GateR,
    mc.GateU1: circuit_pb2.GateType.GateU1,
    mc.GateU2: circuit_pb2.GateType.GateU2,
    mc.GateU3: circuit_pb2.GateType.GateU3,
    mc.GateSWAP: circuit_pb2.GateType.GateSWAP,
    mc.GateISWAP: circuit_pb2.GateType.GateISWAP,
    mc.GateECR: circuit_pb2.GateType.GateECR,
    mc.GateDCX: circuit_pb2.GateType.GateDCX,
    mc.GateRXX: circuit_pb2.GateType.GateRXX,
    mc.GateRYY: circuit_pb2.GateType.GateRYY,
    mc.GateRZZ: circuit_pb2.GateType.GateRZZ,
    mc.GateRZX: circuit_pb2.GateType.GateRZX,
    mc.GateXXplusYY: circuit_pb2.GateType.GateXXplusYY,
    mc.GateXXminusYY: circuit_pb2.GateType.GateXXminusYY,
}

# Reverse mapping: protobuf -> mimiqcircuits
REVERSE_GATEMAP = {v: k for k, v in GATEMAP.items()}

# Operation mapping: mimiqcircuits -> protobuf
OPERATIONMAP = {
    mc.MeasureX: circuit_pb2.OperationType.MeasureX,
    mc.MeasureY: circuit_pb2.OperationType.MeasureY,
    mc.MeasureZ: circuit_pb2.OperationType.MeasureZ,
    mc.Measure: circuit_pb2.OperationType.MeasureZ,
    mc.MeasureXX: circuit_pb2.OperationType.MeasureXX,
    mc.MeasureYY: circuit_pb2.OperationType.MeasureYY,
    mc.MeasureZZ: circuit_pb2.OperationType.MeasureZZ,
    mc.MeasureResetX: circuit_pb2.OperationType.MeasureResetX,
    mc.MeasureResetY: circuit_pb2.OperationType.MeasureResetY,
    mc.MeasureReset: circuit_pb2.OperationType.MeasureResetZ,
    mc.BondDim: circuit_pb2.OperationType.BondDim,
    mc.SchmidtRank: circuit_pb2.OperationType.SchmidtRank,
    mc.VonNeumannEntropy: circuit_pb2.OperationType.VonNeumannEntropy,
    mc.Not: circuit_pb2.OperationType.Not,
}

# Reverse operation mapping: protobuf -> mimiqcircuits
REVERSE_OPERATIONMAP = {v: k for k, v in OPERATIONMAP.items()}

# Kraus Channel mapping: mimiqcircuits -> protobuf
KRAUSCHANNELMAP = {
    mc.Reset: circuit_pb2.KrausChannelType.ResetZ,
    mc.ResetZ: circuit_pb2.KrausChannelType.ResetZ,
    mc.ResetX: circuit_pb2.KrausChannelType.ResetX,
    mc.ResetY: circuit_pb2.KrausChannelType.ResetY,
    mc.ResetZ: circuit_pb2.KrausChannelType.ResetZ,
    mc.AmplitudeDamping: circuit_pb2.KrausChannelType.AmplitudeDamping,
    mc.GeneralizedAmplitudeDamping: circuit_pb2.KrausChannelType.GeneralizedAmplitudeDamping,
    mc.PhaseAmplitudeDamping: circuit_pb2.KrausChannelType.PhaseAmplitudeDamping,
    mc.ThermalNoise: circuit_pb2.KrausChannelType.ThermalNoise,
    mc.PauliX: circuit_pb2.KrausChannelType.PauliX,
    mc.PauliY: circuit_pb2.KrausChannelType.PauliY,
    mc.PauliZ: circuit_pb2.KrausChannelType.PauliZ,
    mc.ProjectiveNoiseX: circuit_pb2.KrausChannelType.ProjectiveNoiseX,
    mc.ProjectiveNoiseY: circuit_pb2.KrausChannelType.ProjectiveNoiseY,
    mc.ProjectiveNoiseZ: circuit_pb2.KrausChannelType.ProjectiveNoiseZ,
    mc.ProjectiveNoise: circuit_pb2.KrausChannelType.ProjectiveNoiseZ,
}

# Reverse Kraus Channel mapping: protobuf -> mimiqcircuits
REVERSE_KRAUSCHANNELMAP = {v: k for k, v in KRAUSCHANNELMAP.items()}


# Forward mapping: mimiqcircuits -> protobuf
OPERATORMAP = {
    mc.SigmaMinus: circuit_pb2.OperatorType.SigmaMinus,
    mc.SigmaPlus: circuit_pb2.OperatorType.SigmaPlus,
    mc.Projector0: circuit_pb2.OperatorType.Projector0,
    mc.Projector1: circuit_pb2.OperatorType.Projector1,
    mc.Projector00: circuit_pb2.OperatorType.Projector00,
    mc.Projector01: circuit_pb2.OperatorType.Projector01,
    mc.Projector10: circuit_pb2.OperatorType.Projector10,
    mc.Projector11: circuit_pb2.OperatorType.Projector11,
    mc.ProjectorX0: circuit_pb2.OperatorType.ProjectorX0,
    mc.ProjectorX1: circuit_pb2.OperatorType.ProjectorX1,
    mc.ProjectorY0: circuit_pb2.OperatorType.ProjectorY0,
    mc.ProjectorY1: circuit_pb2.OperatorType.ProjectorY1,
    mc.DiagonalOp: circuit_pb2.OperatorType.DiagonalOp,
}

# Reverse mapping: protobuf -> mimiqcircuits
REVERSE_OPERATORMAP = {v: k for k, v in OPERATORMAP.items()}


GENERALIZED_GATE_MAP = {
    "QFT": mc.QFT,
    "PhaseGradient": mc.PhaseGradient,
    "PolynomialOracle": mc.PolynomialOracle,
    "Diffusion": mc.Diffusion,
}

EXPRSYMENGINE = {
    se.Add: circuit_pb2.FunctionType.ADD,
    se.Mul: circuit_pb2.FunctionType.MUL,
    se.Pow: circuit_pb2.FunctionType.POW,
    se.sin: circuit_pb2.FunctionType.SIN,
    se.cos: circuit_pb2.FunctionType.COS,
    se.tan: circuit_pb2.FunctionType.TAN,
    se.log: circuit_pb2.FunctionType.LOG,
}

IRRATIONALSYMENGINE = {
    se.pi: circuit_pb2.Irrational.PI,
    se.E: circuit_pb2.Irrational.EULER,
}


# Annotation mapping: mimiqcircuits -> protobuf
ANNOTATIONMAP = {
    mc.QubitCoordinates: circuit_pb2.AnnotationType.QubitCoordinates,
    mc.ShiftCoordinates: circuit_pb2.AnnotationType.ShiftCoordinates,
    mc.Tick: circuit_pb2.AnnotationType.Tick,
    
}

# Reverse annotation mapping: protobuf -> mimiqcircuits
REVERSE_ANNOTATIONMAP = {v: k for k, v in ANNOTATIONMAP.items()}


def toproto(param):
    arg = circuit_pb2.Arg()

    def safe_convert(value):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    if isinstance(param, int):
        arg.argvalue_value.integer_value = param

    elif isinstance(param, float):
        param = safe_convert(param)
        if isinstance(param, int):
            arg.argvalue_value.integer_value = param
        else:
            arg.argvalue_value.double_value = param

    elif isinstance(param, se.Integer):
        arg.argvalue_value.integer_value = param.p

    elif isinstance(param, (se.Rational, se.Float)):
        param = safe_convert(float(param))
        if isinstance(param, int):
            arg.argvalue_value.integer_value = param
        else:
            arg.argvalue_value.double_value = param

    elif isinstance(param, se.RealDouble):
        arg.argvalue_value.double_value = param.real

    elif isinstance(param, se.Symbol):
        arg.symbol_value.value = param.name

    elif param in IRRATIONALSYMENGINE:
        arg.irrational_value = IRRATIONALSYMENGINE[param]

    elif type(param) in EXPRSYMENGINE:
        arg.argfunction_value.mtype = EXPRSYMENGINE[type(param)]

        for a in param.args:
            arg.argfunction_value.args.append(toproto(a))

    elif isinstance(param, complex):
        complex_arg = circuit_pb2.ComplexArg()
        real_arg = circuit_pb2.Arg()
        imag_arg = circuit_pb2.Arg()
        real_arg.argvalue_value.double_value = param.real
        imag_arg.argvalue_value.double_value = param.imag
        complex_arg.real.CopyFrom(real_arg)
        complex_arg.imag.CopyFrom(imag_arg)
        arg.argcomplex_value.CopyFrom(complex_arg)

    elif isinstance(param, se.Basic):
        if param == (se.I):
            evaluated = param.evalf()
            real_part = float(evaluated.real)
            imag_part = float(evaluated.imag)
            complex_arg = circuit_pb2.ComplexArg()
            real_arg = circuit_pb2.Arg()
            imag_arg = circuit_pb2.Arg()
            real_arg.argvalue_value.double_value = real_part
            imag_arg.argvalue_value.double_value = imag_part
            complex_arg.real.CopyFrom(real_arg)
            complex_arg.imag.CopyFrom(imag_arg)
            arg.argcomplex_value.CopyFrom(complex_arg)
        else:
            arg.symbol_value.value = str(param)
    else:
        raise ValueError(f"Unsupported parameter type: {type(param)}")

    return arg


EXPRSPROTO = {v: k for k, v in EXPRSYMENGINE.items()}
IRRATIONALPROTO = {v: k for k, v in IRRATIONALSYMENGINE.items()}


def fromproto(arg):
    def safe_convert(value):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    if arg.HasField("argvalue_value"):
        av = arg.argvalue_value
        if av.HasField("integer_value"):
            return av.integer_value
        elif av.HasField("double_value"):
            return safe_convert(av.double_value)

    elif arg.HasField("irrational_value"):
        return IRRATIONALPROTO[arg.irrational_value]

    elif arg.HasField("symbol_value"):
        return se.Symbol(arg.symbol_value.value)

    elif arg.HasField("argfunction_value"):
        ftype = arg.argfunction_value.mtype
        params = list(map(fromproto, arg.argfunction_value.args))

        if ftype in EXPRSPROTO:
            return EXPRSPROTO[ftype](*params)

        elif ftype == circuit_pb2.FunctionType.DIV:
            return se.Mul(params[0], se.Pow(params[1], -1))

        elif ftype == circuit_pb2.FunctionType.EXP:
            return se.Pow(se.E, params[0])

        else:
            raise ValueError("Unsupported function type in protocol buffer")

    elif arg.HasField("argcomplex_value"):
        real = fromproto(arg.argcomplex_value.real)
        imag = fromproto(arg.argcomplex_value.imag)
        return complex(real, imag)

    else:
        raise ValueError("Unsupported parameter type in protocol buffer")


def toproto_complex(value):
    if hasattr(value, "real") and hasattr(value, "imag"):
        # If the value has real and imaginary attributes, use them
        return circuit_pb2.ComplexArg(real=toproto(value.real), imag=toproto(value.imag))
    else:
        # Otherwise, treat it as a symbolic expression with the imaginary part as zero
        return circuit_pb2.ComplexArg(real=toproto(value), imag=toproto(0))

def fromproto_complex(complex_proto):
    real = fromproto(complex_proto.real)
    imag = fromproto(complex_proto.imag)

    # Only return the real part if the imaginary part is zero
    if imag == 0:
        return real
    return complex(real, imag)

def fromproto_generalized_gate(generalized_proto):
    name = generalized_proto.name
    args = [fromproto(arg) for arg in generalized_proto.args]
    qregsizes = list(generalized_proto.qregsizes)  
    if name in GENERALIZED_GATE_MAP:
        return GENERALIZED_GATE_MAP[name](*qregsizes, *args)

    else:
        raise ValueError(f"Unsupported generalized gate name: {name}")

# Gate conversion functions
def toproto_gate(gate):
    gate_type_enum = GATEMAP.get(type(gate))
    if gate_type_enum is not None:
        params = [toproto(param) for param in gate.getparams()]
        return circuit_pb2.Gate(
            simplegate=circuit_pb2.SimpleGate(mtype=gate_type_enum, parameters=params)
        )

    # Generalized conversion functions
    if hasattr(gate, "name") and gate.name in GENERALIZED_GATE_MAP:
        args = [toproto(param) for param in gate.getparams()]
        
        # Special Case: PolynomialOracle (Ensures Correct Parameter Order)
        if isinstance(gate, mc.PolynomialOracle):
            args = [toproto(arg) for arg in gate._params[2:]]  # Skip nx, ny
        
        return circuit_pb2.Gate(
            generalized=circuit_pb2.Generalized(
                name=gate.name,
                args=args,  # Processed params
                qregsizes=list(getattr(gate, "_qregsizes", [])),  # Ensure qregsizes exist
            )
        )

    # Check if the gate is a PauliString
    if isinstance(gate, mc.PauliString):
        return circuit_pb2.Gate(
            paulistring=circuit_pb2.PauliString(
                numqubits=gate.num_qubits, pauli=gate.pauli
            )
        )

    # Check if the gate is a custom gate
    if isinstance(gate, mc.GateCustom):
        U = [toproto_complex(param) for param in gate.matrix.T]
        return circuit_pb2.Gate(
            customgate=circuit_pb2.CustomGate(numqubits=gate.num_qubits, matrix=U)
        )

    # Check if the gate is a GateDecl
    if isinstance(gate, mc.GateDecl):
        instructions_proto = list(map(toproto_instruction, gate.circuit.instructions))
        arguments = [circuit_pb2.Symbol(value=arg) for arg in gate.arguments]
        return circuit_pb2.GateDecl(
            name=gate.name, args=arguments, instructions=instructions_proto
        )

    # Check if the gate is a GateCall
    if isinstance(gate, mc.GateCall):
        decl_proto = toproto_gate(gate._decl)
        args_proto = [toproto(arg) for arg in gate._args]
        return circuit_pb2.Gate(
            gatecall=circuit_pb2.GateCall(decl=decl_proto, args=args_proto)
        )

    elif isinstance(gate, mc.Control):
        control_gate_proto = toproto_gate(gate.op)
        return circuit_pb2.Gate(
            control=circuit_pb2.Control(
                operation=control_gate_proto, numcontrols=gate.num_controls
            )
        )

    elif isinstance(gate, mc.Power):
        base_gate_proto = toproto_gate(gate.op)

        # Serialize the exponent using the appropriate field
        if isinstance(gate.exponent, int):
            power_proto = circuit_pb2.Power(
                operation=base_gate_proto, int_val=gate.exponent
            )
        elif isinstance(gate.exponent, Fraction):
            power_proto = circuit_pb2.Power(
                operation=base_gate_proto,
                rational_val=circuit_pb2.Rational(
                    num=gate.exponent.numerator, den=gate.exponent.denominator
                ),
            )
        elif isinstance(gate.exponent, float):
            power_proto = circuit_pb2.Power(
                operation=base_gate_proto, double_val=gate.exponent
            )
        else:
            raise ValueError(f"Unsupported exponent type: {type(gate.exponent)}")

        return circuit_pb2.Gate(power=power_proto)
    elif isinstance(gate, mc.Inverse):
        base_gate_proto = toproto_gate(gate.op)
        return circuit_pb2.Gate(inverse=circuit_pb2.Inverse(operation=base_gate_proto))
    elif isinstance(gate, mc.Parallel):
        base_gate_proto = toproto_gate(gate.op)
        return circuit_pb2.Gate(
            parallel=circuit_pb2.Parallel(
                operation=base_gate_proto, numrepeats=gate.num_repeats
            )
        )
    else:
        raise ValueError(f"Unsupported gate type: {type(gate)}")


def fromproto_gatedecl(gatedecl_proto):
    name = gatedecl_proto.name
    arguments = [str(symbol.value) for symbol in gatedecl_proto.args]
    instructions = [fromproto_instruction(inst) for inst in gatedecl_proto.instructions]

    return mc.GateDecl(name=name, arguments=tuple(arguments), circuit=mc.Circuit(instructions))


def fromproto_gate(gate_proto):
    if gate_proto.HasField("simplegate"):
        gate_class = REVERSE_GATEMAP.get(gate_proto.simplegate.mtype)
        if gate_class:
            params = [fromproto(param) for param in gate_proto.simplegate.parameters]
            return gate_class(*params)
        raise ValueError(
            f"Unsupported gate type in proto: {gate_proto.simplegate.mtype}"
        )

    if gate_proto.HasField("paulistring"):
        return mc.PauliString(gate_proto.paulistring.pauli)

    if gate_proto.HasField("customgate"):
        U_matrix = [fromproto_complex(val) for val in gate_proto.customgate.matrix]
        matrix_size = 2**gate_proto.customgate.numqubits 
        original_matrix = np.array(U_matrix).reshape(matrix_size, matrix_size).T
        return mc.GateCustom(matrix=original_matrix)
    
    elif gate_proto.HasField("generalized"):
        return fromproto_generalized_gate(gate_proto.generalized)

    elif gate_proto.HasField("control"):
        control_gate = fromproto_gate(gate_proto.control.operation)
        return mc.Control(gate_proto.control.numcontrols, control_gate)

    elif gate_proto.HasField("power"):
        base_gate = fromproto_gate(gate_proto.power.operation)

        if gate_proto.power.HasField("int_val"):
            power_value = gate_proto.power.int_val
        elif gate_proto.power.HasField("rational_val"):
            power_value = Fraction(
                gate_proto.power.rational_val.num, gate_proto.power.rational_val.den
            )
        elif gate_proto.power.HasField("double_val"):
            power_value = gate_proto.power.double_val
        else:
            raise ValueError("Unsupported power value type in proto")

        return mc.Power(base_gate, power_value)
    elif gate_proto.HasField("inverse"):
        base_gate = fromproto_gate(gate_proto.inverse.operation)

        return mc.Inverse(base_gate)
    elif gate_proto.HasField("parallel"):
        base_gate = fromproto_gate(gate_proto.parallel.operation)
        return mc.Parallel(gate_proto.parallel.numrepeats, base_gate)

    elif gate_proto.HasField("gatecall"):
        # Deserialize the GateDecl using the correct function
        decl = fromproto_gatedecl(gate_proto.gatecall.decl)
        args = [fromproto(arg) for arg in gate_proto.gatecall.args]
        return mc.GateCall(decl, args)

    else:
        raise ValueError(
            f"Unsupported gate type in proto: {gate_proto.WhichOneof('gate')}"
        )


def toproto_operator(operator):
    if isinstance(operator, mc.AbstractOperator):
        operator_type = OPERATORMAP.get(type(operator))
        if operator_type is not None:
            params = [toproto(param) for param in operator.getparams()]
            return circuit_pb2.Operator(
                simpleoperator=circuit_pb2.SimpleOperator(
                    mtype=operator_type, parameters=params
                )
            )

        # Handle RescaledGate
        elif isinstance(operator, mc.RescaledGate):
            base_gate_proto = toproto_gate(operator.get_operation())
            scale_proto = toproto(operator.get_scale())
            return circuit_pb2.Operator(
                rescaledgate=circuit_pb2.RescaledGate(
                    operation=base_gate_proto, scale=scale_proto
                )
            )

        # Handle gates directly by using toproto_gate
        elif isinstance(operator, mc.Gate):
            gate_proto = toproto_gate(operator)
            return circuit_pb2.Operator(
                **{
                    gate_proto.WhichOneof("gate"): getattr(
                        gate_proto, gate_proto.WhichOneof("gate")
                    )
                }
            )

        # Handle custom operators
        elif isinstance(operator, mc.Operator):
            matrix_data = (
                operator.matrix() if callable(operator.matrix) else operator.matrix
            )
            # Manually flatten the matrix using list comprehension
            matrix = [
                toproto_complex(matrix_data[i, j])
                for i in range(matrix_data.rows)
                for j in range(matrix_data.cols)
            ]
            return circuit_pb2.Operator(
                customoperator=circuit_pb2.CustomOperator(
                    numqubits=operator.num_qubits, matrix=matrix
                )
            )
        else:
            raise ValueError(f"Unsupported Operator type: {type(operator)}")
    else:
        raise ValueError(f"Unsupported Operator type: {type(operator)}")


def fromproto_operator(operator_proto):
    # Handle simple operators
    if operator_proto.HasField("simpleoperator"):
        operator_class = REVERSE_OPERATORMAP.get(operator_proto.simpleoperator.mtype)
        if operator_class:
            params = [
                fromproto(param) for param in operator_proto.simpleoperator.parameters
            ]
            return operator_class(*params)

    # Handle rescaled gates
    elif operator_proto.HasField("rescaledgate"):
        base_gate = fromproto_gate(operator_proto.rescaledgate.operation)
        scale = fromproto(operator_proto.rescaledgate.scale)
        return mc.RescaledGate(base_gate, scale)

    # Handle custom operators
    elif operator_proto.HasField("customoperator"):
        matrix = [
            fromproto_complex(val) for val in operator_proto.customoperator.matrix
        ]
        matrix = np.array(matrix).reshape(
            2**operator_proto.customoperator.numqubits,
            2**operator_proto.customoperator.numqubits,
        )
        return mc.Operator(mat=matrix)

    # Handle gates stored within an operator
    elif (
        operator_proto.HasField("simplegate")
        or operator_proto.HasField("customgate")
        or operator_proto.HasField("control")
        or operator_proto.HasField("power")
        or operator_proto.HasField("inverse")
        or operator_proto.HasField("parallel")
        or operator_proto.HasField("paulistring")
    ):
        return fromproto_gate(
            circuit_pb2.Gate(
                **{
                    operator_proto.WhichOneof("operator"): getattr(
                        operator_proto, operator_proto.WhichOneof("operator")
                    )
                }
            )
        )

    else:
        raise ValueError(
            f"Unsupported Operator type in proto: {operator_proto.WhichOneof('operator')}"
        )


def toproto_krauschannel(kraus_channel):
    kraus_channel_type = KRAUSCHANNELMAP.get(type(kraus_channel))
    if kraus_channel_type is not None:
        params = [toproto(param) for param in kraus_channel.getparams()]
        return circuit_pb2.KrausChannel(
            simplekrauschannel=circuit_pb2.SimpleKrausChannel(
                mtype=kraus_channel_type, parameters=params
            )
        )

    if isinstance(kraus_channel, mc.Kraus):
        operators_proto = []
        for op in kraus_channel.krausoperators():
            matrix = op.matrix()

            matrix_elements = []
            # Get the correct matrix dimensions
            num_rows, num_cols = matrix.rows, matrix.cols

            for i in range(num_rows):
                for j in range(num_cols):
                    element = matrix[i, j]
                    matrix_elements.append(toproto_complex(element))

            operators_proto.append(
                circuit_pb2.Operator(
                    customoperator=circuit_pb2.CustomOperator(
                        numqubits=kraus_channel.num_qubits, matrix=matrix_elements
                    )
                )
            )
        return circuit_pb2.KrausChannel(
            customkrauschannel=circuit_pb2.CustomKrausChannel(
                numqubits=kraus_channel.num_qubits, operators=operators_proto
            )
        )

    elif isinstance(kraus_channel, mc.MixedUnitary):
        operators_proto = []
        gates_or_matrices = (
            kraus_channel.unitarygates()
            if hasattr(kraus_channel, "unitarygates")
            else kraus_channel.unitarymatrices()
        )
        probabilities = kraus_channel.probabilities()

        for gate_or_matrix, scale in zip(gates_or_matrices, probabilities):
            gate_proto = toproto_gate(gate_or_matrix)
            scale_proto = toproto(se.sqrt(scale))
            operators_proto.append(
                circuit_pb2.RescaledGate(operation=gate_proto, scale=scale_proto)
            )
        return circuit_pb2.KrausChannel(
            mixedunitarychannel=circuit_pb2.MixedUnitaryChannel(
                operators=operators_proto
            )
        )

    # Add conversion for PauliNoise
    elif isinstance(kraus_channel, mc.PauliNoise):
        # Call probabilities as a method
        probabilities_proto = [toproto(prob) for prob in kraus_channel.probabilities()]
        pauli_strings_proto = [
            circuit_pb2.PauliString(
                numqubits=pauli_str.num_qubits, pauli=pauli_str.pauli
            )
            for pauli_str in kraus_channel.paulistr
        ]
        return circuit_pb2.KrausChannel(
            paulichannel=circuit_pb2.PauliChannel(
                probabilities=probabilities_proto, paulistrings=pauli_strings_proto
            )
        )
    if isinstance(kraus_channel, mc.Depolarizing):
        return toproto_depolarizing(kraus_channel)

    else:
        raise ValueError(f"Unsupported KrausChannel type: {type(kraus_channel)}")


def fromproto_krauschannel(kraus_proto):
    # Handle SimpleKrausChannel type
    if kraus_proto.HasField("simplekrauschannel"):
        kraus_class = REVERSE_KRAUSCHANNELMAP.get(kraus_proto.simplekrauschannel.mtype)
        if kraus_class:
            params = [
                fromproto(param) for param in kraus_proto.simplekrauschannel.parameters
            ]
            return kraus_class(*params)
        raise ValueError(
            f"Unsupported KrausChannel type in proto: {kraus_proto.simplekrauschannel.mtype}"
        )

    elif kraus_proto.HasField("customkrauschannel"):
        operators = []
        for op_proto in kraus_proto.customkrauschannel.operators:
            if op_proto.HasField("customoperator"):
                # Deserialize matrix elements from the CustomOperator
                matrix_elements = [
                    fromproto_complex(val) for val in op_proto.customoperator.matrix
                ]
                num_qubits = op_proto.customoperator.numqubits
                matrix_size = 2**num_qubits

                # Check if the number of elements matches the expected matrix size
                if len(matrix_elements) != matrix_size * matrix_size:
                    raise ValueError(
                        f"Number of elements ({len(matrix_elements)}) does not match expected size ({matrix_size * matrix_size})."
                    )

                # Reshape the list of elements into a symengine matrix
                matrix = se.Matrix(matrix_size, matrix_size, matrix_elements)
                operators.append(mc.Operator(mat=matrix))

            elif op_proto.HasField("simpleoperator"):
                # Handle SimpleOperator by mapping its type and parameters
                operator_class = REVERSE_OPERATORMAP.get(op_proto.simpleoperator.mtype)
                if operator_class:
                    params = [
                        fromproto(param) for param in op_proto.simpleoperator.parameters
                    ]
                    operators.append(operator_class(*params))
                else:
                    raise ValueError(
                        f"Unsupported SimpleOperator type: {op_proto.simpleoperator.mtype}"
                    )
            else:
                raise ValueError("Unsupported operator type in customkrauschannel")

        return mc.Kraus(operators)

    elif kraus_proto.HasField("mixedunitarychannel"):
        probabilities = []
        unitary_matrices = []

        for rescaled_gate_proto in kraus_proto.mixedunitarychannel.operators:
            scale = fromproto(rescaled_gate_proto.scale)
            probabilities.append((scale**2))

            # if rescaled_gate_proto.operation.HasField("customgate"):
            #     matrix_elements = [fromproto_complex(
            #         val) for val in rescaled_gate_proto.operation.customgate.matrix]
            #     num_qubits = rescaled_gate_proto.operation.customgate.numqubits
            #     matrix_size = 2**num_qubits
            #     matrix = se.Matrix(matrix_size, matrix_size, matrix_elements)
            #     unitary_matrices.append(matrix)
            # else:
            gate = fromproto_gate(rescaled_gate_proto.operation)
            unitary_matrices.append(gate)

        return mc.MixedUnitary(probabilities, unitary_matrices)

    # Handle PauliNoise
    elif kraus_proto.HasField("paulichannel"):
        probabilities = [
            fromproto(prob) for prob in kraus_proto.paulichannel.probabilities
        ]
        pauli_strings = [
            mc.PauliString(pauli_str.pauli)
            for pauli_str in kraus_proto.paulichannel.paulistrings
        ]
        return mc.PauliNoise(probabilities, pauli_strings)

    elif kraus_proto.HasField("depolarizingchannel"):
        return fromproto_depolarizing(kraus_proto.depolarizingchannel)
    else:
        raise ValueError(
            f"Unsupported KrausChannel type in proto: {kraus_proto.WhichOneof('krauschannel')}"
        )


def toproto_operation(operation):
    operation_type = OPERATIONMAP.get(type(operation))
    if operation_type is not None:
        return circuit_pb2.Operation(
            simpleoperation=circuit_pb2.SimpleOperation(mtype=operation_type)
        )

    # Check if the operation is a Kraus channel
    if isinstance(operation, mc.krauschannel):
        kraus_proto = toproto_krauschannel(operation)
        kraus_field_name = kraus_proto.WhichOneof("krauschannel")
        return circuit_pb2.Operation(
            **{kraus_field_name: getattr(kraus_proto, kraus_field_name)}
        )

    # Check if the operation is an operator
    if isinstance(operation, mc.AbstractOperator):
        # Check if the operation is a gate
        if isinstance(operation, mc.Gate):
            gate_proto = toproto_gate(operation)
            gate_field_name = gate_proto.WhichOneof("gate")
            return circuit_pb2.Operation(
                **{gate_field_name: getattr(gate_proto, gate_field_name)}
            )
        else:
            operator_proto = toproto_operator(operation)
            operator_field_name = operator_proto.WhichOneof("operator")
            return circuit_pb2.Operation(
                **{operator_field_name: getattr(operator_proto, operator_field_name)}
            )

    # Handle Amplitude
    if isinstance(operation, mc.Amplitude):
        from mimiqcircuits.proto.bitvector import toproto_bitvector

        bitstring_proto = toproto_bitvector(operation.bs)
        return circuit_pb2.Operation(
            amplitude=circuit_pb2.Amplitude(bs=bitstring_proto)
        )

    # Handle ExpectationValue
    if isinstance(operation, mc.ExpectationValue):
        operator_proto = toproto_operator(operation.op)
        return circuit_pb2.Operation(
            expectationvalue=circuit_pb2.ExpectationValue(operator=operator_proto)
        )

    # Handle Barrier
    if isinstance(operation, mc.Barrier):
        return circuit_pb2.Operation(
            barrier=circuit_pb2.Barrier(numqubits=operation.num_qubits)
        )

    # Handle IfStatement operation
    if isinstance(operation, mc.IfStatement):
        from mimiqcircuits.proto.bitvector import toproto_bitvector

        bitstring_proto = toproto_bitvector(operation.bitstring)
        operation_proto = toproto_operation(operation.op)
        return circuit_pb2.Operation(
            ifstatement=circuit_pb2.IfStatement(
                operation=operation_proto, bitstring=bitstring_proto
            )
        )

    # Handle Detector
    if isinstance(operation, mc.Detector):
        notes = [toproto_note(note) for note in operation.get_notes()]
        return circuit_pb2.Operation(
            detector=circuit_pb2.Detector(
                numqubits=operation._num_bits, notes=notes
            )
        )


    # Handle ObservableInclude
    if isinstance(operation, mc.ObservableInclude):
        notes = [toproto_note(note) for note in operation.get_notes()]
        return circuit_pb2.Operation(
            observableinc=circuit_pb2.ObservableInclude(
                numbits=operation._num_bits, notes=notes
            )
        )


    # Handle annotations like Tick, ShiftCoordinates, etc.
    annotation_type = ANNOTATIONMAP.get(type(operation))
    if annotation_type is not None:
        notes = [toproto_note(note) for note in operation.get_notes()]
        return circuit_pb2.Operation(
            simpleannotation=circuit_pb2.SimpleAnnotation(
                mtype=annotation_type, notes=notes
            )
        )

    # Handle Not operation
    if isinstance(operation, mc.Not):
        return circuit_pb2.Operation(
            simpleoperation=circuit_pb2.SimpleOperation(mtype=circuit_pb2.OperationType.Not)
        )
    
    raise ValueError(f"Unsupported operation type: {type(operation)}")


def fromproto_operation(operation_proto):
    if operation_proto.HasField("simpleoperation"):
        operation_type = operation_proto.simpleoperation.mtype
        operation_class = REVERSE_OPERATIONMAP.get(operation_type)
        if operation_class:
            return operation_class()
        else:
            raise ValueError(f"Unsupported SimpleOperation type: {operation_type}")

    # Handle DepolarizingChannel
    elif operation_proto.HasField("depolarizingchannel"):
        depolarizing_proto = operation_proto.depolarizingchannel
        return mc.Depolarizing(depolarizing_proto.numqubits, fromproto(depolarizing_proto.probability))

    # Handle gate operations
    elif (
        operation_proto.HasField("simplegate")
        or operation_proto.HasField("customgate")
        or operation_proto.HasField("control")
        or operation_proto.HasField("power")
        or operation_proto.HasField("inverse")
        or operation_proto.HasField("parallel")
        or operation_proto.HasField("paulistring")
        or operation_proto.HasField("gatecall")
    ):
        return fromproto_gate(
            circuit_pb2.Gate(
                **{
                    operation_proto.WhichOneof("operation"): getattr(
                        operation_proto, operation_proto.WhichOneof("operation")
                    )
                }
            )
        )

    # Handle Kraus channels, including MixedUnitary
    elif (
        operation_proto.HasField("simplekrauschannel")
        or operation_proto.HasField("customkrauschannel")
        or operation_proto.HasField("mixedunitarychannel")
        or operation_proto.HasField("paulichannel")
    ):
        return fromproto_krauschannel(
            circuit_pb2.KrausChannel(
                **{
                    operation_proto.WhichOneof("operation"): getattr(
                        operation_proto, operation_proto.WhichOneof("operation")
                    )
                }
            )
        )

    # Handle operators
    elif (
        operation_proto.HasField("simpleoperator")
        or operation_proto.HasField("customoperator")
        or operation_proto.HasField("rescaledgate")
    ):
        return fromproto_operator(
            circuit_pb2.Operator(
                **{
                    operation_proto.WhichOneof("operation"): getattr(
                        operation_proto, operation_proto.WhichOneof("operation")
                    )
                }
            )
        )

    # Handle Amplitude operation
    elif operation_proto.HasField("amplitude"):
        from mimiqcircuits.proto.bitvector import fromproto_bitvector

        bitstring = fromproto_bitvector(operation_proto.amplitude.bs)
        return mc.Amplitude(bitstring)

    # Handle ExpectationValue operation
    elif operation_proto.HasField("expectationvalue"):
        operator = fromproto_operator(operation_proto.expectationvalue.operator)
        return mc.ExpectationValue(operator)

    # Handle Barrier operation
    elif operation_proto.HasField("barrier"):
        return mc.Barrier(operation_proto.barrier.numqubits)

    # Handle IfStatement operation
    elif operation_proto.HasField("ifstatement"):
        operation = fromproto_operation(operation_proto.ifstatement.operation)
        from mimiqcircuits.proto.bitvector import fromproto_bitvector

        bitstring = mc.BitString(
            fromproto_bitvector(operation_proto.ifstatement.bitstring)
        )
        return mc.IfStatement(operation=operation, bitstring=bitstring)

    # Handle Detector
    elif operation_proto.HasField("detector"):
        # this is the number of classical bits
        num_bits = operation_proto.detector.numqubits
        # Convert Note objects to float values
        notes = [float(note.double_note) if note.HasField("double_note") else float(note.int_note) for note in operation_proto.detector.notes]
        return mc.Detector(num_bits, notes)
    
    # Handle ObservableInclude
    elif operation_proto.HasField("observableinc"):
        num_bits = operation_proto.observableinc.numbits
        notes = [fromproto_note(note) for note in operation_proto.observableinc.notes]
        return mc.ObservableInclude(num_bits, notes)


    # Check if the operation is an annotation
    if operation_proto.HasField("simpleannotation"):
        annotation_class = REVERSE_ANNOTATIONMAP.get(
            operation_proto.simpleannotation.mtype
        )
        if annotation_class is not None:
            notes = [
                fromproto_note(note) for note in operation_proto.simpleannotation.notes
            ]

            # Handle cases where the annotation class expects a list or multiple args
            if annotation_class is mc.Tick:
                return mc.Tick()
            elif annotation_class in {mc.ShiftCoordinates, mc.QubitCoordinates}:
                return annotation_class(notes)

            return annotation_class(*notes)
        
    elif operation_proto.HasField("generalized"):
        return fromproto_generalized_gate(operation_proto.generalized)

    else:
        raise ValueError(
            f"Unsupported operation field in proto: {operation_proto.WhichOneof('operation')}"
        )


def toproto_note(note):
    if isinstance(note, int):
        return circuit_pb2.Note(int_note=note)
    elif isinstance(note, float):
        return circuit_pb2.Note(double_note=note)
    else:
        raise ValueError(f"Unsupported note type: {type(note)}")


def fromproto_note(note_proto):
    if note_proto.HasField("int_note"):
        return float(note_proto.int_note)  # Convert to float for compatibility
    elif note_proto.HasField("double_note"):
        return note_proto.double_note
    else:
        raise ValueError(f"Unsupported Note type in proto: {note_proto}")


def toproto_depolarizing(depolarizing):
    return circuit_pb2.KrausChannel(
        depolarizingchannel=circuit_pb2.DepolarizingChannel(
            numqubits=depolarizing.N,
            probability=toproto(depolarizing.p)
        )
    )

def fromproto_depolarizing(depolarizing_proto):
    num_qubits = depolarizing_proto.numqubits
    probability = fromproto(depolarizing_proto.probability) 
    return mc.Depolarizing(num_qubits, probability)


def toproto_instruction(inst):
    return circuit_pb2.Instruction(
        operation=toproto_operation(inst.operation),
        qtargets=[x + 1 for x in inst.qubits],
        ctargets=[x + 1 for x in inst.bits],
        ztargets=[x + 1 for x in inst.zvars],
    )


def fromproto_instruction(inst_proto):
    return mc.Instruction(
        operation=fromproto_operation(inst_proto.operation),
        qubits=tuple([x - 1 for x in inst_proto.qtargets]),
        bits=tuple([x - 1 for x in inst_proto.ctargets]),
        zvars=tuple([x - 1 for x in inst_proto.ztargets]),
    )


def toproto_circuit(circuit):
    instructions_proto = [toproto_instruction(inst) for inst in circuit.instructions]
    return circuit_pb2.Circuit(instructions=instructions_proto)


def fromproto_circuit(circuit_proto):
    instructions = [fromproto_instruction(inst) for inst in circuit_proto.instructions]
    return mc.Circuit(instructions)
