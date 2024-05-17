#
# Copyright © 2022-2023 University of Strasbourg. All Rights Reserved.
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

from mimiqcircuits.proto import circuit_pb
import mimiqcircuits as mc
import symengine as se
from symengine import *
from fractions import Fraction
import inspect

GATEENUMMAP = {
    "GateU": circuit_pb.GateType.GateU,
    "GateID": circuit_pb.GateType.GateID,
    "GateX": circuit_pb.GateType.GateX,
    "GateY": circuit_pb.GateType.GateY,
    "GateZ": circuit_pb.GateType.GateZ,
    "GateH": circuit_pb.GateType.GateH,
    "GateS": circuit_pb.GateType.GateS,
    "GateT": circuit_pb.GateType.GateT,
    "GateP": circuit_pb.GateType.GateP,
    "GateRX": circuit_pb.GateType.GateRX,
    "GateRY": circuit_pb.GateType.GateRY,
    "GateRZ": circuit_pb.GateType.GateRZ,
    "GateR": circuit_pb.GateType.GateR,
    "GateU1": circuit_pb.GateType.GateU1,
    "GateU2": circuit_pb.GateType.GateU2,
    "GateU3": circuit_pb.GateType.GateU3,
    "GateSWAP": circuit_pb.GateType.GateSWAP,
    "GateISWAP": circuit_pb.GateType.GateISWAP,
    "GateECR": circuit_pb.GateType.GateECR,
    "GateDCX": circuit_pb.GateType.GateDCX,
    "GateRXX": circuit_pb.GateType.GateRXX,
    "GateRYY": circuit_pb.GateType.GateRYY,
    "GateRZZ": circuit_pb.GateType.GateRZZ,
    "GateRZX": circuit_pb.GateType.GateRZX,
    "GateXXplusYY": circuit_pb.GateType.GateXXplusYY,
    "GateXXminusYY": circuit_pb.GateType.GateXXminusYY,
}


def toproto_circuit(circuit):
    instructions_proto = list(map(toproto_instruction, circuit.instructions))
    return circuit_pb.Circuit(instructions=instructions_proto)


def fromproto_circuit(circuit):
    instructions = list(map(fromproto_instruction, circuit.instructions))
    return mc.Circuit(instructions)


def toproto_instruction(inst):
    op = toproto_operation(inst.operation)
    return circuit_pb.Instruction(
        operation=op,
        qtargets=[x+1 for x in inst.qubits],
        ctargets=[x+1 for x in inst.bits]
    )


def fromproto_instruction(inst):
    op = fromproto_operation(inst.operation)
    return mc.Instruction(
        op,
        tuple([x-1 for x in inst.qtargets]),
        tuple([x-1 for x in inst.ctargets])
    )


EXPRSYMENGINE = {
    se.Add: circuit_pb.FunctionType.ADD,
    se.Mul: circuit_pb.FunctionType.MUL,
    se.Pow: circuit_pb.FunctionType.POW,
    se.sin: circuit_pb.FunctionType.SIN,
    se.cos: circuit_pb.FunctionType.COS,
    se.tan: circuit_pb.FunctionType.TAN,
    se.log: circuit_pb.FunctionType.LOG,
}

IRRATIONALSYMENGINE = {
    se.pi: circuit_pb.Irrational.PI,
    se.E: circuit_pb.Irrational.EULER,
}


def toproto_param(param):
    arg = circuit_pb.Arg()

    if isinstance(param, int):
        arg.argvalue_value.integer_value = param

    elif isinstance(param, float):
        arg.argvalue_value.double_value = param

    elif isinstance(param, se.Integer):
        arg.argvalue_value.integer_value = param.p

    elif isinstance(param, (se.Rational, se.Float)):
        arg.argvalue_value.double_value = param

    elif isinstance(param, se.RealDouble):
        arg.argvalue_value.double_value = param.real

    elif isinstance(param, se.Symbol):
        arg.symbol_value.value = param.name

    elif param in IRRATIONALSYMENGINE:
        arg.irrational_value = IRRATIONALSYMENGINE[param]

    elif type(param) in EXPRSYMENGINE:
        arg.argfunction_value.functiontype = EXPRSYMENGINE[type(param)]

        for a in param.args:
            arg.argfunction_value.args.append(toproto_param(a))

    else:
        raise ValueError(f"Unsupported parameter type: {type(param)}")

    return arg


EXPRSPROTO = {v: k for k, v in EXPRSYMENGINE.items()}
IRRATIONALPROTO = {v: k for k, v in IRRATIONALSYMENGINE.items()}


def fromproto_param(arg):
    if arg.HasField("argvalue_value"):
        av = arg.argvalue_value
        if av.HasField("integer_value"):
            return av.integer_value
        elif av.HasField("double_value"):
            return av.double_value

    elif arg.HasField("irrational_value"):
        return IRRATIONALPROTO[arg.irrational_value]

    elif arg.HasField("symbol_value"):
        return se.Symbol(arg.symbol_value.value)

    elif arg.HasField("argfunction_value"):
        ftype = arg.argfunction_value.functiontype
        params = list(map(fromproto_param, arg.argfunction_value.args))

        if ftype in EXPRSPROTO:
            return EXPRSPROTO[ftype](*params)

        elif ftype == circuit_pb.FunctionType.DIV:
            se.Mul(params[0], se.Pow(params[1], -1))

        elif ftype == circuit_pb.FunctionType.EXP:
            se.Pow(se.E, params[0])

        else:
            raise ValueError("Unsupported function type in protocol buffer")

    else:
        raise ValueError("Unsupported parameter type in protocol buffer")


def toproto_gate(gate):
    gate_class_name = gate.__class__.__name__
    gate_type_enum = GATEENUMMAP.get(gate_class_name, None)

    if gate_type_enum is not None:
        params = tuple(getattr(gate, attr) for attr in gate._parnames) if hasattr(
            gate, '_parnames') else ()

        params = list(map(toproto_param, params))

        gate_proto = circuit_pb.Gate(
            gtype=gate_type_enum, parameters=params)
        return gate_proto


def fromproto_gate(gate_proto):
    gate_type_enum = gate_proto.gtype
    gate_class_name = None

    for class_name, enum_value in GATEENUMMAP.items():
        if enum_value == gate_type_enum:
            gate_class_name = class_name
            break

    if gate_class_name is not None:
        gate_class = getattr(mc, gate_class_name, None)
        if gate_class:
            params = list(map(fromproto_param, gate_proto.parameters))
            return gate_class(*params)


def toproto_operation(operation):
    if isinstance(operation, mc.GateCustom):
        return toproto_custom(operation)

    if isinstance(operation, mc.GateCall):
        return toproto_gatecall(operation)

    if isinstance(operation, (mc.QFT, mc.PhaseGradient, mc.Diffusion, mc.PolynomialOracle)):
        return toproto_generalized(operation)

    elif isinstance(operation, mc.Gate):
        gate_proto = toproto_gate(operation)
        return circuit_pb.Operation(gate=gate_proto)

    elif isinstance(operation, mc.Measure):
        return toproto_measure(operation=operation)

    elif isinstance(operation, mc.Barrier):
        return toproto_barrier(operation=operation)

    elif isinstance(operation, mc.Reset):
        return toproto_reset(operation=operation)

    elif isinstance(operation, mc.Control):
        return toproto_control(operation)

    elif isinstance(operation, mc.Parallel):
        return toproto_parallel(operation)

    elif isinstance(operation, mc.Power):
        return toproto_power(operation)

    elif isinstance(operation, mc.Inverse):
        return toproto_inverse(operation)

    elif isinstance(operation, mc.IfStatement):
        return toproto_ifstatement(operation)

    else:
        raise NotImplementedError("Operation not implemented")


def fromproto_operation(operation_proto):

    if operation_proto.HasField("custom"):
        return fromproto_custom(operation_proto.custom)

    if operation_proto.HasField("gate"):
        return fromproto_gate(operation_proto.gate)

    elif operation_proto.HasField("measure"):
        return fromproto_measure(operation_proto.measure)

    elif operation_proto.HasField("reset"):
        return fromproto_reset(operation_proto.reset)

    elif operation_proto.HasField("barrier"):
        return fromproto_barrier(operation_proto.barrier)

    elif operation_proto.HasField("control"):
        return fromproto_control(operation_proto.control)

    elif operation_proto.HasField("parallel"):
        return fromproto_parallel(operation_proto.parallel)

    elif operation_proto.HasField("power"):
        return fromproto_power(operation_proto.power)

    elif operation_proto.HasField("inverse"):
        return fromproto_inverse(operation_proto.inverse)

    elif operation_proto.HasField("ifstatement"):
        return fromproto_ifstatement(operation_proto.ifstatement)

    elif operation_proto.HasField("generalized"):
        return fromproto_generalized(operation_proto.generalized)

    elif operation_proto.HasField("gatecall"):
        return fromproto_gatecall(operation_proto.gatecall)


def toproto_measure(operation):
    return circuit_pb.Operation(measure=circuit_pb.Measure())


def fromproto_measure(operation_proto):
    return mc.Measure()


def toproto_reset(operation):
    return circuit_pb.Operation(reset=circuit_pb.Reset())


def fromproto_reset(operation_proto):
    return mc.Reset()


def toproto_barrier(operation):
    num_qubits = operation.num_qubits
    return circuit_pb.Operation(
        barrier=circuit_pb.Barrier(numqubits=num_qubits)
    )


def fromproto_barrier(operation_proto):
    num_qubits = operation_proto.numqubits
    return mc.Barrier(num_qubits)


def toproto_control(control):
    operation_proto = toproto_operation(control.op)
    control_proto = circuit_pb.Operation(control=circuit_pb.Control(
        numcontrols=control._num_controls, operation=operation_proto))
    return control_proto


def fromproto_control(control_proto):
    operation = fromproto_operation(control_proto.operation)
    return mc.Control(control_proto.numcontrols, operation)


def toproto_parallel(parallel):
    operation_proto = toproto_operation(parallel.op)
    parallel_proto = circuit_pb.Operation(parallel=circuit_pb.Parallel(
        numrepeats=parallel._num_repeats, operation=operation_proto))
    return parallel_proto


def fromproto_parallel(parallel_proto):
    operation = fromproto_operation(parallel_proto.operation)
    return mc.Parallel(parallel_proto.numrepeats, operation)


def toproto_ifstatement(ifstatement):
    operation_proto = toproto_operation(ifstatement.op)
    param = toproto_param(ifstatement.val)
    return circuit_pb.Operation(
        ifstatement=circuit_pb.IfStatement(
            nbits=ifstatement.num_bits,
            operation=operation_proto, value=param
        )
    )


def fromproto_ifstatement(ifstatement_proto):
    operation = fromproto_operation(ifstatement_proto.operation)
    param = fromproto_param(ifstatement_proto.value)
    return mc.IfStatement(operation, ifstatement_proto.nbits, param)


def toproto_power(power):
    operation_proto = toproto_operation(power._op)
    power_proto = circuit_pb.Power(operation=operation_proto)

    if isinstance(power._exponent, float):
        power_proto.double_val = power._exponent
    elif isinstance(power._exponent, int):
        power_proto.int_val = power._exponent
    elif isinstance(power._pexponent, Fraction):
        rational_proto = circuit_pb.Rational(
            num=power._exponent.numerator, den=power._exponent.denominator)
        power_proto.rational_val.CopyFrom(rational_proto)
    else:
        raise ValueError("Unsupported power type in Power operation")

    control_proto = circuit_pb.Operation(power=power_proto)
    return control_proto


def fromproto_power(power_proto):
    operation = fromproto_operation(power_proto.operation)

    if power_proto.HasField("double_val"):
        power = power_proto.double_val
    elif power_proto.HasField("int_val"):
        power = power_proto.int_val
    elif power_proto.HasField("rational_val"):
        rational_val = power_proto.rational_val
        power = Fraction(rational_val.num, rational_val.den)
    else:
        raise ValueError("Unsupported power type in Power operation")

    return mc.Power(operation, power)


def toproto_inverse(inverse):
    operation_proto = toproto_operation(inverse.op)
    inverse_proto = circuit_pb.Operation(
        inverse=circuit_pb.Inverse(operation=operation_proto))
    return inverse_proto


def fromproto_inverse(inverse_proto):
    operation = fromproto_operation(inverse_proto.operation)
    return mc.Inverse(operation)


def toproto_complex(complex_matrix):
    complex_args = []

    for i in range(complex_matrix.rows):
        for j in range(complex_matrix.cols):
            val = complex_matrix[i, j]

            if isinstance(val, se.Symbol):
                # If val is a symbolic variable, create only real argument
                real_arg = circuit_pb.Arg(
                    symbol_value=circuit_pb.Symbol(value=str(val)))
                imag_arg = circuit_pb.Arg(
                    symbol_value=circuit_pb.Symbol(value=str(0.0)))
                complex_arg = circuit_pb.ComplexArg()
                complex_arg.real.CopyFrom(real_arg)
                complex_arg.imag.CopyFrom(imag_arg)
            else:
                # If val is a number, create real and imaginary arguments
                real_arg = circuit_pb.Arg(
                    argvalue_value=circuit_pb.ArgValue(double_value=val.real))
                imag_arg = circuit_pb.Arg(
                    argvalue_value=circuit_pb.ArgValue(double_value=val.imag))
                complex_arg = circuit_pb.ComplexArg()
                complex_arg.real.CopyFrom(real_arg)
                complex_arg.imag.CopyFrom(imag_arg)

            complex_args.append(complex_arg)

    return complex_args


def toproto_custom(custom):
    complex_args = toproto_complex(custom.matrix.T)
    return circuit_pb.Operation(custom=circuit_pb.GateCustom(matrix=complex_args, nqubits=custom.num_qubits))


def fromproto_complex(complex_args):
    complex_matrix = []
    for complex_arg in complex_args:
        real_val = 0.0
        imag_val = 0.0

        if complex_arg.real.HasField('argvalue_value'):
            # If real part is present, use the double value
            real_val = complex_arg.real.argvalue_value.double_value
        elif complex_arg.real.HasField('symbol_value'):
            # If real part is symbolic, reconstruct the symbol
            real_val = se.sympify(complex_arg.real.symbol_value.value)

        if complex_arg.imag.HasField('argvalue_value'):
            # If imag part is present, use the double value
            imag_val = complex_arg.imag.argvalue_value.double_value
        elif complex_arg.imag.HasField('symbol_value'):
            # If imag part is symbolic, reconstruct the symbol
            imag_val = se.sympify(complex_arg.imag.symbol_value.value)

        complex_matrix.append(real_val + imag_val * I)

    return se.Matrix(complex_matrix)


def fromproto_custom(custom_proto):
    complex_matrix = fromproto_complex(custom_proto.matrix)
    num_qubits = custom_proto.nqubits
    complex_matrix = complex_matrix.reshape(2 ** num_qubits, 2 ** num_qubits)
    complex_matrix = complex_matrix.T  # Add transpose operation
    return mc.GateCustom(complex_matrix)


def toproto_generalized(generalized_gate):
    args = []
    for param in generalized_gate.getparams():
        arg_proto = toproto_param(param)
        args.append(arg_proto)

    generalized_proto = circuit_pb.Generalized(
        name=generalized_gate._name, args=args, regsizes=generalized_gate._params)
    return circuit_pb.Operation(generalized=generalized_proto)


def fromproto_generalized(generalized_proto):
    generalized_constructors = {
        "QFT": mc.QFT,
        "PhaseGradient": mc.PhaseGradient,
        "Diffusion": mc.Diffusion,
        "PolynomialOracle": mc.PolynomialOracle,
    }

    name = generalized_proto.name
    args = list(map(fromproto_param, generalized_proto.args))

    if name in generalized_constructors:
        return generalized_constructors[name](*generalized_proto.regsizes, *args)
    else:
        raise NotImplementedError(f"Unrecognized generalized operation {name}")


def toproto_gatedecl(decl):
    instructions_proto = list(map(toproto_instruction, decl.instructions))
    arguments = [circuit_pb.Symbol(value=arg) for arg in decl.arguments]
    return circuit_pb.GateDecl(name=decl.name, args=arguments, instructions=instructions_proto)


def fromproto_gatedecl(gatedecl_proto):
    instructions = [fromproto_instruction(inst)
                    for inst in gatedecl_proto.instructions]
    arguments = [str(symbol) for symbol in gatedecl_proto.args]
    return mc.GateDecl(name=gatedecl_proto.name, arguments=arguments, instructions=instructions)


def toproto_gatecall(gatecall):
    decl_proto = toproto_gatedecl(gatecall._decl)
    args_proto = [toproto_param(arg) for arg in gatecall._args]
    return circuit_pb.Operation(
        gatecall=circuit_pb.GateCall(decl=decl_proto, args=args_proto))


def fromproto_gatecall(gatecall_proto):
    decl = fromproto_gatedecl(gatecall_proto.decl)
    args = [fromproto_param(arg) for arg in gatecall_proto.args]
    return mc.GateCall(decl, args)


exported_functions = {
    name: func
    for name, func in globals().items()
    if inspect.isfunction(func) and (name.startswith('toproto') or name.startswith('fromproto'))
}

# Export all the functions
__all__ = list(exported_functions.keys())
