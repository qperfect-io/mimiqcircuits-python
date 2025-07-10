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

from ctypes import ArgumentError
from fractions import Fraction
from typing import Callable, Optional, Type

import numpy as np
import symengine as se
from symengine.lib.symengine_wrapper import Zero as seZero
from symengine.lib.symengine_wrapper import One as seOne
from symengine.lib.symengine_wrapper import NegativeOne as seNegativeOne

import mimiqcircuits as mc
from mimiqcircuits.proto import circuit_pb2, pauli_pb2
from mimiqcircuits.proto.bitvector import fromproto_bitvector, toproto_bitvector
from mimiqcircuits.operations.gates.generalized.rpauli import RPauli

# ---------------------- Registry Pattern for Conversions ----------------------


class ProtoRegistry:
    """Registry for proto conversion functions based on type."""

    def __init__(self):
        self.toproto_registry = {}
        self.fromproto_registry = {}

    def register_toproto(self, mc_type):
        """Register a function to convert from mimiqcircuits to protobuf.

        Can be used as a decorator:
            @registry.register_toproto(mc.PauliString)
            def convert_paulistring(...):
                ...
        """

        def decorator(converter_func):
            self.toproto_registry[mc_type] = converter_func
            return converter_func

        return decorator

    def register_toproto_direct(self, mc_type, converter_func):
        """Register a function to convert from mimiqcircuits to protobuf."""
        self.toproto_registry[mc_type] = converter_func

    def register_fromproto(self, proto_field):
        """Register a function to convert from protobuf to mimiqcircuits.

        Can be used as a decorator:
            @registry.register_fromproto("paulistring")
            def convert_from_paulistring(...):
                ...
        """

        def decorator(converter_func):
            self.fromproto_registry[proto_field] = converter_func
            return converter_func

        return decorator

    def register_fromproto_direct(self, proto_field, converter_func):
        """Register a function to convert from protobuf to mimiqcircuits."""
        self.fromproto_registry[proto_field] = converter_func

    def get_toproto_converter(self, obj_type: Type) -> Optional[Callable]:
        """Get converter for a specific mimiqcircuits type."""
        return self.toproto_registry.get(obj_type)

    def get_fromproto_converter(self, proto_field: str) -> Optional[Callable]:
        """Get converter for a specific protobuf field."""
        return self.fromproto_registry.get(proto_field)


# Create registries for different object types
gate_registry = ProtoRegistry()
operation_registry = ProtoRegistry()
krauschannel_registry = ProtoRegistry()
operator_registry = ProtoRegistry()
annotation_registry = ProtoRegistry()

# ---------------------- Constants and Mappings ----------------------

# Gate mapping: mimiqcircuits -> protobuf
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
#
# Reverse gate mapping: protobuf -> mimiqcircuits
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
    mc.Pow: circuit_pb2.OperationType.Pow,
}

# Reverse operation mapping: protobuf -> mimiqcircuits
REVERSE_OPERATIONMAP = {v: k for k, v in OPERATIONMAP.items()}

# Generalized operation mapping: mimiqcircuits -> protobuf
GENERALIZEDOPERATIONMAP = {
    mc.Barrier: circuit_pb2.GeneralizedOperationType.Barrier,
    mc.Add: circuit_pb2.GeneralizedOperationType.Add,
    mc.Multiply: circuit_pb2.GeneralizedOperationType.Multiply,
}

# Reverse generalized operation mapping: protobuf -> mimiqcircuits
REVERSE_GENERALIZEDOPERATIONMAP = {v: k for k, v in GENERALIZEDOPERATIONMAP.items()}

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

# Operator mapping: mimiqcircuits -> protobuf
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

# Reverse operator mapping: protobuf -> mimiqcircuits
REVERSE_OPERATORMAP = {v: k for k, v in OPERATORMAP.items()}

# Generalized gate mappings
GENERALIZEDGATEMAP = {
    mc.QFT: circuit_pb2.GeneralizedType.QFT,
    mc.PhaseGradient: circuit_pb2.GeneralizedType.PhaseGradient,
    mc.PolynomialOracle: circuit_pb2.GeneralizedType.PolynomialOracle,
    mc.Diffusion: circuit_pb2.GeneralizedType.Diffusion,
    mc.GateRNZ: circuit_pb2.GeneralizedType.GateRNZ,
}

REVERSE_GENERALIZEDGATEMAP = {v: k for k, v in GENERALIZEDGATEMAP.items()}

# Expression mappings: regex -> protobuf
EXPRSYMENGINE = {
    se.Add: circuit_pb2.FunctionType.ADD,
    se.Mul: circuit_pb2.FunctionType.MUL,
    se.Pow: circuit_pb2.FunctionType.POW,
    se.sin: circuit_pb2.FunctionType.SIN,
    se.cos: circuit_pb2.FunctionType.COS,
    se.tan: circuit_pb2.FunctionType.TAN,
    se.log: circuit_pb2.FunctionType.LOG,
}

# reverse expression mappings: protobuf -> regex
EXPRSPROTO = {v: k for k, v in EXPRSYMENGINE.items()}

# Irrational mappings: mimiqcircuits -> protobuf
IRRATIONALSYMENGINE = {
    se.pi: circuit_pb2.Irrational.PI,
    se.E: circuit_pb2.Irrational.EULER,
}

# Reverse irrational mappings: protobuf -> mimiqcircuits
IRRATIONALPROTO = {v: k for k, v in IRRATIONALSYMENGINE.items()}

# Annotation mappings : mimiqcircuits -> protobuf
ANNOTATIONMAP = {
    mc.QubitCoordinates: circuit_pb2.AnnotationType.QubitCoordinates,
    mc.ShiftCoordinates: circuit_pb2.AnnotationType.ShiftCoordinates,
    mc.Tick: circuit_pb2.AnnotationType.Tick,
}
# Reverse annotation mappings: protobuf -> mimiqcircuits
REVERSE_ANNOTATIONMAP = {v: k for k, v in ANNOTATIONMAP.items()}

# Generalized annotation mappings
GENERALIZEDANNOTATIONMAP = {
    mc.Detector: circuit_pb2.GeneralizedAnnotationType.Detector,
    mc.ObservableInclude: circuit_pb2.GeneralizedAnnotationType.ObservableInclude,
}

# Reverse generalized annotation mappings
REVERSE_GENERALIZEDANNOTATIONMAP = {v: k for k, v in GENERALIZEDANNOTATIONMAP.items()}

# ---------------------- Base Conversion Functions ----------------------


def safe_convert(value):
    """Convert float to int if it represents an integer value."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def toproto_arg(param):
    """Convert a parameter to protocol buffer format."""
    # Dispatcher dictionary for different parameter types
    type_handlers = {
        int: lambda p: circuit_pb2.Arg(
            argvalue_value=circuit_pb2.ArgValue(integer_value=p)
        ),
        float: lambda p: handle_float(p),
        se.Integer: lambda p: circuit_pb2.Arg(
            argvalue_value=circuit_pb2.ArgValue(integer_value=p.p)
        ),
        se.Rational: lambda p: handle_numeric(float(p)),
        se.Float: lambda p: handle_numeric(float(p)),
        se.RealDouble: lambda p: circuit_pb2.Arg(
            argvalue_value=circuit_pb2.ArgValue(double_value=p.real)
        ),
        se.Symbol: lambda p: circuit_pb2.Arg(
            symbol_value=circuit_pb2.Symbol(value=p.name)
        ),
        complex: lambda p: handle_complex(p),
        se.ComplexDouble: lambda p: handle_complex(p.evalf()),
        str: lambda p: circuit_pb2.Arg(symbol_value=circuit_pb2.Symbol(value=p)),
        seZero: lambda p: toproto_arg(0),
        seOne: lambda p: toproto_arg(1),
        seNegativeOne: lambda p: toproto_arg(-1),
    }

    # Special case for irrational numbers
    if param in IRRATIONALSYMENGINE:
        arg = circuit_pb2.Arg()
        arg.irrational_value = IRRATIONALSYMENGINE[param]
        return arg

    # Special case for symengine expressions
    if type(param) in EXPRSYMENGINE:
        arg = circuit_pb2.Arg()
        arg.argfunction_value.mtype = EXPRSYMENGINE[type(param)]
        for a in param.args:
            arg.argfunction_value.args.append(toproto_arg(a))
        return arg

    # Special case for symengine.Basic
    if isinstance(param, se.Basic) and param == (se.I):
        return handle_complex(param)
    
    if isinstance(param, se.Basic) and param.is_Number:
        try:
            return handle_numeric(float(param))
        except Exception:
            pass 

    # Handle the parameter using the appropriate handler
    handler = type_handlers.get(type(param))
    if handler:
        return handler(param)

    raise ValueError(f"Unsupported parameter type: {type(param)}")


def handle_float(value):
    """Handle float values, converting to integer if possible."""
    arg = circuit_pb2.Arg()
    value = safe_convert(value)
    if isinstance(value, int):
        arg.argvalue_value.integer_value = value
    else:
        arg.argvalue_value.double_value = value
    return arg


def handle_numeric(value):
    """Handle numeric values that need to be converted to float first."""
    arg = circuit_pb2.Arg()
    value = safe_convert(value)
    if isinstance(value, int):
        arg.argvalue_value.integer_value = value
    else:
        arg.argvalue_value.double_value = value
    return arg


def handle_complex(value):
    """Handle complex symengine number conversion."""
    real_part = float(value.real)
    imag_part = float(value.imag)
    if imag_part == 0:
        return toproto_arg(real_part)
    else:
        raise ArgumentError("Arguments cannot be complex numbers")


def fromproto_arg(arg):
    """Convert a protocol buffer Arg to a Python/symengine object."""
    # Handle argvalue_value (integer or double)
    if arg.HasField("argvalue_value"):
        av = arg.argvalue_value
        if av.HasField("integer_value"):
            return av.integer_value
        elif av.HasField("double_value"):
            return safe_convert(av.double_value)

    # Handle irrational values
    elif arg.HasField("irrational_value"):
        return IRRATIONALPROTO[arg.irrational_value]

    # Handle symbol values
    elif arg.HasField("symbol_value"):
        return se.Symbol(arg.symbol_value.value)

    # Handle function values
    elif arg.HasField("argfunction_value"):
        return fromproto_function(arg.argfunction_value)

    # Handle complex values
    elif arg.HasField("argcomplex_value"):
        real = fromproto_arg(arg.argcomplex_value.real)
        imag = fromproto_arg(arg.argcomplex_value.imag)
        return complex(real, imag)

    else:
        raise ValueError("Unsupported parameter type in protocol buffer")


def fromproto_function(function_proto):
    """Convert a protocol buffer function to a symengine expression."""
    ftype = function_proto.mtype
    params = list(map(fromproto_arg, function_proto.args))

    # Handle different function types
    if ftype in EXPRSPROTO:
        return EXPRSPROTO[ftype](*params)
    elif ftype == circuit_pb2.FunctionType.DIV:
        return se.Mul(params[0], se.Pow(params[1], -1))
    elif ftype == circuit_pb2.FunctionType.EXP:
        return se.Pow(se.E, params[0])
    else:
        raise ValueError("Unsupported function type in protocol buffer")


def toproto_complex(value):
    """Convert a complex value to protocol buffer format."""
    if hasattr(value, "real") and hasattr(value, "imag"):
        # If the value has real and imaginary attributes, use them
        return circuit_pb2.ComplexArg(
            real=toproto_arg(value.real), imag=toproto_arg(value.imag)
        )
    else:
        # Otherwise, treat it as a symbolic expression with the imaginary part as zero
        return circuit_pb2.ComplexArg(real=toproto_arg(value), imag=toproto_arg(0))


def fromproto_complex(complex_proto):
    """Convert a protocol buffer complex value to a Python complex."""
    real = fromproto_arg(complex_proto.real)
    imag = fromproto_arg(complex_proto.imag)

    # Only return the real part if the imaginary part is zero
    if imag == 0:
        return real
    return complex(real, imag)


def toproto_note(note):
    """Convert a note value to protocol buffer format."""
    if isinstance(note, int):
        return circuit_pb2.Note(int_note=note)
    elif isinstance(note, float):
        return circuit_pb2.Note(double_note=note)
    else:
        raise ValueError(f"Unsupported note type: {type(note)}")


def fromproto_note(note_proto):
    """Convert a protocol buffer note to a Python value."""
    if note_proto.HasField("int_note"):
        return float(note_proto.int_note)  # Convert to float for compatibility
    elif note_proto.HasField("double_note"):
        return note_proto.double_note
    else:
        raise ValueError(f"Unsupported Note type in proto: {note_proto}")


# ---------------------- Gate Conversion Functions ----------------------


def toproto_gate(gate, declcache=None):
    """Convert a gate to protocol buffer format using registry pattern."""
    # Try to use a registered converter first
    converter = gate_registry.get_toproto_converter(type(gate))

    if converter:
        return converter(gate, declcache)

    # Simple gate types from the mapping
    gate_type_enum = GATEMAP.get(type(gate))
    if gate_type_enum is not None:
        params = [toproto_arg(param) for param in gate.getparams()]
        return circuit_pb2.Gate(
            simplegate=circuit_pb2.SimpleGate(mtype=gate_type_enum, parameters=params)
        )

    # Fall back to case-by-case handling
    raise ValueError(f"Unsupported gate type: {type(gate)}")


def fromproto_gate(gate_proto, declcache=None):
    """Convert a protocol buffer gate to a mimiqcircuits gate using registry pattern."""
    # Determine which field is set
    gate_field = gate_proto.WhichOneof("gate")

    # Try to use a registered converter first
    converter = gate_registry.get_fromproto_converter(gate_field)
    if converter:
        return converter(getattr(gate_proto, gate_field), declcache)

    # Fall back to case-by-case handling
    if gate_field == "simplegate":
        gate_class = REVERSE_GATEMAP.get(gate_proto.simplegate.mtype)
        if gate_class:
            params = [
                fromproto_arg(param) for param in gate_proto.simplegate.parameters
            ]
            return gate_class(*params)
        raise ValueError(
            f"Unsupported gate type in proto: {gate_proto.simplegate.mtype}"
        )

    raise ValueError(f"Unsupported gate type in proto: {gate_field}")


# Register specific gate converters
@gate_registry.register_toproto(mc.PauliString)
def toproto_paulistring(gate, declcache=None):
    """Convert a PauliString to protocol buffer format."""
    return circuit_pb2.Gate(
        paulistring=pauli_pb2.PauliString(numqubits=gate.num_qubits, pauli=gate.pauli)
    )


@gate_registry.register_toproto(mc.GateCustom)
def toproto_customgate(gate, declcache=None):
    """Convert a GateCustom to protocol buffer format."""
    U = [toproto_complex(param) for param in gate.matrix.T]
    return circuit_pb2.Gate(
        customgate=circuit_pb2.CustomGate(numqubits=gate.num_qubits, matrix=U)
    )


def toproto_gatedecl(gate, declcache=None):
    """Convert a GateDecl to protocol buffer format."""
    instructions_proto = list(
        map(
            lambda inst: toproto_instruction(inst, declcache),
            gate.circuit.instructions,
        )
    )
    arguments = [circuit_pb2.Symbol(value=str(arg)) for arg in gate.arguments]
    return circuit_pb2.GateDecl(
        name=gate.name, args=arguments, instructions=instructions_proto
    )


@gate_registry.register_toproto(mc.GateCall)
def toproto_gatecall(gate, declcache=None):
    """Convert a GateCall to protocol buffer format."""
    args_proto = [toproto_arg(arg) for arg in gate._args]
    declid = id(gate._decl)

    if declcache is None:
        decl_proto = toproto_gatedecl(gate._decl)
        return circuit_pb2.Gate(
            gatecall=circuit_pb2.GateCall(decl=decl_proto, args=args_proto)
        )
    else:
        if declid not in declcache[0]:
            declcache[0][declid] = toproto_gatedecl(gate._decl, declcache)
            declcache[1].append(declid)

        return circuit_pb2.Gate(
            cachedgatecall=circuit_pb2.CachedGateCall(id=declid, args=args_proto)
        )


@gate_registry.register_toproto(mc.Control)
def toproto_control(gate, declcache=None):
    """Convert a Control gate to protocol buffer format."""
    control_gate_proto = toproto_gate(gate.op, declcache)
    return circuit_pb2.Gate(
        control=circuit_pb2.Control(
            operation=control_gate_proto, numcontrols=gate.num_controls
        )
    )


@gate_registry.register_toproto(mc.Power)
def toproto_power(gate, declcache=None):
    """Convert a Power gate to protocol buffer format."""
    base_gate_proto = toproto_gate(gate.op, declcache)

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


@gate_registry.register_toproto(mc.Inverse)
def toproto_inverse(gate, declcache=None):
    """Convert an Inverse gate to protocol buffer format."""
    base_gate_proto = toproto_gate(gate.op, declcache)
    return circuit_pb2.Gate(inverse=circuit_pb2.Inverse(operation=base_gate_proto))


@gate_registry.register_toproto(mc.Parallel)
def toproto_parallel(gate, declcache=None):
    """Convert a Parallel gate to protocol buffer format."""
    base_gate_proto = toproto_gate(gate.op, declcache)
    return circuit_pb2.Gate(
        parallel=circuit_pb2.Parallel(
            operation=base_gate_proto, numrepeats=gate.num_repeats
        )
    )


# Register generalized gate converter for all gates with a name attribute
def register_generalized_gates():
    """Register converter for all gates in GENERALIZED_GATE_MAP."""
    for gate_class in GENERALIZEDGATEMAP.keys():
        gate_registry.register_toproto_direct(gate_class, toproto_generalized_gate)


def toproto_generalized_gate(gate, declcache=None):
    """Convert a generalized gate to protocol buffer format."""
    args = [toproto_arg(param) for param in gate.getparams()]

    # Special Case: PolynomialOracle (Ensures Correct Parameter Order)
    if isinstance(gate, mc.PolynomialOracle):
        args = [toproto_arg(arg) for arg in gate._params[2:]]  # Skip nx, ny

    return circuit_pb2.Gate(
        generalized=circuit_pb2.Generalized(
            mtype=GENERALIZEDGATEMAP[type(gate)],
            args=args,  # Processed params
            qregsizes=list(getattr(gate, "_qregsizes", [])),  # Ensure qregsizes exist
        )
    )


# Register from_proto converters
@gate_registry.register_fromproto("paulistring")
def fromproto_paulistring(paulistring_proto, declcache=None):
    """Convert a protocol buffer PauliString to a mimiqcircuits PauliString."""
    return mc.PauliString(paulistring_proto.pauli)


@gate_registry.register_fromproto("customgate")
def fromproto_customgate(customgate_proto, declcache=None):
    """Convert a protocol buffer CustomGate to a mimiqcircuits GateCustom."""
    U_matrix = [fromproto_complex(val) for val in customgate_proto.matrix]
    matrix_size = 2**customgate_proto.numqubits
    original_matrix = np.array(U_matrix).reshape(matrix_size, matrix_size).T
    return mc.GateCustom(matrix=original_matrix)


@gate_registry.register_fromproto("generalized")
def fromproto_generalized_gate(generalized_proto, declcache=None):
    """Convert a protocol buffer Generalized gate to a mimiqcircuits gate."""
    name = [k for k, v in GENERALIZEDGATEMAP.items() if v == generalized_proto.mtype][0]
    args = [fromproto_arg(arg) for arg in generalized_proto.args]
    qregsizes = list(generalized_proto.qregsizes)
    if name in GENERALIZEDGATEMAP:
        return name(*qregsizes, *args)
    else:
        raise ValueError(f"Unsupported generalized gate name: {name}")


@gate_registry.register_fromproto("control")
def fromproto_control(control_proto, declcache=None):
    """Convert a protocol buffer Control gate to a mimiqcircuits Control gate."""
    control_gate = fromproto_gate(control_proto.operation, declcache)
    return mc.Control(control_proto.numcontrols, control_gate)


@gate_registry.register_fromproto("power")
def fromproto_power(power_proto, declcache=None):
    """Convert a protocol buffer Power gate to a mimiqcircuits Power gate."""
    base_gate = fromproto_gate(power_proto.operation, declcache)

    if power_proto.HasField("int_val"):
        power_value = power_proto.int_val
    elif power_proto.HasField("rational_val"):
        power_value = Fraction(
            power_proto.rational_val.num, power_proto.rational_val.den
        )
    elif power_proto.HasField("double_val"):
        power_value = power_proto.double_val
    else:
        raise ValueError("Unsupported power value type in proto")

    return mc.Power(base_gate, power_value)


@gate_registry.register_fromproto("inverse")
def fromproto_inverse(inverse_proto, declcache=None):
    """Convert a protocol buffer Inverse gate to a mimiqcircuits Inverse gate."""
    base_gate = fromproto_gate(inverse_proto.operation, declcache)
    return mc.Inverse(base_gate)


@gate_registry.register_fromproto("parallel")
def fromproto_parallel(parallel_proto, declcache=None):
    """Convert a protocol buffer Parallel gate to a mimiqcircuits Parallel gate."""
    base_gate = fromproto_gate(parallel_proto.operation, declcache)
    return mc.Parallel(parallel_proto.numrepeats, base_gate)


@gate_registry.register_fromproto("gatecall")
def fromproto_gatecall(gatecall_proto, declcache=None):
    """Convert a protocol buffer GateCall to a mimiqcircuits GateCall."""
    # Deserialize the GateDecl using the correct function
    decl = fromproto_gatedecl(gatecall_proto.decl)
    args = [fromproto_arg(arg) for arg in gatecall_proto.args]
    return mc.GateCall(decl, args)


@gate_registry.register_fromproto("cachedgatecall")
def fromproto_cachedgatecall(cachedgatecall_proto, declcache):
    """Convert a protocol buffer CachedGateCall to a mimiqcircuits GateCall."""

    if declcache is None:
        raise ValueError("Declcache must be provided for CachedGateCall conversion")

    declid = cachedgatecall_proto.id
    args = [fromproto_arg(arg) for arg in cachedgatecall_proto.args]

    if declid not in declcache[0]:
        raise ValueError(f"GateDecl with ID {declid} not found in cache")

    decl = declcache[0][declid]
    return mc.GateCall(decl, args)


def fromproto_gatedecl(gatedecl_proto, declcache=None):
    """Convert a protocol buffer GateDecl to a mimiqcircuits GateDecl."""
    name = gatedecl_proto.name
    arguments = [se.Symbol(symbol.value) for symbol in gatedecl_proto.args]
    instructions = [
        fromproto_instruction(inst, declcache) for inst in gatedecl_proto.instructions
    ]

    return mc.GateDecl(
        name=name, arguments=tuple(arguments), circuit=mc.Circuit(instructions)
    )


# ---------------------- Operator Conversion Functions ----------------------


def toproto_operator(operator, declcache=None):
    """Convert an operator to protocol buffer format using registry pattern."""
    # Try to use a registered converter first
    converter = operator_registry.get_toproto_converter(type(operator))
    if converter:
        return converter(operator, declcache)

    # Simple operator types from the mapping
    if isinstance(operator, mc.AbstractOperator):
        operator_type = OPERATORMAP.get(type(operator))
        if operator_type is not None:
            params = [toproto_arg(param) for param in operator.getparams()]
            return circuit_pb2.Operator(
                simpleoperator=circuit_pb2.SimpleOperator(
                    mtype=operator_type, parameters=params
                )
            )

    # Fall back to case-by-case handling
    if isinstance(operator, mc.Gate):
        gate_proto = toproto_gate(operator, declcache)
        return circuit_pb2.Operator(
            **{
                gate_proto.WhichOneof("gate"): getattr(
                    gate_proto, gate_proto.WhichOneof("gate")
                )
            }
        )

    raise ValueError(f"Unsupported Operator type: {type(operator)}")


def fromproto_operator(operator_proto, declcache=None):
    """Convert a protocol buffer operator to a mimiqcircuits operator using registry pattern."""
    # Determine which field is set
    operator_field = operator_proto.WhichOneof("operator")

    # Try to use a registered converter first
    converter = operator_registry.get_fromproto_converter(operator_field)
    if converter:
        return converter(getattr(operator_proto, operator_field), declcache)

    # Fall back to case-by-case handling
    if operator_field == "simpleoperator":
        operator_class = REVERSE_OPERATORMAP.get(operator_proto.simpleoperator.mtype)
        if operator_class:
            params = [
                fromproto_arg(param)
                for param in operator_proto.simpleoperator.parameters
            ]
            return operator_class(*params)

    # Handle gate-based operators
    gate_fields = [
        "simplegate",
        "customgate",
        "control",
        "power",
        "inverse",
        "parallel",
        "paulistring",
    ]
    if operator_field in gate_fields:
        return fromproto_gate(
            circuit_pb2.Gate(
                **{operator_field: getattr(operator_proto, operator_field)}
            ),
            declcache,
        )

    raise ValueError(f"Unsupported Operator type in proto: {operator_field}")


# Register specific operator converters
@operator_registry.register_toproto(mc.RescaledGate)
def toproto_rescaledgate(operator, declcache=None):
    """Convert a RescaledGate to protocol buffer format."""
    base_gate_proto = toproto_gate(operator.get_operation(), declcache)
    scale_proto = toproto_arg(operator.get_scale())
    return circuit_pb2.Operator(
        rescaledgate=circuit_pb2.RescaledGate(
            operation=base_gate_proto, scale=scale_proto
        )
    )


@operator_registry.register_toproto(mc.Operator)
def toproto_customoperator(operator, declcache=None):
    """Convert a custom Operator to protocol buffer format."""
    matrix_data = operator.matrix() if callable(operator.matrix) else operator.matrix
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


@operator_registry.register_fromproto("rescaledgate")
def fromproto_rescaledgate(rescaledgate_proto, declcache=None):
    """Convert a protocol buffer RescaledGate to a mimiqcircuits RescaledGate."""
    base_gate = fromproto_gate(rescaledgate_proto.operation, declcache)
    scale = fromproto_arg(rescaledgate_proto.scale)
    return mc.RescaledGate(base_gate, scale)


@operator_registry.register_fromproto("customoperator")
def fromproto_customoperator(customoperator_proto, declcache=None):
    """Convert a protocol buffer CustomOperator to a mimiqcircuits Operator."""
    matrix = [fromproto_complex(val) for val in customoperator_proto.matrix]
    matrix = np.array(matrix).reshape(
        2**customoperator_proto.numqubits,
        2**customoperator_proto.numqubits,
    )
    return mc.Operator(mat=matrix)


# ---------------------- Kraus Channel Conversion Functions ----------------------


def toproto_krauschannel(kraus_channel, declcache=None):
    """Convert a Kraus channel to protocol buffer format using registry pattern."""
    # Try to use a registered converter first
    converter = krauschannel_registry.get_toproto_converter(type(kraus_channel))
    if converter:
        return converter(kraus_channel, declcache)

    # Simple kraus channel types from the mapping
    kraus_channel_type = KRAUSCHANNELMAP.get(type(kraus_channel))
    if kraus_channel_type is not None:
        params = [toproto_arg(param) for param in kraus_channel.getparams()]
        return circuit_pb2.KrausChannel(
            simplekrauschannel=circuit_pb2.SimpleKrausChannel(
                mtype=kraus_channel_type, parameters=params
            )
        )

    # Fall back to case-by-case handling
    raise ValueError(f"Unsupported KrausChannel type: {type(kraus_channel)}")


def fromproto_krauschannel(kraus_proto, declcache=None):
    """Convert a protocol buffer Kraus channel to a mimiqcircuits Kraus channel."""
    # Determine which field is set
    kraus_field = kraus_proto.WhichOneof("krauschannel")

    # Try to use a registered converter first
    converter = krauschannel_registry.get_fromproto_converter(kraus_field)
    if converter:
        return converter(getattr(kraus_proto, kraus_field), declcache)

    # Fall back to case-by-case handling
    if kraus_field == "simplekrauschannel":
        kraus_class = REVERSE_KRAUSCHANNELMAP.get(kraus_proto.simplekrauschannel.mtype)
        if kraus_class:
            params = [
                fromproto_arg(param)
                for param in kraus_proto.simplekrauschannel.parameters
            ]
            return kraus_class(*params)
        raise ValueError(
            f"Unsupported KrausChannel type in proto: {kraus_proto.simplekrauschannel.mtype}"
        )

    raise ValueError(f"Unsupported KrausChannel type in proto: {kraus_field}")


# Register specific krauschannel converters
@krauschannel_registry.register_toproto(mc.Kraus)
def toproto_kraus(kraus_channel, declcache=None):
    """Convert a Kraus to protocol buffer format."""
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


@krauschannel_registry.register_toproto(mc.MixedUnitary)
def toproto_mixedunitary(kraus_channel, declcache=None):
    """Convert a MixedUnitary to protocol buffer format."""
    operators_proto = []
    gates_or_matrices = (
        kraus_channel.unitarygates()
        if hasattr(kraus_channel, "unitarygates")
        else kraus_channel.unitarymatrices()
    )
    probabilities = kraus_channel.probabilities()

    for gate_or_matrix, scale in zip(gates_or_matrices, probabilities):
        gate_proto = toproto_gate(gate_or_matrix, declcache)
        scale_proto = toproto_arg(se.sqrt(scale))
        operators_proto.append(
            circuit_pb2.RescaledGate(operation=gate_proto, scale=scale_proto)
        )
    return circuit_pb2.KrausChannel(
        mixedunitarychannel=circuit_pb2.MixedUnitaryChannel(operators=operators_proto)
    )


@krauschannel_registry.register_toproto(mc.PauliNoise)
def toproto_paulinoise(kraus_channel, declcache=None):
    """Convert a PauliNoise to protocol buffer format."""
    # Call probabilities as a method
    probabilities_proto = [toproto_arg(prob) for prob in kraus_channel.probabilities()]
    pauli_strings_proto = [
        pauli_pb2.PauliString(numqubits=pauli_str.num_qubits, pauli=pauli_str.pauli)
        for pauli_str in kraus_channel.paulistr
    ]
    return circuit_pb2.KrausChannel(
        paulichannel=circuit_pb2.PauliChannel(
            probabilities=probabilities_proto, paulistrings=pauli_strings_proto
        )
    )


@krauschannel_registry.register_toproto(mc.Depolarizing)
def toproto_depolarizing(kraus_channel, declcache=None):
    """Convert a Depolarizing to protocol buffer format."""
    return circuit_pb2.KrausChannel(
        depolarizingchannel=circuit_pb2.DepolarizingChannel(
            numqubits=kraus_channel.N, probability=toproto_arg(kraus_channel.p)
        )
    )


@krauschannel_registry.register_fromproto("customkrauschannel")
def fromproto_customkrauschannel(customkrauschannel_proto, declcache=None):
    """Convert a protocol buffer CustomKrausChannel to a mimiqcircuits Kraus."""
    operators = []
    for op_proto in customkrauschannel_proto.operators:
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
                    fromproto_arg(param) for param in op_proto.simpleoperator.parameters
                ]
                operators.append(operator_class(*params))
            else:
                raise ValueError(
                    f"Unsupported SimpleOperator type: {op_proto.simpleoperator.mtype}"
                )
        else:
            raise ValueError("Unsupported operator type in customkrauschannel")

    return mc.Kraus(operators)


@krauschannel_registry.register_fromproto("mixedunitarychannel")
def fromproto_mixedunitarychannel(mixedunitary_proto, declcache=None):
    """Convert a protocol buffer MixedUnitaryChannel to a mimiqcircuits MixedUnitary."""
    probabilities = []
    unitary_matrices = []

    for rescaled_gate_proto in mixedunitary_proto.operators:
        scale = fromproto_arg(rescaled_gate_proto.scale)
        probabilities.append((scale**2))
        gate = fromproto_gate(rescaled_gate_proto.operation, declcache)
        unitary_matrices.append(gate)

    return mc.MixedUnitary(probabilities, unitary_matrices)


@krauschannel_registry.register_fromproto("paulichannel")
def fromproto_paulichannel(paulichannel_proto, declcache=None):
    """Convert a protocol buffer PauliChannel to a mimiqcircuits PauliNoise."""
    probabilities = [fromproto_arg(prob) for prob in paulichannel_proto.probabilities]
    pauli_strings = [
        mc.PauliString(pauli_str.pauli) for pauli_str in paulichannel_proto.paulistrings
    ]
    return mc.PauliNoise(probabilities, pauli_strings)


@krauschannel_registry.register_fromproto("depolarizingchannel")
def fromproto_depolarizing(depolarizing_proto, declcache=None):
    """Convert a protocol buffer DepolarizingChannel to a mimiqcircuits Depolarizing."""
    num_qubits = depolarizing_proto.numqubits
    probability = fromproto_arg(depolarizing_proto.probability)
    return mc.Depolarizing(num_qubits, probability)


# ---------------------- Operation Conversion Functions ----------------------


def register_simple_operations():
    """Register converter for all operations in SIMPLE_OPERATION_MAP."""
    for operation_class in OPERATIONMAP.keys():
        operation_registry.register_toproto_direct(
            operation_class, toproto_simpleoperation
        )


def toproto_simpleoperation(operation, declcache=None):
    """Convert a simple operation to protocol buffer format."""
    operation_type = OPERATIONMAP.get(type(operation))

    args = [toproto_arg(param) for param in operation.getparams()]

    if operation_type is None:
        raise TypeError("No matching operation type.")

    return circuit_pb2.Operation(
        simpleoperation=circuit_pb2.SimpleOperation(
            mtype=operation_type, parameters=args
        )
    )


def register_generalized_operations():
    """Register converter for all operations in GENERALIZEDOPERATIONMAP."""
    for operation_class in GENERALIZEDOPERATIONMAP.keys():
        operation_registry.register_toproto_direct(
            operation_class, toproto_generalized_operation
        )


def toproto_generalized_operation(operation, declcache=None):
    """Convert a generalized operation to protocol buffer format."""
    generalized_operation_type = GENERALIZEDOPERATIONMAP.get(type(operation))

    if generalized_operation_type is None:
        raise TypeError("No matching generalized operation type.")

    parameters = [toproto_arg(param) for param in operation.getparams()]

    return circuit_pb2.Operation(
        generalizedoperation=circuit_pb2.GeneralizedOperation(
            mtype=generalized_operation_type,
            numqubits=operation.num_qubits,
            numbits=operation.num_bits,
            numzvars=operation.num_zvars,
            parameters=parameters,
        )
    )


def register_simple_annotations():
    """Register converter for all annotations in ANNOTATION_MAP."""
    for annotation_class in ANNOTATIONMAP.keys():
        operation_registry.register_toproto_direct(
            annotation_class, toproto_simpleannotation
        )


def toproto_simpleannotation(simpleannotation, declcache=None):
    annotation_type = ANNOTATIONMAP.get(type(simpleannotation))
    if annotation_type is None:
        raise TypeError("No matching annotation type.")

    notes = [toproto_note(note) for note in simpleannotation.get_notes()]
    return circuit_pb2.Operation(
        simpleannotation=circuit_pb2.SimpleAnnotation(
            mtype=annotation_type, notes=notes
        )
    )


def register_generalized_annotations():
    """Register converter for all annotations in GENERALIZEDANNOTATION_MAP."""
    for annotation_class in GENERALIZEDANNOTATIONMAP.keys():
        operation_registry.register_toproto_direct(
            annotation_class, toproto_generalizedannotation
        )


def toproto_generalizedannotation(generalizedannotation, declcache=None):
    annotation_type = GENERALIZEDANNOTATIONMAP.get(type(generalizedannotation))
    if annotation_type is None:
        raise TypeError("No matching generalized annotation type.")

    notes = [toproto_note(note) for note in generalizedannotation.get_notes()]

    return circuit_pb2.Operation(
        generalizedannotation=circuit_pb2.GeneralizedAnnotation(
            mtype=annotation_type,
            numqubits=generalizedannotation.num_qubits,
            numbits=generalizedannotation.num_bits,
            numzvars=generalizedannotation.num_zvars,
            notes=notes,
        )
    )


def toproto_operation(operation, declcache=None):
    """Convert an operation to protocol buffer format using registry pattern."""
    # Try to use a registered converter first
    converter = operation_registry.get_toproto_converter(type(operation))
    if converter:
        return converter(operation, declcache)

    # print(operation_registry.toproto_registry)

    if isinstance(operation, mc.krauschannel):
        kraus_proto = toproto_krauschannel(operation, declcache)
        kraus_field_name = kraus_proto.WhichOneof("krauschannel")
        return circuit_pb2.Operation(
            **{kraus_field_name: getattr(kraus_proto, kraus_field_name)}
        )

    # Check if the operation is a gate
    if isinstance(operation, mc.Gate):
        gate_proto = toproto_gate(operation, declcache)
        gate_field_name = gate_proto.WhichOneof("gate")
        return circuit_pb2.Operation(
            **{gate_field_name: getattr(gate_proto, gate_field_name)}
        )

    # Check if the operation is an operator (but not a gate)
    if isinstance(operation, mc.AbstractOperator) and not isinstance(
        operation, mc.Gate
    ):
        operator_proto = toproto_operator(operation, declcache)
        operator_field_name = operator_proto.WhichOneof("operator")
        return circuit_pb2.Operation(
            **{operator_field_name: getattr(operator_proto, operator_field_name)}
        )

    # Fall back to case-by-case handling
    raise ValueError(f"Unsupported operation type: {type(operation)}")


@operation_registry.register_fromproto("simpleoperation")
def fromproto_simpleoperation(simpleoperation_proto, declcache=None):
    args = [fromproto_arg(param) for param in simpleoperation_proto.parameters]
    operation_class = REVERSE_OPERATIONMAP.get(simpleoperation_proto.mtype)
    if operation_class:
        return operation_class(*args)
    raise ValueError(f"Unsupported SimpleOperation type: {simpleoperation_proto.mtype}")


@operation_registry.register_fromproto("generalizedoperation")
def fromproto_generalized_operation(generalizedoperation_proto, declcache=None):
    args = [fromproto_arg(param) for param in generalizedoperation_proto.parameters]
    operation_class = REVERSE_GENERALIZEDOPERATIONMAP.get(
        generalizedoperation_proto.mtype
    )

    nums = []
    if generalizedoperation_proto.numqubits != 0:
        nums.append(generalizedoperation_proto.numqubits)
    if generalizedoperation_proto.numbits != 0:
        nums.append(generalizedoperation_proto.numbits)
    if generalizedoperation_proto.numzvars != 0:
        nums.append(generalizedoperation_proto.numzvars)

    if operation_class:
        return operation_class(*nums, *args)
    raise ValueError(
        f"Unsupported SimpleOperation type: {generalizedoperation_proto.mtype}"
    )


@operation_registry.register_fromproto("simpleannotation")
def fromproto_simpleannotation(simpleannotation_proto, declcache=None):
    annotation_class = REVERSE_ANNOTATIONMAP.get(simpleannotation_proto.mtype)
    if annotation_class is None:
        raise ValueError("")

    notes = [fromproto_note(note) for note in simpleannotation_proto.notes]

    # Handle specific annotation classes
    if annotation_class is mc.Tick:
        return mc.Tick()
    elif annotation_class in {mc.ShiftCoordinates, mc.QubitCoordinates}:
        return annotation_class(notes)

    return annotation_class(*notes)


@operation_registry.register_fromproto("generalizedannotation")
def fromproto_generalizedannotation(generalizedannotation_proto, declcache=None):
    annotation_class = REVERSE_GENERALIZEDANNOTATIONMAP.get(
        generalizedannotation_proto.mtype
    )
    if annotation_class is None:
        raise ValueError("")

    notes = [fromproto_note(note) for note in generalizedannotation_proto.notes]

    nums = []
    if generalizedannotation_proto.numqubits != 0:
        nums.append(generalizedannotation_proto.numqubits)
    if generalizedannotation_proto.numbits != 0:
        nums.append(generalizedannotation_proto.numbits)
    if generalizedannotation_proto.numzvars != 0:
        nums.append(generalizedannotation_proto.numzvars)

    return annotation_class(*nums, notes)


def fromproto_operation(operation_proto, declcache=None):
    """Convert a protocol buffer operation to a mimiqcircuits operation."""
    # Determine which field is set
    operation_field = operation_proto.WhichOneof("operation")

    # Try to use a registered converter first
    converter = operation_registry.get_fromproto_converter(operation_field)
    if converter:
        return converter(getattr(operation_proto, operation_field), declcache)

    # Handle gate operations
    gate_fields = [
        "simplegate",
        "customgate",
        "control",
        "power",
        "inverse",
        "parallel",
        "paulistring",
        "gatecall",
        "cachedgatecall",
        "generalized",
    ]
    if operation_field in gate_fields:
        gate_proto = circuit_pb2.Gate(
            **{operation_field: getattr(operation_proto, operation_field)}
        )
        return fromproto_gate(gate_proto, declcache)

    # Handle kraus channel operations
    kraus_fields = [
        "simplekrauschannel",
        "customkrauschannel",
        "mixedunitarychannel",
        "paulichannel",
        "depolarizingchannel",
    ]
    if operation_field in kraus_fields:
        kraus_proto = circuit_pb2.KrausChannel(
            **{operation_field: getattr(operation_proto, operation_field)}
        )
        return fromproto_krauschannel(kraus_proto, declcache)

    # Handle operator operations
    operator_fields = ["simpleoperator", "customoperator", "rescaledgate"]
    if operation_field in operator_fields:
        operator_proto = circuit_pb2.Operator(
            **{operation_field: getattr(operation_proto, operation_field)}
        )
        return fromproto_operator(operator_proto, declcache)

    raise ValueError(f"Unsupported operation field in proto: {operation_field}")


# Register specific operation converters
@operation_registry.register_toproto(mc.Amplitude)
def toproto_amplitude(operation, declcache=None):
    """Convert an Amplitude operation to protocol buffer format."""
    bitstring_proto = toproto_bitvector(operation.bs)
    return circuit_pb2.Operation(amplitude=circuit_pb2.Amplitude(bs=bitstring_proto))


@operation_registry.register_toproto(mc.ExpectationValue)
def toproto_expectationvalue(operation, declcache=None):
    """Convert an ExpectationValue operation to protocol buffer format."""
    operator_proto = toproto_operator(operation.op, declcache)
    return circuit_pb2.Operation(
        expectationvalue=circuit_pb2.ExpectationValue(operator=operator_proto)
    )


@operation_registry.register_toproto(mc.IfStatement)
def toproto_ifstatement(operation, declcache=None):
    """Convert an IfStatement operation to protocol buffer format."""
    bitstring_proto = toproto_bitvector(operation.bitstring)
    operation_proto = toproto_operation(operation.op, declcache)
    return circuit_pb2.Operation(
        ifstatement=circuit_pb2.IfStatement(
            operation=operation_proto, bitstring=bitstring_proto
        )
    )


@operation_registry.register_toproto(mc.Not)
def toproto_not(operation, declcache=None):
    """Convert a Not operation to protocol buffer format."""
    return circuit_pb2.Operation(
        simpleoperation=circuit_pb2.SimpleOperation(mtype=circuit_pb2.OperationType.Not)
    )


@operation_registry.register_fromproto("amplitude")
def fromproto_amplitude(amplitude_proto, declcache=None):
    """Convert a protocol buffer Amplitude to a mimiqcircuits Amplitude."""
    bitstring = mc.BitString(fromproto_bitvector(amplitude_proto.bs))
    return mc.Amplitude(bitstring)


@operation_registry.register_fromproto("expectationvalue")
def fromproto_expectationvalue(expectationvalue_proto, declcache=None):
    """Convert a protocol buffer ExpectationValue to a mimiqcircuits ExpectationValue."""
    operator = fromproto_operator(expectationvalue_proto.operator, declcache)
    return mc.ExpectationValue(operator)


@operation_registry.register_fromproto("ifstatement")
def fromproto_ifstatement(ifstatement_proto, declcache=None):
    """Convert a protocol buffer IfStatement to a mimiqcircuits IfStatement."""
    operation = fromproto_operation(ifstatement_proto.operation, declcache)
    bitstring = mc.BitString(fromproto_bitvector(ifstatement_proto.bitstring))
    return mc.IfStatement(operation=operation, bitstring=bitstring)


# ------------ Block & Repeat -----------


@operation_registry.register_toproto(mc.Block)
def toproto_block(op, declcache=None):
    proto_insts = [
        toproto_instruction(
            mc.Instruction(
                inst.operation,
                qubits=tuple(inst.qubits),
                bits=tuple(inst.bits),
                zvars=tuple(inst.zvars),
            ),
            declcache,
        )
        for inst in op.instructions
    ]
    block_msg = circuit_pb2.Block(
        numqubits=op.num_qubits,
        numbits=op.num_bits,
        numzvars=op.num_zvars,
        instructions=proto_insts,
    )

    return circuit_pb2.Operation(block=block_msg)


@operation_registry.register_fromproto("block")
def fromproto_block(proto, declcache=None):
    instructions = [
        fromproto_instruction(inst, declcache) for inst in proto.instructions
    ]
    return mc.Block(proto.numqubits, proto.numbits, proto.numzvars, instructions)


@operation_registry.register_toproto(mc.Repeat)
def toproto_repeat(op, declcache=None):
    proto_op = toproto_operation(op.get_operation(), declcache)
    return circuit_pb2.Operation(
        repeat=circuit_pb2.Repeat(numrepeats=op.repeats, operation=proto_op)
    )


@operation_registry.register_fromproto("repeat")
def fromproto_repeat(proto, declcache=None):
    inner_op = fromproto_operation(proto.operation, declcache)
    return mc.Repeat(proto.numrepeats, inner_op)


# ---------------------- Instruction Conversion Functions ----------------------


def toproto_instruction(inst, declcache=None):
    """Convert an instruction to protocol buffer format."""
    return circuit_pb2.Instruction(
        operation=toproto_operation(inst.operation, declcache),
        qtargets=[x + 1 for x in inst.qubits],
        ctargets=[x + 1 for x in inst.bits],
        ztargets=[x + 1 for x in inst.zvars],
    )


def fromproto_instruction(inst_proto, declcache=None):
    """Convert a protocol buffer instruction to a mimiqcircuits instruction."""
    return mc.Instruction(
        operation=fromproto_operation(inst_proto.operation, declcache),
        qubits=tuple([x - 1 for x in inst_proto.qtargets]),
        bits=tuple([x - 1 for x in inst_proto.ctargets]),
        zvars=tuple([x - 1 for x in inst_proto.ztargets]),
    )


# ---------------------- Circuit Conversion Functions ----------------------


def toproto_circuit(circuit):
    """Convert a circuit to protocol buffer format."""
    declcache = ({}, [])
    instructions_proto = [
        toproto_instruction(inst, declcache) for inst in circuit.instructions
    ]
    decls_map = {}
    for k, decl in declcache[0].items():
        if isinstance(decl, circuit_pb2.GateDecl):
            decls_map[k] = circuit_pb2.Declaration(gatedecl=decl)
        elif isinstance(decl, circuit_pb2.Block):
            decls_map[k] = circuit_pb2.Declaration(block=decl)
        else:
            raise ValueError(f"Unsupported declaration type: {type(decl)}")

    return circuit_pb2.Circuit(
        instructions=instructions_proto, decls=decls_map, declorder=declcache[1]
    )


def fromproto_circuit(circuit_proto):
    """Convert a protocol buffer circuit to a mimiqcircuits circuit."""
    declcache = ({}, circuit_proto.declorder)
    for k in circuit_proto.declorder:
        decl = circuit_proto.decls[k]
        which = decl.WhichOneof("decl")

        if which == "gatedecl":
            declcache[0][k] = fromproto_gatedecl(decl.gatedecl, declcache)
        elif which == "block":
            declcache[0][k] = fromproto_block(decl.block, declcache)
        else:
            raise ValueError(f"Unknown declaration type in proto: {which}")

    instructions = [
        fromproto_instruction(inst, declcache) for inst in circuit_proto.instructions
    ]
    return mc.Circuit(instructions)


# --- Gate Conversion for RPauli ---
@gate_registry.register_toproto(RPauli)
def toproto_rpauli(gate, declcache=None):
    return circuit_pb2.Gate(
        rpauli=circuit_pb2.RPauli(
            pauli=pauli_pb2.PauliString(
                numqubits=gate.pauli.num_qubits, pauli=str(gate.pauli.pauli)
            ),
            theta=toproto_arg(gate.theta),
        )
    )


@operation_registry.register_fromproto("rpauli")
def fromproto_rpauli_operation(msg, declcache=None):
    return mc.RPauli(fromproto_paulistring(msg.pauli), fromproto_arg(msg.theta))


# ---------------------- Initialize Registries ----------------------


def initialize_registries():
    """Initialize all converter registries."""
    register_generalized_gates()
    register_simple_operations()
    register_generalized_operations()
    register_simple_annotations()
    register_generalized_annotations()
    # Additional initialization can be added here


initialize_registries()
