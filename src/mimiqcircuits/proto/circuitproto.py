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

from mimiqcircuits.bitstrings import BitString
from mimiqcircuits.blockdecl import BlockCall, BlockDecl
from mimiqcircuits.circuit import Circuit
from mimiqcircuits.classical.and_operation import And
from mimiqcircuits.classical.not_operation import Not
from mimiqcircuits.classical.or_operation import Or
from mimiqcircuits.classical.setbits import SetBit0, SetBit1
from mimiqcircuits.classical.xor_operation import Xor
from mimiqcircuits.complex import Add, Multiply, Pow
from mimiqcircuits.gatedecl import GateCall, GateDecl
from mimiqcircuits.instruction import Instruction
from mimiqcircuits.operations.amplitude import Amplitude
from mimiqcircuits.operations.annotations import (
    AncillaAlloc,
    AncillaFree,
    Detector,
    Lost,
    ObservableInclude,
    QubitCoordinates,
    Reloaded,
    ShiftCoordinates,
    Tick,
)
from mimiqcircuits.operations.barrier import Barrier
from mimiqcircuits.operations.block import Block
from mimiqcircuits.operations.control import Control
from mimiqcircuits.operations.entanglement import (
    BondDim,
    SchmidtRank,
    VonNeumannEntropy,
)
from mimiqcircuits.operations.expectationvalue import ExpectationValue
from mimiqcircuits.operations.gates.custom import GateCustom
from mimiqcircuits.operations.gates.customdiagonal import GateCustomDiagonal
from mimiqcircuits.operations.gates.delay import Delay
from mimiqcircuits.operations.gates.gate import Gate
from mimiqcircuits.operations.gates.generalized.diffusion import Diffusion
from mimiqcircuits.operations.gates.generalized.paulistring import PauliString
from mimiqcircuits.operations.gates.generalized.phasegradient import PhaseGradient
from mimiqcircuits.operations.gates.generalized.polynomialoracle import PolynomialOracle
from mimiqcircuits.operations.gates.generalized.qft import QFT
from mimiqcircuits.operations.gates.generalized.rnz import GateRNZ
from mimiqcircuits.operations.gates.standard.dcx import GateDCX
from mimiqcircuits.operations.gates.standard.deprecated import GateU1, GateU2, GateU3
from mimiqcircuits.operations.gates.standard.ecr import GateECR
from mimiqcircuits.operations.gates.standard.hadamard import GateH, GateHXY, GateHYZ
from mimiqcircuits.operations.gates.standard.id import GateID
from mimiqcircuits.operations.gates.standard.interactions import (
    GateRXX,
    GateRYY,
    GateRZX,
    GateRZZ,
    GateXXminusYY,
    GateXXplusYY,
)
from mimiqcircuits.operations.gates.standard.iswap import GateISWAP
from mimiqcircuits.operations.gates.standard.pauli import GateX, GateY, GateZ
from mimiqcircuits.operations.gates.standard.phase import GateP
from mimiqcircuits.operations.gates.standard.rotations import (
    GateR,
    GateRX,
    GateRY,
    GateRZ,
)
from mimiqcircuits.operations.gates.standard.s import GateS
from mimiqcircuits.operations.gates.standard.swap import GateSWAP
from mimiqcircuits.operations.gates.standard.t import GateT
from mimiqcircuits.operations.gates.standard.u import GateU
from mimiqcircuits.operations.ifstatement import IfStatement
from mimiqcircuits.operations.inverse import Inverse
from mimiqcircuits.operations.krauschannel import krauschannel
from mimiqcircuits.operations.losschannel import Check, Loss, MeasureCheck, Reload
from mimiqcircuits.operations.magic import PrepareStateT
from mimiqcircuits.operations.measure import Measure, MeasureX, MeasureY, MeasureZ
from mimiqcircuits.operations.measurepauli import MeasurePauli
from mimiqcircuits.operations.measurereset import (
    MeasureReset,
    MeasureResetX,
    MeasureResetY,
)
from mimiqcircuits.operations.noisechannel.kraus import Kraus
from mimiqcircuits.operations.noisechannel.mixedunitary import MixedUnitary
from mimiqcircuits.operations.noisechannel.standards.ampdamping import (
    AmplitudeDamping,
    GeneralizedAmplitudeDamping,
)
from mimiqcircuits.operations.noisechannel.standards.depolarizing import Depolarizing
from mimiqcircuits.operations.noisechannel.standards.pauli import (
    PauliNoise,
    PauliX,
    PauliY,
    PauliZ,
)
from mimiqcircuits.operations.noisechannel.standards.phaseamplitudedamping import (
    PhaseAmplitudeDamping,
    ThermalNoise,
)
from mimiqcircuits.operations.noisechannel.standards.projectivenoise import (
    ProjectiveNoise,
    ProjectiveNoiseX,
    ProjectiveNoiseY,
    ProjectiveNoiseZ,
)
from mimiqcircuits.operations.operator import AbstractOperator
from mimiqcircuits.operations.operators.custom import Operator
from mimiqcircuits.operations.operators.diagonals import DiagonalOp
from mimiqcircuits.operations.operators.lossyoperator import LossyOperator
from mimiqcircuits.operations.operators.projectors import (
    Projector0,
    Projector00,
    Projector01,
    Projector1,
    Projector10,
    Projector11,
    ProjectorX0,
    ProjectorX1,
    ProjectorY0,
    ProjectorY1,
)
from mimiqcircuits.operations.operators.sigmas import SigmaMinus, SigmaPlus
from mimiqcircuits.operations.pairmeasure import MeasureXX, MeasureYY, MeasureZZ
from mimiqcircuits.operations.parallel import Parallel
from mimiqcircuits.operations.power import Power
from mimiqcircuits.operations.readouterr import ReadoutErr
from mimiqcircuits.operations.repeat import Repeat
from mimiqcircuits.operations.rescaledgates import RescaledGate
from mimiqcircuits.operations.reset import Reset, ResetX, ResetY, ResetZ
from mimiqcircuits.operations.whilestatement import WhileStatement
from mimiqcircuits.proto import circuit_pb2, pauli_pb2
from mimiqcircuits.proto.bitvector import fromproto_bitvector, toproto_bitvector
from mimiqcircuits.operations.gates.generalized.rpauli import RPauli

# ---------------------- Registry Pattern for Conversions ----------------------


_MISSING_CONVERTER = object()


class ProtoRegistry:
    """Registry for proto conversion functions based on type."""

    def __init__(self):
        self.toproto_registry = {}
        self.fromproto_registry = {}
        # Memoizes get_toproto_converter results (including MRO misses) so
        # repeated lookups for the same type skip the MRO walk.
        self._toproto_cache = {}

    def register_toproto(self, mc_type):
        """Register a function to convert from mimiqcircuits to protobuf.

        Can be used as a decorator:
            @registry.register_toproto(mc.PauliString)
            def convert_paulistring(...):
                ...
        """

        def decorator(converter_func):
            self.toproto_registry[mc_type] = converter_func
            self._toproto_cache.clear()
            return converter_func

        return decorator

    def register_toproto_direct(self, mc_type, converter_func):
        """Register a function to convert from mimiqcircuits to protobuf."""
        self.toproto_registry[mc_type] = converter_func
        self._toproto_cache.clear()

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
        """Get converter for a specific mimiqcircuits type.

        Traverses the type's MRO to find a converter, so canonical subclasses
        (e.g., GateCH, a subclass of Control) use the parent class's converter.
        Results — including misses — are cached; the cache is cleared whenever a
        new converter is registered.
        """
        cache = self._toproto_cache
        cached = cache.get(obj_type, _MISSING_CONVERTER)
        if cached is not _MISSING_CONVERTER:
            return cached

        converter = self.toproto_registry.get(obj_type)
        if converter is None:
            for parent_type in obj_type.__mro__[1:]:
                converter = self.toproto_registry.get(parent_type)
                if converter is not None:
                    break

        cache[obj_type] = converter
        return converter

    def get_fromproto_converter(self, proto_field: str) -> Optional[Callable]:
        """Get converter for a specific protobuf field."""
        return self.fromproto_registry.get(proto_field)


# Create registries for different object types
gate_registry = ProtoRegistry()
operation_registry = ProtoRegistry()
krauschannel_registry = ProtoRegistry()
operator_registry = ProtoRegistry()

# ---------------------- Constants and Mappings ----------------------

# Gate mapping: mimiqcircuits -> protobuf
GATEMAP = {
    GateID: circuit_pb2.GateType.GateID,
    GateX: circuit_pb2.GateType.GateX,
    GateY: circuit_pb2.GateType.GateY,
    GateZ: circuit_pb2.GateType.GateZ,
    GateH: circuit_pb2.GateType.GateH,
    GateHXY: circuit_pb2.GateType.GateHXY,
    GateHYZ: circuit_pb2.GateType.GateHYZ,
    GateS: circuit_pb2.GateType.GateS,
    GateT: circuit_pb2.GateType.GateT,
    Delay: circuit_pb2.GateType.Delay,
    GateU: circuit_pb2.GateType.GateU,
    GateP: circuit_pb2.GateType.GateP,
    GateRX: circuit_pb2.GateType.GateRX,
    GateRY: circuit_pb2.GateType.GateRY,
    GateRZ: circuit_pb2.GateType.GateRZ,
    GateR: circuit_pb2.GateType.GateR,
    GateU1: circuit_pb2.GateType.GateU1,
    GateU2: circuit_pb2.GateType.GateU2,
    GateU3: circuit_pb2.GateType.GateU3,
    GateSWAP: circuit_pb2.GateType.GateSWAP,
    GateISWAP: circuit_pb2.GateType.GateISWAP,
    GateECR: circuit_pb2.GateType.GateECR,
    GateDCX: circuit_pb2.GateType.GateDCX,
    GateRXX: circuit_pb2.GateType.GateRXX,
    GateRYY: circuit_pb2.GateType.GateRYY,
    GateRZZ: circuit_pb2.GateType.GateRZZ,
    GateRZX: circuit_pb2.GateType.GateRZX,
    GateXXplusYY: circuit_pb2.GateType.GateXXplusYY,
    GateXXminusYY: circuit_pb2.GateType.GateXXminusYY,
}
#
# Reverse gate mapping: protobuf -> mimiqcircuits
REVERSE_GATEMAP = {v: k for k, v in GATEMAP.items()}

# Operation mapping: mimiqcircuits -> protobuf
OPERATIONMAP = {
    MeasureX: circuit_pb2.OperationType.MeasureX,
    MeasureY: circuit_pb2.OperationType.MeasureY,
    MeasureZ: circuit_pb2.OperationType.MeasureZ,
    Measure: circuit_pb2.OperationType.MeasureZ,
    MeasureXX: circuit_pb2.OperationType.MeasureXX,
    MeasureYY: circuit_pb2.OperationType.MeasureYY,
    MeasureZZ: circuit_pb2.OperationType.MeasureZZ,
    MeasureResetX: circuit_pb2.OperationType.MeasureResetX,
    MeasureResetY: circuit_pb2.OperationType.MeasureResetY,
    MeasureReset: circuit_pb2.OperationType.MeasureResetZ,
    BondDim: circuit_pb2.OperationType.BondDim,
    SchmidtRank: circuit_pb2.OperationType.SchmidtRank,
    VonNeumannEntropy: circuit_pb2.OperationType.VonNeumannEntropy,
    Not: circuit_pb2.OperationType.Not,
    Pow: circuit_pb2.OperationType.Pow,
    SetBit0: circuit_pb2.OperationType.SetBit0,
    SetBit1: circuit_pb2.OperationType.SetBit1,
    Reload: circuit_pb2.OperationType.Reload,
    Check: circuit_pb2.OperationType.Check,
    MeasureCheck: circuit_pb2.OperationType.MeasureCheck,
}

# Reverse operation mapping: protobuf -> mimiqcircuits
REVERSE_OPERATIONMAP = {v: k for k, v in OPERATIONMAP.items()}

# Generalized operation mapping: mimiqcircuits -> protobuf
GENERALIZEDOPERATIONMAP = {
    Barrier: circuit_pb2.GeneralizedOperationType.Barrier,
    Add: circuit_pb2.GeneralizedOperationType.Add,
    Multiply: circuit_pb2.GeneralizedOperationType.Multiply,
    And: circuit_pb2.GeneralizedOperationType.And,
    Or: circuit_pb2.GeneralizedOperationType.Or,
    Xor: circuit_pb2.GeneralizedOperationType.Xor,
}

# Reverse generalized operation mapping: protobuf -> mimiqcircuits
REVERSE_GENERALIZEDOPERATIONMAP = {v: k for k, v in GENERALIZEDOPERATIONMAP.items()}
# Files written before ParityCheck was deprecated still carry its enum value.
REVERSE_GENERALIZEDOPERATIONMAP[circuit_pb2.GeneralizedOperationType.ParityCheck] = Xor

# Kraus Channel mapping: mimiqcircuits -> protobuf
KRAUSCHANNELMAP = {
    Reset: circuit_pb2.KrausChannelType.ResetZ,
    ResetZ: circuit_pb2.KrausChannelType.ResetZ,
    ResetX: circuit_pb2.KrausChannelType.ResetX,
    ResetY: circuit_pb2.KrausChannelType.ResetY,
    ResetZ: circuit_pb2.KrausChannelType.ResetZ,
    AmplitudeDamping: circuit_pb2.KrausChannelType.AmplitudeDamping,
    GeneralizedAmplitudeDamping: circuit_pb2.KrausChannelType.GeneralizedAmplitudeDamping,
    PhaseAmplitudeDamping: circuit_pb2.KrausChannelType.PhaseAmplitudeDamping,
    ThermalNoise: circuit_pb2.KrausChannelType.ThermalNoise,
    PauliX: circuit_pb2.KrausChannelType.PauliX,
    PauliY: circuit_pb2.KrausChannelType.PauliY,
    PauliZ: circuit_pb2.KrausChannelType.PauliZ,
    ProjectiveNoiseX: circuit_pb2.KrausChannelType.ProjectiveNoiseX,
    ProjectiveNoiseY: circuit_pb2.KrausChannelType.ProjectiveNoiseY,
    ProjectiveNoiseZ: circuit_pb2.KrausChannelType.ProjectiveNoiseZ,
    ProjectiveNoise: circuit_pb2.KrausChannelType.ProjectiveNoiseZ,
    PrepareStateT: circuit_pb2.KrausChannelType.PrepareStateT,
}

# Reverse Kraus Channel mapping: protobuf -> mimiqcircuits
REVERSE_KRAUSCHANNELMAP = {v: k for k, v in KRAUSCHANNELMAP.items()}

# Operator mapping: mimiqcircuits -> protobuf
OPERATORMAP = {
    SigmaMinus: circuit_pb2.OperatorType.SigmaMinus,
    SigmaPlus: circuit_pb2.OperatorType.SigmaPlus,
    Projector0: circuit_pb2.OperatorType.Projector0,
    Projector1: circuit_pb2.OperatorType.Projector1,
    Projector00: circuit_pb2.OperatorType.Projector00,
    Projector01: circuit_pb2.OperatorType.Projector01,
    Projector10: circuit_pb2.OperatorType.Projector10,
    Projector11: circuit_pb2.OperatorType.Projector11,
    ProjectorX0: circuit_pb2.OperatorType.ProjectorX0,
    ProjectorX1: circuit_pb2.OperatorType.ProjectorX1,
    ProjectorY0: circuit_pb2.OperatorType.ProjectorY0,
    ProjectorY1: circuit_pb2.OperatorType.ProjectorY1,
    DiagonalOp: circuit_pb2.OperatorType.DiagonalOp,
}

# Reverse operator mapping: protobuf -> mimiqcircuits
REVERSE_OPERATORMAP = {v: k for k, v in OPERATORMAP.items()}

# Generalized gate mappings
GENERALIZEDGATEMAP = {
    QFT: circuit_pb2.GeneralizedType.QFT,
    PhaseGradient: circuit_pb2.GeneralizedType.PhaseGradient,
    PolynomialOracle: circuit_pb2.GeneralizedType.PolynomialOracle,
    Diffusion: circuit_pb2.GeneralizedType.Diffusion,
    GateRNZ: circuit_pb2.GeneralizedType.GateRNZ,
}

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
    QubitCoordinates: circuit_pb2.AnnotationType.QubitCoordinates,
    ShiftCoordinates: circuit_pb2.AnnotationType.ShiftCoordinates,
    Tick: circuit_pb2.AnnotationType.Tick,
    Lost: circuit_pb2.AnnotationType.Lost,
    Reloaded: circuit_pb2.AnnotationType.Reloaded,
    AncillaAlloc: circuit_pb2.AnnotationType.AncillaAlloc,
    AncillaFree: circuit_pb2.AnnotationType.AncillaFree,
}
# Reverse annotation mappings: protobuf -> mimiqcircuits
REVERSE_ANNOTATIONMAP = {v: k for k, v in ANNOTATIONMAP.items()}

# Generalized annotation mappings
GENERALIZEDANNOTATIONMAP = {
    Detector: circuit_pb2.GeneralizedAnnotationType.Detector,
    ObservableInclude: circuit_pb2.GeneralizedAnnotationType.ObservableInclude,
}

# Reverse generalized annotation mappings
REVERSE_GENERALIZEDANNOTATIONMAP = {v: k for k, v in GENERALIZEDANNOTATIONMAP.items()}

# ---------------------- Base Conversion Functions ----------------------


def safe_convert(value):
    """Convert float to int if it represents an integer value."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


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


def _arg_int(p):
    return circuit_pb2.Arg(argvalue_value=circuit_pb2.ArgValue(integer_value=p))


def _arg_se_integer(p):
    return circuit_pb2.Arg(argvalue_value=circuit_pb2.ArgValue(integer_value=p.p))


def _arg_se_realdouble(p):
    return circuit_pb2.Arg(argvalue_value=circuit_pb2.ArgValue(double_value=p.real))


def _arg_se_symbol(p):
    return circuit_pb2.Arg(symbol_value=circuit_pb2.Symbol(value=p.name))


def _arg_string(p):
    return circuit_pb2.Arg(symbol_value=circuit_pb2.Symbol(value=p))


def _arg_function(mtype, args):
    return circuit_pb2.Arg(
        argfunction_value=circuit_pb2.ArgFunction(mtype=mtype, args=args)
    )


def _arg_rational(num, den):
    # `Arg` has no rational variant: a rational `p/q` travels as `DIV(p, q)`
    # of two integers, which the decoders read back as an exact rational.
    if den == 1:
        return _arg_int(int(num))
    return _arg_function(
        circuit_pb2.FunctionType.DIV, [_arg_int(int(num)), _arg_int(int(den))]
    )


def _argbytes(arg):
    return arg.SerializeToString(deterministic=True)


def _sorted_args(args):
    # Symbolics and SymEngine order the terms of a sum or product differently.
    # Sorting them by their encoding gives both languages the same bytes; the
    # decoders rebuild through `+` and `*`, so the order does not change the
    # value.
    return sorted(args, key=_argbytes)


def _arg_se_add(p):
    return _arg_function(
        circuit_pb2.FunctionType.ADD, _sorted_args(map(toproto_arg, p.args))
    )


def _arg_se_quotient(p):
    # SymEngine stores `x/y` as `Mul(x, Pow(y, -1))` and `1/x` as `Pow(x, -1)`.
    # The wire format writes a quotient with a non-numeric denominator as
    # `DIV(num, den)`. A numeric denominator is
    # left to the product and rational rules. Returns `None` when `p` is not
    # such a quotient. Powers go through `_arg_se_pow`.
    num, den = p.as_numer_denom()
    if den.is_Number:
        return None
    return _arg_function(
        circuit_pb2.FunctionType.DIV, [toproto_arg(num), toproto_arg(den)]
    )


def _arg_se_pow(p):
    num, den = p.as_numer_denom()
    if not den.is_Number:
        # The numerator of a power is always 1, but SymEngine returns `1.0`
        # for a float exponent such as `x**-0.5`.
        return _arg_function(
            circuit_pb2.FunctionType.DIV, [_arg_int(1), toproto_arg(den)]
        )
    return _arg_function(circuit_pb2.FunctionType.POW, [toproto_arg(a) for a in p.args])


def _arg_se_mul(p):
    # SymEngine stores `pi/4` as `Mul(1/4, pi)`. The wire format writes a
    # single factor divided by an integer as `DIV(pi, 4)`, and folds any other
    # rational into the product's coefficient, so the expression a user writes
    # as `pi/4` always gives the same bytes and decodes to what the user wrote.
    quotient = _arg_se_quotient(p)
    if quotient is not None:
        return quotient
    args = p.args
    coeff = args[0] if args[0].is_Number else None
    factors = args[1:] if coeff is not None else args
    if (
        isinstance(coeff, se.Rational)
        and coeff.p == 1
        and coeff.q != 1
        and len(factors) == 1
    ):
        return _arg_function(
            circuit_pb2.FunctionType.DIV, [toproto_arg(factors[0]), _arg_int(int(coeff.q))]
        )
    return _arg_function(
        circuit_pb2.FunctionType.MUL, _sorted_args(map(toproto_arg, args))
    )


# Direct type -> handler table for the common cases in toproto_arg. Built once
# at module load (the original code rebuilt it per call).
_TOPROTO_ARG_HANDLERS = {
    int: _arg_int,
    float: handle_float,
    se.Integer: _arg_se_integer,
    se.Rational: lambda p: _arg_rational(p.p, p.q),
    type(se.Rational(1, 2)): lambda p: _arg_rational(p.p, p.q),
    Fraction: lambda p: _arg_rational(p.numerator, p.denominator),
    se.Add: _arg_se_add,
    se.Mul: _arg_se_mul,
    se.Pow: _arg_se_pow,
    se.Float: lambda p: handle_numeric(float(p)),
    se.RealDouble: _arg_se_realdouble,
    se.Symbol: _arg_se_symbol,
    complex: handle_complex,
    se.ComplexDouble: lambda p: handle_complex(p.evalf()),
    str: _arg_string,
    seZero: lambda p: _arg_int(0),
    seOne: lambda p: _arg_int(1),
    seNegativeOne: lambda p: _arg_int(-1),
    np.float64: lambda p: handle_float(float(p)),
    np.float32: lambda p: handle_float(float(p)),
    np.int64: lambda p: _arg_int(int(p)),
    np.int32: lambda p: _arg_int(int(p)),
}


def toproto_arg(param):
    """Convert a parameter to protocol buffer format."""
    handler = _TOPROTO_ARG_HANDLERS.get(type(param))
    if handler is not None:
        return handler(param)

    # Slow paths for irrationals, symengine expressions, and other symbolic
    # values that can't be keyed on exact type.
    if param in IRRATIONALSYMENGINE:
        arg = circuit_pb2.Arg()
        arg.irrational_value = IRRATIONALSYMENGINE[param]
        return arg

    param_type = type(param)
    if param_type in EXPRSYMENGINE:
        arg = circuit_pb2.Arg()
        arg.argfunction_value.mtype = EXPRSYMENGINE[param_type]
        for a in param.args:
            arg.argfunction_value.args.append(toproto_arg(a))
        return arg

    if isinstance(param, se.Rational):
        return _arg_rational(param.p, param.q)

    if isinstance(param, se.Basic):
        if param == se.I:
            return handle_complex(param)
        if param.is_Number:
            try:
                return handle_numeric(float(param))
            except Exception:
                pass

    raise ValueError(f"Unsupported parameter type: {type(param)}")


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
@gate_registry.register_toproto(PauliString)
def toproto_paulistring(gate, declcache=None):
    """Convert a PauliString to protocol buffer format."""
    return circuit_pb2.Gate(
        paulistring=pauli_pb2.PauliString(numqubits=gate.num_qubits, pauli=gate.pauli)
    )


@gate_registry.register_toproto(GateCustom)
def toproto_customgate(gate, declcache=None):
    """Convert a GateCustom to protocol buffer format."""
    U = [toproto_complex(param) for param in gate._U.T]
    return circuit_pb2.Gate(
        customgate=circuit_pb2.CustomGate(numqubits=gate.num_qubits, matrix=U)
    )


@gate_registry.register_toproto(GateCustomDiagonal)
def toproto_customdiagonalgate(gate, declcache=None):
    """Convert a GateCustomDiagonal to protocol buffer format."""
    d = [toproto_complex(entry) for entry in gate._d]
    return circuit_pb2.Gate(
        customdiagonalgate=circuit_pb2.CustomDiagonalGate(
            numqubits=gate.num_qubits, diagonal=d
        )
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
        name=gate.name,
        args=arguments,
        instructions=instructions_proto,
        numancillas=gate.num_ancillas,
    )


@gate_registry.register_toproto(GateCall)
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


@gate_registry.register_toproto(Control)
def toproto_control(gate, declcache=None):
    """Convert a Control gate to protocol buffer format."""
    control_gate_proto = toproto_gate(gate.op, declcache)
    return circuit_pb2.Gate(
        control=circuit_pb2.Control(
            operation=control_gate_proto, numcontrols=gate.num_controls
        )
    )


@gate_registry.register_toproto(Power)
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


@gate_registry.register_toproto(Inverse)
def toproto_inverse(gate, declcache=None):
    """Convert an Inverse gate to protocol buffer format."""
    base_gate_proto = toproto_gate(gate.op, declcache)
    return circuit_pb2.Gate(inverse=circuit_pb2.Inverse(operation=base_gate_proto))


@gate_registry.register_toproto(Parallel)
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
    if isinstance(gate, PolynomialOracle):
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
    return PauliString(paulistring_proto.pauli)


@gate_registry.register_fromproto("customgate")
def fromproto_customgate(customgate_proto, declcache=None):
    """Convert a protocol buffer CustomGate to a mimiqcircuits GateCustom."""
    U_matrix = [fromproto_complex(val) for val in customgate_proto.matrix]
    matrix_size = 2**customgate_proto.numqubits
    original_matrix = np.array(U_matrix).reshape(matrix_size, matrix_size).T
    return GateCustom(matrix=original_matrix)


@gate_registry.register_fromproto("customdiagonalgate")
def fromproto_customdiagonalgate(customdiagonalgate_proto, declcache=None):
    """Convert a protocol buffer CustomDiagonalGate to a GateCustomDiagonal."""
    d = [fromproto_complex(val) for val in customdiagonalgate_proto.diagonal]
    return GateCustomDiagonal(d)


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
    return Control(control_proto.numcontrols, control_gate)


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

    return Power(base_gate, power_value)


@gate_registry.register_fromproto("inverse")
def fromproto_inverse(inverse_proto, declcache=None):
    """Convert a protocol buffer Inverse gate to a mimiqcircuits Inverse gate."""
    base_gate = fromproto_gate(inverse_proto.operation, declcache)
    return Inverse(base_gate)


@gate_registry.register_fromproto("parallel")
def fromproto_parallel(parallel_proto, declcache=None):
    """Convert a protocol buffer Parallel gate to a mimiqcircuits Parallel gate."""
    base_gate = fromproto_gate(parallel_proto.operation, declcache)
    return Parallel(parallel_proto.numrepeats, base_gate)


@gate_registry.register_fromproto("gatecall")
def fromproto_gatecall(gatecall_proto, declcache=None):
    """Convert a protocol buffer GateCall to a mimiqcircuits GateCall."""
    # Deserialize the GateDecl using the correct function
    decl = fromproto_gatedecl(gatecall_proto.decl)
    args = [fromproto_arg(arg) for arg in gatecall_proto.args]
    return GateCall(decl, args)


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
    return GateCall(decl, args)


def fromproto_gatedecl(gatedecl_proto, declcache=None):
    """Convert a protocol buffer GateDecl to a mimiqcircuits GateDecl."""
    name = gatedecl_proto.name
    arguments = [se.Symbol(symbol.value) for symbol in gatedecl_proto.args]
    instructions = [
        fromproto_instruction(inst, declcache) for inst in gatedecl_proto.instructions
    ]

    return GateDecl(
        name=name,
        arguments=tuple(arguments),
        circuit=Circuit(instructions),
        num_ancillas=gatedecl_proto.numancillas,
    )


# ---------------------- Operator Conversion Functions ----------------------


def toproto_operator(operator, declcache=None):
    """Convert an operator to protocol buffer format using registry pattern."""
    # Try to use a registered converter first
    converter = operator_registry.get_toproto_converter(type(operator))
    if converter:
        return converter(operator, declcache)

    # Simple operator types from the mapping
    if isinstance(operator, AbstractOperator):
        operator_type = OPERATORMAP.get(type(operator))
        if operator_type is not None:
            params = [toproto_arg(param) for param in operator.getparams()]
            return circuit_pb2.Operator(
                simpleoperator=circuit_pb2.SimpleOperator(
                    mtype=operator_type, parameters=params
                )
            )

    # Fall back to case-by-case handling
    if isinstance(operator, Gate):
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
        "customdiagonalgate",
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
@operator_registry.register_toproto(RescaledGate)
def toproto_rescaledgate(operator, declcache=None):
    """Convert a RescaledGate to protocol buffer format."""
    base_gate_proto = toproto_gate(operator.get_operation(), declcache)
    scale_proto = toproto_arg(operator.get_scale())
    return circuit_pb2.Operator(
        rescaledgate=circuit_pb2.RescaledGate(
            operation=base_gate_proto, scale=scale_proto
        )
    )


@operator_registry.register_toproto(Operator)
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


@operator_registry.register_toproto(LossyOperator)
def toproto_lossyoperator(operator, declcache=None):
    matrix_data = operator.matrix()
    matrix = [
        toproto_complex(matrix_data[i, j])
        for i in range(matrix_data.rows)
        for j in range(matrix_data.cols)
    ]
    return circuit_pb2.Operator(
        lossyoperator=circuit_pb2.LossyOperator(
            numqubits=operator.num_qubits,
            matrix=matrix,
            lossy=list(operator.lossyqubits()),
        )
    )


@operator_registry.register_fromproto("rescaledgate")
def fromproto_rescaledgate(rescaledgate_proto, declcache=None):
    """Convert a protocol buffer RescaledGate to a mimiqcircuits RescaledGate."""
    base_gate = fromproto_gate(rescaledgate_proto.operation, declcache)
    scale = fromproto_arg(rescaledgate_proto.scale)
    return RescaledGate(base_gate, scale)


@operator_registry.register_fromproto("customoperator")
def fromproto_customoperator(customoperator_proto, declcache=None):
    """Convert a protocol buffer CustomOperator to a mimiqcircuits Operator."""
    matrix = [fromproto_complex(val) for val in customoperator_proto.matrix]
    matrix = np.array(matrix).reshape(
        2**customoperator_proto.numqubits,
        2**customoperator_proto.numqubits,
    )
    return Operator(mat=matrix)


@operator_registry.register_fromproto("lossyoperator")
def fromproto_lossyoperator(lossyoperator_proto, declcache=None):
    matrix = [fromproto_complex(val) for val in lossyoperator_proto.matrix]
    size = 2**lossyoperator_proto.numqubits
    matrix = np.array(matrix).reshape(size, size)
    return LossyOperator(mat=matrix, lossy=tuple(lossyoperator_proto.lossy))


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
@krauschannel_registry.register_toproto(Kraus)
def toproto_kraus(kraus_channel, declcache=None):
    """Convert a Kraus to protocol buffer format."""
    operators_proto = [
        toproto_operator(op, declcache) for op in kraus_channel.krausoperators()
    ]
    return circuit_pb2.KrausChannel(
        customkrauschannel=circuit_pb2.CustomKrausChannel(
            numqubits=kraus_channel.num_qubits, operators=operators_proto
        )
    )


# Per-branch lossy qubit masks: qubit q (1-based) <-> bit q-1.
def _lossy_to_mask(lossy):
    mask = 0
    for q in lossy:
        mask |= 1 << (q - 1)
    return mask


def _mask_to_lossy(mask):
    return tuple(q for q in range(1, mask.bit_length() + 1) if mask & (1 << (q - 1)))


@krauschannel_registry.register_toproto(MixedUnitary)
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

    lossy = getattr(kraus_channel, "lossy", None)
    lossy_masks = (
        [_lossy_to_mask(m) for m in lossy]
        if lossy is not None and any(lossy)
        else []
    )
    return circuit_pb2.KrausChannel(
        mixedunitarychannel=circuit_pb2.MixedUnitaryChannel(
            operators=operators_proto, lossy_masks=lossy_masks
        )
    )


@krauschannel_registry.register_toproto(PauliNoise)
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


@krauschannel_registry.register_toproto(Depolarizing)
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
            operators.append(Operator(mat=matrix))

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
        elif op_proto.HasField("lossyoperator"):
            matrix_elements = [fromproto_complex(val) for val in op_proto.lossyoperator.matrix]
            num_qubits = op_proto.lossyoperator.numqubits
            matrix_size = 2**num_qubits

            if len(matrix_elements) != matrix_size * matrix_size:
                raise ValueError(
                    f"Number of elements ({len(matrix_elements)}) does not match expected size ({matrix_size * matrix_size})."
                )

            matrix = np.array(matrix_elements).reshape(matrix_size, matrix_size)
            operators.append(
                LossyOperator(
                    mat=matrix, lossy=tuple(op_proto.lossyoperator.lossy)
                )
            )
        else:
            raise ValueError("Unsupported operator type in customkrauschannel")

    return Kraus(operators)


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

    masks = list(mixedunitary_proto.lossy_masks)
    if not masks:
        return MixedUnitary(probabilities, unitary_matrices)
    lossy = [_mask_to_lossy(m) for m in masks]
    return MixedUnitary(probabilities, unitary_matrices, lossy=lossy)


@krauschannel_registry.register_fromproto("paulichannel")
def fromproto_paulichannel(paulichannel_proto, declcache=None):
    """Convert a protocol buffer PauliChannel to a mimiqcircuits PauliNoise."""
    probabilities = [fromproto_arg(prob) for prob in paulichannel_proto.probabilities]
    pauli_strings = [
        PauliString(pauli_str.pauli) for pauli_str in paulichannel_proto.paulistrings
    ]
    return PauliNoise(probabilities, pauli_strings)


@krauschannel_registry.register_fromproto("depolarizingchannel")
def fromproto_depolarizing(depolarizing_proto, declcache=None):
    """Convert a protocol buffer DepolarizingChannel to a mimiqcircuits Depolarizing."""
    num_qubits = depolarizing_proto.numqubits
    probability = fromproto_arg(depolarizing_proto.probability)
    return Depolarizing(num_qubits, probability)


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

    if isinstance(operation, krauschannel):
        kraus_proto = toproto_krauschannel(operation, declcache)
        kraus_field_name = kraus_proto.WhichOneof("krauschannel")
        return circuit_pb2.Operation(
            **{kraus_field_name: getattr(kraus_proto, kraus_field_name)}
        )

    # Check if the operation is a gate
    if isinstance(operation, Gate):
        gate_proto = toproto_gate(operation, declcache)
        gate_field_name = gate_proto.WhichOneof("gate")
        return circuit_pb2.Operation(
            **{gate_field_name: getattr(gate_proto, gate_field_name)}
        )

    # Check if the operation is an operator (but not a gate)
    if isinstance(operation, AbstractOperator) and not isinstance(
        operation, Gate
    ):
        operator_proto = toproto_operator(operation, declcache)
        operator_field_name = operator_proto.WhichOneof("operator")
        return circuit_pb2.Operation(
            **{operator_field_name: getattr(operator_proto, operator_field_name)}
        )

    # Fall back to case-by-case handling
    raise ValueError(f"Unsupported operation type: {type(operation)}")


_LEGACY_QUBITLOSS = 16  # pre-redesign OperationType tag, folded into Loss


@operation_registry.register_fromproto("simpleoperation")
def fromproto_simpleoperation(simpleoperation_proto, declcache=None):
    # Legacy: pre-redesign certain loss (QubitLoss) folded into Loss().
    if simpleoperation_proto.mtype == _LEGACY_QUBITLOSS:
        return Loss()
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
    if annotation_class is Tick:
        return Tick()
    elif annotation_class in {ShiftCoordinates, QubitCoordinates}:
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
        "customdiagonalgate",
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
    operator_fields = ["simpleoperator", "customoperator", "rescaledgate", "lossyoperator"]
    if operation_field in operator_fields:
        operator_proto = circuit_pb2.Operator(
            **{operation_field: getattr(operation_proto, operation_field)}
        )
        return fromproto_operator(operator_proto, declcache)

    raise ValueError(f"Unsupported operation field in proto: {operation_field}")


# Register specific operation converters
@operation_registry.register_toproto(Amplitude)
def toproto_amplitude(operation, declcache=None):
    """Convert an Amplitude operation to protocol buffer format."""
    bitstring_proto = toproto_bitvector(operation.bs)
    return circuit_pb2.Operation(amplitude=circuit_pb2.Amplitude(bs=bitstring_proto))


@operation_registry.register_toproto(ExpectationValue)
def toproto_expectationvalue(operation, declcache=None):
    """Convert an ExpectationValue operation to protocol buffer format."""
    operator_proto = toproto_operator(operation.op, declcache)
    return circuit_pb2.Operation(
        expectationvalue=circuit_pb2.ExpectationValue(operator=operator_proto)
    )


@operation_registry.register_toproto(IfStatement)
def toproto_ifstatement(operation, declcache=None):
    """Convert an IfStatement operation to protocol buffer format."""
    bitstring_proto = toproto_bitvector(operation.bitstring)
    operation_proto = toproto_operation(operation.op, declcache)
    return circuit_pb2.Operation(
        ifstatement=circuit_pb2.IfStatement(
            operation=operation_proto, bitstring=bitstring_proto
        )
    )


@operation_registry.register_toproto(Not)
def toproto_not(operation, declcache=None):
    """Convert a Not operation to protocol buffer format."""
    return circuit_pb2.Operation(
        simpleoperation=circuit_pb2.SimpleOperation(mtype=circuit_pb2.OperationType.Not)
    )


@operation_registry.register_fromproto("amplitude")
def fromproto_amplitude(amplitude_proto, declcache=None):
    """Convert a protocol buffer Amplitude to a mimiqcircuits Amplitude."""
    bitstring = BitString(fromproto_bitvector(amplitude_proto.bs))
    return Amplitude(bitstring)


@operation_registry.register_fromproto("expectationvalue")
def fromproto_expectationvalue(expectationvalue_proto, declcache=None):
    """Convert a protocol buffer ExpectationValue to a mimiqcircuits ExpectationValue."""
    operator = fromproto_operator(expectationvalue_proto.operator, declcache)
    return ExpectationValue(operator)


@operation_registry.register_fromproto("ifstatement")
def fromproto_ifstatement(ifstatement_proto, declcache=None):
    """Convert a protocol buffer IfStatement to a mimiqcircuits IfStatement."""
    operation = fromproto_operation(ifstatement_proto.operation, declcache)
    bitstring = BitString(fromproto_bitvector(ifstatement_proto.bitstring))
    return IfStatement(operation=operation, bitstring=bitstring)


@operation_registry.register_toproto(WhileStatement)
def toproto_whilestatement(operation, declcache=None):
    """Convert a WhileStatement operation to protocol buffer format."""
    bitstring_proto = toproto_bitvector(operation.bitstring)
    operation_proto = toproto_operation(operation.op, declcache)
    return circuit_pb2.Operation(
        whilestatement=circuit_pb2.WhileStatement(
            operation=operation_proto, bitstring=bitstring_proto
        )
    )


@operation_registry.register_fromproto("whilestatement")
def fromproto_whilestatement(whilestatement_proto, declcache=None):
    """Convert a protocol buffer WhileStatement to a mimiqcircuits WhileStatement."""
    operation = fromproto_operation(whilestatement_proto.operation, declcache)
    bitstring = BitString(fromproto_bitvector(whilestatement_proto.bitstring))
    return WhileStatement(operation=operation, bitstring=bitstring)


@operation_registry.register_toproto(ReadoutErr)
def toproto_readouterr(operation, declcache=None):
    """Convert a ReadoutErr operation to protobuf format."""
    msg = circuit_pb2.Operation()
    msg.readouterr.p0.CopyFrom(toproto_arg(operation.p0))
    msg.readouterr.p1.CopyFrom(toproto_arg(operation.p1))
    return msg


@operation_registry.register_fromproto("readouterr")
def fromproto_readouterr(readouterr_proto, declcache=None):
    """Convert protobuf ReadoutErr to mimiqcircuits ReadoutErr."""
    return ReadoutErr(
        fromproto_arg(readouterr_proto.p0),
        fromproto_arg(readouterr_proto.p1),
    )


@operation_registry.register_toproto(Loss)
def toproto_loss(operation, declcache=None):
    """Convert a Loss operation to protobuf format."""
    msg = circuit_pb2.Operation()
    msg.loss.p.CopyFrom(toproto_arg(operation.p))
    return msg


@operation_registry.register_fromproto("loss")
def fromproto_loss(loss_proto, declcache=None):
    """Convert protobuf Loss to mimiqcircuits Loss."""
    return Loss(fromproto_arg(loss_proto.p))


@operation_registry.register_toproto(MeasurePauli)
def toproto_measurepauli(operation, declcache=None):
    """Convert a MeasurePauli operation to protobuf format."""
    return circuit_pb2.Operation(
        measurepauli=circuit_pb2.MeasurePauli(
            pauli=pauli_pb2.PauliString(
                numqubits=operation.num_qubits, pauli=operation.pstring()
            )
        )
    )


@operation_registry.register_fromproto("measurepauli")
def fromproto_measurepauli(measurepauli_proto, declcache=None):
    """Convert protobuf MeasurePauli to mimiqcircuits MeasurePauli."""
    return MeasurePauli(measurepauli_proto.pauli.pauli)


# ------------ Block & Repeat -----------


@operation_registry.register_toproto(Block)
def toproto_block(op, declcache=None):
    proto_insts = [
        toproto_instruction(
            Instruction(
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
        numancillas=op.num_ancillas,
    )

    return circuit_pb2.Operation(block=block_msg)


@operation_registry.register_fromproto("block")
def fromproto_block(proto, declcache=None):
    instructions = [
        fromproto_instruction(inst, declcache) for inst in proto.instructions
    ]
    return Block(
        proto.numqubits,
        proto.numbits,
        proto.numzvars,
        instructions,
        num_ancillas=proto.numancillas,
    )


def toproto_blockdecl(decl, declcache=None):
    """Convert a BlockDecl to protocol buffer format."""
    return circuit_pb2.BlockDecl(
        name=decl.name,
        args=[circuit_pb2.Symbol(value=str(arg)) for arg in decl.arguments],
        numqubits=decl.num_qubits,
        numbits=decl.num_bits,
        numzvars=decl.num_zvars,
        instructions=[toproto_instruction(inst, declcache) for inst in decl],
        numancillas=decl.num_ancillas,
    )


def fromproto_blockdecl(proto, declcache=None):
    """Convert a protocol buffer BlockDecl to a mimiqcircuits BlockDecl."""
    # Pushed one by one: the Block constructor deep-copies, which would give
    # nested calls their own copy of a shared declaration.
    body = Block(proto.numqubits, proto.numbits, proto.numzvars)
    for inst in proto.instructions:
        body.push(fromproto_instruction(inst, declcache))
    return BlockDecl(
        proto.name,
        tuple(se.Symbol(arg.value) for arg in proto.args),
        body,
        num_ancillas=proto.numancillas,
    )


@operation_registry.register_toproto(BlockCall)
def toproto_blockcall(op, declcache=None):
    args = [toproto_arg(arg) for arg in op.arguments]
    # Outside a circuit there is no declaration table, so the decl goes inline.
    if declcache is None:
        msg = circuit_pb2.BlockCall(args=args, decl=toproto_blockdecl(op.decl))
        return circuit_pb2.Operation(blockcall=msg)
    declid = id(op.decl)
    if declid not in declcache[0]:
        declcache[0][declid] = toproto_blockdecl(op.decl, declcache)
        declcache[1].append(declid)
    return circuit_pb2.Operation(blockcall=circuit_pb2.BlockCall(id=declid, args=args))


@operation_registry.register_fromproto("blockcall")
def fromproto_blockcall(proto, declcache=None):
    args = [fromproto_arg(arg) for arg in proto.args]
    if proto.HasField("decl"):
        return BlockCall(fromproto_blockdecl(proto.decl, declcache), args)
    if declcache is None or proto.id not in declcache[0]:
        raise ValueError(f"BlockDecl with ID {proto.id} not found in cache")
    return BlockCall(declcache[0][proto.id], args)


@operation_registry.register_toproto(Repeat)
def toproto_repeat(op, declcache=None):
    proto_op = toproto_operation(op.get_operation(), declcache)
    return circuit_pb2.Operation(
        repeat=circuit_pb2.Repeat(numrepeats=op.repeats, operation=proto_op)
    )


@operation_registry.register_fromproto("repeat")
def fromproto_repeat(proto, declcache=None):
    inner_op = fromproto_operation(proto.operation, declcache)
    return Repeat(proto.numrepeats, inner_op)


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
    return Instruction(
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
        elif isinstance(decl, circuit_pb2.BlockDecl):
            decls_map[k] = circuit_pb2.Declaration(blockdecl=decl)
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
        elif which == "blockdecl":
            declcache[0][k] = fromproto_blockdecl(decl.blockdecl, declcache)
        else:
            raise ValueError(f"Unknown declaration type in proto: {which}")

    instructions = [
        fromproto_instruction(inst, declcache) for inst in circuit_proto.instructions
    ]
    return Circuit(instructions)


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
    return RPauli(fromproto_paulistring(msg.pauli), fromproto_arg(msg.theta))


# ------------- Operation-direct gate converters (skip Gate wrapper) ----------
#
# When a Gate appears as a top-level instruction operation, building a
# circuit_pb2.Gate just to unpack its field and rebuild a circuit_pb2.Operation
# costs an extra allocation plus a CopyFrom. The converters below produce the
# Operation message directly. They mirror the existing gate_registry entries
# but write into the matching Operation oneof field. toproto_gate is still
# used for inner gate slots (Control.operation, Power.operation, etc.).


def toproto_operation_simplegate(gate, declcache=None):
    return circuit_pb2.Operation(
        simplegate=circuit_pb2.SimpleGate(
            mtype=GATEMAP[type(gate)],
            parameters=[toproto_arg(p) for p in gate.getparams()],
        )
    )


def toproto_operation_generalized_gate(gate, declcache=None):
    if isinstance(gate, PolynomialOracle):
        args = [toproto_arg(arg) for arg in gate._params[2:]]
    else:
        args = [toproto_arg(p) for p in gate.getparams()]
    return circuit_pb2.Operation(
        generalized=circuit_pb2.Generalized(
            mtype=GENERALIZEDGATEMAP[type(gate)],
            args=args,
            qregsizes=list(getattr(gate, "_qregsizes", [])),
        )
    )


def toproto_operation_paulistring(gate, declcache=None):
    return circuit_pb2.Operation(
        paulistring=pauli_pb2.PauliString(numqubits=gate.num_qubits, pauli=gate.pauli)
    )


def toproto_operation_customgate(gate, declcache=None):
    U = [toproto_complex(p) for p in gate._U.T]
    return circuit_pb2.Operation(
        customgate=circuit_pb2.CustomGate(numqubits=gate.num_qubits, matrix=U)
    )


def toproto_operation_customdiagonalgate(gate, declcache=None):
    d = [toproto_complex(entry) for entry in gate._d]
    return circuit_pb2.Operation(
        customdiagonalgate=circuit_pb2.CustomDiagonalGate(
            numqubits=gate.num_qubits, diagonal=d
        )
    )


def toproto_operation_gatecall(gate, declcache=None):
    args_proto = [toproto_arg(arg) for arg in gate._args]
    if declcache is None:
        decl_proto = toproto_gatedecl(gate._decl)
        return circuit_pb2.Operation(
            gatecall=circuit_pb2.GateCall(decl=decl_proto, args=args_proto)
        )
    declid = id(gate._decl)
    if declid not in declcache[0]:
        declcache[0][declid] = toproto_gatedecl(gate._decl, declcache)
        declcache[1].append(declid)
    return circuit_pb2.Operation(
        cachedgatecall=circuit_pb2.CachedGateCall(id=declid, args=args_proto)
    )


def toproto_operation_control(gate, declcache=None):
    return circuit_pb2.Operation(
        control=circuit_pb2.Control(
            operation=toproto_gate(gate.op, declcache),
            numcontrols=gate.num_controls,
        )
    )


def toproto_operation_power(gate, declcache=None):
    base = toproto_gate(gate.op, declcache)
    if isinstance(gate.exponent, int):
        msg = circuit_pb2.Power(operation=base, int_val=gate.exponent)
    elif isinstance(gate.exponent, Fraction):
        msg = circuit_pb2.Power(
            operation=base,
            rational_val=circuit_pb2.Rational(
                num=gate.exponent.numerator, den=gate.exponent.denominator
            ),
        )
    elif isinstance(gate.exponent, float):
        msg = circuit_pb2.Power(operation=base, double_val=gate.exponent)
    else:
        raise ValueError(f"Unsupported exponent type: {type(gate.exponent)}")
    return circuit_pb2.Operation(power=msg)


def toproto_operation_inverse(gate, declcache=None):
    return circuit_pb2.Operation(
        inverse=circuit_pb2.Inverse(operation=toproto_gate(gate.op, declcache))
    )


def toproto_operation_parallel(gate, declcache=None):
    return circuit_pb2.Operation(
        parallel=circuit_pb2.Parallel(
            operation=toproto_gate(gate.op, declcache),
            numrepeats=gate.num_repeats,
        )
    )


def toproto_operation_rpauli(gate, declcache=None):
    return circuit_pb2.Operation(
        rpauli=circuit_pb2.RPauli(
            pauli=pauli_pb2.PauliString(
                numqubits=gate.pauli.num_qubits, pauli=str(gate.pauli.pauli)
            ),
            theta=toproto_arg(gate.theta),
        )
    )


def register_operation_direct_gates():
    """Register Operation-direct converters for every gate type."""
    for gate_class in GATEMAP:
        operation_registry.register_toproto_direct(
            gate_class, toproto_operation_simplegate
        )
    for gate_class in GENERALIZEDGATEMAP:
        operation_registry.register_toproto_direct(
            gate_class, toproto_operation_generalized_gate
        )
    operation_registry.register_toproto_direct(
        PauliString, toproto_operation_paulistring
    )
    operation_registry.register_toproto_direct(
        GateCustom, toproto_operation_customgate
    )
    operation_registry.register_toproto_direct(
        GateCustomDiagonal, toproto_operation_customdiagonalgate
    )
    operation_registry.register_toproto_direct(
        GateCall, toproto_operation_gatecall
    )
    operation_registry.register_toproto_direct(
        Control, toproto_operation_control
    )
    operation_registry.register_toproto_direct(
        Power, toproto_operation_power
    )
    operation_registry.register_toproto_direct(
        Inverse, toproto_operation_inverse
    )
    operation_registry.register_toproto_direct(
        Parallel, toproto_operation_parallel
    )
    operation_registry.register_toproto_direct(RPauli, toproto_operation_rpauli)


# ---------------------- Initialize Registries ----------------------


def initialize_registries():
    """Initialize all converter registries."""
    register_generalized_gates()
    register_simple_operations()
    register_generalized_operations()
    register_simple_annotations()
    register_generalized_annotations()
    register_operation_direct_gates()
    # Additional initialization can be added here


initialize_registries()
