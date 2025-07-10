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

from mimiqcircuits.__version__ import __version__  # noqa: F401

from mimiqcircuits.circuit import Circuit

from mimiqcircuits.bitstrings import BitString

from mimiqcircuits.operations.operation import Operation
from mimiqcircuits.operations.operator import AbstractOperator

from mimiqcircuits.operations.control import (
    Control,
    register_control_decomposition,
)  # noqa: F401
from mimiqcircuits.operations.inverse import (
    Inverse,
    register_inverse_decomposition,  # noqa: F401
    register_inverse_alias,  # noqa: F401
)
from mimiqcircuits.operations.power import (
    Power,
    register_power_decomposition,  # noqa: F401
    register_power_alias,  # noqa: F401
)
from mimiqcircuits.operations.parallel import Parallel
from mimiqcircuits.operations.barrier import Barrier
from mimiqcircuits.operations.ifstatement import IfStatement
from mimiqcircuits.operations.measure import (
    AbstractMeasurement,
    Measure,
    MeasureX,
    MeasureY,
    MeasureZ,
)
from mimiqcircuits.operations.reset import Reset, ResetX, ResetY, ResetZ
from mimiqcircuits.operations.gates.gate import Gate

from mimiqcircuits.operations.gates.custom import GateCustom

from mimiqcircuits.operations.gates.standard.u import GateU
from mimiqcircuits.operations.gates.standard.id import GateID
from mimiqcircuits.operations.gates.standard.pauli import GateX, GateY, GateZ
from mimiqcircuits.operations.gates.standard.hadamard import (
    GateH,
    GateHXY,
    GateHXZ,
    GateHYZ,
)
from mimiqcircuits.operations.gates.standard.s import GateS, GateSDG
from mimiqcircuits.operations.gates.standard.t import GateT, GateTDG
from mimiqcircuits.operations.gates.standard.sx import GateSX, GateSXDG
from mimiqcircuits.operations.gates.standard.sy import GateSY, GateSYDG
from mimiqcircuits.operations.gates.standard.rotations import (
    GateRX,
    GateRY,
    GateRZ,
    GateR,
)
from mimiqcircuits.operations.gates.standard.deprecated import GateU1, GateU2, GateU3

from mimiqcircuits.operations.gates.standard.cpauli import GateCX, GateCY, GateCZ
from mimiqcircuits.operations.gates.standard.chadamard import GateCH
from mimiqcircuits.operations.gates.standard.phase import GateP
from mimiqcircuits.operations.gates.standard.swap import GateSWAP
from mimiqcircuits.operations.gates.standard.iswap import GateISWAP
from mimiqcircuits.operations.gates.standard.cs import GateCS, GateCSDG
from mimiqcircuits.operations.gates.standard.csx import GateCSX, GateCSXDG
from mimiqcircuits.operations.gates.standard.ecr import GateECR
from mimiqcircuits.operations.gates.standard.dcx import GateDCX
from mimiqcircuits.operations.gates.standard.cphase import GateCP
from mimiqcircuits.operations.gates.standard.cu import GateCU
from mimiqcircuits.operations.gates.standard.crotations import GateCRX, GateCRY, GateCRZ
from mimiqcircuits.operations.gates.standard.interactions import (
    GateRXX,
    GateRYY,
    GateRZZ,
    GateRZX,
    GateXXplusYY,
    GateXXminusYY,
)
from mimiqcircuits.operations.gates.standard.cnx import GateCCX, GateC3X
from mimiqcircuits.operations.gates.standard.cnp import GateCCP
from mimiqcircuits.operations.gates.standard.cswap import GateCSWAP

from mimiqcircuits.operations.gates.generalized.qft import QFT
from mimiqcircuits.operations.gates.generalized.phasegradient import PhaseGradient
from mimiqcircuits.operations.gates.generalized.rnz import GateRNZ

from mimiqcircuits.instruction import Instruction

from mimiqcircuits.remote import MimiqConnection

from mimiqcircuits.gatedecl import GateCall, GateDecl, gatedecl

from mimiqcircuits.lazy import control, parallel, inverse, power, LazyArg, LazyExpr

from mimiqcircuits.operations.gates.generalized.diffusion import Diffusion

from mimiqcircuits.operations.gates.generalized.polynomialoracle import PolynomialOracle

from mimiqcircuits.qcsresults import QCSResults
from mimiqcircuits.canvas import AsciiCanvas, AsciiCircuit
from mimiqcircuits.operations.gates.generalized.paulistring import PauliString
from mimiqcircuits.operations.noisechannel.standards.ampdamping import (
    AmplitudeDamping,
    GeneralizedAmplitudeDamping,
)

from mimiqcircuits.operations.operators.sigmas import (
    SigmaMinus,
    SigmaPlus,
)
from mimiqcircuits.operations.operators.diagonals import DiagonalOp
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
    ProjectorZ0,
    ProjectorZ1,
)

from mimiqcircuits.operations.noisechannel.standards.depolarizing import (
    Depolarizing,
    Depolarizing1,
    Depolarizing2,
)
from mimiqcircuits.operations.operators.custom import Operator
from mimiqcircuits.operations.noisechannel.standards.phaseamplitudedamping import (
    PhaseAmplitudeDamping,
    ThermalNoise,
)
from mimiqcircuits.operations.noisechannel.kraus import Kraus
from mimiqcircuits.operations.noisechannel.mixedunitary import MixedUnitary
from mimiqcircuits.operations.krauschannel import krauschannel
from mimiqcircuits.operations.noisechannel.standards.pauli import (
    PauliNoise,
    PauliX,
    PauliY,
    PauliZ,
)
from mimiqcircuits.operations.amplitude import Amplitude
from mimiqcircuits.operations.expectationvalue import ExpectationValue
from mimiqcircuits.operations.entanglement import (
    SchmidtRank,
    VonNeumannEntropy,
    BondDim,
)
from mimiqcircuits.operations.measurereset import (
    MeasureReset,
    MeasureResetX,
    MeasureResetY,
    MeasureResetZ,
)
from mimiqcircuits.operations.gates.delay import Delay
from mimiqcircuits.operations.noisechannel.standards.projectivenoise import (
    ProjectiveNoise,
    ProjectiveNoiseX,
    ProjectiveNoiseY,
    ProjectiveNoiseZ,
)
from mimiqcircuits.operations.pairmeasure import MeasureXX, MeasureYY, MeasureZZ
from mimiqcircuits.operations.rescaledgates import RescaledGate
from mimiqcircuits.operations.annotations import (
    AbstractAnnotation,
    Detector,
    QubitCoordinates,
    ShiftCoordinates,
    ObservableInclude,
    Tick,
)
from mimiqcircuits.classical.classical_not import Not
from mimiqcircuits.complex import Pow, Add, Multiply
from mimiqcircuits.operations.block import Block
from mimiqcircuits.operations.repeat import Repeat, repeat

# needed to initialize the registers
import mimiqcircuits.proto.circuitproto

from mimiqlink import QPERFECT_CLOUD, QPERFECT_DEV, PLANQK_API
from mimiqcircuits.hamiltonian import (
    Hamiltonian,
    HamiltonianTerm,
    push_expval,
    push_suzukitrotter,
    push_lietrotter,
    push_yoshidatrotter,
)
from mimiqcircuits.operations.gates.generalized.rpauli import RPauli


class GATES:
    """
    Helper class to list gates.

    **NOTE**: Cannot instantiated called directly. Use `GATES.list()` for a list
    of gate types, and `help(GATES)` for the .


    **Single qubit gates**
        :func:`GateX` :func:`GateY` :func:`GateZ` :func:`GateH`
        :func:`GateS` :func:`GateSDG`
        :func:`GateT` :func:`GateTDG`
        :func:`GateSX` :func:`GateSXDG`
        :func:`GateID`

    **Single qubit gates (parametric)**
        :func:`GateU` :func:`GateP`
        :func:`GateRX` :func:`GateRY` :func:`GateRZ` :func:`GateP`

    **Two qubit gates**
        :func:`GateCX` :func:`GateCY` :func:`GateCZ`
        :func:`GateCH`
        :func:`GateSWAP` :func:`GateISWAP`
        :func:`GateCS` :func:`GateCSX`
        :func:`GateECR` :func:`GateDCX`

    **Two qubit gates (parametric)**
        :func:`GateCU`
        :func:`GateCP`
        :func:`GateCRX` :func:`GateCRY` :func:`GateCRZ`
        :func:`GateRXX` :func:`GateRYY` :func:`GateRZZ`
        :func:`GateXXplusYY` :func:`GateXXminusYY`

    **Other**
        :func:`GateCustom`
    """

    def __new__(cls):
        raise NotImplementedError

    @staticmethod
    def list():
        return [
            GateCustom,
            GateU,
            GateID,
            GateX,
            GateY,
            GateZ,
            GateH,
            GateHXY,
            GateHXZ,
            GateHYZ,
            GateS,
            GateSDG,
            GateT,
            GateTDG,
            GateSX,
            GateSXDG,
            GateSY,
            GateSYDG,
            GateRX,
            GateRY,
            GateRZ,
            GateR,
            GateU1,
            GateU2,
            GateU3,
            GateCX,
            GateCY,
            GateCZ,
            GateCH,
            GateSWAP,
            GateISWAP,
            GateCS,
            GateCSDG,
            GateCSX,
            GateCSXDG,
            GateECR,
            GateDCX,
            GateCP,
            GateCU,
            GateCRX,
            GateCRY,
            GateCRZ,
            GateRXX,
            GateRYY,
            GateRZZ,
            GateRZX,
            GateXXplusYY,
            GateXXminusYY,
            GateCCX,
            GateC3X,
            GateCCP,
            GateCSWAP,
            GateP,
        ]


# Export specific classes, and functions.
__all__ = [
    "Circuit",
    "BitString",
    "Operation",
    "Control",
    "Parallel",
    "Inverse",
    "Power",
    "Barrier",
    "IfStatement",
    "AbstractMeasurement",
    "Measure",
    "MeasureX",
    "MeasureY",
    "MeasureZ",
    "Reset",
    "ResetX",
    "ResetY",
    "ResetZ",
    "MeasureReset",
    "MeasureResetX",
    "MeasureResetY",
    "MeasureResetZ",
    "Gate",
    "GateCustom",
    "GateU",
    "GateID",
    "GateX",
    "GateY",
    "GateZ",
    "GateH",
    "GateHXY",
    "GateHXZ",
    "GateHYZ",
    "GateS",
    "GateSDG",
    "GateT",
    "GateTDG",
    "GateSX",
    "GateSXDG",
    "GateSY",
    "GateSYDG",
    "GateRX",
    "GateRY",
    "GateRZ",
    "GateR",
    "GateU1",
    "GateU2",
    "GateU3",
    "GateCX",
    "GateCY",
    "GateCZ",
    "GateCH",
    "GateSWAP",
    "GateISWAP",
    "GateCS",
    "GateCSDG",
    "GateCSX",
    "GateCSXDG",
    "GateECR",
    "GateDCX",
    "GateCP",
    "GateCU",
    "GateCRX",
    "GateCRY",
    "GateCRZ",
    "GateRXX",
    "GateRYY",
    "GateRZZ",
    "GateRZX",
    "GateXXplusYY",
    "GateXXminusYY",
    "GateCCX",
    "GateC3X",
    "GateCCP",
    "GateCSWAP",
    "QFT",
    "PhaseGradient",
    "Instruction",
    "GateP",
    "MimiqConnection",
    "GateCall",
    "GateDecl",
    "gatedecl",
    "control",
    "parallel",
    "inverse",
    "power",
    "LazyArg",
    "LazyExpr",
    "Diffusion",
    "PolynomialOracle",
    "QCSResults",
    "AsciiCanvas",
    "AsciiCircuit",
    "AbstractOperator",
    "PauliString",
    "AmplitudeDamping",
    "GeneralizedAmplitudeDamping",
    "SigmaMinus",
    "SigmaPlus",
    "DiagonalOp",
    "Projector0",
    "Projector00",
    "Projector01",
    "Projector1",
    "Projector10",
    "Projector11",
    "Depolarizing",
    "Depolarizing1",
    "Depolarizing2",
    "ProjectorX0",
    "ProjectorX1",
    "ProjectorY0",
    "ProjectorY1",
    "ProjectorZ0",
    "ProjectorZ1",
    "Operator",
    "PauliNoise",
    "krauschannel",
    "PauliX",
    "PauliY",
    "PauliZ",
    "ThermalNoise",
    "PhaseAmplitudeDamping",
    "Kraus",
    "MixedUnitary",
    "QPERFECT_CLOUD",
    "QPERFECT_DEV",
    "PLANQK_API",
    "Amplitude",
    "ExpectationValue",
    "BondDim",
    "VonNeumannEntropy",
    "SchmidtRank",
    "Delay",
    "ProjectiveNoise",
    "ProjectiveNoiseX",
    "ProjectiveNoiseY",
    "ProjectiveNoiseZ",
    "MeasureZZ",
    "MeasureXX",
    "MeasureYY",
    "RescaledGate",
    "AbstractAnnotation",
    "Detector",
    "QubitCoordinates",
    "ShiftCoordinates",
    "ObservableInclude",
    "Tick",
    "Not",
    "Pow",
    "Add",
    "Multiply",
    "GATES",
    "Hamiltonian",
    "HamiltonianTerm",
    "push_expval",
    "push_suzukitrotter",
    "push_lietrotter",
    "RPauli",
    "GateRNZ",
    "push_yoshidatrotter",
    "Block",
    "Repeat",
    "repeat",
]
