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

from mimiqcircuits.__version__ import __version__

from mimiqcircuits.circuit import Circuit

from mimiqcircuits.bitstrings import BitString

from mimiqcircuits.operations.operation import Operation

from mimiqcircuits.operations.control import Control
from mimiqcircuits.operations.parallel import Parallel
from mimiqcircuits.operations.inverse import Inverse
from mimiqcircuits.operations.power import Power
from mimiqcircuits.operations.barrier import Barrier
from mimiqcircuits.operations.ifstatement import IfStatement
from mimiqcircuits.operations.measure import Measure
from mimiqcircuits.operations.reset import Reset
from mimiqcircuits.operations.gates.gate import Gate

from mimiqcircuits.operations.gates.custom import GateCustom

from mimiqcircuits.operations.gates.standard.u import GateU
from mimiqcircuits.operations.gates.standard.id import GateID
from mimiqcircuits.operations.gates.standard.pauli import GateX, GateY, GateZ
from mimiqcircuits.operations.gates.standard.hadamard import GateH
from mimiqcircuits.operations.gates.standard.s import GateS, GateSDG
from mimiqcircuits.operations.gates.standard.t import GateT, GateTDG
from mimiqcircuits.operations.gates.standard.sx import GateSX, GateSXDG
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

from mimiqcircuits.instruction import Instruction

from mimiqcircuits.remote import MimiqConnection

import mimiqcircuits.matrices as matrices

from mimiqcircuits.gatedecl import GateCall, GateDecl, gatedecl

from mimiqcircuits.lazy import control, parallel, inverse, power, LazyArg, LazyExpr

from mimiqcircuits.operations.gates.generalized.diffusion import Diffusion

from mimiqcircuits.operations.gates.generalized.polynomialoracle import PolynomialOracle

from mimiqcircuits.visualization.results import plothistogram
from mimiqcircuits.qcsresults import QCSResults, save_results, load_results
from mimiqcircuits.canvas import AsciiCanvas, AsciiCircuit
from mimiqlink.connection import QPERFECT_CLOUD, QPERFECT_CLOUD2

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
    "Measure",
    "Reset",
    "Gate",
    "GateCustom",
    "GateU",
    "GateID",
    "GateX",
    "GateY",
    "GateZ",
    "GateH",
    "GateS",
    "GateSDG",
    "GateT",
    "GateTDG",
    "GateSX",
    "GateSXDG",
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
    "matrices",
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
    "plothistogram",
    "QCSResults",
    "save_results",
    "load_results",
    "AsciiCanvas",
    "AsciiCircuit",
    "QPERFECT_CLOUD",
    "QPERFECT_CLOUD2"
]
