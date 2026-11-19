#
# Copyright © 2023-2026 QPerfect. All Rights Reserved.
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
"""Protobuf conversion for :class:`SweepExperiment` and
:class:`EquivalenceExperiment`."""

import numpy as np
from symengine import Symbol

from mimiqcircuits.proto import circuit_pb2, experiment_pb2
from mimiqcircuits.proto.circuitproto import toproto_circuit, fromproto_circuit

_METHODS = {
    "samples": experiment_pb2.SAMPLES,
    "amplitudes": experiment_pb2.AMPLITUDES,
}
_METHODNAMES = {v: k for k, v in _METHODS.items()}


def toproto_parametertable(table):
    return experiment_pb2.ParameterTable(
        symbols=[circuit_pb2.Symbol(value=str(s)) for s in table.listvars()],
        npoints=len(table),
        # numpy arrays are row-major, the same layout as the wire.
        values=table.to_numpy().ravel().tolist(),
    )


def fromproto_parametertable(msg):
    from mimiqcircuits.sweep import ParameterTable

    symbols = [Symbol(s.value) for s in msg.symbols]
    npoints = msg.npoints
    if len(msg.values) != npoints * len(symbols):
        raise ValueError(
            f"a table of {npoints} points over {len(symbols)} symbols needs "
            f"{npoints * len(symbols)} values, not {len(msg.values)}"
        )
    values = np.array(msg.values, dtype=float).reshape(npoints, len(symbols))
    return ParameterTable(symbols, values)


def toproto_sweep(ex):
    return experiment_pb2.SweepExperiment(
        circuit=toproto_circuit(ex.circuit),
        table=toproto_parametertable(ex.table),
    )


def fromproto_sweep(msg):
    from mimiqcircuits.sweep import SweepExperiment

    return SweepExperiment(
        fromproto_circuit(msg.circuit), fromproto_parametertable(msg.table)
    )


def toproto_equivalence(ex):
    return experiment_pb2.EquivalenceExperiment(
        c1=toproto_circuit(ex.c1),
        c2=toproto_circuit(ex.c2),
        method=_METHODS[ex.method],
    )


def fromproto_equivalence(msg):
    from mimiqcircuits.equivalence import EquivalenceExperiment

    if msg.method not in _METHODNAMES:
        raise ValueError(f"unknown equivalence method {msg.method}")
    return EquivalenceExperiment(
        fromproto_circuit(msg.c1),
        fromproto_circuit(msg.c2),
        method=_METHODNAMES[msg.method],
    )
