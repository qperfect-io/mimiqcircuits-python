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
Protobuf I/O utilities for saving and loading MimiqCircuits objects.
"""

from mimiqcircuits.circuit import Circuit
from mimiqcircuits.equivalence import EquivalenceExperiment
from mimiqcircuits.hamiltonian import Hamiltonian
from mimiqcircuits.lossmodel import LossModel
from mimiqcircuits.noisemodel import NoiseModel
from mimiqcircuits.optimization import (
    OptimizationExperiment,
    OptimizationResults,
    OptimizationRun,
)
from mimiqcircuits.qcsresults import QCSResults
from mimiqcircuits.sweep import SweepExperiment
from mimiqcircuits.proto.circuitproto import toproto_circuit, fromproto_circuit
from mimiqcircuits.proto.hamiltonianproto import (
    toproto_hamiltonian,
    fromproto_hamiltonian,
)
from mimiqcircuits.proto.optim import (
    toproto_OptimizationExperiment,
    toproto_OptimizationRun,
    toproto_OptimizationResults,
    fromproto_OptimizationExperiment,
    fromproto_OptimizationRun,
    fromproto_OptimizationResults,
)
from mimiqcircuits.proto.qcsrproto import toproto_qcsr, fromproto_qcsr
from mimiqcircuits.proto import circuit_pb2, hamiltonian_pb2, optim_pb2, qcsresults_pb2

from mimiqcircuits.proto.noisemodelproto import (
    toproto_noisemodel,
    fromproto_noisemodel,
)
from mimiqcircuits.proto.circuitrulesproto import (
    toproto_lossmodel,
    fromproto_lossmodel,
)
from mimiqcircuits.proto import noisemodel_pb2, circuitrules_pb2
from mimiqcircuits.proto.experimentproto import (
    toproto_sweep,
    fromproto_sweep,
    toproto_equivalence,
    fromproto_equivalence,
)
from mimiqcircuits.proto import experiment_pb2


# Mapping for serialization
SAVEPROTO_MAP = {
    Circuit: lambda obj: toproto_circuit(obj).SerializeToString(),
    Hamiltonian: lambda obj: toproto_hamiltonian(obj).SerializeToString(),
    OptimizationExperiment: lambda obj: toproto_OptimizationExperiment(
        obj
    ).SerializeToString(),
    OptimizationRun: lambda obj: toproto_OptimizationRun(obj).SerializeToString(),
    OptimizationResults: lambda obj: toproto_OptimizationResults(
        obj
    ).SerializeToString(),
    QCSResults: lambda obj: toproto_qcsr(obj).SerializeToString(),
    NoiseModel: lambda obj: toproto_noisemodel(obj).SerializeToString(),
    LossModel: lambda obj: toproto_lossmodel(obj).SerializeToString(),
    SweepExperiment: lambda obj: toproto_sweep(obj).SerializeToString(),
    EquivalenceExperiment: lambda obj: toproto_equivalence(obj).SerializeToString(),
}

# Mapping for deserialization
LOADPROTO_MAP = {
    Circuit: (circuit_pb2.Circuit, fromproto_circuit),
    Hamiltonian: (hamiltonian_pb2.Hamiltonian, fromproto_hamiltonian),
    OptimizationExperiment: (
        optim_pb2.OptimizationExperiment,
        fromproto_OptimizationExperiment,
    ),
    OptimizationRun: (optim_pb2.OptimizationRun, fromproto_OptimizationRun),
    OptimizationResults: (
        optim_pb2.OptimizationResults,
        fromproto_OptimizationResults,
    ),
    QCSResults: (qcsresults_pb2.QCSResults, fromproto_qcsr),
    NoiseModel: (noisemodel_pb2.NoiseModel, fromproto_noisemodel),
    LossModel: (circuitrules_pb2.LossModel, fromproto_lossmodel),
    SweepExperiment: (experiment_pb2.SweepExperiment, fromproto_sweep),
    EquivalenceExperiment: (
        experiment_pb2.EquivalenceExperiment,
        fromproto_equivalence,
    ),
}


def saveproto(obj, file):
    """Serialize an object to protobuf and write it to a file.

    The classes that support protobuf expose this as a ``saveproto`` method,
    for example ``circuit.saveproto("circuit.pb")``.

    Args:
        obj: The object to save: a ``Circuit``, ``Hamiltonian``,
            ``OptimizationExperiment``, ``OptimizationRun``,
            ``OptimizationResults``, ``QCSResults``, ``NoiseModel``,
            ``LossModel``, ``SweepExperiment`` or ``EquivalenceExperiment``.
            Subclasses are not accepted.
        file (str or file-like): A path, or a binary file object with a
            ``write`` method.

    Returns:
        int: The number of bytes written.

    Raises:
        TypeError: If the type of ``obj`` is not supported.
        ValueError: If writing to a path fails.

    Examples:
        >>> from mimiqcircuits import *
        >>> from mimiqcircuits.proto.protoio import saveproto, loadproto
        >>> import os, tempfile
        >>> c = Circuit()
        >>> c.push(GateH(), 0)
        1-qubit circuit with 1 instruction:
        └── H @ q[0]
        <BLANKLINE>

        Save the circuit, then load it back by naming the class to read.

        >>> path = os.path.join(tempfile.mkdtemp(), "circuit.pb")
        >>> nbytes = saveproto(c, path)
        >>> loadproto(path, Circuit)
        1-qubit circuit with 1 instruction:
        └── H @ q[0]
        <BLANKLINE>
    """
    serializer = SAVEPROTO_MAP.get(type(obj))
    if serializer is None:
        raise TypeError(f"Unsupported object type: {type(obj)}")

    data = serializer(obj)

    if hasattr(file, "write"):
        return file.write(data)
    try:
        with open(file, "wb") as f:
            return f.write(data)
    except Exception as e:
        raise ValueError(f"Failed to write protobuf file: {e}")


def loadproto(file, cls):
    """Read an object of class ``cls`` from a protobuf file.

    The classes that support protobuf expose this as a static ``loadproto``
    method, for example ``Circuit.loadproto("circuit.pb")``. The file must
    hold an object of class ``cls``: the protobuf format does not record it.

    Args:
        file (str or file-like): A path, or a binary file object with a
            ``read`` method.
        cls (type): The class to read, one of the classes :func:`saveproto`
            accepts.

    Returns:
        object: The loaded object, an instance of ``cls``.

    Raises:
        TypeError: If ``cls`` is not supported.
        ValueError: If the file cannot be read or parsed.

    See Also:
        :func:`saveproto`, which has an example.
    """
    entry = LOADPROTO_MAP.get(cls)
    if entry is None:
        raise TypeError(f"Unsupported class type: {cls}")

    msg_cls, parser = entry
    msg = msg_cls()

    try:
        if hasattr(file, "read"):
            msg.ParseFromString(file.read())
        else:
            with open(file, "rb") as f:
                msg.ParseFromString(f.read())
    except Exception as e:
        raise ValueError(f"Failed to read or parse protobuf file: {e}")

    return parser(msg)


__all__ = ["saveproto", "loadproto"]
