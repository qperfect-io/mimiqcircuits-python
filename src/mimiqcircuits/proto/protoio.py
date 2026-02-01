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

import mimiqcircuits as mc
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
from mimiqcircuits.proto import noisemodel_pb2


# Mapping for serialization
SAVEPROTO_MAP = {
    mc.Circuit: lambda obj: toproto_circuit(obj).SerializeToString(),
    mc.Hamiltonian: lambda obj: toproto_hamiltonian(obj).SerializeToString(),
    mc.OptimizationExperiment: lambda obj: toproto_OptimizationExperiment(
        obj
    ).SerializeToString(),
    mc.OptimizationRun: lambda obj: toproto_OptimizationRun(obj).SerializeToString(),
    mc.OptimizationResults: lambda obj: toproto_OptimizationResults(
        obj
    ).SerializeToString(),
    mc.QCSResults: lambda obj: toproto_qcsr(obj).SerializeToString(),
    mc.NoiseModel: lambda obj: toproto_noisemodel(obj).SerializeToString(),
}

# Mapping for deserialization
LOADPROTO_MAP = {
    mc.Circuit: (circuit_pb2.Circuit, fromproto_circuit),
    mc.Hamiltonian: (hamiltonian_pb2.Hamiltonian, fromproto_hamiltonian),
    mc.OptimizationExperiment: (
        optim_pb2.OptimizationExperiment,
        fromproto_OptimizationExperiment,
    ),
    mc.OptimizationRun: (optim_pb2.OptimizationRun, fromproto_OptimizationRun),
    mc.OptimizationResults: (
        optim_pb2.OptimizationResults,
        fromproto_OptimizationResults,
    ),
    mc.QCSResults: (qcsresults_pb2.QCSResults, fromproto_qcsr),
    mc.NoiseModel: (noisemodel_pb2.NoiseModel, fromproto_noisemodel),
}


def saveproto(obj, file):
    """
    Serialize a MimiqCircuits object and save it to a Protobuf (.pb) file.

    Parameters
    ----------
    obj : Union[Circuit, Hamiltonian, OptimizationExperiment, OptimizationRun, OptimizationResults, QCSResults]
        The object to serialize.
    file : Union[str, file-like]
        The output file path or a file-like object with a write() method.

    Returns
    -------
    int
        Number of bytes written.

    Raises
    ------
    TypeError
        If the object type is unsupported.
    ValueError
        If the file is invalid or writing fails.
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
    """
    Load a MimiqCircuits object from a Protobuf (.pb) file.

    Parameters
    ----------
    file : Union[str, file-like]
        File path or open file-like object to read binary data from.
    cls : Type
        The target class to deserialize into: Circuit, Hamiltonian, OptimizationExperiment, etc.

    Returns
    -------
    object
        The deserialized object of the given class.

    Raises
    ------
    TypeError
        If the class is unsupported.
    ValueError
        If the file is invalid or parsing fails.
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
