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

from mimiqcircuits.proto import hamiltonian_pb2
from mimiqcircuits.proto import pauli_pb2
from mimiqcircuits import *
from mimiqcircuits.proto.circuitproto import toproto_paulistring, fromproto_paulistring


def fromproto_hamiltonian_term(g: hamiltonian_pb2.HamiltonianTerm) -> HamiltonianTerm:
    pauli = fromproto_paulistring(g.pauli)
    return HamiltonianTerm(g.coefficient, pauli, tuple(q - 1 for q in g.qubits))


def toproto_hamiltonian_term(g: HamiltonianTerm) -> hamiltonian_pb2.HamiltonianTerm:
    proto = hamiltonian_pb2.HamiltonianTerm()
    proto.coefficient = g.get_coefficient()
    gate = toproto_paulistring(g.get_operation())
    proto.pauli.CopyFrom(gate.paulistring)
    proto.qubits.extend(q + 1 for q in g.get_qubits())
    return proto


def fromproto_hamiltonian(g: hamiltonian_pb2.Hamiltonian) -> Hamiltonian:
    return Hamiltonian([fromproto_hamiltonian_term(term) for term in g.terms])


def toproto_hamiltonian(g: Hamiltonian) -> hamiltonian_pb2.Hamiltonian:
    proto = hamiltonian_pb2.Hamiltonian()
    proto.terms.extend([toproto_hamiltonian_term(term) for term in g])
    return proto
