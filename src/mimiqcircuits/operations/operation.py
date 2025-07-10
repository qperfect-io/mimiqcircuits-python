#
# Copyright © 2022-2024 University of Strasbourg. All Rights Reserved.
# Copyright © 2023-2025 QPerfect. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use thas file except in compliance with the License.
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

from abc import ABC, abstractmethod
import copy
import mimiqcircuits as mc
import symengine as se
import sympy as sp
from mimiqcircuits.canvas import _gate_name_padding


class Operation(ABC):
    """
    Abstract base class for quantum operations.
    """

    _name = None

    _num_qubits = None
    _num_qregs = 1
    _qregsizes = None

    _num_bits = None
    _num_cregs = 0
    _cregsizes = None
    _num_zvars = None
    _num_zregs = 0
    _zregsizes = None

    _parnames = ()

    @property
    def num_qubits(self):
        return self._num_qubits

    @num_qubits.setter
    def num_qubits(self, value):
        raise ValueError("Cannot set num_qubits. Read only parameter.")

    @property
    def num_qregs(self):
        return self._num_qregs

    @num_qregs.setter
    def num_qregs(self, value):
        raise ValueError("Cannot set num_qregs. Read only parameter.")

    @property
    def num_bits(self):
        return self._num_bits

    @num_bits.setter
    def num_bits(self, value):
        raise ValueError("Cannot set num_bits. Read only parameter.")

    @property
    def num_zvars(self):
        return self._num_zvars

    @num_zvars.setter
    def num_zvars(self, value):
        raise ValueError("Cannot set num_zvars. Read only parameter.")

    @property
    def num_cregs(self):
        return self._num_cregs

    @num_cregs.setter
    def num_cregs(self, value):
        raise ValueError("Cannot set num_cregs. Read only parameter.")

    @property
    def qregsizes(self):
        return self._qregsizes

    @qregsizes.setter
    def qregsizes(self, value):
        raise ValueError("Cannot set qregsizes. Read only parameter.")

    @property
    def cregsizes(self):
        return self._cregsizes

    @cregsizes.setter
    def cregsizes(self, value):
        raise ValueError("Cannot set cregsizes. Read only parameter.")

    @property
    def zregsizes(self):
        return self._zregsizes

    @zregsizes.setter
    def zregsizes(self, value):
        raise ValueError("Cannot set zregsizes. Read only parameter.")

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        raise ValueError("Cannot set name. Read only parameter.")

    @property
    def parnames(self):
        return self._parnames

    @parnames.setter
    def parnames(self, value):
        raise ValueError("Cannot set parnames. Read only parameter.")

    def getparams(self):
        return [getattr(self, pn) for pn in self._parnames]

    def is_symbolic(self):
        return any(
            isinstance(param, (se.Basic, sp.Basic)) and not param.evalf().is_number
            for param in self.getparams()
        )

    def getparam(self, pn):
        if pn not in self.parnames:
            raise ValueError(f"Parameter {pn} not found.")
        return getattr(self, pn)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.__dict__ == other.__dict__

    def copy(self):
        """Creates a shallow copy of the operation.
            To create a full copy use deepcopy() instead.

        Returns:
            Operation: A new Operation object containing references to the same attributes as the original circuit
        """
        return copy.copy(self)

    def deepcopy(self):
        """Creates a copy of the object and for all its attributes

        Returns:
            Operation: A new Operation object fully identical the original circuit
        """
        return copy.deepcopy(self)

    @abstractmethod
    def iswrapper(self):
        pass

    def isopalias(self):
        return False

    def numparams(self):
        return len(self.getparams())

    def _decompose(self, circ, qubits, bits, zvars):
        return circ.push(self, *qubits, *bits, *zvars)

    def decompose(self):
        return self._decompose(
            mc.Circuit(),
            range(self.num_qubits),
            range(self.num_bits),
            range(self.num_zvars),
        )

    def evaluate(self, d):
        return self

    def isidentity(self):
        return False

    @classmethod
    def isunitary(self):
        """Check if the class represents a unitary operator.

        By default, this method returns `False` unless explicitly overridden in a subclass.
        """
        return False

    def asciiwidth(self, qubits, bits, zvars):
        # Calculate padding for qubits, bits, and zvars
        namepadding = _gate_name_padding(qubits, bits, zvars)

        # Calculate the width contribution of zvars
        zvar_width = len(",".join(map(str, zvars))) if zvars else 0

        # Assume operation has a string representation (e.g., __str__)
        operation_repr = str(self)

        # Calculate the width: | + (num + space) + name + zvars (if any) + |
        if zvars and not qubits:
            # If there are zvars but no qubits, ensure width includes zvars fully
            return 1 + max(namepadding + len(operation_repr), zvar_width) + 1
        else:
            # Regular case where zvars and qubits are both present or only qubits
            return 1 + namepadding + len(operation_repr) + 1

    def get_operation(self):
        return self

    def listvars(self):
        vars = set()
        for param in self.getparams():
            if isinstance(param, se.Basic):
                vars.update(param.atoms(se.Symbol))
        return list(vars)


# export operations
__all__ = ["Operation"]
