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


import copy
from mimiqcircuits.instruction import Instruction
from mimiqcircuits.operations.operation import Operation
from mimiqcircuits.push import push_instruction_container


class Block(Operation):
    r"""Block operation: group and reuse a sequence of instructions.

    The `Block` class represents a reusable subcircuit. It encapsulates a fixed number
    of qubits, bits, and z-variables, along with a list of instructions. Blocks are
    used to define logical units in a circuit that can be inserted as composite operations.

    You can construct a block in several ways:

    - `Block()`: Create an empty block with 0 qubits, bits, and zvars.
    - `Block(circuit)`: Copy instructions from a circuit (or block) into a new block.
    - `Block(instructions)`: Create a block from a list of `Instruction` objects.
    - `Block(num_qubits, num_bits, num_zvars[, instructions])`: Fully specify a block.

    Notes:
        - Once created, a block has a fixed number of qubits, bits, and zvars.
        - Adding an instruction that exceeds the declared dimensions will raise a `ValueError`.
        - Blocks are deep-copied upon construction to avoid accidental mutations.

    Examples:
        >>> from mimiqcircuits import *
        >>> c = Circuit()

        >>> c.push(GateCX(), 1, 2)
        3-qubit circuit with 1 instructions:
        └── CX @ q[1], q[2]
        <BLANKLINE>

        <BLANKLINE>
        >>> c.push(GateCX(), 1, 3)
        4-qubit circuit with 2 instructions:
        ├── CX @ q[1], q[2]
        └── CX @ q[1], q[3]
        <BLANKLINE>

        >>> c.push(MeasureZZ(), 1, 2, 1)
        4-qubit, 2-bit circuit with 3 instructions:
        ├── CX @ q[1], q[2]
        ├── CX @ q[1], q[3]
        └── MZZ @ q[1,2], c[1]
        <BLANKLINE>

        >>> block = Block(c)
        >>> block
        4-qubit, 2-bit block ... with 3 instructions:
        ├── CX @ q[1], q[2]
        ├── CX @ q[1], q[3]
        └── MZZ @ q[1,2], c[1]

        >>> main = Circuit()
        >>> main.push(block, 0, 1, 2, 3, 0, 1)
        4-qubit, 2-bit circuit with 1 instructions:
        └── block ... @ q[0,1,2,3], c[0,1]
        <BLANKLINE>

        >>> main.decompose()
        4-qubit, 2-bit circuit with 3 instructions:
        ├── CX @ q[1], q[2]
        ├── CX @ q[1], q[3]
        └── MZZ @ q[1,2], c[1]
        <BLANKLINE>

    See Also:
        - :class:`Instruction`
        - :class:`Circuit`
    """

    _name = "Block"
    _num_qregs = 1
    _num_cregs = 1
    _num_zregs = 1
    _parnames = ()

    def __init__(self, *args):
        if len(args) == 0:
            self.instructions = []
            self._nq = 0
            self._nc = 0
            self._nz = 0

        elif len(args) == 1:
            arg = args[0]
            if hasattr(arg, "instructions"):
                self.instructions = copy.deepcopy(arg.instructions)
            elif isinstance(arg, list):
                if not all(isinstance(i, Instruction) for i in arg):
                    raise TypeError(
                        "All items in the list must be Instruction instances"
                    )
                self.instructions = copy.deepcopy(arg)
            else:
                raise TypeError("Expected a Circuit or list of Instructions")

            self._nq = self._infer_qubits()
            self._nc = self._infer_bits()
            self._nz = self._infer_zvars()

        elif len(args) == 4:
            self._nq, self._nc, self._nz = args[:3]
            if not isinstance(args[3], list):
                raise TypeError(
                    "Expected a list of Instruction objects as the fourth argument"
                )
            if not all(isinstance(i, Instruction) for i in args[3]):
                raise TypeError(
                    "All items in the instruction list must be Instruction instances"
                )
            self.instructions = copy.deepcopy(args[3])

        elif len(args) == 3:
            self._nq, self._nc, self._nz = args
            self.instructions = []

        else:
            raise ValueError("Invalid arguments to Block constructor")

        super().__init__()
        self._num_qubits = self._nq
        self._num_bits = self._nc
        self._num_zvars = self._nz
        self._qregsizes = [1] * self._num_qubits if self._num_qubits > 0 else []
        self._cregsizes = [1] * self._num_bits if self._num_bits > 0 else []
        self._zregsizes = [1] * self._num_zvars if self._num_zvars > 0 else []

    def _infer_qubits(self):
        return (
            max((max(i.qubits, default=-1) for i in self.instructions), default=-1) + 1
        )

    def _infer_bits(self):
        return max((max(i.bits, default=-1) for i in self.instructions), default=-1) + 1

    def _infer_zvars(self):
        return (
            max((max(i.zvars, default=-1) for i in self.instructions), default=-1) + 1
        )

    def push(self, operation, *args):
        return push_instruction_container(
            self, operation, *args, check_fn=self._check_instruction_block
        )

    def _check_instruction_block(self, inst):
        if max(inst.qubits, default=-1) >= self._nq:
            raise ValueError(
                f"Too many qubits: max qubit index {max(inst.qubits)} exceeds allowed {self._nq - 1}"
            )
        if max(inst.bits, default=-1) >= self._nc:
            raise ValueError(
                f"Too many bits: max bit index {max(inst.bits)} exceeds allowed {self._nc - 1}"
            )
        if max(inst.zvars, default=-1) >= self._nz:
            raise ValueError(
                f"Too many zvars: max zvar index {max(inst.zvars)} exceeds allowed {self._nz - 1}"
            )

    def __iter__(self):
        return iter(self.instructions)

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        return (
            self.instructions[idx]
            if isinstance(idx, int)
            else Block(self.instructions[idx])
        )

    def __str__(self):
        return f"block {id(self):x}"

    def __repr__(self):
        if not self.instructions:
            return "empty circuit"
        blockid = self.blockid()
        parts = []
        if self._nq > 0:
            parts.append(f"{self._nq}-qubit")
        if self._nc > 0:
            parts.append(f"{self._nc}-bit")
        if self._nz > 0:
            parts.append(f"{self._nz}-zvar")
        head = f"{', '.join(parts)} {blockid} with {len(self)} instructions"
        lines = [head + ":"]
        for inst in self.instructions[:-1]:
            lines.append(f"├── {inst}")
        lines.append(f"└── {self.instructions[-1]}")
        return "\n".join(lines)

    def __call__(self, circuit, *targets):
        nq, nc, nz = self._nq, self._nc, self._nz
        if len(targets) != nq + nc + nz:
            raise ValueError(
                f"Expected {nq} qubits, {nc} bits, {nz} zvars but got {len(targets)} total targets"
            )

        qtargets = targets[:nq]
        ctargets = targets[nq : nq + nc]
        ztargets = targets[nq + nc :]

        return self._decompose(circuit, qtargets, ctargets, ztargets)

    def _decompose(self, circuit, qtargets, ctargets, ztargets):
        for inst in self.instructions:
            q = [qtargets[i] for i in inst.qubits]
            c = [ctargets[i] for i in inst.bits]
            z = [ztargets[i] for i in inst.zvars]
            circuit.push(inst.operation, *q, *c, *z)
        return circuit

    def iswrapper(self):
        return False

    def blockid(self):
        return f"block {hex(id(self))[2:]}"

    def format_with_targets(self, qubits, bits, zvars):
        label = self.blockid()
        parts = []
        if qubits:
            parts.append("q[" + ",".join(map(str, qubits)) + "]")
        if bits:
            parts.append("c[" + ",".join(map(str, bits)) + "]")
        if zvars:
            parts.append("z[" + ",".join(map(str, zvars)) + "]")
        return f"{label} @ {', '.join(parts)}" if parts else label


__all__ = ["Block"]
