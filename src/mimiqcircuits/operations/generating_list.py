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
"""Utilities for generating operation lists and trees."""

import inspect
from collections import defaultdict
from mimiqcircuits.operations.gates.custom import GateCustom
from mimiqcircuits.operations.gates.customdiagonal import GateCustomDiagonal
from mimiqcircuits.operations.gates.standard.chadamard import GateCH
from mimiqcircuits.operations.gates.standard.cnp import GateCCP
from mimiqcircuits.operations.gates.standard.cnx import GateC3X, GateCCX
from mimiqcircuits.operations.gates.standard.cpauli import GateCX, GateCY, GateCZ
from mimiqcircuits.operations.gates.standard.cphase import GateCP
from mimiqcircuits.operations.gates.standard.crotations import GateCRX, GateCRY, GateCRZ
from mimiqcircuits.operations.gates.standard.cs import GateCS, GateCSDG
from mimiqcircuits.operations.gates.standard.cswap import GateCSWAP
from mimiqcircuits.operations.gates.standard.csx import GateCSX, GateCSXDG
from mimiqcircuits.operations.gates.standard.cu import GateCU
from mimiqcircuits.operations.gates.standard.dcx import GateDCX
from mimiqcircuits.operations.gates.standard.deprecated import GateU1, GateU2, GateU3
from mimiqcircuits.operations.gates.standard.ecr import GateECR
from mimiqcircuits.operations.gates.standard.hadamard import (
    GateH,
    GateHXY,
    GateHXZ,
    GateHYZ,
)
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
from mimiqcircuits.operations.gates.standard.s import GateS, GateSDG
from mimiqcircuits.operations.gates.standard.swap import GateSWAP
from mimiqcircuits.operations.gates.standard.sx import GateSX, GateSXDG
from mimiqcircuits.operations.gates.standard.sy import GateSY, GateSYDG
from mimiqcircuits.operations.gates.standard.t import GateT, GateTDG
from mimiqcircuits.operations.gates.standard.u import GateU


class InheritanceTree:
    """Inheritance tree of the classes derived from a base class.

    Used to build the class hierarchies shown in the documentation of
    :class:`OPERATIONS`, :class:`GATES` and the other listing classes.
    Collect classes with :meth:`extract_classes` (and optionally
    functions with :meth:`extract_gate_functions`), then render the tree
    with :meth:`print_tree` or :meth:`generate_ascii_tree`.

    Args:
        base_class (type): The root of the tree.

    Examples:
        >>> from mimiqcircuits import *
        >>> import mimiqcircuits as mc
        >>> inheritance_tree = InheritanceTree(Operation)
        >>> inheritance_tree.extract_classes(mc)
        >>> inheritance_tree.extract_gate_functions(mc)
        >>> inheritance_tree.print_tree()
        :class:`Operation`
            :class:`AbstractAnnotation`
                :class:`AncillaAlloc`
                :class:`AncillaFree`
                :class:`Detector`
                :class:`Lost`
                :class:`ObservableInclude`
                :class:`QubitCoordinates`
                :class:`Reloaded`
                :class:`ShiftCoordinates`
                :class:`Tick`
            :class:`AbstractClassical`
                :class:`And`
                :class:`Not`
                :class:`Or`
                :class:`SetBit0`
                :class:`SetBit1`
                :class:`Xor`
            :class:`AbstractMeasurement`
                :class:`Measure`
                :class:`MeasurePauli`
                :class:`MeasureReset`
                :class:`MeasureResetX`
                :class:`MeasureResetY`
                :class:`MeasureResetZ`
                :class:`MeasureX`
                :class:`MeasureXX`
                :class:`MeasureY`
                :class:`MeasureYY`
                :class:`MeasureZ`
                :class:`MeasureZZ`
            :class:`AbstractOperator`
                :class:`DiagonalOp`
                :class:`Gate`
                    :class:`Control`
                        :class:`GateC3X`
                        :class:`GateCCP`
                        :class:`GateCCX`
                        :class:`GateCH`
                        :class:`GateCP`
                        :class:`GateCRX`
                        :class:`GateCRY`
                        :class:`GateCRZ`
                        :class:`GateCS`
                        :class:`GateCSDG`
                        :class:`GateCSWAP`
                        :class:`GateCSX`
                        :class:`GateCSXDG`
                        :class:`GateCU`
                        :class:`GateCX`
                        :class:`GateCY`
                        :class:`GateCZ`
                    :class:`Delay`
                    :class:`Diffusion`
                    :class:`GateCall`
                    :class:`GateCustom`
                    :class:`GateCustomDiagonal`
                    :class:`GateDCX`
                    :class:`GateECR`
                    :class:`GateH`
                    :class:`GateHXY`
                    :class:`GateHXZ`
                    :class:`GateHYZ`
                    :class:`GateID`
                    :class:`GateISWAP`
                    :class:`GateP`
                    :class:`GateR`
                    :class:`GateRNZ`
                    :class:`GateRX`
                    :class:`GateRXX`
                    :class:`GateRY`
                    :class:`GateRYY`
                    :class:`GateRZ`
                    :class:`GateRZX`
                    :class:`GateRZZ`
                    :class:`GateSWAP`
                    :class:`GateU`
                    :class:`GateU1`
                    :class:`GateU2`
                    :class:`GateU3`
                    :class:`GateX`
                    :class:`GateXXminusYY`
                    :class:`GateXXplusYY`
                    :class:`GateY`
                    :class:`GateZ`
                    :class:`Inverse`
                        :class:`GateISWAPDG`
                        :class:`GateSDG`
                        :class:`GateSXDG`
                        :class:`GateSYDG`
                        :class:`GateTDG`
                    :class:`Parallel`
                    :class:`PauliString`
                    :class:`PhaseGradient`
                    :class:`PolynomialOracle`
                    :class:`Power`
                        :class:`GateS`
                        :class:`GateSX`
                        :class:`GateSY`
                        :class:`GateT`
                    :class:`QFT`
                    :class:`RPauli`
                :class:`LossyOperator`
                :class:`Operator`
                :class:`Projector0`
                :class:`Projector00`
                :class:`Projector01`
                :class:`Projector1`
                :class:`Projector10`
                :class:`Projector11`
                :class:`ProjectorX0`
                :class:`ProjectorX1`
                :class:`ProjectorY0`
                :class:`ProjectorY1`
                :class:`ProjectorZ0`
                :class:`ProjectorZ1`
                :class:`RescaledGate`
                :class:`SigmaMinus`
                :class:`SigmaPlus`
            :class:`Add`
            :class:`Amplitude`
            :class:`Barrier`
            :class:`Block`
            :class:`BlockCall`
            :class:`BondDim`
            :class:`Check`
            :class:`ExpectationValue`
            :class:`IfStatement`
            :class:`Loss`
            :class:`MeasureCheck`
            :class:`Multiply`
            :class:`Pow`
            :class:`ReadoutErr`
            :class:`Reload`
            :class:`Repeat`
            :class:`SchmidtRank`
            :class:`VonNeumannEntropy`
            :class:`WhileStatement`
            :class:`krauschannel`
                :class:`AmplitudeDamping`
                :class:`Depolarizing`
                :class:`Depolarizing1`
                :class:`Depolarizing2`
                :class:`GeneralizedAmplitudeDamping`
                :class:`Kraus`
                :class:`MixedUnitary`
                :class:`PauliNoise`
                :class:`PauliX`
                :class:`PauliY`
                :class:`PauliZ`
                :class:`PhaseAmplitudeDamping`
                :class:`PrepareStateT`
                :class:`ProjectiveNoise`
                :class:`ProjectiveNoiseX`
                :class:`ProjectiveNoiseY`
                :class:`ProjectiveNoiseZ`
                :class:`Reset`
                :class:`ResetX`
                :class:`ResetY`
                :class:`ResetZ`
                :class:`ThermalNoise`
        <BLANKLINE>
    """

    def __init__(self, base_class):
        """Create an empty tree rooted at ``base_class``."""
        self.base_class = base_class
        self.class_tree = defaultdict(list)
        self.function_registry = defaultdict(list)

    def extract_classes(self, module):
        """Add every subclass of the base class found in ``module``.

        Each class is placed under its first base class. Aliases (names
        that differ from the class's own name) are skipped.

        Args:
            module: The module to scan.
        """
        classes = inspect.getmembers(module, inspect.isclass)

        for name, cls in classes:
            # Skip aliases such as the deprecated `PrepareMagic = PrepareStateT`,
            # so each class appears once under its own name.
            if name != cls.__name__:
                continue
            if issubclass(cls, self.base_class) and cls != self.base_class:
                parent = cls.__bases__[0].__name__
                self.class_tree[parent].append(name)

    def extract_gate_functions(self, module):
        """Add the functions of ``module`` that build instances of the base class.

        Every public function that can be called without arguments is
        called once; if it returns an instance of the base class, it is
        listed under the class of the result. Functions that raise are
        skipped.

        Args:
            module: The module to scan.
        """
        functions = inspect.getmembers(module, inspect.isfunction)

        for name, func in functions:
            try:
                if name.startswith("_"):
                    continue

                sig = inspect.signature(func)
                if any(
                    param.default == inspect.Parameter.empty
                    and param.kind
                    in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                    for param in sig.parameters.values()
                ):
                    continue

                result = func()

                if isinstance(result, self.base_class):
                    parent_name = type(result).__name__
                    self.function_registry[parent_name].append(name)

            except Exception:
                continue

    def generate_tree_output(self, class_name=None, level=0):
        """Return the tree below ``class_name`` as indented ``:class:`` lines.

        Args:
            class_name (str): Name of the class to start from. Defaults to
                the base class.
            level (int): Indentation level of the first line.

        Returns:
            str: The tree, one class or function per line.
        """
        if class_name is None:
            class_name = self.base_class.__name__

        result = "    " * level + f":class:`{class_name}`\n"

        for func in sorted(self.function_registry.get(class_name, [])):
            result += "    " * (level + 1) + f":func:`{func}`\n"

        for child_class in sorted(self.class_tree.get(class_name, [])):
            result += self.generate_tree_output(child_class, level + 1)
        return result

    def print_tree(self):
        """Print the output of :meth:`generate_tree_output`."""
        print(self.generate_tree_output())

    def generate_ascii_tree(self, class_name=None, prefix="", is_last=True):
        """Return the tree below ``class_name`` drawn with ASCII branches.

        Args:
            class_name (str): Name of the class to start from. Defaults to
                the base class.
            prefix (str): Text placed before each line, used for recursion.
            is_last (bool): Whether this node is the last child of its
                parent, used for recursion.

        Returns:
            str: The tree, one class or function per line.
        """
        if class_name is None:
            class_name = self.base_class.__name__

        branch = "└── " if is_last else "├── "
        line = f"{prefix}{branch}{class_name}" if prefix else class_name
        lines = [line]

        # Combine class children and function children
        class_children = sorted(self.class_tree.get(class_name, []))
        func_children = sorted(self.function_registry.get(class_name, []))
        all_children = class_children + func_children

        for idx, child in enumerate(all_children):
            child_is_last = idx == len(all_children) - 1
            next_prefix = prefix + ("    " if is_last else "│   ")

            if child in self.class_tree or child in self.function_registry:
                # Recursive call for subclasses
                lines.append(
                    self.generate_ascii_tree(child, next_prefix, child_is_last)
                )
            else:
                # Function node
                branch_func = "└── " if child_is_last else "├── "
                lines.append(f"{next_prefix}{branch_func}{child}")

        return "\n".join(lines)


class OPERATIONS:
    """All supported operations. Use ``help(OPERATIONS)`` to view the hierarchy.

    This class only carries documentation; it cannot be instantiated.
    """

    def __new__(cls):
        raise NotImplementedError(
            "Use `help(OPERATIONS)` to view available operations."
        )


class ANNOTATIONS:
    """Annotations. Use ``help(ANNOTATIONS)`` to view the hierarchy.

    This class only carries documentation; it cannot be instantiated.
    """

    def __new__(cls):
        raise NotImplementedError(
            "Use `help(ANNOTATIONS)` to view available annotations."
        )


class GATES:
    """Gates. Use ``help(GATES)`` to view the hierarchy.

    This class only carries documentation; it cannot be instantiated.
    """

    def __new__(cls):
        raise NotImplementedError("Use `help(GATES)` to view available gates.")


class KRAUSCHANNELS:
    """Kraus channels. Use ``help(KRAUSCHANNELS)`` to view the hierarchy.

    This class only carries documentation; it cannot be instantiated.
    """

    def __new__(cls):
        raise NotImplementedError(
            "Use `help(KRAUSCHANNELS)` to view available channels."
        )


class MEASUREMENTS:
    """Measurements. Use ``help(MEASUREMENTS)`` to view the hierarchy.

    This class only carries documentation; it cannot be instantiated.
    """

    def __new__(cls):
        raise NotImplementedError(
            "Use `help(MEASUREMENTS)` to view available measurements."
        )


class SIMPLEGATES:
    """Standard gates. Use ``help(SIMPLEGATES)`` to view the full list.

    This class only carries documentation; it cannot be instantiated.
    """

    def __new__(cls):
        raise NotImplementedError(
            "Use `help(SIMPLEGATES)` to view available simple gates."
        )


class CLASSICALOPERATIONS:
    """Classical operations. Use ``help(CLASSICALOPERATIONS)`` to view the hierarchy.

    This class only carries documentation; it cannot be instantiated.
    """

    def __new__(cls):
        raise NotImplementedError(
            "Use `help(CLASSICALOPERATIONS)` to view available classical operations."
        )

    @staticmethod
    def list():
        """Return the classes of the standard gates.

        Returns:
            list: Gate classes, from :class:`GateCustom` to :class:`GateP`.
        """
        return [
            GateCustom,
            GateCustomDiagonal,
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


SIMPLEGATES_GROUPS = {
    "Single qubit gates": [
        GateX,
        GateY,
        GateZ,
        GateH,
        GateS,
        GateSDG,
        GateT,
        GateTDG,
        GateSX,
        GateSXDG,
        GateSY,
        GateSYDG,
        GateID,
    ],
    "Single qubit gates (parametric)": [
        GateU,
        GateP,
        GateRX,
        GateRY,
        GateRZ,
        GateR,
        GateU1,
        GateU2,
        GateU3,
    ],
    "Two qubit gates": [
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
    ],
    "Two qubit gates (parametric)": [
        GateCU,
        GateCP,
        GateCRX,
        GateCRY,
        GateCRZ,
        GateRXX,
        GateRYY,
        GateRZZ,
        GateRZX,
        GateXXplusYY,
        GateXXminusYY,
    ],
    "Multi-qubit gates (special)": [
        GateCCX,
        GateC3X,
        GateCCP,
        GateCSWAP,
        GateHXY,
        GateHXZ,
        GateHYZ,
        GateCustom,
        GateCustomDiagonal,
    ],
}


def generate_ascii_gate_tree_grouped(groups):
    """Return a text listing of gate classes grouped by section.

    Args:
        groups (dict): Map from section title to a list of gate classes,
            such as ``SIMPLEGATES_GROUPS``.

    Returns:
        str: The sections in order, each followed by its gate names sorted
        alphabetically and drawn as a tree.
    """
    lines = ["Available quantum gates:\n"]
    for section, gates in groups.items():
        lines.append(f"*{section}*")
        gate_names = sorted(g.__name__ for g in gates)
        for i, name in enumerate(gate_names):
            branch = "└── " if i == len(gate_names) - 1 else "├── "
            lines.append(branch + name)
        lines.append("")  # empty line between groups
    return "\n".join(lines)


# RST-safe fenced code block for Sphinx docstrings
def _rst_fenced(text: str) -> str:
    """Wrap text in a RST fenced code block for Sphinx docstrings."""
    indented = "\n".join("   " + line for line in text.splitlines())
    return f"\n.. code-block::\n\n{indented}\n"


def attach_inheritance_tree_to_docstring(
    base, module, target_object, title, start_from
):
    """Append an ASCII inheritance tree to the docstring of ``target_object``.

    The tree is wrapped in an RST ``code-block`` so that Sphinx keeps its
    layout.

    Args:
        base: Root class used to collect the classes.
        module: Module scanned for classes and functions.
        target_object: Object whose ``__doc__`` is extended.
        title (str): Heading written above the tree.
        start_from (str): Name of the class the tree starts from.
    """

    tree = InheritanceTree(base)
    tree.extract_classes(module)
    tree.extract_gate_functions(module)

    ascii_tree = tree.generate_ascii_tree(class_name=start_from)

    block = _rst_fenced(ascii_tree)

    existing = target_object.__doc__ or ""
    target_object.__doc__ = existing + f"\n\n{title}:\n" + block


__all__ = ["OPERATIONS", "GATES", "KRAUSCHANNELS", "MEASUREMENTS", "SIMPLEGATES"]
