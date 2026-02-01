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
from mimiqcircuits import *


class InheritanceTree:
    """Class to generate and display an inheritance tree for classes that inherit from a base class.

    >>> from mimiqcircuits import *
    >>> import mimiqcircuits as mc
    >>> inheritance_tree = InheritanceTree(Operation)
    >>> inheritance_tree.extract_classes(mc)
    >>> inheritance_tree.extract_gate_functions(mc)
    >>> inheritance_tree.print_tree()
    :class:`Operation`
        :class:`AbstractAnnotation`
            :class:`Detector`
            :class:`ObservableInclude`
            :class:`QubitCoordinates`
            :class:`ShiftCoordinates`
            :class:`Tick`
        :class:`AbstractClassical`
            :class:`And`
            :class:`Not`
            :class:`Or`
            :class:`ParityCheck`
            :class:`SetBit0`
            :class:`SetBit1`
            :class:`Xor`
        :class:`AbstractMeasurement`
            :class:`Measure`
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
                    :func:`GateC3X`
                    :func:`GateCCX`
                    :func:`GateCH`
                    :func:`GateCS`
                    :func:`GateCSDG`
                    :func:`GateCSWAP`
                    :func:`GateCSX`
                    :func:`GateCSXDG`
                    :func:`GateCX`
                    :func:`GateCY`
                    :func:`GateCZ`
                :class:`Delay`
                :class:`Diffusion`
                :class:`GateCall`
                :class:`GateCustom`
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
                    :func:`GateSDG`
                    :func:`GateSXDG`
                    :func:`GateTDG`
                    :class:`GateSYDG`
                :class:`Parallel`
                :class:`PauliString`
                :class:`PhaseGradient`
                :class:`PolynomialOracle`
                :class:`Power`
                    :func:`GateS`
                    :func:`GateSX`
                    :func:`GateT`
                    :class:`GateSY`
                :class:`QFT`
                :class:`RPauli`
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
        :class:`BondDim`
        :class:`ExpectationValue`
        :class:`IfStatement`
        :class:`Multiply`
        :class:`Pow`
        :class:`ReadoutErr`
        :class:`Repeat`
        :class:`SchmidtRank`
        :class:`VonNeumannEntropy`
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
        """Initialize with the base class (e.g., Operation)."""
        self.base_class = base_class
        self.class_tree = defaultdict(list)
        self.function_registry = defaultdict(list)  # <-- added

    def extract_classes(self, module):
        """Extract all classes from the module that inherit from the base class."""
        classes = inspect.getmembers(module, inspect.isclass)

        for name, cls in classes:
            if issubclass(cls, self.base_class) and cls != self.base_class:
                parent = cls.__bases__[0].__name__
                self.class_tree[parent].append(name)

    def extract_gate_functions(self, module):
        """Automatically extract gate-like functions that return subclasses of the base class."""
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
        """Recursively generate the class hierarchy tree as a string."""
        if class_name is None:
            class_name = self.base_class.__name__

        result = "    " * level + f":class:`{class_name}`\n"

        for func in sorted(self.function_registry.get(class_name, [])):  # <-- added
            result += "    " * (level + 1) + f":func:`{func}`\n"  # <-- added

        for child_class in sorted(self.class_tree.get(class_name, [])):
            result += self.generate_tree_output(child_class, level + 1)
        return result

    def print_tree(self):
        """Print the generated tree."""
        print(self.generate_tree_output())

    def generate_ascii_tree(self, class_name=None, prefix="", is_last=True):
        """
        Generate a clean ASCII tree showing class/function hierarchy.
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


def attach_inheritance_tree_to_docstring(
    base_class,
    module,
    target_object=None,
    title="Hierarchy of Operations",
    start_from=None,
):
    tree = InheritanceTree(base_class)
    tree.extract_classes(module)
    tree.extract_gate_functions(module)

    tree_str = tree.generate_ascii_tree(class_name=start_from)

    target = target_object if target_object is not None else base_class

    if target.__doc__ is None:
        target.__doc__ = ""

    target.__doc__ += f"\n\n{title}:\n{tree_str}"


class OPERATIONS:
    """All supported quantum operations. Use `help(OPERATIONS)` to view the hierarchy."""

    def __new__(cls):
        raise NotImplementedError(
            "Use `help(OPERATIONS)` to view available operations."
        )


class ANNOTATIONS:
    """Annotation-related operations. Use `help(ANNOTATIONS)` to view the hierarchy."""

    def __new__(cls):
        raise NotImplementedError(
            "Use `help(ANNOTATIONS)` to view available annotations."
        )


class GATES:
    """Gate operations. Use `help(GATES)` to view the hierarchy."""

    def __new__(cls):
        raise NotImplementedError("Use `help(GATES)` to view available gates.")


class KRAUSCHANNELS:
    """Kraus channel operations. Use `help(KRAUSCHANNELS)` to view the hierarchy."""

    def __new__(cls):
        raise NotImplementedError(
            "Use `help(KRAUSCHANNELS)` to view available channels."
        )


class MEASUREMENTS:
    """Measurement operations. Use `help(MEASUREMENTS)` to view the hierarchy."""

    def __new__(cls):
        raise NotImplementedError(
            "Use `help(MEASUREMENTS)` to view available measurements."
        )


class SIMPLEGATES:
    """Simple gates. Use `help(SIMPLEGATES)` to view the full list."""

    def __new__(cls):
        raise NotImplementedError(
            "Use `help(SIMPLEGATES)` to view available simple gates."
        )


class CLASSICALOPERATIONS:
    """Classical operations. Use `help(CLASSICALOPERATIONS)` to view the hierarchy."""

    def __new__(cls):
        raise NotImplementedError(
            "Use `help(CLASSICALOPERATIONS)` to view available classical operations."
        )

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
    ],
}


def generate_ascii_gate_tree_grouped(groups):
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
    """
    Attach an ASCII inheritance tree to a class or module docstring,
    wrapped in a proper Sphinx RST code-block so formatting is preserved.
    """

    tree = InheritanceTree(base)
    tree.extract_classes(module)
    tree.extract_gate_functions(module)

    ascii_tree = tree.generate_ascii_tree(class_name=start_from)

    block = _rst_fenced(ascii_tree)

    existing = target_object.__doc__ or ""
    target_object.__doc__ = existing + f"\n\n{title}:\n" + block


__all__ = ["OPERATIONS", "GATES", "KRAUSCHANNELS", "MEASUREMENTS", "SIMPLEGATES"]
