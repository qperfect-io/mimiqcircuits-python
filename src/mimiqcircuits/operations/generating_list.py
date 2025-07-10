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

import inspect
from collections import defaultdict


class InheritanceTree:
    """Class to generate and display an inheritance tree for classes that inherit from a base class.

    >>> from mimiqcircuits import *
    >>> import mimiqcircuits as mc
    >>> inheritance_tree = InheritanceTree(Operation)
    >>> inheritance_tree.extract_classes(mc)
    >>> inheritance_tree.print_tree()
    :class:`Operation`
        :class:`AbstractAnnotation`
            :class:`Detector`
            :class:`ObservableInclude`
            :class:`QubitCoordinates`
            :class:`ShiftCoordinates`
            :class:`Tick`
        :class:`AbstractMeasurement`
            :class:`Measure`
            :class:`MeasureReset`
            :class:`MeasureResetX`
            :class:`MeasureResetY`
            :class:`MeasureResetZ`
            :class:`MeasureX`
            :class:`MeasureY`
            :class:`MeasureZ`
        :class:`AbstractOperator`
            :class:`DiagonalOp`
            :class:`Gate`
                :class:`Control`
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
                    :class:`GateSYDG`
                :class:`Parallel`
                :class:`PauliString`
                :class:`PhaseGradient`
                :class:`PolynomialOracle`
                :class:`Power`
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
        :class:`MeasureXX`
        :class:`MeasureYY`
        :class:`MeasureZZ`
        :class:`Multiply`
        :class:`Not`
        :class:`Pow`
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

    def extract_classes(self, module):
        """Extract all classes from the module that inherit from the base class."""
        classes = inspect.getmembers(module, inspect.isclass)

        for name, cls in classes:
            if issubclass(cls, self.base_class) and cls != self.base_class:
                parent = cls.__bases__[0].__name__
                self.class_tree[parent].append(name)

    def generate_tree_output(self, class_name=None, level=0):
        """Recursively generate the class hierarchy tree as a string."""
        if class_name is None:
            class_name = self.base_class.__name__

        result = "    " * level + f":class:`{class_name}`\n"

        for child_class in sorted(self.class_tree.get(class_name, [])):
            result += self.generate_tree_output(child_class, level + 1)
        return result

    def print_tree(self):
        """Print the generated tree."""
        print(self.generate_tree_output())
