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
"""Concrete circuit-preparation passes.

Each wraps a circuit rewrite behind the :class:`AbstractPass` interface so
it composes in a :class:`PassPipeline`. The clustering gate-fusion pass
lives in :mod:`mimiqcircuits.fusion` as :class:`FusePass`.

Qubit reordering and SWAP removal are server-side only in Python: they have
no local implementation, so a remote backend translates the corresponding
``reorderqubits`` / ``remove_swaps`` knobs into server options directly
rather than through a client-side pass.
"""

from mimiqcircuits.backends.passes import AbstractPass, PassResult, PassSpec


class CanonicalDecomposePass(AbstractPass):
    """Pass that decomposes a circuit to its canonical basis.

    Wraps :meth:`Circuit.decompose`. Qubit indices are unchanged, so
    :attr:`PassResult.qubit_permutation` is ``None``.
    """

    def spec(self):
        return PassSpec.from_dict("canonical_decompose")

    def apply(self, ctx, circuit):
        return circuit.decompose(), PassResult()
