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

import warnings

from mimiqcircuits.classical.xor_operation import Xor


class ParityCheck:
    """
    Deprecated alias of :class:`Xor`.

    ``ParityCheck(N)`` returns ``Xor(N)``, which computes the same parity of
    N-1 classical bits into the first bit, and emits a ``DeprecationWarning``.
    No object is ever an instance of ``ParityCheck``.

    Examples:
        >>> import warnings
        >>> from mimiqcircuits import *
        >>> with warnings.catch_warnings():
        ...     warnings.simplefilter("ignore", DeprecationWarning)
        ...     ParityCheck(5)
        c[?0] = c[?1] ^ c[?2] ^ c[?3] ^ c[?4]
    """

    def __new__(cls, N=3):
        warnings.warn(
            "ParityCheck is deprecated, use Xor instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return Xor(N)


__all__ = ["ParityCheck"]
