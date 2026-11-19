#
# Copyright © 2023-2026 QPerfect. All Rights Reserved.
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

"""Default probe-circuit constructors for the anti-capability harness.

Each entry maps a capability
token to a zero-argument factory that returns a minimal
:class:`Circuit` whose execution requires the named capability.

Registered eagerly at module-import time so callers see the populated
:data:`CAP_PROBES` registry after `from mimiqcircuits.backends import
CAP_PROBES`.
"""

from __future__ import annotations

from mimiqcircuits.circuit import Circuit
from mimiqcircuits.operations.gates.standard.hadamard import GateH
from mimiqcircuits.operations.gates.standard.pauli import GateX
from mimiqcircuits.operations.gates.standard.rotations import GateRX
from mimiqcircuits.operations.ifstatement import IfStatement
from mimiqcircuits.operations.measure import Measure
from mimiqcircuits.operations.noisechannel.kraus import Kraus
from mimiqcircuits.operations.noisechannel.standards.depolarizing import Depolarizing1
from mimiqcircuits.operations.operators.custom import Operator
from mimiqcircuits.operations.operators.lossyoperator import LossyOperator
from mimiqcircuits.operations.reset import Reset
from mimiqcircuits.bitstrings import BitString

from mimiqcircuits.backends.capabilities import register_cap_probe


def _probe_feed_forward():
    """A classically-controlled gate after a measurement."""
    c = Circuit()
    c.push(Measure(), 0, 0)
    c.push(IfStatement(GateX(), BitString("1")), 0, 0)
    return c


def _probe_midcircuit_measure():
    """Measurement followed by a gate on the same qubit."""
    c = Circuit()
    c.push(Measure(), 0, 0)
    c.push(GateX(), 0)
    return c


def _probe_midcircuit_reset():
    """Reset surrounded by gates."""
    c = Circuit()
    c.push(GateH(), 0)
    c.push(Reset(), 0)
    c.push(GateH(), 0)
    return c


def _probe_reset_after_measure():
    """Reset immediately after a measurement on the same qubit."""
    c = Circuit()
    c.push(Measure(), 0, 0)
    c.push(Reset(), 0)
    return c


def _probe_noise():
    """A trace-preserving mixed-unitary Kraus channel."""
    c = Circuit()
    c.push(GateH(), 0)
    c.push(Depolarizing1(0.1), 0)
    return c


def _probe_loss():
    """A runtime loss channel: a ``Kraus`` with a ``LossyOperator`` branch,
    whose outcome depends on the quantum state. The default ``can_handle``
    rejects these on backends that don't advertise ``"loss"`` (see
    ``Backend.can_handle``). Pre-resolvable loss (``Loss``/``Reload``/
    ``Check``/``MeasureCheck``) is resolved upstream and needs no capability.
    """
    import numpy as np

    survival = Operator(np.array([[1, 0], [0, np.sqrt(0.9)]], dtype=np.complex128))
    lossy = LossyOperator(np.array([[0, np.sqrt(0.1)], [0, 0]], dtype=np.complex128))
    c = Circuit()
    c.push(Kraus([survival, lossy]), 0)
    return c


def _probe_parametric():
    """A symbolic rotation gate with an unbound parameter. Executing
    this requires either ``:parametric`` support (bind path) or
    raises an unbound-symbolic error.
    """
    import sympy

    theta = sympy.Symbol("theta")
    c = Circuit()
    c.push(GateRX(theta), 0)
    return c


# Register at import time so callers see a populated CAP_PROBES dict.
register_cap_probe("feed_forward", _probe_feed_forward)
register_cap_probe("midcircuit_measure", _probe_midcircuit_measure)
register_cap_probe("midcircuit_reset", _probe_midcircuit_reset)
register_cap_probe("reset_after_measure", _probe_reset_after_measure)
register_cap_probe("noise", _probe_noise)
register_cap_probe("loss", _probe_loss)
register_cap_probe("parametric", _probe_parametric)


__all__ = [
    "_probe_feed_forward",
    "_probe_midcircuit_measure",
    "_probe_midcircuit_reset",
    "_probe_reset_after_measure",
    "_probe_noise",
    "_probe_loss",
    "_probe_parametric",
]
