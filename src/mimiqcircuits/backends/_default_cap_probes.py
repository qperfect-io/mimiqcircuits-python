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

Mirrors the Julia ``_register_default_cap_probes!`` set in
``AbstractQCSs.jl/src/capabilities.jl``. Each entry maps a capability
token to a zero-argument factory that returns a minimal
:class:`Circuit` whose execution requires the named capability.

Registered eagerly at module-import time so callers see the populated
:data:`CAP_PROBES` registry after `from mimiqcircuits.backends import
CAP_PROBES`.
"""

from __future__ import annotations

import mimiqcircuits as mc
from mimiqcircuits import BitString

from mimiqcircuits.backends.capabilities import register_cap_probe


def _probe_feed_forward():
    """A classically-controlled gate after a measurement."""
    c = mc.Circuit()
    c.push(mc.Measure(), 0, 0)
    c.push(mc.IfStatement(mc.GateX(), BitString("1")), 0, 0)
    return c


def _probe_midcircuit_measure():
    """Measurement followed by a gate on the same qubit."""
    c = mc.Circuit()
    c.push(mc.Measure(), 0, 0)
    c.push(mc.GateX(), 0)
    return c


def _probe_midcircuit_reset():
    """Reset surrounded by gates."""
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.Reset(), 0)
    c.push(mc.GateH(), 0)
    return c


def _probe_reset_after_measure():
    """Reset immediately after a measurement on the same qubit."""
    c = mc.Circuit()
    c.push(mc.Measure(), 0, 0)
    c.push(mc.Reset(), 0)
    return c


def _probe_noise():
    """A trace-preserving mixed-unitary Kraus channel."""
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.Depolarizing1(0.1), 0)
    return c


def _probe_loss():
    """A non-trace-preserving loss-bearing operation. The default
    ``can_handle`` rejects these on backends that don't advertise
    ``"loss"`` (see ``Backend.can_handle``).
    """
    c = mc.Circuit()
    c.push(mc.GateH(), 0)
    c.push(mc.LossErr(0.1), 0)
    return c


def _probe_parametric():
    """A symbolic rotation gate with an unbound parameter. Executing
    this requires either ``:parametric`` support (bind path) or
    raises an unbound-symbolic error.
    """
    import sympy

    theta = sympy.Symbol("theta")
    c = mc.Circuit()
    c.push(mc.GateRX(theta), 0)
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
