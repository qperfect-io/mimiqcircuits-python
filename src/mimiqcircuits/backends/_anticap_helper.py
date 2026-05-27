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

"""Shared helper for the ``test_can_handle_rejects_anticap`` conformance
test family. Each downstream backend repository (`mimiqengines-python`,
`tensorweaver-python`, `quantanium-python-rs`, and the in-tree
:class:`MimiqRemoteBackend`) imports :func:`assert_anticap_rejected`
and feeds its concrete simulator.

The helper iterates :data:`mimiqcircuits.backends.CAP_PROBES`,
constructs the probe circuit for each undeclared capability, and
asserts the backend rejects it either by:

- returning :class:`Inadmissible` from ``can_handle``, or
- raising on ``execute`` / ``compile``.

Silent acceptance of an undeclared capability is a "positive lie" —
exactly what this kit exists to catch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mimiqcircuits.backends.capabilities import (
    CAP_PROBES,
    Inadmissible,
)

if TYPE_CHECKING:
    from mimiqcircuits.backends.backend import Backend


def assert_anticap_rejected(sim: "Backend", *, skip_caps: tuple[str, ...] = ()) -> dict:
    """Run the anti-capability harness against ``sim``.

    For every capability in :data:`CAP_PROBES` that ``sim`` does NOT
    declare, build the probe circuit and assert rejection. Returns a
    dict mapping ``cap`` to the rejection mode (``"can_handle"`` or
    ``"raises"``) for use in test-side assertions / diagnostics.

    ``skip_caps`` lets a backend opt out of probing a specific
    capability whose probe is known to be too aggressive for the
    backend's API surface (e.g. ``parametric`` on a backend whose
    forwarder normalises symbolic params before reaching dispatch).
    Skipped caps should be rare and documented at the call site.
    """
    declared = set(sim.capabilities())
    rejections: dict[str, str] = {}
    failures: list[str] = []

    for cap, factory in CAP_PROBES.items():
        if cap in declared or cap in skip_caps:
            continue

        circuit = factory()

        try:
            verdict = sim.can_handle(circuit)
        except Exception:
            # can_handle itself raised — count as rejection.
            rejections[cap] = "raises"
            continue

        if isinstance(verdict, Inadmissible):
            rejections[cap] = "can_handle"
            continue

        # can_handle accepted the circuit; the backend must raise on
        # the actual execution path.
        try:
            sim.execute(circuit, nsamples=1)
        except Exception:
            rejections[cap] = "raises"
            continue

        # Silent acceptance — positive-lie capability.
        failures.append(cap)

    if failures:
        raise AssertionError(
            f"{type(sim).__name__} silently accepts undeclared "
            f"capabilities: {sorted(failures)}. Either declare them in "
            f"capabilities() or wire can_handle / execute to reject."
        )

    return rejections


__all__ = ["assert_anticap_rejected"]
