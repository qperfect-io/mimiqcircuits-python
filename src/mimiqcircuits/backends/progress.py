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

"""Progress reporting for :meth:`LocalBackend.execute`.

The ``execute`` driver and backends open named stages and advance them; the
concrete sink decides whether and how to render. :class:`NoProgress` is
the zero-cost default; :class:`TqdmProgress` draws bars with tqdm.

A sink is driven through three verbs, all no-ops on :class:`NoProgress`:

- ``progress.stage(name, total=...)`` opens a bar and returns a stage.
- ``stage.step(n=1)`` advances it.
- ``stage.finish()`` closes it.

The driver owns the trajectory bar and, on the single-evolve path, an
execution bar fed by the backend's per-step ``callback``. A backend with
a countable compression step emits its own stage by overriding
:meth:`LocalBackend.compile_progress`.
"""

from __future__ import annotations

import abc


class Stage(abc.ABC):
    """A single progress bar opened by :meth:`Progress.stage`.

    Subclasses implement :meth:`step` and :meth:`finish`.
    """

    @abc.abstractmethod
    def step(self, n: int = 1) -> None:
        """Advance the bar by ``n`` steps."""

    @abc.abstractmethod
    def finish(self) -> None:
        """Close the bar."""


class Progress(abc.ABC):
    """Destination for execution progress, given to
    :meth:`LocalBackend.execute` as ``progress``.

    Subclasses implement :meth:`stage`. :class:`NoProgress` draws nothing and
    :class:`TqdmProgress` draws tqdm bars.

    Examples:
        A sink that prints every stage it is asked to open.

        >>> from mimiqcircuits import *
        >>> from mimiqcircuits.backends import NoProgress, Progress
        >>> class PrintProgress(Progress):
        ...     def stage(self, name, *, total=None):
        ...         print(f"{name}: {total} steps")
        ...         return NoProgress().stage(name, total=total)
        >>> bar = PrintProgress().stage("trajectories", total=100)
        trajectories: 100 steps
        >>> bar.step()
        >>> bar.finish()
    """

    @abc.abstractmethod
    def stage(self, name: str, *, total=None) -> Stage:
        """Open a stage.

        Args:
            name: Name of the stage, such as ``"trajectories"``,
                ``"execute"`` or ``"compression"``.
            total: Number of steps, or ``None`` when unknown.

        Returns:
            The :class:`Stage` to advance and close.
        """


class _NoStage(Stage):
    def step(self, n: int = 1) -> None:
        pass

    def finish(self) -> None:
        pass


_NO_STAGE = _NoStage()


class NoProgress(Progress):
    """Sink that draws nothing; the default of :meth:`LocalBackend.execute`.

    Its stages ignore :meth:`Stage.step` and :meth:`Stage.finish`.
    """

    def stage(self, name: str, *, total=None) -> Stage:
        return _NO_STAGE


class _TqdmStage(Stage):
    def __init__(self, name: str, total, leave: bool):
        from tqdm.auto import tqdm

        self._bar = tqdm(total=total, desc=str(name), leave=leave)

    def step(self, n: int = 1) -> None:
        self._bar.update(n)

    def finish(self) -> None:
        self._bar.close()


class TqdmProgress(Progress):
    """Sink that draws each stage as a tqdm bar.

    ``execute(progress=True)`` uses it with the default arguments.

    Args:
        leave: Keep each bar on screen after its stage finishes. The default,
            ``False``, clears it, so successive stages do not pile up.
    """

    def __init__(self, *, leave: bool = False):
        self._leave = leave

    def stage(self, name: str, *, total=None) -> Stage:
        return _TqdmStage(name, total, self._leave)


def to_progress(progress) -> Progress:
    """Turn the ``progress`` argument of :meth:`LocalBackend.execute` into a
    :class:`Progress`.

    Args:
        progress: A :class:`Progress`, returned as it is, or a bool:
            ``True`` gives :class:`TqdmProgress`, ``False`` gives
            :class:`NoProgress`.

    Returns:
        The :class:`Progress` sink.

    Raises:
        TypeError: For any other type.
    """
    if isinstance(progress, Progress):
        return progress
    if isinstance(progress, bool):
        return TqdmProgress() if progress else NoProgress()
    raise TypeError(
        f"progress must be a bool or Progress, got {type(progress).__name__}"
    )


def execution_progress_callback(progress: Progress, user, box: list):
    """Wrap a per-step ``callback`` so the execution bar advances on each
    step the backend reports, then forwards to ``user`` unchanged.

    Backends report a step by calling ``callback(i, total)``: ``i`` is the
    1-based step index and ``total`` the step count (or ``None`` when
    unknown). The stage is opened lazily on the first step so its total
    comes from the backend. ``box`` is a one-element list that receives
    the stage so the driver can close it once evolution returns.
    """

    def cb(i, total=None, *args, **kwargs):
        if box[0] is None:
            box[0] = progress.stage(
                "execute", total=total if isinstance(total, int) else None
            )
        box[0].step()
        if user is not None:
            return user(i, total, *args, **kwargs)
        return None

    return cb
