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

import math
from statistics import mean, median, stdev


def _format_duration(t):
    if t == 0:
        return "0s"

    # round to 3 significant digits
    rt = float(f"{t:.3g}")

    if rt >= 1:
        minutes = int(rt // 60)
        seconds = rt % 60

        hours = int(minutes // 60)
        minutes = minutes % 60

        days = int(hours // 24)
        hours = hours % 24

        strings = []
        if days > 0:
            strings.append(f"{days}d")
        if hours > 0:
            strings.append(f"{hours}h")
        if minutes > 0:
            strings.append(f"{minutes}m")
        if seconds > 0:
            # Check if seconds is an integer effectively
            if seconds.is_integer():
                strings.append(f"{int(seconds)}s")
            else:
                strings.append(f"{seconds}s")

        return " ".join(strings)
    elif rt >= 1e-3:
        val = float(f"{rt * 1e3:.3g}")
        if val.is_integer():
            return f"{int(val)}ms"
        return f"{val}ms"
    elif rt >= 1e-6:
        val = float(f"{rt * 1e6:.3g}")
        if val.is_integer():
            return f"{int(val)}μs"
        return f"{val}μs"
    else:
        val = float(f"{rt * 1e9:.3g}")
        if val.is_integer():
            return f"{int(val)}ns"
        return f"{val}ns"


def _get_sorted_timings(timings):
    valid_timings = [(k, v) for k, v in timings.items() if v > 1e-7]
    # Sort so "total" is last (key 2), others are first (key 1), then alphabetically
    valid_timings.sort(key=lambda x: (2 if x[0] == "total" else 1, x[0]))
    return valid_timings


def _display_stdev(values):
    """Return a display-safe standard deviation.

    `statistics.stdev` raises on `nan`/`inf` inputs; for display we
    prefer surfacing `nan` rather than crashing repr/html rendering.
    """
    if len(values) < 2 or any(not math.isfinite(v) for v in values):
        return math.nan
    return stdev(values)


class QCSResults:
    """Results of a circuit execution.

    Remote executions (``get_result``/``get_results`` on a connection) and
    local backends return this object. Every argument is optional and stored
    as an attribute of the same name; missing lists and dictionaries start
    empty.

    Args:
        simulator (str, optional): The name of the simulator.
        version (str, optional): The version of the simulator.
        fidelities (list, optional): One fidelity estimate per execution of
            the circuit.
        avggateerrors (list, optional): One average multi-qubit gate error
            estimate per execution.
        cstates (list[BitString], optional): The sampled classical registers,
            one :class:`BitString` per sample.
        zstates (list, optional): The sampled complex-valued ``z`` registers,
            one sequence of numbers per sample.
        amplitudes (dict, optional): Statevector amplitudes, mapping a
            :class:`BitString` to a complex number.
        timings (dict, optional): Time in seconds spent in each phase of the
            computation, keyed by phase name (for example ``"total"``).

    Examples:
        Results normally come from a connection:

        >>> from mimiqcircuits import *
        >>> conn = MimiqConnection()
        >>> conn.connect()  # doctest: +SKIP
        >>> c = Circuit()
        >>> c.push(GateH(), 0)
        1-qubit circuit with 1 instruction:
        └── H @ q[0]
        <BLANKLINE>
        >>> c.push(Measure(), 0, 0)
        1-qubit, 1-bit circuit with 2 instructions:
        ├── H @ q[0]
        └── M @ q[0], c[0]
        <BLANKLINE>
        >>> job = conn.submit(c, nsamples=1000)  # doctest: +SKIP
        >>> res = conn.get_result(job)  # doctest: +SKIP

        Here we build one by hand to show how to inspect it: five samples of
        a two-bit register and one amplitude.

        >>> res = QCSResults(
        ...     simulator="MIMIQ-StateVector",
        ...     version="1.0",
        ...     fidelities=[1.0],
        ...     avggateerrors=[0.0],
        ...     cstates=[BitString("00"), BitString("11"), BitString("11"),
        ...              BitString("00"), BitString("11")],
        ...     amplitudes={BitString("00"): 0.7071 + 0j},
        ...     timings={"total": 0.0012, "sample": 0.0002},
        ... )

        The printed summary shows timings, fidelity, and the most sampled
        states.

        >>> res
        QCSResults:
        ├── simulator: MIMIQ-StateVector 1.0
        ├── timings:
        │    ├── sample time: 200μs
        │    └── total time: 1.2ms
        ├── fidelity estimate: 1
        ├── average multi-qubit gate error estimate: 0
        ├── most sampled:
        │    ├── bs"11" => 3
        │    └── bs"00" => 2
        ├── 1 executions
        ├── 1 amplitudes
        └── 5 samples

        The raw data is available as attributes.

        >>> len(res.cstates)
        5
        >>> res.cstates[0]
        bs"00"
        >>> res.amplitudes[BitString("00")]
        (0.7071+0j)

        :meth:`histogram` counts how often each state was sampled.

        >>> res.histogram()
        {bs"00": 2, bs"11": 3}
    """

    def __init__(
        self,
        simulator=None,
        version=None,
        fidelities=None,
        avggateerrors=None,
        cstates=None,
        zstates=None,
        amplitudes=None,
        timings=None,
    ):
        self.simulator = simulator
        self.version = version
        self.fidelities = fidelities if fidelities is not None else []
        self.avggateerrors = avggateerrors if avggateerrors is not None else []
        self.cstates = cstates if cstates is not None else []
        self.zstates = zstates if zstates is not None else []
        self.amplitudes = amplitudes if amplitudes is not None else {}
        self.timings = timings if timings is not None else {}

    def __repr__(self):
        result_str = "QCSResults:\n"

        if self.simulator is not None:
            result_str += f"├── simulator: {self.simulator} {self.version}\n"

        # filter out the ones < 1e-7 and sort
        timings = _get_sorted_timings(self.timings)

        if timings:
            result_str += "├── timings:\n"
            for key, value in timings[:-1]:
                result_str += f"│    ├── {key} time: {_format_duration(value)}\n"

            key, value = timings[-1]
            result_str += f"│    └── {key} time: {_format_duration(value)}\n"

        if len(self.fidelities) == 1:
            from mimiqcircuits.backends.fidelity import as_lower_bound, _to_fidelity
            fid = as_lower_bound(_to_fidelity(self.fidelities[0]))
            result_str += f"├── fidelity estimate: {fid:.3g}\n"
            result_str += f"├── average multi-qubit gate error estimate: {self.avggateerrors[0]:.3g}\n"
        elif len(self.fidelities) != 0:
            from mimiqcircuits.backends.fidelity import as_lower_bound, _to_fidelity
            fidelity_values = [
                as_lower_bound(_to_fidelity(f)) for f in self.fidelities
            ]
            result_str += "├── fidelity estimate:\n"
            result_str += f"│    ├── min, max: {min(fidelity_values):.3g}, {max(fidelity_values):.3g}\n"
            result_str += f"│    ├── mean: {mean(fidelity_values):.3g}\n"
            result_str += f"│    ├── median: {median(fidelity_values):.3g}\n"
            result_str += f"│    └── std: {_display_stdev(fidelity_values):.3g}\n"
            result_str += "├── average multi-qubit gate error estimate:\n"
            result_str += f"│    ├── min, max: {min(self.avggateerrors):.3g}, {max(self.avggateerrors):.3g}\n"
            result_str += f"│    ├── mean: {mean(self.avggateerrors):.3g}\n"
            result_str += f"│    ├── median: {median(self.avggateerrors):.3g}\n"
            result_str += f"│    └── std: {_display_stdev(self.avggateerrors):.3g}\n"

        if len(self.cstates) != 0:
            hist = self.histogram()
            outcomes = sorted(hist, key=hist.get, reverse=True)[0:5]
            result_str += "├── most sampled:\n"
            for bs in outcomes[:-1]:
                result_str += f'│    ├── bs"{bs.to01()}" => {hist[bs]}\n'

            bs = outcomes[-1]
            result_str += f'│    └── bs"{bs.to01()}" => {hist[bs]}\n'

        if len(self.zstates) > 0 and any(len(zstate) > 0 for zstate in self.zstates):
            result_str += "├── zreg (most sampled):\n"
            z_hist = self.histzvars()
            for zstate, count in list(z_hist.items())[:-1]:
                result_str += f"│    ├── [{', '.join(map(str, zstate))}] => {count}\n"
            zstate, count = list(z_hist.items())[-1]
            result_str += f"│    └── [{', '.join(map(str, zstate))}] => {count}\n"

        result_str += f"├── {len(self.fidelities)} executions\n"
        result_str += f"├── {len(self.amplitudes)} amplitudes\n"
        result_str += f"└── {len(self.cstates)} samples"

        return result_str

    def _repr_html_(self):
        html_output = "<table><tbody>"

        # QCSResults Header
        html_output += '<tr><td colspan=2 style="text-align:center;"><strong>QCSResults</strong></td></tr>'
        html_output += "<tr><td colspan=2><hr></td></tr>"

        # Simulator Information
        html_output += '<tr><td colspan=2 style="text-align:center;"><strong>Simulator</strong></td></tr>'
        html_output += f'<tr><td colspan=2 style="text-align:center;">{self.simulator} {self.version}</td></tr>'
        html_output += "<tr><td colspan=2><hr></td></tr>"

        # Timings
        html_output += '<tr><td colspan=2 style="text-align:center;"><strong>Timings</strong></td></tr>'
        for key, value in _get_sorted_timings(self.timings):
            html_output += f'<tr><td style="text-align:left;">{key} time</td><td>{_format_duration(value)}</td></tr>'
        html_output += "<tr><td colspan=2><hr></td></tr>"

        # Fidelity Estimates
        if self.fidelities:
            from mimiqcircuits.backends.fidelity import as_lower_bound, _to_fidelity
            html_output += "<tr><td colspan=2></td></tr>"
            html_output += '<tr><td colspan=2 style="text-align:center;"><strong>Fidelity estimate</strong></td></tr>'
            if len(self.fidelities) == 1:
                fidelity_values = [
                    as_lower_bound(_to_fidelity(self.fidelities[0]))
                ]
                html_output += f'<tr><td style="text-align:left;">Single run value</td><td>{round(fidelity_values[0], 3)}</td></tr>'
            else:
                fidelity_values = [
                    as_lower_bound(_to_fidelity(f)) for f in self.fidelities
                ]
                html_output += f'<tr><td style="text-align:left;">Mean</td><td>{round(mean(fidelity_values), 3)}</td></tr>'
                html_output += f'<tr><td style="text-align:left;">Median</td><td>{round(median(fidelity_values), 3)}</td></tr>'
                html_output += f'<tr><td style="text-align:left;">Standard Deviation</td><td>{round(_display_stdev(fidelity_values), 3)}</td></tr>'
            html_output += "<tr><td colspan=2><hr></td></tr>"

        # Average Multiqubit Error Estimates
        if self.avggateerrors:
            html_output += "<tr><td colspan=2></td></tr>"
            html_output += '<tr><td colspan=2 style="text-align:center;"><strong>Average multiqubit error estimate</strong></td></tr>'
            if len(self.avggateerrors) == 1:
                html_output += f'<tr><td style="text-align:left;">Single run value</td><td>{round(self.avggateerrors[0], 3)}</td></tr>'
            else:
                html_output += f'<tr><td style="text-align:left;">Mean</td><td>{round(mean(self.avggateerrors), 3)}</td></tr>'
                html_output += f'<tr><td style="text-align:left;">Median</td><td>{round(median(self.avggateerrors), 3)}</td></tr>'
                html_output += f'<tr><td style="text-align:left;">Standard Deviation</td><td>{round(_display_stdev(self.avggateerrors), 3)}</td></tr>'
            html_output += "<tr><td colspan=2><hr></td></tr>"

        # Statistics
        html_output += "<tr><td colspan=2></td></tr>"
        html_output += '<tr><td colspan=2 style="text-align:center;"><strong>Statistics</strong></td></tr>'
        html_output += f'<tr><td style="text-align:left;">Number of executions</td><td>{len(self.fidelities)}</td></tr>'
        html_output += f'<tr><td style="text-align:left;">Number of samples</td><td>{len(self.cstates)}</td></tr>'
        html_output += f'<tr><td style="text-align:left;">Number of amplitudes</td><td>{len(self.amplitudes)}</td></tr>'
        html_output += "<tr><td colspan=2><hr></td></tr>"

        # Sampled Classical States
        if self.cstates:
            html_output += "<tr><td colspan=2></td></tr>"
            html_output += '<tr><td colspan=2 style="text-align:center;"><strong>Samples</strong></td></tr>'
            hist = self.histogram()
            outcomes = sorted(hist, key=hist.get, reverse=True)[0:10]
            for bs in outcomes:
                html_output += f'<tr><td style="text-align:left;font-family: monospace;">{hex(bs.tointeger())}</td><td style="text-align:left;font-family: monospace;">{bs.to01()}</td><td style="font-family: monospace;">{hist[bs]}</td></tr>'
            html_output += "<tr><td colspan=2><hr></td></tr>"

        # Statevector Amplitudes
        if self.amplitudes:
            html_output += "<tr><td colspan=2></td></tr>"
            html_output += '<tr><td colspan=2 style="text-align:center;"><strong>Statevector Amplitudes</strong></td></tr>'
            for bs, amp in self.amplitudes.items():
                html_output += f'<tr><td style="text-align:left;">{hex(bs.tointeger())}</td><td style="text-align:left;">{bs.to01()}</td><td>{amp:.3f}</td></tr>'
            html_output += "<tr><td colspan=2><hr></td></tr>"

        if len(self.zstates) > 0 and any(len(zstate) > 0 for zstate in self.zstates):
            html_output += "<tr><td colspan=2></td></tr>"
            html_output += '<tr><td colspan=2 style="text-align:center;"><strong>Z States</strong></td></tr>'
            z_hist = self.histzvars()
            for zstate, count in z_hist.items():
                html_output += f'<tr><td style="text-align:left;font-family: monospace;">[{", ".join(map(str, zstate))}]</td><td style="text-align:left;font-family: monospace;">{count}</td></tr>'
            html_output += "<tr><td colspan=2><hr></td></tr>"

        html_output += "</tbody></table>"

        return html_output

    def histogram(self):
        """Count how often each classical state was sampled.

        Returns:
            dict: A mapping from each sampled :class:`BitString` in
            ``cstates`` to its number of occurrences (``int``), in order of
            first appearance.

        Examples:
            >>> from mimiqcircuits import *
            >>> res = QCSResults(cstates=[BitString("01"), BitString("10"), BitString("01")])
            >>> res.histogram()
            {bs"01": 2, bs"10": 1}
        """
        hist = {}

        for cstate in self.cstates:
            if cstate in hist:
                hist[cstate] += 1
            else:
                hist[cstate] = 1

        return hist

    def histzvars(self):
        """Count how often each ``z`` register state was sampled.

        Returns:
            dict: A mapping from each state in ``zstates``, converted to a
            tuple, to its number of occurrences (``int``), in order of first
            appearance.

        Examples:
            >>> from mimiqcircuits import *
            >>> res = QCSResults(zstates=[[1.0], [0.5], [1.0]])
            >>> res.histzvars()
            {(1.0,): 2, (0.5,): 1}
        """
        hist = {}

        for zstate in self.zstates:
            i = tuple(zstate)  # Make it hashable
            if i in hist:
                hist[i] += 1
            else:
                hist[i] = 1

        return hist

    def saveproto(self, file):
        """Save the results to a protobuf file.

        Args:
            file (str or file-like): A path, or a binary file object with a
                ``write`` method.

        Returns:
            int: The number of bytes written.

        Raises:
            ValueError: If the file cannot be written.

        Examples:
            >>> from mimiqcircuits import *
            >>> import os, tempfile
            >>> res = QCSResults(
            ...     simulator="MIMIQ-StateVector",
            ...     version="1.0",
            ...     cstates=[BitString("01"), BitString("01"), BitString("10")],
            ... )

            Write the results to a file, then read them back with the static
            method :meth:`loadproto`.

            >>> path = os.path.join(tempfile.mkdtemp(), "results.pb")
            >>> nbytes = res.saveproto(path)
            >>> loaded = QCSResults.loadproto(path)
            >>> loaded.histogram() == res.histogram()
            True
        """
        from mimiqcircuits.proto.protoio import saveproto

        return saveproto(self, file)

    @staticmethod
    def loadproto(file):
        """Load results from a protobuf file written by :meth:`saveproto`.

        This is a static method: call it on the class.

        Args:
            file (str or file-like): A path, or a binary file object with a
                ``read`` method.

        Returns:
            QCSResults: The loaded results.

        Raises:
            ValueError: If the file cannot be read or parsed.

        See Also:
            :meth:`saveproto`, which has an example.
        """
        from mimiqcircuits.proto.protoio import loadproto

        return loadproto(file, QCSResults)


__all__ = ["QCSResults"]
