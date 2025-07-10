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
from mimiqcircuits.proto.qcsrproto import toproto_qcsr, fromproto_qcsr

from mimiqcircuits.bitstrings import bitvec_to_int
from statistics import mean, median, stdev


class QCSResults:
    """
    Represents the results of quantum computations obtained from quantum cloud services (QCS).

    Args:
        simulator (str): The name of the quantum simulator.
        version (str): The version of the quantum simulator.
        fidelities (list): List of fidelity estimates from different executions.
        avggateerrors (list): List of average multi-qubit gate errors from different executions.
        cstates (list): List of classical states obtained from executions.
        zstates (list): Not used in the current implementation.
        amplitudes (dict): Dictionary of statevector amplitudes for different quantum states.
        timings (dict): Dictionary of timing information for different phases of the computation.

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

        # filter out the ones < 1e-7
        timings = {k: v for k, v in self.timings.items() if v > 1e-7}

        result_str += "├── timings:\n"
        for key, value in list(timings.items())[:-1]:
            result_str += f"│    ├── {key} time: {value}s\n"

        key, value = list(timings.items())[-1]
        result_str += f"│    └── {key} time: {value}s\n"

        if len(self.fidelities) == 1:
            result_str += f"├── fidelity estimate: {self.fidelities[0]:.3g}\n"
            result_str += f"├── average multi-qubit gate error estimate: {self.avggateerrors[0]:.3g}\n"
        elif len(self.fidelities) != 0:
            result_str += "├── fidelity estimate:\n"
            result_str += f"│    ├── min, max: {min(self.fidelities):.3g}, {max(self.fidelities):.3g}\n"
            result_str += f"│    ├── mean: {mean(self.fidelities):.3g}\n"
            result_str += f"│    ├── median: {median(self.fidelities):.3g}\n"
            result_str += f"│    └── std: {stdev(self.fidelities):.3g}\n"
            result_str += "├── average multi-qubit gate error estimate:\n"
            result_str += f"│    ├── min, max: {min(self.avggateerrors):.3g}, {max(self.avggateerrors):.3g}\n"
            result_str += f"│    ├── mean: {mean(self.avggateerrors):.3g}\n"
            result_str += f"│    ├── median: {median(self.avggateerrors):.3g}\n"
            result_str += f"│    └── std: {stdev(self.avggateerrors):.3g}\n"

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
        for key, value in self.timings.items():
            html_output += f'<tr><td style="text-align:left;">{key} time</td><td>{value}s</td></tr>'
        html_output += "<tr><td colspan=2><hr></td></tr>"

        # Fidelity Estimates
        if self.fidelities:
            html_output += "<tr><td colspan=2></td></tr>"
            html_output += '<tr><td colspan=2 style="text-align:center;"><strong>Fidelity estimate</strong></td></tr>'
            if len(self.fidelities) == 1:
                html_output += f'<tr><td style="text-align:left;">Single run value</td><td>{round(self.fidelities[0], 3)}</td></tr>'
            else:
                html_output += f'<tr><td style="text-align:left;">Mean</td><td>{round(mean(self.fidelities), 3)}</td></tr>'
                html_output += f'<tr><td style="text-align:left;">Median</td><td>{round(median(self.fidelities), 3)}</td></tr>'
                html_output += f'<tr><td style="text-align:left;">Standard Deviation</td><td>{round(stdev(self.fidelities), 3)}</td></tr>'
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
                html_output += f'<tr><td style="text-align:left;">Standard Deviation</td><td>{round(stdev(self.avggateerrors), 3)}</td></tr>'
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
                html_output += f'<tr><td style="text-align:left;font-family: monospace;">{hex(bitvec_to_int(bs))}</td><td style="text-align:left;font-family: monospace;">{bs.to01()}</td><td style="font-family: monospace;">{hist[bs]}</td></tr>'
            html_output += "<tr><td colspan=2><hr></td></tr>"

        # Statevector Amplitudes
        if self.amplitudes:
            html_output += "<tr><td colspan=2></td></tr>"
            html_output += '<tr><td colspan=2 style="text-align:center;"><strong>Statevector Amplitudes</strong></td></tr>'
            for bs, amp in self.amplitudes.items():
                html_output += f'<tr><td style="text-align:left;">{hex(bitvec_to_int(bs.bits))}</td><td style="text-align:left;">{bs.to01()}</td><td>{amp:.3f}</td></tr>'
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
        """Histogram of the obtained classical states' occurrences.

        Returns:
            A dictionary of classical states (bitarray) and their occurrences
            (float).

        Raises:
            TypeError: If a non QCSResults object is passed.
        """
        hist = {}

        for cstate in self.cstates:
            if cstate in hist:
                hist[cstate] += 1
            else:
                hist[cstate] = 1

        return hist

    def histzvars(self):
        """
        Histogram of the obtained zstates' occurrences.
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
        """Save QCSResults object to a Protocol Buffers file.

        Examples:

            >>> from mimiqcircuits import *
            >>> from symengine import *
            >>> import tempfile
            >>> x, y = symbols("x y")
            >>> c = Circuit()
            >>> c.push(GateH(), 0)
            1-qubit circuit with 1 instructions:
            └── H @ q[0]
            <BLANKLINE>
            >>> conn = MimiqConnection()
            >>> conn.connect()
            Connection:
            ├── url: https://mimiq.qperfect.io/api
            ├── Computing time: 597/10000 minutes
            ├── Executions: 451/10000
            ├── Max time limit per request: 180 minutes
            └── status: open
            <BLANKLINE>
            >>> job = conn.execute(c)
            >>> res = conn.get_result(job)
            >>> res
            QCSResults:
            ├── simulator: MIMIQ-StateVector 0.18.0
            ├── timings:
            │    ├── parse time: 5.9552e-05s
            │    ├── apply time: 1.6682e-05s
            │    ├── total time: 0.00019081399999999998s
            │    ├── compression time: 4.575e-06s
            │    └── sample time: 5.299e-05s
            ├── fidelity estimate: 1
            ├── average multi-qubit gate error estimate: 0
            ├── most sampled:
            │    ├── bs"1" => 513
            │    └── bs"0" => 487
            ├── 1 executions
            ├── 0 amplitudes
            └── 1000 samples

            >>> tmpfile = tempfile.NamedTemporaryFile(suffix=".pb", delete=True)
            >>> res.saveproto(tmpfile.name)
            7169
            >>> res.loadproto(tmpfile.name)
            QCSResults:
            ├── simulator: MIMIQ-StateVector 0.18.0
            ├── timings:
            │    ├── parse time: 5.9552e-05s
            │    ├── apply time: 1.6682e-05s
            │    ├── total time: 0.00019081399999999998s
            │    ├── compression time: 4.575e-06s
            │    └── sample time: 5.299e-05s
            ├── fidelity estimate: 1
            ├── average multi-qubit gate error estimate: 0
            ├── most sampled:
            │    ├── bs"1" => 513
            │    └── bs"0" => 487
            ├── 1 executions
            ├── 0 amplitudes
            └── 1000 samples

            Note:
                This example uses a temporary file to demonstrate the save and load functionality.
                You can save your file with any name at any location using:

                .. code-block:: python

                    res.saveproto("example.pb")
                    res.loadproto("example.pb")
        """
        if isinstance(file, str):
            with open(file, "wb") as f:
                return f.write(toproto_qcsr(self).SerializeToString())
        elif hasattr(file, "write"):
            return file.write(toproto_qcsr(self).SerializeToString())

    @staticmethod
    def loadproto(file):
        """Load QCSResults object from a Protocol Buffers file.

        The :func:`loadproto` method is a static method and should be called on the class,
        not on an instance of the class.

        Note:

            Look for example in :func:`QCSResults.saveproto`
        """
        from mimiqcircuits.proto import qcsresults_pb2

        qcs_results_proto = qcsresults_pb2.QCSResults()

        if isinstance(file, str):
            with open(file, "rb") as f:
                qcs_results_proto.ParseFromString(f.read())
        elif hasattr(file, "read"):
            qcs_results_proto.ParseFromString(file.read())

        return fromproto_qcsr(qcs_results_proto)


__all__ = ["QCSResults"]
