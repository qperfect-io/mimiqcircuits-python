from mimiqcircuits.proto.qcsrproto import toproto_qcsr, fromproto_qcsr
from mimiqcircuits.proto import qcsresults_pb
from mimiqcircuits.bitstates import bitvec_to_int


class QCSResults:
    def __init__(self, simulator=None, version=None, fidelities=None, avggateerrors=None, cstates=None, zstates=None, amplitudes=None, timings=None):
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
        for key, value in self.timings.items():
            result_str += f"├── {key} time: {value}s\n"
        result_str += f"├── fidelity estimate (min,max): ({min(self.fidelities):.3f}, {max(self.fidelities):.3f})\n"
        result_str += f"├── average ≥2-qubit gate error (min,max): ({min(self.avggateerrors):.3f}, {max(self.avggateerrors):.3f})\n"
        result_str += f"├── {len(self.fidelities)} executions\n"
        result_str += f"├── {len(self.amplitudes)} amplitudes\n"
        result_str += f"└── {len(self.cstates)} samples"
        return result_str

    def _repr_html_(self):
        html_output = "<h3>QCSResults</h3>"

        html_output += "<h4>Simulator</h4>"
        html_output += "<table>"
        html_output += f"<tr><td>{self.simulator} {self.version}</td></tr>"
        html_output += "</table>"

        html_output += "<h4>Timings</h4>"
        html_output += "<table>"
        for key, value in self.timings.items():
            html_output += f"<tr><td>{key} time</td><td>{value}s</td></tr>"
        html_output += "</table>"

        html_output += "<h4>Accuracy</h4>"
        html_output += "<table>"
        html_output += "<tr><td>Estimate</td><td>minimum</td><td>maximum</td></tr>"
        html_output += f"<tr><td>Fidelity</td><td>{min(self.fidelities):.3f}</td><td>{max(self.fidelities):.3f}</td></tr>"
        html_output += f"<tr><td>Average (>=2)-qubit gate error</td><td>{min(self.avggateerrors):.3f}</td><td>{max(self.avggateerrors):.3f}</td></tr>"
        html_output += "</table>"

        html_output += "<h4>Statistics</h4>"
        html_output += "<table>"
        html_output += f"<tr><td>Executions</td><td>{len(self.fidelities)}</td></tr>"
        html_output += f"<tr><td>Amplitudes</td><td>{len(self.amplitudes)}</td></tr>"
        html_output += f"<tr><td>Samples</td><td>{len(self.cstates)}</td></tr>"
        html_output += "</table>"

        if len(self.cstates) > 0:
            hist = self.histogram()
            outcomes = sorted(hist, key=hist.get, reverse=True)[0:10]

            html_output += "<h4>Sampled Classical States</h4>"
            html_output += "<table>"
            html_output += "<tr><td>Hex</td><td>Bitstring</td><td>Counts</td>"
            for bs in outcomes:
                html_output += f"<tr><td>{hex(bitvec_to_int(bs))}</td><td>{bs.to01()}</td><td>{hist[bs]}</td></tr>"
            html_output += "</table>"

        if len(self.amplitudes) > 0:
            html_output += "<h4>Statevector Amplitudes</h4>"
            html_output += "<table>"
            html_output += "<tr><td>Hex</td><td>Bitstring</td><td>Amplitude</td>"
            for bs, amp in self.amplitudes.items():
                html_output += f"<tr><td>{hex(bitvec_to_int(bs.bits))}</td><td>{bs.to01()}</td><td>{amp:.3f}</td></tr>"
            html_output += "</table>"

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

    def saveproto(self, filename):
        with open(filename, "wb") as f:
            return f.write(toproto_qcsr(self).SerializeToString())

    @staticmethod
    def loadproto(filename):
        qcs_results_proto = qcsresults_pb.QCSResults()
        with open(filename, "rb") as f:
            qcs_results_proto.ParseFromString(f.read())
        return fromproto_qcsr(qcs_results_proto)


__all__ = ['QCSResults']
