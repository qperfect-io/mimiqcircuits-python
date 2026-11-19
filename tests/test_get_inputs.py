import json
import os
from types import SimpleNamespace

from mimiqcircuits.get import get_inputs


class _FakeConnection:
    def downloadJobFiles(self, execution, destdir):
        files = {
            "request.json": "{}",
            "circuits.json": json.dumps({"circuits": [{"file": "c.qasm", "type": "qasm"}]}),
            "c.qasm": "OPENQASM 2.0;\nqreg q[1];\n",
        }
        for name, text in files.items():
            with open(os.path.join(destdir, name), "w") as f:
                f.write(text)
        return list(files)


def test_get_inputs_qasm_path_survives():
    conn = SimpleNamespace(connection=_FakeConnection())
    inputs, _ = get_inputs(conn, "job")
    assert os.path.isfile(inputs[0])
    with open(inputs[0]) as f:
        assert f.read().startswith("OPENQASM 2.0;")
