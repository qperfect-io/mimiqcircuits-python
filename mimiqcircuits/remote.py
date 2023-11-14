#
# Copyright © 2022-2023 University of Strasbourg. All Rights Reserved.
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

import mimiqlink
import hashlib
import tempfile
import json
import os
from mimiqcircuits.circuit import Circuit
from mimiqcircuits.qcsresults import QCSResults
from mimiqcircuits.__version__ import __version__
import numpy as np
from time import sleep

# maximum nbumber of samples allowed
MAX_SAMPLES = 2**16

# default value for the number of samples
DEFAULT_SAMPLES = 1000

# minimum and maximum bond dimension allowed
MIN_BONDDIM = 1
MAX_BONDDIM = 2**12

# default bond dimension
DEFAULT_BONDDIM = 256

# default time limit
DEFAULT_TIME_LIMIT = 5 * 60

# default algorithm
DEFAULT_ALGORITHM = "auto"

RESULTSPB_FILE = "results.pb"
CIRCUITPB_FILE = "circuit.pb"
CIRCUITQASM_FILE = "circuit.qasm"


def _hash_file(filename):
    """
    Calculate the SHA-256 hash of a file.

    Args:
        filename (str): The name of the file.

    Returns:
        str: The hexadecimal representation of the file's SHA-256 hash.
    """
    sha256_hash = hashlib.sha256()
    hash = ""

    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
        hash = sha256_hash.hexdigest()

    return hash


class MimiqConnection(mimiqlink.MimiqConnection):
    """Represents a connection to the Mimiq Server.

    Inherits from: mimiqlink.MimiqConnection python.
    """

    def execute(self, circuit, label="pyapi_v" + __version__,
                algorithm=DEFAULT_ALGORITHM, nsamples=DEFAULT_SAMPLES,
                bitstates=None, timelimit=DEFAULT_TIME_LIMIT, bonddim=None,
                seed=None):
        """
        Execute a circuit simulation using the MIMIQ cloud services.

        Args:
            circuit (Circuit): The quantum circuit to be executed.
            label (str): The label for the execution (default: "circuitsimu").
            algorithm (str): The algorithm to be used for execution (default: "auto").
            nsamples (int): The number of samples to generate (default: 1000).
            bitstates (list): List of bitstates for conditional execution (default: None).
            timelimit (int): The time limit for execution in seconds (default: 5 * 60).
            bonddim (int): The bond dimension for the MPS algorithm (default: None).
            seed (int): The seed for generating random numbers (default: randomly generated). If provided,
                uses the specified seed.

        Returns:
            str: The execution identifier.

        Raises:
            ValueError: If bonddim, nsamples, or timelimit exceeds the allowed limits.

        Examples:
            >>> from mimiqcircuits import *
            >>> conn=MimiqConnection()
            >>> conn.connect()
            >>> c=Circuit()
            >>> c.push(GateH(),1)
                2-qubit circuit with 1 instructions:
                 └── H @ q1

            >>> job=conn.execute(c,algorithm="auto")
            >>> res=conn.get_results(job)
            >>> res.samples
                {BitState('00'): 505, BitState('01'): 495}
        """
        if bitstates is None:
            bitstates = []
        else:
            nq = circuit.num_qubits()
            for b in bitstates:
                if len(b) != nq:
                    raise ValueError(
                        "The number of qubits in the bitstring is not equal to the number of qubits in the circuit.")

        # if seed is none default it to a random int64 seed
        if seed is None:
            seed = np.random.randint(0, 2**63 - 1)

        if (algorithm == "auto" or algorithm == "mps") and bonddim is None:
            bonddim = DEFAULT_BONDDIM

        if bonddim is not None and (bonddim < MIN_BONDDIM or bonddim > MAX_BONDDIM):
            raise ValueError(
                f"bonddim must be between {MIN_BONDDIM} and {MAX_BONDDIM}")

        if nsamples > MAX_SAMPLES:
            raise ValueError(f"nsamples must be less than {MAX_SAMPLES}")

        with tempfile.TemporaryDirectory() as tmpdir:
            # save the circuit in json format
            if isinstance(circuit, Circuit):
                circuit_filename = os.path.join(tmpdir, CIRCUITPB_FILE)
                circuit.saveproto(circuit_filename)
            elif isinstance(circuit, str):
                if not os.path.isfile(circuit):
                    raise FileNotFoundError(
                        f"File {circuit} not found.")

                circuit_filename = os.path.join(tmpdir, CIRCUITQASM_FILE)
                os.path.copy(circuit, circuit_filename)
            else:
                raise TypeError(
                    "circuit must be a Circuit object or a OPENQASM file")

            circuit_hash = _hash_file(circuit_filename)

            jsonbitstates = ['bs' + o.to01() for o in bitstates]

            pars = {"algorithm": algorithm, "bitstates": jsonbitstates,
                    "samples": nsamples, "seed": seed,
                    "apilang: ": "python", "apiversion": __version__,
                    "circuitsapiversion": __version__}

            if bonddim is not None:
                pars["bondDimension"] = bonddim

            req = {
                "executor": "Circuits",
                "timelimit": timelimit,
                "files": [
                    {
                        "name": os.path.basename(circuit_filename),
                        "hash": circuit_hash
                    }
                ],
                "parameters": pars
            }

            req_filename = os.path.join(tmpdir, "parameters.json")

            with open(req_filename, "w") as f:
                json.dump(req, f)

            emutype = "CIRC"
            return self.request(
                emutype,
                algorithm,
                label,
                timelimit,
                [req_filename, circuit_filename]
            )

    def get_results(self, execution, interval=10):
        """Retrieve the results of a completed execution.

        Args:
            execution (str): The execution identifier.
            interval (int): The interval (in seconds) for checking job status (default: 10).

        Returns:
            Results: An instance of the QCSResults class.

        Raises:
            RuntimeError: If the remote job encounters an error.

        Examples:
            >>> from mimiqcircuits import *
            >>> conn=MimiqConnection()
            >>> conn.connect()
            >>> c=Circuit()
            >>> c.push(GateH(),1)
                2-qubit circuit with 1 instructions:
                 └── H @ q1

            >>> job=conn.execute(c,algorithm="auto")
            >>> res=conn.get_results(job)
            >>> res.samples
                {BitState('00'): 504, BitState('01'): 496}
        """
        while not self.isJobDone(execution):
            sleep(interval)

        infos = self.requestInfo(execution)

        if infos['status'] == "ERROR":
            raise RuntimeError("Remote job errored.")

        with tempfile.TemporaryDirectory() as tmpdir:
            names = self.downloadResults(execution, destdir=tmpdir)

            if RESULTSPB_FILE not in names:
                raise RuntimeError(
                    "File not found in results. Update Your library.")

            results = QCSResults.loadproto(
                os.path.join(tmpdir, RESULTSPB_FILE))

        return results

    def get_inputs(self, execution):
        """Retrieve the inputs (circuit and parameters) of the execution.

        Args:
            execution (str): The execution identifier.

        Returns:
            dict: parameters (dict) of the execution.

        Raises:
            RuntimeError: If the required files are not found in  inputs. Update your library.

        Examples:
            >>> from mimiqcircuits import *
            >>> conn=MimiqConnection()
            >>> conn.connect()
            >>> c=Circuit()
            >>> c.push(GateH(),1)
                2-qubit circuit with 1 instructions:
                 └── H @ q1

            >>> job=conn.execute(c,algorithm="auto")
            >>> res=conn.get_results(job)
            >>> res.samples
                {BitState('00'): 504, BitState('01'): 496}

            >>> import json
            >>> c, parameters = conn.get_inputs(job)
            >>> print(json.dumps(parameters))
                {"executor": "Circuits", "timelimit": 300, "files": [{"name": "circuit.json", "hash": "4955f6540ec79fd1c34cfadc0dd0bd37b3d4f28bbce60208198f5d0bd7f486c7"}],
                "parameters": {"algorithm": "auto", "bitstates": [], "samples": 1000, "seed": 2875985032347405045, "bondDimension": 256}}
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            names = self.downloadJobFiles(execution, destdir=tmpdir)

            if CIRCUITPB_FILE not in names or CIRCUITQASM_FILE not in names:
                raise RuntimeError(
                    "File not found in inputs. Update Your library.")

            if "parameters.jl" not in names:
                raise RuntimeError(
                    "File not found in inputs. Update Your library.")

            with open(os.path.join(tmpdir, "parameters.jl"), "r") as f:
                parameters = json.load(f)

            if CIRCUITPB_FILE in names:
                circuit = Circuit.loadproto(
                    os.path.join(tmpdir, CIRCUITPB_FILE))
            elif CIRCUITQASM_FILE in names:
                circuit = os.path.join(tmpdir, CIRCUITQASM_FILE)
            else:
                raise FileNotFoundError(
                    "Circuit file not found within results. Update your library.")

        return circuit, parameters
