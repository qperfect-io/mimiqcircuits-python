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
from mimiqcircuits.qcsr import QcsrFile
from mimiqcircuits.bitstates import BitState
import numpy as np
from time import sleep

RESULTS_SAMPLES_FILENAME = "samples.qcsr"
RESULTS_AMPLITUES_FILENAME = "amplitudes.qcsr"
RESULTS_PARAMETERS_FILENAME = "results.json"

REQUEST_CIRCUIT_FILENAME = "circuit.json"
REQUEST_PARAMETERS_FILENAME = "parameters.json"

DEFAULT_ALGORITHM = "auto"
DEFAULT_BONDDIM = 256
DEFAULT_SAMPLES = 1000
DEFAULT_TIME_LIMIT = 5 * 60

MAX_SAMPLES = 2**16
MIN_BONDDIM = 1
MAX_BONDDIM = 2**12
MAX_TIME_LIMIT = 30 * 60


class AllEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(AllEncoder, self).default(obj)


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


class Results:
    """
    Represents the results of a execution.

    Attributes:
        execution (str): The execution identifier.
        results (dict): A dictionary containing the result parameters.
        samples (dict): A dictionary containing the sample results.
        amplitudes (dict): A dictionary containing the amplitude results.
    """

    def __init__(self,
                 execution: str, results: dict,
                 samples: dict, amplitudes: dict):
        self.execution = execution
        self.results = results
        self.samples = samples
        self.amplitudes = amplitudes


class MimiqConnection(mimiqlink.MimiqConnection):
    """
    Represents a connection to the Mimiq Server.

    Inherits from: mimiqlink.MimiqConnection python.

    """

    def execute(self, circuit, label="circuitsimu",
                algorithm=DEFAULT_ALGORITHM, nsamples=DEFAULT_SAMPLES,
                bitstates=None, timelimit=DEFAULT_TIME_LIMIT, bonddim=None, seed=None):
        """
        Execute a circuit simulation using the Mimiq Server.

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

        **Example of the usage:**

        >>> from mimiqcircuits import *

        >>> conn=MimiqConnection()
        >>> conn.connect()
        >>> c=Circuit()
        >>> c.add_gate(GateH(),1)
        >>> c

        >>> 2-qubit circuit with 1 gates
            └── H @ q1

        >>> job=conn.execute(c,algorithm="auto")

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

        if timelimit > MAX_TIME_LIMIT:
            raise ValueError(
                f"timelimit must be less than {MAX_TIME_LIMIT} seconds ({MAX_TIME_LIMIT / 60} miutes)"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            # save the circuit in json format
            circuit_filename = os.path.join(tmpdir, REQUEST_CIRCUIT_FILENAME)
            with open(circuit_filename, "w") as f:
                json.dump(circuit.to_json(), f, cls=AllEncoder)

            circuit_hash = _hash_file(circuit_filename)

            jsonbitstates = ['bs' + o.to01() for o in bitstates]

            pars = {"algorithm": algorithm, "bitstates": jsonbitstates,
                    "samples": nsamples, "seed": seed}

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

            req_filename = os.path.join(tmpdir, REQUEST_PARAMETERS_FILENAME)

            with open(req_filename, "w") as f:
                json.dump(req, f)

            return self.request(
                algorithm,
                label,
                [req_filename, circuit_filename]
            )

    def get_results(self, execution, interval=10):
        """
        Retrieve the results of a completed execution.

        Args:
            execution (str): The execution identifier.
            interval (int): The interval (in seconds) for checking job status (default: 10).

        Returns:
            Results: An instance of the Results class containing the execution, results, samples, amplitudes.

        Raises:
            RuntimeError: If the remote job encounters an error.

        **Example of the usage:**

        >>> from mimiqcircuits import *

        >>> conn=MimiqConnection()
        >>> conn.connect()
        >>> c=Circuit()
        >>> c.add_gate(GateH(),1)
        >>> c

        >>> 2-qubit circuit with 1 gates
            └── H @ q1

        >>> job=conn.execute(c,algorithm="auto")
        >>> res=conn.get_results(job)
        >>> res.samples

        >>> {BitState('00'): 504, BitState('01'): 496}

        """
        while not self.isJobDone(execution):
            sleep(interval)

        infos = self.requestInfo(execution)

        if infos['status'] == "ERROR":
            raise RuntimeError("Remote job errored.")

        with tempfile.TemporaryDirectory() as tmpdir:
            names = self.downloadResults(execution, destdir=tmpdir)

            if RESULTS_PARAMETERS_FILENAME not in names:
                raise RuntimeError(
                    "File not found in results. Update Your library.")

            if RESULTS_SAMPLES_FILENAME not in names:
                raise RuntimeError(
                    "File not found in results. Update your library.")

            if RESULTS_AMPLITUES_FILENAME not in names:
                raise RuntimeError(
                    "File not found in results. Update your library.")

            with open(os.path.join(tmpdir, RESULTS_PARAMETERS_FILENAME), "r") as f:
                results = json.load(f)

            with QcsrFile(os.path.join(tmpdir, RESULTS_SAMPLES_FILENAME), "rb") as f:
                samples = {}
                for x in f.read():
                    samples[BitState(x[0])] = x[1]

            with QcsrFile(os.path.join(tmpdir, RESULTS_AMPLITUES_FILENAME), "rb") as f:
                amplitudes = {}
                for x in f.read():
                    amplitudes[BitState(x[0])] = x[1]

        return Results(execution, results, samples, amplitudes)

    def get_inputs(self, execution):
        """
        Retrieve the inputs (circuit and parameters) of the execution.

        Args:
            execution (str): The execution identifier.

        Returns:
            dict: parameters (dict) of the execution.

        Raises:
            RuntimeError: If the required files are not found in  inputs. Update your library.

        **Example of usage:**

        >>> from mimiqcircuits import *
        >>> conn=MimiqConnection()
        >>> conn.connect()
        >>> c=Circuit()
        >>> c.add_gate(GateH(),1)
        >>> c

        >>> 2-qubit circuit with 1 gates
            └── H @ q1

        >>> job=conn.execute(c,algorithm="auto")
        >>> res=conn.get_results(job) 
        >>> res.samples

        >>> {BitState('00'): 504, BitState('01'): 496}

        >>> import json
        >>> c, parameters = conn.get_inputs(job)
        >>> print(json.dumps(parameters))

        >>> {"executor": "Circuits", "timelimit": 300, "files": [{"name": "circuit.json", "hash": "4955f6540ec79fd1c34cfadc0dd0bd37b3d4f28bbce60208198f5d0bd7f486c7"}],
             "parameters": {"algorithm": "auto", "bitstates": [], "samples": 1000, "seed": 2875985032347405045, "bondDimension": 256}}

        """
        with tempfile.TemporaryDirectory() as tmpdir:
            names = self.downloadJobFiles(execution, destdir=tmpdir)

            if REQUEST_CIRCUIT_FILENAME not in names:
                raise RuntimeError(
                    "File not found in inputs. Update Your library.")

            if REQUEST_PARAMETERS_FILENAME not in names:
                raise RuntimeError(
                    "File not found in inputs. Update Your library.")

            with open(os.path.join(tmpdir, REQUEST_PARAMETERS_FILENAME), "r") as f:
                parameters = json.load(f)

            with open(os.path.join(tmpdir, REQUEST_CIRCUIT_FILENAME), "r") as f:
                circuit = Circuit.from_json(json.load(f))

        return circuit, parameters


__all__ = ["MimiqConnection", "Results"]
