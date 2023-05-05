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
MAX_TIME_LIMIT = 5 * 60


def _hash_file(filename):
    sha256_hash = hashlib.sha256()
    hash = ""

    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
        hash = sha256_hash.hexdigest()

    return hash


class Results:
    def __init__(self,
                 execution: str, results: dict,
                 samples: dict, amplitudes: dict):
        self.execution = execution
        self.results = results
        self.samples = samples
        self.amplitudes = amplitudes


class MimiqConnection(mimiqlink.MimiqConnection):
    def execute(self, circuit, label="circuitsimu",
                algorithm=DEFAULT_ALGORITHM, nsamples=DEFAULT_SAMPLES,
                bitstates=None, timelimit=DEFAULT_TIME_LIMIT, bonddim=None):
        if bitstates is None:
            bitstates = []

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
                json.dump(circuit.to_json(), f)

            circuit_hash = _hash_file(circuit_filename)

            jsonbitstates = ['bs' + o.to01() for o in bitstates]

            pars = {"algorithm": algorithm,
                    "bitstates": jsonbitstates, "samples": nsamples}

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
