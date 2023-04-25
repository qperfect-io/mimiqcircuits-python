#
# Copyright © 2023 University of Strasbourg. All Rights Reserved.
# See AUTHORS.md for the list of authors.
#

import mimiqlink
import hashlib
import tempfile
import json
import os
import bson
from mimiqcircuits.circuit import Circuit

from time import sleep


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
                algorithm="auto", nsamples=1000,
                bitstates=None, timelimit=5 * 60, bonddim=None):
        if bitstates is None:
            bitstates = []

        if (algorithm == "auto" or algorithm == "mps") and bonddim is None:
            bonddim = 256

        if bonddim < 1 or bonddim > 2**12:
            raise ValueError("bonddim must be between 1 and 4096")

        if nsamples > 2**16:
            raise ValueError("nsamples must be less than 2^16")

        if timelimit > 30 * 60:
            raise ValueError("timelimit must be less than 30 minutes")

        with tempfile.TemporaryDirectory() as tmpdir:
            # save the circuit in json format
            circuit_filename = os.path.join(tmpdir, "circuit.json")
            with open(circuit_filename, "w") as f:
                json.dump(circuit.to_json(), f)

            circuit_hash = _hash_file(circuit_filename)

            pars = {"algorithm": algorithm,
                    "bitstates": bitstates, "samples": nsamples}

            if bonddim is not None:
                pars["bonddim"] = bonddim

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

            return self.request(algorithm, label, [req_filename, circuit_filename])

    def get_results(self, execution, interval=10):
        while not self.isJobDone(execution):
            sleep(interval)

        infos = self.requestInfo(execution)

        if infos['status'] == "ERROR":
            raise RuntimeError("Remote job errored.")

        with tempfile.TemporaryDirectory() as tmpdir:
            names = self.downloadResults(execution, destdir=tmpdir)

            if "results.json" not in names:
                raise RuntimeError(
                    "File not found in results. Update Your library.")

            if "samples.bson" not in names:
                raise RuntimeError(
                    "File not found in results. Update your library.")

            if "amplitudes.bson" not in names:
                raise RuntimeError(
                    "File not found in results. Update your library.")

            with open(os.path.join(tmpdir, "results.json"), "r") as f:
                results = json.load(f)

            with open(os.path.join(tmpdir, "samples.bson"), "rb") as f:
                # PERF: this can cause problem. there is an alternative
                # which is
                # samples_generator = bons.decode_file_ter(f)
                # which returns a generator
                samples = bson.decode_all(f.read())

            with open(os.path.join(tmpdir, "amplitudes.bson"), "rb") as f:
                # PERF: same as samples.bson
                amplitudes = bson.decode_all(f.read())

        return Results(execution, results, samples, amplitudes)

    def get_inputs(self, execution):
        with tempfile.TemporaryDirectory() as tmpdir:
            names = self.downloadJobFiles(execution, destdir=tmpdir)

            if "parameters.json" not in names:
                raise RuntimeError(
                    "File not found in inputs. Update Your library.")

            if "parameters.json" not in names:
                raise RuntimeError(
                    "File not found in inputs. Update Your library.")

            with open(os.path.join(tmpdir, "parameters.json"), "r") as f:
                parameters = json.load(f)

            with open(os.path.join(tmpdir, "circuit.json"), "r") as f:
                circuit = Circuit.from_json(json.load(f))

        return circuit, parameters


__all__ = ["MimiqConnection", "Results"]
