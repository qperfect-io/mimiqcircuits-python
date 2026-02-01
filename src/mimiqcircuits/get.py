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


CIRCUITS_MANIFEST = "circuits.json"

OPTIMIZE_MANIFEST = "optimize.json"

REQUEST_MANIFEST = "request.json"

import tempfile
import json
import os
from time import sleep
from mimiqcircuits.circuit import Circuit
from mimiqcircuits.qcsresults import QCSResults
from mimiqcircuits.__version__ import __version__
from mimiqcircuits.remote import RemoteConnection, TYPE_PROTO, TYPE_QASM, TYPE_STIM
import mimiqcircuits as mc


class QCSError:
    def __init__(self, error):
        self.error = error

    def __repr__(self):
        return self.error


def is_circuits_job(base_names):
    return CIRCUITS_MANIFEST in base_names


def is_optimization_job(base_names):
    return OPTIMIZE_MANIFEST in base_names


def is_validinputs(execution, base_names):
    has_req = REQUEST_MANIFEST in base_names
    has_circ = is_circuits_job(base_names)
    has_opt = is_optimization_job(base_names)

    if not has_req:
        raise RuntimeError(
            f"Execution {execution} is invalid: missing required '{REQUEST_MANIFEST}'. "
            f"Found: {base_names}"
        )

    if not has_circ and not has_opt:
        raise RuntimeError(
            f"Execution {execution} is invalid: expected either '{CIRCUITS_MANIFEST}' or '{OPTIMIZE_MANIFEST}'. "
            f"Found: {base_names}"
        )

    if has_circ and has_opt:
        raise RuntimeError(
            f"Execution {execution} is invalid: both '{CIRCUITS_MANIFEST}' and '{OPTIMIZE_MANIFEST}' present (ambiguous). "
            f"Found: {base_names}"
        )

    return True


def load_inputs(base_names, tmpdir):
    joblist = []

    if is_circuits_job(base_names):
        circuits_json = os.path.join(tmpdir, CIRCUITS_MANIFEST)
        with open(circuits_json, "r") as f:
            parameters = json.load(f)

        for cinfo in parameters["circuits"]:
            fpath = os.path.join(tmpdir, cinfo["file"])
            # None if absent
            ctype = cinfo.get("type")

            if ctype == TYPE_PROTO:
                joblist.append(Circuit.loadproto(fpath))
            elif ctype in (TYPE_STIM, TYPE_QASM):
                joblist.append(fpath)
            else:
                raise ValueError(
                    f"Unsupported circuit type '{ctype}' for file '{cinfo['file']}'."
                )

        return joblist, parameters

    if is_optimization_job(base_names):
        from mimiqcircuits.optimization import OptimizationExperiment

        optimize_json = os.path.join(tmpdir, OPTIMIZE_MANIFEST)
        with open(optimize_json, "r") as f:
            parameters = json.load(f)

        for einfo in parameters["experiments"]:
            fpath = os.path.join(tmpdir, einfo["file"])
            joblist.append(OptimizationExperiment.loadproto(fpath))

        return joblist, parameters

    # Should be unreachable
    raise RuntimeError(f"No recognized manifest. Found: {base_names}")


def get_results(self, execution, interval=1):
    """Retrieve the results of a completed execution.

    Args:
        execution (str): The execution identifier.
        interval (int): The interval (in seconds) for checking job status (default: 1).

    Returns:
        List[QCSResults | OptimizationRun | OptimizationResults]: A list of result instances.

    Raises:
        RuntimeError: If the remote job encounters an error.
    """
    # Wait for the job to finish
    while not self.connection.isJobDone(execution):
        sleep(interval)

    infos = self.connection.requestInfo(execution)

    if infos.status == "ERROR":
        error_message = infos.get("errorMessage", "Remote job errored.")
        raise RuntimeError(f"Remote job errored: {error_message}")
    elif infos.status == "CANCELED":
        raise RuntimeError("Remote job canceled.")

    with tempfile.TemporaryDirectory(prefix="mimiq_res_") as tmpdir:
        names = self.connection.downloadResults(execution, destdir=tmpdir)

        results_file_path = os.path.join(tmpdir, "results.json")

        if "results.json" not in names or not os.path.isfile(results_file_path):
            raise RuntimeError(f"No results found in execution {execution}.")

        with open(results_file_path, "r") as f:
            results_list = json.load(f)

        results = []
        for result in results_list:
            if "error" in result:
                results.append(QCSError(result["error"]))
            else:
                fname = os.path.join(tmpdir, result["file"])
                if not os.path.isfile(fname):
                    raise RuntimeError(f"Missing result file {fname}")
                if "optresult" in os.path.basename(fname):
                    from mimiqcircuits.optimization import (
                        OptimizationResults,
                        OptimizationRun,
                    )

                    if result.get("history", False):
                        results.append(OptimizationResults.loadproto(fname))
                    else:
                        results.append(OptimizationRun.loadproto(fname))
                else:
                    results.append(QCSResults.loadproto(fname))

    return results


def get_result(self, execution, **kwargs):
    """Retrieve the first result if multiple are found.

    Args:
        execution (str): The execution identifier.
        **kwargs: Additional keyword arguments for result retrieval.

    Returns:
        QCSResults | OptimizationRun | OptimizationResults: The first result found.

    Raises:
        RuntimeWarning: If multiple results are found.
    """
    results = self.get_results(execution, **kwargs)

    if len(results) > 1:
        print("Warning: Multiple results found. Returning the first one.")

    return results[0]


def get_inputs(self, execution):
    """Retrieve the inputs (circuits or experiments and parameters) of the execution.

    Args:
        execution (str): The execution identifier.

    Returns:
        tuple: A tuple containing a list of Circuit or OptimizationExperiment objects and parameters (dict).

    Raises:
        RuntimeError: If required files are not found in the inputs.
    """
    with tempfile.TemporaryDirectory(prefix="mimiq_in_") as tmpdir:
        names = self.connection.downloadJobFiles(execution, destdir=tmpdir)
        base_names = [os.path.basename(name) for name in names]

        print(f"Downloaded files: {base_names}")

        is_validinputs(execution, base_names)

        joblist, parameters = load_inputs(base_names, tmpdir)
        return joblist, parameters


def get_input(self, execution, **kwargs):
    """Retrieve the first circuit or experiment and parameters of the execution.

    Args:
        execution (str): The execution identifier.

    Returns:
        tuple: A tuple containing the first Circuit or OptimizationExperiment object and parameters (dict).

    Raises:
        RuntimeError: If required files are not found in the inputs.
    """
    circuits, parameters = self.get_inputs(execution, **kwargs)

    if len(circuits) > 1:
        print("Warning: Multiple results found. Returning the first one.")

    return circuits[0], parameters


# Register on RemoteConnection
RemoteConnection.get_input = get_input
RemoteConnection.get_inputs = get_inputs
RemoteConnection.get_result = get_result
RemoteConnection.get_results = get_results

__all__ = ["get_inputs", "get_input", "get_results", "get_result", "QCSError"]
