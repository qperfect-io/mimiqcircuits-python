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
from mimiqcircuits.remote import RemoteConnection, TYPE_PROTO, TYPE_QASM, TYPE_STIM


class QCSError:
    """The error reported for one circuit of a remote job that failed.

    :meth:`get_results` puts it in the list of results, in place of the
    result of the failed circuit. Its ``error`` attribute holds the message.
    """

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
    """Wait for a remote job to finish and return its results.

    Available as a method of the connection: ``conn.get_results(job)``. The
    call blocks, checking the job status every ``interval`` seconds, then
    downloads the results.

    Args:
        execution (str): The job identifier returned by ``submit``.
        interval (float): Seconds between status checks. Defaults to 1.

    Returns:
        list: One entry per circuit or experiment of the job, in order: a
        :class:`QCSResults`, an ``OptimizationResults`` for optimization jobs,
        or a :class:`QCSError` for a circuit that failed.

    Raises:
        RuntimeError: If the whole job failed or was canceled, or its result
            files are missing.

    Examples:
        >>> from mimiqcircuits import *
        >>> conn = MimiqConnection()  # doctest: +SKIP
        >>> conn.connect()  # doctest: +SKIP
        >>> c = Circuit().push(GateH(), 0).push(Measure(), 0, 0)
        >>> job = conn.submit(c, nsamples=100)  # doctest: +SKIP
        >>> results = conn.get_results(job)  # doctest: +SKIP
    """
    # Wait for the job to finish
    while not self.connection.isJobDone(execution):
        sleep(interval)

    infos = self.connection.requestInfo(execution)

    if infos.status == "ERROR":
        # The MIMIQ cloud reports the reason in errorMessage, Quantum Hive in messages.
        error_message = (
            infos.get("errorMessage", None)
            or ", ".join(msg["message"] for msg in infos.get("messages", None) or [])
            or "Remote job errored."
        )
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
                    from mimiqcircuits.optimization import OptimizationResults

                    results.append(OptimizationResults.loadproto(fname))
                else:
                    results.append(QCSResults.loadproto(fname))

    return results


def get_result(self, execution, **kwargs):
    """Wait for a remote job to finish and return its first result.

    Available as a method of the connection: ``conn.get_result(job)``. Use it
    for jobs with a single circuit; if the job has more, a warning is printed
    and only the first result is returned.

    Args:
        execution (str): The job identifier returned by ``submit``.
        **kwargs: Passed to :meth:`get_results`, such as ``interval``.

    Returns:
        QCSResults: The first result, or an ``OptimizationResults`` or a
        :class:`QCSError`, as for :meth:`get_results`.

    Raises:
        RuntimeError: As for :meth:`get_results`.
    """
    results = self.get_results(execution, **kwargs)

    if len(results) > 1:
        print("Warning: Multiple results found. Returning the first one.")

    return results[0]


def get_inputs(self, execution):
    """Download the inputs of a remote job: its circuits and its parameters.

    Available as a method of the connection: ``conn.get_inputs(job)``. The
    names of the downloaded files are printed.

    Args:
        execution (str): The job identifier returned by ``submit``.

    Returns:
        tuple: ``(inputs, parameters)``. ``inputs`` is a list with one
        :class:`Circuit` per circuit job entry (or the path of a downloaded
        file for circuits submitted as QASM or Stim; the files are left in a
        new temporary directory, which is not removed), or one
        ``OptimizationExperiment`` per experiment of an optimization job.
        ``parameters`` is the dict of job parameters.

    Raises:
        RuntimeError: If the job files are not a valid circuit or
            optimization job.
        ValueError: If a circuit file has an unsupported type.
    """
    # Not a TemporaryDirectory: QASM and Stim inputs are returned as paths
    # into this directory, so it must outlive the call.
    tmpdir = tempfile.mkdtemp(prefix="mimiq_in_")
    names = self.connection.downloadJobFiles(execution, destdir=tmpdir)
    base_names = [os.path.basename(name) for name in names]

    print(f"Downloaded files: {base_names}")

    is_validinputs(execution, base_names)

    return load_inputs(base_names, tmpdir)


def get_input(self, execution):
    """Download the inputs of a remote job and return the first one with the parameters.

    Available as a method of the connection: ``conn.get_input(job)``. If the
    job has more than one input, a warning is printed and only the first is
    returned.

    Args:
        execution (str): The job identifier returned by ``submit``.

    Returns:
        tuple: ``(input, parameters)``, with ``input`` the first entry of
        the list :meth:`get_inputs` returns.

    Raises:
        RuntimeError: As for :meth:`get_inputs`.
    """
    circuits, parameters = self.get_inputs(execution)

    if len(circuits) > 1:
        print("Warning: Multiple results found. Returning the first one.")

    return circuits[0], parameters


# Register on RemoteConnection
RemoteConnection.get_input = get_input
RemoteConnection.get_inputs = get_inputs
RemoteConnection.get_result = get_result
RemoteConnection.get_results = get_results

__all__ = ["get_inputs", "get_input", "get_results", "get_result", "QCSError"]
