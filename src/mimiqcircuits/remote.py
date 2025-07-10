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

import mimiqlink
import tempfile
import json
import os
import shutil
from mimiqcircuits.circuit import Circuit
from mimiqcircuits.qcsresults import QCSResults
from mimiqcircuits.__version__ import __version__
import numpy as np
from time import sleep

# maximum number of samples allowed
MAX_SAMPLES = 2**16

# default value for the number of samples
DEFAULT_SAMPLES = 1000

# minimum and maximum bond dimension allowed
MIN_BONDDIM = 1
MAX_BONDDIM = 2**12

# minimum and maximum entanglement dimension allowed
MIN_ENTDIM = 4
MAX_ENTDIM = 64

# default bond dimension
DEFAULT_BONDDIM = 256

# default entanglement dimension
DEFAULT_ENTDIM = 16

# default time limit
DEFAULT_TIME_LIMIT = 30

# default algorithm
DEFAULT_ALGORITHM = "auto"

RESULTSPB_FILE = "results.pb"

CIRCUIT_FNAME = "circuit"

EXTENSION_PROTO = "pb"
EXTENSION_QASM = "qasm"
EXTENSION_STIM = "stim"

TYPE_PROTO = "proto"
TYPE_QASM = "qasm"
TYPE_STIM = "stim"


def _file_is_openqasm2(file_path: str) -> bool:
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("//") or not line:
                continue
            return line.startswith("OPENQASM 2.0;")
    return False


def _file_may_be_stim(filepath: str) -> bool:
    STIM_KEYWORDS = {
        "I", "X", "Y", "Z", "C_XYZ", "C_ZYX", "H", "H_XY", "H_XZ", "H_YZ", "S",
        "SQRT_X", "SQRT_X_DAG", "SQRT_Y", "SQRT_Y_DAG", "SQRT_Z", "SQRT_Z_DAG", "S_DAG",
        "CNOT", "CX", "CXSWAP", "CY", "CZ", "CZSWAP", "ISWAP", "ISWAP_DAG",
        "SQRT_XX", "SQRT_XX_DAG", "SQRT_YY", "SQRT_YY_DAG", "SQRT_ZZ", "SQRT_ZZ_DAG",
        "SWAP", "SWAPCX", "SWAPCZ", "XCX", "XCY", "XCZ", "YCX", "YCY", "YCZ", "ZCX", "ZCY", "ZCZ",
        "CORRELATED_ERROR", "DEPOLARIZE1", "DEPOLARIZE2", "E", "ELSE_CORRELATED_ERROR",
        "HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1", "PAULI_CHANNEL_1", "PAULI_CHANNEL_2",
        "X_ERROR", "Y_ERROR", "Z_ERROR", "M", "MR", "MRX", "MRY", "MRZ", "MX", "MY", "MZ",
        "R", "RX", "RY", "RZ", "MXX", "MYY", "MZZ", "MPP", "SPP", "SPP_DAG",
        "REPEAT", "DETECTOR", "MPAD", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "TICK"
    }

    try:
        with open(filepath, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if line.startswith("#") or line == "":
                    continue
                first_word = line.split()[0] if line.split() else ""
                if first_word in STIM_KEYWORDS:
                    return True
    except FileNotFoundError:
        raise FileNotFoundError(f"File {filepath} not found.")
    return False


class QCSError:
    def __init__(self, error):
        self.error = error

    def __repr__(self):
        return self.error


class RemoteConnection:
    """Base class for connections to the Mimiq Server.

    This class provides common functionality for both MimiqConnection and PlanqkConnection.
    """

    def __init__(self, connection: mimiqlink.AbstractConnection):
        """Initialize a remote connection using a specific connection type.

        Args:
            connection: A mimiqlink connection object (MimiqConnection or PlanqkConnection)
        """
        self.connection = connection

    def __get_timelimit(self):
        """Fetch the maximum time limit for execution from the server."""
        # For MimiqConnection, get from user_limits
        if isinstance(self.connection, mimiqlink.MimiqConnection) and hasattr(
            self.connection, "user_limits"
        ):
            limits = self.connection.user_limits
            if limits and limits.get("enabledMaxTimeout"):
                return limits.get("maxTimeout", DEFAULT_TIME_LIMIT)

        return DEFAULT_TIME_LIMIT

    # Forward the attribute/method call to the connection object
    def __getattr__(self, name):
        original = getattr(self.connection, name)

        # It it is not callable, jsut return it
        if not callable(original):
            return original

        # If it is a method, we need to wrap the return value
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            if isinstance(result, type(self.connection)):
                return self
            return result

        return wrapped

    def execute(
        self,
        circuits,  # Can be a single Circuit object or a list of Circuit objects or QASM file paths
        label="pyapi_v" + __version__,
        algorithm=DEFAULT_ALGORITHM,
        nsamples=DEFAULT_SAMPLES,
        bitstrings=None,
        timelimit=None,
        bonddim=None,
        entdim=None,
        fuse=None,
        reorderqubits=None,
        seed=None,
        qasmincludes=None,
        force=False,
        **kwargs,
    ):
        """
        Execute a circuit or a list of quantum circuits on the Mimiq server.

        Args:
            circuits (Circuit or list of Circuits or str): A single Circuit object, a list of Circuit objects,
                                                           or QASM file paths representing the circuits to be executed.
            label (str): A label for the execution. Defaults to "pyapi_v" + __version__.
            algorithm (str): The algorithm to use. Defaults to "auto".
            nsamples (int): The number of samples to collect. Defaults to DEFAULT_SAMPLES.
            bitstrings (list of str, optional): Specific bitstrings to measure. Defaults to None.
            timelimit (int, optional): The maximum execution time in minutes. Defaults to None.
            bonddim (int, optional): The bond dimension to use. Defaults to None.
            entdim (int, optional): The entanglement dimension to use. Defaults to None.
            fuse (bool, optional): Whether to fuse gates. Defaults to None (let the remote service decide).
            reorderqubits (bool, optional): Whether to reorder qubits. Defaults to None (let the remote service decide).
            seed (int, optional): A seed for random number generation. Defaults to None.
            qasmincludes (list of str, optional): Additional QASM includes. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            object: A handle to the execution, typically used to retrieve results.

        Raises:
            ValueError: If nsamples exceeds MAX_SAMPLES, bond/entanglement dimensions are out of bounds,
                        or if a circuit contains unevaluated symbolic parameters.
            FileNotFoundError: If a QASM file is not found.
            TypeError: If the circuits argument is not a Circuit object or a valid file path.
        """

        if nsamples > MAX_SAMPLES:
            raise ValueError(f"nsamples must be less than {MAX_SAMPLES}")

        if timelimit is None:
            timelimit = self.__get_timelimit()

        maxtimelimit = self.__get_timelimit()
        if timelimit > maxtimelimit:
            raise ValueError(
                f"Timelimit cannot be set more than {maxtimelimit} minutes."
            )

        if bitstrings is None:
            bitstrings = []
        elif isinstance(circuits, Circuit):
            nq = circuits.num_qubits()
            for b in bitstrings:
                if len(b) != nq:
                    raise ValueError(
                        "The number of qubits in the bitstring is not equal to the number of qubits in the circuit."
                    )

        if seed is None:
            seed = int(np.random.randint(0, np.iinfo(np.int_).max, dtype=np.int_))

        # Set bond and entangling dimensions based on algorithm
        if algorithm == "auto" or algorithm == "mps":
            if bonddim is None:
                bonddim = DEFAULT_BONDDIM
            if entdim is None:
                entdim = DEFAULT_ENTDIM

        if bonddim is not None and (bonddim < MIN_BONDDIM or bonddim > MAX_BONDDIM):
            raise ValueError(f"bonddim must be between {MIN_BONDDIM} and {MAX_BONDDIM}")

        # Check for entdim constraint in specific algorithms
        if algorithm in ["mps", "auto"]:
            actual_entdim = DEFAULT_ENTDIM if entdim is None else entdim
            if actual_entdim < MIN_ENTDIM and not force:
                raise ValueError(
                    f"entdim must be between {MIN_ENTDIM} and {MAX_ENTDIM}"
                )
            elif actual_entdim < MIN_ENTDIM and force:
                print(
                    f"Warning: Running simulation with entdim={actual_entdim}. Results may be misleading."
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            allfiles = []
            circuit_files = []

            # Ensure circuits is a list
            if isinstance(circuits, (Circuit, str)):
                circuits = [circuits]

            if len(circuits) > 1 and algorithm == "auto":
                raise ValueError(
                    "The 'auto' algorithm is not supported in batch mode. Please specify 'mps' or 'statevector' for batch executions."
                )

            if not circuits:
                raise ValueError(
                    "The provided list of circuits is empty. At least one circuit is required."
                )

            for i, c in enumerate(circuits):
                if not isinstance(c, (Circuit, str)):
                    raise TypeError(
                        f"Invalid type at index {i}: expected Circuit or QASM file path, got {type(c).__name__}."
                    )

                if isinstance(c, Circuit) and len(c) == 0:
                    raise ValueError(
                        f"Empty Circuit object at index {i} is not allowed."
                    )

                if isinstance(c, str) and not os.path.isfile(c):
                    raise ValueError(
                        f"Invalid QASM file path at index {i}: {c} does not exist."
                    )

            for i, circuit in enumerate(circuits):
                if isinstance(circuit, Circuit) and circuit.is_symbolic():
                    raise ValueError(
                        "The circuit contains unevaluated symbolic parameters and cannot be processed until all parameters are fully evaluated."
                    )
                if isinstance(circuit, Circuit):
                    circuit_filename = os.path.join(
                        tmpdir, f"{CIRCUIT_FNAME}{i + 1}.{EXTENSION_PROTO}"
                    )
                    circuit.saveproto(circuit_filename)
                    circuit_files.append(
                        {
                            "file": os.path.basename(circuit_filename),
                            "type": TYPE_PROTO,
                        }
                    )
                    allfiles.append(circuit_filename)
                elif isinstance(circuit, str):
                    if not os.path.isfile(circuit):
                        raise FileNotFoundError(f"File {circuit} not found.")

                    # Case: QASM
                    if _file_is_openqasm2(circuit):
                        circuit_filename = os.path.join(tmpdir, f"{CIRCUIT_FNAME}{i+1}.{EXTENSION_QASM}")
                        shutil.copyfile(circuit, circuit_filename)
                        if qasmincludes is None:
                            qasmincludes = []
                        circuit_files.append({
                            "file": os.path.basename(circuit_filename),
                            "type": TYPE_QASM
                        })
                        allfiles.append(circuit_filename)

                    # Case: STIM
                    elif _file_may_be_stim(circuit):
                        circuit_filename = os.path.join(tmpdir, f"{CIRCUIT_FNAME}{i+1}.{EXTENSION_STIM}")
                        shutil.copyfile(circuit, circuit_filename)
                        circuit_files.append({
                            "file": os.path.basename(circuit_filename),
                            "type": TYPE_STIM
                        })
                        allfiles.append(circuit_filename)

                    # Unknown type
                    else:
                        raise ValueError(
                            f"File {circuit} is neither a valid OpenQASM 2.0 file nor a recognizable STIM file."
                        )
                else:
                    raise TypeError(
                        "circuits must be Circuit objects or paths to QASM or STIM files"
                    )

            jsonbitstrings = ["bs" + o.to01() for o in bitstrings]

            pars = {
                "algorithm": algorithm,
                "bitstrings": jsonbitstrings,
                "samples": nsamples,
                "seed": seed,
                "circuits": circuit_files,
            }

            if bonddim is not None:
                pars["bondDimension"] = bonddim

            if entdim is not None:
                pars["entDimension"] = entdim

            if fuse is not None:
                pars["fuse"] = fuse

            if reorderqubits is not None:
                pars["reorderQubits"] = reorderqubits

            # Add any additional keyword arguments
            pars.update(kwargs)

            # Save the parameters to a JSON file
            pars_filename = os.path.join(tmpdir, "circuits.json")
            with open(pars_filename, "w") as f:
                json.dump(pars, f)

            # Prepare the request
            req = {
                "executor": "Circuits",
                "timelimit": timelimit,
                "apilang": "python",
                "apiversion": __version__,
                "circuitsapiversion": __version__,
            }

            reqfile = os.path.join(tmpdir, "request.json")
            with open(reqfile, "w") as f:
                json.dump(req, f)

            # Make the request to the server
            emutype = "CIRC"

            sleep(0.1)
            return self.connection.request(
                emutype,
                algorithm,
                label,
                timelimit,
                [reqfile, pars_filename] + allfiles,
            )

    def get_results(self, execution, interval=1):
        """Retrieve the results of a completed execution.

        Args:
            execution (str): The execution identifier.
            interval (int): The interval (in seconds) for checking job status (default: 1).

        Returns:
            List[QCSResults]: A list of QCSResults instances.

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
                    results.append(QCSResults.loadproto(fname))

        return results

    def get_result(self, execution, **kwargs):
        """Retrieve the first result if multiple are found.

        Args:
            execution (str): The execution identifier.
            **kwargs: Additional keyword arguments for result retrieval.

        Returns:
            QCSResults: The first result found.

        Raises:
            RuntimeWarning: If multiple results are found.
        """
        results = self.get_results(execution, **kwargs)

        if len(results) > 1:
            print("Warning: Multiple results found. Returning the first one.")

        return results[0]

    def get_inputs(self, execution):
        """Retrieve the inputs (circuits and parameters) of the execution.

        Args:
            execution (str): The execution identifier.

        Returns:
            tuple: A tuple containing a list of Circuit objects and parameters (dict).

        Raises:
            RuntimeError: If required files are not found in the inputs.
        """
        with tempfile.TemporaryDirectory(prefix="mimiq_in_") as tmpdir:
            names = self.connection.downloadJobFiles(execution, destdir=tmpdir)

            # Print the downloaded files for debugging purposes
            print(f"Downloaded files: {names}")

            # Get the base names of the downloaded files
            base_names = [os.path.basename(name) for name in names]

            # Check if the circuits.json and request.json files are present
            if "circuits.json" not in base_names or "request.json" not in base_names:
                raise RuntimeError(
                    f"{execution} is not a valid execution for MimiqCircuits: missing necessary files"
                )

            # Load the parameters from the circuits.json file
            circuits_file_path = os.path.join(tmpdir, "circuits.json")
            with open(circuits_file_path, "r") as f:
                parameters = json.load(f)

            circuits = []
            for c in parameters["circuits"]:
                if c["type"] == TYPE_PROTO:
                    circuit = Circuit.loadproto(os.path.join(tmpdir, c["file"]))
                    circuits.append(circuit)
                else:
                    # case of STIM and QASM files
                    circuits.append(os.path.join(tmpdir, c["file"]))

            if len(circuits) == 0:
                raise RuntimeError(
                    "No valid circuit files found. Input parameters not valid."
                )

        return circuits, parameters

    def get_input(self, execution, **kwargs):
        """Retrieve the first circuit and parameters of the execution.

        Args:
            execution (str): The execution identifier.

        Returns:
            tuple: A tuple containing the first Circuit object and parameters (dict).

        Raises:
            RuntimeError: If required files are not found in the inputs.
        """
        circuits, parameters = self.get_inputs(execution, **kwargs)

        if len(circuits) > 1:
            print("Warning: Multiple results found. Returning the first one.")

        return circuits[0], parameters

    def __repr__(self):
        """Return a string representation of the connection."""
        return self.connection.__repr__()

    def __str__(self):
        """Return a string representation of the connection."""
        return self.connection.__str__()


class MimiqConnection(RemoteConnection):
    """Represents a connection to the Mimiq Server via direct cloud connection.

    This is a wrapper around mimiqlink.MimiqConnection to provide the circuit execution API.
    """

    def __init__(self, url=None):
        """Initialize a MimiqConnection.

        Args:
            url (str, optional): The URL of the Mimiq server. Defaults to None (using default cloud URL).
        """
        connection = mimiqlink.MimiqConnection(url)
        super().__init__(connection)


class PlanqkConnection(RemoteConnection):
    """Represents a connection to the Mimiq Server via PlanQK.

    This is a wrapper around mimiqlink.PlanqkConnection to provide the circuit execution API.
    """

    def __init__(self, url=None, consumer_key=None, consumer_secret=None):
        """Initialize a PlanqkConnection.

        Args:
            url (str, optional): The URL of the PlanQK API. Defaults to None (using default PlanQK URL).
            consumer_key (str, optional): The consumer key for PlanQK authentication. Defaults to None.
            consumer_secret (str, optional): The consumer secret for PlanQK authentication. Defaults to None.
        """
        connection = mimiqlink.PlanqkConnection(url, consumer_key, consumer_secret)
        super().__init__(connection)


__all__ = ["MimiqConnection", "PlanqkConnection", "RemoteConnection"]
