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
"""Remote connection and execution utilities."""

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
        "I",
        "X",
        "Y",
        "Z",
        "C_XYZ",
        "C_ZYX",
        "H",
        "H_XY",
        "H_XZ",
        "H_YZ",
        "S",
        "SQRT_X",
        "SQRT_X_DAG",
        "SQRT_Y",
        "SQRT_Y_DAG",
        "SQRT_Z",
        "SQRT_Z_DAG",
        "S_DAG",
        "CNOT",
        "CX",
        "CXSWAP",
        "CY",
        "CZ",
        "CZSWAP",
        "ISWAP",
        "ISWAP_DAG",
        "SQRT_XX",
        "SQRT_XX_DAG",
        "SQRT_YY",
        "SQRT_YY_DAG",
        "SQRT_ZZ",
        "SQRT_ZZ_DAG",
        "SWAP",
        "SWAPCX",
        "SWAPCZ",
        "XCX",
        "XCY",
        "XCZ",
        "YCX",
        "YCY",
        "YCZ",
        "ZCX",
        "ZCY",
        "ZCZ",
        "CORRELATED_ERROR",
        "DEPOLARIZE1",
        "DEPOLARIZE2",
        "E",
        "ELSE_CORRELATED_ERROR",
        "HERALDED_ERASE",
        "HERALDED_PAULI_CHANNEL_1",
        "PAULI_CHANNEL_1",
        "PAULI_CHANNEL_2",
        "X_ERROR",
        "Y_ERROR",
        "Z_ERROR",
        "M",
        "MR",
        "MRX",
        "MRY",
        "MRZ",
        "MX",
        "MY",
        "MZ",
        "R",
        "RX",
        "RY",
        "RZ",
        "MXX",
        "MYY",
        "MZZ",
        "MPP",
        "SPP",
        "SPP_DAG",
        "REPEAT",
        "DETECTOR",
        "MPAD",
        "OBSERVABLE_INCLUDE",
        "QUBIT_COORDS",
        "SHIFT_COORDS",
        "TICK",
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

    def submit(
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
        mpsmethod=None,
        traversal=None,
        noisemodel=None,
        streaming=None,
        **kwargs,
    ):
        """
        Submit a circuit or a list of quantum circuits to the Mimiq server.
        Returns a Job object (non-blocking).

        Args:
            circuits (Circuit or list of Circuits or str): A single Circuit object, a list of Circuit objects,
                                                           or QASM file paths representing the circuits to be executed.
            label (str): A label for the execution. Defaults to "pyapi_v" + __version__.
            algorithm (str): The algorithm to use. Defaults to "auto".
            nsamples (int): The number of samples to collect. Defaults to DEFAULT_SAMPLES.
            seed (int, optional): A seed for random number generation. Defaults to None.
            bitstrings (list of str, optional): Specific bitstrings to measure. Defaults to None.
            timelimit (int, optional): The maximum execution time in minutes. Defaults to None.
            bonddim (int, optional): The bond dimension to use. Defaults to None.
            entdim (int, optional): The entanglement dimension to use. Defaults to None.
            fuse (bool, optional): Whether to fuse gates. Defaults to None (let the remote service decide).
            reorderqubits (bool, optional): Whether to reorder qubits. Defaults to None (let the remote service decide).
            mpsmethod (str, optional): whether to use variational ("vmpoa", "vmpob") or direct ("dmpo") methods for MPO application in MPS simulations. Defaults to None (let the remote service decide).
            traversal (str, optional): method to traverse the circuit while compressing it into MPOs. Can be "sequential" (default) or "bfs" (Breadth-First Search). Defaults to None.
            noisemodel (NoiseModel, optional): A NoiseModel object to be applied to the circuit(s) before execution. Defaults to None.
            streaming (bool, optional): whether or not to use the streaming simulator. Defaults to None (let the remote service decide).
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
        from mimiqcircuits.circuittester import CircuitTesterExperiment

        if isinstance(circuits, CircuitTesterExperiment):
            c = circuits.build_circuit()
            return self.submit(
                c,
                label=label,
                algorithm=algorithm,
                nsamples=nsamples,
                bitstrings=bitstrings,
                timelimit=timelimit,
                bonddim=bonddim,
                entdim=entdim,
                fuse=fuse,
                reorderqubits=reorderqubits,
                seed=seed,
                qasmincludes=qasmincludes,
                force=force,
                mpsmethod=mpsmethod,
                traversal=traversal,
                noisemodel=noisemodel,
                streaming=streaming,
                **kwargs,
            )

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

            if noisemodel is not None:
                from mimiqcircuits.noisemodel import apply_noise_model, NoiseModel

                if not isinstance(noisemodel, NoiseModel):
                    raise TypeError(
                        f"noisemodel must be a NoiseModel object, got {type(noisemodel).__name__}."
                    )

                new_circuits = []
                for i, c in enumerate(circuits):
                    if isinstance(c, Circuit):
                        new_circuits.append(apply_noise_model(c, noisemodel))
                    elif isinstance(c, str):
                        raise ValueError(
                            f"Cannot apply NoiseModel to file path at index {i}. "
                            "Please load the circuit into a Circuit object first."
                        )
                    else:
                        # Will be caught by later checks, but safe to keep valid circuits
                        new_circuits.append(c)
                circuits = new_circuits

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
                        circuit_filename = os.path.join(
                            tmpdir, f"{CIRCUIT_FNAME}{i + 1}.{EXTENSION_QASM}"
                        )
                        shutil.copyfile(circuit, circuit_filename)
                        if qasmincludes is None:
                            qasmincludes = []
                        circuit_files.append(
                            {
                                "file": os.path.basename(circuit_filename),
                                "type": TYPE_QASM,
                            }
                        )
                        allfiles.append(circuit_filename)

                    # Case: STIM
                    elif _file_may_be_stim(circuit):
                        circuit_filename = os.path.join(
                            tmpdir, f"{CIRCUIT_FNAME}{i + 1}.{EXTENSION_STIM}"
                        )
                        shutil.copyfile(circuit, circuit_filename)
                        circuit_files.append(
                            {
                                "file": os.path.basename(circuit_filename),
                                "type": TYPE_STIM,
                            }
                        )
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

            if traversal is not None:
                if traversal not in ["sequential", "bfs"]:
                    raise ValueError("traversal must be one of 'sequential' or 'bfs'.")
                pars["mpoTraversal"] = traversal

            if mpsmethod is not None:
                if mpsmethod not in ["vmpoa", "vmpob", "dmpo"]:
                    raise ValueError(
                        "mpsmethod must be one of 'vmpoa', 'vmpob', or 'dmpo'."
                    )
                pars["mpsMethod"] = mpsmethod

            if streaming is not None:
                pars["streaming"] = streaming

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

    def execute(self, *args, **kwargs):
        import warnings

        warnings.warn(
            "execute() is deprecated and will be blocking in the future. "
            "Use submit() for non-blocking execution.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.submit(*args, **kwargs)

    def schedule(self, *args, **kwargs):
        """Deprecated alias for submit."""
        import warnings
        warnings.warn(
            "schedule() is deprecated. Use submit() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.submit(*args, **kwargs)

    def check_equivalence(
        self,
        experiment,
        **kwargs,
    ):
        """
        Executes a CircuitTesterExperiment and verifies the results.
        Blocks until execution is complete.

        Args:
            experiment (CircuitTesterExperiment): The experiment to run.
            **kwargs: Arguments passed to execute.

        Returns:
            float: The verification score (probability of all-zero state).
        """
        job = self.submit(experiment, **kwargs)
        # Assuming job has get_results()
        if hasattr(job, "get_results"):
            results = job.get_results()
            return experiment.interpret_results(results[0])
        else:

            # Fallback or error if job type is unexpected
            raise RuntimeError(
                "Job object returned by submit does not support get_results()."
            )

    def optimize(
        self,
        experiments,
        label="pyapi_v" + __version__,
        algorithm=DEFAULT_ALGORITHM,
        nsamples=DEFAULT_SAMPLES,
        timelimit=None,
        bonddim=None,
        entdim=None,
        fuse=None,
        reorderqubits=None,
        seed=None,
        history=False,
        force=False,
        debug=False,
        **kwargs,
    ):
        from mimiqcircuits.optimization_remote import optimize_impl

        return optimize_impl(
            self,
            experiments,
            label=label,
            algorithm=algorithm,
            nsamples=nsamples,
            timelimit=timelimit,
            bonddim=bonddim,
            entdim=entdim,
            fuse=fuse,
            reorderqubits=reorderqubits,
            seed=seed,
            history=history,
            force=force,
            debug=debug,
            **kwargs,
        )

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
