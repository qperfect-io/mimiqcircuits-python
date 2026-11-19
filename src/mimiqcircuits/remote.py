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

import tempfile
import json
import os
import shutil
from mimiqcircuits.circuit import Circuit
from mimiqcircuits.__version__ import __version__
from mimiqcircuits.proto import WIRE_FORMAT_VERSION
import numpy as np
from time import sleep
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mimiqlink

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


MPS_METHODS = ("vmpoa", "vmpob", "dmpo")
MPO_TRAVERSALS = ("sequential", "bfs", "frontierpack")


def _set_exec_opts(
    pars,
    mpsmethod=None,
    mpotraversal=None,
    streaming=None,
    remove_swaps=None,
    canonicaldecompose=None,
    fuse=None,
    reorderqubits=None,
    reorderqubits_seed=None,
):
    """Write the optional execution options shared by ``submit`` and ``optimize``.

    ``None`` leaves the choice to the remote service.
    """
    if mpsmethod is not None:
        if mpsmethod not in MPS_METHODS:
            raise ValueError(f"mpsmethod must be one of {MPS_METHODS}.")
        pars["mpsMethod"] = mpsmethod

    if mpotraversal is not None:
        if mpotraversal not in MPO_TRAVERSALS:
            raise ValueError(f"mpotraversal must be one of {MPO_TRAVERSALS}.")
        pars["mpoTraversal"] = mpotraversal

    for key, val in (
        ("streaming", streaming),
        ("removeSwaps", remove_swaps),
        ("canonicalDecompose", canonicaldecompose),
        ("fuse", fuse),
        ("reorderQubits", reorderqubits),
        ("reorderQubitsSeed", reorderqubits_seed),
    ):
        if val is not None:
            pars[key] = val

    return pars


class RemoteConnection:
    """Base class for connections to the MIMIQ remote services.

    It wraps a ``mimiqlink`` connection and adds the circuit execution API:
    :meth:`submit`, :meth:`optimize`, :meth:`check_equivalence`, and the
    ``get_result``/``get_results``/``get_input``/``get_inputs`` retrieval
    methods. Any other attribute or method (``connect``, ``close``,
    ``loadtoken``, ``savetoken``, ``requests``, ``stopExecution``, ...) is
    forwarded to the wrapped connection. Forwarded methods that return the
    wrapped connection return this object instead, so calls such as
    ``conn.connect()`` can be chained.

    Use :class:`MimiqConnection` or :class:`PlanqkConnection` rather than this
    class directly.
    """

    def __init__(self, connection: "mimiqlink.AbstractConnection"):
        """Wrap an existing ``mimiqlink`` connection.

        Args:
            connection: A ``mimiqlink`` connection object, such as
                ``mimiqlink.MimiqConnection``, ``mimiqlink.QhiveConnection``
                or ``mimiqlink.PlanqkConnection``.
        """
        self.connection = connection

    def __get_timelimit(self):
        """Fetch the maximum time limit for execution from the server."""
        import mimiqlink

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

        # If it is not callable, just return it
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
        mpscutoff=None,
        remove_swaps=None,
        canonicaldecompose=None,
        fuse=None,
        reorderqubits=None,
        reorderqubits_seed=None,
        seed=None,
        qasmincludes=None,
        force=False,
        mpsmethod=None,
        mpotraversal=None,
        noisemodel=None,
        streaming=None,
    ):
        """Submit one or more circuits for execution on the remote service.

        The call uploads the circuits and returns at once, without waiting for
        the execution to finish. Pass the returned identifier to
        ``get_result`` or ``get_results`` to wait for and download the
        results.

        Circuits are sent in protobuf format. File paths are uploaded as they
        are and must point to an OpenQASM 2.0 file (its first non-comment line
        is ``OPENQASM 2.0;``) or a STIM file. An
        :class:`~mimiqcircuits.EquivalenceExperiment` is turned into its test
        circuit with ``build_circuit()`` and submitted.

        Args:
            circuits (Circuit | str | EquivalenceExperiment | list): A circuit,
                a path to an OpenQASM 2.0 or STIM file, an equivalence
                experiment, or a list of circuits and file paths.
            label (str): A label to identify the execution. Defaults to
                ``"pyapi_v"`` followed by the package version.
            algorithm (str): The simulator to use: ``"auto"``, ``"mps"`` or
                ``"statevector"``. ``"auto"`` lets the service choose and
                accepts a single circuit only. Defaults to ``"auto"``.
            nsamples (int): The number of measurement samples to collect. At
                most ``MAX_SAMPLES`` (65536). Defaults to 1000.
            bitstrings (list of BitString, optional): Bitstrings whose
                amplitudes are computed. When ``circuits`` is a single
                :class:`~mimiqcircuits.Circuit`, each must have one bit per
                qubit. Defaults to None (no amplitudes).
            timelimit (int, optional): The maximum execution time in minutes.
                Defaults to None, which uses the largest time limit allowed for
                the account (30 minutes if the service sets none).
            bonddim (int, optional): The MPS bond dimension, between 1 and
                4096. Defaults to None, which means 256 for the ``"auto"`` and
                ``"mps"`` algorithms.
            entdim (int, optional): The MPS entanglement dimension, at least
                4. Defaults to None, which means 16 for the ``"auto"`` and
                ``"mps"`` algorithms.
            mpscutoff (float, optional): Singular value truncation cutoff for
                MPS simulation. Smaller values give higher accuracy at higher
                cost. Must be non-negative. Defaults to None (the service
                decides).
            remove_swaps (bool, optional): Whether to remove SWAP gates by
                relabeling qubits. Defaults to None (the service decides).
            canonicaldecompose (bool, optional): Whether to decompose the
                circuit into ``GateU`` and ``GateCX``. Defaults to None (the
                service decides).
            fuse (bool, optional): Whether to fuse gates. Defaults to None (the
                service decides).
            reorderqubits (bool or str, optional): Whether to reorder qubits,
                or the name of the reordering method (for example
                ``"greedy"``, ``"spectral"``, ``"rcm"``, ``"sa_warm_start"``,
                ``"sa_only"``, ``"memetic"``, ``"multilevel"``, ``"grasp"``,
                ``"hybrid"``). Defaults to None (the service decides).
            reorderqubits_seed (int, optional): Seed for the qubit reordering,
                independent of ``seed``. Defaults to None (derived from
                ``seed``).
            seed (int, optional): Seed for the simulation's random number
                generator. Defaults to None (a random seed).
            qasmincludes (list of str, optional): Accepted for compatibility
                and not used.
            force (bool): If True, allow ``entdim`` below 4 and print a
                warning instead of raising. Defaults to False.
            mpsmethod (str, optional): How MPOs are applied in MPS simulation:
                ``"vmpoa"`` or ``"vmpob"`` (variational) or ``"dmpo"``
                (direct). Defaults to None (the service decides).
            mpotraversal (str, optional): How the circuit is traversed when
                compressing it into MPOs: ``"sequential"``, ``"bfs"``
                (breadth-first search) or ``"frontierpack"`` (packs
                independent ready gates into the open MPO, good for brickwork
                circuits). Defaults to None (the service decides).
            noisemodel (NoiseModel, optional): A noise model applied to every
                :class:`~mimiqcircuits.Circuit` before upload. It cannot be
                applied to file paths. Defaults to None.
            streaming (bool, optional): Whether to use the streaming
                simulator. Defaults to None (the service decides).

        Returns:
            str: The identifier of the execution on the remote service.

        Raises:
            ValueError: If ``nsamples``, ``timelimit``, ``bonddim``, ``entdim``,
                ``mpscutoff``, ``mpsmethod`` or ``mpotraversal`` is out of
                range; if a bitstring has the wrong length; if the list of
                circuits is empty, contains an empty circuit, a circuit with
                unevaluated symbolic parameters, or a path that does not exist
                or is neither OpenQASM 2.0 nor STIM; if ``algorithm`` is
                ``"auto"`` with more than one circuit; or if ``noisemodel`` is
                given with a file path.
            TypeError: If an item of ``circuits`` is not a circuit or a path,
                if ``noisemodel`` is not a :class:`~mimiqcircuits.NoiseModel`,
                or if a preparation flag has the wrong type.
            mimiqlink.MimiqConnectionError: If the connection is not
                authenticated or the upload fails.

        Examples:
            >>> from mimiqcircuits import *
            >>> conn = MimiqConnection()
            >>> conn.connect()  # doctest: +SKIP

            Build a Bell-state circuit and measure both qubits.

            >>> c = Circuit()
            >>> c.push(GateH(), 0)
            1-qubit circuit with 1 instruction:
            └── H @ q[0]
            <BLANKLINE>
            >>> c.push(GateCX(), 0, 1)
            2-qubit circuit with 2 instructions:
            ├── H @ q[0]
            └── CX @ q[0], q[1]
            <BLANKLINE>
            >>> c.push(Measure(), range(2), range(2))
            2-qubit, 2-bit circuit with 4 instructions:
            ├── H @ q[0]
            ├── CX @ q[0], q[1]
            ├── M @ q[0], c[0]
            └── M @ q[1], c[1]
            <BLANKLINE>

            Submit it: the call returns an identifier straight away.

            >>> job = conn.submit(c, nsamples=1000, algorithm="statevector")  # doctest: +SKIP

            Wait for the execution to finish and download its results.

            >>> res = conn.get_result(job)  # doctest: +SKIP
            >>> res.histogram()  # doctest: +SKIP
        """
        from mimiqcircuits.equivalence import EquivalenceExperiment

        if isinstance(circuits, EquivalenceExperiment):
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
                mpscutoff=mpscutoff,
                remove_swaps=remove_swaps,
                canonicaldecompose=canonicaldecompose,
                fuse=fuse,
                reorderqubits=reorderqubits,
                reorderqubits_seed=reorderqubits_seed,
                seed=seed,
                qasmincludes=qasmincludes,
                force=force,
                mpsmethod=mpsmethod,
                mpotraversal=mpotraversal,
                noisemodel=noisemodel,
                streaming=streaming,
            )

        from mimiqcircuits.backends.passes import check_prep_flags

        flags = check_prep_flags(
            fuse=fuse,
            canonicaldecompose=canonicaldecompose,
            reorderqubits=reorderqubits,
            remove_swaps=remove_swaps,
            reorderqubits_seed=reorderqubits_seed,
        )
        fuse = flags["fuse"]
        canonicaldecompose = flags["canonicaldecompose"]
        reorderqubits = flags["reorderqubits"]
        remove_swaps = flags["remove_swaps"]
        reorderqubits_seed = flags["reorderqubits_seed"]

        if nsamples > MAX_SAMPLES:

            raise ValueError(f"nsamples must be at most {MAX_SAMPLES}")

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
            if actual_entdim > MAX_ENTDIM or (actual_entdim < MIN_ENTDIM and not force):
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

            if mpscutoff is not None:
                if mpscutoff < 0:
                    raise ValueError("mpscutoff must be non-negative")
                pars["mpsCutoff"] = mpscutoff

            _set_exec_opts(
                pars,
                mpsmethod=mpsmethod,
                mpotraversal=mpotraversal,
                streaming=streaming,
                remove_swaps=remove_swaps,
                canonicaldecompose=canonicaldecompose,
                fuse=fuse,
                reorderqubits=reorderqubits,
                reorderqubits_seed=reorderqubits_seed,
            )

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
                "wireformatversion": WIRE_FORMAT_VERSION,
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
        """Submit circuits for execution. Deprecated: use :meth:`submit`.

        Takes the same arguments as :meth:`submit`, emits a
        :class:`DeprecationWarning`, and returns the execution identifier
        without waiting for the execution to finish.
        """
        import warnings

        warnings.warn(
            "execute() is deprecated and will be blocking in the future. "
            "Use submit() for non-blocking execution.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.submit(*args, **kwargs)

    def check_equivalence(
        self,
        experiment,
        **kwargs,
    ):
        """Run an equivalence experiment remotely and return its score.

        Submits ``experiment`` with :meth:`submit`, waits for the results and
        passes the first one to ``experiment.interpret_results``.

        Args:
            experiment (EquivalenceExperiment): The experiment to run.
            **kwargs: Keyword arguments passed to :meth:`submit`.

        Returns:
            float: The verification score (probability of the all-zero state).

        Raises:
            RuntimeError: If the remote job fails or is canceled.
        """
        job = self.submit(experiment, **kwargs)
        return experiment.interpret_results(self.get_result(job))

    def optimize(
        self,
        experiments,
        label="pyapi_v" + __version__,
        algorithm=DEFAULT_ALGORITHM,
        nsamples=DEFAULT_SAMPLES,
        timelimit=None,
        bonddim=None,
        entdim=None,
        mpscutoff=None,
        remove_swaps=None,
        canonicaldecompose=None,
        fuse=None,
        reorderqubits=None,
        reorderqubits_seed=None,
        seed=None,
        history=False,
        force=False,
        debug=False,
        mpsmethod=None,
        mpotraversal=None,
        streaming=None,
    ):
        """Submit one or more optimization experiments to the remote service.

        The call uploads the experiments and returns at once. Pass the
        returned identifier to ``get_result`` or ``get_results`` to wait for
        the execution and obtain :class:`~mimiqcircuits.OptimizationResults`.

        Args:
            experiments (OptimizationExperiment or list): An experiment or a
                list of experiments.
            label (str): A label to identify the execution. Defaults to
                ``"pyapi_v"`` followed by the package version.
            algorithm (str): The simulator to use: ``"auto"``, ``"mps"`` or
                ``"statevector"``. ``"auto"`` accepts a single experiment only.
                Defaults to ``"auto"``.
            nsamples (int): The number of samples used to estimate expectation
                values. At most 65536. Defaults to 1000.
            timelimit (int, optional): The maximum execution time in minutes.
                Defaults to None (the largest limit allowed for the account).
            bonddim (int, optional): The MPS bond dimension. Defaults to None
                (256 for ``"auto"`` and ``"mps"``).
            entdim (int, optional): The MPS entanglement dimension. Defaults to
                None (16 for ``"auto"`` and ``"mps"``).
            mpscutoff (float, optional): Singular value truncation cutoff for
                MPS simulation. Defaults to None (the service decides).
            remove_swaps (bool, optional): Whether to remove SWAP gates.
                Defaults to None (the service decides).
            canonicaldecompose (bool, optional): Whether to decompose into
                ``GateU`` and ``GateCX``. Defaults to None (the service
                decides).
            fuse (bool, optional): Whether to fuse gates. Defaults to None (the
                service decides).
            reorderqubits (bool or str, optional): Whether to reorder qubits,
                or the reordering method. Defaults to None (the service
                decides).
            reorderqubits_seed (int, optional): Seed for the qubit reordering.
                Defaults to None (derived from ``seed``).
            seed (int, optional): Seed for the random number generator.
                Defaults to None (a random seed).
            history (bool): If True, record every evaluation of the
                optimization. Defaults to False.
            force (bool): If True, allow ``entdim`` below 4 with a warning.
                Defaults to False.
            debug (bool): If True, keep the temporary upload directory.
                Defaults to False.
            mpsmethod (str, optional): ``"vmpoa"``, ``"vmpob"`` or ``"dmpo"``.
                Defaults to None (the service decides).
            mpotraversal (str, optional): ``"sequential"``, ``"bfs"`` or
                ``"frontierpack"``. Defaults to None (the service decides).
            streaming (bool, optional): Whether to use the streaming
                simulator. Defaults to None (the service decides).

        Returns:
            str: The identifier of the execution on the remote service.

        Raises:
            ValueError: If a parameter is out of range, no experiment is
                given, or ``algorithm`` is ``"auto"`` with more than one
                experiment.
            TypeError: If an item is not an
                :class:`~mimiqcircuits.OptimizationExperiment`, or a
                preparation flag has the wrong type.

        Examples:
            >>> from mimiqcircuits import *
            >>> conn = MimiqConnection()
            >>> conn.connect()  # doctest: +SKIP

            Build a one-parameter circuit and the quantity to minimise.

            >>> from symengine import symbols
            >>> x = symbols("x")
            >>> c = Circuit()
            >>> c.push(GateRX(x), 0)
            1-qubit circuit with 1 instruction:
            └── RX(x) @ q[0]
            <BLANKLINE>
            >>> c.push(ExpectationValue(GateZ()), 0, 0)
            1-qubit, 1-zvar circuit with 2 instructions:
            ├── RX(x) @ q[0]
            └── ⟨Z⟩ @ q[0], z[0]
            <BLANKLINE>

            Minimise z[0] starting from x = 0.5, then fetch the best value.

            >>> ex = OptimizationExperiment(c, {x: 0.5}, zregister=0)
            >>> job = conn.optimize(ex)  # doctest: +SKIP
            >>> res = conn.get_result(job)  # doctest: +SKIP
            >>> res.get_best()  # doctest: +SKIP
        """
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
            mpscutoff=mpscutoff,
            remove_swaps=remove_swaps,
            canonicaldecompose=canonicaldecompose,
            fuse=fuse,
            reorderqubits=reorderqubits,
            reorderqubits_seed=reorderqubits_seed,
            seed=seed,
            history=history,
            force=force,
            debug=debug,
            mpsmethod=mpsmethod,
            mpotraversal=mpotraversal,
            streaming=streaming,
        )

    def __repr__(self):
        """Return a string representation of the connection."""
        return self.connection.__repr__()

    def __str__(self):
        """Return a string representation of the connection."""
        return self.connection.__str__()


class MimiqConnection(RemoteConnection):
    """Connection to the MIMIQ cloud services.

    Wraps a ``mimiqlink.MimiqConnection``, or a ``mimiqlink.QhiveConnection``
    for Quantum Hive deployments, and adds the circuit execution API described
    in :class:`RemoteConnection`. The object is created unauthenticated: call
    ``connect`` (or ``loadtoken``) before submitting anything.

    Examples:
        Connect to the default MIMIQ cloud. Without arguments ``connect``
        opens a login page in the browser.

        >>> from mimiqcircuits import *
        >>> conn = MimiqConnection()
        >>> conn.connect()  # doctest: +SKIP

        Save the session token to reuse it later without logging in again.

        >>> conn.savetoken("qperfect.json")  # doctest: +SKIP
        >>> conn = MimiqConnection().loadtoken("qperfect.json")  # doctest: +SKIP

        Submit a circuit and wait for its results.

        >>> c = Circuit()
        >>> c.push(GateH(), 0)
        1-qubit circuit with 1 instruction:
        └── H @ q[0]
        <BLANKLINE>
        >>> c.push(Measure(), 0, 0)
        1-qubit, 1-bit circuit with 2 instructions:
        ├── H @ q[0]
        └── M @ q[0], c[0]
        <BLANKLINE>
        >>> job = conn.submit(c, nsamples=100)  # doctest: +SKIP
        >>> res = conn.get_result(job)  # doctest: +SKIP

        List past executions and close the session.

        >>> conn.requests()  # doctest: +SKIP
        >>> conn.close()  # doctest: +SKIP
    """

    def __init__(self, url=None, auth_url=None):
        """Create an unauthenticated connection.

        Args:
            url (str, optional): The URL of the MIMIQ service. Defaults to None
                (``mimiqlink.QPERFECT_CLOUD``, or
                ``mimiqlink.QPERFECT_CLOUD_V2`` when ``auth_url`` is given).
            auth_url (str, optional): The URL of the authentication server.
                Giving one, or giving ``mimiqlink.QPERFECT_CLOUD_V2`` or
                ``mimiqlink.QPERFECT_DEV`` as ``url``, connects to Quantum
                Hive, which authenticates through Keycloak. The known URLs pick
                their own authentication server, so ``auth_url`` is only
                needed for other deployments. Defaults to None.

        Warns:
            FutureWarning: When connecting to the current MIMIQ web service,
                which will be deprecated in favour of MIMIQ V2. Set the
                ``MIMIQ_V2_WARNING`` environment variable to ``off``, ``false``,
                ``no`` or ``0`` to silence it.
        """
        import mimiqlink

        if auth_url is not None or url in (mimiqlink.QPERFECT_CLOUD_V2, mimiqlink.QPERFECT_DEV):
            connection = mimiqlink.QhiveConnection(url, auth_url)
        else:
            if os.environ.get("MIMIQ_V2_WARNING", "").strip().lower() not in ("0", "false", "no", "off"):
                import warnings

                warnings.warn(
                    "You are connecting to the current MIMIQ web service, which "
                    "will be deprecated. See "
                    "https://docs.qperfect.io/mimiqcircuits-python/migration_v2.html. "
                    "Set MIMIQ_V2_WARNING=off to silence this warning.",
                    FutureWarning,
                    stacklevel=2,
                )
            connection = mimiqlink.MimiqConnection(url)
        super().__init__(connection)


class PlanqkConnection(RemoteConnection):
    """Connection to the MIMIQ services through the PlanQK gateway.

    Wraps a ``mimiqlink.PlanqkConnection`` and adds the circuit execution API
    described in :class:`RemoteConnection`. Call ``connect`` before
    submitting anything.

    Examples:
        Connect with explicit credentials. Without arguments, ``connect``
        reads ``PLANQK_CONSUMER_KEY`` and ``PLANQK_CONSUMER_SECRET`` from the
        environment.

        >>> from mimiqcircuits import *
        >>> from mimiqcircuits.remote import PlanqkConnection
        >>> conn = PlanqkConnection()
        >>> conn.connect("my_consumer_key", "my_consumer_secret")  # doctest: +SKIP

        Submit and retrieve results as with :class:`MimiqConnection`.

        >>> c = Circuit()
        >>> c.push(GateX(), 0)
        1-qubit circuit with 1 instruction:
        └── X @ q[0]
        <BLANKLINE>
        >>> c.push(Measure(), 0, 0)
        1-qubit, 1-bit circuit with 2 instructions:
        ├── X @ q[0]
        └── M @ q[0], c[0]
        <BLANKLINE>
        >>> job = conn.submit(c)  # doctest: +SKIP
        >>> res = conn.get_result(job)  # doctest: +SKIP
    """

    def __init__(self, url=None, consumer_key=None, consumer_secret=None):
        """Create an unauthenticated PlanQK connection.

        Args:
            url (str, optional): The URL of the PlanQK gateway. Defaults to
                None (``mimiqlink.PLANQK_API``).
            consumer_key (str, optional): The PlanQK consumer key. Defaults to
                None.
            consumer_secret (str, optional): The PlanQK consumer secret.
                Defaults to None.
        """
        import mimiqlink

        connection = mimiqlink.PlanqkConnection(url, consumer_key, consumer_secret)
        super().__init__(connection)


__all__ = ["MimiqConnection", "PlanqkConnection", "RemoteConnection"]
