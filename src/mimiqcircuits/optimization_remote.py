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

import tempfile
import json
import os
import shutil
import atexit
from mimiqcircuits.__version__ import __version__
import numpy as np
from time import sleep
import mimiqcircuits as mc
from mimiqcircuits.remote import *
from mimiqcircuits.optimization import *
from mimiqcircuits.remote import (
    RemoteConnection,
    DEFAULT_ALGORITHM,
    DEFAULT_BONDDIM,
    DEFAULT_ENTDIM,
    DEFAULT_SAMPLES,
    MAX_BONDDIM,
    MAX_ENTDIM,
    MAX_SAMPLES,
    EXTENSION_PROTO,
    TYPE_PROTO,
    MIN_BONDDIM,
    MIN_ENTDIM,
)


def optimize_impl(
    self,
    experiments,
    label="pyapi_v" + __version__,
    algorithm=DEFAULT_ALGORITHM,
    nsamples=DEFAULT_SAMPLES,
    timelimit=None,
    bonddim=None,
    entdim=None,
    remove_swaps=None,
    canonical_decompose=None,
    fuse=None,
    reorderqubits=None,
    seed=None,
    history=False,
    force=False,
    debug=False,
    **kwargs,
):
    r"""
    Submit one or more quantum optimization experiments to the remote execution backend.

    This function prepares and launches one or more `OptimizationExperiment` instances
    using the specified algorithm and execution parameters. It handles input serialization,
    temporary workspace creation, and submission via the `RemoteConnection`.

    Args:
        experiments (OptimizationExperiment or list of OptimizationExperiment): A single `OptimizationExperiment` or a list of experiments to optimize.
        label (str, optional): A label for the execution. Defaults to `"pyapi_v" + __version__`.
        algorithm (str, optional): Optimization algorithm to use. One of `"mps"`, `"statevector"`, or `"auto"`. Defaults to `"auto"`.
        nsamples (int, optional): Number of samples used to estimate expectation values. Defaults to `DEFAULT_SAMPLES`.
        timelimit (int, optional): Maximum execution time in minutes. If `None`, uses the backend’s default.
        bonddim (int, optional): Bond dimension for MPS simulation. Used only if algorithm is `"mps"` or `"auto"`.
        entdim (int, optional): Entanglement dimension for MPS simulation. Used only if algorithm is `"mps"` or `"auto"`.
        remove_swaps (bool, optional): Whether to remove SWAP gates. If `None`, backend chooses automatically.
        canonical_decompose (bool, optional): Whether to decompose the circuit into GateU and GateCX. If `None`, backend chooses automatically.
        fuse (bool, optional): Whether to fuse single-qubit gates. If `None`, backend chooses automatically.
        reorderqubits (bool, optional): Whether to reorder qubits before execution. If `None`, backend chooses automatically.
        seed (int, optional): Seed for random number generation. If `None`, a random seed is chosen.
        history (bool, optional): If `True`, records the full optimization trajectory. Defaults to `False`.
        force (bool, optional): If `True`, bypasses entdim range checks. Use with caution.
        debug (bool, optional): If `True`, keeps the temporary working directory for debugging. Defaults to `False`.
        **kwargs: Additional backend-specific keyword arguments.

    Raises:
        ValueError: If any parameters are out of allowed bounds, or no experiments are provided.
        TypeError: If any item in `experiments` is not an `OptimizationExperiment`.

    Returns:
        str: Remote job identifier (job ID or label) returned by the backend.

    Notes:
        - If `algorithm="auto"` and multiple experiments are provided, an error is raised.
        - Temporary files are created in a workspace and deleted automatically unless `debug=True`.
        - All experiments are serialized to protobuf before submission.
    """
    if nsamples > MAX_SAMPLES:
        raise ValueError(f"nsamples must be less than {MAX_SAMPLES}")

    if timelimit is None:
        timelimit = self._RemoteConnection__get_timelimit()

    maxtimelimit = self._RemoteConnection__get_timelimit()
    if timelimit > maxtimelimit:
        raise ValueError(f"Timelimit cannot be set more than {maxtimelimit} minutes.")

    if seed is None:
        seed = int(np.random.randint(0, np.iinfo(np.int_).max, dtype=np.int_))

    if isinstance(experiments, OptimizationExperiment):
        experiments = [experiments]

    if len(experiments) > 1 and algorithm == "auto":
        raise ValueError(
            "The 'auto' algorithm is not supported in batch mode. Please specify 'mps' or 'statevector' for batch optimizations."
        )

    if not experiments:
        raise ValueError("At least one OptimizationExperiment must be provided.")

    tmpdir = tempfile.mkdtemp(prefix="mimiq_opt_")

    if not debug:
        atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))
    else:
        print("Debug mode ON — temp folder will not be deleted.")

    experiment_files = []
    experiment_entries = []

    for i, ex in enumerate(experiments):
        if not isinstance(ex, OptimizationExperiment):
            raise TypeError(f"Expected OptimizationExperiment, got {type(ex)}")
        fname = f"experiment_{i + 1}.{EXTENSION_PROTO}"
        fpath = os.path.join(tmpdir, fname)
        ex.saveproto(fpath)
        experiment_files.append(fpath)
        experiment_entries.append({"file": fname, "type": TYPE_PROTO})

    pars = {
        "experiments": experiment_entries,
        "algorithm": algorithm,
        "samples": nsamples,
        "seed": seed,
        "history": history,
    }

    if algorithm in ["auto", "mps"]:
        if bonddim is None:
            bonddim = DEFAULT_BONDDIM
        if entdim is None:
            entdim = DEFAULT_ENTDIM

    if bonddim is not None:
        if bonddim < MIN_BONDDIM or bonddim > MAX_BONDDIM:
            raise ValueError(f"bonddim must be between {MIN_BONDDIM} and {MAX_BONDDIM}")
        pars["bondDimension"] = bonddim

    if algorithm in ["mps", "auto"]:
        actual_entdim = DEFAULT_ENTDIM if entdim is None else entdim
        if actual_entdim < MIN_ENTDIM and not force:
            raise ValueError(f"entdim must be between {MIN_ENTDIM} and {MAX_ENTDIM}")
        elif actual_entdim < MIN_ENTDIM and force:
            print(
                f"Warning: Running optimization with entdim={actual_entdim}. Results may be misleading."
            )
        pars["entDimension"] = actual_entdim

    if remove_swaps is not None:
        pars["removeSwaps"] = remove_swaps

    if canonical_decompose is not None:
        pars["canonicalDecompose"] = canonical_decompose

    if fuse is not None:
        pars["fuse"] = fuse

    if reorderqubits is not None:
        pars["reorderQubits"] = reorderqubits

    if kwargs:
        import warnings

        for key in kwargs:
            warnings.warn(
                f"Unknown option '{key}' passed to optimize(). This option will be sent to the backend but may be ignored.",
                stacklevel=2,
            )

    pars.update(kwargs)

    pars_file = os.path.join(tmpdir, "optimize.json")
    with open(pars_file, "w") as f:
        json.dump(pars, f)

    req = {
        "executor": "Optimize",
        "timelimit": timelimit,
        "apilang": "python",
        "apiversion": __version__,
        "circuitsapiversion": __version__,
    }

    req_file = os.path.join(tmpdir, "request.json")
    with open(req_file, "w") as f:
        json.dump(req, f)

    sleep(0.1)

    return self.connection.request(
        "CIRC", algorithm, label, timelimit, [req_file, pars_file] + experiment_files
    )


__all__ = ["optimize_impl"]
