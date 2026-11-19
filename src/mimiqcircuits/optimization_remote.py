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
from mimiqcircuits.proto import WIRE_FORMAT_VERSION
import numpy as np
from time import sleep
from mimiqcircuits.remote import *
from mimiqcircuits.optimization import *
from mimiqcircuits.remote import (
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
    _set_exec_opts,
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
    r"""Submit optimization experiments to the MIMIQ service.

    This is the implementation of ``RemoteConnection.optimize``; ``self`` is
    the connection. The experiments are saved to Protobuf files in a
    temporary folder, sent with the request, and the call returns without
    waiting. Fetch the :class:`~mimiqcircuits.OptimizationResults` with the
    connection's ``get_results``.

    Args:
        self (RemoteConnection): The connection to submit through.
        experiments (OptimizationExperiment or list): One experiment, or a
            list of them.
        label (str): Label of the job. Defaults to ``"pyapi_v"`` followed by
            the package version.
        algorithm (str): Simulator to evaluate the cost with: ``"auto"``,
            ``"mps"`` or ``"statevector"``. Defaults to ``"auto"``.
        nsamples (int): Number of shots per evaluation.
        timelimit (int): Time limit of the job in minutes. ``None`` uses the
            largest allowed.
        bonddim (int): Bond dimension of the MPS simulator. ``None`` uses the
            default with ``"mps"`` and ``"auto"``.
        entdim (int): Entanglement dimension of the MPS simulator. ``None``
            uses the default with ``"mps"`` and ``"auto"``.
        mpscutoff (float): Singular-value cutoff of the MPS simulator.
            ``None`` lets the service choose.
        remove_swaps (bool): Server-side SWAP removal. ``None`` lets the
            service choose.
        canonicaldecompose (bool): Server-side decomposition into the
            canonical gate set. ``None`` lets the service choose.
        fuse (bool): Server-side gate fusion. ``None`` lets the service
            choose.
        reorderqubits (bool or str): Server-side qubit reordering, or the
            name of the reordering method. ``None`` lets the service choose.
        reorderqubits_seed (int): Seed of the reordering, separate from the
            simulation seed. ``None`` uses the simulation seed.
        seed (int): Seed of the simulation. ``None`` draws one at random.
        history (bool): Also return every evaluated run. Defaults to
            ``False``.
        force (bool): Allow an ``entdim`` below the minimum, with a printed
            warning. Defaults to ``False``.
        debug (bool): Keep the temporary folder after the process exits, and
            print a notice. Defaults to ``False``.
        mpsmethod (str): MPO application method of the MPS simulator:
            ``"dmpo"``, ``"vmpoa"`` or ``"vmpob"``. ``None`` lets the service
            choose.
        mpotraversal (str): MPO compression traversal: ``"sequential"``,
            ``"bfs"`` or ``"frontierpack"``. ``None`` lets the service
            choose.
        streaming (bool): Use the streaming MPS simulator. ``None`` lets the
            service choose.

    Returns:
        The identifier of the submitted job.

    Raises:
        ValueError: If no experiment is given, more than one is given with
            ``algorithm="auto"``, or an option is out of range
            (``nsamples``, ``timelimit``, ``bonddim``, ``entdim``,
            ``mpscutoff``, ``mpsmethod``, ``mpotraversal``).
        TypeError: If an item of ``experiments`` is not an
            :class:`~mimiqcircuits.OptimizationExperiment`, or a preparation
            flag has the wrong type.

    Examples:
        >>> from mimiqcircuits import *
        >>> from symengine import symbols
        >>> theta = symbols("theta")
        >>> c = Circuit().push(GateRY(theta), 0)
        >>> c.push(ExpectationValue(GateZ()), 0, 0)  # doctest: +ELLIPSIS
        1-qubit, 1-zvar circuit with 2 instructions:
        ...
        >>> ex = OptimizationExperiment(c, {theta: 0.1})
        >>> conn = MimiqConnection()
        >>> conn.connect()  # doctest: +SKIP
        >>> job = conn.optimize(ex, algorithm="statevector", nsamples=1)  # doctest: +SKIP
        >>> res = conn.get_results(job)[0]  # doctest: +SKIP
        >>> res.get_best().cost  # doctest: +SKIP
    """
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

    pars_file = os.path.join(tmpdir, "optimize.json")
    with open(pars_file, "w") as f:
        json.dump(pars, f)

    req = {
        "executor": "Optimize",
        "timelimit": timelimit,
        "apilang": "python",
        "apiversion": __version__,
        "circuitsapiversion": __version__,
        "wireformatversion": WIRE_FORMAT_VERSION,
    }

    req_file = os.path.join(tmpdir, "request.json")
    with open(req_file, "w") as f:
        json.dump(req, f)

    sleep(0.1)

    return self.connection.request(
        "CIRC", algorithm, label, timelimit, [req_file, pars_file] + experiment_files
    )


__all__ = ["optimize_impl"]
