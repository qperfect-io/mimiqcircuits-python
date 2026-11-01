#
# Copyright © 2023-2026 QPerfect. All Rights Reserved.
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

"""Variational optimization of a parametric circuit on a local backend."""

import numpy as np
from scipy.optimize import approx_fprime, minimize

from mimiqcircuits.optimization import (
    OptimizationExperiment,
    OptimizationResults,
    OptimizationRun,
)

SUPPORTED_OPTIMIZERS = (
    "BFGS",
    "LBFGS",
    "CG",
    "NELDERMEAD",
    "NEWTON",
    "COBYLA",
    "CMAES",
)

_SCIPY_METHODS = {
    "BFGS": "BFGS",
    "LBFGS": "L-BFGS-B",
    "CG": "CG",
    "NELDERMEAD": "Nelder-Mead",
    "NEWTON": "Newton-CG",
    "COBYLA": "COBYLA",
}

_EXPERIMENT_KEYS = ("optimizer", "label", "maxiters", "zregister")


def _minimize(cost, x0, optimizer, maxiters):
    """Run ``optimizer`` on ``cost`` from ``x0``; the best point is tracked by
    the caller, so the return value is not needed."""
    if optimizer == "CMAES":
        try:
            import cma
        except ImportError as e:
            raise ImportError(
                "the CMAES optimizer needs the `cma` package: "
                "pip install 'mimiqcircuits[cmaes]'"
            ) from e
        sigma0 = float(np.std(x0)) or 0.5
        opts = {"verbose": -9}
        if maxiters is not None:
            opts["maxiter"] = maxiters
        cma.fmin2(cost, x0, sigma0, options=opts)
        return

    options = {} if maxiters is None else {"maxiter": maxiters}
    kwargs = {}
    if optimizer == "COBYLA":
        kwargs["tol"] = 1e-6
    elif optimizer == "NEWTON":
        # scipy's Newton-CG needs both derivatives; the steps balance
        # truncation against round-off of the nested differences.
        def jac(x):
            return approx_fprime(x, cost, 1e-6)

        kwargs["jac"] = jac
        kwargs["hess"] = lambda x: approx_fprime(x, jac, 1e-4)

    minimize(cost, x0, method=_SCIPY_METHODS[optimizer], options=options, **kwargs)


def as_experiments(experiment, initparams, kwargs):
    """Normalize the arguments of ``Backend.optimize`` into a list of
    experiments and whether a single one was given.

    The experiment fields (``optimizer``, ``label``, ``maxiters``,
    ``zregister``) are popped from ``kwargs`` when a circuit is given, so
    what is left is meant for execution.
    """
    if isinstance(experiment, (OptimizationExperiment, list, tuple)):
        if initparams is not None:
            raise TypeError(
                "initparams is read from the OptimizationExperiment; "
                "do not pass it as well"
            )
        repeated = [k for k in _EXPERIMENT_KEYS if k in kwargs]
        if repeated:
            raise TypeError(
                f"{', '.join(repeated)} is read from the OptimizationExperiment; "
                "do not pass it as well"
            )
        single = isinstance(experiment, OptimizationExperiment)
        experiments = [experiment] if single else list(experiment)
        for ex in experiments:
            if not isinstance(ex, OptimizationExperiment):
                raise TypeError(f"expected an OptimizationExperiment, got {type(ex).__name__}")
        return experiments, single

    fields = {k: kwargs.pop(k) for k in _EXPERIMENT_KEYS if k in kwargs}
    return [OptimizationExperiment(experiment, dict(initparams or {}), **fields)], True


def optimize(backend, experiment, initparams=None, *, return_history=False, **kwargs):
    """Implementation of :meth:`LocalBackend.optimize`."""
    experiments, single = as_experiments(experiment, initparams, kwargs)
    results = [_optimize_one(backend, ex, return_history, kwargs) for ex in experiments]
    return results[0] if single else results


def _optimize_one(backend, experiment, return_history, kwargs):
    if experiment.optimizer not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer: {experiment.optimizer}. Supported optimizers are: "
            + ", ".join(SUPPORTED_OPTIMIZERS)
        )

    zregister = experiment.zregister
    if zregister >= experiment.circuit.num_zvars():
        raise ValueError(
            f"zregister {zregister} is out of range: the circuit has "
            f"{experiment.circuit.num_zvars()} z-registers"
        )

    keys = list(experiment.initparams.keys())
    x0 = np.array([float(experiment.initparams[k]) for k in keys])

    best = None
    history = []

    def cost(x):
        nonlocal best
        params = {k: float(v) for k, v in zip(keys, x)}
        results = backend.execute(experiment.circuit.evaluate(params), **kwargs)
        value = float(np.mean([complex(z[zregister]).real for z in results.zstates]))
        run = OptimizationRun(cost=value, parameters=params, results=results)
        if best is None or value < best.cost:
            best = run
        if return_history:
            history.append(run)
        return value

    _minimize(cost, x0, experiment.optimizer, experiment.maxiters)

    return OptimizationResults(best=best, history=history)
