import pytest
from mimiqcircuits import Circuit, GateRX, GateRY, QCSResults
from mimiqcircuits.optimization import (
    OptimizationExperiment,
    OptimizationRun,
    OptimizationResults,
)
from symengine import symbols


# Test OptimizationExperiment
def test_valid_optimization_experiment():
    x, y = symbols("x y")
    c = Circuit().push(GateRX(x), 0).push(GateRY(y), 1)
    exp = OptimizationExperiment(
        circuit=c,
        initparams={x: 0.1, y: 2.0},
        optimizer="BFGS",
        label="test",
        maxiters=100,
        zregister=0,
    )
    assert exp.is_valid()
    assert exp.num_params() == 2
    assert exp.num_qubits() == 2
    assert exp.get_param(x) == 0.1


def test_optimization_experiment_param_mismatch():
    x, y = symbols("x y")
    c = Circuit().push(GateRX(x), 0).push(GateRY(y), 1)
    with pytest.raises(ValueError):
        OptimizationExperiment(
            circuit=c,
            initparams={x: 0.1},  # missing y
            optimizer="BFGS",
            label="fail",
            maxiters=100,
            zregister=0,
        )


def test_invalid_method():
    x = symbols("x")
    c = Circuit().push(GateRX(x), 0)
    with pytest.raises(ValueError):
        OptimizationExperiment(
            circuit=c,
            initparams={x: 0.5},
            optimizer="COBYLA",
            label="fail",
            maxiters=100,
            zregister=-1,
        )


def test_optimization_experiment_evaluation():
    x, y = symbols("x y")
    c = Circuit().push(GateRX(x), 0).push(GateRY(y), 1)
    exp = OptimizationExperiment(
        circuit=c,
        initparams={x: 0.1, y: 0.2},
        optimizer="CG",
        label="eval",
        maxiters=None,
        zregister=0,
    )
    new_exp = exp.change_parameters({x: 0.5})
    assert new_exp.get_param(x) == 0.5
    assert new_exp.get_param(y) == 0.2


def test_change_list_of_parameters():
    x, y = symbols("x y")
    c = Circuit().push(GateRX(x), 0).push(GateRY(y), 1)
    exp = OptimizationExperiment(
        circuit=c,
        initparams={x: 1.0, y: 2.0},
        optimizer="CG",
        label="changelistofparameters",
        maxiters=None,
        zregister=0,
    )
    updated = exp.change_list_of_parameters([3.0, 4.0])
    assert updated.get_param(x) == 3.0
    assert updated.get_param(y) == 4.0


def test_proto_roundtrip(tmp_path):
    x = symbols("x")
    c = Circuit().push(GateRX(x), 0)
    exp = OptimizationExperiment(
        circuit=c,
        initparams={x: 1.0},
        optimizer="CG",
        label="proto",
        maxiters=10,
        zregister=0,
    )
    path = tmp_path / "experiment.pb"
    exp.saveproto(str(path))
    loaded = OptimizationExperiment.loadproto(str(path))
    assert isinstance(loaded, OptimizationExperiment)
    assert loaded.get_param(x) == 1.0


# Test OptimizationRun
def test_optimization_run_basic():
    run = OptimizationRun(
        cost=1.23,
        parameters={"x": 0.1, "y": 0.2},
        results=QCSResults(),
    )
    assert run.get_cost() == 1.23
    assert run.get_param("x") == 0.1
    assert isinstance(run.get_resultofbest(), QCSResults)


def test_run_proto_roundtrip(tmp_path):
    x = symbols("x")
    result = QCSResults("", "")
    run = OptimizationRun(0.42, {x: 0.5}, result)
    path = tmp_path / "run.pb"
    run.saveproto(str(path))
    loaded = OptimizationRun.loadproto(str(path))

    assert isinstance(loaded, OptimizationRun)
    assert loaded.get_param(x) == 0.5
    assert loaded.get_params() == {x: 0.5}
    assert isinstance(loaded.get_resultofbest(), QCSResults)


# Test OptimizationResults
def test_optimization_results():
    x = symbols("x")
    r1 = OptimizationRun(0.5, {x: 1.0}, QCSResults("", ""))
    r2 = OptimizationRun(0.3, {x: 0.8}, QCSResults("", ""))
    results = OptimizationResults(best=r2, history=[r1, r2])

    assert results.get_best().get_cost() == 0.3
    all_results = results.get_resultsofhistory()
    assert len(all_results) == 2
    assert isinstance(all_results[0], QCSResults)
    assert results.get_best().get_param(x) == 0.8


def test_results_proto_roundtrip(tmp_path):
    x = symbols("x")
    run = OptimizationRun(0.1, {x: 0.2}, QCSResults("", ""))
    res = OptimizationResults(best=run, history=[run])
    path = tmp_path / "results.pb"
    res.saveproto(str(path))
    loaded = OptimizationResults.loadproto(str(path))
    assert isinstance(loaded, OptimizationResults)
    assert loaded.get_best().get_cost() == 0.1
