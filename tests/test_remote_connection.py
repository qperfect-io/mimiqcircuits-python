"""Client-side behaviour of `MimiqConnection` that needs no server: argument
checks in `submit`, and the helpers built on `submit` and `get_result`."""

import pytest

from mimiqcircuits import Circuit, GateH, Measure, EquivalenceExperiment
from mimiqcircuits.remote import MimiqConnection, MAX_ENTDIM


def _offline_connection(monkeypatch):
    # Skip __init__: it would open a connection to the service.
    conn = MimiqConnection.__new__(MimiqConnection)
    conn.connection = None
    monkeypatch.setattr(conn, "_RemoteConnection__get_timelimit", lambda: 30, raising=False)
    return conn


def test_entdim_above_the_maximum_is_refused(monkeypatch):
    conn = _offline_connection(monkeypatch)
    c = Circuit().push(GateH(), 0).push(Measure(), 0, 0)

    with pytest.raises(ValueError, match="entdim"):
        conn.submit(c, algorithm="mps", entdim=MAX_ENTDIM + 1)


def test_check_equivalence_scores_the_first_result(monkeypatch):
    conn = _offline_connection(monkeypatch)
    monkeypatch.setattr(conn, "submit", lambda experiment, **kwargs: "job-1")
    monkeypatch.setattr(conn, "get_result", lambda job: ("result of", job))

    experiment = EquivalenceExperiment(Circuit().push(GateH(), 0), Circuit().push(GateH(), 0))
    monkeypatch.setattr(experiment, "interpret_results", lambda result: result)

    assert conn.check_equivalence(experiment) == ("result of", "job-1")
