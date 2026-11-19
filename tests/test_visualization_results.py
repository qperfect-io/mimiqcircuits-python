import pytest

# matplotlib is an optional dependency.
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from mimiqcircuits import BitString, QCSResults
from mimiqcircuits.visualization.results import plothistogram


def _results(fidelities):
    return QCSResults(
        simulator="sim",
        version="1",
        fidelities=fidelities,
        avggateerrors=[],
        cstates=[BitString("100"), BitString("100"), BitString("010")],
    )


def _texts(fig):
    return [t.get_text() for t in fig.axes[0].texts]


@pytest.mark.parametrize("mimiqstyle", [True, False])
def test_nobitstrings_hides_the_bits(mimiqstyle, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    fig = plothistogram(_results([1.0]), mimiqstyle=mimiqstyle, nobitstrings=True)
    assert _texts(fig) == []


def test_empty_fidelities_are_allowed():
    fig = plothistogram(_results([]), mimiqstyle=False)
    assert "fidelity" not in fig.axes[0].get_title()


def test_bar_labels_match_tointeger():
    fig = plothistogram(_results([1.0]), mimiqstyle=False)
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert labels[0] == hex(BitString("100").tointeger())


def test_bar_labels_match_the_notebook_table():
    import re

    res = _results([1.0])
    fig = plothistogram(res, mimiqstyle=False)
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert labels == re.findall(r">(0x[0-9a-f]+)<", res._repr_html_())
