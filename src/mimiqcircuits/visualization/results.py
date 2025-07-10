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
from matplotlib import pyplot as plt
from mimiqcircuits.qcsresults import QCSResults
from mimiqcircuits.bitstrings import bitvec_to_int


# define colors for plots
QPERFECTION_COLORS = {
    "metallic_seaweed": "#0c7e8f",
    "vivid_tangelo": "#EC7016",
    "antique_fuchsia": "#A4598D",
    "bangladesh_green": "#006E51",
    "dark_brown_tangelo": "#96694A",
    "rhythm": "#7E6A98",
}

QPERFECT_COLORS = {
    "teal": QPERFECTION_COLORS["metallic_seaweed"],
    "orange": QPERFECTION_COLORS["vivid_tangelo"],
    "purple": QPERFECTION_COLORS["antique_fuchsia"],
    "green": QPERFECTION_COLORS["bangladesh_green"],
    "brown": QPERFECTION_COLORS["dark_brown_tangelo"],
    "violet": QPERFECTION_COLORS["rhythm"],
}

# The colorscheme is taken by https://github.com/joshdick/onedark.vim
# which is released under MIT license with the following notice:
# Copyright (c) 2015 Joshua Dick
MIMIQ_COLORS = {
    "black": "#282C34",
    "red": "#E06C75",
    "green": "#98C379",
    "yellow": "#E5C07B",
    "blue": "#61AFEF",
    "magenta": "#C678DD",
    "cyan": "#56B6C2",
    "white": "#ABB2BF",
}


def _plothist(outcomes, counts, title, nobitstrings=False):
    labels = [hex(bitvec_to_int(arr)) for arr in outcomes]
    nbars = len(counts)
    nsamples = sum(counts)

    fig = plt.figure()

    bars = plt.bar(labels, counts)

    # alternating colors for bars
    for i in range(len(bars)):
        color = "yellow" if i % 2 == 0 else "magenta"
        bars[i].set_color(color)

    if not nobitstrings:
        bottom = max(counts) * 0.02
        for i, outcome in enumerate(outcomes):
            plt.text(
                i,
                bottom,
                outcome.to01(),
                ha="center",
                rotation=90,
                color="black",
                size=6,
                family="monospace",
            )

    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.xlim(-1, nbars)
    plt.ylabel(f"counts ({nsamples} samples)")
    plt.title(title)
    plt.tight_layout()
    return fig


def plothistogram(results, num_outcomes=15, mimiqstyle=True, nobitstrings=False):
    """Plots the histogram of the obtained classical states' occurrences.

    Arguments:
        results: a QCSResults object.
        num_states (int): the maximum number of bitstrings plotted (default 15)

    Raises:
        TypeError: If a non QCSResults object is passed.
    """
    if not isinstance(results, QCSResults):
        raise TypeError("First argument is not a valid QCSResults object")

    hist = results.histogram()

    outcomes = sorted(hist, key=hist.get, reverse=True)[0:num_outcomes]
    counts = [hist[x] for x in outcomes]

    title = f"{results.simulator} {results.version} fidelity(min:{min(results.fidelities):.3f}, max:{max(results.fidelities):.3f})"

    if not mimiqstyle:
        return _plothist(outcomes, counts, title, nobitstrings=nobitstrings)

    with plt.style.context("mimiqcircuits.visualization.mimiq"):
        fig = _plothist(outcomes, counts, title)

    plt.show()
    return fig


__all__ = ["plothistogram"]
