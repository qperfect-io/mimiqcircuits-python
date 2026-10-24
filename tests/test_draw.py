from mimiqcircuits import Circuit, GateCX, GateH, Measure


def test_draw_packs_independent_instructions(capsys):
    c = Circuit()
    c.push(GateH(), range(3))
    c.push(Measure(), range(2), range(2))
    c.draw()
    lines = capsys.readouterr().out.splitlines()

    def columns(gate):
        return [line.index(gate) for line in lines if gate in line]

    # the three H gates act on different qubits, so they share one column
    hcols = columns("┤H├")
    assert len(hcols) == 3 and len(set(hcols)) == 1

    # both measurements write to c, so they are drawn one after the other
    mcols = columns("┤M├")
    assert len(mcols) == 2 and len(set(mcols)) == 2


def test_draw_aligns_broadcast_runs(capsys):
    c = Circuit()
    c.push(GateH(), range(4))
    c.push(GateCX(), range(0, 8, 2), range(1, 8, 2))
    c.draw()
    lines = capsys.readouterr().out.splitlines()

    # one push of CX gates is drawn as one layer, although q[4]..q[7] are free
    # in the first column
    xcols = [line.index("┤X├") for line in lines if "┤X├" in line]
    assert len(xcols) == 4 and len(set(xcols)) == 1


def test_draw_sequential(capsys):
    c = Circuit()
    c.push(GateH(), range(3))
    c.draw(compact=False)
    lines = capsys.readouterr().out.splitlines()

    # every instruction gets its own column
    hcols = [line.index("┤H├") for line in lines if "┤H├" in line]
    assert len(hcols) == 3 and len(set(hcols)) == 3
