import mimiqcircuits as mc


def test_qcsresults_amplitude_keys_roundtrip(tmp_path):
    # Keys that are not palindromes, so a bit-order mistake shows up.
    amplitudes = {
        mc.BitString("1000000011"): 0.5 + 0.25j,
        mc.BitString("0110"): -1j,
        mc.BitString("1" + "0" * 64): 0.125,
    }
    cstates = [mc.BitString("1000000011"), mc.BitString("0000000001")]
    res = mc.QCSResults(
        "sim", "1", [1.0], [0.0], cstates, [], amplitudes=amplitudes
    )

    fname = tmp_path / "res.pb"
    res.saveproto(str(fname))
    loaded = mc.QCSResults.loadproto(str(fname))

    assert loaded.amplitudes == amplitudes
    assert loaded.cstates == cstates
