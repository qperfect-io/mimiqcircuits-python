import pytest
import numpy as np
import symengine as se
import sympy as sp
from mimiqcircuits import *

import mimiqcircuits as mc


# Define helper functions if needed
def randunitary(dim):
    # Generates a random unitary matrix using sympy
    A = sp.randMatrix(dim, dim)
    Q, R = sp.Matrix(A).QRdecomposition()
    return np.array(Q).astype(np.complex128)


def is_close(matrix1, matrix2, tol=1e-8):
    matrix1 = np.array(matrix1.tolist(), dtype=np.complex128)
    matrix2 = np.array(matrix2.tolist(), dtype=np.complex128)
    diff_matrix = np.abs(matrix1 - matrix2)
    max_diff = np.max(diff_matrix)
    return max_diff < tol


# Test noise channel definitions
def test_noises_definition():
    assert hasattr(mc, "krauschannel")
    assert hasattr(mc, "Kraus")
    assert hasattr(mc, "MixedUnitary")
    assert hasattr(mc, "PauliNoise")
    assert hasattr(mc, "AmplitudeDamping")
    assert hasattr(mc, "GeneralizedAmplitudeDamping")
    assert hasattr(mc, "PhaseAmplitudeDamping")
    assert hasattr(mc, "ThermalNoise")
    assert hasattr(mc, "Depolarizing")


# Test mixed unitary assignment
def test_mixed_unitary_assignment():
    assert not mc.krauschannel.ismixedunitary()
    assert not mc.Kraus.ismixedunitary()
    assert not mc.AmplitudeDamping.ismixedunitary()
    assert not mc.GeneralizedAmplitudeDamping.ismixedunitary()
    assert not mc.PhaseAmplitudeDamping.ismixedunitary()
    assert not mc.ThermalNoise.ismixedunitary()

    assert mc.MixedUnitary.ismixedunitary()
    assert mc.PauliNoise.ismixedunitary()
    assert mc.Depolarizing.ismixedunitary()


# Test Kraus channel
def test_kraus():
    p = np.random.rand()
    Emats = [
        np.array([[1, 0], [0, np.sqrt(1 - p)]]),
        np.array([[0, np.sqrt(p)], [0, 0]]),
    ]
    kch = Kraus(Emats)

    for i, Ek in enumerate(kch.krausmatrices()):
        assert is_close(Ek, Emats[i])

    # Test wrong Kraus: not normalized and check regex pattern match
    Emats = [
        np.array([[1, 0], [0, np.sqrt(1 - p)]]),
        np.array([[0, np.sqrt(p)], [1, 0]]),
    ]
    with pytest.raises(ValueError, match=r"List of Kraus matrices should fulfill"):
        Kraus(Emats)

    # Test wrong Kraus: dimension and check regex pattern match
    Emats = [
        np.array([[1, 0, 0], [0, np.sqrt(1 - p), 0], [0, 0, 1]]),
        np.array([[0, np.sqrt(p), 0], [0, 0, 0], [0, 0, 0]]),
    ]
    with pytest.raises(
        ValueError, match=r"Dimension of operator has to be 2\^n with n>=1"
    ):
        Kraus(Emats)


# Test Mixed Unitary channel
def test_mixed_unitary():
    probs2, probs4 = np.random.rand(2), np.random.rand(4)
    probs2 /= np.sum(probs2)
    probs4 /= np.sum(probs4)

    Umats2 = [randunitary(2), randunitary(2)]
    much2 = MixedUnitary(probs2.tolist(), Umats2)

    Umats4 = [randunitary(4) for _ in range(4)]
    much4 = MixedUnitary(probs4.tolist(), Umats4)

    for i, Uk in enumerate(much2.unitarymatrices()):
        if hasattr(Uk, "matrix"):
            Uk = np.array(Uk.matrix.tolist(), dtype=np.complex128)
        assert is_close(Uk, Umats2[i])

    for i, Uk in enumerate(much4.unitarymatrices()):
        if hasattr(Uk, "matrix"):
            Uk = np.array(Uk.matrix.tolist(), dtype=np.complex128)
        assert is_close(Uk, Umats4[i])

    # Test wrong MixedUnitary: not unitary
    Umats_wrong = [randunitary(2), np.array([[1, 0], [1, -1]])]
    with pytest.raises(ValueError):
        MixedUnitary(probs2.tolist(), Umats_wrong)

    # Test wrong MixedUnitary: not normalized
    probs_wrong = probs2 * 0.5
    with pytest.raises(ValueError):
        MixedUnitary(probs_wrong.tolist(), Umats2)

    # Test wrong MixedUnitary: dimension
    Umats_wrong = [
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]),
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
    ]
    with pytest.raises(ValueError):
        MixedUnitary(probs2.tolist(), Umats_wrong)


# Test PauliNoise channel
def test_pauli_noise():
    probs2, probs4 = np.random.rand(2), np.random.rand(4)
    probs2 /= np.sum(probs2)
    probs4 /= np.sum(probs4)

    ops_str = ["I", "X"]
    pauliN = PauliNoise(probs2.tolist(), ops_str)
    for i, Uk in enumerate(pauliN.unitarymatrices()):
        assert is_close(Uk, mc.PauliString(ops_str[i])._matrix())

    ops_str = ["II", "XX", "YY", "ZX"]
    pauliN = PauliNoise(probs4.tolist(), ops_str)
    for i, Uk in enumerate(pauliN.unitarymatrices()):
        assert is_close(Uk, mc.PauliString(ops_str[i])._matrix())

    # Test wrong PauliNoise: non-Pauli
    ops_str = ["I", "K"]
    with pytest.raises(
        ValueError, match=r"Pauli string can only contain I, X, Y, or Z"
    ):
        PauliNoise(probs2.tolist(), ops_str)

    # Test wrong PauliNoise: dimensions
    ops_str = ["I", "XX"]
    with pytest.raises(
        ValueError, match=r"All Pauli strings must be of the same length."
    ):
        PauliNoise(probs2.tolist(), ops_str)

    # Test wrong PauliNoise: non-matching lengths
    ops_str = ["II", "XX"]
    with pytest.raises(
        ValueError,
        match=r"Lists of probabilities and Pauli strings must have the same length.",
    ):
        PauliNoise(probs4.tolist(), ops_str)


# Tests for PauliX class
def test_pauli_x():
    p = np.random.rand()
    px = PauliX(p)

    E1 = mc.GateID().matrix() * np.sqrt(1 - p)
    E2 = mc.GateX().matrix() * np.sqrt(p)

    for i, Ek in enumerate(px.krausmatrices()):
        if i == 0:
            assert is_close(Ek, E1)
        elif i == 1:
            assert is_close(Ek, E2)

    # Test invalid probability values
    with pytest.raises(ValueError, match=r"Probability should be between 0 and 1\."):
        PauliX(1.1)
    with pytest.raises(ValueError, match=r"Probability should be between 0 and 1\."):
        PauliX(-0.1)


# Tests for PauliY class
def test_pauli_y():
    p = np.random.rand()
    py = PauliY(p)

    E1 = mc.GateID().matrix() * np.sqrt(1 - p)
    E2 = mc.GateY().matrix() * np.sqrt(p)

    for i, Ek in enumerate(py.krausmatrices()):
        if i == 0:
            assert is_close(Ek, E1)
        elif i == 1:
            assert is_close(Ek, E2)

    # Test invalid probability values
    with pytest.raises(ValueError, match=r"Probability should be between 0 and 1\."):
        PauliY(1.1)
    with pytest.raises(ValueError, match=r"Probability should be between 0 and 1\."):
        PauliY(-0.1)


# Tests for PauliZ class
def test_pauli_z():
    p = np.random.rand()
    pz = PauliZ(p)

    E1 = mc.GateID().matrix() * np.sqrt(1 - p)
    E2 = mc.GateZ().matrix() * np.sqrt(p)

    for i, Ek in enumerate(pz.krausmatrices()):
        if i == 0:
            assert is_close(Ek, E1)
        elif i == 1:
            assert is_close(Ek, E2)

    # Test invalid probability values
    with pytest.raises(ValueError, match=r"Probability should be between 0 and 1\."):
        PauliZ(1.1)
    with pytest.raises(ValueError, match=r"Probability should be between 0 and 1\."):
        PauliZ(-0.1)


# Test Depolarizing channel
def test_depolarizing():
    depol = Depolarizing(1, 0.5)
    umats = depol.unitarymatrices()

    pmats = [
        mc.GateID().matrix(),
        mc.GateX().matrix(),
        mc.GateY().matrix(),
        mc.GateZ().matrix(),
    ]
    for i in range(len(umats)):
        assert is_close(umats[i], pmats[i])

    # N=2
    depol = Depolarizing(2, 0.3)
    umats = depol.unitarymatrices()

    pmats2 = [np.kron(M1, M2) for M1 in pmats for M2 in pmats]
    for i in range(len(umats)):
        assert is_close(umats[i], pmats2[i])

    # Test wrong Depolarizing: p value
    with pytest.raises(ValueError, match="Probability p needs to be between 0 and 1."):
        Depolarizing(1, 2)
    with pytest.raises(ValueError, match="Probability p needs to be between 0 and 1."):
        Depolarizing(1, -0.1)


def test_amplitude_damping():
    gamma = np.random.rand()
    ad = AmplitudeDamping(gamma)
    E1 = mc.DiagonalOp(float(1), float(np.sqrt(1 - gamma))).matrix()
    E2 = mc.SigmaMinus(float(np.sqrt(gamma))).matrix()

    for i, Ek in enumerate(ad.krausmatrices()):
        if i == 0:
            assert is_close(Ek, E1)
        elif i == 1:
            assert is_close(Ek, E2)

    # Test wrong gamma value
    with pytest.raises(ValueError, match=r"Value of gamma must be between 0 and 1\."):
        AmplitudeDamping(1.1)

    with pytest.raises(ValueError, match=r"Value of gamma must be between 0 and 1\."):
        AmplitudeDamping(-0.1)


# Test Generalized Amplitude Damping channel
def test_generalized_amplitude_damping():
    p = np.random.rand()
    gamma = np.random.rand()
    gad = GeneralizedAmplitudeDamping(p, gamma)
    E1 = mc.DiagonalOp(
        float(np.sqrt(p)), float(np.sqrt(p) * np.sqrt(1 - gamma))
    ).matrix()
    E2 = mc.DiagonalOp(
        float(np.sqrt(1 - p) * np.sqrt(1 - gamma)), float(np.sqrt(1 - p))
    ).matrix()
    E3 = mc.SigmaMinus(float(np.sqrt(p) * np.sqrt(gamma))).matrix()
    E4 = mc.SigmaPlus(float(np.sqrt(1 - p) * np.sqrt(gamma))).matrix()

    for i, Ek in enumerate(gad.krausmatrices()):
        if i == 0:
            assert is_close(Ek, E1)
        elif i == 1:
            assert is_close(Ek, E2)
        elif i == 2:
            assert is_close(Ek, E3)
        elif i == 3:
            assert is_close(Ek, E4)

    # Test wrong p value
    with pytest.raises(ValueError, match=r"Value of p must be between 0 and 1\."):
        GeneralizedAmplitudeDamping(1.1, gamma)

    with pytest.raises(ValueError, match=r"Value of p must be between 0 and 1\."):
        GeneralizedAmplitudeDamping(-0.1, gamma)

    # Test wrong gamma value
    with pytest.raises(ValueError, match=r"Value of gamma must be between 0 and 1\."):
        GeneralizedAmplitudeDamping(p, 1.1)

    with pytest.raises(ValueError, match=r"Value of gamma must be between 0 and 1\."):
        GeneralizedAmplitudeDamping(p, -0.1)


def test_phase_amplitude_damping():
    p = 0.5
    gamma = 0.5
    beta = 0.2
    pad = PhaseAmplitudeDamping(p, gamma, beta)

    K = np.sqrt(1 - gamma) * (1 - 2 * beta) / (1 - gamma * p)
    pref1 = np.sqrt(1 - gamma * p)
    pref2 = np.sqrt(1 - gamma * (1 - p) - (1 - gamma * p) * K**2)
    pref3 = np.sqrt(gamma * p)
    pref4 = np.sqrt(gamma * (1 - p))

    E1 = mc.DiagonalOp(pref1 * K, pref1).matrix()
    E2 = mc.Projector0(pref2).matrix()
    E3 = mc.SigmaMinus(pref3).matrix()
    E4 = mc.SigmaPlus(pref4).matrix()

    for i, Ek in enumerate(pad.krausmatrices()):
        if i == 0:
            assert is_close(Ek, E1)
        elif i == 1:
            assert is_close(Ek, E2)
        elif i == 2:
            assert is_close(Ek, E3)
        elif i == 3:
            assert is_close(Ek, E4)

    # Test wrong parameter values
    with pytest.raises(ValueError, match=r"Value of p must be between 0 and 1\."):
        PhaseAmplitudeDamping(1.1, gamma, beta)
    with pytest.raises(ValueError, match=r"Value of p must be between 0 and 1\."):
        PhaseAmplitudeDamping(-0.1, gamma, beta)
    with pytest.raises(ValueError, match=r"Value of gamma must be between 0 and 1\."):
        PhaseAmplitudeDamping(p, 1.1, beta)
    with pytest.raises(ValueError, match=r"Value of gamma must be between 0 and 1\."):
        PhaseAmplitudeDamping(p, -0.1, beta)
    with pytest.raises(ValueError, match=r"Value of beta must be between 0 and 1\."):
        PhaseAmplitudeDamping(p, gamma, 1.1)
    with pytest.raises(ValueError, match=r"Value of beta must be between 0 and 1\."):
        PhaseAmplitudeDamping(p, gamma, -0.1)


def test_thermal_noise():
    T1 = 2.0
    T2 = 1.0
    time = 1.0
    ne = 0.1
    tn = ThermalNoise(T1, T2, time, ne)

    Gamma1 = 1 / T1
    Gamma2 = 1 / T2

    p = 1 - ne
    gamma = 1 - np.exp(-Gamma1 * time)
    beta = 0.5 * (1 - np.exp(-(Gamma2 - Gamma1 / 2) * time))

    pad = PhaseAmplitudeDamping(p, gamma, beta)
    kmats_pad = pad.krausmatrices()
    kmats_tn = tn.krausmatrices()

    for i in range(len(kmats_pad)):
        assert is_close(kmats_pad[i], kmats_tn[i])

    # Test wrong parameter values
    with pytest.raises(ValueError, match=r"Value of T1 must be >= 0\."):
        ThermalNoise(-0.1, T2, time, ne)
    with pytest.raises(ValueError, match=r"Value of T2 must be <= 2 \* T1\."):
        ThermalNoise(T1, 2 * T1 + 0.1, time, ne)
    with pytest.raises(ValueError, match=r"Value of time must be >= 0\."):
        ThermalNoise(T1, T2, -0.1, ne)
    with pytest.raises(ValueError, match=r"Value of ne must be between 0 and 1\."):
        ThermalNoise(T1, T2, time, 1.1)
    with pytest.raises(ValueError, match=r"Value of ne must be between 0 and 1\."):
        ThermalNoise(T1, T2, time, -0.1)
