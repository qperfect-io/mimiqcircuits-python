import pytest
import mimiqcircuits as mc
import numpy as np
from random import random


# Helper function to check if two circuits are equivalent
def circuits_are_equivalent(circuit1, circuit2):
    # Ensure both circuits have the same number of instructions
    if len(circuit1.instructions) != len(circuit2.instructions):
        return False

    # Iterate through the instructions and compare
    for inst1, inst2 in zip(circuit1.instructions, circuit2.instructions):
        if (
            inst1.qubits != inst2.qubits
            or inst1.bits != inst2.bits
            or inst1.zvars != inst2.zvars
        ):
            return False

        # Check if the instruction types are the same
        if type(inst1) != type(inst2):
            return False
    return True


# Base test function for noise channels
def base_noise_channel_test(circuit, noise_channel, tmpdir):
    temp_file = tmpdir.join("test.pb")
    temp_filename = str(temp_file)

    circuit.push(noise_channel, 1)
    circuit.saveproto(temp_filename)

    loaded_circuit = mc.Circuit.loadproto(temp_filename)

    # Check if the loaded circuit has the same instructions as the original circuit
    assert circuits_are_equivalent(circuit, loaded_circuit)


# Test Cases


def test_phase_amplitude_damping_proto(tmpdir):
    circuit = mc.Circuit()
    pad = mc.PhaseAmplitudeDamping(1, 1, 1)
    base_noise_channel_test(circuit, pad, tmpdir)


def test_amplitude_damping_proto(tmpdir):
    circuit = mc.Circuit()
    ad = mc.AmplitudeDamping(0.5)
    base_noise_channel_test(circuit, ad, tmpdir)


def test_generalized_amplitude_damping_proto(tmpdir):
    circuit = mc.Circuit()
    gad = mc.GeneralizedAmplitudeDamping(0.5, 0.3)
    base_noise_channel_test(circuit, gad, tmpdir)


def test_thermal_noise_proto(tmpdir):
    circuit = mc.Circuit()
    tn = mc.ThermalNoise(2.0, 1.0, 1.0, 0.1)
    base_noise_channel_test(circuit, tn, tmpdir)


def test_pauli_x_proto(tmpdir):
    circuit = mc.Circuit()
    px = mc.PauliX(0.5)
    base_noise_channel_test(circuit, px, tmpdir)


def test_pauli_y_proto(tmpdir):
    circuit = mc.Circuit()
    py = mc.PauliY(0.5)
    base_noise_channel_test(circuit, py, tmpdir)


def test_pauli_z_proto(tmpdir):
    circuit = mc.Circuit()
    pz = mc.PauliZ(0.5)
    base_noise_channel_test(circuit, pz, tmpdir)


def test_pauli_noise_proto(tmpdir):
    circuit = mc.Circuit()
    pn = mc.PauliNoise([0.25, 0.25, 0.25, 0.25], ["I", "X", "Y", "Z"])
    base_noise_channel_test(circuit, pn, tmpdir)


def test_mixed_unitary_proto(tmpdir):
    circuit = mc.Circuit()
    # Define complex matrices for MixedUnitary
    unitary1 = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    unitary2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    mu = mc.MixedUnitary([0.5, 0.5], [unitary1, unitary2])
    base_noise_channel_test(circuit, mu, tmpdir)


def test_kraus_proto(tmpdir):
    circuit = mc.Circuit()
    # Define normalized complex matrices for Kraus operators
    E1 = np.array([[1, 0], [0, np.sqrt(0.5)]], dtype=np.complex128)
    E2 = np.array([[0, np.sqrt(0.5)], [0, 0]], dtype=np.complex128)
    kraus = mc.Kraus([E1, E2])
    base_noise_channel_test(circuit, kraus, tmpdir)

# Test Cases for gates and operations
def base_gate_operation_test(circuit, operation, tmpdir, *args):
    temp_file = tmpdir.join("test.pb")
    temp_filename = str(temp_file)

    circuit.push(operation, *args)
    circuit.saveproto(temp_filename)

    loaded_circuit = mc.Circuit.loadproto(temp_filename)

    assert circuits_are_equivalent(circuit, loaded_circuit)


def test_cnot_proto(tmpdir):
    circuit = mc.Circuit()
    cnot = mc.GateCX()
    base_gate_operation_test(circuit, cnot, tmpdir, 0, 1)

def test_cz_proto(tmpdir):
    circuit = mc.Circuit()
    cz = mc.GateCZ()
    base_gate_operation_test(circuit, cz, tmpdir, 0, 1)

def test_swap_proto(tmpdir):
    circuit = mc.Circuit()
    swap = mc.GateSWAP()
    base_gate_operation_test(circuit, swap, tmpdir, 0, 1)

def test_gate_u_proto(tmpdir):
    circuit = mc.Circuit()
    gate_u = mc.GateU(0.5, 0.3, 0.1)
    base_gate_operation_test(circuit, gate_u, tmpdir, 0)

def test_gate_custom_proto(tmpdir):
    circuit = mc.Circuit()
    custom_matrix = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    gate_custom = mc.GateCustom(custom_matrix)
    base_gate_operation_test(circuit, gate_custom, tmpdir, 0)

def test_gate_x_proto(tmpdir):
    circuit = mc.Circuit()
    gate_x = mc.GateX()
    base_gate_operation_test(circuit, gate_x, tmpdir, 0)

def test_gate_y_proto(tmpdir):
    circuit = mc.Circuit()
    gate_y = mc.GateY()
    base_gate_operation_test(circuit, gate_y, tmpdir, 0)

def test_gate_z_proto(tmpdir):
    circuit = mc.Circuit()
    gate_z = mc.GateZ()
    base_gate_operation_test(circuit, gate_z, tmpdir, 0)

def test_gate_h_proto(tmpdir):
    circuit = mc.Circuit()
    gate_h = mc.GateH()
    base_gate_operation_test(circuit, gate_h, tmpdir, 0)

def test_gate_s_proto(tmpdir):
    circuit = mc.Circuit()
    gate_s = mc.GateS()
    base_gate_operation_test(circuit, gate_s, tmpdir, 0)

def test_gate_s_proto(tmpdir):
    circuit = mc.Circuit()
    gate_s = mc.GateSDG()
    base_gate_operation_test(circuit, gate_s, tmpdir, 0)

def test_gate_t_proto(tmpdir):
    circuit = mc.Circuit()
    gate_t = mc.GateTDG()
    base_gate_operation_test(circuit, gate_t, tmpdir, 0)

def test_gate_t_proto(tmpdir):
    circuit = mc.Circuit()
    gate_t = mc.GateT()
    base_gate_operation_test(circuit, gate_t, tmpdir, 0)

def test_gate_rx_proto(tmpdir):
    circuit = mc.Circuit()
    gate_rx = mc.GateRX(np.pi / 4)
    base_gate_operation_test(circuit, gate_rx, tmpdir, 0)

def test_gate_ry_proto(tmpdir):
    circuit = mc.Circuit()
    gate_ry = mc.GateRY(np.pi / 4)
    base_gate_operation_test(circuit, gate_ry, tmpdir, 0)

def test_gate_rz_proto(tmpdir):
    circuit = mc.Circuit()
    gate_rz = mc.GateRZ(random())
    base_gate_operation_test(circuit, gate_rz, tmpdir, 0)

def test_gate_xxplusyy_proto(tmpdir):
    circuit = mc.Circuit()
    gate_xxplusyy = mc.GateXXplusYY(np.pi / 4, random())
    base_gate_operation_test(circuit, gate_xxplusyy, tmpdir, 0, 1)

def test_gate_xxminusyy_proto(tmpdir):
    circuit = mc.Circuit()
    gate_xxplusyy = mc.GateXXminusYY(np.pi / 4, random())
    base_gate_operation_test(circuit, gate_xxplusyy, tmpdir, 0, 1)

def test_gate_Control_proto(tmpdir):
    circuit = mc.Circuit()
    Control = mc.Control(3, mc.GateR(np.pi / 4, random()))
    base_gate_operation_test(circuit, Control, tmpdir, 0, 1, 2, 3)

def test_Amplitude_proto(tmpdir):
    circuit = mc.Circuit()
    amp = mc.Amplitude(mc.BitString("01"))
    base_gate_operation_test(circuit, amp, tmpdir, 0)

def test_ExpectationValue_control_proto(tmpdir):
    circuit = mc.Circuit()
    exp = mc.ExpectationValue(mc.Control(1, mc.Power(mc.GateX(), 1/2)))
    base_gate_operation_test(circuit, exp, tmpdir, 0, 1, 0)

def test_gate_ExpectationValue_proto(tmpdir):
    circuit = mc.Circuit()
    exp = mc.ExpectationValue(mc.PauliString("XYZI"))
    base_gate_operation_test(circuit, exp, tmpdir, 0, 1, 2, 3, 0)

def test_gate_BondDim_proto(tmpdir):
    circuit = mc.Circuit()
    exp = mc.BondDim()
    base_gate_operation_test(circuit, exp, tmpdir, 0, 0)

def test_gate_SchmidtRank_proto(tmpdir):
    circuit = mc.Circuit()
    sch = mc.SchmidtRank()
    base_gate_operation_test(circuit, sch, tmpdir, 0, 0)

def test_VonNeumannEntropy_proto(tmpdir):
    circuit = mc.Circuit()
    von = mc.VonNeumannEntropy()
    base_gate_operation_test(circuit, von, tmpdir, 0, 0)

def test_IfStatement_proto(tmpdir):
    circuit = mc.Circuit()
    ifs = mc.IfStatement(mc.GateX(), mc.BitString('1'))
    base_gate_operation_test(circuit, ifs, tmpdir, 0, 0)

def test_MeasureX_proto(tmpdir):
    circuit = mc.Circuit()
    mx = mc.MeasureX()
    base_gate_operation_test(circuit, mx, tmpdir, 0, 0)

def test_MeasureY_proto(tmpdir):
    circuit = mc.Circuit()
    my = mc.MeasureY()
    base_gate_operation_test(circuit, my, tmpdir, 0, 0)

def test_MeasureZ_proto(tmpdir):
    circuit = mc.Circuit()
    mz = mc.MeasureZ()
    base_gate_operation_test(circuit, mz, tmpdir, 0, 0)

def test_Measure_proto(tmpdir):
    circuit = mc.Circuit()
    m = mc.Measure()
    base_gate_operation_test(circuit, m, tmpdir, 0, 0)

def test_MeasureResetX_proto(tmpdir):
    circuit = mc.Circuit()
    mrx = mc.MeasureResetX()
    base_gate_operation_test(circuit, mrx, tmpdir, 0, 0)

def test_MeasureResetZ_proto(tmpdir):
    circuit = mc.Circuit()
    mrz = mc.MeasureResetZ()
    base_gate_operation_test(circuit, mrz, tmpdir, 0, 0)

def test_gate_MeasureResetY_proto(tmpdir):
    circuit = mc.Circuit()
    mry = mc.MeasureResetY()
    base_gate_operation_test(circuit, mry, tmpdir, 0, 0)

def test_Reset_proto(tmpdir):
    circuit = mc.Circuit()
    res = mc.Reset()
    base_gate_operation_test(circuit, res, tmpdir, 0)

def test_gate_Power_proto(tmpdir):
    circuit = mc.Circuit()
    pow = mc.Power(mc.GateR(random(), random()), random())
    base_gate_operation_test(circuit, pow, tmpdir, 0)

def test_Inverse_proto(tmpdir):
    circuit = mc.Circuit()
    inv = mc.Inverse(mc.GateR(random(), random()))
    base_gate_operation_test(circuit, inv, tmpdir, 0)

def test_Inverse_p_proto(tmpdir):
    circuit = mc.Circuit()
    inv = mc.Inverse(mc.Parallel(2, mc.GateR(random(), random())))
    base_gate_operation_test(circuit, inv, tmpdir, 0, 1)

def test_gate_delay_proto(tmpdir):
    circuit = mc.Circuit()
    delay = mc.Delay()
    base_gate_operation_test(circuit, delay, tmpdir, 0)

def test_Not_proto(tmpdir):
    circuit = mc.Circuit()
    not_gate = mc.Not()
    base_gate_operation_test(circuit, not_gate, tmpdir, 0)