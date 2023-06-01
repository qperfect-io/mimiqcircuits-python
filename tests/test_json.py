import mimiqcircuits as mc
import random


def test_allgates():
    c = mc.Circuit()
    print(c)

    simple_1q = [mc.GateX, mc.GateY, mc.GateZ, mc.GateH, mc.GateS,
                 mc.GateSDG, mc.GateTDG, mc.GateSX, mc.GateSXDG, mc.GateID]

    for gatetype in simple_1q:
        for n in range(0, 8):
            c.add_gate(gatetype(), n)

    json = c.to_json()
    c2_simple_1q = mc.Circuit.from_json(json)

    assert c == c2_simple_1q

    params_1q_3p = [mc.GateU]
    params_1q_2p = [mc.GateR]
    params_1q_1p = [mc.GateP, mc.GateRX, mc.GateRY, mc.GateRZ]

    for gatetype in params_1q_3p:
        for n in range(0, 8):
            c.add_gate(gatetype(*[random.random() for r in range(0, 3)]), n)

    for gatetype in params_1q_2p:
        for n in range(0, 8):
            c.add_gate(gatetype(*[random.random() for r in range(0, 2)]), n)

    for gatetype in params_1q_1p:
        for n in range(0, 8):
            c.add_gate(gatetype(random.random()), n)

    json = c.to_json()
    c2_params_1q = mc.Circuit.from_json(json)

    assert c == c2_params_1q

    simple_2q = [mc.GateCX, mc.GateCY, mc.GateCZ, mc.GateCH,
                 mc.GateSWAP, mc.GateISWAP, mc.GateISWAPDG]

    for gatetype in simple_2q:
        for n in [(0, 7), (0, 3), (2, 4), (7, 1), (7, 0), (4, 2)]:
            c.add_gate(gatetype(), *n)

    json = c.to_json()
    c2_simple_2q = mc.Circuit.from_json(json)

    assert c == c2_simple_2q

    params_2q_4p = [mc.GateCU]
    params_2q_2p = [mc.GateCR]
    params_2q_1p = [mc.GateCP, mc.GateCRX, mc.GateCRY, mc.GateCRZ]

    for gatetype in params_2q_4p:
        for n in [(0, 7), (0, 3), (2, 4), (7, 1), (7, 0), (4, 2)]:
            c.add_gate(gatetype(*[random.random() for r in range(0, 4)]), *n)

    for gatetype in params_2q_2p:
        for n in [(0, 7), (0, 3), (2, 4), (7, 1), (7, 0), (4, 2)]:
            c.add_gate(gatetype(*[random.random() for r in range(0, 2)]), *n)

    for gatetype in params_2q_1p:
        for n in [(0, 7), (0, 3), (2, 4), (7, 1), (7, 0), (4, 2)]:
            c.add_gate(gatetype(random.random()), *n)

    json = c.to_json()
    c2_params_2q = mc.Circuit.from_json(json)

    assert c == c2_params_2q


def test_barrier():
    c = mc.Circuit()
    c.add_gate(mc.GateX(), 0)
    c.add_gate(mc.GateCX(), 0, 1)
    c.add_barrier(0, 1)
    c.add_gate(mc.GateCX(), 1, 0)
    c.add_barrier(1, 0)
    c.add_gate(mc.GateCX(), 0, 1)

    json = c.to_json()
    c_barrier = mc.Circuit.from_json(json)

    assert c == c_barrier
