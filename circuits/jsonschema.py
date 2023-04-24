import json
from .quantumcircuit import CircuitGate, Circuit
from .gates import *


import inspect

CIRCUIT_SCHEMA = {
    "type": "object",
    "properties": {
        "num_qubits": {
            "type": "integer"
        },
        "gates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "gate_name": {
                        "type": "string"
                    },
                    "qubits": {
                        "type": "array",
                        "items": {
                            "type": "integer"
                        }
                    },
                    "params": {
                        "oneOf": [
                            {"type": "number"},
                            {"type": "array", "items": {"type": "number"}},
                            {"type": "string"}
                        ]
                    }
                },
                "required": ["gate_name", "qubits"]
            }
        }
    },
    "required": ["num_qubits", "gates"]
}


def parnames(gate):
    gate_class = gate.__class__
    if issubclass(gate_class, type) and issubclass(gate_class, object):
        return {}
    sig = inspect.signature(gate_class.__init__)
    return {p: sig.parameters[p].default for p in sig.parameters if p != 'self'}


def tojson(circuit):
    gates = []

    for g in circuit:
        if isinstance(g.gate, GateCustom):
            gate_dict = {'gate_name': str(g.gate), 'qubits': list(
                g.qubits), 'matrix': g.gate.matrix.tolist()}
        elif isinstance(g.gate, Gate):
            gate_dict = {'gate_name': g.gate.__class__.__name__,
                         'qubits': list(g.qubits)}
            parnames_dict = parnames(g.gate)
            params = []
            for p in parnames_dict:
                if p in parnames_dict:
                    params.append(getattr(g.gate, p))
                else:
                    params.append(parnames_dict[p])
            gate_dict["params"] = params
        gates.append(gate_dict)
    data = {"num_qubits": len(
        set(qubit for gate in circuit for qubit in gate.qubits)), 'gates': gates}

    return json.dumps(data)


def fromjson(json_str):
    data = json.loads(json_str)
    gates = []
    gate_classes = {"GateH": GateH, "GateX": GateX, "GateY": GateY, "GateZ": GateZ, "GateR": GateR,
                    "GateU3": GateU3, "GateU": GateU, "GateU1": GateU1, "GateU2": GateU2, "GateCX": GateCX,
                    "GateCY": GateCY, "GateCZ": GateCZ, "GateCCX": GateCCX, "GateSWAP": GateSWAP,
                    "GateISWAPDG": GateISWAP, "GateISWAP": GateISWAPDG, "GateP": GateP, "GateCP": GateCP,
                    "GateS": GateS, "GateCU": GateCU, "GateSDG": GateSDG, "GateT": GateT, "GateTDG": GateTDG,
                    "GateSXDG": GateSXDG, "GateSX": GateSX, "GateID": GateID, "GateCSWAP": GateCSWAP, "GateCH": GateCH,
                    "GateCX": GateCX, "GateCY": GateCY, "GateCZ": GateCZ, "GateRX": GateRX, "GateRY": GateRY, "GateRZ": GateRZ,
                    "GateCRX": GateCRX, "GateCRY": GateCRY, "GateCRZ": GateCRZ, "Custom": GateCustom}

    for gate in data["gates"]:
        gate_name = gate.pop("gate_name")
        gate_class = gate_classes[gate_name]
        targets = gate.pop("qubits")
        if gate_class == GateCustom:
            gate_obj = GateCustom.from_dict(gate)
        else:
            gate_params = gate.pop("params", [])
            gate_obj = gate_class(*gate_params)
        gates.append(CircuitGate(gate_obj, *tuple(targets)))
    return Circuit(gates)
