# Circuits (`circuits-python`)

A library to handle quantum circuits for **QPerfect**'s MIMIQ Emulator.

## Installation
pip install circuits or pip install .

## Usage

```python

import circuits
from circuits import gates, quantumcircuit, test, jsonschema

circuit=quantumcircuit.Circuit()

circuit.add_gate(gates.GateH(),0)

circuit.add_gate(gates.GateCH(),0,2)

circuit.add_circuitgate(quantumcircuit.CircuitGate(gates.GateCSWAP(), 0,1,2))

import numpy as np

matrix = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0]])

circuit.add_circuitgate(quantumcircuit.CircuitGate(gates.GateCustom(matrix), 1,2))
circuit.add_gate(gates.GateCustom(matrix), 3,4)

print(circuit)

```

# COPYRIGHT

Copyright © 2023 University of Strasbourg. All Rights Reserved.

# AUTHORS

See the [AUTHORS.md](AUTHORS.md) file for the list of authors.
