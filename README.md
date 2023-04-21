# Circuits (`circuits-python`)

A library to handle quantum circuits for **QPerfect**'s MIMIQ Emulator.

## Installation
pip install circuits or pip install .

## Usage

```python

from circuits import *
from circuits.quantumcircuit import *
from circuits.gates import *
from circuits.jsonschema import *

circuit=Circuit()

circuit.add_gate(GateH(),0)

circuit.add_gate(GateCH(),0,2)

circuit.add_circuitgate(CircuitGate(GateCSWAP(), 0,1,2))

import numpy as np

matrix = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0]])

circuit.add_circuitgate(CircuitGate(GateCustom(matrix), 1,2))
circuit.add_gate(GateCustom(matrix), 3,4)

print(circuit)

print("......................................................")

json_str = jsonschema.tojson(circuit)
print(json_str)

json_str2 = jsonschema.fromjson(json_str)
print(json_str2)

```

# COPYRIGHT

Copyright © 2023 University of Strasbourg. All Rights Reserved.

# AUTHORS

See the [AUTHORS.md](AUTHORS.md) file for the list of authors.
