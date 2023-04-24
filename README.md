# Python interface to MIMIQ Circuits (`mimiqcircuits`)

A library to handle quantum circuits for **QPerfect**'s MIMIQ Emulator.

## Installation

```
pip install "mimiqcircuits @ git+https://github.com/qperfect-io/circuits-python.git
````

## Usage

### Handling circuits

```python
import mimiqcircuits as mc

c = mc.Circuit()

print(c)
# will print:
# empty circuit

c.add_gate(mc.GateX(), 4)

print(c)
# will print:
#5-qubit circuit with 1 gates
# └── X @ q4

c.add_gate(mc.GateCX(), 1, 8)

print(c)
# will print:
# 9-qubit circuit with 2 gates
# ├── X @ q4
# └── CX @ q1, q8

import json

json.dumps(c.to_json())
# will give:
# '{"gates": [{"name": "X", "targets": [5]}, {"name": "CX", "targets": [2, 9]}]}'
```

# COPYRIGHT

Copyright © 2023 University of Strasbourg. All Rights Reserved.

# AUTHORS

See the [AUTHORS.md](AUTHORS.md) file for the list of authors.
