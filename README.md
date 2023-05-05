# Python interface to MIMIQ Circuits (`mimiqcircuits`)

A library to handle quantum circuits for **QPerfect**'s MIMIQ Emulator.

## Installation

```
pip install "mimiqcircuits @ git+https://github.com/qperfect-io/mimiqcircuits-python.git
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

## COPYRIGHT

Copyright © 2022-2023 University of Strasbourg. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

