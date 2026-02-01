# MIMIQ Circuits Python

[![Build Status](https://github.com/qperfect-io/mimiqcircuits-python/workflows/Test/badge.svg)](https://github.com/qperfect-io/mimiqcircuits-python/actions)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://docs.qperfect.io/mimiqcircuits-python/)
[![PyPI version](https://badge.fury.io/py/mimiqcircuits.svg)](https://pypi.org/project/mimiqcircuits/)
[![Python versions](https://img.shields.io/pypi/pyversions/mimiqcircuits.svg)](https://pypi.org/project/mimiqcircuits/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**MIMIQ Circuits** is a comprehensive Python library for building, simulating, and executing quantum circuits on QPerfect's MIMIQ Virtual Quantum Computer. Design quantum algorithms, test them locally, and scale them on high-performance cloud simulators.

Part of the [MIMIQ](https://qperfect.io) ecosystem by [QPerfect](https://qperfect.io).

## Features

- 🔧 **Build quantum circuits** with a Pythonic, intuitive API
- ☁️ **Execute on MIMIQ cloud** with automatic job management
- 🔄 **Import/Export OpenQASM** and Stim files for interoperability
- 🎯 **Multiple simulation algorithms** (state vector, MPS, and more)
- 📈 **Expectation values and measurements** for quantum observables
- 🌐 **Seamless cloud integration** with authentication and connection management
- 🧪 **Noise modeling** with realistic quantum error channels

## Installation

### From PyPI (Recommended)

```bash
pip install mimiqcircuits
```

### With Visualization Support

```bash
pip install mimiqcircuits[visualization]
```

### From GitHub (Latest Development Version)

```bash
pip install "mimiqcircuits @ git+https://github.com/qperfect-io/mimiqcircuits-python.git"
```

### Requirements

- Python 3.8 or higher
- Supported on Linux, macOS, and Windows

## Quick Start

### Building Your First Circuit

```python
from mimiqcircuits import *

# Create a Bell state circuit
circuit = Circuit()
circuit.push(GateH(), 0)
circuit.push(GateCX(), 0, 1)
circuit.push(Measure(), 0, 0)
circuit.push(Measure(), 1, 1)

# Print the circuit
circuit.draw()
```

### Connecting to MIMIQ Cloud

```python
from mimiqcircuits import *

# Establish connection (opens browser for authentication)
conn = MimiqConnection()
conn.connect()

# Alternative: Connect with credentials directly
# conn.connectUser("your.email@example.com", "yourpassword")
```

### Executing a Circuit

```python
from mimiqcircuits import *

# Create a multi-qubit entangled state
circuit = Circuit()
circuit.push(GateH(), 0)
for i in range(1, 10):
    circuit.push(GateCX(), 0, i)
circuit.push(Measure(), range(10), range(10))

# Connect to MIMIQ
conn = MimiqConnection()
conn.connect()

# Execute on MIMIQ remote services
job = conn.execute(circuit, algorithm="auto", nsamples=1000)

# Wait for completion and retrieve results
results = conn.get_results(job)

# Visualize results
from mimiqcircuits.visualization import plothistogram
plothistogram(results)
```

### Working with OpenQASM

```python
from mimiqcircuits import *

# Connect to MIMIQ
conn = MimiqConnection()
conn.connect()

# Execute a QASM file
job = conn.execute("path/to/circuit.qasm", algorithm="statevector")
results = conn.get_results(job)
```

## Circuit Building Examples

### Quantum Fourier Transform

```python
from mimiqcircuits import *

circuit = Circuit()
# QFT on 5 qubits
circuit.push(QFT(5), *range(5))
circuit.push(Barrier(5), *range(5))
circuit.push(inverse(QFT(5)), *range(5))

circuit.draw()
```

### Parameterized Circuits

```python
from mimiqcircuits import *
import numpy as np

circuit = Circuit()
theta = np.pi / 4

# Rotation gates with parameters
circuit.push(GateRX(theta), 0)
circuit.push(GateRY(theta), 1)
circuit.push(GateRZ(theta), 2)

# Custom controlled rotation
circuit.push(GateCRX(theta), 0, 1)

circuit.deaw()
```

### Expectation Values

```python
from mimiqcircuits import *

circuit = Circuit()
circuit.push(GateH(), 0)
circuit.push(GateCX(), 0, 1)

# Measure expectation value of Z operator
circuit.push(ExpectationValue(GateZ()), 0, 0)

conn = MimiqConnection()
conn.connect()
results = conn.execute(circuit, algorithm="statevector")
```

### Noisy Simulations

```python
from mimiqcircuits import *

circuit = Circuit()
circuit.push(GateH(), 0)
circuit.push(Depolarizing1(0.01), 0)  # 1% depolarizing noise
circuit.push(GateCX(), 0, 1)
circuit.push(AmplitudeDamping(0.05), 0)  # 5% amplitude damping
circuit.push(Measure(), range(2), range(2))
```

## Available Simulation Algorithms

MIMIQ supports multiple simulation algorithms optimized for different use cases:

- **`auto`** - Automatically selects the best algorithm
- **`statevector`** - Full state vector simulation (exact, up to ~30 qubits)
- **`mps`** - Matrix Product State (efficient for certain circuit structures)

```python
# Specify algorithm during execution
job = conn.execute(circuit, algorithm="mps", nsamples=10000)
```

## Supported Gates and Operations

### Single-Qubit Gates

`GateH`, `GateX`, `GateY`, `GateZ`, `GateS`, `GateT`, `GateSX`, `GateTDG`, `GateSDG`, `GateID`, `GateRX`, `GateRY`, `GateRZ`, `GateU`, `GateP`

### Two-Qubit Gates

`GateCX`, `GateCY`, `GateCZ`, `GateCH`, `GateSWAP`, `GateISWAP`, `GateDCX`, `GateECR`

### Three-Qubit Gates

`GateCCX` (Toffoli), `GateCSWAP` (Fredkin)

### Multi-Controlled Gates

`GateCU`, `GateCRX`, `GateCRY`, `GateCRZ`, `GateCP`

### Composite Gates

`QFT` (Quantum Fourier Transform), `PhaseGradient`, `PolynomialOracle`

### Measurements

`Measure`, `MeasureReset`, `ExpectationValue`

### Noise Channels

`Depolarizing`, `AmplitudeDamping`, `PhaseDamping`, `PauliNoise`, `ThermalNoise`

### Circuit Control

`Barrier`, `Reset`, `IfStatement` (conditional operations)

## Julia Version

For Julia users, check out [MimiqCircuits.jl](https://github.com/qperfect-io/MimiqCircuits.jl) - the native Julia implementation with full feature parity.

## Related Packages

- **[mimiqlink-python](https://github.com/qperfect-io/mimiqlink-python)** - Connection and authentication library (included in mimiqcircuits)
- **[MimiqCircuits.jl](https://github.com/qperfect-io/MimiqCircuits.jl)** - Julia version of this library

## Access to MIMIQ

MIMIQ Circuits is currently in active development and available for early adopters. To execute circuits on MIMIQ's cloud services, you need an active subscription.

- 🌐 **[Register on our website](https://qperfect.io)** to request access
- 📧 Contact us at <contact@qperfect.io> for organizational subscriptions
- 💬 We value your feedback to improve MIMIQ Circuits

## Contributing

We welcome contributions from the quantum computing community! Whether you're fixing bugs, adding features, improving documentation, or sharing ideas:

1. 🐛 **Report issues** on [GitHub Issues](https://github.com/qperfect-io/mimiqcircuits-python/issues)
2. 💡 **Suggest features** through GitHub Discussions
3. 🔧 **Submit pull requests** - please read our contributing guidelines first
4. 📝 **Improve documentation** - even small fixes help!

## Support and Community

- 📧 **Email:** <mimiq.support@qperfect.io>
- 🐛 **Bug Reports:** [GitHub Issues](https://github.com/qperfect-io/mimiqcircuits-python/issues)
- 💬 **Discussions:** [GitHub Discussions](https://github.com/qperfect-io/mimiqcircuits-python/discussions)
- 🌐 **Website:** [qperfect.io](https://qperfect.io)

## COPYRIGHT

> Copyright © 2022-2023 University of Strasbourg. All Rights Reserved.
>
> Licensed under the Apache License, Version 2.0 (the "License");
> you may not use this file except in compliance with the License.
> You may obtain a copy of the License at
>
>     <http://www.apache.org/licenses/LICENSE-2.0>
>
> Unless required by applicable law or agreed to in writing, software
> distributed under the License is distributed on an "AS IS" BASIS,
> WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
> See the License for the specific language governing permissions and
> limitations under the License.

---

**Made with ❤️ by the QPerfect team**
