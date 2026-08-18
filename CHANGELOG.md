# Changelog

All notable changes to `mimiqcircuits` (Python) are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.26.7] — 2026-08-18

`GateCustom.matrix` is a method again, as it is on every other operation. Read it
as `gate.matrix()`; `gate.matrix` on its own now gives you the bound method
rather than the matrix.

### Fixed
- `Instruction.matrix()` no longer raises for instructions holding a `GateCustom`. `GateCustom.__init__` assigned `self.matrix`, which shadowed its own `matrix()` method, so the method was unreachable and `Instruction.matrix()` tried to call a matrix. The matrix now lives in a private field and `matrix()` works, which also makes `RescaledGate` accept a `GateCustom`. This mattered more after the fusion changes below, since `GateCustom` is the normal output of `fuse_circuit`.

### Changed
- `fuse_circuit` and `FusePass` build each fused block by contracting every member gate directly against the accumulating matrix, instead of embedding it into the block's full space first. Together with the changes below, fusion is 2.7–5.0x faster across bricklayer, QFT and quantum volume circuits at `max_support` 3–5. The fused unitary is unchanged, but the summation order is not, so entries may differ in the last few bits of the mantissa (measured within 3 units in the last place).
- `reorder_qubits_matrix` embeds a NumPy matrix by contracting it against an identity, replacing the chain of Kronecker products and the index permutation that followed it. Every entry of that product is a sum with at most one non-zero term, so results are bit-for-bit identical to before. Symbolic matrices keep the SymEngine path, whose index permutation is now derived from the `nq` basis indices and inverted by scatter rather than round-tripping all `2**nq` indices through `BitString` and calling `argsort`.
- `GateCustom` skips the elementwise symbolic scan when handed a NumPy array of numeric dtype, where it could never match, and checks unitarity against that array rather than converting its SymEngine copy back to NumPy.

## [0.26.6] — 2026-08-18

### Changed
- `GateCustom` checks unitarity with NumPy when the matrix is fully numeric, instead of building a dense SymEngine product and walking its entries in Python. Constructing a 9-qubit block drops from about 15 seconds to under half a second, which matters now that `fuse_circuit` can emit blocks that wide. Symbolic matrices are unaffected.

### Fixed
- `fuse_circuit` and `FusePass` merge the clusters owning a gate's wires instead of only extending a single one, so `max_support` above 2 now produces wider blocks. On entangling circuits every wire is owned after the first layer, which made every later gate start a fresh cluster and pinned the output at the `max_support=2` result. A quantum volume circuit on 30 qubits at depth 30 now fuses to 442 blocks at `max_support=2` and 159 at `max_support=4`, where before it gave 442 at every setting.

### Docs
- Fixed the heading hierarchy in the entanglement dynamics demo, which rendered several sections as separate top-level pages.

## [0.26.5] — 2026-08-16

### Fixed
- `GateCustom` accepts matrices of any power-of-two size. It previously derived the qubit count incorrectly and rejected matrices wider than 3 qubits, which also made `fuse_circuit` and `FusePass` raise for `max_support` above 3.

### Docs
- The use case demos are rendered from notebooks hosted in the documentation instead of linking to blog posts. Sphinx now uses `myst-nb`, with notebook execution disabled.

## [0.26.4] — 2026-08-05

### Fixed
- `Block.is_symbolic` and `Block.evaluate` now descend into the block's instructions, so parameters inside a `Block` are evaluated.

### Docs
- Condensed the changelog entries for earlier releases.

## [0.26.3] — 2026-07-17

### Added
- `CanonicalDecomposePass` decomposes a circuit to its canonical basis via `Circuit.decompose`, composing in a `PassPipeline` like `FusePass`.
- `execute` and `submit` accept the circuit-preparation knobs `fuse`, `fuse_threshold`, `canonicaldecompose`, `reorderqubits`, and `remove_swaps` as shorthands for a `PassPipeline`. Locally, `reorderqubits` and `remove_swaps` raise; the remote backend forwards all of them to the server. Combining the knobs with an explicit `passes=` raises.

## [0.26.2] — 2026-07-17

### Added
- `fuse_circuit(circuit, max_support=2)` and the `FusePass` pass replace runs of adjacent unitary gates with a single `GateCustom` block, preserving the circuit unitary. Measurements, resets, noise channels, `Barrier`, control flow, and gates with symbolic parameters act as fusion boundaries. `max_support` caps the block width.
- `FusePass` takes a `qubit_threshold` keyword (default 0): fusion is skipped for circuits with fewer qubits.

### Fixed
- `reorder_qubits_matrix`, used by `Instruction.matrix()` when embedding a gate into a larger register, returned a wrong matrix for non-contiguous or reordered target wires, placing the gate on the wrong wires.

## [0.26.1] — 2026-06-28

### Added
- `CircuitDAG`, `traverse_by_bfs`, `traverse_by_dfs`, `topological_sort_by_bfs`, `topological_sort_by_dfs`, and `to_networkx` are now exported from the top-level `mimiqcircuits` namespace, and the manual documents the `Circuit.dag()` view.

### Fixed
- `lower_losses` now writes the classical bit for every single-qubit measurement on a lost qubit (`MeasureX`, `MeasureY`, `MeasureReset`, …), not just `Measure`/`MeasureZ`.
- `CircuitDAG` now has a readable `repr` and raises a clear `TypeError` when given a `Circuit` instead of a vertex count.

## [0.26.0] — 2026-06-26

### Added
- `LocalBackend.apply_segment(state, insts)` applies a batch of instructions at once, so compressing backends keep gate fusion across runtime-loss circuits. The default applies one instruction at a time.

### Changed
- The runtime-loss driver `LocalBackend.evolve_with_loss` now flushes each run of all-present instructions through `apply_segment` instead of applying them one by one. Results are unchanged for non-compressing backends.

## [0.25.0] — 2026-06-22

### Added
- `Loss(p)` operation for qubit loss (`Loss()` means certain loss), plus `Reload`, `Check`, and `MeasureCheck`.
- `Lost` and `Reloaded` annotations, which record loss and reload events without affecting execution.
- Loss resolution split into three functions, with matching `Circuit` methods: `sample_losses` draws the random `Loss(p)` events, `lower_losses` rewrites loss bookkeeping into primitives, and `resolve_losses` runs both to produce a circuit that runs on any backend.
- `sample_loss_scenario` builds a "what-if" loss scenario by forcing the selected `Loss` sites and resolving.
- `lossmodel_rewrite` exposes the per-instruction `LossModel` decision, so backends can apply the same loss rules at runtime as `lower_losses` does offline.
- `LossState` register and the `LocalBackend.evolve_with_loss` driver. A backend gains runtime `LossyOperator` loss by returning `True` from `uses_loss_driver` and implementing `apply_instruction` and `sample_kraus`.
- `MixedUnitary` accepts a `lossy` keyword marking which qubits leak in each branch. `Circuit.sample_mixedunitaries` then emits a `Loss` on those qubits when a lossy branch is drawn.

### Changed
- Protobuf serialization renames the loss operations and adds the `Lost` / `Reloaded` annotations. The old `QubitLoss` tag still decodes as `Loss()`, so circuits saved by older versions keep loading. `WIRE_FORMAT_VERSION` is now `1.1.0`.
- `MixedUnitaryChannel` gains an optional `lossy_masks` field. Circuits without lossy branches serialize unchanged.

### Deprecated
- `LossErr`, `QubitLoss`, `QubitReload`, `CheckLoss`, and `MeasureCheckLoss` are aliases for the new operations and will be removed in a future release. They are functions rather than classes, so `isinstance(op, QubitLoss)` no longer works; test against `Loss`, `Reload`, `Check`, or `MeasureCheck` instead.

## [0.24.2] — 2026-06-01

### Fixed
- `Circuit.dag()`, `traverse_by_bfs`, and `traverse_by_dfs` now treat `BondDim`, `SchmidtRank`, and `VonNeumannEntropy` as depending on every qubit, so they stay after the gates that affect the bond they probe.

## [0.24.1] — 2026-06-01

### Fixed
- `Circuit.dag()`, `traverse_by_bfs`, and `traverse_by_dfs` now treat `Amplitude` as depending on every qubit, so it stays after the gates whose state it reads.

## [0.24.0] — 2026-05-31

### Added
- `Circuit.dag()` returns a `CircuitDAG` whose vertices are instruction positions and whose edges record the qubit, bit, and z-variable dependencies between instructions. `traverse_by_bfs` and `traverse_by_dfs` iterate the instructions in topological order, and `to_networkx` exports the graph as a `networkx.DiGraph`. networkx is an optional dependency, installed with the `graph` extra.

### Changed
- `Circuit.instructions` is now a read-only property. Instructions must be added or removed through `push`, `insert`, `append`, or `remove`, which keep the cached dependency graph and the qubit/bit/z-variable counts consistent.

## [0.23.2] — 2026-05-31

### Added
- Progress reporting for `LocalBackend.execute`. Pass `progress=True` (or a `TqdmProgress`) to draw bars for the compression, execution, and trajectory phases. The default `progress=False` keeps the pipeline cost-free.
- `Progress` sink interface with `NoProgress` and `TqdmProgress` implementations. Adds a `tqdm` dependency.

## [0.23.1] — 2026-05-28

### Fixed
- `mimiqcircuits.__version__` now comes from the installed package metadata instead of a hard-coded literal, which had drifted out of sync with `pyproject.toml`.

## [0.23.0] — 2026-05-27

### Added
- `mimiqcircuits.WIRE_FORMAT_VERSION` constant (initial value `"1.0.0"`) declaring the version of the MIMIQ wire format, independently of the package version. Mirrors `MimiqCircuitsBase.WIRE_FORMAT_VERSION`.

### Changed
- The JSON request envelope for `submit` and `optimize` now carries a `wireformatversion` key alongside `circuitsapiversion`. Older executors ignore it.
- `circuitproto.toproto_circuit` is 1.4-2x faster, through memoized converter lookup and gate converters that write the `Operation` proto directly.

### Fixed
- `Backend.can_handle` now rejects circuits with noise channels when the backend has not declared `"noise"`, and circuits with free symbolic parameters when it has not declared `"parametric"`. Previously both surfaced as opaque evolve-time errors.

## [0.21.8]

Changelog tracking begins with this version. See git history for prior changes.
