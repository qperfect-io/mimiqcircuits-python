# Changelog

All notable changes to `mimiqcircuits` (Python) are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
