# Changelog

All notable changes to `mimiqcircuits` (Python) are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.30.0] - 2026-10-24

### Added
- `ParameterTable` holds the values of a parameter sweep: a fixed list of symbols and one row of values per point. It is built from an array, from one column per symbol (`ParameterTable.from_columns({theta: xs, phi: ys})`), or from a list of dictionaries (`ParameterTable.from_dicts`).
- `gridtable({theta: xs, phi: ys})` builds the `ParameterTable` of every combination of the given values.
- Parameter tables compose: `a * b` is the Cartesian product, `a.hcat(b)` pairs the points of two tables of the same length, and `a.vcat(b)` appends the points of `b` after those of `a`.
- `SweepExperiment(circuit, table)` pairs a symbolic circuit with the table of values to run it with, and checks that the table binds exactly the free symbols of the circuit. `ex[i]` is the circuit evaluated at point `i`.
- `LocalBackend.execute` accepts a `SweepExperiment` and returns one result per point, with the same per-point seeds as the equivalent `param_grid`.
- `t[i:j]` on a `ParameterTable` returns the table of those points, and `ex[i:j]` on a `SweepExperiment` the experiment over them.
- `ParameterTable` and `SweepExperiment` compare structurally with `==`; a `ParameterTable` is also hashable.
- `simplify(circuit)` reduces a circuit by running a list of passes until the number of two-qubit gates, then of instructions, stops going down; the unitary is kept exactly, global phase included, and `passes` selects which passes run. It is the same algorithm as in MimiqCircuitsBase.jl and runs in the native extension.
- `remove_identities(circuit)` removes gates whose matrix is exactly the identity, such as `GateID` or `GateRZ(0)`.
- `cancel_inverses(circuit)` removes pairs of gates that multiply to the identity, such as `CX CX` or `S SDG`, including across gates that the second one commutes with, and moves multi-controlled X gates by structural rules that hold at any width.
- `merge_rotations(circuit)` adds up the angles of consecutive rotations of the same kind on the same qubits, `GateRZ(a) GateRZ(b)` into `GateRZ(a + b)`, across commuting gates and with symbolic angles, and merges different single-qubit gates of one rotation family, Z/S/T/P/U1, X/SX or Y/SY and their inverses, so `T T` becomes `S`.
- `resynthesize_1q(circuit)` replaces each run of single-qubit gates with one `GateU`, or removes it when it is the identity.
- `resynthesize_2q(circuit)` rewrites each two-qubit block with the minimal number of `GateCX`, from zero to three, found from its KAK decomposition, when that lowers its two-qubit gate count.
- `resynthesize_linear(circuit)` rewrites every block of `GateX`, `GateCX` and `GateSWAP`, of any width, from the affine map it computes, using Patel-Markov-Hayes synthesis, when that lowers its two-qubit gate count; it is not one of the default passes of `simplify`.
- `SimplifyPass(passes=None, maxiter=10)` runs `simplify` as a circuit-preparation pass, keeping the unitary exactly; `passes` names the simplification passes to run, so the pass can be described to a remote executor.
- `execute(…, simplify=True)` on a local backend adds the backend's `simplify_pass`, `SimplifyPass()` by default, between removing swaps and canonical decomposition; `simplify=False` drops a backend's own. It is off unless asked for. The remote backend refuses it for now; simplify the circuit before submitting instead.

### Changed
- `PrepareMagic` is renamed `PrepareStateT`. `PrepareMagic` remains as a deprecated alias. The protobuf enum value keeps its number, so the wire format is unchanged.
- `evaluate` on a wrapper (`Inverse`, `Power`, `Control`, `Parallel`, `Repeat`, `IfStatement`, `WhileStatement`, `Block`, …) returns the operation untouched when it has no free symbols, and otherwise rebuilds it through the simplifying builders, so `Inverse(GateRX(x)).evaluate({x: 0.3})` is `GateRX(-0.3)`.
- `Power` and `.power()` reject an exponent that is not a real number, such as a symbol, with a `TypeError`.
- `Backend.execute` no longer lists `fuse_threshold`; it stays on `LocalBackend.execute`, since the remote service has no fuse threshold.
- `reorderqubits_seed` accepts any integer, including NumPy integers.
- `Circuit.evaluate` is about 7x faster on circuits with many parameters: it converts the parameter dict once instead of at every gate, and reuses instructions that have nothing to substitute instead of rebuilding them.
- `QCSResults.cstates` holds `BitString`s for results decoded from protobuf, such as every remote execution, as it already did for local backends. A key from `amplitudes` now matches its entry in `histogram()`. Code that calls `bitarray` methods directly on `results.cstates[i]` needs `.bits`; indexing, iteration, `to01()` and `len()` work unchanged.
- `rescale` and `rescale_inplace` are defined once on `AbstractOperator`, so every operator has both (`SigmaPlus` gains `rescale_inplace`). On a gate, `rescale` returns a `RescaledGate` and `rescale_inplace` raises `TypeError`.

### Deprecated
- `ParityCheck` is deprecated. `ParityCheck(N)` warns and returns `Xor(N)`, which computes the same parity and allows the target to be one of the inputs. Circuits saved with `ParityCheck` load as `Xor`, and new circuits never save it.

### Removed
The deprecated aliases below are gone. Replace `LossErr(p)` with `Loss(p)`, `QubitLoss()` with `Loss()`, `QubitReload()` with `Reload()`, `CheckLoss()` with `Check()`, `MeasureCheckLoss()` with `MeasureCheck()`, `MimiqConnection.schedule` with `submit`, and `qasmstring`/`qasmsave` with `mimiqcircuits.qasm.dumps`/`dump`.

- `RescaledGate.rescale_in_place`; use `rescale_inplace`.
- The deprecated loss operations `LossErr`, `QubitLoss`, `QubitReload`, `CheckLoss` and `MeasureCheckLoss`.
- The deprecated `schedule` method of `MimiqConnection` and `PlanqkConnection`.
- The legacy `qasmstring` and `qasmsave` aliases in `mimiqcircuits.qasm.serializer`.
- Unused module-level helpers: `mimiqcircuits.bitstrings.int_to_bitvec`, `mimiqcircuits.matrices.pmatrixpi`, `ctrl`, `gphase`, `gphasepi`, `rxmatrix`, `rymatrix`, `rzmatrix` and their `*pi` variants, `mimiqcircuits.numerics.pmatrixpi`, `ctrl` and `kron`, and `mimiqcircuits.operations.noisechannel.standards.projectivenoise.checknormalizlength`.
- The unused `mimiqcircuits.unicode` module.
- The `quantanium` extra.

### Fixed
- A circuit with an `RPauli` before a measurement or an expectation value no longer raises `AttributeError: ... has no attribute 'op'` when a backend analyses it for sampling. Wrappers now follow the rule MimiqCircuitsBase.jl has: `iswrapper()` is true exactly when `get_operation()` returns the wrapped operation, and code that looks through wrappers calls `get_operation()`. `Power` and `Inverse` gain `get_operation()`, `ExpectationValue.iswrapper()` returns `True`, and `RPauli.iswrapper()` returns `False`, both as in Julia, so a nested `RPauli` prints without brackets (`CR("XY", 0.3)`), also as in Julia.
- `Operator.unwrappedmatrix()` and `LossyOperator.unwrappedmatrix()` return a `complex128` NumPy array instead of the operator's own SymEngine matrix, and raise `UnexpectedSymbolics` when an entry is symbolic.
- `GateRNZ` and `RPauli` support `unwrappedmatrix()`, alone and inside wrappers such as `Control` and `Inverse`.
- `repr` of a `MixedUnitary` holding a `GateCustom` no longer raises.
- `listvars` and `is_symbolic` look inside `Block` and every wrapper, so `SweepExperiment` accepts a symbol that appears only inside a `Block`.
- The preparation flags (`fuse`, `canonicaldecompose`, `remove_swaps`, `reorderqubits`, `fuse_threshold`, `reorderqubits_seed`) raise `TypeError` on a value of the wrong type, locally and in `MimiqConnection.submit`, instead of reading `fuse=0` as a request to fuse.
- `GateCustom.getparam` and `GateCustomDiagonal.getparam` raise `IndexError` for an index outside `1..n` instead of wrapping around.
- `ParameterTable.from_dicts` raises when a point gives the same symbol both as a symbol and as its name.
- `decompose(c, basis, wrap=True)` no longer fails on a non-unitary operation that has a decomposition, such as `MeasureReset`, `MeasurePauli` or `PrepareStateT`. It is wrapped into a `Block` holding its decomposition, and the gates in it are wrapped into a `GateDecl` as usual. `mimiqcircuits.qasm.dumps` writes a `Block` inline, so it now exports `MeasurePauli` and `PrepareStateT`.
- `mimiqcircuits.qasm.dumps` writes an `IfStatement` as `if(c==n)` when it reads the whole classical register, and raises `QASMSerializationError` otherwise; it used to raise for every `IfStatement`.
- `mimiqcircuits.qasm.loads` reads `if(c==n)` statements, which raised `TypeError`.
- `QASMBasis` decomposes `MeasureReset` into `Measure` and `Reset`, so `mimiqcircuits.qasm.dumps` exports `MeasureReset`, `MeasureResetX` and `MeasureResetY` for a circuit of any number of bits.
- `QASMBasis` raises `DecompositionError` on a `WhileStatement` instead of keeping it as terminal, since OpenQASM 2.0 has no loops.
- The operation hierarchy from `InheritanceTree.extract_classes` lists each class once under its own name, so the deprecated alias `PrepareMagic` no longer shows next to `PrepareStateT`.
- `GateRY(theta).control(n)` returns the controlled gate instead of `None`, so evaluating a circuit holding `Control(1, GateRY(theta))` no longer raises `TypeError`.
- `lossmodel_rewrite` no longer requires `rng`; it defaults to a fresh `random.Random()`, as in `lower_losses` and `sample_losses`.
- `saveproto` on a `NoiseModel` containing a `CustomNoiseRule` raises a `ValueError` saying custom rules cannot be serialized, and `loadproto` refuses a stored custom rule instead of decoding it into a rule that matches every instruction.
- `from mimiqcircuits.operations.gates.custom import *` and `from mimiqcircuits.operations.krauschannel import *` no longer raise `AttributeError`: their `__all__` listed the undefined names `Custom` and `Noise`.
- `t[[i, j]]` on a `ParameterTable` returns the table of the selected points, and `ex[[i, j]]` on a `SweepExperiment` the experiment over them, instead of a dictionary of mismatched values.
- `haslossybranch()` works on every Kraus channel: it is `True` when a `Kraus` channel has a `LossyOperator` branch and `False` for channels without loss branches, such as `PauliX(0.1)`, which used to raise `AttributeError`.
- `Circuit.sample_mixedunitaries()` without an `rng` no longer raises `TypeError`.
- `QCSResults` displays in Jupyter without printing a traceback, and `plothistogram` no longer raises `TypeError`.
- `bitvec_to_int` accepts a mutable `bitarray` instead of raising `TypeError`.
- `CircuitTesterExperiment.interpret_results` reports the probability of the all-zero outcome instead of always 0.0 with the sampling method.

### Build
- The native extension depends on `num-complex`.

### CI
- `vX.Y.Z-private` tags build wheels for Linux, macOS and Windows plus the sdist and upload them to the internal `python/index` package index.

### Docs
- The installation guide and the local simulation page present Exaqt, QPerfect's free local statevector simulator (`pip install mimiq-exaqt`), in place of Quantanium.
- The `Detector` and `AbstractAnnotation` docstrings no longer open with a section title, which broke the API summary page.
- The symbolic operations manual explains parameter sweeps with `ParameterTable`, `gridtable` and `SweepExperiment`, and `param_grid` as their shorthand.
- `LocalBackend.execute(param_grid=...)` is documented: the `execute` docstring describes the argument, its return value and seeding, and the symbolic operations manual has a section on running parameter sweeps.
- `listvars` on operations, instructions, blocks and circuits has a docstring: what it looks into, what it returns, and that the order of the symbols is not specified.
- The `Block` docstring example matches the current output, and the modules under `mimiqcircuits.operations.decompositions` are regular packages, so `pytest --doctest-modules src` collects them.
- The noise rules (`GlobalReadoutNoise`, `ExactQubitReadoutNoise`, `SetQubitReadoutNoise`, `OperationInstanceNoise`, `ExactOperationInstanceQubitNoise`, `SetOperationInstanceQubitNoise`, `IdleNoise`, `SetIdleQubitNoise`, `CustomNoiseRule`) have docstrings with examples; `IdleNoise` and `SetIdleQubitNoise` state that they replace explicit `Delay` instructions.
- `Circuit.evaluate`, `Circuit.listvars`, `Circuit.getparams`, the annotations' `get_notes`, `repeat`, `gatedecl`, `Depolarizing1`, `Depolarizing2`, `Gate`, `AbstractMeasurement`, `LazyExpr` and `LazyArg` have docstrings.

## [0.29.0] - 2026-10-17

### Added
- `Backend.fuse_pass`, `reorder_pass`, `canonical_decompose_pass` and `remove_swaps_pass` build the pass each flag installs, so a backend with its own pass of that kind keeps it when a flag asks; `reorder_pass` and `remove_swaps_pass` raise `NotImplementedError` by default, since no generic local pass exists.
- `resolve_pipeline(backend, passes, **flags)` resolves circuit preparation; `LocalBackend._resolve_prep_passes` stays as an alias of `_resolve_pipeline` for mimiq-qiskit.
- `LocalBackend.execute` takes `reorderqubits_seed`, which seeds the reorder pass and on its own does not request reordering.
- `PassPipeline.__add__`.
- `GridSynthRewrite(epsilon=1e-10, seed=0)` approximates `GateRZ` with Clifford+T gates to operator-norm error `epsilon`, from `1e-100` up to `1`, using the Ross-Selinger gridsynth algorithm, with the same output as in MimiqCircuitsBase.jl.
- `CliffordTBasis(rotations=rule)` selects the rule that synthesizes the remaining Z-rotations.
- `MeasurePauli(pauli)` measures an `N`-qubit Pauli product observable into one classical bit, and decomposes into `GateH`, `GateS`, `GateSDG`, `GateCX` and one `Measure`, so `QASMBasis` can decompose it.
- `PrepareMagic()` prepares one qubit in the magic state `T|+⟩` and decomposes into `Reset`, `GateH`, `GateT`.
- `pauli_based_computation(circuit)`, also available as `Circuit.pauli_based_computation`, rewrites a Clifford+T circuit into `π/8` Pauli rotations and `MeasurePauli`, reproducing the measurement statistics of the circuit but not its final state.
- `pauli_based_computation(circuit, magic_states=True)` replaces each `π/8` rotation by magic-state injection with `PrepareMagic`, `MeasurePauli` and classical feed-forward, tracking the Pauli frame with 3-bit `Xor` operations only.
- Wire format 1.3.0 adds the `MeasurePauli` operation and the `PrepareMagic` Kraus channel type to `circuit.proto`; clients send this version with every request, so a release carrying it cannot submit any job until the executor speaks 1.3.0. It also writes symbolic parameters as exact expression trees instead of rounded numbers, a patch-level change that older decoders read to the same values.
- `CliffordTBasis`, `FlattenedBasis` and `RuleBasis` decompose the operation inside an `IfStatement` with their own rules and put each resulting instruction under the same condition, so classically controlled gates no longer raise `DecompositionError`.
- `pauli_based_computation` supports a classically controlled Pauli operator, or a `PauliString`, `Block`, `Parallel`, `Repeat` or `GateCall` made only of Paulis, as one conditional `PauliString`, folded into the Pauli frame with `magic_states=True`.
- `pauli_based_computation` supports `Reset`, `ResetX`, `ResetY`, `MeasureReset`, `MeasureResetX` and `MeasureResetY` as a measurement followed by a conditional Pauli, each reset writing its outcome to a new bit after the circuit's bits.
- `pauli_based_computation` turns a classically controlled T or T† gate into a conditional `RPauli`, and refuses it with `magic_states=True`.
- `pauli_based_computation` handles `MeasureX`, `MeasureY`, `MeasureXX`, `MeasureYY`, `MeasureZZ` and `MeasurePauli` itself, whatever `basis` does with them.
- `pauli_based_computation` explains in its error why it refuses a classically controlled Clifford gate: the Clifford after it would depend on the condition bits.
- `StimBasis` keeps `MeasurePauli`, `MeasureXX`, `MeasureYY`, `MeasureZZ`, `PauliNoise`, `GateHXY` and `GateHYZ` as terminal, since Stim has them natively.
- `StimBasis` keeps `GateRXX`, `GateRYY`, `GateRZZ` and `RPauli` at ±π/2 as terminal (Stim's `SQRT_XX`, `SQRT_YY`, `SQRT_ZZ`, `SPP`) and decomposes them at other angles.
- `StimBasis` keeps `IfStatement` of `GateX`, `GateY` or `GateZ` on one bit equal to 1 as terminal (Stim's feedback `CX rec[-k] q`), rewrites the same gate conditioned on 0 as the gate followed by that conditional, and raises `DecompositionError` for any other conditional operation.

### Changed
Preparation flags now adjust the backend's `default_passes` instead of replacing them, and `passes=` is appended after them. A caller that passed `False` for every flag to get the defaults now gets no pass of those kinds: pass `None`, or leave the flag out.

- `Circuit.draw` packs instructions on disjoint wires into shared columns, so independent gates are drawn side by side instead of one per column. Instructions that read or write the classical or Z register keep their circuit order and are never drawn in parallel with each other. Consecutive instructions of the same operation type on disjoint wires, such as those from one broadcast push, start in the same column.
- `Circuit.draw(compact=False)` keeps the previous layout, with every instruction in its own column.
- `Circuit.draw` labels an operation on classical bits or z-variables only with its targets, such as `c[0] = c[1] ^ c[2]`, instead of a list of indices followed by the operation, such as `0,1,2 ═> ^`.
- An `Xor`, `And` or `Or` instruction whose target is also an input prints in compound form, such as `c[2] ^= c[1]` for `c[2] = c[2] ^ c[1]`.
- Each preparation flag of `execute` defaults to `None`, which keeps the backend's pass of that kind; `False` drops it, and any other value replaces it or adds it in the fixed flag order.
- On a local backend, `passes=` together with flags no longer raises `ValueError`: it is appended after the adjusted backend passes, without deduplication. Remote submission still rejects the pair.
- `reorderqubits` and `remove_swaps` raise `NotImplementedError` only when requested on a backend that does not override `reorder_pass` or `remove_swaps_pass`.
- `QCSResults.saveproto` and `loadproto`, and every other conversion of bit vectors to protobuf `BitVector` messages, are several times faster, write the same bytes, and now also accept `"0"`/`"1"` strings.
- `Instruction.matrix` and `reorder_qubits_matrix` embed numeric gate matrices into larger registers about 40x faster, with identical results.
- `matrix()` and `unwrappedmatrix()` now cache the matrix of named wrapper gates such as `GateT`, `GateCX` and `GateCCX`, which speeds up `fuse_circuit`.
- `unwrappedmatrix()` on named wrapper gates such as `GateT` and `GateCX` now returns a read-only array, as it already did for parameter-free gates; copy it before writing into it.
- `add_noise`, `decorate` and the `decorate_on_match_*` functions and methods now run in linear time.
- `Circuit.insert` inserts a whole `Circuit` in one step instead of one instruction at a time.
- An in-place `decorate_on_match_*` call whose generator raises now leaves the circuit unchanged.
- `decompose` is faster with every built-in basis, about 2x with `CanonicalBasis` on large circuits.
- `Circuit.push` and `Block.push` are faster when every target is a plain `int`.
- Constructing `Instruction`, `GateT`, `GateS`, `GateSX` and `GateSY` objects is faster.
- `QASMBasis` keeps only OpenQASM 2.0 built-ins and `qelib1.inc` gates terminal, so `dumps` writes other gates as gate definitions, decomposes measurement variants, and rejects noise, classical operations and annotations with `DecompositionError`.
- `CanonicalRewrite.matches` accepts every operation with a canonical decomposition, whatever `CanonicalBasis` keeps terminal, as in MimiqCircuitsBase.jl, so other bases can expand `ResetX` and `ResetY`.
- `GateCS`, `GateCSDG`, `GateCSX` and `GateCSXDG` decompose as in MimiqCircuitsBase.jl, so `GateCSX` gives 7 canonical gates instead of 28.
- `GateCRX` and `GateISWAPDG` decompose as in MimiqCircuitsBase.jl.
- `QFT` and `PhaseGradient` decompositions use exact fractions of π instead of floating-point coefficients, as in MimiqCircuitsBase.jl.
- `CliffordTBasis()` synthesizes arbitrary rotations with `GridSynthRewrite()`, and `CliffordTBasis(sk_depth=d)` with `SolovayKitaevRewrite(d)`.
- `CliffordTBasis` stores the rotation rule in its `rotations` attribute.
- `circuit.proto` declares `OperationType.QubitLoss = 16` again, marked deprecated and read only when decoding old files, matching `MimiqCircuitsBase.jl`, with no change on the wire.
- `CliffordTBasis` keeps as terminal every operation that `StimBasis` keeps, plus `GateT`, `GateTDG`, classical operations and any conditional of its terminal operations, so measurements and resets in every Pauli basis, Pauli noise, `GateHXY`, `GateHYZ` and Pauli rotations by ±π/2 are no longer expanded.
- `Depolarizing.getparams()` returns `[p]` and `parnames` is `("p",)`, without the number of qubits, as in MimiqCircuitsBase.jl.
- `GateCustom.getparams()` and `GateCustomDiagonal.getparams()` return the matrix entries in column-major order and the diagonal entries, with their 1-based indices as `parnames`, as in MimiqCircuitsBase.jl, so `is_symbolic` sees symbolic entries.
- `Circuit.push` and `Block.push` raise `ValueError` for an operator that is not a gate, such as `Projector0`, `SigmaMinus`, `DiagonalOp` or `RescaledGate`, as MimiqCircuitsBase.jl does.
- `Power` and `Gate.power` store a float exponent that is exactly a fraction with denominator at most 1000 as a `Fraction`, or as an `int`, so `Power(GateCX(), 0.5)` saves the same bytes as in MimiqCircuitsBase.jl and `GateP(pi).power(0.5)` is `GateP(pi/2)`.
- `GateU.inverse()` returns `+0.0` instead of `-0.0` for a zero float angle, as in MimiqCircuitsBase.jl.
- Symbolic quotients with a non-numeric denominator, such as `1/x` or `x/y`, are saved to protobuf as `DIV(num, den)`, with the same bytes as MimiqCircuitsBase.jl; files saved earlier still load.

### Removed
- The `sk_depth` attribute of `CliffordTBasis` is removed; use `basis.rotations.depth` instead.

### Fixed
- `extract_projection` removes an absorbed measurement from `quantum_circuit` even when a later gate on other qubits follows it. It used to keep every instruction up to the last kept one, so a final `Measure` that fusion or another reordering pass moved ahead of such a gate stayed there, and the circuit ran one evolution per shot instead of sampling. Brickwork circuit, 18 qubits, `fuse=True`, 1000 shots on qsim: 28 s to 0.07 s.
- `extract_projection` keeps the projection as wide as the source register when it drops a dead write. Dropping the only instruction on the highest classical bit used to shrink `projection.num_bits()`, so the sampling path returned `cstates` shorter than the trajectory path for the same circuit.
- `Circuit.draw` wraps an instruction onto the next block when it would reach the last column, where it used to overwrite the end of the wire.
- `fuse_threshold` is no longer silently inert when `fuse` is unset: on its own
  it selects fusion at that threshold, and with `fuse=False` it raises
  `ValueError` instead of being ignored.

- `QCSResults.saveproto` now writes amplitude keys in the same bit order as `loadproto` and MimiqCircuitsBase.jl read them.
- `matrix()` on a parameter-free gate now returns a copy, so writing into the result no longer changes what later calls return.
- `Barrier`, `QubitCoordinates` and `ShiftCoordinates` survive `copy`, `deepcopy` and `pickle`; they used to come back as a `LazyExpr`, which broke `Circuit.deepcopy()` and the non-in-place `add_noise` and `decorate` on circuits containing them.
- `Circuit.insert` of a `Circuit` at a negative index now keeps the inserted instructions together and in order.
- Inserting a circuit into itself with `Circuit.insert` no longer loops forever.
- A `decorate_on_match_*` generator that returns a `Circuit` now inserts its instructions once per match, instead of decorating them again or looping forever with `before=True`.
- `MeasureXX` and `MeasureYY` decompose into circuits that measure `X⊗X` and `Y⊗Y`, where `MeasureXX` measured `Z⊗Z` and `MeasureYY` left an extra `X⊗X` on the state.
- A multi-controlled X with five or more controls decomposes into the right unitary.
- `PolynomialOracle` and a `Power` whose base does not decompose into a single gate raise `DecompositionError`, where decomposing them looped forever.
- `Power(Control(n, U), p)` decomposes into `Control(n, Power(U, p))`, and a `Power` of `PhaseGradient` into phase gates.
- A nested `Control` returns the named gate of the flattened operation, so `Control(1, GateCX())` is a `GateCCX`.
- `GateU(0, 0, 0, γ).power(p)` and other diagonal `GateU` powers return finite angles instead of NaN, so a multi-controlled phase-only `GateU` decomposes.
- `StimBasis` keeps `Depolarizing1` and `Depolarizing2` as terminal, as `MimiqCircuitsBase.jl` does, instead of raising `DecompositionError`.
- `SpecialAngleRewrite` treats a non-finite angle as not special instead of raising.
- Symbolic parameters built from integers, rationals, `π` and `ℯ`, such as `π/4`, `2π/3` or `x*π/4`, are saved to protobuf exactly and with the same bytes in Julia and Python, so a loaded circuit equals the saved one.
- `StimBasis` drops a phase-only `GateU(0, 0, 0, γ)`, such as the one `RPauli("II", θ)` decomposes into, instead of raising `DecompositionError`, as `CliffordTBasis` does.
- `GateRX`, `GateRY`, `GateRZ`, `GateR`, `GateP` and `GateU1` with a symbolic parameter, and `GateU`, can be raised to a `Fraction` power instead of raising `TypeError`.

### Build
- The package includes a compiled Rust extension, `mimiqcircuits._core`, and builds with maturin instead of hatchling.
- Wheels are platform-specific: one abi3 wheel per platform for CPython 3.10 and newer, plus a `cp314t` wheel for free-threaded CPython 3.14.
- Installing from the sdist, as on platforms without a published wheel, requires Rust 1.85 or newer.
- The sdist leaves out the CI configuration, `scripts/` and the logo.
- The `protobuf` requirement is raised to `>=6.33,<7`, since the checked-in `*_pb2.py` modules fail to import under older runtimes.

### CI
- Test and documentation jobs install a Rust toolchain, and the GitLab test job runs the Rust unit tests.
- The PyPI workflow builds the wheels and the sdist with maturin.
- GitLab jobs cache the Rust toolchain and the cargo registry.

### Docs
- The installation guide and the README state that installing from source needs a Rust toolchain.
- The API reference documents the decomposition bases and rewrite rules of `mimiqcircuits.decomposition`.
- The circuits manual shows how to decompose a circuit to `CliffordTBasis` and choose the gridsynth tolerance.

### Migration
Files that earlier versions saved with amplitudes load with each key's bits mirrored within every byte. The file does not record which encoder wrote it, so repair only files you know predate this release, by reversing the bits of each key byte:

```python
from mimiqcircuits.proto import qcsresults_pb2

msg = qcsresults_pb2.QCSResults()
with open("old.pb", "rb") as f:
    msg.ParseFromString(f.read())
for entry in msg.amplitudes:
    entry.key.data = bytes(int(f"{b:08b}"[::-1], 2) for b in entry.key.data)
with open("fixed.pb", "wb") as f:
    f.write(msg.SerializeToString())
```

`QASMBasis` keeps the same operations terminal as MimiqCircuitsBase.jl. Code that expected `GateSX`, `GateP`, `GateECR` or noise channels to pass through it unchanged should decompose with `CanonicalBasis` or strip the noise first.

`CliffordTBasis()` now approximates arbitrary rotations with `GridSynthRewrite()` instead of Solovay-Kitaev, so decomposed circuits change: they are shorter and more accurate. Pass `sk_depth=3` to get the previous output.

The output of `CliffordTBasis` can now contain the Stim operations listed under **Changed**; code that expects only the elementary Clifford+T gates must expand them itself.

`Circuit.push` no longer accepts operators that are not gates. Use them inside `ExpectationValue`, a Kraus channel or another operation that takes an operator. Code that rebuilds a `Depolarizing` from `getparams()` must pass the number of qubits separately, for example `Depolarizing(op.num_qubits, *op.getparams())`.

## [0.28.0] - 2026-09-22

### Added
- `CompiledProjection` (`mimiqcircuits.backends`) compiles a projection circuit once and runs it over a whole block of shots, where `evaluate_projection` interprets it per shot. `LocalBackend._execute_sampling` uses it. GHZ 100 qubits, 1024 shots: 145 ms to under 1 ms.
- `evaluate_projection` accepts `And`, `Or`, `Xor` and `ParityCheck`, the operations of the classical language it was missing. It stays the reference `CompiledProjection` is tested against.
- `MimiqConnection(url, auth_url)` takes an authentication URL and opens a `mimiqlink.QhiveConnection` when one is given, or when `url` is `QPERFECT_DEV`. Quantum Hive authenticates through Keycloak rather than the MIMIQ cloud login, so the two need different connections; every other URL still gets `mimiqlink.MimiqConnection` and behaves as before.

### Changed
- `BitString.__setitem__` copies the buffer as a `bitarray` rather than through a Python list. Still O(n), since the buffer stays frozen for `__hash__`: 200 writes on a 200-bit string, 0.36 ms to 0.09 ms.
- `extract_projection` absorbs a trailing `Not`, `And`, `Or`, `Xor`, `ParityCheck`, `SetBit0` or `SetBit1`, where it absorbed only `Measure` and `MeasureReset` before. **This changes which path such a circuit takes**: it evolves once and projects per shot, where it used to evolve once per shot. GHZ 24 qubits, 1000 shots with a `Not`: 527 ms to 7 ms. See `Upgrading` below.
- `BitString` shares an argument that is already a `frozenbitarray` rather than copying it. The buffer's bit order is kept as given, as the copying path also did, so anything reading the raw buffer must check `.endian`.
- A write to a classical bit that the projection overwrites without reading is dropped, so reusing a classical bit no longer costs a per-shot write that nothing observes.

### Fixed
- `_is_writing_op` reads a wrapper's own bit and z-var footprint before unwrapping. `IfStatement`, `PairMeasure` and `ExpectationValue` write a bit or a z-var while wrapping an operator that writes nothing, so they reported no write and did not block their bits: a measurement feeding an `IfStatement`'s condition could be absorbed past it. Matches `AbstractQCSs.jl`, which fixed this in 0.21.7.
- `get_results` reports the server's reason when a remote job errors on Quantum Hive. Its API returns a `messages` list where the MIMIQ cloud returns `errorMessage`, so the failure came back as the bare "Remote job errored."

### Upgrading
- A circuit ending in measurements plus classical logic now takes the sampling path. `results.fidelities` and `results.avggateerrors` carry one entry rather than `nsamples`, and a seeded run gives different samples, because the two paths consume randomness differently.

### Build
- `mimiqlink` is required at `>=0.9,<0.10`, its first release with `QhiveConnection`.

## [0.27.1] — 2026-09-07

`bitarray` is now required at `>=3.10`, up from `>=2.9`. `BitString` only uses the
`bitarray`/`frozenbitarray` constructors, so nothing in the public API changes, but an
environment pinned to `bitarray 2.x` will no longer resolve.

### Changed
- `bitarray` is required at `>=3.10,<4`, its first release with supported free-threading — it declares itself safe to run without the GIL, so it no longer forces the interpreter back into GIL-mode on import. The 2.x line also publishes no wheel for the free-threaded ABI, so the old `<3` bound made installing on Python 3.14t build `bitarray` from source, which needs a C compiler on the user's machine.
- `symengine` is required at `>=0.14.1,<0.15`. The floor is its first release publishing a wheel for the free-threaded CPython ABI (`cp314t`); the cap closes at the next version a `0.y.z` dependency is allowed to break in, so an incompatible release fails to resolve instead of failing at runtime.

### CI
- The test job runs against Python 3.10, 3.13 and 3.14t, where it previously ran only the image's 3.13.

### Build
- The package installs and passes its suite on free-threaded CPython 3.14 (`python3.14t`). Free-threading is not yet in effect there: `symengine` does not declare itself safe to run without the GIL, so importing `mimiqcircuits` re-enables it and the interpreter falls back to GIL-mode execution. The declaration is merged upstream but unreleased as of `symengine 0.14.1`, so lifting this needs a `symengine` release plus a deliberate bump of the bound above, with the suite run against it. Setting `PYTHON_GIL=0` overrides the fallback and keeps the GIL off; the released wheels are built with `WITH_SYMENGINE_THREAD_SAFE`, so this overrides a stale declaration rather than running unprotected code, but it is process-wide and applies to every extension.

## [0.27.0] — 2026-08-19

### Added
- `GateCustomDiagonal(d)`, an `n`-qubit gate given by the `2**n` entries of its diagonal, as `GateCustom` is given by a full matrix. Storing only the diagonal keeps a wide diagonal block at `2**n` numbers rather than `4**n`; entries must be phases. It decomposes exactly into parity rotations (`GateRNZ`/`GateRZ`) through a Walsh-Hadamard expansion of its phases, so backends that do not know the gate still handle it.
- `fuse_circuit` and `FusePass` take `max_diagonal_support` (default: `max_support`), the width budget for runs where every gate is diagonal in the computational basis. Such runs now fuse into a `GateCustomDiagonal` and are composed by multiplying diagonals elementwise, never building the dense block. A cluster keeps the diagonal budget only while all of its gates are diagonal; the first dense gate joining it brings it back under `max_support`. With the default budget the emitted blocks cover the same gates as before, but an all-diagonal block comes out as `GateCustomDiagonal` rather than `GateCustom`.
- Wire format 1.2.0: the `circuit.proto` schema gains a `CustomDiagonalGate` message, carried by the `Gate`, `Operator` and `Operation` oneofs. Additive, so an older decoder ignores it and a newer one still reads old payloads; a client emitting `GateCustomDiagonal` needs an executor speaking 1.2.0.

### Fixed
- The decomposition of `GateCustom` is exact: it used to drop a global phase and, on matrices with degenerate singular values such as diagonal ones, could put a wrong phase on a single amplitude (fidelity as low as 0.53).

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
- `GateCustom` checks unitarity with NumPy for numeric matrices, so wide blocks such as those `fuse_circuit` emits are built much faster (9 qubits: about 15 s to under 0.5 s).

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
