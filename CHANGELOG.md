# Changelog

All notable changes to `mimiqcircuits` (Python) are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.23.2] — 2026-05-31

### Added
- Progress reporting for `LocalBackend.execute`. Pass `progress=True`
  (or a `TqdmProgress`) to draw bars for the compression, single-shot
  execution, and trajectory phases; the default `progress=False`
  (`NoProgress`) keeps the pipeline cost-free.
- `Progress` sink interface with `NoProgress` and `TqdmProgress`
  implementations. The driver owns the trajectory bar and, on the
  single-evolve path, an execution bar driven by the backend's per-step
  `callback`; backends with a countable compression step emit their own
  stage by overriding `compile_progress`. Adds a `tqdm` dependency.

## [0.23.1] — 2026-05-28

### Fixed
- `mimiqcircuits.__version__` is now sourced from the installed
  package metadata via `importlib.metadata.version("mimiqcircuits")`
  rather than a hard-coded literal in `__version__.py`. The literal
  had drifted to `0.21.8` while `pyproject.toml` was at `0.23.0`,
  so anything reading `__version__` (e.g. the `apiversion` /
  `circuitsapiversion` envelope fields, the default `pyapi_v…`
  job label) was lying. Single source of truth now lives in
  `pyproject.toml`.

## [0.23.0] — 2026-05-27

### Added
- `mimiqcircuits.WIRE_FORMAT_VERSION` constant (initial value
  `"1.0.0"`) declaring the version of the MIMIQ wire format —
  the union of the ProtoBuf schemas and the JSON request/response
  envelope — independently of the package's own release version.
  Mirrors `MimiqCircuitsBase.WIRE_FORMAT_VERSION` byte-for-byte. See
  `WIRE_FORMAT.md` in `MimiqCircuitsBase.jl/` for the full surface
  and the bump-trigger checklist.

### Fixed
- `Backend.can_handle` now rejects circuits with noise channels when
  the backend has not declared `"noise"`, and circuits with free
  symbolic parameters when the backend has not declared `"parametric"`.
  Previously both slipped past admission and surfaced as opaque
  evolve-time errors. Mirrors the matching tightening in
  `AbstractQCSs.jl`.

### Changed
- The JSON request envelope for `submit` and `optimize` now carries a
  `wireformatversion` key (the value of
  `mimiqcircuits.WIRE_FORMAT_VERSION`) alongside the existing
  `circuitsapiversion`. Executors aware of the new field will use it
  for compatibility checks; older executors ignore it.
- `circuitproto.toproto_circuit` is ~1.4-2x faster. `ProtoRegistry`
  converter lookup is memoized (MRO walk runs once per type, cache
  cleared on registration); `toproto_arg`'s handler dict is hoisted to
  module scope so it isn't rebuilt per parameter; gate converters write
  the `Operation` proto directly instead of building a `Gate` and
  unpacking it (`Control`, `Power`, `Inverse`, `Parallel`,
  `PauliString`, `GateCustom`, `GateCall`, `RPauli`, and everything in
  `GATEMAP` / `GENERALIZEDGATEMAP`).

## [0.21.8]

Changelog tracking begins with this version. See git history for prior changes.
