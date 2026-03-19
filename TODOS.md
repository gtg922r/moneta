# Moneta — Deferred Work

## Infrastructure

### Migrate from mypy to ty

**What:** Replace mypy with Astral's ty type checker for all-Astral toolchain (uv + ruff + ty).

**Why:** ty is Rust-based and significantly faster than mypy. Unifies the toolchain under one ecosystem (Astral) for simpler dependency management and consistent behavior.

**Context:** As of March 2026, ty (v0.0.23) has an 81% false positive rate on Pydantic v2 codebases due to incomplete support for TypeVar defaults, ParamSpec, and TypedDict narrowing. The Pydantic team is tracking this at pydantic/pydantic-ai#3970. When ty reaches 1.0 with Pydantic v2 compatibility verified, swap `mypy` for `ty` in `pyproject.toml` dev deps and update the CI typecheck job from `mypy src/` to `ty check src/`. Remove `[tool.mypy]` and `[tool.pydantic-mypy]` config sections, add `[tool.ty]` section.

**Effort:** S
**Priority:** P3
**Depends on:** ty 1.0 release with Pydantic v2 compatibility

## Streaming Aggregation for Large Simulation Counts

**What:** Compute query statistics (percentiles, probabilities) incrementally without storing the full `(N_runs, T_steps, N_assets)` result matrix in memory.

**Why:** The current ResultStore pre-allocates the full array. At 100K runs × 360 steps × 5 assets × 8 bytes = 1.44GB. At 1M runs = 14.4GB — exceeds typical machine RAM. Streaming aggregation would allow arbitrarily large run counts by computing statistics on-the-fly.

**Pros:** Enables 1M+ simulations for high-precision probability estimates. Removes memory as a scaling bottleneck.

**Cons:** More complex orchestrator loop. Some query types (distribution histograms) require knowing the full range upfront, which means either a two-pass approach or adaptive binning. Percentiles require approximate algorithms (t-digest or similar).

**Context:** Phase 1 targets 10K-100K runs which fit comfortably in memory. This becomes necessary if users want very high precision on tail probabilities (e.g., "what's the 1st percentile outcome?") or if Phase 2 adds many more asset types expanding the per-run footprint. The orchestrator loop (`engine/orchestrator.py`) would need a streaming mode where processors still mutate state, but instead of `results.record(state, t)` copying the full slice, it updates running statistics. Libraries like `welford` (online variance) or `tdigest` (approximate percentiles) could help.

**Effort:** M (medium) — algorithm design + orchestrator refactor + query engine changes
**Priority:** P3
**Depends on:** Phase 1 complete. Should be done before Phase 3's cartesian product sweeps (which multiply run count by scenario count).
