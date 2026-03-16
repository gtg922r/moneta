# Moneta — Phase 1 Implementation Plan

**Date:** 2026-03-16
**Spec:** `_SPEC.md` (reviewed)
**Target:** Working `moneta run` CLI with Monte Carlo simulation, queries, and reports

---

## Implementation Strategy

### Build Order: Bottom-Up, Test-First

Each step builds one layer of the system, from foundation to integration. Every step produces a self-contained, tested commit. Tests are written alongside (or before) the code they cover.

```
  DEPENDENCY GRAPH (build order is left-to-right, top-to-bottom)

  Step 1          Step 2              Step 3
  scaffolding ──▶ types + models ──▶  loader + presets
                                         │
                                    Step 4│
                                    state + ResultStore
                                    ┌────┴────┐
                              Step 5a       Step 5b
                              growth +      events +
                              inflation     transfers
                                    └────┬────┘
                                    Step 6│
                                    orchestrator
                                    ┌────┴────┐
                              Step 7│         │
                              query engine    │
                              ┌────┴────┐     │
                        Step 8a       Step 8b │
                        terminal      HTML    │
                        output        report  │
                              └────┬────┘     │
                                   └──────────┘
                                    Step 9│
                                    CLI
                                    Step 10│
                                    sweep mode
                                    Step 11│
                                    e2e tests + examples
```

### Commit Philosophy

1. **Each step = one atomic commit** with passing tests
2. **Tests in the same commit** as the code they test
3. **No forward references** — each commit only uses code that exists
4. **Run full test suite after each commit** to catch regressions
5. **Git init at Step 1** — every step builds on a real, versioned history

### Sub-Agent Strategy

Sub-agents protect the main context window while implementing each step. Here's the approach:

**When to use sub-agents:**
- Each implementation step (1-11) is delegated to a sub-agent
- The sub-agent gets: `_SPEC.md`, `_PHASE_1_PLAN.md`, and relevant existing source files
- After each agent completes, the main context reviews the diff and runs tests

**Sequential vs parallel:**
- **Steps 5a + 5b** (growth/inflation + events/transfers) can run in **parallel** — they're independent processors that both depend only on `engine/state.py`
- **Steps 8a + 8b** (terminal + HTML report) can run in **parallel** — they're independent output renderers that both depend on query results
- **All other steps run sequentially** — each depends on the previous step's output

**Context provided to each sub-agent:**
```
  EVERY agent gets:
    - _SPEC.md (full spec)
    - _PHASE_1_PLAN.md (this file — their step's section)
    - pyproject.toml (dependencies)

  ADDITIONALLY, each agent gets:
    - Source files it depends on (e.g., Step 6 gets state.py + all processors)
    - Test fixtures and conftest.py
    - The specific step instructions from this plan
```

**Feedback loop between steps:**
- After each step, run `uv run pytest` to verify all tests pass
- If a step reveals a design issue (e.g., the state dataclass needs an extra field), update this plan and the spec before continuing
- Steps 6 (orchestrator) and 9 (CLI) are **checkpoint steps** — pause and verify the system works end-to-end before proceeding

---

## Step-by-Step Plan

### Step 1: Project Scaffolding

**Goal:** Working `uv` project with directory structure, dependencies, and pytest running.

**Files created:**
```
moneta/
├── pyproject.toml
├── src/moneta/__init__.py
├── src/moneta/parser/__init__.py
├── src/moneta/engine/__init__.py
├── src/moneta/engine/processors/__init__.py
├── src/moneta/query/__init__.py
├── src/moneta/output/__init__.py
├── src/moneta/presets/__init__.py
├── tests/__init__.py
├── tests/conftest.py
└── .gitignore
```

**Details:**
- `uv init` with `src` layout
- `pyproject.toml` with all dependencies from spec
- `conftest.py` with shared fixtures: `seeded_rng` (returns `np.random.default_rng(42)`)
- `git init` + initial commit
- All `__init__.py` files are empty (just establish package structure)

**Validation:** `uv run pytest` runs and finds 0 tests, exits 0.

**Commit message:** `feat: initialize project with uv, dependencies, and package structure`

---

### Step 2: Parser — Custom Types + Pydantic Models

**Goal:** All Pydantic custom types and model classes. Loading a YAML string into a fully validated `ScenarioModel` works.

**Files created/modified:**
```
src/moneta/parser/types.py      # Custom annotated types
src/moneta/parser/models.py     # Pydantic model classes
tests/test_parser/test_types.py # Type parsing tests
tests/test_parser/test_models.py # Model validation tests
```

**parser/types.py — Pydantic custom types:**
- `AnnualRate` — parses `"7% annually"`, `"3% monthly"`, or float passthrough
- `Duration` — parses `"10 years"`, `"6 months"`, returns int (months)
- `ProbabilityWindow` — parses `"20% within 3 years"`, `"60% within 5-6 years"`
- `MultiplierRange` — parses `["2x", "5x"]` → `(2.0, 5.0)`
- `CurrencyAmount` — parses `"$850,000"` or numeric passthrough
- All implemented as `Annotated[T, BeforeValidator(parse_fn)]`
- All parse functions use `float()` (locale-independent, `.` decimal only)

**parser/models.py — Pydantic models:**
```python
class ScenarioConfig(BaseModel):
    name: str
    time_horizon: Duration
    simulations: int = 10000
    time_step: Literal["monthly"] = "monthly"
    seed: int | None = None

class GrowthConfig(BaseModel):
    model: Literal["gbm"]
    expected_return: AnnualRate
    volatility: AnnualRate

# ... OR preset reference:
class PresetGrowthConfig(BaseModel):
    preset: str

class InflationConfig(BaseModel):
    model: Literal["mean_reverting"]
    long_term_rate: AnnualRate
    volatility: AnnualRate
    mean_reversion_speed: float = 0.5  # θ, reasonable default

class PresetInflationConfig(BaseModel):
    preset: str

class LiquidityEvent(BaseModel):
    probability: ProbabilityWindow
    valuation_range: MultiplierRange

class TransferConfig(BaseModel):
    transfer_to: str  # validated against asset registry later

class InvestmentAsset(BaseModel):
    type: Literal["investment"]
    initial_balance: CurrencyAmount
    growth: GrowthConfig | PresetGrowthConfig

class IlliquidEquityAsset(BaseModel):
    type: Literal["illiquid_equity"]
    current_valuation: CurrencyAmount
    shares: int | None = None
    liquidity_events: list[LiquidityEvent]
    on_liquidation: TransferConfig

Asset = Annotated[InvestmentAsset | IlliquidEquityAsset, Field(discriminator="type")]

class GlobalConfig(BaseModel):
    inflation: InflationConfig | PresetInflationConfig

class ProbabilityQuery(BaseModel): ...
class PercentilesQuery(BaseModel): ...
class ExpectedQuery(BaseModel): ...
class DistributionQuery(BaseModel): ...
Query = Annotated[...query union..., Field(discriminator="type")]

class SweepScenario(BaseModel):
    label: str
    overrides: dict  # raw dict, deep-merged with base model

class SweepConfig(BaseModel):
    scenarios: list[SweepScenario]

class ScenarioModel(BaseModel):
    """Top-level model for a .moneta.yaml file."""
    scenario: ScenarioConfig
    assets: dict[str, Asset]
    global_config: GlobalConfig = Field(alias="global")
    queries: list[Query]
    sweep: SweepConfig | None = None
```

**Cross-reference validation (model_validator):**
- All `transfer_to` references point to existing asset names
- No self-transfers (asset transferring to itself)
- No circular transfers
- All query `of` / expression asset names exist
- Query `at` times don't exceed `time_horizon`

**Tests (test_types.py):**
- Each type: valid string → correct value, invalid string → ValidationError
- Edge cases: 0%, negative values, missing units, wrong types
- Passthrough: numeric values pass through unchanged

**Tests (test_models.py):**
- Valid complete model → parses successfully
- Missing required field → specific error message
- Unknown asset type → discriminator error
- Bad reference in transfer_to → validation error
- Self-transfer → validation error
- Query time > horizon → validation error

**Commit message:** `feat: parser types and Pydantic models with discriminated unions and validation`

---

### Step 3: Parser — Loader + Presets

**Goal:** Load a `.moneta.yaml` file from disk, resolve presets, and return a validated `ScenarioModel`.

**Files created/modified:**
```
src/moneta/parser/loader.py
src/moneta/presets/__init__.py
src/moneta/presets/data/sp500.yaml
src/moneta/presets/data/us_inflation.yaml
src/moneta/presets/data/us_treasuries.yaml
src/moneta/presets/data/tech_startup_equity.yaml
src/moneta/presets/data/total_market.yaml
tests/test_parser/test_loader.py
tests/fixtures/simple_model.moneta.yaml
tests/fixtures/equity_model.moneta.yaml
```

**parser/loader.py:**
- `load_model(path: Path) -> ScenarioModel` — main entry point
- `_load_yaml(path)` — `yaml.safe_load()`, handle FileNotFoundError, encoding errors
- `_resolve_presets(raw_dict)` — walk the dict, replace `preset: name` with preset config
- `_deep_merge(base, overrides)` — for sweep mode (recursive dict merge)

**presets/__init__.py:**
- `get_preset(name: str) -> dict` — load bundled YAML from `data/` directory
- Use `importlib.resources` to load bundled YAML files
- Known presets: sp500, us_inflation, us_treasuries, tech_startup_equity, total_market

**Preset YAML files:** Each contains the parameters that would replace the `preset:` reference.
```yaml
# presets/data/sp500.yaml
model: gbm
expected_return: 7% annually
volatility: 15% annually
```

**Tests (test_loader.py):**
- Load valid model file → ScenarioModel with correct values
- Load model with presets → presets resolved correctly
- File not found → clear error
- Malformed YAML → clear error
- Unknown preset → clear error listing available presets

**Commit message:** `feat: YAML loader with preset resolution and bundled preset library`

---

### Step 4: Engine — SimulationState + ResultStore

**Goal:** State and result data structures, fully specified, with initialization from a ScenarioModel.

**Files created/modified:**
```
src/moneta/engine/state.py
tests/test_engine/test_state.py
```

**engine/state.py:**
```python
@dataclass
class SimulationState:
    balances: np.ndarray           # float64[n_runs, n_assets]
    events_fired: np.ndarray       # bool[n_runs, n_events]
    inflation_rate: np.ndarray     # float64[n_runs]
    cum_inflation: np.ndarray      # float64[n_runs]
    step: int
    asset_names: list[str]
    asset_index: dict[str, int]

    @classmethod
    def from_scenario(cls, model: ScenarioModel, n_runs: int) -> "SimulationState":
        # Initialize balances from asset configs
        # Initialize events_fired to all False
        # Initialize inflation_rate to initial rate from config
        # Initialize cum_inflation to 1.0

@dataclass
class ResultStore:
    balances: np.ndarray           # float64[n_runs, n_steps, n_assets]
    cum_inflation: np.ndarray      # float64[n_runs, n_steps]
    event_fired_at: np.ndarray     # int32[n_runs, n_events]
    asset_names: list[str]
    asset_index: dict[str, int]
    n_runs: int
    n_steps: int
    n_assets: int

    @classmethod
    def allocate(cls, model: ScenarioModel, n_runs: int) -> "ResultStore":
        # Pre-allocate all arrays with correct shapes

    def record(self, state: SimulationState, step: int) -> None:
        # Copy current state slice into the result arrays
```

**Tests:**
- `from_scenario` creates correct shapes
- Initial balances match asset configs
- `allocate` creates correctly shaped zero arrays
- `record` copies state correctly at given step index

**Commit message:** `feat: SimulationState and ResultStore with pre-allocation and initialization`

---

### Step 5a: Engine — Growth + Inflation Processors

**Goal:** GBM growth and Ornstein-Uhlenbeck inflation processors, fully tested.

> **Can run in parallel with Step 5b** — independent processors, both depend only on state.py

**Files created/modified:**
```
src/moneta/engine/processors/__init__.py  # Processor protocol
src/moneta/engine/processors/growth.py
src/moneta/engine/processors/inflation.py
tests/test_engine/test_growth.py
tests/test_engine/test_inflation.py
```

**processors/__init__.py:**
```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Processor(Protocol):
    def step(self, state: SimulationState, dt: float, rng: np.random.Generator) -> None: ...
```

**processors/growth.py (GBM):**
```python
class GrowthProcessor:
    def __init__(self, asset_configs: dict[str, GrowthConfig]):
        # Pre-compute per-asset drift and vol arrays
        # Only apply growth to investment-type assets

    def step(self, state, dt, rng):
        # S(t+dt) = S(t) * exp((μ - σ²/2)·dt + σ·√dt·Z)
        # Z = rng.standard_normal(size=(n_runs, n_growth_assets))
        # Vectorized multiply on the growth-asset columns of state.balances
```

**processors/inflation.py (Ornstein-Uhlenbeck):**
```python
class InflationProcessor:
    def __init__(self, config: InflationConfig):
        self.mu = config.long_term_rate
        self.theta = config.mean_reversion_speed
        self.sigma = config.volatility

    def step(self, state, dt, rng):
        # dx = θ·(μ - x)·dt + σ·√dt·Z
        # state.inflation_rate += dx
        # state.cum_inflation *= (1 + state.inflation_rate * dt)
```

**Tests (test_growth.py):**
- Seeded deterministic: known seed → exact balance after 1 step
- Statistical (100K runs, 120 steps): mean return within 1% of theoretical
- Statistical: std within 5% of theoretical
- Zero balance stays zero
- Extreme vol (10000%) → no NaN
- Growth only applies to investment-type columns

**Tests (test_inflation.py):**
- Seeded deterministic: exact rate after 1 step
- Statistical: mean rate converges toward long_term_rate
- Cumulative inflation tracks correctly
- Can go negative (deflation)
- Extreme parameters → no NaN

**Commit message:** `feat: GBM growth and Ornstein-Uhlenbeck inflation processors with statistical tests`

---

### Step 5b: Engine — Events + Transfers Processors

**Goal:** Hazard-rate event processor and transfer processor, fully tested.

> **Can run in parallel with Step 5a** — independent processors, both depend only on state.py

**Files created/modified:**
```
src/moneta/engine/processors/events.py
src/moneta/engine/processors/transfer.py
tests/test_engine/test_events.py
tests/test_engine/test_transfer.py
```

**processors/events.py (Hazard Rate):**
```python
class EventProcessor:
    def __init__(self, events: list[EventConfig]):
        # Pre-compute hazard rates from probability windows
        # h = 1 - (1 - p)^(1/window_months)

    def step(self, state, dt, rng):
        # For each event that hasn't fired:
        #   if current step is within the event's window:
        #     draw U ~ Uniform(0,1) for all runs
        #     fire where U < h AND NOT events_fired
        #     set events_fired = True for fired runs
        #     draw liquidation value from multiplier range
        #     set balance = current_valuation * multiplier for fired runs
```

**processors/transfer.py:**
```python
class TransferProcessor:
    def __init__(self, transfers: list[TransferConfig]):
        # Map: (source_asset_idx, dest_asset_idx, event_idx)

    def step(self, state, dt, rng):
        # For each transfer:
        #   newly_fired = events_fired AND NOT already_transferred
        #   state.balances[newly_fired, dest] += state.balances[newly_fired, source]
        #   state.balances[newly_fired, source] = 0
```

**Tests (test_events.py):**
- Seeded: event fires at expected step
- Statistical: fire frequency matches probability within 95% CI
- Event fires at most once per run
- Multiple events on same asset: first wins
- 100% probability → fires in every run (within window)
- 0% probability → never fires
- Overlapping windows handled correctly
- Valuation drawn from range bounds

**Tests (test_transfer.py):**
- Transfer moves balance on event fire
- Source zeroed, dest increased
- Balance conservation: sum unchanged
- No transfer if event hasn't fired
- Transfer only happens once per event

**Commit message:** `feat: hazard-rate event and transfer processors with statistical tests`

---

### Step 6: Engine — Orchestrator *(CHECKPOINT)*

**Goal:** Full simulation loop. Given a `ScenarioModel`, produce a `ResultStore` with correct data.

**Files created/modified:**
```
src/moneta/engine/orchestrator.py
tests/test_engine/test_orchestrator.py
```

**engine/orchestrator.py:**
```python
def build_pipeline(model: ScenarioModel) -> list[Processor]:
    """Build the fixed processor pipeline from the scenario model."""
    # 1. EventProcessor (if any events defined)
    # 2. TransferProcessor (if any transfers defined)
    # 3. GrowthProcessor (if any growth configs)
    # 4. InflationProcessor

def run_simulation(model: ScenarioModel, seed: int | None = None) -> ResultStore:
    n_runs = model.scenario.simulations
    rng = np.random.default_rng(seed)
    state = SimulationState.from_scenario(model, n_runs)
    results = ResultStore.allocate(model, n_runs)
    pipeline = build_pipeline(model)

    total_steps = model.scenario.time_horizon  # already in months
    for t in range(total_steps):
        state.step = t
        for processor in pipeline:
            processor.step(state, dt=1/12, rng=rng)
        results.record(state, t)

    return results
```

**Tests (test_orchestrator.py):**
- Simple model (1 asset, GBM only): results have correct shape
- Pipeline runs processors in correct order (verified via mock)
- Seeded run produces bit-identical results on re-run
- Results recorded at each time step
- Model with no events: pipeline skips EventProcessor/TransferProcessor

**CHECKPOINT VALIDATION:**
```bash
# At this point we can run a full simulation and inspect the result arrays.
# This is the first end-to-end test of the engine.
uv run python -c "
from moneta.parser.loader import load_model
from moneta.engine.orchestrator import run_simulation
model = load_model('tests/fixtures/equity_model.moneta.yaml')
results = run_simulation(model, seed=42)
print(f'Shape: {results.balances.shape}')
print(f'Final p50: {np.median(results.balances[:, -1, 0]):.0f}')
"
```

**Commit message:** `feat: Monte Carlo orchestrator with fixed pipeline and seeded execution`

---

### Step 7: Query Engine

**Goal:** Parse query expressions and evaluate all query types against a ResultStore.

**Files created/modified:**
```
src/moneta/query/expressions.py
src/moneta/query/engine.py
tests/test_query/test_expressions.py
tests/test_query/test_engine.py
```

**query/expressions.py — Recursive descent parser:**
```
Grammar:
  expr     → compare
  compare  → additive (('>' | '<' | '>=' | '<=') additive)?
  additive → term (('+' | '-') term)*
  term     → factor (('*' | '/') factor)*
  factor   → NUMBER | ASSET_NAME | '(' expr ')'
```
- Tokenizer: split on whitespace and operators, recognize numbers and asset names
- Parser: recursive descent, returns an AST (dataclass nodes)
- Evaluator: walks AST, resolves asset names via `asset_index`, operates on `float64[n_runs]` arrays

**query/engine.py:**
```python
def evaluate_queries(
    queries: list[Query],
    results: ResultStore,
) -> list[QueryResult]:
    """Evaluate all queries against the result store."""

class QueryResult:
    label: str
    query_type: str
    # Type-specific results:
    probability: float | None        # for probability queries
    percentiles: dict[int, float] | None  # for percentile queries (by time point)
    expected: ExpectedResult | None  # mean, median, std
    distribution: DistributionResult | None  # histogram bins + counts
```

- **Probability:** count `expression_true / n_runs` at the specified step
- **Percentiles:** `np.percentile(values, [10, 25, 50, 75, 90])` at each time point
- **Expected:** `np.mean`, `np.median`, `np.std`
- **Distribution:** `np.histogram(values, bins=N)`
- **Inflation adjustment:** `values / cum_inflation[:, step]` before computing

**Tests (test_expressions.py):**
- Parse each operator: +, -, *, /, >, <, >=, <=
- Asset name resolution to array columns
- Operator precedence: `a + b > c` parsed as `(a + b) > c`
- Unknown asset → clear error with available names
- Malformed expression → clear error

**Tests (test_engine.py):**
- Construct a known ResultStore (hand-crafted arrays)
- Probability query: 60 of 100 runs above threshold → 60%
- Percentile query: known data → exact percentiles
- Expected query: known data → exact mean/median/std
- Distribution query: known data → correct histogram
- Inflation adjustment: values divided by cum_inflation

**Commit message:** `feat: query expression parser and evaluation engine with inflation adjustment`

---

### Step 8a: Terminal Output

**Goal:** Rich-formatted terminal output for query results.

> **Can run in parallel with Step 8b**

**Files created/modified:**
```
src/moneta/output/terminal.py
tests/test_output/test_terminal.py
```

**output/terminal.py:**
- `render_results(results: list[QueryResult], timing: float, scenario: ScenarioConfig) -> str`
- Format probability results as `label .............. XX.X% probability`
- Format percentile tables with Rich Table
- Format currency: `$1.23M`, `$612K`, `$1,234`
- Format sweep comparison table
- Header: `─── Moneta ── N simulations ── X year horizon ── Yms ───`

**Tests (test_terminal.py):**
- Currency formatting: various ranges ($612K, $1.23M, $23.5M)
- Probability result formatting
- Percentile table formatting
- Sweep comparison table with multiple scenarios

**Commit message:** `feat: Rich terminal output with currency formatting and percentile tables`

---

### Step 8b: HTML Report

**Goal:** Interactive Plotly HTML report generation.

> **Can run in parallel with Step 8a**

**Files created/modified:**
```
src/moneta/output/report.py
tests/test_output/test_report.py
```

**output/report.py:**
- `generate_report(results: ResultStore, query_results: list[QueryResult], output_path: Path)`
- **Fan chart:** Compute p10/p25/p50/p75/p90 at each time step, plot as filled areas
- **Distribution histogram:** At queried time points, plot histogram of values
- **Sample paths:** Plot 5-10 random individual simulation paths
- **Probability timeline:** For probability queries, compute probability at every step (not just the queried one)
- Self-contained HTML (Plotly.js bundled inline)
- For sweep mode: overlaid fan charts + comparison table

**Tests (test_report.py):**
- **Data-layer tests:** Fan chart data computation returns correct percentile arrays
- **Data-layer tests:** Histogram computation returns correct bins/counts
- **Data-layer tests:** Sample path selection returns valid path indices
- **Snapshot test:** Generate report with fixed seed, compare HTML to golden file
- Report file is valid HTML (basic structural check)

**Commit message:** `feat: interactive Plotly HTML report with fan charts, histograms, and sample paths`

---

### Step 9: CLI *(CHECKPOINT)*

**Goal:** Working `moneta run` and `moneta validate` commands.

**Files created/modified:**
```
src/moneta/cli.py
tests/test_integration/test_end_to_end.py
tests/fixtures/golden/  (golden output files)
```

**cli.py:**
```python
@click.group()
def main(): ...

@main.command()
@click.argument("model_file", type=click.Path(exists=True))
@click.option("--simulations", type=int, default=None)
@click.option("--seed", type=int, default=None)
@click.option("--output", type=click.Path(), default="./output")
@click.option("--no-report", is_flag=True)
@click.option("--verbose", is_flag=True)
@click.option("--format", "output_format", type=click.Choice(["table", "json", "csv"]), default="table")
def run(model_file, simulations, seed, output, no_report, verbose, output_format):
    """Run Monte Carlo simulation on a model file."""
    # 1. Load model
    # 2. Override simulations/seed if provided
    # 3. Run simulation (with timing)
    # 4. Evaluate queries
    # 5. Render terminal output
    # 6. Generate HTML report (unless --no-report)
    # 7. Handle errors: catch MonetaError, display friendly message

@main.command()
@click.argument("model_file", type=click.Path(exists=True))
def validate(model_file):
    """Validate a model file without running simulation."""
```

**Error handling wrapper:**
- Define `MonetaError` base exception
- CLI catches it, prints friendly message, exits 1
- With `--verbose`, print full traceback

**Integration tests (test_end_to_end.py):**
- `moneta run simple_model.moneta.yaml --seed 42` → exit 0, output matches golden file
- `moneta run equity_model.moneta.yaml --seed 42` → output matches golden file
- `moneta validate simple_model.moneta.yaml` → exit 0
- `moneta validate nonexistent.yaml` → exit 1, error message
- `moneta run bad_model.yaml` → exit 1, validation error
- `moneta run model.yaml --format json` → valid JSON output
- `moneta run model.yaml --no-report` → no HTML file generated

**CHECKPOINT VALIDATION:**
```bash
uv run moneta run tests/fixtures/equity_model.moneta.yaml --seed 42
# Should produce terminal output + HTML report
```

**Commit message:** `feat: CLI with run and validate commands, golden-file integration tests`

---

### Step 10: Sweep Mode

**Goal:** Named scenario sweep with comparison output.

**Files created/modified:**
```
src/moneta/parser/loader.py      # add deep_merge
src/moneta/engine/orchestrator.py # add run_sweep
src/moneta/output/terminal.py     # add comparison table
src/moneta/output/report.py       # add overlaid charts
tests/fixtures/sweep_model.moneta.yaml
tests/test_integration/test_end_to_end.py  # add sweep tests
```

**Changes:**
- `loader.py` — `deep_merge(base_dict, overrides_dict)`: recursive dict merge. List values are replaced, not appended. For each sweep scenario, deep-merge overrides onto the base model dict, then re-parse through Pydantic.
- `orchestrator.py` — `run_sweep(model, seed) -> list[tuple[str, ResultStore]]`: run simulation for each scenario, return labeled results. Share the same base seed, offset by scenario index for independence.
- `terminal.py` — comparison table: for each query, show results across all scenarios in a Rich Table.
- `report.py` — overlaid fan charts: one set of percentile bands per scenario, different colors, with legend.

**Tests:**
- Sweep with 2 scenarios: both produce results, comparison table rendered
- Deep merge: nested override replaces correctly, non-overridden fields preserved
- Sweep with no scenarios: treated as single-scenario run
- Golden file test for sweep output

**Commit message:** `feat: named scenario sweep mode with comparison output`

---

### Step 11: Examples + Final Polish

**Goal:** Example model files, final integration tests, cleanup.

**Files created/modified:**
```
examples/retirement_basic.moneta.yaml
examples/retirement_with_equity.moneta.yaml
examples/scenario_comparison.moneta.yaml
```

**Examples:**
1. **retirement_basic** — Single investment portfolio with GBM growth. Simple percentile query.
2. **retirement_with_equity** — Investment portfolio + startup equity with liquidity events. Probability + percentile queries with inflation adjustment.
3. **scenario_comparison** — Same as #2 but with sweep mode: conservative/base/optimistic equity outcomes.

**Polish:**
- Verify all examples run successfully
- Ensure `--verbose` shows useful timing info
- Ensure error messages are helpful for common mistakes
- Final pass: run full test suite, check coverage
- Update any stale comments or docstrings

**Commit message:** `feat: example model files and final integration polish`

---

## Validation Checklist

After all steps complete, verify:

```
  ┌────────────────────────────────────────────────────────────────┐
  │  VALIDATION CHECKLIST                                          │
  ├────────────────────────────────────────────────────────────────┤
  │  □  uv run pytest — all tests pass                             │
  │  □  uv run pytest --cov — coverage >80%                        │
  │  □  moneta run examples/retirement_basic.moneta.yaml           │
  │  □  moneta run examples/retirement_with_equity.moneta.yaml     │
  │  □  moneta run examples/scenario_comparison.moneta.yaml        │
  │  □  moneta validate examples/retirement_basic.moneta.yaml      │
  │  □  moneta run nonexistent.yaml — friendly error               │
  │  □  moneta run --seed 42 ... — deterministic output            │
  │  □  moneta run --format json ... — valid JSON                  │
  │  □  moneta run --no-report ... — no HTML file                  │
  │  □  HTML report opens in browser, charts are interactive       │
  │  □  Sweep mode produces comparison table + overlaid charts     │
  │  □  All presets load correctly                                 │
  │  □  Git log shows clean, atomic commit history                 │
  └────────────────────────────────────────────────────────────────┘
```

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Pydantic custom types harder than expected | Medium | Low | Fall back to explicit validators. Same result, more verbose. |
| Query expression parser edge cases | Medium | Medium | Keep grammar simple. Defer complex expressions to Phase 2. |
| Plotly HTML report too large (>5MB) | Low | Low | Use `plotly.io.to_html(full_html=True, include_plotlyjs='cdn')` as fallback. |
| Statistical tests flaky | Medium | Low | Use large N (100K), wide confidence intervals, and fixed seeds. |
| NumPy version compatibility | Low | Low | Pin `numpy>=2.0` which is stable. |
| Hazard rate math edge cases | Medium | Medium | Comprehensive unit tests for boundary probabilities (0%, 100%, overlapping windows). |
