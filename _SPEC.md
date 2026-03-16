# Moneta — Product Specification

**Version:** 0.1.0 (Phase 1)
**Date:** 2026-03-16
**Status:** Reviewed

## Vision

Moneta is a **probabilistic financial modeling engine** — a local-first personal tool for defining your current financial state, modeling how it evolves over time with uncertainty, and asking probabilistic questions about future outcomes.

Instead of spreadsheet projections that pretend the future is certain, Moneta runs thousands of Monte Carlo simulations to give you distributions: "There's a 43% chance your net worth exceeds $2M in 10 years" — not "You'll have $1.8M."

## Core Principles

1. **Uncertainty is a feature, not a bug.** Every projection is a distribution, not a point estimate.
2. **Text-based, version-controlled models.** Financial scenarios are YAML files you can diff, share, and iterate on.
3. **Local-first, single-user.** Runs on your machine. No servers, no accounts, no data leaves your laptop.
4. **Extensible engine.** New financial capabilities (tax, income, expenses) are pluggable processors — no core rewrites.
5. **Ask questions, get probabilities.** The query system turns simulation results into actionable answers.

---

## Architecture

### System Overview

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                         MONETA SYSTEM                               │
  │                                                                     │
  │  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
  │  │  MODEL FILE   │───▶│    PARSER    │───▶│   SCENARIO            │  │
  │  │  (.moneta.yaml)│   │              │    │                       │  │
  │  │  - scenario   │    │  - YAML load │    │  assets (typed union) │  │
  │  │  - assets     │    │  - Pydantic  │    │  inflation config     │  │
  │  │  - global     │    │    custom    │    │  queries              │  │
  │  │  - queries    │    │    types     │    │  sweep scenarios      │  │
  │  │  - sweep      │    │  - presets   │    │  asset registry       │  │
  │  └──────────────┘    └──────────────┘    └──────────┬────────────┘  │
  │                                                      │              │
  │                                            ┌─────────▼─────────┐    │
  │                                            │  SIMULATION ENGINE │    │
  │                                            │                   │    │
  │  ┌──────────────┐                          │  ┌─────────────┐  │    │
  │  │  PROCESSORS   │                         │  │ Orchestrator │  │    │
  │  │  (4 total)    │                         │  │             │  │    │
  │  │               │    ┌─────────────────┐  │  │ for t in T: │  │    │
  │  │ 1.events      │    │  MONTE CARLO    │  │  │  processors │  │    │
  │  │ 2.transfers   │◀───│  ORCHESTRATOR   │◀─│  │  .step()    │  │    │
  │  │ 3.growth      │    │                 │  │  │  record(t)  │  │    │
  │  │ 4.inflation   │    │  N runs         │  │  └─────────────┘  │    │
  │  │               │    │  vectorized     │  │                   │    │
  │  │ mutate state  │    │  seeded RNG     │  │                   │    │
  │  │ in-place      │    └────────┬────────┘  └───────────────────┘    │
  │  └──────────────┘             │                                    │
  │                      ┌─────────▼─────────┐                          │
  │                      │  RESULT STORE      │                          │
  │                      │  balances[N,T,A]   │                          │
  │                      │  cum_infl[N,T]     │                          │
  │                      │  event_timing[N,E] │                          │
  │                      └─────────┬─────────┘                          │
  │                                │                                    │
  │                ┌───────────────┼───────────────┐                    │
  │                ▼               ▼               ▼                    │
  │  ┌──────────────────┐ ┌──────────────┐ ┌────────────────┐          │
  │  │  QUERY ENGINE     │ │  TERMINAL    │ │  HTML REPORT   │          │
  │  │                   │ │  OUTPUT      │ │  (Plotly)      │          │
  │  │  •probability()   │ │  (Rich)      │ │                │          │
  │  │  •percentiles()   │ │              │ │  •distributions│          │
  │  │  •expected()      │ │  •summary    │ │  •fan charts   │          │
  │  │  •distribution()  │ │  •queries    │ │  •comparisons  │          │
  │  └──────────────────┘ │  •timing     │ │  •sample paths │          │
  │                        └──────────────┘ └────────────────┘          │
  └─────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.12+ | Best ecosystem for numerics + plotting. NumPy vectorization = C speed for array ops. |
| Package manager | `uv` | Fast dependency resolution, built-in venv, lockfile support. No pip/conda friction. |
| Numerics | NumPy | Vectorized Monte Carlo — all runs simulated as matrix operations. |
| Model format | YAML + Pydantic custom types | Readable YAML. Annotated values (`7% annually`) parsed via Pydantic `BeforeValidator`. |
| Validation | Pydantic v2 | Parse-time validation with discriminated unions for asset types. Clear error messages. |
| Terminal output | Rich | Formatted tables, progress bars, colored output. |
| Visualization | Plotly | Interactive HTML reports with hover, zoom, scenario toggle. |
| CLI framework | Click | Explicit, widely known, decorator-based routing. |
| Testing | pytest | With seeded RNG for deterministic + statistical tests. |
| Distribution | PyPI via `uv publish` | Install with `uv tool install moneta` or `uvx moneta`. |

### Simulation Pipeline (per time step)

Each monthly time step executes 4 processors in a fixed, deterministic order. The orchestrator then snapshots state into the result store.

```
  ┌─────────────────────────────────────────────────────────┐
  │  Per time step t:                                       │
  │                                                         │
  │  1. EVENTS      → Check if discrete events fire         │
  │  2. TRANSFERS   → Execute triggered asset transfers     │
  │  3. GROWTH      → Apply stochastic growth models        │
  │  4. INFLATION   → Step inflation process, update cum    │
  │  ─────────────────────────────────────────────────────  │
  │  5. RECORD      → Orchestrator snapshots state (not a   │
  │                    processor — orchestrator's job)       │
  └─────────────────────────────────────────────────────────┘
```

Processors **mutate state in-place** — no copying per step. The orchestrator copies the current slice into the pre-allocated result store after all processors run. This order is documented and invariant. Liquidation proceeds receive growth starting the *next* month.

---

## Model File Format

Models are defined in `.moneta.yaml` files with five top-level sections: `scenario`, `assets`, `global`, `queries`, and `sweep`.

### Full Example

```yaml
# retirement_projection.moneta.yaml

scenario:
  name: "Retirement projection — base case"
  time_horizon: 30 years
  simulations: 10000
  time_step: monthly
  seed: 42                          # optional, for reproducibility

assets:
  investment_portfolio:
    type: investment
    initial_balance: 850000
    growth:
      model: gbm                    # geometric Brownian motion
      expected_return: 7% annually
      volatility: 15% annually

  startup_equity:
    type: illiquid_equity
    current_valuation: 500000
    shares: 50000

    liquidity_events:
      - probability: 20% within 3 years
        valuation_range: [2x, 5x]
      - probability: 60% within 5-6 years
        valuation_range: [3x, 10x]

    on_liquidation:
      transfer_to: investment_portfolio

global:
  inflation:
    model: mean_reverting           # Ornstein-Uhlenbeck process
    long_term_rate: 3% annually
    volatility: 1% annually

queries:
  - type: probability
    expression: investment_portfolio + startup_equity > 2000000
    at: 10 years
    label: "$2M net worth at year 10"

  - type: percentiles
    values: [10, 25, 50, 75, 90]
    of: investment_portfolio
    at: [5, 10, 15, 20, 25, 30] years
    adjust_for: inflation
    label: "Portfolio value distribution (real $)"

  - type: probability
    expression: investment_portfolio > 1500000
    at: 15 years
    adjust_for: inflation
    label: "$1.5M portfolio (real $) at year 15"

sweep:
  scenarios:
    - label: conservative
      overrides:
        assets:
          startup_equity:
            liquidity_events:
              - probability: 20% within 3 years
                valuation_range: [1x, 3x]
              - probability: 60% within 5-6 years
                valuation_range: [2x, 5x]
    - label: optimistic
      overrides:
        assets:
          startup_equity:
            liquidity_events:
              - probability: 20% within 3 years
                valuation_range: [3x, 10x]
              - probability: 60% within 5-6 years
                valuation_range: [5x, 20x]
```

### Presets

Common stochastic models are available as built-in presets:

```yaml
assets:
  portfolio:
    type: investment
    initial_balance: 850000
    growth:
      preset: sp500                  # 7% real return, 15% vol (GBM)

global:
  inflation:
    preset: us_inflation             # 3% long-term, 1% vol (mean-reverting)
```

**Phase 1 presets:**

| Preset | Model | Parameters | Source |
|--------|-------|------------|--------|
| `sp500` | GBM | μ=7%, σ=15% annually | Historical S&P 500 real returns |
| `us_inflation` | Mean-reverting | μ=3%, σ=1% annually | Historical CPI |
| `us_treasuries` | Mean-reverting | μ=4%, σ=3% annually | Historical 10Y yield |
| `tech_startup_equity` | Hazard-rate | See below | Industry data |
| `total_market` | GBM | μ=7%, σ=12% annually | Total US stock market |

### Annotated Value Types (Pydantic Custom Types)

Model file values like `7% annually` are parsed by Pydantic custom types via `BeforeValidator`, not a separate expression parser. Each field's type annotation determines the parsing grammar:

```
  TYPE                    PARSES                          RESULT
  ─────────────────────── ─────────────────────────────── ────────────────────
  AnnualRate              "7% annually"                   0.07
                          "3% monthly"                    0.03 (monthly)
                          0.07                            0.07 (passthrough)
  Duration                "10 years"                      120 (months)
                          "6 months"                      6
                          360                             360 (passthrough)
  ProbabilityWindow       "20% within 3 years"            (0.20, 0, 36)
                          "60% within 5-6 years"          (0.60, 60, 72)
  MultiplierRange         [2x, 5x]                        (2.0, 5.0)
  CurrencyAmount          "$850,000"                      850000
                          850000                          850000 (passthrough)
```

All parsing uses Python `float()` (locale-independent). Model files must use `.` as the decimal separator.

### Query Expressions

Queries support basic arithmetic on asset names. These are parsed by a real expression parser (recursive descent) since they have operators, variables, and comparisons:

```yaml
queries:
  - type: probability
    expression: investment_portfolio + startup_equity > 2000000
    at: 10 years

  - type: probability
    expression: investment_portfolio - initial_investment > 0
    at: 5 years
    label: "Positive real return at 5 years"
```

Supported operators: `+`, `-`, `*`, `/`, `>`, `<`, `>=`, `<=`
Supported references: asset names, `initial_<asset>` for starting values

---

## Stochastic Models

### Geometric Brownian Motion (GBM)

Used for market returns (investments, equities). Models growth with log-normal distribution:

```
  dS = μ·S·dt + σ·S·dW

  where:
    S  = asset value
    μ  = expected return (drift)
    σ  = volatility
    dW = Wiener process increment
    dt = time step (1/12 year for monthly)
```

**Implementation:** Vectorized across all runs using the exact solution:

```python
S(t+dt) = S(t) * exp((μ - σ²/2)·dt + σ·√dt·Z)
where Z ~ N(0,1)
```

NumPy generates all Z values for all runs in a single call per time step.

### Mean-Reverting Process (Ornstein-Uhlenbeck)

Used for inflation, interest rates. Tends toward a long-term mean:

```
  dx = θ·(μ - x)·dt + σ·dW

  where:
    x  = current rate
    μ  = long-term mean rate
    θ  = mean-reversion speed (how fast it reverts)
    σ  = volatility
```

Inflation "meanders" but gravitates toward the long-term average. Can go negative (deflation) — this is realistic and allowed.

### Hazard-Rate Events

Used for discrete probabilistic events (liquidity events, job changes, etc.). Each time step has a conditional probability of the event occurring, calibrated to match user-specified cumulative probabilities.

```
  User specifies: "20% probability within 3 years"

  Engine computes: monthly hazard rate h such that
    1 - (1-h)^36 = 0.20
    h ≈ 0.00617 per month

  Each month, for each simulation run:
    draw U ~ Uniform(0,1)
    if U < h and event hasn't fired yet → event fires
```

For multiple events on the same asset (e.g., "20% in 3 years OR 60% in 5-6 years"), the engine handles overlapping windows. Once any event fires, the asset is liquidated and subsequent events are no-ops.

**Valuation on liquidation:** When an event fires, the liquidation value is drawn from the specified range (uniform distribution over the multiplier range applied to current valuation).

---

## Engine Architecture

### Processor Protocol

Each engine capability implements a simple protocol. Processors **mutate state in-place** and return nothing:

```python
from typing import Protocol
import numpy as np

class Processor(Protocol):
    def step(
        self,
        state: SimulationState,
        dt: float,
        rng: np.random.Generator,
    ) -> None:
        """Mutate state by one time step. All runs processed simultaneously."""
        ...
```

### SimulationState (fully specified)

```python
@dataclass
class SimulationState:
    """Mutable working state for all simulation runs.

    All arrays have first dimension = n_runs. Processors mutate in-place.
    The orchestrator snapshots into ResultStore after each time step.
    """
    # Asset balances: current value of each asset in each run
    balances: np.ndarray           # float64[n_runs, n_assets]

    # Event tracking: which events have fired in each run
    events_fired: np.ndarray       # bool[n_runs, n_events]

    # Inflation process state
    inflation_rate: np.ndarray     # float64[n_runs] — current annual rate
    cum_inflation: np.ndarray      # float64[n_runs] — cumulative factor (starts at 1.0)

    # Metadata
    step: int                      # current time step index
    asset_names: list[str]         # ordered asset names (column mapping)
    asset_index: dict[str, int]    # name → column index (shared with query engine)
```

### ResultStore (fully specified)

```python
@dataclass
class ResultStore:
    """Pre-allocated storage for simulation results.

    All arrays allocated upfront. Orchestrator fills one time slice per step.
    Query engine reads from these arrays.
    """
    # Main timeseries: asset balances at each time step
    balances: np.ndarray           # float64[n_runs, n_steps, n_assets]

    # Inflation: cumulative factor at each step (for real-dollar queries)
    cum_inflation: np.ndarray      # float64[n_runs, n_steps]

    # Event timing: which month each event fired (-1 if never)
    event_fired_at: np.ndarray     # int32[n_runs, n_events]

    # Metadata
    asset_names: list[str]
    asset_index: dict[str, int]    # shared with SimulationState
    n_runs: int
    n_steps: int
    n_assets: int
```

### Phase 1 Processors

| Processor | Responsibility | Mutates |
|-----------|---------------|---------|
| `EventProcessor` | Fire discrete events based on hazard rates | `events_fired`, `balances` (sets liquidation value) |
| `TransferProcessor` | Move value between assets on event triggers | `balances` (source → dest) |
| `GrowthProcessor` | Apply GBM growth to investment-type assets | `balances` |
| `InflationProcessor` | Step O-U process, update cumulative factor | `inflation_rate`, `cum_inflation` |

### Monte Carlo Orchestrator

```python
def run_simulation(scenario: Scenario, seed: int | None = None) -> ResultStore:
    rng = np.random.default_rng(seed)
    state = SimulationState.from_scenario(scenario)
    results = ResultStore.allocate(scenario)     # pre-allocate all arrays

    for t in range(scenario.total_steps):
        state.step = t
        for processor in scenario.pipeline:
            processor.step(state, dt=1/12, rng=rng)
        results.record(state, t)                 # copy current slice into results

    return results
```

For sweep mode, the orchestrator runs once per scenario variant (sequentially), collecting results for comparison. Memory: one `ResultStore` in memory at a time per scenario.

### Pydantic Model Types (Discriminated Unions)

Asset types use Pydantic discriminated unions for type-safe, per-type validation:

```python
class InvestmentAsset(BaseModel):
    type: Literal["investment"]
    initial_balance: CurrencyAmount
    growth: GrowthConfig

class IlliquidEquityAsset(BaseModel):
    type: Literal["illiquid_equity"]
    current_valuation: CurrencyAmount
    shares: int | None = None
    liquidity_events: list[LiquidityEvent]
    on_liquidation: TransferConfig

# Pydantic routes to correct model based on "type" field
Asset = Annotated[
    InvestmentAsset | IlliquidEquityAsset,
    Field(discriminator="type")
]
```

Phase 2 extensibility: add new asset types (e.g., `IncomeStreamAsset`) to the union.

### Vectorization Strategy

The engine does NOT loop over individual simulation runs. All runs are processed simultaneously as NumPy array operations:

```
  Shape: (n_runs, n_assets)

  Example: 10,000 runs × 5 assets
  - Growth: single NumPy multiply across (10000, 5) array
  - Events: single uniform draw of shape (10000,), compare to threshold
  - Transfer: masked assignment on the (10000, 5) array

  Total for 10K runs × 30 years × 12 months:
  = 360 vectorized operations, each touching 50K floats
  = milliseconds on modern hardware
```

---

## Query Engine

The query engine operates on the completed `ResultStore`.

### Query Types

**Probability queries:**
```yaml
- type: probability
  expression: portfolio + equity > 2000000
  at: 10 years
```
→ Counts the fraction of runs where the expression is true at the specified time.

**Percentile queries:**
```yaml
- type: percentiles
  values: [10, 25, 50, 75, 90]
  of: portfolio
  at: [5, 10, 15, 20, 25, 30] years
  adjust_for: inflation
```
→ Computes percentiles of the asset value distribution at each specified time.

**Expected value queries:**
```yaml
- type: expected
  of: portfolio + equity
  at: 10 years
  adjust_for: inflation
```
→ Returns mean, median, and standard deviation.

**Distribution queries:**
```yaml
- type: distribution
  of: portfolio
  at: 10 years
  bins: 50
```
→ Full histogram of outcomes for the HTML report.

### Inflation Adjustment

When `adjust_for: inflation` is specified, values are divided by `cum_inflation` for each run at the queried time step. This converts nominal dollars to real (present-day) dollars. Since inflation varies per run, the adjustment itself is stochastic.

### Asset Name Resolution

Both the model validator and the query engine resolve asset names via a shared `asset_index: dict[str, int]` mapping. Built once during model parsing, passed to both the validator (checking `transfer_to` references) and the query expression evaluator (resolving names to result array columns).

---

## Output

### Terminal Output (Rich)

Every run produces a terminal summary:

```
$ moneta run retirement_projection.moneta.yaml

─── Moneta ── 10,000 simulations ── 30 year horizon ── 142ms ───

Query Results:
  $2M net worth at year 10 .............. 43.2% probability
  $1.5M portfolio (real $) at year 15 ... 61.8% probability

Portfolio value distribution (real $):
  ┌─────────┬────────┬────────┬────────┬────────┬────────┐
  │         │  p10   │  p25   │  p50   │  p75   │  p90   │
  ├─────────┼────────┼────────┼────────┼────────┼────────┤
  │  5 yr   │ $612K  │ $742K  │ $921K  │ $1.14M │ $1.41M │
  │ 10 yr   │ $891K  │ $1.12M │ $1.48M │ $1.95M │ $2.56M │
  │ 15 yr   │ $1.08M │ $1.51M │ $2.18M │ $3.14M │ $4.49M │
  │ 20 yr   │ $1.22M │ $1.94M │ $3.12M │ $5.01M │ $7.98M │
  │ 25 yr   │ $1.31M │ $2.41M │ $4.33M │ $7.77M │ $13.8M │
  │ 30 yr   │ $1.35M │ $2.93M │ $5.92M │ $11.9M │ $23.5M │
  └─────────┴────────┴────────┴────────┴────────┴────────┘

📈 Full report: ./output/retirement_projection_report.html
```

When using sweep mode, the terminal shows a comparison table:

```
Scenario Comparison:
  ┌───────────────┬────────────────────────────┬──────────────────┐
  │ Scenario      │ $2M at yr 10               │ p50 at yr 10     │
  ├───────────────┼────────────────────────────┼──────────────────┤
  │ conservative  │ 28.1%                      │ $1.12M           │
  │ base          │ 43.2%                      │ $1.48M           │
  │ optimistic    │ 59.7%                      │ $1.91M           │
  │ moonshot      │ 74.3%                      │ $2.64M           │
  └───────────────┴────────────────────────────┴──────────────────┘
```

### HTML Report (Plotly)

Each run generates a self-contained interactive HTML file with:

1. **Fan charts** — Asset value over time showing p10/p25/p50/p75/p90 bands
2. **Distribution histograms** — Full outcome distribution at queried time points
3. **Probability timeline** — How the probability of a query changes over time
4. **Scenario comparison** (sweep mode) — Overlaid fan charts and comparison tables
5. **Sample paths** — A handful of individual simulation paths to build intuition

The report is self-contained (no external dependencies) and opens in the default browser.

---

## CLI Interface

```
moneta run <model.moneta.yaml> [options]

Options:
  --simulations N     Override simulation count (default: from model file)
  --seed N            Set random seed for reproducibility
  --output DIR        Output directory (default: ./output/)
  --no-report         Skip HTML report generation
  --verbose           Show timing details and debug info
  --format FORMAT     Terminal output format: table (default), json, csv
```

### Examples

```bash
# Run a model
moneta run retirement.moneta.yaml

# Quick iteration with fewer sims
moneta run retirement.moneta.yaml --simulations 1000

# Reproducible run
moneta run retirement.moneta.yaml --seed 42

# JSON output for piping to other tools
moneta run retirement.moneta.yaml --format json | jq '.queries[0].probability'

# Validate a model file without running
moneta validate retirement.moneta.yaml
```

---

## Sweep Mode (Named Scenarios)

The `sweep` block defines named scenario variants with explicit YAML overrides:

```yaml
sweep:
  scenarios:
    - label: conservative
      overrides:
        assets:
          startup_equity:
            liquidity_events:
              - probability: 20% within 3 years
                valuation_range: [1x, 3x]
              - probability: 60% within 5-6 years
                valuation_range: [2x, 5x]
    - label: optimistic
      overrides:
        assets:
          startup_equity:
            liquidity_events:
              - probability: 20% within 3 years
                valuation_range: [3x, 10x]
              - probability: 60% within 5-6 years
                valuation_range: [5x, 20x]
```

Each scenario is the base model with its overrides deep-merged on top. The orchestrator runs each scenario sequentially (one `ResultStore` in memory at a time), then aggregates query results for comparison.

The HTML report in sweep mode includes:
- Overlaid fan charts (one band per scenario)
- Comparison table (all queries × all scenarios)

Cartesian product sweeps (varying multiple parameters across a grid) are deferred to Phase 3.

---

## Project Structure

```
moneta/
├── pyproject.toml              # uv project config, dependencies, CLI entry point
├── uv.lock                     # lockfile
├── src/
│   └── moneta/
│       ├── __init__.py
│       ├── cli.py              # CLI entry point (Click)
│       ├── parser/
│       │   ├── __init__.py
│       │   ├── loader.py       # YAML loading, preset resolution, deep-merge
│       │   ├── types.py        # Pydantic custom types (AnnualRate, Duration, etc.)
│       │   └── models.py       # Pydantic models (Scenario, Asset union, Query, etc.)
│       ├── engine/
│       │   ├── __init__.py
│       │   ├── orchestrator.py # Monte Carlo loop + recording
│       │   ├── state.py        # SimulationState + ResultStore dataclasses
│       │   └── processors/
│       │       ├── __init__.py # Processor protocol + processor registry
│       │       ├── growth.py   # GBM and other growth models
│       │       ├── events.py   # Hazard-rate discrete events
│       │       ├── inflation.py# Mean-reverting inflation (O-U process)
│       │       └── transfer.py # Asset-to-asset transfers on event triggers
│       ├── query/
│       │   ├── __init__.py
│       │   ├── engine.py       # Query evaluation on ResultStore
│       │   └── expressions.py  # Query expression parser (recursive descent)
│       ├── output/
│       │   ├── __init__.py
│       │   ├── terminal.py     # Rich terminal output
│       │   └── report.py       # Plotly HTML report generator
│       └── presets/
│           ├── __init__.py     # Preset lookup (load bundled YAML)
│           └── data/
│               ├── sp500.yaml
│               ├── us_inflation.yaml
│               ├── us_treasuries.yaml
│               ├── tech_startup_equity.yaml
│               └── total_market.yaml
├── tests/
│   ├── conftest.py             # Shared fixtures, seeded RNG helper
│   ├── test_parser/
│   │   ├── test_types.py       # Custom type parsing (AnnualRate, Duration, etc.)
│   │   ├── test_models.py      # Model validation, discriminated unions, references
│   │   └── test_loader.py      # YAML loading, preset resolution, deep-merge
│   ├── test_engine/
│   │   ├── test_orchestrator.py# Pipeline order, recording, sweep execution
│   │   ├── test_growth.py      # GBM: deterministic + statistical tests
│   │   ├── test_events.py      # Hazard rate: deterministic + statistical tests
│   │   ├── test_inflation.py   # O-U process: deterministic + statistical tests
│   │   └── test_transfer.py    # Balance conservation, event-triggered transfers
│   ├── test_query/
│   │   ├── test_engine.py      # Query evaluation on known result matrices
│   │   └── test_expressions.py # Expression parsing, operator precedence, errors
│   ├── test_output/
│   │   ├── test_terminal.py    # Terminal formatting, currency display
│   │   └── test_report.py      # Report data structures + HTML snapshot tests
│   ├── test_integration/
│   │   └── test_end_to_end.py  # Golden file tests with fixed seeds
│   └── fixtures/
│       ├── simple_model.moneta.yaml
│       ├── equity_model.moneta.yaml
│       ├── sweep_model.moneta.yaml
│       └── golden/             # Expected outputs for golden tests
├── examples/
│   ├── retirement_basic.moneta.yaml
│   ├── retirement_with_equity.moneta.yaml
│   └── scenario_comparison.moneta.yaml
├── _SPEC.md                    # This file
└── _PHASE_1_PLAN.md            # Implementation plan
```

---

## Error Handling Philosophy

Moneta errors should read like a helpful colleague, not a stack trace.

### Principles

1. **Validate early.** Catch all model file errors at parse time, before simulation.
2. **Point to the source.** Every error references the file, line, and field.
3. **Suggest fixes.** Typo detection, "did you mean?" suggestions, valid value hints.
4. **Never show internals.** No Python tracebacks in normal operation (`--verbose` shows them).

### Error Categories

| Phase | Error | User Sees |
|-------|-------|-----------|
| Parse | Malformed YAML | `retirement.moneta.yaml:12 — invalid YAML syntax` |
| Parse | Invalid expression | `retirement.moneta.yaml:8 — can't parse '7% anually'. Did you mean '7% annually'?` |
| Validate | Missing field | `Asset 'portfolio' (type: investment) missing required field 'initial_balance'` |
| Validate | Bad reference | `transfer_to: 'foo' — no asset named 'foo'. Available: investment_portfolio, startup_equity` |
| Validate | Self-transfer | `Asset 'equity' transfers to itself — this would zero the balance` |
| Validate | Circular flow | `Circular transfer detected: A → B → A` |
| Validate | Invalid probability | `Probability must be 0-100%, got '150%'` |
| Engine | Numeric overflow | `Simulation halted: NaN detected in asset 'portfolio' at month 247. Check volatility parameters.` |
| Query | Impossible time | `Query asks about year 40 but time_horizon is 30 years` |
| Query | Unknown asset | `Query references 'savings' but no asset with that name exists` |
| Output | Can't write | `Can't write to ./output/: permission denied` |

---

## Testing Strategy

### Test Pyramid

```
  ┌─────────────────────┐
  │   Integration (few) │  Golden file: known model + fixed seed = exact output
  │                     │  HTML report: snapshot + data-layer tests
  ├─────────────────────┤
  │  Statistical (some) │  Run N sims, assert distribution properties within
  │                     │  confidence intervals (mean, variance, event freq)
  ├─────────────────────┤
  │    Unit (many)      │  Type parsing, model validation, query eval,
  │                     │  terminal formatting — all deterministic
  └─────────────────────┘
```

### Seeded RNG for Reproducibility

All stochastic tests use `np.random.default_rng(seed=42)`. Golden-file integration tests produce bit-identical results across runs.

### Statistical Tests

For stochastic processors (GBM, inflation, events):
- Run 100K simulations with known parameters
- Assert sample mean within 1% of theoretical mean
- Assert sample std within 5% of theoretical std
- Assert event frequency within 95% confidence interval of specified probability

### Chaos Tests

- 10000% volatility — no crash, no NaN
- 99.9% event probability — fires in essentially every run
- 0.001% event probability — fires in essentially no runs
- 1M simulations — completes without OOM
- Empty model (no assets) — clean error message
- 100-year horizon — completes in reasonable time

---

## Phase 1 Scope

### In Scope

- [x] YAML parser with Pydantic custom types for annotated values
- [x] Discriminated union asset types (investment, illiquid_equity)
- [x] GBM growth processor
- [x] Mean-reverting inflation processor (Ornstein-Uhlenbeck)
- [x] Hazard-rate discrete event processor
- [x] Asset transfer processor
- [x] Monte Carlo orchestrator (vectorized, seeded, in-place mutation)
- [x] Fully specified SimulationState and ResultStore
- [x] Query engine (probability, percentiles, expected, distribution)
- [x] Query expression parser (recursive descent)
- [x] Inflation-adjusted queries
- [x] Terminal output (Rich)
- [x] Interactive HTML report (Plotly) with fan charts, histograms, sample paths
- [x] Sweep mode with named scenarios (YAML deep-merge)
- [x] Built-in presets (sp500, us_inflation, us_treasuries, tech_startup_equity, total_market)
- [x] CLI with `run` and `validate` commands (Click)
- [x] Comprehensive test suite (unit + statistical + integration + chaos)
- [x] Example model files

### NOT in Scope (Phase 1)

| Item | Rationale | Phase |
|------|-----------|-------|
| Tax engine (income + capital gains) | Major new processor — needs its own design cycle | 2 |
| Income / expense streams | Requires recurring cash flow modeling | 2 |
| Sensitivity analysis | Useful but not core — requires meta-simulation | 2 |
| Cartesian product sweeps | Named scenarios sufficient for Phase 1 | 3 |
| Goal-seeking queries | Optimization problem, not simulation | 3 |
| Scenario diffing | Side-by-side model comparison | 3 |
| Real portfolio import (CSV/API) | Integration complexity | 3 |
| Custom time steps (weekly, daily) | Monthly sufficient for financial planning | 3 |
| Correlated assets (multivariate) | Requires joint distributions | 2 |
| Model file imports (!include) | Adds file resolution complexity | 3 |
| Natural language queries | LLM integration | 4 |
| Web UI | Local tool, CLI-first | 4 |

---

## Dependencies

```toml
[project]
name = "moneta"
requires-python = ">=3.12"
dependencies = [
    "numpy>=2.0",
    "pyyaml>=6.0",
    "pydantic>=2.0",
    "plotly>=5.0",
    "rich>=13.0",
    "click>=8.0",
]

[project.scripts]
moneta = "moneta.cli:main"

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
]
```

Minimal, well-maintained dependency tree. No exotic packages.

---

## Roadmap

```
  PHASE 1 — Core Engine (this spec)
  ├── Monte Carlo engine with GBM, events, inflation, transfers
  ├── YAML model files with Pydantic custom types
  ├── Probabilistic queries with inflation adjustment
  ├── Terminal + interactive HTML output
  ├── Named scenario sweep mode
  └── Built-in presets

  PHASE 2 — Financial Depth
  ├── Tax engine (income + capital gains, federal + state)
  ├── Income streams (salary, dividends, Social Security)
  ├── Expense modeling (fixed, variable, one-time)
  ├── Contribution modeling (401k, IRA, taxable)
  ├── Sensitivity analysis (auto-rank input impact)
  ├── Correlated assets (multivariate normal)
  └── Rebalancing strategies

  PHASE 3 — Power User
  ├── Goal-seeking ("what savings rate for 90% chance of $X?")
  ├── Cartesian product sweeps + dot-path parameter syntax
  ├── Scenario diffing (side-by-side model comparison)
  ├── Real portfolio import (CSV, brokerage APIs)
  ├── Custom processor plugin API (user-defined processors)
  └── Model file imports (!include)

  PHASE 4 — Platform
  ├── Web UI for interactive exploration
  ├── Natural language queries ("when can I retire?")
  ├── Shareable model library / community presets
  └── Time-series data feeds (real market data)
```
