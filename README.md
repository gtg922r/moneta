# Moneta

**Probabilistic financial modeling engine.** Define your financial situation, model how it evolves with uncertainty, and ask questions about the future in terms of probabilities — not false certainties.

```
$ moneta run retirement.moneta.yaml --seed 42

─── Moneta ── 10,000 simulations ── 30 year horizon ── 203ms ───

Query Results:
  $2M net worth at year 10 ................. 83.6% probability
  $1.5M portfolio (real $) at year 15 ...... 78.9% probability

Portfolio value distribution (real $):
  ┌───────┬──────────┬─────────┬────────┬────────┬─────────┐
  │       │      p10 │     p25 │    p50 │    p75 │     p90 │
  ├───────┼──────────┼─────────┼────────┼────────┼─────────┤
  │  5 yr │ $669.39K │ $833.2K │ $1.09M │ $1.59M │  $2.82M │
  │ 10 yr │ $896.83K │  $1.53M │ $3.01M │ $4.58M │  $6.21M │
  │ 15 yr │ $946.62K │  $1.72M │ $3.32M │ $5.51M │   $8.1M │
  │ 20 yr │   $1.05M │  $1.94M │ $3.77M │ $6.64M │ $10.45M │
  │ 25 yr │   $1.13M │  $2.16M │ $4.27M │ $8.01M │ $13.45M │
  │ 30 yr │   $1.24M │   $2.4M │ $4.89M │ $9.48M │ $16.76M │
  └───────┴──────────┴─────────┴────────┴────────┴─────────┘
```

## Why Moneta?

Spreadsheets give you one number: "You'll have $1.8M." That's a lie — the future is uncertain.

Moneta runs **thousands of Monte Carlo simulations** to give you distributions: *"There's an 84% chance your net worth exceeds $2M in 10 years."* It models market volatility, inflation uncertainty, probabilistic events (like a startup exit), and scheduled cash flows (income, withdrawals, expenses) — all in a simple YAML file you can version-control and iterate on.

## Install

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# Install from GitHub
uv tool install git+https://github.com/gtg922r/moneta.git

# Or run directly without installing
uvx --from git+https://github.com/gtg922r/moneta.git moneta run model.moneta.yaml
```

### Use with Claude Code (recommended)

The best way to use Moneta is with the **`/moneta` agent skill** for [Claude Code](https://claude.com/claude-code). Instead of writing YAML by hand, describe your financial situation conversationally and the agent builds the model, runs the simulation, and explains the results.

**Setup:**

```bash
# 1. Clone the repo
git clone https://github.com/gtg922r/moneta.git
cd moneta

# 2. Install dependencies
uv sync

# 3. Start Claude Code in the moneta directory
claude
```

Claude Code automatically discovers the `/moneta` skill from `.claude/skills/moneta/SKILL.md`. Type `/moneta` and describe what you want to model:

```
> /moneta

  I have a $500K investment portfolio and $300K in startup equity.
  I'm contributing $3K/month to the portfolio. The startup might
  exit in 3-5 years at 2-8x. I want to know if I can retire in
  20 years with $6K/month spending.
```

The agent will:
1. Build a `.moneta.yaml` model from your description
2. Choose appropriate presets (S&P 500 growth, US inflation)
3. Run the simulation and generate an interactive report
4. Explain the results in plain English
5. Offer to adjust parameters or add scenarios

### For development

```bash
git clone https://github.com/gtg922r/moneta.git
cd moneta
uv sync --extra dev
uv run moneta run examples/retirement_basic.moneta.yaml --seed 42
```

## Quick Start

Create a file called `my_plan.moneta.yaml`:

```yaml
scenario:
  name: "My retirement projection"
  time_horizon: 30 years
  simulations: 10000

assets:
  portfolio:
    type: investment
    initial_balance: $500,000
    growth:
      preset: sp500            # 7% return, 15% volatility (historical S&P 500)

global:
  inflation:
    preset: us_inflation       # 3% long-term, mean-reverting

cash_flows:
  contributions:
    amount: $2,000 monthly
    asset: portfolio
    end: 20 years

  retirement_spending:
    amount: -$6,000 monthly
    asset: portfolio
    start: 20 years
    adjust_for: inflation

queries:
  - type: probability
    expression: portfolio > 1000000
    at: 20 years
    label: "$1M before retirement"

  - type: percentiles
    values: [10, 25, 50, 75, 90]
    of: portfolio
    at: [10 years, 20 years, 30 years]
    adjust_for: inflation
    label: "Portfolio value (real $)"
```

Run it:

```bash
uv run moneta run my_plan.moneta.yaml --seed 42
```

## Features

### Stochastic Growth Models

Investment assets grow using **Geometric Brownian Motion** — the same model used in quantitative finance. Growth meanders realistically: some years up 20%, some down 15%, trending toward the expected return over time.

```yaml
growth:
  model: gbm
  expected_return: 7% annually
  volatility: 15% annually
```

Or use a built-in preset:

| Preset | Model | Parameters |
|--------|-------|------------|
| `sp500` | GBM | 7% return, 15% volatility |
| `total_market` | GBM | 7% return, 12% volatility |
| `us_inflation` | Mean-reverting | 3% long-term, 1% volatility |
| `us_treasuries` | Mean-reverting | 4% long-term, 3% volatility |

### Probabilistic Events

Model uncertain future events like a startup acquisition or job change. Specify the probability and timing window — Moneta calibrates the math:

```yaml
startup_equity:
  type: illiquid_equity
  current_valuation: $500,000
  liquidity_events:
    - probability: 20% within 3 years
      valuation_range: [2x, 5x]
    - probability: 60% within 5-6 years
      valuation_range: [3x, 10x]
  on_liquidation:
    transfer_to: portfolio
```

### Cash Flows

Model income, withdrawals, and one-time expenses. Amounts can be inflation-adjusted to maintain purchasing power:

```yaml
cash_flows:
  salary:
    amount: $3,000 monthly
    asset: portfolio
    end: 20 years

  retirement_spending:
    amount: -$6,000 monthly
    asset: portfolio
    start: 20 years
    adjust_for: inflation      # amount grows with simulated inflation

  college_tuition:
    amount: -$50,000 annually
    asset: portfolio
    start: 18 years
    end: 22 years
```

When withdrawals exceed the balance, Moneta clamps at zero and tracks the **shortfall** — the cumulative unmet withdrawals. Query it directly: `expression: shortfall > 0` tells you the probability of running out of money.

Set `allow_negative: true` on a cash flow to allow balances to go negative for analysis.

### Scenario Comparison

Compare different assumptions side-by-side with sweep mode:

```yaml
sweep:
  scenarios:
    - label: conservative
      overrides:
        assets:
          portfolio:
            growth:
              model: gbm
              expected_return: 4% annually
              volatility: 10% annually
    - label: aggressive
      overrides:
        assets:
          portfolio:
            growth:
              model: gbm
              expected_return: 10% annually
              volatility: 20% annually
```

```
Scenario Comparison:
  ┌───────────────┬──────────────────────────────┬─────────────┐
  │ Scenario      │ Portfolio > $500K at year 10 │ p50 at yr 5 │
  ├───────────────┼──────────────────────────────┼─────────────┤
  │ base          │                        24.6% │     $269.4K │
  │ conservative  │                         4.0% │     $238.9K │
  │ aggressive    │                        44.6% │     $302.7K │
  └───────────────┴──────────────────────────────┴─────────────┘
```

### Query Types

Ask four kinds of questions about your simulation results:

```yaml
# "What are the odds?"
- type: probability
  expression: portfolio + equity > 2000000
  at: 10 years

# "What's the range of outcomes?"
- type: percentiles
  values: [10, 25, 50, 75, 90]
  of: portfolio
  at: [5 years, 10 years, 20 years]
  adjust_for: inflation

# "What's the average?"
- type: expected
  of: portfolio
  at: 10 years

# "Show me the full distribution"
- type: distribution
  of: portfolio
  at: 10 years
```

### Interactive HTML Reports

Every run generates an interactive Plotly HTML report with fan charts, distribution histograms, sample paths, and scenario comparisons:

```bash
uv run moneta run model.moneta.yaml --seed 42
# Opens ./output/model_report.html in your browser
```

Skip the report with `--no-report` for fast iteration.

## CLI Reference

```
moneta run <file.moneta.yaml> [options]    Run Monte Carlo simulation
moneta validate <file.moneta.yaml>         Validate model file without running

Options:
  --simulations N     Override simulation count
  --seed N            Random seed for reproducibility
  --output DIR        Output directory (default: ./output/)
  --no-report         Skip HTML report generation
  --verbose           Show timing details and debug info
  --format json|csv   Machine-readable output
```

## Value Syntax

Moneta uses human-readable value syntax in YAML files:

| Type | Examples |
|------|----------|
| Annual rate | `7% annually`, `0.5% monthly` |
| Duration | `10 years`, `6 months` |
| Currency | `$850,000`, `850000` |
| Probability window | `20% within 3 years`, `60% within 5-6 years` |
| Multiplier range | `[2x, 5x]` |
| Cash flow amount | `$5,000 monthly`, `-$50,000 annually`, `-$100,000` |

## How It Works

Moneta runs a **vectorized Monte Carlo simulation** using NumPy. All simulation runs are processed simultaneously as array operations — 10,000 simulations over 30 years complete in ~200ms.

Each monthly time step executes a fixed processor pipeline:

1. **Events** — Fire probabilistic events (equity liquidation, etc.)
2. **Transfers** — Move value between assets on event triggers
3. **Cash Flows** — Apply scheduled income, withdrawals, expenses
4. **Growth** — Apply stochastic growth (Geometric Brownian Motion)
5. **Inflation** — Step mean-reverting inflation process

The result is a `(N_runs x T_steps x N_assets)` matrix that the query engine analyzes to produce probabilities, percentiles, and distributions.

## Financial Correctness

People make real decisions with Moneta. The engine is verified by a comprehensive **financial correctness test suite** (523 tests) covering:

- **Mathematical identities**: GBM exact formula verification, O-U process properties, hazard rate calibration to 12 decimal places
- **Analytical benchmarks**: Zero-volatility deterministic growth verified to the cent (`$100K * exp(0.07 * 10) = $201,375.27`)
- **Statistical properties**: Mean/variance convergence over 500K simulations with rigorous confidence intervals
- **Conservation laws**: Transfer value conservation, cash flow balance identity, shortfall arithmetic
- **Numerical stability**: Floating-point precision with non-round values, cumulative inflation chain stability over 360 months

## Examples

The [`examples/`](examples/) directory contains ready-to-run models:

- **[retirement_basic.moneta.yaml](examples/retirement_basic.moneta.yaml)** — Simple portfolio growth with presets
- **[retirement_with_equity.moneta.yaml](examples/retirement_with_equity.moneta.yaml)** — Portfolio + startup equity with liquidity events
- **[retirement_with_spending.moneta.yaml](examples/retirement_with_spending.moneta.yaml)** — Full lifecycle: contributions, retirement spending, college tuition
- **[scenario_comparison.moneta.yaml](examples/scenario_comparison.moneta.yaml)** — Sweep mode comparing return assumptions

## Development

```bash
git clone https://github.com/gtg922r/moneta.git
cd moneta
uv sync --extra dev
uv run pytest                    # Run all 523 tests
uv run pytest --cov              # With coverage report
uv run moneta run examples/retirement_basic.moneta.yaml --seed 42
```

## License

MIT
