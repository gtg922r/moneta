# Moneta Model File Reference

Moneta is a probabilistic financial modeling engine. Models are defined in `.moneta.yaml` files with five top-level sections.

## Complete Example

```yaml
scenario:
  name: "Retirement projection with startup equity"
  time_horizon: 30 years
  simulations: 10000
  seed: 42                          # optional, for reproducibility

assets:
  investment_portfolio:
    type: investment
    initial_balance: $850,000
    growth:
      model: gbm
      expected_return: 7% annually
      volatility: 15% annually

  startup_equity:
    type: illiquid_equity
    current_valuation: $500,000
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
    model: mean_reverting
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
    at: [5 years, 10 years, 15 years, 20 years, 25 years, 30 years]
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

## Value Syntax

| Type | Examples | Meaning |
|------|----------|---------|
| Annual rate | `7% annually` | 0.07 annual rate |
| Monthly rate | `0.5% monthly` | Annualized to 6% |
| Duration | `10 years`, `6 months` | Time period |
| Probability window | `20% within 3 years` | 20% chance in first 3 years |
| Probability range | `60% within 5-6 years` | 60% chance between year 5 and 6 |
| Multiplier range | `[2x, 5x]` | Uniform distribution over 2-5x |
| Currency | `$850,000`, `850000` | Dollar amount |

## Asset Types

### `investment` — Liquid assets with stochastic growth
```yaml
asset_name:
  type: investment
  initial_balance: $500,000
  growth:
    model: gbm                    # Geometric Brownian Motion
    expected_return: 7% annually  # average annual return (drift)
    volatility: 15% annually      # annual standard deviation
```
Or use a preset: `growth: { preset: sp500 }`

### `illiquid_equity` — Equity with probabilistic liquidity events
```yaml
asset_name:
  type: illiquid_equity
  current_valuation: $500,000
  shares: 50000                   # optional
  liquidity_events:
    - probability: 20% within 3 years
      valuation_range: [2x, 5x]  # multiplier on current_valuation
    - probability: 60% within 5-6 years
      valuation_range: [3x, 10x]
  on_liquidation:
    transfer_to: other_asset_name # proceeds flow to this asset
```

## Growth Models

- **`gbm`** (Geometric Brownian Motion) — For market returns. Log-normal, meanders with drift.
- **`mean_reverting`** — For inflation/interest rates. Reverts toward a long-term mean.

## Built-in Presets

| Preset | Model | Parameters |
|--------|-------|------------|
| `sp500` | GBM | 7% return, 15% volatility |
| `total_market` | GBM | 7% return, 12% volatility |
| `us_inflation` | Mean-reverting | 3% long-term, 1% volatility |
| `us_treasuries` | Mean-reverting | 4% long-term, 3% volatility |
| `tech_startup_equity` | GBM | 15% return, 40% volatility |

Usage: `growth: { preset: sp500 }` or `inflation: { preset: us_inflation }`

## Global Settings

```yaml
global:
  inflation:
    model: mean_reverting
    long_term_rate: 3% annually
    volatility: 1% annually
    mean_reversion_speed: 0.5     # optional, default 0.5
```
Or use a preset: `inflation: { preset: us_inflation }`

## Query Types

### Probability — "What are the odds X happens?"
```yaml
- type: probability
  expression: portfolio + equity > 2000000
  at: 10 years
  adjust_for: inflation           # optional — converts to real dollars
  label: "Descriptive label"
```
Operators: `+`, `-`, `*`, `/`, `>`, `<`, `>=`, `<=`
References: asset names, `initial_<asset>` for starting values

### Percentiles — "What's the range of outcomes?"
```yaml
- type: percentiles
  values: [10, 25, 50, 75, 90]
  of: portfolio                   # asset name or expression
  at: [5 years, 10 years, 20 years]  # one or multiple time points
  adjust_for: inflation           # optional
  label: "Portfolio distribution"
```

### Expected — "What's the average outcome?"
```yaml
- type: expected
  of: portfolio + equity
  at: 10 years
  adjust_for: inflation           # optional
  label: "Expected net worth"
```

### Distribution — "Show me the full histogram"
```yaml
- type: distribution
  of: portfolio
  at: 10 years
  bins: 50                        # optional, default 50
  label: "Outcome distribution"
```

## Sweep Mode — Named Scenario Comparison

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
Each scenario deep-merges its overrides onto the base model. Run produces a comparison table.

## CLI

```bash
uv run moneta run <file.moneta.yaml> [options]
uv run moneta validate <file.moneta.yaml>

Options:
  --simulations N     Override simulation count
  --seed N            Random seed for reproducibility
  --output DIR        Output directory (default: ./output/)
  --no-report         Skip HTML report generation
  --format json|csv   Machine-readable output
```

## Simulation Pipeline

Each monthly time step runs processors in fixed order:
1. **Events** — fire discrete events (liquidity, etc.)
2. **Transfers** — move value between assets
3. **Growth** — apply GBM growth to investments
4. **Inflation** — step inflation process

Liquidation proceeds receive growth starting the *next* month.

## Validation Rules

- `transfer_to` must reference an existing asset name
- No self-transfers (asset transferring to itself)
- Query `at` time must not exceed `time_horizon`
- Query expressions can only reference defined asset names
- Probabilities must be 0-100%
