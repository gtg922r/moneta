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
| Cash flow amount | `$5,000 monthly`, `-$50,000 annually`, `-$100,000` | Periodic or one-time flow |

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

## Cash Flows

Cash flows model money moving in and out of assets over time — income, withdrawals, contributions, and one-time expenses. Define them in a top-level `cash_flows` section.

### Recurring flows — periodic income or spending

Recurring flows have an `amount` with a frequency (`monthly` or `annually`) and optional `start`/`end` bounds.

```yaml
cash_flows:
  salary_contribution:
    amount: $3,000 monthly        # positive = deposit into asset
    asset: portfolio
    start: 0 months               # optional, defaults to beginning
    end: 20 years                  # optional, defaults to time_horizon

  retirement_spending:
    amount: -$5,000 monthly       # negative = withdrawal from asset
    asset: portfolio
    start: 20 years
    end: 40 years
    adjust_for: inflation         # amount grows with simulated inflation

  college_tuition:
    amount: -$50,000 annually     # annual amount spread across 12 months
    asset: portfolio
    start: 18 years
    end: 22 years
```

### One-time flows — lump-sum events

One-time flows have an `amount` without a frequency and use `at` instead of `start`/`end`.

```yaml
cash_flows:
  house_down_payment:
    amount: -$100,000             # no frequency = one-time
    asset: portfolio
    at: 5 years                   # applied once at this time
```

### Cash flow fields

| Field | Required | Description |
|-------|----------|-------------|
| `amount` | yes | `$N monthly`, `$N annually`, or `$N` (one-time). Prefix `-` for withdrawals. |
| `asset` | yes | Name of the asset this flow applies to (must exist in `assets`). |
| `start` | no | When the recurring flow begins (default: month 0). |
| `end` | no | When the recurring flow stops (default: `time_horizon`). |
| `at` | no | For one-time flows only — the single point in time to apply. |
| `adjust_for` | no | Set to `inflation` to scale the amount by cumulative simulated inflation. |
| `allow_negative` | no | Default `false`. When false, balances are clamped to zero and unmet withdrawals are tracked as **shortfall**. Set to `true` to allow balances to go negative. |

### Shortfall tracking

When `allow_negative` is false (the default), any withdrawal that would push an asset's balance below zero is clamped. The unmet portion accumulates in a virtual field called `shortfall`, which can be used in queries:

```yaml
queries:
  - type: probability
    expression: shortfall > 0
    at: 40 years
    label: "Probability of running out of money"

  - type: expected
    of: shortfall
    at: 40 years
    label: "Expected cumulative shortfall"
```

### Pipeline position

Cash flows are applied in step 3 of the monthly pipeline (after events and transfers, before growth). This means:
- Equity liquidation proceeds are in the account before withdrawals occur
- The post-cash-flow balance is what gets growth applied

### Amount syntax

| Format | Example | Meaning |
|--------|---------|---------|
| Positive monthly | `$3,000 monthly` | Deposit $3K every month |
| Negative monthly | `-$5,000 monthly` | Withdraw $5K every month |
| Positive annual | `$50,000 annually` | Deposit ~$4,167/month (spread over 12 months) |
| Negative annual | `-$50,000 annually` | Withdraw ~$4,167/month (spread over 12 months) |
| One-time positive | `$100,000` | Deposit $100K once (use with `at`) |
| One-time negative | `-$100,000` | Withdraw $100K once (use with `at`) |

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
moneta run <file.moneta.yaml> [options]
moneta validate <file.moneta.yaml>

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
3. **Cash flows** — apply income, withdrawals, expenses; clamp balances and track shortfall
4. **Growth** — apply GBM growth to investments
5. **Inflation** — step inflation process

Liquidation proceeds receive growth starting the *next* month.

## Validation Rules

- `transfer_to` must reference an existing asset name
- No self-transfers (asset transferring to itself)
- Query `at` time must not exceed `time_horizon`
- Query expressions can only reference defined asset names or `shortfall`
- Probabilities must be 0-100%
- Cash flow `asset` must reference an existing asset name
- Cash flow `at`, `start`, and `end` must not exceed `time_horizon`
- Cash flow `start` must be before `end`
- A cash flow cannot have both `at` and `start`/`end`
- One-time flows (no frequency) must specify `at`
- Recurring flows (with frequency) must use `start`/`end`, not `at`
