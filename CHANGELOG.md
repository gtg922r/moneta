# Changelog

All notable changes to Moneta are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/). Versions follow [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-03-19

Initial public release. Probabilistic financial modeling engine with Monte Carlo simulation.

### Added

- **Monte Carlo simulation engine** — vectorized NumPy-based engine running 10,000+ simulations in ~200ms
- **Investment assets** with Geometric Brownian Motion (GBM) stochastic growth
- **Illiquid equity** with probabilistic liquidity events, hazard-rate calibration, and configurable valuation ranges
- **Inflation modeling** — mean-reverting Ornstein-Uhlenbeck process
- **Cash flows** — recurring (monthly/annual) income, withdrawals, and one-time expenses with inflation adjustment
- **Shortfall tracking** — balance clamping with cumulative unmet withdrawal tracking
- **Asset transfers** — automatic transfer on liquidity events (e.g., equity proceeds → portfolio)
- **Query engine** — probability, percentile, expected value, and distribution queries with expression parser
- **Inflation-adjusted queries** — `adjust_for: inflation` converts results to real dollars
- **Named scenario sweeps** — compare conservative/base/optimistic assumptions side-by-side
- **Built-in presets** — `sp500`, `total_market`, `us_inflation`, `us_treasuries`, `tech_startup_equity`
- **Interactive HTML reports** — Plotly-based fan charts, histograms, probability timelines, and scenario comparisons
- **Rich terminal output** — formatted tables with currency formatting and percentile bands
- **CLI** — `moneta run` and `moneta validate` commands via Click
- **Human-readable YAML syntax** — `$850,000`, `7% annually`, `20% within 3 years`, `[2x, 5x]`
- **`/moneta` agent skill** for Claude Code — conversational financial model building
- **Installable via `npx skills add gtg922r/moneta`** — one-command skill install with CLI auto-install
- **523 tests** with 85%+ coverage including financial correctness verification suite
- **CI pipeline** — ruff linting, mypy strict type checking, pytest on Python 3.12 + 3.13
- **Pre-commit hooks** — ruff check + format on staged files
- **PEP 561** `py.typed` marker for downstream type checking

### Example models included

- `retirement_basic.moneta.yaml` — simple portfolio growth with presets
- `retirement_with_equity.moneta.yaml` — portfolio + startup equity with liquidity events
- `retirement_with_spending.moneta.yaml` — full lifecycle with contributions, retirement spending, college tuition
- `scenario_comparison.moneta.yaml` — sweep mode comparing return assumptions

[0.1.0]: https://github.com/gtg922r/moneta/releases/tag/v0.1.0
