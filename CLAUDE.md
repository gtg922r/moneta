## Moneta

Use `/moneta` to get help building financial models. The skill guides you through creating .moneta.yaml files, running Monte Carlo simulations, and interpreting results.

Run models with `moneta run <file> --seed 42`. Validate with `moneta validate <file>`. Inside the repo dev environment, use `uv run moneta` instead.

## Development Workflow

### Setup

```bash
uv sync --extra dev
```

### Commands

| Task | Command |
|------|---------|
| Lint | `uv run ruff check` |
| Lint + auto-fix | `uv run ruff check --fix` |
| Format | `uv run ruff format` |
| Format check (no changes) | `uv run ruff format --check` |
| Type check | `uv run mypy src/` |
| Run tests | `uv run pytest -q` |
| Run tests with coverage | `uv run pytest --cov=moneta --cov-report=term-missing` |
| Validate a model | `uv run moneta validate <file>` |
| Run a model | `uv run moneta run <file> --seed 42` |

### Quality Bar

CI enforces all of the following on every push and PR:

- **Lint:** `ruff check` must pass with zero warnings (rules: E, F, W, I, UP, B, SIM, RUF)
- **Format:** `ruff format --check` must produce no diffs (line-length 88)
- **Type check:** `mypy --strict` must pass on `src/` with the Pydantic plugin enabled
- **Tests:** `pytest` must pass on Python 3.12 and 3.13, with ≥85% coverage

### Before Committing

Run lint and format before committing. If you have pre-commit hooks installed (`pre-commit install`), ruff runs automatically on staged files. Otherwise:

```bash
uv run ruff check --fix && uv run ruff format
```

Type checking runs in CI only (too slow for pre-commit hooks). Run it manually before pushing:

```bash
uv run mypy src/
```

## Releasing

To release a new version:

1. Ensure CI is green on main
2. Update `CHANGELOG.md` with the new version's changes
3. Bump `version` in `pyproject.toml` (single source of truth)
4. Commit: `git commit -m "release: vX.Y.Z"`
5. Tag: `git tag vX.Y.Z`
6. Push: `git push origin main --tags`

The `release.yml` workflow handles the rest: lint + typecheck + test → build → GitHub Release with artifacts. The tag version must match `pyproject.toml` version or the release will fail.

## gstack

Use the `/browse` skill from gstack for all web browsing. Never use `mcp__claude-in-chrome__*` tools.

Available skills:
- `/moneta` - Financial modeling assistant (craft models, run simulations, explain results)
- `/plan-ceo-review` - CEO-level plan review
- `/plan-eng-review` - Engineering plan review
- `/review` - Code review
- `/ship` - Ship code
- `/browse` - Web browsing (use this for all web browsing)
- `/qa` - QA testing
- `/setup-browser-cookies` - Set up browser cookies
- `/retro` - Retrospective
