# Contributing to Windmojo

Thanks for your interest in contributing to Windmojo!

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/windmojo.git
   cd windmojo
   ```
3. Install dependencies:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv sync --all-extras
   ```
4. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

**Python**: 3.10 or higher (managed by UV)

```bash
# Run tests (no network access and no spot data needed)
uv run pytest tests/ -v

# Run with coverage
uv run pytest --cov=windmojo

# Command-line interface
uv run python cli.py --help
uv run python -m windmojo.data_fetcher --help
```

To try changes end to end you need a spot with measurements (see "Step by Step: A New
Spot" in the [README](README.md)). Spot data, configs and models stay in the gitignored
`data/` and `config/` directories; don't commit them.

## Code Style

- **Formatter**: Black (line length 88, version pinned in `pyproject.toml`)
- **Import sorting**: isort (black profile)
- **Type hints** and **docstrings** for public functions and classes

```bash
# Format code (CI checks windmojo/, tests/ and cli.py)
uv run black windmojo/ tests/ cli.py
uv run isort windmojo/ tests/ cli.py

# Type check
uv run mypy windmojo/
```

The pre-commit hooks (`uv run pre-commit install`) run the same tools on every commit.

## Changes to Features or Training

- Training and live forecasts must compute identical features: no feature may look back
  more than 7 days or use statistics over the whole input
  (`tests/test_pipeline.py::TestLiveParity`). After changing the feature list, update
  `tests/feature_names.txt`
- Claims about accuracy need a test on independent data: walk-forward (train only on
  data before the test period), ideally on more than one spot. A single test split is
  not enough; see [FEATURE_RELEVANCE.md](research/03_FEATURE_ENGINEERING/FEATURE_RELEVANCE.md)
  for how features were compared
- Feature selection is settled (all features); new features are welcome if they add
  information the weather model does not already have

## Pull Request Process

1. Ensure your code follows the style guidelines above
2. Add tests for new functionality
3. Update documentation if needed (README, SIMPLIFIED_ARCHITECTURE.md, DEPLOYMENT.md, CHANGELOG.md)
4. Keep PRs focused - one feature or fix per PR
5. Write a clear PR description explaining the change

## Areas of Interest

We welcome contributions in these areas:

- **Probabilistic output**: chance of ≥ 4 m/s per hour, quantiles, gust forecast
- **Lead time**: training on forecasts made one or more days ahead
- **Spot layout**: `cli.py download` and `cli.py features` for `data/spots/`, so new
  spots need no legacy config
- **More spots**: results for coastal, thermal and mountain spots
- **Documentation**: tutorials, examples, translations
- **Testing**: expanding test coverage

## Reporting Issues

When reporting bugs, please include:

- Python version and OS
- The command you ran and the config (without API keys)
- Expected vs actual behavior
- Relevant error messages or logs

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
