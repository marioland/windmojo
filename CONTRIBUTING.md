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
# Run tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest --cov=windmojo

# Run a module
uv run python -m windmojo.data_fetcher --help
```

## Code Style

- **Formatter**: Black (line length 88)
- **Import sorting**: isort (black profile)
- **Type hints**: Required for all public functions
- **Docstrings**: Required for all public functions and classes

```bash
# Format code
uv run black windmojo/ tests/
uv run isort windmojo/ tests/

# Type check
uv run mypy windmojo/
```

## Pull Request Process

1. Ensure your code follows the style guidelines above
2. Add tests for new functionality
3. Update documentation if needed
4. Keep PRs focused - one feature or fix per PR
5. Write a clear PR description explaining the change

## Areas of Interest

We welcome contributions in these areas:

- **Feature engineering**: New atmospheric feature ideas
- **Alternative ML models**: LightGBM, neural networks, ensemble methods
- **Multi-location support**: Improvements to the area/spot architecture
- **Documentation**: Tutorials, examples, translations
- **Testing**: Expanding test coverage
- **Performance**: Optimizing data download and feature engineering

## Reporting Issues

When reporting bugs, please include:

- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Relevant error messages or logs

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
