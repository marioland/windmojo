"""Windmojo CLI - Command-line interface for wind forecasting."""

from windmojo.cli.main import cli

__all__ = ["cli", "main"]


def main():
    """Main CLI entry point."""
    cli()
