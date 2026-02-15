"""
Windmojo CLI - Main entry point.

Unified command-line interface for wind forecasting tasks.
"""

from pathlib import Path

import click

from windmojo.__version__ import __version__


@click.group()
@click.version_option(version=__version__, prog_name="windmojo")
def cli():
    """
    Windmojo - ML-powered wind forecasting.

    Download synoptic data, extract features, and train models.
    """
    pass


from windmojo.cli.commands.cache import cache
from windmojo.cli.commands.diagram import diagram

# Import commands
from windmojo.cli.commands.download import download

# Register commands
cli.add_command(download)
cli.add_command(cache)
cli.add_command(diagram)


if __name__ == "__main__":
    cli()
