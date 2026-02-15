"""
Cache management commands.

Manage downloaded synoptic data cache.
"""

from pathlib import Path

import click

from windmojo.data.cache import CacheManager


@click.group()
def cache():
    """Manage synoptic data cache."""
    pass


@cache.command()
@click.option(
    "--cache-dir",
    default="data/.cache/synoptic",
    help="Cache directory",
    type=click.Path(),
)
def status(cache_dir):
    """Show cache status and statistics."""
    manager = CacheManager(cache_dir)

    cached_dates = manager.list_cached()
    cache_size = manager.get_cache_size()

    # Convert bytes to MB
    cache_size_mb = cache_size / (1024 * 1024)

    click.echo(f"Cache directory: {manager.cache_dir}")
    click.echo(f"Total files: {len(cached_dates)}")
    click.echo(f"Total size: {cache_size_mb:.2f} MB")

    if cached_dates:
        click.echo(f"\nDate range:")
        click.echo(f"  First: {cached_dates[0]}")
        click.echo(f"  Last: {cached_dates[-1]}")


@cache.command()
@click.option(
    "--cache-dir",
    default="data/.cache/synoptic",
    help="Cache directory",
    type=click.Path(),
)
def list(cache_dir):
    """List all cached dates."""
    manager = CacheManager(cache_dir)

    cached_dates = manager.list_cached()

    if not cached_dates:
        click.echo("No cached files found.")
        return

    click.echo(f"Cached dates ({len(cached_dates)} total):")
    for date in cached_dates:
        click.echo(f"  {date}")


@cache.command()
@click.option(
    "--cache-dir",
    default="data/.cache/synoptic",
    help="Cache directory",
    type=click.Path(),
)
@click.option(
    "--older-than",
    type=int,
    help="Only clear files older than N days",
)
@click.option(
    "--yes",
    is_flag=True,
    help="Skip confirmation prompt",
)
def clear(cache_dir, older_than, yes):
    """Clear cache files."""
    manager = CacheManager(cache_dir)

    cached_dates = manager.list_cached()

    if not cached_dates:
        click.echo("No cached files to clear.")
        return

    # Confirm with user
    if older_than:
        msg = f"Clear {len(cached_dates)} files older than {older_than} days?"
    else:
        msg = f"Clear all {len(cached_dates)} cached files?"

    if not yes:
        if not click.confirm(msg):
            click.echo("Cancelled.")
            return

    # Clear cache
    manager.clear(older_than_days=older_than)

    click.echo("Cache cleared successfully.")
