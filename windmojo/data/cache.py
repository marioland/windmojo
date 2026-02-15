"""
Cache management for downloaded synoptic data.

Provides caching layer to avoid re-downloading data and enable
quick resume of interrupted downloads.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class CacheManager:
    """Manage cached synoptic grid data."""

    def __init__(self, cache_dir: str = "data/.cache/synoptic"):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for cache storage
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def is_cached(self, date: str) -> bool:
        """
        Check if data for date is cached.

        Args:
            date: Date string (YYYY-MM-DD)

        Returns:
            True if cached, False otherwise
        """
        filepath = self._get_cache_path(date)
        return filepath.exists()

    def get(self, date: str) -> Optional[pd.DataFrame]:
        """
        Load cached data for date.

        Args:
            date: Date string (YYYY-MM-DD)

        Returns:
            DataFrame if cached, None otherwise
        """
        filepath = self._get_cache_path(date)

        if not filepath.exists():
            return None

        try:
            logger.debug(f"Loading from cache: {date}")
            return pd.read_parquet(filepath)
        except Exception as e:
            logger.warning(f"Failed to load cache for {date}: {e}")
            return None

    def put(self, date: str, data: pd.DataFrame) -> Path:
        """
        Save data to cache.

        Args:
            date: Date string (YYYY-MM-DD)
            data: DataFrame to cache

        Returns:
            Path to cached file
        """
        filepath = self._get_cache_path(date)

        try:
            data.to_parquet(filepath, engine="pyarrow", compression="snappy")
            logger.debug(f"Cached: {date} ({len(data):,} points)")
            return filepath
        except Exception as e:
            logger.error(f"Failed to cache {date}: {e}")
            raise

    def list_cached(self) -> List[str]:
        """
        List all cached dates.

        Returns:
            List of date strings (YYYY-MM-DD)
        """
        cached = []
        for filepath in sorted(self.cache_dir.glob("synoptic_grid_*.parquet")):
            # Extract date from filename: synoptic_grid_YYYYMMDD.parquet
            date_str = filepath.stem.replace("synoptic_grid_", "")
            # Convert YYYYMMDD to YYYY-MM-DD
            try:
                date_obj = datetime.strptime(date_str, "%Y%m%d")
                cached.append(date_obj.strftime("%Y-%m-%d"))
            except ValueError:
                continue

        return cached

    def clear(self, older_than_days: Optional[int] = None):
        """
        Clear cache.

        Args:
            older_than_days: Only clear files older than N days.
                           If None, clear all.
        """
        if older_than_days is None:
            # Clear all
            count = 0
            for filepath in self.cache_dir.glob("synoptic_grid_*.parquet"):
                filepath.unlink()
                count += 1
            logger.info(f"Cleared {count} cached files")
        else:
            # Clear old files
            cutoff = datetime.now() - timedelta(days=older_than_days)
            count = 0

            for filepath in self.cache_dir.glob("synoptic_grid_*.parquet"):
                # Check file modification time
                mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
                if mtime < cutoff:
                    filepath.unlink()
                    count += 1

            logger.info(
                f"Cleared {count} cached files older than {older_than_days} days"
            )

    def get_cache_size(self) -> int:
        """
        Get total cache size in bytes.

        Returns:
            Cache size in bytes
        """
        total = 0
        for filepath in self.cache_dir.glob("synoptic_grid_*.parquet"):
            total += filepath.stat().st_size
        return total

    def _get_cache_path(self, date: str) -> Path:
        """
        Get cache filepath for date.

        Args:
            date: Date string (YYYY-MM-DD)

        Returns:
            Path to cache file
        """
        # Convert YYYY-MM-DD to YYYYMMDD for filename
        date_str = date.replace("-", "")
        return self.cache_dir / f"synoptic_grid_{date_str}.parquet"
