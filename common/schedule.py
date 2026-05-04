"""Utilities for handling initialization schedules shared by multiple scripts."""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class Schedule:
    start: dt.datetime
    end: dt.datetime
    delta_hours: int


DEFAULT_FMT = "%Y%m%d%H"


def _parse_datetime(value: str | None, fallback: dt.datetime | None) -> dt.datetime:
    if value:
        return dt.datetime.strptime(value, DEFAULT_FMT)
    if fallback is None:
        raise ValueError('A start/end datetime must be provided either via CLI or environment variables.')
    return fallback


def build_schedule(
    start: str | None = None,
    end: str | None = None,
    delta_hours: int | None = None,
    *,
    default_start: dt.datetime | None = None,
    default_end: dt.datetime | None = None,
    default_delta_hours: int = 6,
    env_prefix: str = 'AI_FORECAST',
) -> Schedule:
    """Construct a schedule using CLI overrides, environment variables, or defaults."""
    env_start = os.getenv(f'{env_prefix}_START')
    env_end = os.getenv(f'{env_prefix}_END')
    env_delta = os.getenv(f'{env_prefix}_DELTA_HOURS')

    schedule_start = start or env_start
    schedule_end = end or env_end
    schedule_delta = delta_hours if delta_hours is not None else (int(env_delta) if env_delta else default_delta_hours)

    return Schedule(
        start=_parse_datetime(schedule_start, default_start),
        end=_parse_datetime(schedule_end, default_end),
        delta_hours=schedule_delta,
    )


def generate_init_times(schedule: Schedule) -> List[dt.datetime]:
    """Return every initialization time from ``start`` to ``end`` inclusive."""
    times: List[dt.datetime] = []
    current = schedule.start
    delta = dt.timedelta(hours=schedule.delta_hours)
    while current <= schedule.end:
        times.append(current)
        current += delta
    return times


def format_datetime(value: dt.datetime) -> str:
    return value.strftime(DEFAULT_FMT)


def parse_datetime_list(values: Iterable[str]) -> List[dt.datetime]:
    return [dt.datetime.strptime(value, DEFAULT_FMT) for value in values]
