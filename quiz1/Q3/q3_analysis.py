"""Analysis of firestation dispatch data for Quiz 1, Question 3.

This module performs data quality assessment and exploratory analysis on the
call logs stored in ``data/data.csv``.  The script reads the dataset and prints
a summary without writing any new files so that the original data remains
untouched.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = ROOT / "data" / "data.csv"

DATETIME_FORMAT = "%m/%d/%y %H:%M"


def load_raw_rows(path: Path = RAW_DATA_PATH) -> List[Dict[str, str]]:
    """Load the CSV file and strip whitespace from every field."""
    rows: List[Dict[str, str]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            cleaned = {key: (value or "").strip() for key, value in raw_row.items()}
            rows.append(cleaned)
    return rows


@dataclass
class CallRecord:
    """Normalized representation of a single dispatch record."""

    case_id: int
    xref_id: str
    incident_number: str
    dispatch_units: Tuple[str, ...]
    dispatch_created: datetime
    alarm_time: Optional[datetime]
    call_complete: Optional[datetime]
    shift: str
    first_unit: str

    @property
    def unit_count(self) -> int:
        return len(self.dispatch_units)

    @property
    def resolution_hours(self) -> Optional[float]:
        if self.call_complete is None:
            return None
        delta = self.call_complete - self.dispatch_created
        return delta.total_seconds() / 3600.0

    @property
    def activity_time(self) -> Optional[datetime]:
        return self.alarm_time or self.dispatch_created

    @property
    def activity_week(self) -> Optional[int]:
        if self.activity_time is None:
            return None
        return self.activity_time.isocalendar().week

    @property
    def activity_hour(self) -> Optional[int]:
        if self.activity_time is None:
            return None
        return self.activity_time.hour


def parse_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(value, DATETIME_FORMAT)
    except ValueError:
        return None


def normalise_unit_name(value: str) -> str:
    return value.replace(" ", "").upper()


def split_dispatch_units(value: str) -> Tuple[str, ...]:
    if not value:
        return tuple()
    parts = [normalise_unit_name(part) for part in value.split(",") if part.strip()]
    return tuple(parts)


def build_call_records(rows: Iterable[Dict[str, str]]) -> List[CallRecord]:
    records: List[CallRecord] = []
    for index, row in enumerate(rows, start=1):
        dispatch_created = parse_datetime(row.get("DISPATCH CREATED DATE", ""))
        if dispatch_created is None:
            # Skip rows with no dispatch timestamp because downstream metrics rely on it.
            continue
        alarm_time = parse_datetime(row.get("ALARM DATE TIME", ""))
        call_complete = parse_datetime(row.get("CALL COMPLETE", ""))
        shift = row.get("SHIFT", "").upper() or "UNKNOWN"
        first_unit = normalise_unit_name(row.get("1ST UNIT ON SCENE", "")) or "UNKNOWN"
        dispatch_units = split_dispatch_units(row.get("DISPATCH UNIT", ""))
        record = CallRecord(
            case_id=index,
            xref_id=row.get("XREF ID", ""),
            incident_number=row.get("INCIDENT NUMBER", ""),
            dispatch_units=dispatch_units,
            dispatch_created=dispatch_created,
            alarm_time=alarm_time,
            call_complete=call_complete,
            shift=shift,
            first_unit=first_unit,
        )
        records.append(record)
    return records


def data_quality_report(rows: Iterable[Dict[str, str]]) -> Dict[str, object]:
    rows_list = list(rows)
    total_rows = len(rows_list)

    missing_counts: Dict[str, int] = {}
    for row in rows_list:
        for key, value in row.items():
            if not value:
                missing_counts[key] = missing_counts.get(key, 0) + 1

    duplicate_incidents = Counter(row.get("INCIDENT NUMBER", "") for row in rows_list)
    duplicate_incident_numbers = {key: count for key, count in duplicate_incidents.items() if count > 1}

    summary = {
        "total_rows": total_rows,
        "missing_counts": missing_counts,
        "missing_percentages": {
            key: (count / total_rows) * 100.0 for key, count in missing_counts.items()
        },
        "duplicate_incident_numbers": duplicate_incident_numbers,
    }
    return summary
def average_resolution_hours(records: Iterable[CallRecord]) -> Optional[float]:
    hours = [record.resolution_hours for record in records if record.resolution_hours is not None]
    if not hours:
        return None
    return mean(hours)


def average_units_per_call(records: Iterable[CallRecord]) -> Optional[float]:
    unit_counts = [record.unit_count for record in records if record.unit_count > 0]
    if not unit_counts:
        return None
    return mean(unit_counts)


def shift_activity(records: Iterable[CallRecord]) -> Counter:
    counter: Counter = Counter()
    for record in records:
        counter[record.shift] += 1
    return counter


def week_hour_matrix(records: Iterable[CallRecord]) -> Tuple[List[int], List[int], Dict[int, Dict[int, int]]]:
    grid: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    weeks: set[int] = set()
    hours: set[int] = set()
    for record in records:
        week = record.activity_week
        hour = record.activity_hour
        if week is None or hour is None:
            continue
        grid[week][hour] += 1
        weeks.add(week)
        hours.add(hour)
    week_list = sorted(weeks)
    hour_list = sorted(hours)
    return week_list, hour_list, grid
def print_report() -> None:
    raw_rows = load_raw_rows()
    quality = data_quality_report(raw_rows)
    records = build_call_records(raw_rows)

    print("DATA QUALITY OVERVIEW")
    print(f"Total rows: {quality['total_rows']}")
    print("Missing values per column:")
    for key, count in sorted(quality["missing_counts"].items()):
        pct = quality["missing_percentages"][key]
        print(f"  - {key}: {count} ({pct:.2f}% of rows)")
    print("Duplicate incident numbers:")
    if quality["duplicate_incident_numbers"]:
        for key, count in quality["duplicate_incident_numbers"].items():
            print(f"  - {key}: {count} entries")
    else:
        print("  None detected")

    resolution = average_resolution_hours(records)
    if resolution is not None:
        print(f"\nAverage resolution time (dispatch to completion): {resolution:.2f} hours")

    avg_units = average_units_per_call(records)
    if avg_units is not None:
        print(f"Average number of dispatched units per call: {avg_units:.2f}")

    print("\nCall volume by shift:")
    for shift_name, count in shift_activity(records).most_common():
        print(f"  - Shift {shift_name}: {count} calls")
    weeks, hours, grid = week_hour_matrix(records)
    busiest_calls = -1
    busiest_week: Optional[int] = None
    busiest_hour: Optional[int] = None
    total_tabulated = 0
    for week in weeks:
        for hour in hours:
            value = grid.get(week, {}).get(hour, 0)
            total_tabulated += value
            if value > busiest_calls:
                busiest_calls = value
                busiest_week = week
                busiest_hour = hour

    if busiest_week is not None and busiest_hour is not None:
        print(
            "\nBusiest ISO week/hour: week"
            f" {busiest_week} at {busiest_hour:02d}:00 with {busiest_calls} calls"
        )
        print(f"Total calls included in grid: {total_tabulated}")
    else:
        print("\nInsufficient timestamp data to summarise weekly/hourly activity.")


if __name__ == "__main__":
    print_report()
