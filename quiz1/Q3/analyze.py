"""Fire call dataset analysis for Quiz 1, Question 3."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import sqrt
from pathlib import Path
from random import Random
from statistics import StatisticsError, mean, median, stdev
from typing import Dict, List, Optional, Sequence, Tuple

DATA_PATH = Path(__file__).resolve().parent / "data" / "data.csv"
DATETIME_FORMAT = "%m/%d/%y %H:%M"


@dataclass
class IncidentRecord:
    """Parsed representation of a single incident."""

    xref_id: str
    incident_number: str
    shift: Optional[str]
    dispatch_units: List[str]
    first_unit: Optional[str]
    dispatch_dt: datetime
    alarm_dt: Optional[datetime]
    call_complete_raw: Optional[datetime]

    def cleaned_call_complete(self) -> Optional[datetime]:
        """Project the completion time onto the dispatch date.

        The raw data only retains the time-of-day for the completion timestamp.
        Every completion occurs on either 9/4/25 or 9/5/25, regardless of the
        actual dispatch date. The cleaning step reuses the dispatch date and
        preserves the provided time-of-day. If the provided time appears earlier
        than the dispatch timestamp, the event is assumed to conclude within the
        following 24 hours.
        """

        if self.call_complete_raw is None:
            return None
        projected = self.dispatch_dt.replace(
            hour=self.call_complete_raw.hour, minute=self.call_complete_raw.minute
        )
        if projected < self.dispatch_dt:
            projected += timedelta(days=1)
        return projected

    def call_duration_minutes(self) -> Optional[float]:
        complete = self.cleaned_call_complete()
        if complete is None:
            return None
        return (complete - self.dispatch_dt).total_seconds() / 60

    def alarm_baseline(self) -> Optional[datetime]:
        """Return an alarm timestamp aligned to the dispatch date.

        The raw alarm column sometimes uses inconsistent month/day values. When
        the delta is implausibly large (> 12 hours) the function assumes the
        time-of-day is correct but the day needs to be aligned with the dispatch
        date. The aligned timestamp is never allowed to be more than 12 hours
        after dispatch, which prevents artifacts from being interpreted as
        multi-day delays.
        """

        if self.alarm_dt is None:
            return None
        delta = self.alarm_dt - self.dispatch_dt
        if abs(delta.total_seconds()) <= 12 * 60 * 60:
            return self.alarm_dt
        aligned = self.dispatch_dt.replace(
            hour=self.alarm_dt.hour, minute=self.alarm_dt.minute
        )
        # if the aligned value still precedes the dispatch, treat it as a
        # pre-dispatch event (alarm triggered just before dispatch)
        if aligned > self.dispatch_dt + timedelta(hours=12):
            aligned -= timedelta(days=1)
        return aligned

    def alarm_to_close_minutes(self) -> Optional[float]:
        alarm = self.alarm_baseline()
        complete = self.cleaned_call_complete()
        if alarm is None or complete is None:
            return None
        return (complete - alarm).total_seconds() / 60

    def alarm_gap_minutes(self) -> Optional[float]:
        alarm = self.alarm_baseline()
        if alarm is None:
            return None
        gap = (self.dispatch_dt - alarm).total_seconds() / 60
        if gap < 0:
            # Treat negative values as zero response delay.
            gap = 0.0
        if gap > 12 * 60:
            # Extreme outliers are treated as missing information.
            return None
        return gap

    def dispatch_hour(self) -> int:
        return self.dispatch_dt.hour

    def dispatch_weekday(self) -> int:
        return self.dispatch_dt.weekday()  # Monday == 0

    def unit_count(self) -> int:
        return len([u for u in self.dispatch_units if u])


def parse_datetime(value: str) -> Optional[datetime]:
    value = value.strip()
    if not value:
        return None
    return datetime.strptime(value, DATETIME_FORMAT)


def load_records() -> List[IncidentRecord]:
    records: List[IncidentRecord] = []
    with DATA_PATH.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            units = [token.strip() for token in row["DISPATCH UNIT"].split(",")]
            records.append(
                IncidentRecord(
                    xref_id=row["XREF ID"].strip(),
                    incident_number=row["INCIDENT NUMBER"].strip(),
                    shift=row["SHIFT"].strip() or None,
                    dispatch_units=units,
                    first_unit=row["1ST UNIT ON SCENE"].strip() or None,
                    dispatch_dt=parse_datetime(row["DISPATCH CREATED DATE"]),
                    alarm_dt=parse_datetime(row["ALARM DATE TIME"]),
                    call_complete_raw=parse_datetime(row["CALL COMPLETE"]),
                )
            )
    return records


def summarize_missing(records: Sequence[IncidentRecord]) -> Dict[str, int]:
    counts = Counter()
    for record in records:
        if record.shift is None:
            counts["SHIFT"] += 1
        if record.alarm_dt is None:
            counts["ALARM DATE TIME"] += 1
        if record.call_complete_raw is None:
            counts["CALL COMPLETE"] += 1
        if record.unit_count() == 0:
            counts["DISPATCH UNIT"] += 1
        if record.first_unit is None:
            counts["1ST UNIT ON SCENE"] += 1
    return counts


def duplicate_incidents(records: Sequence[IncidentRecord]) -> List[Tuple[str, int]]:
    counts = Counter(record.incident_number for record in records)
    return [(number, count) for number, count in counts.items() if count > 1]


# --- Exploratory statistics -------------------------------------------------

def dispatch_range(records: Sequence[IncidentRecord]) -> Tuple[datetime, datetime]:
    dispatch_dates = [record.dispatch_dt for record in records]
    return min(dispatch_dates), max(dispatch_dates)


def completion_range(records: Sequence[IncidentRecord]) -> Tuple[datetime, datetime]:
    cleaned = [record.cleaned_call_complete() for record in records]
    cleaned = [dt for dt in cleaned if dt is not None]
    return min(cleaned), max(cleaned)


def compute_call_duration_stats(records: Sequence[IncidentRecord]) -> Dict[str, float]:
    durations = [record.call_duration_minutes() for record in records]
    durations = [value for value in durations if value is not None]
    return {
        "count": len(durations),
        "mean": mean(durations),
        "median": median(durations),
        "min": min(durations),
        "max": max(durations),
    }


def compute_alarm_close_stats(records: Sequence[IncidentRecord]) -> Dict[str, float]:
    deltas = [record.alarm_to_close_minutes() for record in records]
    deltas = [value for value in deltas if value is not None]
    return {
        "count": len(deltas),
        "mean": mean(deltas),
        "median": median(deltas),
        "min": min(deltas),
        "max": max(deltas),
    }


def compute_alarm_gap_stats(records: Sequence[IncidentRecord]) -> Dict[str, float]:
    gaps = [record.alarm_gap_minutes() for record in records]
    gaps = [value for value in gaps if value is not None]
    return {
        "count": len(gaps),
        "mean": mean(gaps),
        "median": median(gaps),
        "min": min(gaps),
        "max": max(gaps),
    }


def shift_counts(records: Sequence[IncidentRecord]) -> Counter:
    counts = Counter(record.shift or "Unknown" for record in records)
    return counts


def hourly_weekday_pivot(records: Sequence[IncidentRecord]) -> Dict[str, Dict[int, int]]:
    table: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for record in records:
        weekday = weekdays[record.dispatch_weekday()]
        hour = record.dispatch_hour()
        table[weekday][hour] += 1
    return table


# --- Clustering -------------------------------------------------------------

FeatureVector = List[float]


def build_feature_vectors(records: Sequence[IncidentRecord]) -> Tuple[List[FeatureVector], List[IncidentRecord]]:
    filtered_records: List[IncidentRecord] = []
    features: List[FeatureVector] = []
    shift_map = {"A": 0.0, "B": 1.0, "C": 2.0}
    for record in records:
        duration = record.call_duration_minutes()
        gap = record.alarm_gap_minutes()
        alarm_close = record.alarm_to_close_minutes()
        shift_value = shift_map.get(record.shift)
        if None in (duration, gap, alarm_close, shift_value):
            continue
        features.append(
            [
                duration,
                gap,
                alarm_close,
                float(record.dispatch_hour()),
                float(record.unit_count()),
                shift_value,
            ]
        )
        filtered_records.append(record)
    return features, filtered_records


def standardize_features(vectors: Sequence[FeatureVector]) -> Tuple[List[FeatureVector], List[float], List[float]]:
    if not vectors:
        return [], [], []
    columns = list(zip(*vectors))
    means = [mean(column) for column in columns]
    stds: List[float] = []
    for column, column_mean in zip(columns, means):
        try:
            column_std = stdev(column)
        except StatisticsError:  # pragma: no cover - defensive guard
            column_std = 0.0
        if column_std == 0:
            column_std = 1.0
        stds.append(column_std)
    standardized: List[FeatureVector] = []
    for vector in vectors:
        standardized.append(
            [
                (value - column_mean) / column_std
                for value, column_mean, column_std in zip(vector, means, stds)
            ]
        )
    return standardized, means, stds


def squared_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return sqrt(squared_distance(a, b))


def kmeans(vectors: Sequence[FeatureVector], k: int, *, seed: int = 42, max_iter: int = 100) -> Tuple[List[int], List[FeatureVector]]:
    rng = Random(seed)
    n = len(vectors)
    if k <= 0 or k > n:
        raise ValueError("k must be between 1 and the number of data points")
    indices = list(range(n))
    rng.shuffle(indices)
    centroids = [vectors[idx][:] for idx in indices[:k]]
    assignments = [0] * n
    for _ in range(max_iter):
        changed = False
        # assignment step
        for i, vector in enumerate(vectors):
            distances = [squared_distance(vector, centroid) for centroid in centroids]
            cluster = distances.index(min(distances))
            if assignments[i] != cluster:
                assignments[i] = cluster
                changed = True
        # update step
        new_centroids: List[FeatureVector] = []
        for cluster in range(k):
            members = [vectors[i] for i, label in enumerate(assignments) if label == cluster]
            if not members:
                new_centroids.append(vectors[rng.randrange(n)][:])
                continue
            centroid = [0.0] * len(vectors[0])
            for member in members:
                for j, value in enumerate(member):
                    centroid[j] += value
            centroid = [value / len(members) for value in centroid]
            new_centroids.append(centroid)
        centroids = new_centroids
        if not changed:
            break
    return assignments, centroids


def silhouette_score(vectors: Sequence[FeatureVector], labels: Sequence[int]) -> float:
    clusters: Dict[int, List[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        clusters[label].append(index)
    if len(clusters) <= 1:
        return 0.0

    scores: List[float] = []
    for index, label in enumerate(labels):
        same_cluster = clusters[label]
        if len(same_cluster) == 1:
            scores.append(0.0)
            continue
        point = vectors[index]
        # Mean distance to own cluster
        a = sum(
            euclidean_distance(point, vectors[other])
            for other in same_cluster
            if other != index
        ) / (len(same_cluster) - 1)
        # Mean distance to nearest other cluster
        b = None
        for other_label, indices in clusters.items():
            if other_label == label:
                continue
            distance = sum(
                euclidean_distance(point, vectors[other])
                for other in indices
            ) / len(indices)
            if b is None or distance < b:
                b = distance
        assert b is not None
        scores.append((b - a) / max(a, b))
    return sum(scores) / len(scores)


def inertia(vectors: Sequence[FeatureVector], labels: Sequence[int], centroids: Sequence[FeatureVector]) -> float:
    total = 0.0
    for vector, label in zip(vectors, labels):
        total += squared_distance(vector, centroids[label])
    return total


def dbscan(vectors: Sequence[FeatureVector], eps: float, min_samples: int) -> List[int]:
    n = len(vectors)
    labels = [None] * n
    cluster_id = 0

    def region_query(idx: int) -> List[int]:
        point = vectors[idx]
        neighbors: List[int] = []
        for other_idx, candidate in enumerate(vectors):
            if euclidean_distance(point, candidate) <= eps:
                neighbors.append(other_idx)
        return neighbors

    for idx in range(n):
        if labels[idx] is not None:
            continue
        neighbors = region_query(idx)
        if len(neighbors) < min_samples:
            labels[idx] = -1
            continue
        labels[idx] = cluster_id
        queue = [neighbor for neighbor in neighbors if neighbor != idx]
        while queue:
            neighbor_idx = queue.pop()
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
            if labels[neighbor_idx] is not None:
                continue
            labels[neighbor_idx] = cluster_id
            neighbor_neighbors = region_query(neighbor_idx)
            if len(neighbor_neighbors) >= min_samples:
                for candidate in neighbor_neighbors:
                    if labels[candidate] is None:
                        queue.append(candidate)
        cluster_id += 1
    return labels


def silhouette_for_dbscan(vectors: Sequence[FeatureVector], labels: Sequence[int]) -> float:
    filtered: List[FeatureVector] = []
    filtered_labels: List[int] = []
    mapping: Dict[int, int] = {}
    next_label = 0
    for vector, label in zip(vectors, labels):
        if label == -1:
            continue
        if label not in mapping:
            mapping[label] = next_label
            next_label += 1
        filtered.append(vector)
        filtered_labels.append(mapping[label])
    if not filtered or len(set(filtered_labels)) <= 1:
        return 0.0
    return silhouette_score(filtered, filtered_labels)


def main() -> None:
    records = load_records()
    print(f"Total incidents: {len(records)}")
    start, end = dispatch_range(records)
    print(f"Dispatch records span {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}")
    c_start, c_end = completion_range(records)
    print(f"Call completion (cleaned) spans {c_start:%Y-%m-%d %H:%M} to {c_end:%Y-%m-%d %H:%M}")

    missing = summarize_missing(records)
    print("Missing values:")
    for key, value in missing.items():
        print(f"  {key}: {value}")

    duplicates = duplicate_incidents(records)
    print(f"Duplicate incident numbers: {duplicates}")

    duration_stats = compute_call_duration_stats(records)
    print("Call duration statistics (minutes):", duration_stats)

    alarm_close_stats = compute_alarm_close_stats(records)
    print("Alarm-to-close statistics (minutes):", alarm_close_stats)

    alarm_gap_stats = compute_alarm_gap_stats(records)
    print("Alarm-to-dispatch gap (minutes):", alarm_gap_stats)

    shifts = shift_counts(records)
    print("Shift counts:")
    for shift_label, count in shifts.items():
        print(f"  {shift_label}: {count}")

    pivot = hourly_weekday_pivot(records)
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    hours = list(range(24))
    print("Hourly incidents by weekday:")
    for weekday in weekdays:
        row_total = sum(pivot[weekday][hour] for hour in hours)
        print(
            weekday,
            {hour: pivot[weekday][hour] for hour in hours},
            "total",
            row_total,
        )
    column_totals = {hour: sum(pivot[weekday][hour] for weekday in weekdays) for hour in hours}
    print("Column totals:", column_totals)
    print("Grand total:", sum(column_totals.values()))

    # --- clustering ---
    raw_features, filtered_records = build_feature_vectors(records)
    standardized, feature_means, feature_stds = standardize_features(raw_features)
    print(f"Records used for clustering: {len(standardized)}")
    print("Feature means:", feature_means)
    print("Feature std devs:", feature_stds)

    if standardized:
        k = 3
        kmeans_labels, centroids = kmeans(standardized, k)
        score = silhouette_score(standardized, kmeans_labels)
        inertia_value = inertia(standardized, kmeans_labels, centroids)
        print(f"K-means (k={k}) silhouette: {score:.3f}, inertia: {inertia_value:.3f}")
        cluster_sizes = Counter(kmeans_labels)
        print("K-means cluster sizes:", dict(cluster_sizes))

        eps = 1.25
        min_samples = 25
        dbscan_labels = dbscan(standardized, eps=eps, min_samples=min_samples)
        dbscan_clusters = Counter(label for label in dbscan_labels if label != -1)
        dbscan_noise = sum(1 for label in dbscan_labels if label == -1)
        dbscan_silhouette = silhouette_for_dbscan(standardized, dbscan_labels)
        print(
            f"DBSCAN (eps={eps}, min_samples={min_samples}) clusters: {dict(dbscan_clusters)}, noise: {dbscan_noise}, silhouette: {dbscan_silhouette:.3f}"
        )


if __name__ == "__main__":
    main()
