from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import time
from typing import Any
import urllib.request

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "data_config.json"
DEFAULT_CACHE = ROOT / ".cache_data"
EXTRACTION_SCHEMA_VERSION = 1

try:
    import orjson
    def _loads(value: str | bytes) -> Any:
        return orjson.loads(value)
except ImportError:
    _loads = json.loads


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _timestamp(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp must have an explicit time zone")
    return int(parsed.timestamp())


def _utc(value: int | float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")


def _config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _config_sha256() -> str:
    return hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()


def _preprocessing_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _hours(config: dict[str, Any], days: int | None = None) -> list[tuple[str, str]]:
    first = date.fromisoformat(config["source_start_date"])
    last = date.fromisoformat(config["source_end_date"])
    if days is not None:
        last = min(last, first + timedelta(days=days - 1))
    result = []
    current = first
    while current <= last:
        for hour in range(24):
            stem = f"{current.isoformat()}-{hour}"
            url = config["source_url_template"].format(date=current.isoformat(), hour=hour)
            result.append((stem, url))
        current += timedelta(days=1)
    return result


class _HashingReader:
    def __init__(self, source: Any) -> None:
        self.source = source
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self.source.read(size)
        self.digest.update(chunk)
        self.bytes_read += len(chunk)
        return chunk


def _extraction_spec(config: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": EXTRACTION_SCHEMA_VERSION,
            "event_type": config["event_type"], "event_action": config["event_action"]}


def _cached_hour_manifest(target: Path, url: str, config: dict[str, Any]) -> dict[str, Any] | None:
    manifest_path = target.with_suffix(target.suffix + ".manifest.json")
    if not target.is_file() or not manifest_path.is_file():
        return None
    try:
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(saved, dict) or saved.get("source_url") != url:
            return None
        legacy_spec = {"schema_version": 1, "event_type": "WatchEvent", "event_action": "started"}
        if saved.get("extraction_spec", legacy_spec) != _extraction_spec(config):
            return None
        if _sha256_file(target) != saved.get("extracted_sha256"):
            return None
        if "extraction_spec" not in saved:
            saved["extraction_spec"] = legacy_spec
            _json_write(manifest_path, saved)
        return saved
    except (OSError, ValueError, TypeError):
        return None


def _extract_hour(stem: str, url: str, cache: Path, config: dict[str, Any]) -> dict[str, Any]:
    target = cache / "hours" / f"{stem}.stars.jsonl.gz"
    manifest_path = target.with_suffix(target.suffix + ".manifest.json")
    saved = _cached_hour_manifest(target, url, config)
    if saved is not None:
        return saved
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(config["download_retries"]):
        try:
            counts: Counter[str] = Counter()
            minimum, maximum = None, None
            request = urllib.request.Request(url, headers={"User-Agent": "recsys-course-homework/1.0"})
            with urllib.request.urlopen(request, timeout=config["request_timeout_seconds"]) as response:
                stream = _HashingReader(response)
                expected_size = response.headers.get("Content-Length")
                source_etag = response.headers.get("ETag")
                with temporary.open("wb") as compact_file:
                    with gzip.GzipFile(filename="", mode="wb", fileobj=compact_file, mtime=0) as compact_gzip:
                        with gzip.GzipFile(fileobj=stream, mode="rb") as source_gzip:
                            for line in source_gzip:
                                counts["raw_events"] += 1
                                try:
                                    event = _loads(line)
                                except (ValueError, TypeError):
                                    counts["invalid_json"] += 1
                                    continue
                                if event.get("type") != config["event_type"]:
                                    continue
                                counts["watch_events"] += 1
                                if event.get("payload", {}).get("action") != config["event_action"]:
                                    counts["wrong_watch_action"] += 1
                                    continue
                                repository = event.get("repository") or {}
                                actor = event.get("actor")
                                attributes = event.get("actor_attributes") or {}
                                login = actor if isinstance(actor, str) else (actor or {}).get("login")
                                if not login or not repository.get("id") or not event.get("created_at"):
                                    counts["missing_identity_or_time"] += 1
                                    continue
                                if event.get("public") is False or repository.get("private") is True:
                                    counts["nonpublic"] += 1
                                    continue
                                if attributes.get("type") == "Bot" or str(login).casefold().endswith("[bot]"):
                                    counts["explicit_bot"] += 1
                                    continue
                                try:
                                    timestamp = _timestamp(event["created_at"])
                                    item_id = int(repository["id"])
                                except (ValueError, TypeError, OverflowError):
                                    counts["invalid_identity_or_time"] += 1
                                    continue
                                owner = repository.get("owner") or ""
                                if isinstance(owner, dict):
                                    owner = owner.get("login") or owner.get("name") or ""
                                user_hash = hashlib.sha256(str(login).strip().casefold().encode()).hexdigest()
                                language = str(repository.get("language") or "").strip()
                                owner = str(owner).strip().casefold()
                                row = [user_hash, item_id, timestamp, language, owner]
                                compact_gzip.write((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
                                counts["retained_star_events"] += 1
                                counts["with_language"] += bool(language)
                                counts["with_owner"] += bool(owner)
                                minimum = timestamp if minimum is None else min(minimum, timestamp)
                                maximum = timestamp if maximum is None else max(maximum, timestamp)
                while stream.read(1024 * 1024):
                    pass
                if expected_size is not None and stream.bytes_read != int(expected_size):
                    raise IOError(f"Compressed size mismatch: {url}")
                result = {
                    "extraction_spec": _extraction_spec(config),
                    "source_url": url,
                    "source_sha256": stream.digest.hexdigest(),
                    "source_compressed_bytes": stream.bytes_read,
                    "source_etag": source_etag,
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "extracted_sha256": _sha256_file(temporary),
                    "extracted_bytes": temporary.stat().st_size,
                    "counts": dict(counts),
                    "min_event_time_utc": _utc(minimum) if minimum is not None else None,
                    "max_event_time_utc": _utc(maximum) if maximum is not None else None,
                }
            temporary.replace(target)
            _json_write(manifest_path, result)
            return result
        except Exception as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt + 1 < config["download_retries"]:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Failed extracting {url}: {last_error}") from last_error


def download_data(cache_dir: str | Path | None = None, days: int | None = None,
                  workers: int | None = None) -> dict[str, Any]:
    cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    config = _config()
    config_sha256 = _config_sha256()
    entries = _hours(config, days)
    results: dict[str, Any] = {}
    worker_count = workers or config["download_workers"]
    print(f"Extracting {len(entries)} source hours with {worker_count} workers; full raw archives are not stored.", flush=True)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(_extract_hour, stem, url, cache, config): stem for stem, url in entries}
        for future in as_completed(futures):
            stem = futures[future]
            result = future.result()
            results[stem] = result
            count = sum(value["counts"].get("retained_star_events", 0) for value in results.values())
            byte_count = sum(value["source_compressed_bytes"] for value in results.values())
            print(f"[{len(results)}/{len(entries)}] {stem}: {result['counts'].get('retained_star_events', 0):,} stars; total {count:,}; {byte_count / 1e6:.1f} MB streamed", flush=True)
    totals: Counter[str] = Counter()
    for result in results.values():
        totals.update(result["counts"])
    summary = {
        "dataset_name": config["dataset_name"],
        "config_sha256": config_sha256,
        "hours": len(results),
        "source_compressed_bytes": sum(value["source_compressed_bytes"] for value in results.values()),
        "extracted_bytes": sum(value["extracted_bytes"] for value in results.values()),
        "counts": dict(totals),
        "min_event_time_utc": min(value["min_event_time_utc"] for value in results.values() if value["min_event_time_utc"]),
        "max_event_time_utc": max(value["max_event_time_utc"] for value in results.values() if value["max_event_time_utc"]),
        "source_fingerprint_sha256": hashlib.sha256("\n".join(f"{stem}:{results[stem]['source_sha256']}" for stem in sorted(results)).encode()).hexdigest(),
    }
    _json_write(cache / "acquisition_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _matrix(rows: list[tuple[int, int, int]], shape: tuple[int, int], timestamps: bool = False) -> sparse.csr_matrix:
    if not rows:
        return sparse.csr_matrix(shape, dtype=np.float64 if timestamps else np.float32)
    values = np.asarray(rows, dtype=np.int64)
    data = values[:, 2].astype(np.float64) if timestamps else np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (values[:, 0], values[:, 1])), shape=shape)


def prepare_data(cache_dir: str | Path | None = None, download: bool = True) -> dict[str, Path]:
    cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    config = _config()
    config_sha256 = _config_sha256()
    preprocessing_sha256 = _preprocessing_sha256()
    if download:
        download_data(cache)
    train_cutoff = _timestamp(config["train_cutoff_utc"])
    validation_cutoff = _timestamp(config["validation_cutoff_utc"])
    pairs: dict[tuple[str, int], int] = {}
    metadata: dict[int, tuple[int, str, str]] = {}
    raw_counts: Counter[str] = Counter()
    hours = _hours(config)
    hour_manifests: dict[str, dict[str, Any]] = {}
    for stem, url in hours:
        path = cache / "hours" / f"{stem}.stars.jsonl.gz"
        manifest = _cached_hour_manifest(path, url, config)
        if manifest is None:
            raise ValueError(f"Source hour cache {stem} is missing, corrupt or has a different extraction spec; run --prepare with downloads enabled")
        hour_manifests[stem] = manifest
        with gzip.open(path, "rb") as source:
            for line in source:
                user, item, timestamp, language, owner = _loads(line)
                raw_counts["retained_star_events"] += 1
                key = (user, item)
                previous = pairs.get(key)
                if previous is None or timestamp < previous:
                    pairs[key] = timestamp
                if timestamp < train_cutoff:
                    old = metadata.get(item)
                    if old is None or (timestamp, language, owner) > old:
                        metadata[item] = (timestamp, language, owner)
    raw_counts["unique_user_item_pairs"] = len(pairs)
    raw_counts["duplicate_star_events"] = raw_counts["retained_star_events"] - len(pairs)
    raw_counts["unique_users"] = len({user for user, item in pairs})
    raw_counts["unique_items"] = len({item for user, item in pairs})
    acquisition_counts: Counter[str] = Counter()
    for manifest in hour_manifests.values():
        acquisition_counts.update(manifest.get("counts", {}))
    for name, value in acquisition_counts.items():
        if name != "retained_star_events":
            raw_counts[name] = value
    raw_counts.setdefault("invalid_json", 0)
    empty_source_hours = [stem for stem, manifest in hour_manifests.items()
                          if not manifest.get("counts", {}).get("raw_events", 0)]
    source_fingerprint = None
    if all(manifest.get("source_sha256") for manifest in hour_manifests.values()):
        source_fingerprint = hashlib.sha256("\n".join(f"{stem}:{hour_manifests[stem]['source_sha256']}" for stem in sorted(hour_manifests)).encode()).hexdigest()
    original_train = [(user, item, timestamp) for (user, item), timestamp in pairs.items() if timestamp < train_cutoff]
    item_counts = Counter(item for user, item, timestamp in original_train)
    eligible_items = [item for item, count in item_counts.items() if count >= config["min_train_item_interactions"]]
    eligible_items.sort(key=lambda item: (-item_counts[item], item))
    items = set(eligible_items[:config["max_items"]])
    train = [row for row in original_train if row[1] in items]
    iterations = 0
    while True:
        iterations += 1
        users_count = Counter(user for user, item, timestamp in train)
        items_count = Counter(item for user, item, timestamp in train)
        keep_users = {user for user, count in users_count.items() if count >= config["min_train_user_interactions"]}
        keep_items = {item for item, count in items_count.items() if count >= config["min_train_item_interactions"]}
        reduced = [row for row in train if row[0] in keep_users and row[1] in keep_items]
        if len(reduced) == len(train):
            break
        train = reduced
    users = sorted({user for user, item, timestamp in train})
    items_sorted = sorted({item for user, item, timestamp in train})
    user_map = {user: index for index, user in enumerate(users)}
    item_map = {item: index for index, item in enumerate(items_sorted)}
    shape = (len(users), len(items_sorted))
    splits: dict[str, list[tuple[int, int, int]]] = {"train": [], "validation": [], "test": []}
    split_report: dict[str, Counter[str]] = {name: Counter() for name in splits}
    for (user, item), timestamp in pairs.items():
        split = "train" if timestamp < train_cutoff else ("validation" if timestamp < validation_cutoff else "test")
        report = split_report[split]
        report["raw_unique_pairs"] += 1
        if user not in user_map:
            report["dropped_unknown_user"] += 1
            continue
        if item not in item_map:
            report["dropped_unknown_item"] += 1
            continue
        splits[split].append((user_map[user], item_map[item], timestamp))
        report["kept_pairs"] += 1
    language_values = sorted({metadata[item][1] for item in items_sorted if metadata[item][1]})
    owner_values = sorted({metadata[item][2] for item in items_sorted if metadata[item][2]})
    language_map = {value: index + 1 for index, value in enumerate(language_values)}
    owner_map = {value: index + 1 for index, value in enumerate(owner_values)}
    language = np.asarray([language_map.get(metadata[item][1], 0) for item in items_sorted], dtype=np.int32)
    owner = np.asarray([owner_map.get(metadata[item][2], 0) for item in items_sorted], dtype=np.int32)
    final_item_counts = np.bincount([item for user, item, timestamp in splits["train"]], minlength=shape[1])
    language_distribution = []
    for category, label in enumerate(["unknown", *language_values]):
        mask = language == category
        if np.any(mask):
            language_distribution.append({"language": label, "n_items": int(np.count_nonzero(mask)),
                                          "train_interactions": int(final_item_counts[mask].sum())})
    language_distribution.sort(key=lambda row: (-row["train_interactions"], row["language"]))
    owner_item_counts = np.bincount(owner, minlength=len(owner_values) + 1)[1:]
    processed = cache / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, rows in splits.items():
        paths[name] = processed / f"{name}.npz"
        sparse.save_npz(paths[name], _matrix(rows, shape))
        paths[f"{name}_timestamps"] = processed / f"{name}_timestamps.npz"
        sparse.save_npz(paths[f"{name}_timestamps"], _matrix(rows, shape, timestamps=True))
    paths["item_features"] = processed / "item_features.npz"
    np.savez_compressed(paths["item_features"], language=language, owner=owner)
    _json_write(processed / "private_mappings.json", {"users_sha256": users, "repository_ids": items_sorted, "language_categories": language_values, "owner_categories": owner_values})
    all_timestamps = list(pairs.values())
    summary = {
        "dataset_name": config["dataset_name"],
        "source": "https://www.gharchive.org/",
        "config_sha256": config_sha256,
        "preprocessing_sha256": preprocessing_sha256,
        "source_file_date_start": config["source_start_date"],
        "source_file_date_end_inclusive": config["source_end_date"],
        "source_selection_note": config.get("source_selection_note", "Fixed source interval selected before model evaluation."),
        "source_hours": len(hours),
        "source_compressed_bytes": sum(manifest.get("source_compressed_bytes", 0) for manifest in hour_manifests.values()),
        "source_fingerprint_sha256": source_fingerprint,
        "extraction_spec": _extraction_spec(config),
        "empty_source_hour_count": len(empty_source_hours),
        "empty_source_hours": empty_source_hours,
        "raw": dict(raw_counts),
        "event_time_start_utc": _utc(min(all_timestamps)),
        "event_time_end_utc": _utc(max(all_timestamps)),
        "train_cutoff_utc": config["train_cutoff_utc"],
        "validation_cutoff_utc": config["validation_cutoff_utc"],
        "split_rule": "train timestamp < train cutoff; validation train cutoff <= timestamp < validation cutoff; test timestamp >= validation cutoff",
        "deduplication": "one positive per user/repository, at earliest observed timestamp across the full source interval",
        "catalog_selection": "top items by training-only frequency; ties broken by historical repository ID; then iterative training-only user/item minimum-degree filtering",
        "min_train_user_interactions": config["min_train_user_interactions"],
        "min_train_item_interactions": config["min_train_item_interactions"],
        "max_items": config["max_items"],
        "filter_iterations": iterations,
        "n_users": shape[0],
        "n_items": shape[1],
        "n_interactions": sum(len(rows) for rows in splits.values()),
        "splits": {name: dict(counts) for name, counts in split_report.items()},
        "density_train": len(splits["train"]) / max(1, shape[0] * shape[1]),
        "language_categories_including_unknown": len(language_values) + 1,
        "owner_categories_including_unknown": len(owner_values) + 1,
        "items_missing_language": int(np.count_nonzero(language == 0)),
        "items_missing_owner": int(np.count_nonzero(owner == 0)),
        "language_distribution": language_distribution,
        "owners_with_multiple_items": int(np.count_nonzero(owner_item_counts >= 2)),
        "items_with_shared_owner": int(owner_item_counts[owner_item_counts >= 2].sum()),
        "metadata_snapshot_rule": "latest historical star-event repository snapshot strictly before training cutoff; no post-cutoff metadata or current popularity features",
        "prior_seen_rule": "global earliest-pair deduplication makes train/validation/test user-item pairs disjoint",
        "identity_limitations": "2014 events lack a numeric actor ID; casefolded historical login is hashed before caching. Renames can fragment a user identity.",
        "archive_time_warning": "Historical filenames use collector-local hours; timestamps include offsets and are converted to UTC before splitting.",
        "license_note": config["license_note"],
    }
    summary["requirements_met"] = {
        "interactions": config["min_final_interactions"] <= summary["n_interactions"] <= config["max_final_interactions"],
        "users": shape[0] >= config["min_users"],
        "items": shape[1] >= config["min_items"],
    }
    paths["summary"] = processed / "summary.json"
    _json_write(paths["summary"], summary)
    print(json.dumps(summary, indent=2), flush=True)
    return paths


def processed_cache_valid(cache_dir: str | Path | None = None) -> bool:
    processed = (Path(cache_dir) if cache_dir else DEFAULT_CACHE) / "processed"
    required = [processed / f"{name}{suffix}.npz" for name in ("train", "validation", "test") for suffix in ("", "_timestamps")]
    required.extend([processed / "item_features.npz", processed / "summary.json"])
    if not all(path.is_file() for path in required):
        return False
    try:
        summary = json.loads((processed / "summary.json").read_text(encoding="utf-8"))
        return (isinstance(summary, dict) and summary.get("config_sha256") == _config_sha256()
                and summary.get("preprocessing_sha256") == _preprocessing_sha256())
    except (OSError, ValueError, TypeError):
        return False


def load_data(cache_dir: str | Path | None = None) -> dict[str, Any]:
    processed = (Path(cache_dir) if cache_dir else DEFAULT_CACHE) / "processed"
    if not processed_cache_valid(cache_dir):
        raise ValueError("Processed cache is missing or stale for the current config/pipeline. Run prepare_data() or python data_pipeline.py --prepare.")
    result: dict[str, Any] = {}
    for name in ("train", "validation", "test"):
        result[name] = sparse.load_npz(processed / f"{name}.npz").tocsr()
        result[f"{name}_timestamps"] = sparse.load_npz(processed / f"{name}_timestamps.npz").tocsr()
    with np.load(processed / "item_features.npz") as values:
        result["item_features"] = {name: values[name] for name in values.files}
    result["summary"] = json.loads((processed / "summary.json").read_text(encoding="utf-8"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Stream source archives and cache compact star rows")
    parser.add_argument("--prepare", action="store_true", help="Download missing hours and prepare sparse matrices")
    parser.add_argument("--no-download", action="store_true", help="Prepare from already extracted cache only")
    parser.add_argument("--days", type=int, help="Limit extraction pilot to first N source days")
    parser.add_argument("--workers", type=int, help="Concurrent download workers")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    if args.days is not None and args.days < 1:
        parser.error("--days must be positive")
    if args.prepare:
        if args.days is not None:
            parser.error("--days is for extraction pilot only; --prepare uses the fixed config interval")
        prepare_data(args.cache_dir, download=not args.no_download)
    elif args.download:
        download_data(args.cache_dir, days=args.days, workers=args.workers)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
