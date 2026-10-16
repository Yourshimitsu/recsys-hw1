from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import fnmatch
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlparse


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
AUTHOR = "Томак Андрей Владимирович"
REQUIRED_MODELS = {"ItemKNN", "iALS", "EASE", "FM metadata", "Recency iALS"}
CONTROLS = {"Popularity", "FM IDs only"}
ALL_MODELS = REQUIRED_MODELS | CONTROLS
METRICS = ("ndcg_at_10", "recall_at_10", "precision_at_10", "hit_rate_at_10", "catalog_coverage_at_10")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REQUIRED_FILES = (
    "README.md", "experiments.ipynb", "requirements.txt", ".gitignore",
    "data_pipeline.py", "evaluation.py", "run_experiments.py", "verify_submission.py",
    "config/data_config.json", "models/item_knn.py", "models/implicit_als.py",
    "models/ease.py", "models/fm.py", "models/recency_als.py",
    "results/run_manifest.json", "results/test_metrics.csv", "results/validation_trials.csv",
    "results/selected_parameters.json", "results/dataset_summary.json",
    "results/evaluation_details.json", "results/source_archives.json",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(relative):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON number {value} in {relative}")
    return json.loads((ROOT / relative).read_text(encoding="utf-8"), parse_constant=reject_constant)


def read_csv(relative):
    with (ROOT / relative).open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def number(value, name, minimum=None, maximum=None):
    value = float(value)
    require(math.isfinite(value), f"{name} must be finite")
    require(minimum is None or value >= minimum, f"{name} must be >= {minimum}")
    require(maximum is None or value <= maximum, f"{name} must be <= {maximum}")
    return value


def integer(value, name, minimum=1):
    require(not isinstance(value, bool), f"{name} must be an integer, not boolean")
    value = number(value, name, minimum=minimum)
    require(value.is_integer(), f"{name} must be an integer")
    return int(value)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


class Report:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, label, callback):
        try:
            detail = callback()
        except Exception as error:
            self.failed += 1
            print(f"FAIL | {label}: {type(error).__name__}: {error}")
        else:
            self.passed += 1
            print(f"PASS | {label}: {detail}")


def check_required_files():
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
    require(not missing, "Missing: " + ", ".join(missing))
    empty = [name for name in REQUIRED_FILES if (ROOT / name).stat().st_size == 0]
    require(not empty, "Empty: " + ", ".join(empty))
    require(any((ROOT / "tests").glob("test_*.py")), "No unit tests found")
    return f"{len(REQUIRED_FILES)} required files and unit tests present"


def check_readme():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", text)
    require(AUTHOR in normalized, f"README must name {AUTHOR}")
    placeholders = re.findall(r"\b(?:TODO|FILLME|TBD|PLACEHOLDER)\b|Фамилия\s+Имя\s+Отчество", text, re.I)
    require(not placeholders, "README contains placeholders: " + ", ".join(placeholders))
    for match in re.finditer(r"\bФИО\b", text, re.I):
        nearby = re.sub(r"\s+", " ", text[max(0, match.start() - 100) : match.end() + 120])
        require(AUTHOR in nearby, "README contains a ФИО placeholder without the actual author nearby")
    return "correct author; no unfinished name/TODO placeholders"


def _ignore_rules():
    result = []
    for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            negative = line.startswith("!")
            result.append((line[1:] if negative else line, negative))
    return result


def _ignored(relative, rules):
    relative = relative.replace("\\", "/").strip("/")
    parts = relative.split("/")
    ignored = False
    for raw_pattern, negative in rules:
        directory = raw_pattern.endswith("/")
        anchored = raw_pattern.startswith("/")
        pattern = raw_pattern.strip("/")
        targets = ["/".join(parts[:i]) for i in range(1, len(parts) + (0 if directory else 1))]
        if "/" in pattern or anchored:
            matched = any(fnmatch.fnmatchcase(target, pattern) for target in targets)
        else:
            names = parts[:-1] if directory else parts
            matched = any(fnmatch.fnmatchcase(part, pattern) for part in names)
        if matched:
            ignored = not negative
    return ignored


def public_candidates():
    git = shutil.which("git")
    if git:
        probe = subprocess.run([git, "-C", str(ROOT), "rev-parse", "--is-inside-work-tree"],
                               capture_output=True, text=True)
        if probe.returncode == 0 and probe.stdout.strip() == "true":
            result = subprocess.run(
                [git, "-C", str(ROOT), "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "."],
                capture_output=True, check=True,
            )
            files = sorted({value.decode("utf-8") for value in result.stdout.split(b"\0") if value})
            return files, "Git tracked + untracked, excluding ignored untracked files"
    rules = _ignore_rules()
    files = []
    for directory, dirs, names in os.walk(ROOT, followlinks=False):
        dirs[:] = [name for name in dirs if name != ".git"]
        for name in dirs:
            path = Path(directory) / name
            relative = path.relative_to(ROOT).as_posix()
            if path.is_symlink() and not _ignored(relative + "/sentinel", rules):
                files.append(relative)
        for name in names:
            relative = (Path(directory) / name).relative_to(ROOT).as_posix()
            if not _ignored(relative, rules):
                files.append(relative)
    return sorted(files), "root .gitignore scan (folder has no Git repository)"


def check_public_files(list_files=False):
    rules = _ignore_rules()
    sentinels = (".cache_data/processed/summary.json", ".cache_models/readme.txt",
                 "data/raw.jsonl", ".venv/pyvenv.cfg", "models/__pycache__/fm.pyc",
                 "weights/model.pt", "weights/model.npy", "weights/model.npz")
    require(all(_ignored(path, rules) for path in sentinels),
            ".gitignore must exclude data/model caches, environments, bytecode, and array/weight files")
    files, method = public_candidates()
    forbidden_dirs = {".cache_data", ".cache_models", "data", "datasets", "raw_data", "weights", "checkpoints", ".venv", "venv", "__pycache__",
                      ".ipynb_checkpoints", ".pytest_cache"}
    forbidden_suffixes = (".npz", ".npy", ".pt", ".pth", ".pkl", ".pickle", ".parquet", ".onnx",
                          ".safetensors", ".ckpt", ".bin", ".model", ".weights", ".h5", ".hdf5", ".joblib", ".jsonl", ".jsonl.gz",
                          ".csv.gz", ".sqlite", ".sqlite3", ".db", ".tar", ".tar.gz", ".zip", ".gz",
                          ".mp4", ".mov", ".webm", ".mp3", ".m4a", ".wav", ".pyc", ".nbc", ".nbi")
    unsafe = []
    for relative in files:
        path = ROOT / relative
        parts = set(Path(relative).parts)
        if (parts & forbidden_dirs or relative.lower().endswith(forbidden_suffixes)
                or "private_mappings" in path.name.casefold()
                or path.name == "experiments.unexecuted.ipynb"
                or (path.suffix.lower() == ".csv" and not relative.startswith("results/"))):
            unsafe.append(relative)
        require(path.exists(), f"Candidate file is missing (tracked deletion?): {relative}")
        require(path.resolve().is_relative_to(ROOT.resolve()), f"Candidate points outside submission: {relative}")
        require(path.stat().st_size < 90 * 1024 * 1024, f"Candidate is too large for a reviewable GitHub submission: {relative}")
    require(not unsafe, "Data/weights/cache or private files in public candidates: " + ", ".join(unsafe))
    missing = sorted(set(REQUIRED_FILES) - set(files))
    require(not missing, "Required files excluded from public candidates: " + ", ".join(missing))
    if list_files:
        for relative in files:
            print(f"  READY FILE | {relative}")
    return f"{len(files)} public candidates; no data/weights/caches; {method}"


def check_notebook():
    import nbformat
    notebook = nbformat.read(ROOT / "experiments.ipynb", as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    require(code_cells, "Notebook has no code cells")
    plots = tables = 0
    for index, cell in enumerate(notebook.cells, start=1):
        if cell.cell_type != "code":
            continue
        count = cell.get("execution_count")
        require(isinstance(count, int) and not isinstance(count, bool) and count > 0,
                f"Code cell {index} has not been executed")
        for output in cell.get("outputs", []):
            require(output.get("output_type") != "error", f"Code cell {index} contains an error output")
            data = output.get("data", {})
            plots += int(any(mime.startswith("image/") or mime == "application/vnd.plotly.v1+json" for mime in data))
            html = data.get("text/html", "")
            if isinstance(html, list):
                html = "".join(html)
            tables += int("<table" in str(html).lower() or "application/vnd.dataresource+json" in data)
    require(plots > 0, "Executed notebook has no embedded plot outputs")
    require(tables > 0, "Executed notebook has no rendered result tables")
    return f"valid notebook; {len(code_cells)} executed code cells, {plots} plots, {tables} tables; no errors"


def check_manifest():
    manifest = load_json("results/run_manifest.json")
    require(manifest.get("completed") is True, "run_manifest.completed must be true")
    require(SHA256.fullmatch(str(manifest.get("code_sha256", ""))), "Missing/invalid code fingerprint")
    require(manifest.get("test_used_after_selection") is True, "Missing final test-after-selection declaration")
    require(manifest.get("primary_metric") == "ndcg_at_10", "Unexpected primary metric")
    sys.path.insert(0, str(ROOT))
    previous_cache = os.environ.get("NUMBA_CACHE_DIR")
    try:
        with tempfile.TemporaryDirectory(prefix="recsys-fingerprint-") as cache:
            os.environ["NUMBA_CACHE_DIR"] = cache
            runner = importlib.import_module("run_experiments")
            current = runner._fingerprint()
    finally:
        sys.path.pop(0)
        if previous_cache is None:
            os.environ.pop("NUMBA_CACHE_DIR", None)
        else:
            os.environ["NUMBA_CACHE_DIR"] = previous_cache
    require(current == manifest["code_sha256"], "Source changed after experiments; rerun experiments and execute notebook")
    return f"completed run and matching source fingerprint {current[:12]}…"


def _validate_metric_rows(rows, expected_models, label, unique):
    require(rows, f"{label} is empty")
    names = [row.get("model") for row in rows]
    require(set(names) == expected_models, f"{label} models differ: got {sorted(set(names))}, expected {sorted(expected_models)}")
    if unique:
        require(len(names) == len(set(names)), f"Duplicate models in {label}")
    cohorts = set()
    for index, row in enumerate(rows, start=1):
        for metric in METRICS:
            number(row[metric], f"{label} row {index} {metric}", 0, 1)
        cohorts.add(integer(row["n_users"], f"{label} row {index} n_users"))
        for key in ("fit_seconds", "evaluation_seconds"):
            if key in row:
                number(row[key], f"{label} row {index} {key}", 0)
        require(isinstance(json.loads(row["parameters"]), dict), f"Invalid parameters in {label} row {index}")
    require(len(cohorts) == 1, f"{label} uses inconsistent evaluation-user counts")
    return cohorts.pop()


def check_test_metrics():
    rows = read_csv("results/test_metrics.csv")
    cohort = _validate_metric_rows(rows, ALL_MODELS, "test metrics", unique=True)
    selected = load_json("results/selected_parameters.json")["selected"]
    for row in rows:
        name = row["model"]
        expected_role = "baseline" if name == "Popularity" else "ablation" if name == "FM IDs only" else "required"
        require(row.get("role") == expected_role, f"Incorrect role for {name}")
        require(json.loads(row["parameters"]) == selected[name], f"Test parameters do not match frozen selection for {name}")
    return f"five required models + two controls; bounded finite metrics; common cohort of {cohort} users"


def check_dataset():
    summary = load_json("results/dataset_summary.json")
    config = load_json("config/data_config.json")
    flags = summary.get("requirements_met", {})
    require(all(flags.get(key) is True for key in ("interactions", "users", "items")),
            "dataset requirements_met must explicitly confirm interactions, users, and items")
    interactions = integer(summary["n_interactions"], "n_interactions")
    users = integer(summary["n_users"], "n_users")
    items = integer(summary["n_items"], "n_items")
    number(interactions, "n_interactions", max(100_000, config["min_final_interactions"]), min(2_000_000, config["max_final_interactions"]))
    number(users, "n_users", max(1000, config["min_users"]))
    number(items, "n_items", max(1000, config["min_items"]))
    split_counts = [integer(summary["splits"][name]["kept_pairs"], f"{name} kept_pairs")
                    for name in ("train", "validation", "test")]
    require(sum(split_counts) == interactions, "Split pair counts do not sum to n_interactions")
    number(summary["density_train"], "density_train", 0, 1)
    require(summary["train_cutoff_utc"] == config["train_cutoff_utc"] and
            summary["validation_cutoff_utc"] == config["validation_cutoff_utc"], "Summary cutoffs differ from current config")
    require(summary["source_file_date_start"] == config["source_start_date"] and
            summary["source_file_date_end_inclusive"] == config["source_end_date"], "Summary source interval differs from current config")
    require(summary.get("config_sha256") == hashlib.sha256((ROOT / "config/data_config.json").read_bytes()).hexdigest(),
            "Dataset summary config fingerprint is stale")
    require(summary.get("preprocessing_sha256") == hashlib.sha256((ROOT / "data_pipeline.py").read_bytes()).hexdigest(),
            "Dataset summary preprocessing fingerprint is stale")
    return f"{interactions:,} interactions, {users:,} users, {items:,} items; all three nonempty splits meet thresholds"


def check_selection():
    selection = load_json("results/selected_parameters.json")
    require(selection.get("test_not_used_for_selection") is True, "Selection must explicitly exclude test labels")
    require(selection.get("primary_metric") == "ndcg_at_10", "Selection primary metric must be NDCG@10")
    selected = selection["selected"]
    require(set(selected) == ALL_MODELS, "Frozen parameters must include five models and two controls")
    require(selected["FM IDs only"] == selected["FM metadata"], "FM ID ablation must retain metadata-FM hyperparameters")
    for name, value in selected["iALS"].items():
        require(selected["Recency iALS"].get(name) == value, f"Recency ablation changed baseline ALS {name}")
    trials = read_csv("results/validation_trials.csv")
    _validate_metric_rows(trials, REQUIRED_MODELS, "validation trials", unique=False)
    spaces = selection["search_spaces"]
    require(set(spaces) == REQUIRED_MODELS, "Search spaces must cover exactly the five tuned models")
    for name in REQUIRED_MODELS:
        model_trials = [row for row in trials if row["model"] == name]
        recorded = [canonical(json.loads(row["parameters"])) for row in model_trials]
        declared = [canonical(params) for params in spaces[name]]
        require(recorded == declared, f"Validation trials do not cover the declared search space in order for {name}")
        require([integer(row["trial"], f"{name} trial") for row in model_trials] == list(range(1, len(model_trials) + 1)),
                f"Validation trial numbers are incomplete or out of order for {name}")
        winner = max(model_trials, key=lambda row: float(row["ndcg_at_10"]))
        require(json.loads(winner["parameters"]) == selected[name], f"Frozen {name} parameters are not the first validation maximum")
    manifest = load_json("results/run_manifest.json")
    require(selection.get("seed") == manifest.get("seed"), "Selection and run seeds differ")
    return f"{len(trials)} validation trials; frozen first-maximum choices; paired ablation settings retained"


def check_provenance():
    archives = load_json("results/source_archives.json")
    require(isinstance(archives, list) and archives, "No source archive provenance")
    urls, stems, compressed = set(), {}, 0
    empty_stems = set()
    raw_events_total = retained_events_total = 0
    for index, entry in enumerate(archives, start=1):
        url = entry["source_url"]
        parsed = urlparse(url)
        require(parsed.scheme == "https" and parsed.hostname == "data.gharchive.org" and parsed.path.endswith(".json.gz"),
                f"Unexpected archive source URL at row {index}: {url}")
        require(url not in urls, f"Duplicate archive source URL: {url}")
        urls.add(url)
        for key in ("source_sha256", "extracted_sha256"):
            require(SHA256.fullmatch(str(entry.get(key, ""))), f"Invalid {key} for {url}")
        compressed += integer(entry["source_compressed_bytes"], f"archive {index} compressed bytes")
        require(isinstance(entry.get("counts"), dict), f"Missing extraction counts for {url}")
        raw_events = integer(entry["counts"].get("raw_events", 0), f"archive {index} raw_events", minimum=0)
        retained = integer(entry["counts"].get("retained_star_events", 0), f"archive {index} retained_star_events", minimum=0)
        require(retained <= raw_events, f"Retained-event count exceeds raw-event count for {url}")
        for key, value in entry["counts"].items():
            integer(value, f"archive {index} {key}", minimum=0)
        raw_events_total += raw_events
        retained_events_total += retained
        stem = Path(parsed.path).name.removesuffix(".json.gz")
        stems[stem] = entry["source_sha256"]
        if raw_events == 0:
            empty_stems.add(stem)
    summary = load_json("results/dataset_summary.json")
    manifest = load_json("results/run_manifest.json")
    config = load_json("config/data_config.json")
    expected_hours = (date.fromisoformat(config["source_end_date"]) - date.fromisoformat(config["source_start_date"])).days * 24 + 24
    require(len(archives) == expected_hours == summary["source_hours"] == manifest["source_archive_count"],
            "Source-archive counts do not match the declared complete interval")
    first = date.fromisoformat(config["source_start_date"])
    expected_stems = {f"{(first + timedelta(days=offset)).isoformat()}-{hour}"
                      for offset in range(expected_hours // 24) for hour in range(24)}
    require(set(stems) == expected_stems, "Source archive filenames do not cover every configured day/hour exactly")
    require(summary.get("empty_source_hour_count", 0) == len(empty_stems) and
            set(summary.get("empty_source_hours", [])) == empty_stems,
            "Zero-event source hours differ from the documented dataset summary")
    require(raw_events_total == summary["raw"]["raw_events"] and
            retained_events_total == summary["raw"]["retained_star_events"],
            "Archive event-count totals disagree with the dataset summary")
    require(compressed == summary["source_compressed_bytes"], "Source compressed-byte totals disagree")
    fingerprint = hashlib.sha256("\n".join(f"{stem}:{stems[stem]}" for stem in sorted(stems)).encode()).hexdigest()
    require(fingerprint == summary["source_fingerprint_sha256"], "Dataset source fingerprint disagrees with published archive hashes")
    return f"{len(archives)} unique source URLs ({len(empty_stems)} documented empty hours); SHA-256 aggregate verified without cached data"


def check_evaluation_details():
    details = load_json("results/evaluation_details.json")
    rows = read_csv("results/test_metrics.csv")
    metrics = {row["model"]: float(row["ndcg_at_10"]) for row in rows}
    require(integer(details["n_test_users"], "n_test_users") == integer(rows[0]["n_users"], "test n_users"),
            "Evaluation details and test-user counts disagree")
    trials = read_csv("results/validation_trials.csv")
    require(integer(details["n_validation_users"], "n_validation_users") == integer(trials[0]["n_users"], "validation n_users"),
            "Evaluation details and validation-user counts disagree")
    for key, a, b in (("metadata_ablation", "FM metadata", "FM IDs only"), ("recency_ablation", "Recency iALS", "iALS")):
        comparison = details[key]
        delta = number(comparison["mean_delta"], f"{key} mean_delta", -1, 1)
        low = number(comparison["ci_low"], f"{key} ci_low", -1, 1)
        high = number(comparison["ci_high"], f"{key} ci_high", -1, 1)
        require(low <= high, f"{key} has inverted confidence bounds")
        require(math.isclose(delta, metrics[a] - metrics[b], abs_tol=1e-8), f"{key} delta disagrees with test metrics")
    selected = load_json("results/selected_parameters.json")["selected"]
    for name in ("FM metadata", "FM IDs only"):
        losses = details["fm_training_bce"][name]
        require(len(losses) == selected[name]["epochs"], f"Missing FM training losses for {name}")
        for value in losses:
            number(value, f"{name} training BCE", 0)
    require(details.get("bootstrap_note"), "Missing bootstrap scope/limitations note")
    return "paired bootstrap deltas match test metrics; finite intervals, cohort counts, and complete FM loss curves"


def run_tests():
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    with tempfile.TemporaryDirectory(prefix="recsys-readiness-") as cache:
        env["NUMBA_CACHE_DIR"] = cache
        result = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"],
                                cwd=ROOT, env=env, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=600)
    output = (result.stdout + result.stderr).strip()
    if output:
        print(output)
    require(result.returncode == 0, f"Unit tests failed with exit code {result.returncode}")
    require(re.search(r"Ran [1-9][0-9]* tests?", output), "Test discovery did not run any tests")
    return "unit tests passed from standalone homework folder"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tests", action="store_true", help="Also run unittest discovery offline")
    parser.add_argument("--list-files", action="store_true", help="Print every public submission candidate")
    args = parser.parse_args()
    print(f"Offline submission readiness: {ROOT}")
    report = Report()
    report.check("Required files", check_required_files)
    report.check("README author and completion", check_readme)
    report.check("Public candidate files and ignore rules", lambda: check_public_files(args.list_files))
    report.check("Executed notebook", check_notebook)
    report.check("Completed run and code fingerprint", check_manifest)
    report.check("Final test metrics", check_test_metrics)
    report.check("Dataset requirements", check_dataset)
    report.check("Validation selection and ablations", check_selection)
    report.check("Source provenance", check_provenance)
    report.check("Evaluation details", check_evaluation_details)
    if args.run_tests:
        report.check("Unit tests", run_tests)
    else:
        print("INFO | Unit tests not rerun; add --run-tests to include them")
    status = "PASS" if report.failed == 0 else "FAIL"
    print(f"{status}: {report.passed} checks passed; {report.failed} checks failed.")
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
