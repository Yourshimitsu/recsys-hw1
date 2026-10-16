from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse


_PLOT_SETTINGS = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "axes.spines.top": False,
    "axes.spines.right": False,
}
_SPLIT_COLORS = {"train": "#247ba0", "validation": "#e9a23b", "test": "#bc5068"}


def _save(fig, directory: Path, filename: str) -> Path:
    path = directory / filename
    try:
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)
    return path


def _empty(axis, message: str) -> None:
    axis.text(0.5, 0.5, message, transform=axis.transAxes, ha="center", va="center")
    axis.set_xticks([])
    axis.set_yticks([])


def create_eda_figures(data: dict, out_dir: str | Path = "figures") -> list[Path]:
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    train = sparse.csr_matrix(data["train"], copy=True)
    train.sum_duplicates()
    train.eliminate_zeros()
    if any(data[name].shape != train.shape for name in ("validation", "test")):
        raise ValueError("train, validation and test must have matching dimensions")
    language = np.asarray(data["item_features"]["language"])
    if language.shape != (train.shape[1],):
        raise ValueError("language metadata must contain one category id per item")

    with plt.rc_context(_PLOT_SETTINGS):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
        fig.suptitle("Данные: активность, популярность и временной протокол", fontsize=16)

        axis = axes[0, 0]
        user_counts = np.diff(train.indptr)
        positive = user_counts[user_counts > 0]
        if positive.size:
            if positive.max() == 1:
                bins = np.array([0.5, 1.5])
            else:
                bins = np.geomspace(0.8, positive.max() + 1, min(40, int(positive.max()) + 1))
            axis.hist(positive, bins=bins, color=_SPLIT_COLORS["train"], edgecolor="white", linewidth=0.5)
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.grid(axis="y", alpha=0.2)
        else:
            _empty(axis, "В train нет взаимодействий")
        axis.set(title="Активность пользователей в train", xlabel="Взаимодействий на пользователя, лог. шкала",
                 ylabel="Число пользователей, лог. шкала")

        axis = axes[0, 1]
        item_counts = np.diff(train.tocsc().indptr)
        popularity = np.sort(item_counts[item_counts > 0])[::-1]
        if popularity.size:
            axis.loglog(np.arange(1, popularity.size + 1), popularity, color=_SPLIT_COLORS["train"], linewidth=1.7)
            axis.grid(alpha=0.2, which="major")
        else:
            _empty(axis, "Нет объектов с взаимодействиями")
        axis.set(title="Популярность объектов в train", xlabel="Ранг объекта, лог. шкала",
                 ylabel="Число взаимодействий, лог. шкала")

        axis = axes[1, 0]
        plotted = False
        for name, label in (("train", "Train"), ("validation", "Validation"), ("test", "Test")):
            timestamps = np.asarray(data[f"{name}_timestamps"].data, dtype=np.float64)
            if timestamps.size == 0:
                continue
            if not np.isfinite(timestamps).all():
                plt.close(fig)
                raise ValueError("timeline timestamps must be finite UTC epoch seconds")
            days, counts = np.unique(np.floor(timestamps / 86400).astype(np.int64), return_counts=True)
            dates = np.datetime64("1970-01-01", "D") + days.astype("timedelta64[D]")
            axis.plot(dates, counts, marker="o", markersize=4, linewidth=1.5,
                      label=label, color=_SPLIT_COLORS[name])
            plotted = True
        if plotted:
            for key, label, color in (
                ("train_cutoff_utc", "Граница train", _SPLIT_COLORS["train"]),
                ("validation_cutoff_utc", "Граница validation", _SPLIT_COLORS["validation"]),
            ):
                value = data["summary"].get(key)
                if value:
                    cutoff = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
                    axis.axvline(cutoff, color=color, linestyle="--", linewidth=1, label=label)
            axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=7))
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m", tz=timezone.utc))
            axis.legend(fontsize=8, ncol=2, loc="best")
            axis.grid(alpha=0.2)
        else:
            _empty(axis, "Нет временных меток")
        axis.set(title="Взаимодействия по календарным дням UTC", xlabel="Дата UTC", ylabel="Число взаимодействий")

        axis = axes[1, 1]
        distribution = data["summary"].get("language_distribution")
        if distribution is not None:
            known_languages = [
                (str(row["language"]), int(row["n_items"])) for row in distribution
                if str(row["language"]).strip().casefold() not in {"", "unknown"}
                and int(row["n_items"]) > 0
            ]
            known_languages.sort(key=lambda row: (-row[1], row[0]))
            known_languages = known_languages[:10]
            labels = [name for name, count in known_languages]
            counts = np.asarray([count for name, count in known_languages])
            category_axis_label = "Язык"
        else:
            categories, counts = np.unique(language[language != 0], return_counts=True)
            order = np.lexsort((categories, -counts))[:10]
            categories, counts = categories[order], counts[order]
            labels = [f"Категория {int(value)}" for value in categories]
            category_axis_label = "ID категории языка"
        if counts.size:
            positions = np.arange(counts.size)
            axis.barh(positions, counts, color="#5a9178")
            axis.set_yticks(positions, labels)
            axis.invert_yaxis()
            axis.grid(axis="x", alpha=0.2)
        else:
            _empty(axis, "Метаданные языка отсутствуют")
        missing = int(np.count_nonzero(language == 0))
        total = language.size
        fraction = missing / total if total else 0.0
        axis.set(title=f"Язык: до 10 частых категорий; нет данных {missing}/{total} ({fraction:.1%})",
                 xlabel="Число объектов каталога", ylabel=category_axis_label)

        return [_save(fig, directory, "eda_overview.png")]


def _read_rows(path: Path, required: tuple[str, ...]) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"{path.name} does not contain required columns: {required}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path.name} contains no experiment results")
    return rows


def _numbers(rows: list[dict], key: str) -> np.ndarray:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Result column {key} must contain finite values")
    return values


def create_result_figures(
    results_dir: str | Path = "results", out_dir: str | Path = "figures",
) -> list[Path]:
    source = Path(results_dir)
    rows = _read_rows(source / "test_metrics.csv", ("model", "ndcg_at_10", "recall_at_10"))
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    names = [row["model"] for row in rows]
    palette = plt.get_cmap("tab10")
    colors = {name: palette(index % 10) for index, name in enumerate(names)}

    with plt.rc_context(_PLOT_SETTINGS):
        fig, axes = plt.subplots(1, 2, figsize=(13, max(4, 0.5 * len(rows) + 1.5)),
                                 sharey=True, constrained_layout=True)
        fig.suptitle("Финальное сравнение на test: полный каталог, K = 10", fontsize=15)
        positions = np.arange(len(rows))
        for axis, key, label in zip(axes, ("ndcg_at_10", "recall_at_10"), ("NDCG@10 — основная метрика", "Recall@10")):
            values = _numbers(rows, key)
            bars = axis.barh(positions, values, color=[colors[name] for name in names], height=0.7)
            axis.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=4, fontsize=9)
            axis.set_yticks(positions, names)
            axis.set_xlim(0, max(float(values.max()) * 1.28, 0.01))
            axis.set(title=label, xlabel="Среднее по пользователям")
            axis.grid(axis="x", alpha=0.2)
        axes[0].invert_yaxis()
        saved.append(_save(fig, directory, "test_comparison.png"))

        validation_path = source / "validation_trials.csv"
        if validation_path.exists():
            trials = _read_rows(validation_path, ("model", "trial", "fit_seconds", "ndcg_at_10", "recall_at_10"))
            model_order = list(dict.fromkeys(row["model"] for row in trials))
            fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
            fig.suptitle("Подбор на validation: каждая точка — конфигурация", fontsize=15)
            for index, name in enumerate(model_order):
                group = [row for row in trials if row["model"] == name]
                ndcg = _numbers(group, "ndcg_at_10")
                recall = _numbers(group, "recall_at_10")
                seconds = _numbers(group, "fit_seconds")
                color = colors.get(name, palette(index % 10))
                axes[0].scatter(seconds, ndcg, color=color, label=name, s=38, alpha=0.85)
                axes[1].scatter(recall, ndcg, color=color, label=name, s=38, alpha=0.85)
                best = int(np.argmax(ndcg))
                for axis, x_values in ((axes[0], seconds), (axes[1], recall)):
                    axis.scatter(x_values[best], ndcg[best], color=color, marker="*", s=180,
                                 edgecolors="black", linewidths=0.6, zorder=4)
            if np.all(_numbers(trials, "fit_seconds") > 0):
                axes[0].set_xscale("log")
            axes[0].set(title="Качество и время обучения", xlabel="Время обучения, с", ylabel="Validation NDCG@10")
            axes[1].set(title="Две метрики одного протокола", xlabel="Validation Recall@10", ylabel="Validation NDCG@10")
            axes[0].legend(fontsize=8, loc="best")
            for axis in axes:
                axis.grid(alpha=0.2)
            saved.append(_save(fig, directory, "validation_tradeoffs.png"))

        details_path = source / "evaluation_details.json"
        if details_path.exists():
            details = json.loads(details_path.read_text(encoding="utf-8"))
            curves = {name: np.asarray(values, dtype=np.float64)
                      for name, values in details.get("fm_training_bce", {}).items() if len(values)}
            if curves:
                fig, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
                for index, (name, losses) in enumerate(curves.items()):
                    if losses.ndim != 1 or not np.isfinite(losses).all():
                        plt.close(fig)
                        raise ValueError("FM loss curves must be finite one-dimensional arrays")
                    axis.plot(np.arange(1, losses.size + 1), losses, marker="o", markersize=3,
                              color=colors.get(name, palette(index % 10)), label=name, linewidth=1.7)
                axis.set(title="Обучение FM после выбора параметров", xlabel="Эпоха", ylabel="Средняя обучающая BCE")
                axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
                axis.legend(fontsize=9)
                axis.grid(alpha=0.2)
                saved.append(_save(fig, directory, "fm_training_loss.png"))
    return saved
