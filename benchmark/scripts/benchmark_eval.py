"""Standalone scorer for the compact AVC-Benchmark CSV.

The compact input schema is:

* ``id``;
* ``human_macro``, ``human_temporal``, ``human_fine_grained``;
* scalar model columns named exactly as their figure legends;
* three-dimensional model columns named ``MODEL__macro``,
  ``MODEL__temporal``, and ``MODEL__fine_grained``.

Every numeric scalar column added to the CSV is automatically treated as a
new one-dimensional predictor. Every complete ``__macro/__temporal/__fine_grained``
group is automatically treated as a new three-dimensional predictor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr


DEFAULT_DELTAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]
DIMS = ["macro", "temporal", "fine_grained"]
RESERVED = {"id", "human_macro", "human_temporal", "human_fine_grained"}


def row_nanmean(values):
    values = np.asarray(values, dtype=float)
    count = np.isfinite(values).sum(axis=1)
    total = np.nansum(values, axis=1)
    return np.divide(total, count, out=np.full(len(values), np.nan), where=count > 0)


def logistic5(x, b1, b2, b3, b4, b5):
    z = np.clip(b2 * (x - b3), -60.0, 60.0)
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(z))) + b4 * x + b5


def finite_pair(pred, gt):
    mask = np.isfinite(pred) & np.isfinite(gt)
    return np.asarray(pred, dtype=float)[mask], np.asarray(gt, dtype=float)[mask]


def plcc5(pred, gt):
    p, g = finite_pair(pred, gt)
    if len(p) < 3 or np.std(p) == 0 or np.std(g) == 0:
        return np.nan, False
    p0 = [np.max(g), 0.1, np.mean(p), 0.1, 0.1]
    try:
        params, _ = curve_fit(logistic5, p, g, p0=p0, maxfev=10000)
        return float(pearsonr(logistic5(p, *params), g).statistic), True
    except Exception:
        return float(pearsonr(p, g).statistic), False


def srcc(pred, gt):
    p, g = finite_pair(pred, gt)
    if len(p) < 3 or np.std(p) == 0 or np.std(g) == 0:
        return np.nan
    return float(spearmanr(p, g).statistic)


def pairwise(pred, gt, delta):
    p, g = finite_pair(pred, gt)
    if len(p) < 2:
        return np.nan, 0
    i, j = np.triu_indices(len(p), 1)
    dg, dp = g[i] - g[j], p[i] - p[j]
    use = np.abs(dg) > delta
    n = int(use.sum())
    if n == 0:
        return np.nan, 0
    correct = (dp[use] * dg[use] > 0).sum() + 0.5 * (dp[use] == 0).sum()
    return float(correct / n), n


def discover_predictors(df):
    grouped = {}
    scalar = []
    for col in df.columns:
        if col in RESERVED:
            continue
        if col.endswith("__macro"):
            grouped.setdefault(col[:-7], {})["macro"] = col
        elif col.endswith("__temporal"):
            grouped.setdefault(col[:-10], {})["temporal"] = col
        elif col.endswith("__fine_grained"):
            grouped.setdefault(col[:-14], {})["fine_grained"] = col
        else:
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().any():
                scalar.append((col, numeric.to_numpy(float)))

    predictors = {}
    for name, columns in grouped.items():
        if all(dim in columns for dim in DIMS):
            values = {dim: pd.to_numeric(df[columns[dim]], errors="coerce").to_numpy(float) for dim in DIMS}
            values["overall_arithmetic"] = row_nanmean(np.column_stack([values[dim] for dim in DIMS]))
            predictors[name] = values
    for name, values in scalar:
        if name not in predictors:
            predictors[name] = {"overall_arithmetic": values}
    return predictors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("benchmark.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("metrics"))
    parser.add_argument("--deltas", type=float, nargs="*", default=DEFAULT_DELTAS)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    missing_human = {"human_macro", "human_temporal", "human_fine_grained"} - set(df.columns)
    if missing_human:
        raise ValueError(f"Missing human target columns: {sorted(missing_human)}")
    gt_dims = {dim: pd.to_numeric(df[f"human_{dim}"], errors="coerce").to_numpy(float) for dim in DIMS}
    gt_dims["overall_arithmetic"] = row_nanmean(np.column_stack([gt_dims[dim] for dim in DIMS]))
    model_map = discover_predictors(df)

    rows = []
    for predictor, values in model_map.items():
        dimensions = DIMS + ["overall_arithmetic"] if all(dim in values for dim in DIMS) else ["overall_arithmetic"]
        for dimension in dimensions:
            pred = values[dimension]
            gt = gt_dims[dimension]
            p, _ = finite_pair(pred, gt)
            plcc, fit_ok = plcc5(pred, gt)
            rank = srcc(pred, gt)
            for delta in args.deltas:
                acc, pair_n = pairwise(pred, gt, float(delta))
                rows.append({
                    "predictor": predictor,
                    "dimension": dimension,
                    "n": int(len(p)),
                    "delta": float(delta),
                    "pairwise_acc": acc,
                    "pairwise_pairs": pair_n,
                    "PLCC_5PL": plcc,
                    "PLCC_mapping_success": fit_ok,
                    "SRCC": rank,
                })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "benchmark_metrics.csv", index=False, encoding="utf-8-sig")
    summary = {
        "csv": str(args.csv),
        "rows": int(len(df)),
        "predictors": sorted(model_map),
        "dimensions": {name: [dim for dim in DIMS if dim in values] for name, values in model_map.items()},
        "deltas": [float(x) for x in args.deltas],
        "pairwise_rule": "abs(gt_i - gt_j) > delta; prediction ties receive 0.5 credit",
        "plcc": "Pearson correlation after unbounded five-parameter logistic fit; fallback to raw Pearson on fit failure",
        "new_numeric_columns": "automatically discovered as scalar predictors unless they use the __macro/__temporal/__fine_grained grouped schema",
    }
    (args.output_dir / "benchmark_metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
