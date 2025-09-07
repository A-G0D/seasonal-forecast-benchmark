"""Run the benchmark and write the eval artifacts (baseline/improved per-series
results, the comparison summary as json + markdown). A JSONL trace of every fit
goes to the configured log path. Run it with `python eval/run_benchmark.py`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# allow running directly: put the project root on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.determinism import set_seed
from shared.obs import Observer

from src.backtest import Benchmark, BacktestResult
from src.metrics import LOWER_IS_BETTER
from src.registry import BASELINE_MODELS, IMPROVED_MODELS
from src.series import SERIES_LIBRARY

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "eval"


def _load_config() -> dict:
    with (ROOT / "config.json").open(encoding="utf-8") as fh:
        return json.load(fh)


def _dump_results(results: list[BacktestResult], path: Path) -> None:
    payload = [r.as_record(include_forecast=False) for r in results]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _winner(baseline: dict, improved: dict, metric: str) -> str:
    # all our metrics are lower-is-better
    return "improved" if improved[metric] < baseline[metric] else "baseline"


def main() -> None:
    cfg = _load_config()
    set_seed(cfg["seed"])
    horizon = cfg["horizon"]

    specs = [SERIES_LIBRARY[name] for name in cfg["series"]]

    log_path = ROOT / cfg.get("log_path", "logs/trace.jsonl")
    with Observer("backtest", sink=log_path, deterministic=True) as obs:
        base_bench = Benchmark(specs, BASELINE_MODELS, horizon=horizon,
                               seed=cfg["seed"])
        imp_bench = Benchmark(specs, IMPROVED_MODELS, horizon=horizon,
                              seed=cfg["seed"])
        base_results = base_bench.run(observer=obs)
        imp_results = imp_bench.run(observer=obs)

    EVAL_DIR.mkdir(exist_ok=True)
    _dump_results(base_results, EVAL_DIR / "baseline_results.json")
    _dump_results(imp_results, EVAL_DIR / "improved_results.json")

    base_agg = Benchmark.aggregate(base_results)
    imp_agg = Benchmark.aggregate(imp_results)

    # best (lowest) value per metric on each side
    best_base = {m: min(base_agg[k][m] for k in base_agg) for m in LOWER_IS_BETTER}
    best_imp = {m: min(imp_agg[k][m] for k in imp_agg) for m in LOWER_IS_BETTER}

    comparison = {
        "horizon": horizon,
        "seed": cfg["seed"],
        "series": cfg["series"],
        "baseline_per_model": base_agg,
        "improved_per_model": imp_agg,
        "baseline_best": best_base,
        "improved_best": best_imp,
        "winner_by_metric": {
            m: _winner(best_base, best_imp, m) for m in LOWER_IS_BETTER
        },
    }
    (EVAL_DIR / "comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    _write_markdown(comparison)
    print("wrote eval artifacts to", EVAL_DIR)
    improved_wins = sum(
        1 for m in LOWER_IS_BETTER if comparison["winner_by_metric"][m] == "improved"
    )
    print(f"improved wins on {improved_wins}/{len(LOWER_IS_BETTER)} metrics")


def _write_markdown(c: dict) -> None:
    lines: list[str] = []
    lines.append("# Baseline vs improved forecasters")
    lines.append("")
    lines.append(
        f"Holdout horizon: **{c['horizon']}** steps. Seed: {c['seed']}. "
        f"Averaged over series: {', '.join(c['series'])}."
    )
    lines.append("")
    lines.append("## Per-model averages")
    lines.append("")
    header = "| model | " + " | ".join(m.upper() for m in LOWER_IS_BETTER) + " |"
    sep = "|" + "---|" * (len(LOWER_IS_BETTER) + 1)
    lines.append(header)
    lines.append(sep)
    for group in (c["baseline_per_model"], c["improved_per_model"]):
        for model, scores in group.items():
            row = "| " + model + " | " + " | ".join(
                f"{scores[m]:.3f}" for m in LOWER_IS_BETTER
            ) + " |"
            lines.append(row)
    lines.append("")
    lines.append("## Best on each side")
    lines.append("")
    lines.append("| metric | best baseline | best improved | winner |")
    lines.append("|---|---|---|---|")
    for m in LOWER_IS_BETTER:
        lines.append(
            f"| {m.upper()} | {c['baseline_best'][m]:.3f} | "
            f"{c['improved_best'][m]:.3f} | {c['winner_by_metric'][m]} |"
        )
    lines.append("")
    improved_wins = sum(
        1 for m in LOWER_IS_BETTER if c["winner_by_metric"][m] == "improved"
    )
    lines.append(
        f"The improved models win on {improved_wins} of {len(LOWER_IS_BETTER)} "
        f"metrics against the strongest baseline."
    )
    lines.append("")
    (EVAL_DIR / "COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
