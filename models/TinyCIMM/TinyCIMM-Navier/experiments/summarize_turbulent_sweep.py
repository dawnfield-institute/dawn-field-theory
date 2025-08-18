import argparse
import json
import math
from pathlib import Path
from collections import defaultdict


def wilson_ci(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = successes / n
    denom = 1 + (z**2) / n
    center = p + (z**2) / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + (z**2) / (4 * n)) / n)
    low = (center - margin) / denom
    high = (center + margin) / denom
    return (p, max(0.0, low), min(1.0, high))


def main():
    parser = argparse.ArgumentParser(description="Summarize turbulent sweep results with Wilson CIs")
    parser.add_argument("results_json", help="Path to comprehensive_live_cimm_results.json")
    parser.add_argument("--outdir", help="Directory to write summary files (defaults to JSON's parent)")
    args = parser.parse_args()

    results_path = Path(args.results_json)
    if not results_path.exists():
        raise FileNotFoundError(f"Results JSON not found: {results_path}")

    with results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    phase4 = data.get("phase_4_turbulent_challenge", {})

    # Aggregate by (challenge_name, complexity)
    agg = defaultdict(lambda: {"successes": 0, "total": 0, "ttb": []})

    for key, entry in phase4.items():
        # Each entry should have a challenge_config; skip aggregates if any
        cfg = entry.get("challenge_config")
        if not cfg:
            continue
        name = cfg.get("name", "unknown")
        complexity = cfg.get("complexity")
        # normalize complexity to a string like 1.00 to avoid float key issues
        try:
            c_key = f"{float(complexity):.2f}"
        except (TypeError, ValueError):
            c_key = str(complexity)

        bt = bool(entry.get("breakthrough_detected", False))
        agg[(name, c_key)]["total"] += 1
        if bt:
            agg[(name, c_key)]["successes"] += 1
            step = entry.get("breakthrough_step")
            try:
                if step is not None:
                    agg[(name, c_key)]["ttb"].append(int(step))
            except Exception:
                pass

    # Build rows with Wilson CI
    rows = []
    for (name, c_key), stats in sorted(agg.items(), key=lambda x: (x[0][0], float(x[0][1]))):
        s = stats["successes"]
        n = stats["total"]
        rate, ci_low, ci_high = wilson_ci(s, n)
        ttb_list = stats.get("ttb", [])
        ttb_mean = sum(ttb_list)/len(ttb_list) if ttb_list else None
        ttb_median = None
        if ttb_list:
            ttb_sorted = sorted(ttb_list)
            mid = len(ttb_sorted)//2
            if len(ttb_sorted) % 2 == 1:
                ttb_median = ttb_sorted[mid]
            else:
                ttb_median = 0.5 * (ttb_sorted[mid-1] + ttb_sorted[mid])
        rows.append(
            {
                "challenge": name,
                "complexity": c_key,
                "successes": s,
                "total": n,
                "rate": round(rate, 4),
                "ci_low": round(ci_low, 4),
                "ci_high": round(ci_high, 4),
                "ttb_mean": round(ttb_mean, 2) if ttb_mean is not None else None,
                "ttb_median": round(ttb_median, 2) if ttb_median is not None else None,
            }
        )

    outdir = Path(args.outdir) if args.outdir else results_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    out_json = outdir / "sweep_success_by_challenge_complexity.json"
    out_csv = outdir / "sweep_success_by_challenge_complexity.csv"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    # Write CSV
    headers = ["challenge", "complexity", "successes", "total", "rate", "ci_low", "ci_high", "ttb_mean", "ttb_median"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(
                f"{r['challenge']},{r['complexity']},{r['successes']},{r['total']},{r['rate']},{r['ci_low']},{r['ci_high']},{'' if r['ttb_mean'] is None else r['ttb_mean']},{'' if r['ttb_median'] is None else r['ttb_median']}\n"
            )

    # Print compact summary to stdout
    print("Per-challenge/per-complexity breakthrough rates (95% Wilson CI):")
    # Group for printing by challenge
    by_challenge = defaultdict(list)
    for r in rows:
        by_challenge[r["challenge"]].append(r)
    for name, lst in by_challenge.items():
        lst_sorted = sorted(lst, key=lambda r: float(r["complexity"]))
        print(f"  {name}:")
        for r in lst_sorted:
            ttb_part = ""
            if r.get("ttb_mean") is not None:
                ttb_part = f"; TTB mean/med={r['ttb_mean']}/{r.get('ttb_median')}"
            print(
                f"    c={r['complexity']}: {r['successes']}/{r['total']} = {r['rate']:.2f} (CI {r['ci_low']:.2f}-{r['ci_high']:.2f}){ttb_part}"
            )

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
