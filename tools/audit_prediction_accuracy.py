#!/usr/bin/env python3
"""Honest self-portrait: the distribution of DFT's prediction accuracy.

Motivation. The corpus's famous numbers -- alpha at 5.7 ppm, Koide at 0.5 ppm -- are quoted
everywhere, and it is easy to read them as typical. They may instead be the good tail of a
broad distribution, in which case "DFT derives constants to ppm" overstates what the corpus
actually contains. This tool answers that from the results JSONs rather than from READMEs,
because READMEs quote selectively and their tables mix errors with scores.

WHAT COUNTS AS A PREDICTION. Only a PREDICTED value paired with a MEASURED value in the SAME
dict -- co-location is the evidence that they refer to one quantity. That excludes internal
numerical diagnostics (pac_error, idempotency_error, drift, convergence residuals), which are
checks on the machinery rather than claims about nature, and which would otherwise flood the
distribution with machine-precision noise and make the corpus look far more accurate than it is.

KNOWN LIMITS, stated rather than hidden:
  * A dict pairing is a heuristic. Some genuine predictions record only an error and no pair,
    and are missed. Some pairs are internal comparisons that survive the exclusion list.
  * Nothing here weights by importance -- a headline constant and an incidental check count
    the same. The distribution is of RECORDED COMPARISONS, not of the framework's claims.
  * Duplicates across re-runs are collapsed on (experiment, label, rounded error).

Usage:  python tools/audit_prediction_accuracy.py [--json OUT] [--top N]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PRED = re.compile(r"^(dft|predicted|prediction|derived|theoretical|theory|expected|formula|"
                  r"calc(ulated)?|model)(_|$)|_(predicted|derived|dft|expected|theory)$")
MEAS = re.compile(r"^(measured|observed|experimental|actual|codata|pdg|reference|target|"
                  r"empirical|literature)(_|$)|_(measured|observed|actual|codata|exact)$")
# machinery checks, not claims about nature
INTERNAL = re.compile(r"pac_error|idempoten|drift|convergen|residual_norm|machine|"
                      r"roundtrip|round_trip|reconstruction|selftest|self_test|"
                      r"conservation|closure_error|numerical")


def rel_error(pred, meas):
    try:
        p, m = float(pred), float(meas)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(p) and math.isfinite(m)) or m == 0:
        return None
    if abs(p) > 1e12 or abs(m) > 1e12:
        return None
    r = abs(p - m) / abs(m)
    return r if 0 < r < 1e3 else None


def harvest(obj, ctx, out, depth=0):
    """Find dicts holding BOTH a predicted-like and a measured-like numeric value."""
    if depth > 8:
        return
    if isinstance(obj, dict):
        preds = {k: v for k, v in obj.items()
                 if PRED.search(k.lower()) and isinstance(v, (int, float))}
        meass = {k: v for k, v in obj.items()
                 if MEAS.search(k.lower()) and isinstance(v, (int, float))}
        if preds and meass:
            label = None
            for key in ("name", "quantity", "label", "constant", "description", "test"):
                if isinstance(obj.get(key), str):
                    label = obj[key][:60]; break
            blob = " ".join(list(preds) + list(meass) + [label or ""]).lower()
            if not INTERNAL.search(blob):
                pk, pv = sorted(preds.items())[0]
                mk, mv = sorted(meass.items())[0]
                r = rel_error(pv, mv)
                if r is not None:
                    out.append(dict(area=ctx["area"], exp=ctx["exp"],
                                    label=label or f"{pk} vs {mk}",
                                    pred_key=pk, meas_key=mk,
                                    predicted=float(pv), measured=float(mv),
                                    rel_error=r, ppm=r * 1e6, file=ctx["file"]))
        for v in obj.values():
            harvest(v, ctx, out, depth + 1)
    elif isinstance(obj, list):
        for v in obj[:60]:
            harvest(v, ctx, out, depth + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    files = sorted((ROOT / "experiments").glob("**/results/*.json"))
    recs, unread = [], 0
    for f in files:
        parts = f.relative_to(ROOT / "experiments").parts
        ctx = dict(area=parts[0], exp=parts[1] if len(parts) > 1 else "?",
                   file=str(f.relative_to(ROOT)))
        try:
            harvest(json.loads(f.read_text(encoding="utf-8")), ctx, recs)
        except Exception:
            unread += 1

    seen, uniq = set(), []
    for r in recs:
        k = (r["exp"], r["label"], round(r["ppm"], 3))
        if k not in seen:
            seen.add(k); uniq.append(r)

    print(f"files scanned {len(files)}  unreadable {unread}")
    print(f"paired predicted/measured comparisons: {len(recs)}  ({len(uniq)} distinct)\n")
    if not uniq:
        return 1

    ppm = sorted(r["ppm"] for r in uniq)
    n = len(ppm)
    bands = [(0, 1, "< 1 ppm"), (1, 10, "1-10 ppm"), (10, 100, "10-100 ppm"),
             (100, 1e3, "100 ppm - 0.1%"), (1e3, 1e4, "0.1% - 1%"),
             (1e4, 1e5, "1% - 10%"), (1e5, 1e9, "> 10%")]
    print(f"{'band':<18}{'count':>7}{'share':>8}  histogram")
    for lo, hi, name in bands:
        c = sum(1 for v in ppm if lo <= v < hi)
        print(f"{name:<18}{c:>7}{c/n:>7.1%}  {'#' * min(60, round(60*c/n))}")

    gm = math.exp(sum(math.log(v) for v in ppm) / n)
    print(f"\n  median {ppm[n//2]:,.1f} ppm    geometric mean {gm:,.1f} ppm")
    print(f"  <= 10 ppm: {sum(1 for v in ppm if v<=10)/n:.1%}"
          f"    >= 0.1%: {sum(1 for v in ppm if v>=1e3)/n:.1%}")

    print(f"\nbest {args.top} (with provenance):")
    for r in sorted(uniq, key=lambda r: r["ppm"])[:args.top]:
        print(f"  {r['ppm']:11.4f} ppm  {r['exp']:<22}{r['label'][:44]}")
    print(f"\nworst {args.top}:")
    for r in sorted(uniq, key=lambda r: -r["ppm"])[:args.top]:
        print(f"  {r['ppm']:11.0f} ppm  {r['exp']:<22}{r['label'][:44]}")

    by = defaultdict(list)
    for r in uniq:
        by[r["exp"]].append(r["ppm"])
    print(f"\n{'experiment':<26}{'n':>5}{'median ppm':>13}{'best ppm':>12}")
    for e, vs in sorted(by.items(), key=lambda kv: sorted(kv[1])[len(kv[1])//2])[:14]:
        vs = sorted(vs)
        print(f"{e:<26}{len(vs):>5}{vs[len(vs)//2]:>13,.1f}{vs[0]:>12,.3f}")

    if args.json:
        Path(args.json).write_text(json.dumps(uniq, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
