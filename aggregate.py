#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sweep results/ and produce ONE verdict table across all theses.

Without this you end up, three weeks in, with two hundred CSVs and no memory of
which run produced which number under which budget.  This script is deliberately
dumb: it reads the summary.json files the experiment scripts already write, and
refuses to interpret anything they did not state themselves.

Two rules it enforces, because they are the ones that get broken quietly:

  1. A run produced in ``--smoke`` mode is NEVER a result.  Smoke summaries are
     detected and listed separately, never mixed into a verdict.
  2. A verdict backed by fewer than ``--min-seeds`` seeds is downgraded to
     "insuffisant", whatever the script concluded.  A single seed decides
     nothing, and the honest failure mode here is to forget that.

Usage
-----
    python aggregate.py                       # table to stdout
    python aggregate.py --csv synthese.csv    # also write it
    python aggregate.py --min-seeds 3         # default
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))

# Every verdict string the scripts are allowed to emit, normalised.
VERDICTS = {
    "confirmee": "confirmée",
    "confirmée": "confirmée",
    "confirmed": "confirmée",
    "infirmee": "infirmée",
    "infirmée": "infirmée",
    "refuted": "infirmée",
    "non concluante": "non concluante",
    "inconclusive": "non concluante",
    "exploratoire": "exploratoire",
    "exploratory": "exploratoire",
    "go": "GO",
    "no-go": "NO-GO",
    "nogo": "NO-GO",
}

ORDER = ["confirmée", "infirmée", "non concluante", "exploratoire",
         "GO", "NO-GO", "insuffisant", "?"]


def norm_verdict(v: Any) -> str:
    if not isinstance(v, str):
        return "?"
    return VERDICTS.get(v.strip().lower(), v.strip())


def looks_like_smoke(path: str, blob: Dict[str, Any]) -> bool:
    """A smoke summary must never reach a verdict table."""
    if "smoke" in path.lower():
        return True
    for key in ("smoke", "is_smoke"):
        if bool(blob.get(key, False)):
            return True
    cfg = blob.get("config") or blob.get("cfg") or {}
    if isinstance(cfg, dict):
        if bool(cfg.get("smoke", False)):
            return True
        # A run with a handful of steps is a smoke run whatever it calls itself.
        for key in ("steps", "c1_steps", "e0_steps"):
            n = cfg.get(key)
            if isinstance(n, (int, float)) and 0 < n <= 50:
                return True
    return False


def count_seeds(blob: Dict[str, Any]) -> Optional[int]:
    for key in ("n_seeds", "seeds", "num_seeds"):
        v = blob.get(key)
        if isinstance(v, int):
            return v
        if isinstance(v, (list, tuple)):
            return len(v)
    cfg = blob.get("config") or blob.get("cfg") or {}
    if isinstance(cfg, dict):
        for key in ("n_seeds", "seeds"):
            v = cfg.get(key)
            if isinstance(v, int):
                return v
            if isinstance(v, (list, tuple)):
                return len(v)
    return None


def walk_verdicts(blob: Any, prefix: str = "") -> List[Dict[str, Any]]:
    """Find every {name -> verdict} pair, whatever nesting the script chose.

    The six scripts were written independently and do not agree on a schema, so
    this walks rather than assumes.  A dict with a 'verdict' key is a verdict;
    a flat {name: "confirmee"} mapping is a verdict table.
    """
    found: List[Dict[str, Any]] = []
    if isinstance(blob, dict):
        if "verdict" in blob and isinstance(blob.get("verdict"), str):
            # This dict IS one verdict record.  Return immediately: scanning its
            # own string values as well would count it twice (once as the record,
            # once as a bare {key: "confirmee"} pair).
            return [{
                "prediction": (prefix.rstrip(".") if prefix else blob.get("name", "?")),
                "verdict": norm_verdict(blob["verdict"]),
                "evidence": blob.get("evidence", blob.get("statement", "")),
            }]
        for k, v in blob.items():
            if isinstance(v, str) and norm_verdict(v) in ORDER and norm_verdict(v) != "?":
                found.append({"prediction": f"{prefix}{k}" if prefix else k,
                              "verdict": norm_verdict(v), "evidence": ""})
            elif isinstance(v, (dict, list)):
                found.extend(walk_verdicts(v, prefix=f"{prefix}{k}." if prefix else f"{k}."))
    elif isinstance(blob, list):
        for i, v in enumerate(blob):
            found.extend(walk_verdicts(v, prefix=f"{prefix}{i}."))
    return found


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=os.path.join(HERE, "results"))
    p.add_argument("--min-seeds", type=int, default=3,
                   help="below this, a verdict is downgraded to 'insuffisant'")
    p.add_argument("--csv", default=None, help="also write the table here")
    p.add_argument("--show-smoke", action="store_true",
                   help="list the smoke summaries that were excluded")
    args = p.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.results, "**", "*.json"), recursive=True))
    if not paths:
        print(f"Aucun summary trouvé sous {args.results}.")
        print("Lance au moins une expérience, ou vérifie --results.")
        return 1

    rows: List[Dict[str, Any]] = []
    smoked: List[str] = []
    unreadable: List[str] = []

    for path in paths:
        try:
            with open(path, encoding="utf-8") as fh:
                blob = json.load(fh)
        except Exception as exc:                       # noqa: BLE001
            unreadable.append(f"{path}: {exc}")
            continue
        if not isinstance(blob, dict):
            continue
        if looks_like_smoke(path, blob):
            smoked.append(path)
            continue

        seeds = count_seeds(blob)
        group = os.path.basename(os.path.dirname(path)) or "?"
        for v in walk_verdicts(blob):
            verdict = v["verdict"]
            note = ""
            if seeds is not None and seeds < args.min_seeds:
                note = f"downgradé: {seeds} graine(s) < {args.min_seeds}"
                verdict = "insuffisant"
            elif seeds is None:
                note = "nombre de graines non déclaré"
            rows.append({
                "groupe": group,
                "prediction": v["prediction"],
                "verdict": verdict,
                "graines": seeds if seeds is not None else "?",
                "note": note,
                "fichier": os.path.relpath(path, HERE),
            })

    if not rows:
        print("Aucun verdict exploitable.")
        if smoked:
            print(f"({len(smoked)} summary de mode --smoke exclus, comme il se doit.)")
        return 1

    rows.sort(key=lambda r: (r["groupe"], ORDER.index(r["verdict"])
                             if r["verdict"] in ORDER else 99, r["prediction"]))

    w_g = max(6, max(len(str(r["groupe"])) for r in rows))
    w_p = min(52, max(10, max(len(str(r["prediction"])) for r in rows)))
    w_v = max(8, max(len(str(r["verdict"])) for r in rows))

    line = "-" * (w_g + w_p + w_v + 26)
    print(line)
    print("SYNTHÈSE DES VERDICTS")
    print(line)
    print(f"{'groupe':{w_g}}  {'prédiction':{w_p}}  {'verdict':{w_v}}  {'graines':>7}  note")
    print(line)
    for r in rows:
        pred = str(r["prediction"])
        if len(pred) > w_p:
            pred = pred[:w_p - 1] + "…"
        print(f"{str(r['groupe']):{w_g}}  {pred:{w_p}}  {str(r['verdict']):{w_v}}  "
              f"{str(r['graines']):>7}  {r['note']}")
    print(line)

    tally: Dict[str, int] = {}
    for r in rows:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    print("  " + " · ".join(f"{k}: {v}" for k, v in
                            sorted(tally.items(), key=lambda kv: ORDER.index(kv[0])
                                   if kv[0] in ORDER else 99)))
    if smoked:
        print(f"  {len(smoked)} summary de mode --smoke exclus (ce ne sont pas des résultats).")
        if args.show_smoke:
            for s in smoked:
                print(f"    - {os.path.relpath(s, HERE)}")
    if unreadable:
        print(f"  {len(unreadable)} fichier(s) illisible(s) :")
        for u in unreadable:
            print(f"    - {u}")
    print(line)

    if args.csv:
        import csv
        with open(args.csv, "w", encoding="utf-8", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"écrit -> {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
