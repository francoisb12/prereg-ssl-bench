#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Re-read experiment A's raw numbers and describe what P2.1 could not see.

WHY THIS IS NOT A NEW TEST
--------------------------
P2.1 predicted that the pure relational objective would collapse the effective
rank towards 1.  It did not: rank went 13.4 -> 15.2 and the verdict is, and
stays, "infirmee".  That verdict is not to be revisited.

But P2.1's falsifier (`rank < 3`) tested TOTAL collapse, while the phenomenon
actually present is a severe PARTIAL concentration that lives in a different
metric.  Choosing a new threshold now, knowing the answer, would be HARKing --
so this script deliberately does not choose one.  It reports quantities, with
verdict class "exploratoire", which aggregate.py keeps separate from
confirmee/infirmee.

To turn any of this into evidence you need a threshold fixed BEFORE new data
exists.  That has been done for the rank-vs-transfer question: see
PREREG["P6_rank_does_not_predict_transfer"] in xp_B_relational_vs_prototype.py,
registered before B was ever run.

Costs nothing: pure JSON re-analysis, no GPU, no training.

    python reanalyse_A.py
    python reanalyse_A.py --summary results/A/A_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))


def mean_sd(xs: List[float]):
    xs = [x for x in xs if x == x]
    if not xs:
        return float("nan"), float("nan")
    return st.mean(xs), (st.pstdev(xs) if len(xs) > 1 else 0.0)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", default=os.path.join(HERE, "results", "A", "A_summary.json"))
    p.add_argument("--out", default=None, help="défaut : <dossier du summary>/A2_exploratory.json")
    args = p.parse_args(argv)

    if not os.path.exists(args.summary):
        print(f"Introuvable : {args.summary}\n"
              f"Lance d'abord `python xp/xp_A_diagnostics.py`, ou passe --summary.")
        return 1
    with open(args.summary, encoding="utf-8") as fh:
        blob = json.load(fh)

    per_arm = (blob.get("A2_raw") or {}).get("per_arm") or {}
    if not per_arm:
        print("Ce summary ne contient pas A2_raw.per_arm : rien à relire.")
        return 1

    proj_dim = int((blob.get("config") or {}).get("proj_dim") or 0) or None
    rows: Dict[str, Any] = {}
    for arm, runs in per_arm.items():
        ceil = mean_sd([r.get("std_ceiling_normalised", float("nan")) for r in runs])[0]
        if not (ceil == ceil) or ceil <= 0:
            ceil = 1.0 / math.sqrt(proj_dim) if proj_dim else float("nan")
        rank_m, rank_s = mean_sd([r.get("eff_rank_z", float("nan")) for r in runs])
        sig_m, sig_s = mean_sd([r.get("mean_std_z_normalised", float("nan")) for r in runs])
        knn_m, knn_s = mean_sd([r.get("knn_acc", float("nan")) for r in runs])
        al_m, _ = mean_sd([r.get("alignment", float("nan")) for r in runs])
        un_m, _ = mean_sd([r.get("uniformity", float("nan")) for r in runs])
        rows[arm] = {
            "n_seeds": len(runs),
            "eff_rank_mean": rank_m, "eff_rank_sd": rank_s,
            "eff_rank_frac_of_dim": (rank_m / proj_dim) if proj_dim else None,
            # THE metric P2.1 should have used: how much of the spread that is
            # geometrically AVAILABLE on the unit sphere does the code actually use.
            "sigma_over_ceiling": (sig_m / ceil) if ceil == ceil else None,
            "sigma_normalised_mean": sig_m, "sigma_normalised_sd": sig_s,
            "hard_ceiling_1_over_sqrt_d": ceil,
            "alignment_mean": al_m, "uniformity_mean": un_m,
            "knn_mean": knn_m, "knn_sd": knn_s,
        }

    # Rank vs transfer, described only.  The pre-registered test lives in B.
    ranked = sorted(rows.items(), key=lambda kv: kv[1]["eff_rank_mean"])
    anomaly = None
    if len(ranked) >= 2:
        lo_name, lo = ranked[0]
        pairs = [(n, v) for n, v in ranked[1:]
                 if v["knn_mean"] == v["knn_mean"] and v["knn_mean"] < lo["knn_mean"]]
        if pairs:
            hi_name, hi = pairs[0]
            pooled = math.sqrt((lo["knn_sd"] ** 2 + hi["knn_sd"] ** 2) / 2) or float("nan")
            anomaly = {
                "lowest_rank_arm": lo_name, "compared_to": hi_name,
                "rank_ratio": hi["eff_rank_mean"] / lo["eff_rank_mean"],
                "knn_gap_pp": 100.0 * (lo["knn_mean"] - hi["knn_mean"]),
                "pooled_sd_pp": 100.0 * pooled,
                "gap_in_pooled_sd": ((lo["knn_mean"] - hi["knn_mean"]) / pooled)
                                    if pooled == pooled and pooled > 0 else None,
                "reading": f"{hi_name} a un rang effectif "
                           f"{hi['eff_rank_mean'] / lo['eff_rank_mean']:.1f}x superieur a "
                           f"{lo_name} et un kNN INFERIEUR. Le rang ne suit pas le transfert.",
            }

    out = {
        "source_summary": os.path.abspath(args.summary),
        "status": "RELECTURE EXPLORATOIRE : aucune de ces lignes n'est un test",
        "why": "P2.1 reste 'infirmee' et n'est pas revisitee. Son falsifieur (rang < 3) "
               "testait l'effondrement TOTAL ; la concentration reelle se mesure en "
               "sigma/plafond et en uniformite. Choisir un seuil maintenant, en connaissant "
               "la reponse, serait du HARKing : aucun seuil n'est donc applique ici.",
        "where_the_real_test_lives": "PREREG['P6_rank_does_not_predict_transfer'] dans "
                                     "xp_B_relational_vs_prototype.py, enregistree avant "
                                     "le premier run de B.",
        "proj_dim": proj_dim,
        "predictions": {
            "P2.4_spread_utilisation": {
                "thesis": "T2 (descriptif)",
                "statement": "part de l'etalement geometriquement disponible que chaque "
                             "objectif utilise reellement, sur le code L2-normalise",
                "falsifier": "AUCUN : rapporte sans seuil, par construction",
                "verdict": "exploratoire",
                "evidence": {k: {"sigma_over_ceiling": v["sigma_over_ceiling"],
                                 "eff_rank_frac_of_dim": v["eff_rank_frac_of_dim"],
                                 "alignment": v["alignment_mean"],
                                 "uniformity": v["uniformity_mean"],
                                 "n_seeds": v["n_seeds"]} for k, v in rows.items()},
            },
            "P2.5_rank_vs_transfer": {
                "thesis": "non prevue : apparue dans les donnees",
                "statement": "le rang effectif ne predit pas la qualite du transfert",
                "falsifier": "AUCUN ici. Test pre-enregistre : P6 dans l'experience B",
                "verdict": "exploratoire",
                "evidence": anomaly or {"note": "aucune inversion rang/kNN dans ces bras"},
            },
        },
        "per_arm": rows,
    }

    dest = args.out or os.path.join(os.path.dirname(os.path.abspath(args.summary)),
                                    "A2_exploratory.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    w = max(len(a) for a in rows) if rows else 8
    line = "-" * (w + 62)
    print(line)
    print("RELECTURE EXPLORATOIRE DE A2 : aucun seuil, aucun verdict de test")
    print(line)
    print(f"{'bras':{w}} {'rang/dim':>9} {'sigma/plafond':>14} {'align':>8} {'unif':>8} {'kNN':>8}")
    for arm, v in ranked:
        fr = f"{100 * v['eff_rank_frac_of_dim']:.0f}%" if v["eff_rank_frac_of_dim"] else "?"
        so = f"{100 * v['sigma_over_ceiling']:.1f}%" if v["sigma_over_ceiling"] else "?"
        print(f"{arm:{w}} {fr:>9} {so:>14} {v['alignment_mean']:8.3f} "
              f"{v['uniformity_mean']:8.3f} {100 * v['knn_mean']:7.1f}%")
    print(line)
    if anomaly:
        print("  " + anomaly["reading"])
        if anomaly.get("gap_in_pooled_sd") is not None:
            print(f"  ecart kNN {anomaly['knn_gap_pp']:+.2f} pp "
                  f"= {anomaly['gap_in_pooled_sd']:.1f} ecarts-types groupes "
                  f"({rows[anomaly['lowest_rank_arm']]['n_seeds']} graines)")
        print("  -> test pre-enregistre : P6 dans l'experience B, seuil fixe avant le run")
    print(line)
    print(f"ecrit -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
