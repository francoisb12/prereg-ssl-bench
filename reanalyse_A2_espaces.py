#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Re-read experiment A2 in the space the downstream metric actually lives in.

WHY THIS IS NOT A NEW TEST
--------------------------
A-001 reported an anomaly: gram_vicreg has 4.8x the effective rank of gram and
loses 2.3 points of kNN.  Journal entry A-002 (2026-08-26) retracted that
framing: `eff_rank_z` is measured on the PROJECTOR output while `knn_acc` is
measured on the BACKBONE.  Between those two arms the projector rank moves
x4.83 and the backbone rank moves by 0.04 sigma.

This script does NOT test any thesis.  It measures three things that A2 logged
partially or not at all, so that the mediator of the -2.29 pp can be named or
ruled out.  Everything here is verdict class "exploratoire".  Choosing a
threshold now, knowing the answer, would be HARKing.

WHAT IT MEASURES
----------------
1. TRAJECTORY (free, CSV only, no GPU, no data).  `eff_rank_h` is written to the
   CSV from step 0 onwards even though the summary does not carry it, so the
   initialisation value and the whole trajectory are already on disk.  Two
   questions: is the final value above or below initialisation, and are the arms
   still moving at the last step, i.e. is "undertrained" still live?

2. CENTRED RANK (needs the checkpoints).  `effective_rank` is UNCENTRED by
   deliberate design (harness docstring: the mean direction is part of the
   collapse story).  For z that is defensible.  For h it is not innocent: h is
   post-ReLU and therefore non-negative, so its spectrum is dominated by the
   shared mean direction, which can mask a real between-arm difference.  A-002's
   "backbone rank unchanged" is measured on the uncentred rank and must be
   checked centred before it is quoted again.

3. THE 2 x 2 GRID (needs the checkpoints).  A-002's reversal was confounded
   three ways at once: augmented views vs clean images, train pool vs test
   split, N = 512 vs N = 10000 (centring was matched: both numbers uncentred).
   The grid separates N from the other two; A2's own code logs
   eff_rank_h_cleantrain_c512, which separates augmentation from split at
   N = 512 (JOURNAL A-005).  Quoting it as a fact invites the correct objection
   that RankMe's Appendix G fixes N at 25600 and shows 10000 is needed for 95%
   of the final rank, so N = 512 is simply too small.  This pass measures the
   SAME statistic on both distributions at both N, centred and uncentred, so
   each axis can be read on its own:

       augmented views (train pool)  x  {512, 10000}
       clean test split              x  {512, 10000}

   The augmented views are rebuilt with the run's own frozen procedure
   (make_fixed_diagnostic_views at the run's seed), so N = 512 reproduces the
   number the run logged and N = 10000 extends the SAME permutation.  Subsets
   are nested by construction: the first 512 of the 10000 are the original 512.

4. CLASS ALLOCATION (needs the checkpoints).  Equal rank can hide a different
   allocation.  Trace-form Fisher ratio on L2-normalised backbone features:
   between-class scatter over within-class scatter.  If gram and gram_vicreg
   share a rank but differ here, the mediator is named: same capacity, worse
   allocation.  Labels are a DIAGNOSTIC ONLY, never training; features frozen.

   Measured on the CLEAN test split, because that is what the kNN probe reads.
   A2's own `eff_rank_h` is measured on AUGMENTED views, which is a second
   mismatch on top of the space mismatch.

USAGE
-----
    python reanalyse_A2_espaces.py --results results/A
    python reanalyse_A2_espaces.py --results results/A --checkpoints

On Colab, with the results already on Drive:

    !cd /content/ubergang_xp && python reanalyse_A2_espaces.py \
        --results /content/drive/MyDrive/francois/ubergang_xp/results/A \
        --checkpoints --data-root /content/ubergang_local/data

Cost: the trajectory pass is instant.  The checkpoint pass is 9 checkpoints x one
forward over the CIFAR-10 test split: a couple of minutes on an A100, roughly a
quarter of an hour on a Colab CPU runtime.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics as st
import sys
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["gram", "gram_vicreg", "gram_vicreg_sphere"]


def mean_sd(xs: List[float]):
    xs = [x for x in xs if x == x]
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return st.mean(xs), st.stdev(xs)


def se_of_difference(a: List[float], b: List[float]) -> float:
    """Standard error of (mean(b) - mean(a)), the quantity A-002 quotes in sigma."""
    a = [x for x in a if x == x]
    b = [x for x in b if x == x]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return math.sqrt(st.stdev(a) ** 2 / len(a) + st.stdev(b) ** 2 / len(b))


# ---------------------------------------------------------------------------
# 1.  Trajectory, from the CSVs.  No torch, no GPU, no data.
# ---------------------------------------------------------------------------

def trajectory(results_dir: str) -> dict:
    out: Dict[str, dict] = {}
    for arm in ARMS:
        per_seed = []
        for path in sorted(glob.glob(os.path.join(results_dir, f"A2_{arm}_seed*.csv"))):
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            pts = []
            for r in rows:
                v, s = r.get("eff_rank_h", ""), r.get("step", "")
                if v in ("", None) or s in ("", None):
                    continue
                try:
                    fv = float(v)
                except ValueError:
                    continue
                if fv != fv:                      # NaN rows are diverged points
                    continue
                pts.append((int(float(s)), fv))
            if not pts:
                continue
            pts.sort()
            first, last = pts[0], pts[-1]
            # "still moving": change over the last quarter of training, as a
            # fraction of the total change since init.  A number, not a threshold.
            cut = first[0] + 0.75 * (last[0] - first[0])
            tail = [v for s, v in pts if s >= cut]
            total = last[1] - first[1]
            tail_change = (tail[-1] - tail[0]) if len(tail) >= 2 else float("nan")
            per_seed.append({
                "file": os.path.basename(path),
                "n_points": len(pts),
                "step_first": first[0], "eff_rank_h_at_init": first[1],
                "step_last": last[0], "eff_rank_h_final": last[1],
                "total_change": total,
                "last_quarter_change": tail_change,
                "last_quarter_over_total": (tail_change / total)
                                           if total not in (0.0,) and total == total
                                           else float("nan"),
            })
        if per_seed:
            out[arm] = {
                "per_seed": per_seed,
                "eff_rank_h_at_init_mean_sd": mean_sd([p["eff_rank_h_at_init"] for p in per_seed]),
                "eff_rank_h_final_mean_sd": mean_sd([p["eff_rank_h_final"] for p in per_seed]),
                "last_quarter_change_mean_sd": mean_sd([p["last_quarter_change"] for p in per_seed]),
            }
    return out


# ---------------------------------------------------------------------------
# 2 and 3.  Centred rank and class allocation, from the checkpoints.
# ---------------------------------------------------------------------------

def checkpoint_pass(results_dir: str, data_root: str, encoder_variant: str,
                    n_eval: Optional[int], device_str: Optional[str],
                    grid_ns: Optional[List[int]] = None) -> dict:
    """Reload each checkpoint and measure h on the split the kNN probe reads.

    Everything that can be read from the run's own ``.config.json`` IS read from
    it (arch, hidden, proj_dim, dataset, image_size, data_source) rather than
    retyped on the command line: a mismatch there would silently build a
    different model and produce numbers that look perfectly fine.  The
    evaluation data is built by A's own ``build_eval_data``, so the
    preprocessing is identical to the probe's rather than merely similar.
    """
    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, os.path.join(HERE, "lib"))
    sys.path.insert(0, os.path.join(HERE, "xp"))
    from harness import make_encoder, effective_rank, get_device       # noqa: E402
    import xp_A_diagnostics as A                                       # noqa: E402

    device = torch.device(device_str) if device_str else get_device()
    print(f"[reanalyse] device = {device}")
    grid_ns = sorted(set(int(n) for n in (grid_ns or [512, 10000])))
    n_grid_max = max(grid_ns)
    view_cache: Dict[int, object] = {}

    def augmented_views(cfg: dict, seed: int):
        """The run's own frozen diagnostic views, extended to n_grid_max.

        Same dataset, same construction and same seed as the run: at n = 512
        this reproduces the `eff_rank_h` the CSV logged, and a larger n extends
        the same permutation rather than drawing a fresh sample.
        """
        if seed in view_cache:
            return view_cache[seed]
        a = A.build_parser().parse_args([])
        a.synthetic = (cfg.get("data_source") == "synthetic")
        a.synthetic_train = getattr(a, "synthetic_train", 192)
        a.synthetic_classes = getattr(a, "synthetic_classes", 4)
        a.dataset = cfg.get("dataset", getattr(a, "dataset", "cifar10"))
        a.image_size = cfg.get("image_size", getattr(a, "image_size", 32))
        a.data_root = data_root
        a.seed = int(cfg.get("seed", 0))
        a.train_subset = cfg.get("train_subset")
        ds, _nc, source = A.build_ssl_data(a)
        n = min(n_grid_max, len(ds))
        if n < n_grid_max:
            print(f"[reanalyse] pool de {len(ds)} images : N plafonne a {n}")
        v1, _v2 = A.make_fixed_diagnostic_views(ds, n, seed=seed)
        print(f"[reanalyse] vues augmentees ({source}) graine {seed} : "
              f"{tuple(v1.shape)}")
        view_cache[seed] = v1
        return v1

    @torch.no_grad()
    def backbone_of(model, x: "torch.Tensor", bs: int = 256) -> "torch.Tensor":
        model.eval()
        out = []
        for i in range(0, x.shape[0], bs):
            out.append(model.backbone(x[i:i + bs].to(device)).float().cpu())
        return torch.cat(out)

    def rank_grid(H: "torch.Tensor", tag: str) -> Dict[str, float]:
        """Effective rank of the first n rows of H, centred and uncentred.

        Nested subsets: the first 512 rows of the 10000 ARE the original 512,
        so the two columns differ by N alone.
        """
        out: Dict[str, float] = {}
        for n in grid_ns:
            if H.shape[0] < n:
                out[f"eff_rank_h_{tag}_c{n}"] = float("nan")
                out[f"eff_rank_h_{tag}_u{n}"] = float("nan")
                continue
            Hn = H[:n]
            out[f"eff_rank_h_{tag}_c{n}"] = float(
                effective_rank(Hn - Hn.mean(dim=0, keepdim=True)))
            out[f"eff_rank_h_{tag}_u{n}"] = float(effective_rank(Hn))
        return out

    loader_cache: Dict[tuple, object] = {}

    def eval_loader(cfg: dict):
        key = (cfg.get("data_source"), cfg.get("dataset"), cfg.get("image_size"),
               n_eval if n_eval is not None else cfg.get("eval_subset"))
        if key in loader_cache:
            return loader_cache[key]
        a = A.build_parser().parse_args([])
        # build_eval_data reads a few attributes that the parser does NOT define:
        # apply_smoke sets them at run time.  Set them all explicitly here rather
        # than depend on which ones happen to exist, and use apply_smoke's own
        # values so a synthetic split matches the one the probe actually saw.
        a.synthetic = (cfg.get("data_source") == "synthetic")
        a.synthetic_train = getattr(a, "synthetic_train", 192)
        a.synthetic_classes = getattr(a, "synthetic_classes", 4)
        a.dataset = cfg.get("dataset", getattr(a, "dataset", "cifar10"))
        a.image_size = cfg.get("image_size", getattr(a, "image_size", 32))
        a.eval_subset = n_eval if n_eval is not None else cfg.get("eval_subset")
        a.data_root = data_root
        _tr, te, _chance = A.build_eval_data(a)
        dl = DataLoader(te, batch_size=256, shuffle=False, num_workers=0)
        kind = "synthetique" if a.synthetic else a.dataset
        print(f"[reanalyse] split de test ({kind}) : {len(te)} images")
        loader_cache[key] = dl
        return dl

    @torch.no_grad()
    def features(model, dl):
        model.eval()
        H, Y = [], []
        for batch in dl:
            if isinstance(batch, dict):
                x = batch.get("x", batch.get("v1"))
                y = batch.get("label", batch.get("y"))
            else:
                x, y = batch[0], batch[1]
            H.append(model.backbone(x.to(device)).float().cpu())
            Y.append(y if torch.is_tensor(y) else torch.as_tensor(y))
        return torch.cat(H), torch.cat(Y)

    def fisher_ratio(H, y) -> float:
        """Trace-form between/within scatter on L2-NORMALISED features.

        The kNN probe normalises, so this does too: otherwise the ratio would
        partly measure a change of scale that the probe cannot see.
        """
        Hn = H / H.norm(dim=1, keepdim=True).clamp_min(1e-8)
        mu = Hn.mean(0, keepdim=True)
        between = within = 0.0
        for c in torch.unique(y):
            m = (y == c)
            nc = int(m.sum())
            if nc < 2:
                continue
            Hc = Hn[m]
            muc = Hc.mean(0, keepdim=True)
            between += nc * float(((muc - mu) ** 2).sum())
            within += float(((Hc - muc) ** 2).sum())
        if within <= 0:
            return float("nan")
        return between / within

    out: Dict[str, dict] = {}
    for arm in ARMS:
        rows = []
        for ck_path in sorted(glob.glob(os.path.join(results_dir, f"A2_{arm}_seed*.ckpt.pt"))):
            name = os.path.basename(ck_path)
            cfg_path = ck_path[: -len(".ckpt.pt")] + ".config.json"
            if not os.path.exists(cfg_path):
                print(f"[reanalyse] {name}: REFUS, {os.path.basename(cfg_path)} absent. "
                      f"Reconstruire le modele de memoire donnerait des chiffres faux "
                      f"sans le dire.")
                continue
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f).get("config", {})
            try:
                blob = torch.load(ck_path, map_location="cpu", weights_only=False)
            except Exception as exc:                                    # noqa: BLE001
                print(f"[reanalyse] {name}: illisible ({type(exc).__name__}: {exc})")
                continue
            state = blob.get("model", blob) if isinstance(blob, dict) else blob
            backbone, feat_dim = make_encoder(arch=cfg.get("arch", "resnet18"),
                                              dataset=encoder_variant)
            model = A.EncoderProjector(backbone, feat_dim,
                                       int(cfg.get("hidden", 512)),
                                       int(cfg.get("proj_dim", 128))).to(device)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                # Never average a half-loaded model into a number.
                print(f"[reanalyse] {name}: REFUS, state_dict incompatible "
                      f"({len(missing)} manquants, {len(unexpected)} en trop)")
                continue
            H, y = features(model, eval_loader(cfg))
            Hc = H - H.mean(dim=0, keepdim=True)
            try:
                seed = int(name.split("seed")[-1].split(".")[0])
            except (ValueError, IndexError):
                print(f"[reanalyse] {name}: REFUS, graine illisible dans le nom")
                continue
            Haug = backbone_of(model, augmented_views(cfg, seed))
            row = {
                "file": name, "seed": seed,
                "step": int(blob.get("step", -1)) if isinstance(blob, dict) else -1,
                "n_images": int(H.shape[0]),
                "n_augmented": int(Haug.shape[0]),
                "eff_rank_h_clean_uncentred": float(effective_rank(H)),
                "eff_rank_h_clean_centred": float(effective_rank(Hc)),
                "fisher_ratio_clean": fisher_ratio(H, y),
                "mean_h_norm": float(H.norm(dim=1).mean()),
            }
            row.update(rank_grid(Haug, "aug"))
            row.update(rank_grid(H, "clean"))
            rows.append(row)
            grid_txt = "  ".join(
                f"{d}/N={n} c={row[f'eff_rank_h_{d}_c{n}']:.1f}"
                for d in ("aug", "clean") for n in grid_ns)
            print(f"[reanalyse] {name}: {grid_txt}  Fisher={row['fisher_ratio_clean']:.4f}")
        if rows:
            agg = {
                "per_seed": rows,
                "eff_rank_h_clean_uncentred": mean_sd([r["eff_rank_h_clean_uncentred"] for r in rows]),
                "eff_rank_h_clean_centred": mean_sd([r["eff_rank_h_clean_centred"] for r in rows]),
                "fisher_ratio_clean": mean_sd([r["fisher_ratio_clean"] for r in rows]),
            }
            for k in rows[0]:
                if k.startswith("eff_rank_h_aug_") or k.startswith("eff_rank_h_clean_c") \
                   or k.startswith("eff_rank_h_clean_u"):
                    if k in ("eff_rank_h_clean_uncentred", "eff_rank_h_clean_centred"):
                        continue
                    agg[k] = mean_sd([r[k] for r in rows])
            out[arm] = agg
    return out


def contrasts(ck: dict) -> dict:
    """gram vs gram_vicreg on every measured quantity, in sigma."""
    out: Dict[str, dict] = {}
    if "gram" not in ck or "gram_vicreg" not in ck:
        return out
    keys = [k for k in ck["gram"]["per_seed"][0]
            if k.startswith("eff_rank_h_") or k == "fisher_ratio_clean"]
    for key in keys:
        a = [r[key] for r in ck["gram"]["per_seed"]]
        b = [r[key] for r in ck["gram_vicreg"]["per_seed"]]
        ma, _ = mean_sd(a)
        mb, _ = mean_sd(b)
        se = se_of_difference(a, b)
        out[key] = {
            "gram": ma, "gram_vicreg": mb, "difference": mb - ma,
            "se_of_difference": se,
            "sigma": ((mb - ma) / se) if se == se and se > 0 else float("nan"),
        }
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=os.path.join(HERE, "results", "A"))
    p.add_argument("--out", default=None, help="defaut : <results>/A2_espaces.json")
    p.add_argument("--checkpoints", action="store_true",
                   help="charger les checkpoints (rang centre + ratio de Fisher)")
    p.add_argument("--data-root", default=os.path.join(HERE, "data"))
    p.add_argument("--encoder-variant", default="cifar",
                   help="seul reglage non lisible dans les .config.json des runs")
    p.add_argument("--n-eval", type=int, default=None,
                   help="borner le split de test, pour un essai rapide")
    p.add_argument("--grid-n", default="512,10000",
                   help="les N de la grille 2 x 2, separes par des virgules")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    if not os.path.isdir(args.results):
        print(f"introuvable : {args.results}", file=sys.stderr)
        return 2

    res: Dict[str, object] = {
        "what_this_is": "exploratoire. Aucune these testee, aucun seuil. "
                        "Voir JOURNAL.md entree A-002.",
        "results_dir": os.path.abspath(args.results),
    }

    print("=" * 74)
    print("1. TRAJECTOIRE DE eff_rank_h, depuis les CSV")
    print("=" * 74)
    tr = trajectory(args.results)
    res["trajectory"] = tr
    if not tr:
        print("  aucun CSV avec une colonne eff_rank_h exploitable.")
    for arm, blk in tr.items():
        i_m, i_s = blk["eff_rank_h_at_init_mean_sd"]
        f_m, f_s = blk["eff_rank_h_final_mean_sd"]
        q_m, q_s = blk["last_quarter_change_mean_sd"]
        print(f"  {arm:22s} init {i_m:7.2f} +/- {i_s:5.2f}   "
              f"final {f_m:7.2f} +/- {f_s:5.2f}   "
              f"dernier quart {q_m:+7.2f} +/- {q_s:5.2f}")
    print("\n  Lecture : si le dernier quart bouge encore autant que le reste,"
          "\n  l'explication 'sous-entraine' est vivante et rien d'autre ne se conclut.")

    if args.checkpoints:
        print()
        print("=" * 74)
        print("GRILLE 2 x 2 (distribution x N) ET ALLOCATION PAR CLASSE")
        print("=" * 74)
        grid_ns = sorted(set(int(x) for x in str(args.grid_n).split(",") if x.strip()))
        ck = checkpoint_pass(args.results, args.data_root, args.encoder_variant,
                             args.n_eval, args.device, grid_ns=grid_ns)
        res["checkpoints"] = ck
        res["contrasts_gram_vs_gram_vicreg"] = contrasts(ck)
        print()
        for arm, blk in ck.items():
            u_m, u_s = blk["eff_rank_h_clean_uncentred"]
            c_m, c_s = blk["eff_rank_h_clean_centred"]
            f_m, f_s = blk["fisher_ratio_clean"]
            print(f"  {arm:22s} non centre {u_m:7.2f} +/- {u_s:5.2f}   "
                  f"centre {c_m:7.2f} +/- {c_s:5.2f}   "
                  f"Fisher {f_m:.4f} +/- {f_s:.4f}")
        print()
        for key, c in res["contrasts_gram_vs_gram_vicreg"].items():
            print(f"  gram -> gram_vicreg  {key:30s} {c['difference']:+10.4f}  "
                  f"({c['sigma']:+.2f} sigma)")
        print()
        print("  GRILLE, rang effectif CENTRE du backbone (moyenne sur les graines)")
        head = "  {:22s}".format("bras") + "".join(
            f"{d} N={n:<6d}".rjust(16) for d in ("aug", "clean") for n in grid_ns)
        print(head)
        for arm, blk in ck.items():
            line = f"  {arm:22s}"
            for d in ("aug", "clean"):
                for n in grid_ns:
                    k = f"eff_rank_h_{d}_c{n}"
                    if k in blk:
                        m, sd = blk[k]
                        line += f"{m:9.1f} +/-{sd:4.1f}".rjust(16)
                    else:
                        line += "n/a".rjust(16)
            print(line)
        print()
        print("  CONTRASTE gram -> gram_vicreg sur la grille centree")
        for d in ("aug", "clean"):
            for n in grid_ns:
                k = f"eff_rank_h_{d}_c{n}"
                c = res["contrasts_gram_vs_gram_vicreg"].get(k)
                if c:
                    print(f"    {d:5s} N={n:<6d} {c['gram']:7.1f} -> {c['gram_vicreg']:7.1f}  "
                          f"{c['difference']:+8.1f}  ({c['sigma']:+.2f} sigma)")
        print()
        print("  Lecture, dans cet ordre :")
        print("  1. Comparer les deux colonnes N d'une MEME distribution : c'est l'effet")
        print("     de N seul. RankMe annexe G dit qu'il faut 10000 pour 95 % du rang,")
        print("     donc un ecart important ici veut dire que N = 512 ne mesurait rien.")
        print("  2. Comparer aug et clean a N EGAL : c'est l'effet de la distribution")
        print("     seule, et c'est le seul chiffre citable.")
        print("  3. Si le contraste tient a N = 10000 sur les deux distributions, le fait")
        print("     est propre. S'il disparait, A-002 mesurait N, pas l'espace, et le")
        print("     JOURNAL doit le dire.")
        print("  Non centre disponible dans le JSON sous _u<N>, meme grille.")

    out_path = args.out or os.path.join(args.results, "A2_espaces.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"\n-> {out_path}")
    print("Rappel : verdict class 'exploratoire'. Rien ici ne confirme "
          "ni n'infirme une these.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
