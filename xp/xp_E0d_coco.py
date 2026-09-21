#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xp_E0d_coco.py -- E0 with a hole the predictor cannot see across, a regime
                  gate that can be populated, and a null-predictor control.
                  Same question as E0b / E0c, third design.

WHAT E0c ESTABLISHED (JOURNAL E-003, 2026-09-16)
-----------------------------------------------
The predictor was no longer myopic (rf 23 px, loss still falling at the last
step) and the ruler `acf_at_rf` reads coherent structure when there is some
(synthetic calibration, results/E0c/calib_ruler_synthetic.txt).  Yet the regime
was empty again: the residual got WHITER (ell 1.6 px, from 3.2 in E0b), and
acf_at_rf sat at zero on things, on stuff and on boundaries alike.  Diagnosis:

  1. With hidden blocks of 8 px and a receptive field of 23 px, every hidden
     pixel is INTERPOLATED from visible context.  The model never has to guess
     something it cannot see, so its errors are never coherent across a region;
     what remains is interpolation grain, identical on a face and on grass.
     Tuning (depth, mask ratio) moved ell DOWN.  The residual production, not
     the settings, is what must change.
  2. The regime precondition read RAW ell_acf (first 1/e crossing).  On the
     calibration, a field with a coherent 32 px envelope still has 0% of cells
     with ell > rf: the grain pins the first crossing whatever lies underneath.
     That gate could not be satisfied by any residual containing grain.

WHAT E0d CHANGES, AND WHY EACH CHANGE
-------------------------------------
  * HOLE > RECEPTIVE FIELD.  Hidden units are super-blocks of `hide` = 32 px
    (4 x 4 cells), 1 in 4 hidden (25%), complementary groups, each super-block
    hidden exactly once in the diagnostic sweep.  rf stays 23 px, so the core
    of a hidden block (farther than 11 px from any visible pixel, about 10 x 10
    px) is BLIND: the model must extrapolate.  A coherent misprediction of a
    hidden object is now possible, and it is the only mechanism that can
    correlate the residual beyond rf.  The script refuses to run if hide <= rf.
  * REGIME GATE ON acf_at_rf.  Fraction of labelled cells with acf_at_rf above
    ACF_FLOOR = 0.05 must reach REGIME_MIN_FRAC = 0.10.  Derivation from the
    calibration: grain-only field q95 = +0.022 and E0c itself had 2% above
    0.05; a grain field carrying a 16 px coherent envelope has about 13% above
    0.05.  ell > rf is still reported, for continuity, with no gate role.
  * NULL-PREDICTOR CONTROL.  In the blind core the error is "image minus a
    prior", whose range is the range of the image itself (long on sky, short on
    gravel).  To separate "what the model failed to see" from "how smooth the
    image is", the same acf_at_rf is read on the squared deviation of the image
    from its own mean colour (the residual of a predictor that knows nothing).
    GO requires the trained residual to beat BOTH controls (naive short-lag
    statistic, null-predictor acf_at_rf) by GO_MARGIN_OVER_NAIVE.
  * SIGN, pre-registered.  T4 predicts thing > stuff (AUC > 0.5).  With a
    populated regime and AUC <= 0.40 the verdict is "infirmee, signe inverse",
    reported apart: it would mean the criterion tracks image smoothness.
  * HOLE DEPTH per cell (Chebyshev distance, in cells, to the nearest visible
    cell under the variant that hides it) is logged; exploratory AUCs on the
    blind core (depth >= 2) and on the ring (depth == 1) are reported, not
    gated.
  * ell_acf_smooth is dropped (E-003, point 7).  Residual magnitude r is a
    DECLARED secondary this time.
  * FUSE unchanged: abort at 20% of the budget if the loss gained < 1% since 10%.
  * SEEDS: seed 0 is a pilot.  Seeds 1 and 2 are run only if seed 0's regime is
    populated; on an empty regime they would decide nothing (E-002, E-003).

Everything else is imported: data and labels from xp_E0b_coco, the acf ruler
from xp_E0c_coco, model, masks, residuals, ell estimators and thresholds from
xp_E_router.

USAGE
-----
  python xp/xp_E0d_coco.py --smoke --cpu
  python xp/xp_E0d_coco.py --seed 0 --data-root /content/ubergang_local/data/cocostuff

Cost: same model and budget as E0c (36 min A100 per seed measured), the 4
diagnostic variants instead of 3 add a few minutes.  Budget 45 min per seed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from lib.harness import Run, set_seed  # noqa: E402
import xp_E_router as ER               # noqa: E402
import xp_E0b_coco as EB               # noqa: E402  (data, labels, AUC helper)
import xp_E0c_coco as EC               # noqa: E402  (radial_acf_at)

CELL_THING, CELL_STUFF, CELL_EXCLUDED = EB.CELL_THING, EB.CELL_STUFF, EB.CELL_EXCLUDED
ACF_FLOOR = 0.05              # acf_at_rf above this = "coherent beyond rf" (calibration q95 of grain = 0.022)
REGIME_MIN_FRAC = 0.10        # fraction of labelled cells above ACF_FLOOR needed for a populated regime
INVERTED_AUC = 0.40           # populated regime and AUC <= this = sign inverted, reported apart
PLATEAU_MIN_GAIN = 0.01       # loss must improve >= 1% between 10% and 20% of budget

PREREG_E0D = {
    "statement": "P0d: on COCO images at 128 px, with hidden blocks (32 px) LARGER than the "
                 "predictor's receptive field (23 px), so that the model must extrapolate in "
                 "the core of each hole, the normalised residual autocorrelation read at the "
                 "receptive field (acf_at_rf) is higher on THING cells than on STUFF cells, "
                 "and separates them better than (i) the naive short-lag structuredness "
                 "statistic and (ii) the same acf_at_rf read on a null predictor (the image's "
                 "own mean colour)",
    "primary_statistic": "auc_acf_at_rf",
    "controls": ["auc_naive_acf_energy", "auc_null_acf_at_rf"],
    "secondary_statistics": ["auc_ell_acf", "auc_ell_freq", "auc_residual_magnitude"],
    "exploratory": ["auc_acf_at_rf_core", "auc_acf_at_rf_ring", "auc_acf_at_rf_stuck"],
    "decision_rule": f"GO iff mean AUC >= {ER.GO_AUC} over >= 3 seeds AND margin over EACH "
                     f"control >= {ER.GO_MARGIN_OVER_NAIVE} AND mean - std >= {ER.NOGO_AUC}; "
                     f"NO-GO iff mean < {ER.NOGO_AUC}; 'signe inverse' iff mean <= {INVERTED_AUC} "
                     f"with a populated regime (reported apart from a plain separation failure)",
    "precondition": f"fraction of labelled cells with acf_at_rf > {ACF_FLOOR} must reach "
                    f"{REGIME_MIN_FRAC}; otherwise 'regime empty' (NO-GO by construction), "
                    f"reported apart. Thresholds derived from results/E0c/calib_ruler_synthetic.txt "
                    f"(grain only: q95 = 0.022; grain + 16 px envelope: ~13% above 0.05) and "
                    f"from E0c itself (2% above 0.05, empty)",
    "residual_production": "super-blocks of 32 px, 1 in 4 hidden (25%), complementary groups; "
                           "hide > rf is enforced by the script; loss on hidden cells only",
    "seeds": "seed 0 first (pilot); seeds 1 and 2 only if seed 0's regime is populated",
    "fuse": f"abort at 20% of the budget if the loss gained < {PLATEAU_MIN_GAIN:.0%} since 10%",
    "registered": "2026-09-16, before the first run of this script, after E-003",
}


# =============================================================================
# 1.  Super-block masks: hidden = the super-blocks whose group EQUALS the variant
# =============================================================================

def _hidden_masks_super(batch_idx: np.ndarray, cfg: "ER.Cfg", grid: int, hide_cells: int,
                        device: torch.device, variant: np.ndarray
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """hidden_region [B,G,G] on CPU (bool, cell granularity), mask_px [B,1,H,W] on device.

    Super-blocks are ``hide_cells`` x ``hide_cells`` cells.  Groups are balanced
    (ER.deterministic_mask_groups), so with S*S super-blocks and V variants each
    variant hides S*S/V of them, and over the V variants each super-block is
    hidden exactly once.
    """
    S = grid // hide_cells
    groups, _own = ER.deterministic_mask_groups(
        batch_idx, S * S, cfg.mask_variants, cfg.mask_seed)
    hidden_sb = torch.from_numpy(groups) == torch.from_numpy(np.asarray(variant)).view(-1, 1)
    hidden_sb = hidden_sb.view(-1, S, S)
    hidden_region = hidden_sb.repeat_interleave(hide_cells, dim=1) \
                             .repeat_interleave(hide_cells, dim=2)
    mask_px = hidden_region.float().unsqueeze(1).repeat_interleave(
        cfg.patch, dim=2).repeat_interleave(cfg.patch, dim=3)
    return hidden_region, mask_px.to(device)


def _hole_depth(hidden_region: torch.Tensor) -> torch.Tensor:
    """[B,G,G] bool -> [B,G,G] long: Chebyshev distance in cells from each hidden
    cell to the nearest visible cell (0 on visible cells)."""
    B, G, _ = hidden_region.shape
    covered = (~hidden_region).float().unsqueeze(1)
    depth = torch.zeros_like(covered)
    for k in range(1, G + 1):
        grown = F.max_pool2d(covered, kernel_size=3, stride=1, padding=1)
        newly = (grown > 0) & (covered == 0)
        depth[newly] = float(k)
        covered = grown
        if bool((covered > 0).all()):
            break
    return depth.squeeze(1).long()


@torch.no_grad()
def e0d_diagnostic_sweep(model: "ER.HierarchicalMaskedPredictor", pool: Dataset,
                         cfg: "ER.Cfg", hide_cells: int, device: torch.device,
                         indices: np.ndarray) -> Dict[str, np.ndarray]:
    """Every cell measured while HIDDEN, exactly once (one variant hides its super-block).

    Also returns the null-predictor field (squared deviation of the image from
    its own mean colour) and the hole depth of every cell under its hiding variant.
    """
    model.eval()
    G = cfg.size // cfg.patch
    idx_all = np.asarray(indices, dtype=np.int64)
    n = int(len(idx_all))
    res_region = np.zeros((n, G, G), dtype=np.float64)
    res_pix = np.zeros((n, cfg.size, cfg.size), dtype=np.float64)
    null_pix = np.zeros((n, cfg.size, cfg.size), dtype=np.float64)
    cnt_region = np.zeros((n, G, G), dtype=np.float64)
    depth_region = np.zeros((n, G, G), dtype=np.int64)
    loader = DataLoader(Subset(pool, idx_all.tolist()), batch_size=min(cfg.bs, 128),
                        shuffle=False, num_workers=cfg.workers, drop_last=False)
    pos = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        bidx = batch["idx"].numpy()
        b = x.shape[0]
        acc_r = torch.zeros(b, G, G, device=device)
        acc_c = torch.zeros(b, G, G, device=device)
        acc_d = torch.zeros(b, G, G, device=device)
        acc_p = torch.zeros(b, cfg.size, cfg.size, device=device)
        acc_pc = torch.zeros(b, cfg.size, cfg.size, device=device)
        for v in range(cfg.mask_variants):
            hidden_region, mask_px = _hidden_masks_super(
                bidx, cfg, G, hide_cells, device, variant=np.full(b, v, dtype=np.int64))
            xin = model.apply_mask(x, mask_px)
            preds, _ = model(xin)
            res_cells, pix0 = ER._targets_and_residuals(x, preds, cfg.patch)
            hr = hidden_region.to(device).float()
            acc_r += res_cells[0] * hr
            acc_c += hr
            acc_d += _hole_depth(hidden_region).to(device).float() * hr
            hp = mask_px.squeeze(1)
            acc_p += pix0 * hp
            acc_pc += hp
        null = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2).mean(dim=1)
        res_region[pos:pos + b] = (acc_r / acc_c.clamp_min(1)).cpu().numpy()
        res_pix[pos:pos + b] = (acc_p / acc_pc.clamp_min(1)).cpu().numpy()
        null_pix[pos:pos + b] = null.cpu().numpy()
        cnt_region[pos:pos + b] = acc_c.cpu().numpy()
        depth_region[pos:pos + b] = acc_d.round().long().cpu().numpy()
        pos += b
    model.train()
    return {"res_region": res_region, "res_pix": res_pix, "null_pix": null_pix,
            "cnt_region": cnt_region, "hole_depth": depth_region, "idx": idx_all}


# =============================================================================
# 2.  Verdict: precondition on acf_at_rf, then the E0 rule against BOTH controls
# =============================================================================

def e0d_verdict(summaries: List[dict]) -> dict:
    prim = ER.mean_std([s["auc_acf_at_rf"] for s in summaries])
    naive = ER.mean_std([s["auc_naive_acf_energy"] for s in summaries])
    null = ER.mean_std([s["auc_null_acf_at_rf"] for s in summaries])
    regime = ER.mean_std([s["frac_acf_above_floor"] for s in summaries])
    m, sd = prim
    margin_naive = m - naive[0]
    margin_null = m - null[0]
    margin = min(margin_naive, margin_null)
    regime_empty = (not np.isfinite(regime[0])) or regime[0] < REGIME_MIN_FRAC
    if regime_empty:
        verdict, decision, reason = "non concluante", "NO-GO", "regime_empty"
    elif not np.isfinite(m):
        verdict, decision, reason = "non concluante", "NO-GO", "nan"
    elif m >= ER.GO_AUC and margin >= ER.GO_MARGIN_OVER_NAIVE and (m - sd) >= ER.NOGO_AUC:
        verdict, decision, reason = "confirmee", "GO", "primary_passes"
    elif m <= INVERTED_AUC:
        verdict, decision, reason = "infirmee", "NO-GO", "primary_inverted"
    elif m < ER.NOGO_AUC:
        verdict, decision, reason = "infirmee", "NO-GO", "primary_below_nogo"
    elif margin < ER.GO_MARGIN_OVER_NAIVE:
        verdict, decision, reason = "non concluante", "NO-GO", "no_margin_over_controls"
    else:
        verdict, decision, reason = "non concluante", "NO-GO", "between_thresholds"
    return {
        "prediction": PREREG_E0D["statement"],
        "verdict": verdict, "decision": decision, "reason": reason,
        "primary_statistic": "auc_acf_at_rf",
        "auc_mean": m, "auc_std": sd,
        "auc_naive_mean": naive[0], "auc_naive_std": naive[1],
        "auc_null_mean": null[0], "auc_null_std": null[1],
        "margin_over_naive": margin_naive, "margin_over_null": margin_null,
        "margin_over_controls": margin,
        "regime_frac_acf_above_floor": regime[0], "regime_min_frac": REGIME_MIN_FRAC,
        "acf_floor": ACF_FLOOR, "regime_empty": bool(regime_empty),
        "n_seeds": len(summaries),
        "secondary": {k: ER.mean_std([s[k] for s in summaries])
                      for k in ("auc_ell_acf", "auc_ell_freq", "auc_residual_magnitude",
                                "auc_acf_at_rf_core", "auc_acf_at_rf_ring",
                                "frac_ell_above_rf", "censored_frac")},
    }


# =============================================================================
# 3.  The run
# =============================================================================

def run_e0d(seed: int, cfg: "ER.Cfg", args, outdir: str, device: torch.device,
            resume: bool = True) -> dict:
    set_seed(seed)
    name = f"E0d_coco_seed{seed}"
    hide_cells = int(args.hide) // cfg.patch
    run = Run(name=name, outdir=outdir,
              config={"stage": "E0d", "seed": seed, "n_images": args.n_images,
                      "purity": args.purity, "thing_max_label": args.thing_max_label,
                      "hide_px": int(args.hide), "mask_ratio": 1.0 / cfg.mask_variants,
                      "smoke": bool(args.smoke), **cfg.to_dict()},
              resume=resume, higher_is_better=False)

    if args.smoke:
        pool: Dataset = EB.SyntheticThingsStuff(args.n_images, cfg.size, cfg.patch,
                                                args.purity, seed=0)
    else:
        pool = EB.CocoStuffCells(os.path.join(args.data_root, "val2017"),
                                 os.path.join(args.data_root, "stuffthingmaps", "val2017"),
                                 cfg.size, cfg.patch, args.n_images, seed=0,
                                 purity=args.purity, thing_max_label=args.thing_max_label)
    G = cfg.size // cfg.patch
    if G % hide_cells != 0 or hide_cells < 1:
        raise RuntimeError(f"[E0d] --hide {args.hide} px must be a multiple of patch "
                           f"({cfg.patch}) dividing the grid ({G} cells).")
    S = G // hide_cells
    if (S * S) % cfg.mask_variants != 0:
        raise RuntimeError(f"[E0d] {S * S} super-blocks are not divisible by "
                           f"--mask-groups {cfg.mask_variants}; groups would be unbalanced.")

    diag_idx = np.sort(np.random.RandomState(seed).permutation(len(pool))[:cfg.e0_diag_n])
    diag_idx = diag_idx.astype(np.int64)
    cells = pool.cell_labels_for(diag_idx)
    n_thing = int((cells == CELL_THING).sum())
    n_stuff = int((cells == CELL_STUFF).sum())
    n_excl = int((cells == CELL_EXCLUDED).sum())
    tot = cells.size
    hist = pool.label_histogram()
    print(f"[E0d] label histogram (id, pixels), boundary <= {args.thing_max_label}:")
    print(f"      thing side: {hist['thing_side_top_ids']}")
    print(f"      stuff side: {hist['stuff_side_top_ids']}")
    print(f"[E0d] diagnostic cells: {tot} | {n_thing} thing ({100 * n_thing / tot:.1f}%) | "
          f"{n_stuff} stuff ({100 * n_stuff / tot:.1f}%) | {n_excl} excluded "
          f"({100 * n_excl / tot:.1f}%)")
    labeled = n_thing + n_stuff
    if labeled == 0 or min(n_thing, n_stuff) < 0.02 * labeled:
        raise RuntimeError("[E0d] one class holds under 2% of labelled cells; fix "
                           "--thing-max-label or --purity BEFORE spending GPU time.")

    model = ER._make_model(cfg, levels=1, device=device)
    rf0 = float(model.receptive_field_px[0])
    window = cfg.win_mult * cfg.patch
    censor = window / 2.0
    blind = max(0.0, float(args.hide) - 2.0 * ((rf0 - 1.0) / 2.0))
    print(f"[E0d] receptive field = {rf0:.1f} px | hidden block = {args.hide} px "
          f"(blind core about {blind:.0f} px) | window = {window} px (ell censored at "
          f"{censor:.1f} px) | hidden fraction = 1/{cfg.mask_variants} of {S * S} super-blocks")
    if float(args.hide) <= rf0:
        raise RuntimeError(f"[E0d] hidden block {args.hide} px must EXCEED the receptive "
                           f"field ({rf0:.0f} px): otherwise every hidden pixel is "
                           f"interpolated and the regime is empty by construction (E-003).")
    if rf0 > ER.ELL_HEADROOM_RATIO * censor:
        raise RuntimeError(f"[E0d] rf {rf0:.0f} px exceeds {ER.ELL_HEADROOM_RATIO:.2f} x "
                           f"censor ({censor:.0f} px). Lower --depth or raise --win-mult.")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[E0d] predictor: depth {cfg.depth}, width {cfg.width}, {n_params / 1e6:.2f} M params")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_step, snap_prev = 0, None
    ck = run.load_ckpt() if resume else None
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = int(ck["step"])
        if ck.get("snap_prev_path") and os.path.exists(ck["snap_prev_path"]):
            snap_prev = dict(np.load(ck["snap_prev_path"]))
            old_idx = np.asarray(snap_prev.get("idx", np.zeros(0)), dtype=np.int64)
            if old_idx.shape != diag_idx.shape or not np.array_equal(old_idx, diag_idx):
                raise RuntimeError(f"[E0d] stale snapshot for {name}: delete {name}.* "
                                   f"and run this seed fresh.")
        print(f"[E0d] resumed at step {start_step}")

    snap_prev_path = os.path.join(outdir, f"{name}.snap_prev.npz")
    snap_step = max(1, cfg.e0_steps - cfg.e0_lam_gap)

    N = int(cfg.e0_steps)
    marks = {int(N * 0.05): "l05", int(N * 0.10): "l10", int(N * 0.20): "l20"}
    tail_len = max(20, N // 100)
    recent: List[float] = []
    fuse: Dict[str, float] = {}

    loader = ER._pool_loader(pool, cfg, seed)
    it = iter(loader)
    model.train()
    t0 = time.time()
    step = start_step
    while step < cfg.e0_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        x = batch["x"].to(device, non_blocking=True)
        bidx = batch["idx"].numpy()
        variant = np.random.randint(0, cfg.mask_variants, size=x.shape[0])
        hidden_region, mask_px = _hidden_masks_super(bidx, cfg, G, hide_cells, device,
                                                     variant=variant)
        hidden_region = hidden_region.to(device)
        for pg in opt.param_groups:
            pg["lr"] = ER._lr_at(step, cfg, cfg.e0_steps)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            xin = model.apply_mask(x, mask_px)
            preds, _ = model(xin)
            res_cells, _pix = ER._targets_and_residuals(x, preds, cfg.patch)
            w = hidden_region.float()
            loss = (res_cells[0] * w).sum() / w.sum().clamp_min(1.0)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        step += 1

        lv = float(loss.detach())
        recent.append(lv)
        if len(recent) > tail_len:
            recent.pop(0)
        if step in marks and recent:
            fuse[marks[step]] = float(np.mean(recent))
            print(f"[E0d] loss at {100 * step / N:.0f}% of budget: {fuse[marks[step]]:.5f}")
            if marks[step] == "l20" and "l10" in fuse and not args.smoke:
                gain = (fuse["l10"] - fuse["l20"]) / max(fuse["l10"], 1e-12)
                run.log(step, plateau_gain_10_to_20=gain)
                if gain < PLATEAU_MIN_GAIN and not args.no_abort_on_plateau:
                    raise SystemExit(
                        f"[E0d] PLATEAU: loss gained {100 * gain:.2f}% between 10% and "
                        f"20% of the budget (< {100 * PLATEAU_MIN_GAIN:.0f}%). The predictor "
                        f"has stopped learning; aborting to save GPU time; "
                        f"--no-abort-on-plateau overrides.")

        if step % cfg.log_every == 0:
            run.log(step, loss=lv, lr=ER._lr_at(step, cfg, cfg.e0_steps), seed=seed,
                    imgs_per_s=step * cfg.bs / max(1e-6, time.time() - t0))
        if step == snap_step and snap_prev is None:
            snap_prev = e0d_diagnostic_sweep(model, pool, cfg, hide_cells, device, diag_idx)
            np.savez_compressed(snap_prev_path, **snap_prev)
            print(f"[E0d] first residual snapshot at step {step}")
        if step % cfg.ckpt_every == 0 or step == cfg.e0_steps:
            run.save_ckpt(step, model=model.state_dict(), opt=opt.state_dict(),
                          snap_prev_path=snap_prev_path if snap_prev is not None else None,
                          best_metric=lv)

    if snap_prev is None:
        snap_prev = e0d_diagnostic_sweep(model, pool, cfg, hide_cells, device, diag_idx)
        np.savez_compressed(snap_prev_path, **snap_prev)
    snap_now = e0d_diagnostic_sweep(model, pool, cfg, hide_cells, device, diag_idx)
    cov = snap_now["cnt_region"]
    if not np.allclose(cov, 1.0):
        raise RuntimeError(f"[E0d] sweep coverage is not 1 everywhere (min {cov.min()}, "
                           f"max {cov.max()}): some cell was hidden 0 or 2+ times.")

    r_prev = snap_prev["res_region"].astype(np.float64)
    r_now = snap_now["res_region"].astype(np.float64)
    n = min(r_prev.shape[0], r_now.shape[0])
    r_prev, r_now, cells = r_prev[:n], r_now[:n], cells[:n]
    lam_raw = (r_prev - r_now) / (r_prev + cfg.eps_r)
    lam_drift = lam_raw - float(np.nanmean(lam_raw))

    res_pix = torch.from_numpy(snap_now["res_pix"][:n]).float().to(device)
    null_pix = torch.from_numpy(snap_now["null_pix"][:n]).float().to(device)
    valid = torch.ones_like(res_pix)
    chunks, acf_rf, acf_null = [], [], []
    for a in range(0, n, 64):
        blk = res_pix[a:a + 64]
        chunks.append(ER.field_statistics(blk, valid[a:a + 64], cfg.patch,
                                          chunk=cfg.ell_chunk, win_mult=cfg.win_mult))
        acf_rf.append(EC.radial_acf_at(blk, cfg.patch, cfg.win_mult, radii=[rf0],
                                       chunk=cfg.ell_chunk)[float(rf0)])
        acf_null.append(EC.radial_acf_at(null_pix[a:a + 64], cfg.patch, cfg.win_mult,
                                         radii=[rf0], chunk=cfg.ell_chunk)[float(rf0)])
    stats = {k: torch.cat([c[k] for c in chunks], 0).cpu().numpy() for k in chunks[0]}
    acf_at_rf = torch.cat(acf_rf, 0).cpu().numpy()
    null_acf_at_rf = torch.cat(acf_null, 0).cpu().numpy()
    hole_depth = snap_now["hole_depth"][:n]

    table = {
        "image_id": np.repeat(np.arange(n)[:, None, None], G, 1).repeat(G, 2).reshape(-1),
        "pool_idx": np.repeat(diag_idx[:n][:, None, None], G, 1).repeat(G, 2).reshape(-1),
        "row": np.tile(np.arange(G)[:, None], (n, 1, G)).reshape(-1),
        "col": np.tile(np.arange(G)[None, :], (n, G, 1)).reshape(-1),
        "r": r_now.reshape(-1), "r_prev": r_prev.reshape(-1),
        "lam_raw": lam_raw.reshape(-1), "lam_drift": lam_drift.reshape(-1),
        "ell_acf": stats["ell_acf"].reshape(-1),
        "ell_freq": stats["ell_freq"].reshape(-1),
        "acf_energy": stats["acf_energy"].reshape(-1),
        "censored": stats["censored"].reshape(-1),
        "acf_at_rf": acf_at_rf.reshape(-1),
        "null_acf_at_rf": null_acf_at_rf.reshape(-1),
        "hole_depth": hole_depth.reshape(-1).astype(np.int64),
        "cell_label": cells.reshape(-1).astype(np.int64),
    }
    table_path = os.path.join(outdir, f"{name}.regions.npz")
    np.savez_compressed(table_path, **table)

    lab = table["cell_label"]
    labeled_mask = lab >= 0
    aucs = {
        "auc_acf_at_rf": EB._labeled_auc(table["acf_at_rf"], lab),
        "auc_naive_acf_energy": EB._labeled_auc(table["acf_energy"], lab),
        "auc_null_acf_at_rf": EB._labeled_auc(table["null_acf_at_rf"], lab),
        "auc_ell_acf": EB._labeled_auc(table["ell_acf"], lab),
        "auc_ell_freq": EB._labeled_auc(table["ell_freq"], lab),
        "auc_residual_magnitude": EB._labeled_auc(table["r"], lab),
        "auc_acf_at_rf_core": EB._labeled_auc(table["acf_at_rf"], lab, table["hole_depth"] >= 2),
        "auc_acf_at_rf_ring": EB._labeled_auc(table["acf_at_rf"], lab, table["hole_depth"] == 1),
    }
    stuck = table["lam_raw"] < np.nanquantile(table["lam_raw"], 0.5)
    aucs["auc_acf_at_rf_stuck"] = EB._labeled_auc(table["acf_at_rf"], lab, stuck)
    aucs["auc_naive_stuck"] = EB._labeled_auc(table["acf_energy"], lab, stuck)

    acf_lab = table["acf_at_rf"][labeled_mask]
    ell_lab = table["ell_acf"][labeled_mask]
    frac_acf = float(np.nanmean(acf_lab > ACF_FLOOR))
    summary = {
        "seed": seed, "receptive_field_px": rf0, "hide_px": int(args.hide),
        "blind_core_px": blind, "measurement_window_px": int(window),
        "mask_ratio": 1.0 / cfg.mask_variants,
        "censored_frac": float(np.mean(table["censored"][labeled_mask])),
        "n_cells_total": int(lab.size), "n_cells_labeled": int(labeled_mask.sum()),
        "frac_thing_cells": float((lab == CELL_THING).sum() / max(1, labeled_mask.sum())),
        "frac_excluded_cells": float((lab == CELL_EXCLUDED).sum() / lab.size),
        "frac_acf_above_floor": frac_acf,
        "frac_acf_above_floor_thing": float(np.nanmean(table["acf_at_rf"][lab == CELL_THING] > ACF_FLOOR)),
        "frac_acf_above_floor_stuff": float(np.nanmean(table["acf_at_rf"][lab == CELL_STUFF] > ACF_FLOOR)),
        "frac_null_above_floor": float(np.nanmean(table["null_acf_at_rf"][labeled_mask] > ACF_FLOOR)),
        "frac_ell_above_rf": float(np.nanmean(ell_lab > rf0)),
        "frac_core_cells": float(np.mean(table["hole_depth"][labeled_mask] >= 2)),
        "mean_ell_thing": float(np.nanmean(table["ell_acf"][lab == CELL_THING])),
        "mean_ell_stuff": float(np.nanmean(table["ell_acf"][lab == CELL_STUFF])),
        "mean_acf_at_rf_thing": float(np.nanmean(table["acf_at_rf"][lab == CELL_THING])),
        "mean_acf_at_rf_stuff": float(np.nanmean(table["acf_at_rf"][lab == CELL_STUFF])),
        "mean_null_acf_at_rf_thing": float(np.nanmean(table["null_acf_at_rf"][lab == CELL_THING])),
        "mean_null_acf_at_rf_stuff": float(np.nanmean(table["null_acf_at_rf"][lab == CELL_STUFF])),
        "mean_r_thing": float(np.nanmean(table["r"][lab == CELL_THING])),
        "mean_r_stuff": float(np.nanmean(table["r"][lab == CELL_STUFF])),
        "loss_marks": fuse, "n_diag_images": int(n),
        "regions_table": os.path.basename(table_path),
        **aucs,
    }
    run.log(cfg.e0_steps, **{k: v for k, v in summary.items() if isinstance(v, (int, float))})
    run.finish(summary)
    print(f"[E0d seed {seed}] PRIMARY AUC(acf_at_rf)={aucs['auc_acf_at_rf']:.3f}  "
          f"controls: naive={aucs['auc_naive_acf_energy']:.3f} "
          f"null={aucs['auc_null_acf_at_rf']:.3f}  |  secondaries: "
          f"ell_acf={aucs['auc_ell_acf']:.3f} ell_freq={aucs['auc_ell_freq']:.3f} "
          f"r={aucs['auc_residual_magnitude']:.3f}  |  core={aucs['auc_acf_at_rf_core']:.3f} "
          f"ring={aucs['auc_acf_at_rf_ring']:.3f}")
    print(f"[E0d seed {seed}] regime: {100 * frac_acf:.2f}% of labelled cells have "
          f"acf_at_rf > {ACF_FLOOR} ({100 * summary['frac_acf_above_floor_thing']:.2f}% thing, "
          f"{100 * summary['frac_acf_above_floor_stuff']:.2f}% stuff; null predictor "
          f"{100 * summary['frac_null_above_floor']:.2f}%); floor {100 * REGIME_MIN_FRAC:.0f}%. "
          f"For continuity: {100 * summary['frac_ell_above_rf']:.2f}% with ell > rf.")
    return summary


# =============================================================================
# 4.  CLI
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""download (Colab; a new VM has none of this):
  mkdir -p /content/ubergang_local/data/cocostuff && cd /content/ubergang_local/data/cocostuff
  wget -q http://images.cocodataset.org/zips/val2017.zip && unzip -q val2017.zip
  wget -q http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip
  mkdir -p stuffthingmaps && unzip -q stuffthingmaps_trainval2017.zip 'val2017/*' -d stuffthingmaps
""")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--outdir", default=os.path.join(_ROOT, "results", "E0d"))
    p.add_argument("--data-root", default=os.path.join(_ROOT, "data", "cocostuff"))
    p.add_argument("--n-images", type=int, default=5000)
    p.add_argument("--e0-steps", type=int, default=10000)
    p.add_argument("--e0-lam-gap", type=int, default=2500)
    p.add_argument("--e0-diag-n", type=int, default=1024)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--patch", type=int, default=8)
    p.add_argument("--hide", type=int, default=32,
                   help="hidden super-block side in px; must exceed the receptive field")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=10, help="rf0 = 1 + 2*(1+depth); 10 -> 23 px")
    p.add_argument("--win-mult", type=int, default=8, help="window = win_mult*patch; 8 -> 64 px")
    p.add_argument("--mask-groups", type=int, default=4,
                   help="hidden fraction = 1/groups of the super-blocks; 4 -> 25%%")
    p.add_argument("--purity", type=float, default=0.7)
    p.add_argument("--thing-max-label", type=int, default=90)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-abort-on-plateau", action="store_true")
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.smoke:
        # 48 px, 8 px cells, 16 px super-blocks (3 x 3 = 9), 1 in 3 hidden, rf 7 px < 16.
        args.size, args.patch, args.hide, args.width, args.depth = 48, 8, 16, 16, 2
        args.win_mult, args.bs, args.workers, args.mask_groups = 4, 16, 0, 3
        args.e0_steps, args.e0_lam_gap, args.e0_diag_n, args.n_images = 60, 20, 32, 64
        if args.seed is None:
            args.seed = 0
    seeds = [args.seed] if args.seed is not None else \
        [int(s) for s in str(args.seeds).split(",") if s != ""]
    device = torch.device("cpu") if args.cpu else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not args.cpu and not args.smoke:
        raise SystemExit("[E0d] no CUDA device found and --cpu was not passed. At 128 px "
                         "with depth 10 this run takes DAYS on CPU. Select a GPU runtime, "
                         "remount Drive, re-download COCO on the new VM, then relaunch.")
    os.makedirs(args.outdir, exist_ok=True)

    cfg = ER.Cfg(dataset="coco_stuff", size=args.size, patch=args.patch,
                 width=args.width, depth=args.depth, levels=1,
                 bs=args.bs, lr=args.lr, workers=args.workers,
                 e0_steps=args.e0_steps, e0_lam_gap=args.e0_lam_gap,
                 e0_diag_n=args.e0_diag_n, win_mult=args.win_mult,
                 mask_variants=args.mask_groups, amp=not args.cpu)

    print("=" * 78)
    print(f"E0d (T4, region labels, hole > receptive field)  |  device={device}  "
          f"seeds={seeds}  size={cfg.size} patch={cfg.patch} hide={args.hide}px "
          f"win={cfg.win_mult * cfg.patch}px  hidden=1/{cfg.mask_variants}")
    print(f"outdir = {args.outdir}")
    print("=" * 78)

    summaries = [run_e0d(s, cfg, args, args.outdir, device, resume=args.resume)
                 for s in seeds]
    verdict = e0d_verdict(summaries)
    payload = {"prereg": PREREG_E0D,
               "config": {**cfg.to_dict(), "n_images": args.n_images, "purity": args.purity,
                          "thing_max_label": args.thing_max_label, "hide_px": int(args.hide),
                          "mask_ratio": 1.0 / cfg.mask_variants, "smoke": bool(args.smoke)},
               "per_seed": summaries, "E0d_verdict": verdict}
    out = os.path.join(args.outdir, "E0d_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)
    print("-" * 78)
    print(f"E0d DECISION: {verdict['decision']} ({verdict['reason']})  "
          f"primary AUC={verdict['auc_mean']:.3f} +/- {verdict['auc_std']:.3f}, "
          f"naive={verdict['auc_naive_mean']:.3f}, null={verdict['auc_null_mean']:.3f}, "
          f"margin over controls={verdict['margin_over_controls']:.3f}, "
          f"regime={100 * verdict['regime_frac_acf_above_floor']:.1f}% above "
          f"{ACF_FLOOR} (floor {100 * REGIME_MIN_FRAC:.0f}%), n_seeds={verdict['n_seeds']}")
    if len(seeds) < 3:
        print("NOTE: fewer than 3 seeds -> PILOT reading, not the pre-registered verdict.")
    if args.smoke:
        print("SMOKE RUN: pipeline only, the numbers mean nothing.")
    print(f"summary -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
