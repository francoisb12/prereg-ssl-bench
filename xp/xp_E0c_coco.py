#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xp_E0c_coco.py -- E0 with a predictor that can see, and a ruler that reads
                  beyond the fine grain.  Same question as E0b, corrected design.

WHAT E0b ESTABLISHED (JOURNAL E-002, 2026-09-16)
-----------------------------------------------
The instrument is alive (residual magnitude separates thing/stuff at 0.568 and
boundary cells at 0.623), but the regime T4 wants to test was EMPTY: 0.2% of
cells had a residual correlated beyond the 11 px receptive field, and `ell`
sat at ~3 px on things and on stuff alike.  Two causes, both design:

  1. A myopic predictor.  rf 11 px, hidden blocks of 8 px, 75% of blocks
     hidden: it saw a 1.5 px ring around each block, itself hidden 3 times in
     4.  It learned a local blur in ~500 steps and nothing afterwards (loss
     flat from step 1000 to 10000).  Its residual was "the image minus a blur"
     everywhere, and the fine grain of that field is the same on a face and on
     grass.
  2. A ruler pinned at the finest scale.  The first 1/e crossing of the radial
     ACF measures the FASTEST fluctuation in the window; any grain pins it at
     ~3 px whatever lies underneath.  corr(ell, acf_energy) = 0.87: at those
     scales `ell` and the naive control were the same statistic.

WHAT E0c CHANGES, AND WHY EACH CHANGE
-------------------------------------
  * EYES.  depth 10 -> rf 23 px.  The predictor sees across the hidden block
    and its neighbours, so it can learn shapes, not only a blur.
  * CONTEXT.  33% of blocks hidden (1 group in 3) instead of 75%.  In a conv
    predictor context is LOCAL: with 75% hidden the receptive field holds on
    average 2 visible blocks out of 9 and the model is starved.  The bench's
    author put it as a curriculum argument: you learn what a car is by seeing
    whole cars, and only then can you recognise one from a few pixels.  Whether
    that holds in general is an open question (global-attention models like
    MAE do well with 75% hidden precisely because their context is global); for
    a local predictor whose residual we want to read, it is the right call.
    Masks are complementary groups, so the hidden fraction is 1/V with V an
    integer: V=3 gives 33%, the largest value at or under the requested 40%.
  * WINDOW.  win_mult 8 -> 64 px, ell censored at 32 px; rf 23 stays under the
    harness headroom rule (rf <= 0.75 * censor = 24).
  * RULER.  A third estimator, `acf_at_rf`: the normalised radial ACF read
    EXACTLY at lag = receptive field.  For a grain-only field it is ~0; for
    grain + a coherent blob it is the fraction of variance that is long-range.
    No smoothing, hence no smoothing bias.  It is the PRIMARY statistic of
    P0c, because it is the literal operationalisation of "correlated beyond
    what the model sees".  `ell_acf` and `ell_freq` are kept as secondaries,
    plus `ell_acf_smooth` (ell on the residual box-filtered at rf/2) as an
    exploratory read, biased upward by construction and labelled so.
  * PRECONDITION, reported separately from the verdict: the fraction of
    labelled cells whose RAW ell_acf exceeds rf.  Under 5% the regime is
    declared EMPTY and the run is NO-GO "by construction", not "ell fails".
  * FUSE.  At 20% of the budget, if the loss improved by less than 1% since
    10%, the predictor is too weak for its task and the run aborts (E0b's loss
    went 0.165 -> 0.168 over that interval).  --no-abort-on-plateau overrides.

Everything else is imported verbatim: data and labels from xp_E0b_coco, model,
masks, residuals, ell estimators and thresholds from xp_E_router.

USAGE
-----
  python xp/xp_E0c_coco.py --smoke --cpu
  python xp/xp_E0c_coco.py --seed 0 --data-root /content/ubergang_local/data/cocostuff

Cost at the defaults (128 px, depth 10, bs 128, 10000 steps): about 2.5x E0b,
so budget 2 to 3 h A100 per seed.  Run --seed 0 first.
"""

from __future__ import annotations

import argparse
import json
import math
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

CELL_THING, CELL_STUFF, CELL_EXCLUDED = EB.CELL_THING, EB.CELL_STUFF, EB.CELL_EXCLUDED
REGIME_MIN_FRAC = 0.05        # below this, "ell > rf" is an empty regime
PLATEAU_MIN_GAIN = 0.01       # loss must improve >= 1% between 10% and 20% of budget

PREREG_E0C = {
    "statement": "P0c: on COCO images at 128 px, with a predictor whose receptive "
                 "field (23 px) exceeds the hidden block (8 px) and its neighbours, "
                 "the normalised residual autocorrelation read AT the receptive "
                 "field (acf_at_rf) separates THING cells from STUFF cells better "
                 "than the naive short-lag structuredness statistic",
    "primary_statistic": "auc_acf_at_rf",
    "secondary_statistics": ["auc_ell_acf", "auc_ell_freq", "auc_ell_acf_smooth"],
    "decision_rule": "same thresholds as E0/E0b, applied to the primary statistic: "
                     f"GO iff mean AUC >= {ER.GO_AUC} over >= 3 seeds AND margin over "
                     f"naive >= {ER.GO_MARGIN_OVER_NAIVE} AND mean - std >= {ER.NOGO_AUC}; "
                     f"NO-GO iff mean < {ER.NOGO_AUC}",
    "precondition": f"fraction of labelled cells with RAW ell_acf > receptive field must "
                    f"reach {REGIME_MIN_FRAC}; otherwise the verdict is 'regime empty' "
                    f"(NO-GO by construction), reported apart from a separation failure",
    "mask": "1 block in 3 hidden (33%), complementary groups, each block hidden exactly "
            "once in the diagnostic sweep; chosen at or under the 40% ceiling requested "
            "on the grounds that a local predictor needs visible context inside its "
            "receptive field to learn anything beyond a blur",
    "fuse": f"abort at 20% of the budget if the loss gained < {PLATEAU_MIN_GAIN:.0%} since 10%",
    "registered": "2026-09-16, before the first run of this script, after E-002",
}


# =============================================================================
# 1.  Inverted complementary masks: hidden = the sample's ONE group
# =============================================================================

def _hidden_masks_inv(batch_idx: np.ndarray, cfg: "ER.Cfg", grid: int,
                      device: torch.device, variant: np.ndarray
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """hidden_region [B,G,G] on CPU (bool), mask_px [B,1,H,W] on device.

    ER.mask_from_groups hides every block whose group DIFFERS from the variant
    (ratio (V-1)/V).  Here the block whose group EQUALS the variant is hidden
    (ratio 1/V), so V=3 hides one third.  Over the V variants each block is
    hidden exactly once, which is what the diagnostic sweep relies on.
    """
    groups, _own = ER.deterministic_mask_groups(
        batch_idx, grid * grid, cfg.mask_variants, cfg.mask_seed)
    hidden = torch.from_numpy(groups) == torch.from_numpy(np.asarray(variant)).view(-1, 1)
    hidden_region = hidden.view(-1, grid, grid)
    mask_px = hidden_region.float().unsqueeze(1).repeat_interleave(
        cfg.patch, dim=2).repeat_interleave(cfg.patch, dim=3)
    return hidden_region, mask_px.to(device)


@torch.no_grad()
def e0c_diagnostic_sweep(model: "ER.HierarchicalMaskedPredictor", pool: Dataset,
                         cfg: "ER.Cfg", device: torch.device, indices: np.ndarray
                         ) -> Dict[str, np.ndarray]:
    """Every region measured while HIDDEN, exactly once (one variant hides it)."""
    model.eval()
    G = cfg.size // cfg.patch
    idx_all = np.asarray(indices, dtype=np.int64)
    n = int(len(idx_all))
    res_region = np.zeros((n, G, G), dtype=np.float64)
    res_pix = np.zeros((n, cfg.size, cfg.size), dtype=np.float64)
    cnt_region = np.zeros((n, G, G), dtype=np.float64)
    loader = DataLoader(Subset(pool, idx_all.tolist()), batch_size=min(cfg.bs, 128),
                        shuffle=False, num_workers=cfg.workers, drop_last=False)
    pos = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        bidx = batch["idx"].numpy()
        b = x.shape[0]
        acc_r = torch.zeros(b, G, G, device=device)
        acc_c = torch.zeros(b, G, G, device=device)
        acc_p = torch.zeros(b, cfg.size, cfg.size, device=device)
        acc_pc = torch.zeros(b, cfg.size, cfg.size, device=device)
        for v in range(cfg.mask_variants):
            hidden_region, mask_px = _hidden_masks_inv(
                bidx, cfg, G, device, variant=np.full(b, v, dtype=np.int64))
            xin = model.apply_mask(x, mask_px)
            preds, _ = model(xin)
            res_cells, pix0 = ER._targets_and_residuals(x, preds, cfg.patch)
            hr = hidden_region.to(device).float()
            acc_r += res_cells[0] * hr
            acc_c += hr
            hp = mask_px.squeeze(1)
            acc_p += pix0 * hp
            acc_pc += hp
        res_region[pos:pos + b] = (acc_r / acc_c.clamp_min(1)).cpu().numpy()
        res_pix[pos:pos + b] = (acc_p / acc_pc.clamp_min(1)).cpu().numpy()
        cnt_region[pos:pos + b] = acc_c.cpu().numpy()
        pos += b
    model.train()
    return {"res_region": res_region, "res_pix": res_pix,
            "cnt_region": cnt_region, "idx": idx_all}


# =============================================================================
# 2.  The ruler that reads at a chosen lag: normalised radial ACF at r = rf
# =============================================================================

@torch.no_grad()
def radial_acf_at(field: torch.Tensor, patch: int, win_mult: int,
                  radii: Sequence[float], chunk: int = 32) -> Dict[float, torch.Tensor]:
    """Normalised, radially averaged ACF of ``field`` [B,H,W] read at the given
    radii (px), per region, on the same window as ER.field_statistics.

    Mirrors field_statistics' estimator (fully observed field, so the missing-
    data normalisation degenerates to the usual one) and returns the profile
    value at each requested radius instead of the 1/e crossing.  Returns
    {radius: [B,G,G]}.
    """
    B, H, W = field.shape
    G = H // patch
    win = int(win_mult) * patch
    pad_px = (win - patch) // 2
    dev = field.device
    f_pad = F.pad(field.unsqueeze(1), (pad_px,) * 4, mode="reflect")
    pad = 2 * win
    rbin, _r, nbins = ER._radial_bins(pad, dev)
    rbin_flat = rbin.reshape(-1)
    counts = torch.zeros(nbins, device=dev).index_add_(
        0, rbin_flat, torch.ones_like(rbin_flat, dtype=torch.float32)).clamp_min(1.0)
    out = {float(r): torch.zeros(B, G * G, device=dev) for r in radii}
    for b0 in range(0, B, chunk):
        b1 = min(B, b0 + chunk)
        fw = F.unfold(f_pad[b0:b1], kernel_size=win, stride=patch)
        nb = fw.shape[0]
        fw = fw.transpose(1, 2).reshape(nb * G * G, win, win)
        x = fw - fw.mean(dim=(-2, -1), keepdim=True)
        m = torch.ones_like(x)
        Fx = torch.fft.rfft2(x, s=(pad, pad))
        Fm = torch.fft.rfft2(m, s=(pad, pad))
        num = torch.fft.irfft2(Fx.real ** 2 + Fx.imag ** 2, s=(pad, pad))
        den = torch.fft.irfft2(Fm.real ** 2 + Fm.imag ** 2, s=(pad, pad))
        ok = den > 8.0
        acf = torch.where(ok, num / den.clamp_min(1e-6), torch.zeros_like(num))
        acf0 = acf[:, :1, :1].clamp_min(1e-12)
        acf = torch.fft.fftshift(acf / acf0, dim=(-2, -1))
        okm = torch.fft.fftshift(ok.float(), dim=(-2, -1))
        n = acf.shape[0]
        prof = torch.zeros(n, nbins, device=dev).index_add_(
            1, rbin_flat, (acf * okm).reshape(n, -1))
        prof = prof / torch.zeros(n, nbins, device=dev).index_add_(
            1, rbin_flat, okm.reshape(n, -1)).clamp_min(1.0)
        for r in radii:
            k = int(min(nbins - 1, max(0, round(float(r)))))
            out[float(r)][b0:b1] = prof[:, k].view(nb, G * G)
        del Fx, Fm, num, den, acf, okm, prof
    return {r: v.view(B, G, G) for r, v in out.items()}


def _smooth(field: torch.Tensor, k: int) -> torch.Tensor:
    """Box filter, odd kernel k, same size.  Used ONLY for the exploratory
    ell_acf_smooth: smoothing at scale k imposes a correlation length ~k on
    every cell, so this statistic is biased upward by construction."""
    k = max(3, int(k) | 1)
    return F.avg_pool2d(field.unsqueeze(1), kernel_size=k, stride=1,
                        padding=k // 2, count_include_pad=False).squeeze(1)


# =============================================================================
# 3.  Verdict: precondition first, then the E0 rule on the primary statistic
# =============================================================================

def e0c_verdict(summaries: List[dict]) -> dict:
    prim = ER.mean_std([s["auc_acf_at_rf"] for s in summaries])
    naive = ER.mean_std([s["auc_naive_acf_energy"] for s in summaries])
    regime = ER.mean_std([s["frac_ell_above_rf"] for s in summaries])
    m, sd = prim
    margin = m - naive[0]
    regime_empty = (not np.isfinite(regime[0])) or regime[0] < REGIME_MIN_FRAC
    if regime_empty:
        verdict, decision, reason = "non concluante", "NO-GO", "regime_empty"
    elif not np.isfinite(m):
        verdict, decision, reason = "non concluante", "NO-GO", "nan"
    elif m >= ER.GO_AUC and margin >= ER.GO_MARGIN_OVER_NAIVE and (m - sd) >= ER.NOGO_AUC:
        verdict, decision, reason = "confirmee", "GO", "primary_passes"
    elif m < ER.NOGO_AUC:
        verdict, decision, reason = "infirmee", "NO-GO", "primary_below_nogo"
    else:
        verdict, decision, reason = "non concluante", "NO-GO", "between_thresholds"
    return {
        "prediction": PREREG_E0C["statement"],
        "verdict": verdict, "decision": decision, "reason": reason,
        "primary_statistic": "auc_acf_at_rf",
        "auc_mean": m, "auc_std": sd,
        "auc_naive_mean": naive[0], "auc_naive_std": naive[1],
        "margin_over_naive": margin,
        "regime_frac_ell_above_rf": regime[0], "regime_min_frac": REGIME_MIN_FRAC,
        "regime_empty": bool(regime_empty),
        "n_seeds": len(summaries),
        "secondary": {k: ER.mean_std([s[k] for s in summaries])
                      for k in ("auc_ell_acf", "auc_ell_freq", "auc_ell_acf_smooth",
                                "auc_residual_magnitude", "censored_frac")},
    }


# =============================================================================
# 4.  The run
# =============================================================================

def run_e0c(seed: int, cfg: "ER.Cfg", args, outdir: str, device: torch.device,
            resume: bool = True) -> dict:
    set_seed(seed)
    name = f"E0c_coco_seed{seed}"
    run = Run(name=name, outdir=outdir,
              config={"stage": "E0c", "seed": seed, "n_images": args.n_images,
                      "purity": args.purity, "thing_max_label": args.thing_max_label,
                      "mask_ratio": 1.0 / cfg.mask_variants, "smoke": bool(args.smoke),
                      **cfg.to_dict()},
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

    diag_idx = np.sort(np.random.RandomState(seed).permutation(len(pool))[:cfg.e0_diag_n])
    diag_idx = diag_idx.astype(np.int64)
    cells = pool.cell_labels_for(diag_idx)
    n_thing = int((cells == CELL_THING).sum())
    n_stuff = int((cells == CELL_STUFF).sum())
    n_excl = int((cells == CELL_EXCLUDED).sum())
    tot = cells.size
    hist = pool.label_histogram()
    print(f"[E0c] label histogram (id, pixels), boundary <= {args.thing_max_label}:")
    print(f"      thing side: {hist['thing_side_top_ids']}")
    print(f"      stuff side: {hist['stuff_side_top_ids']}")
    print(f"[E0c] diagnostic cells: {tot} | {n_thing} thing ({100 * n_thing / tot:.1f}%) | "
          f"{n_stuff} stuff ({100 * n_stuff / tot:.1f}%) | {n_excl} excluded "
          f"({100 * n_excl / tot:.1f}%)")
    labeled = n_thing + n_stuff
    if labeled == 0 or min(n_thing, n_stuff) < 0.02 * labeled:
        raise RuntimeError("[E0c] one class holds under 2% of labelled cells; fix "
                           "--thing-max-label or --purity BEFORE spending GPU time.")

    model = ER._make_model(cfg, levels=1, device=device)
    rf0 = float(model.receptive_field_px[0])
    window = cfg.win_mult * cfg.patch
    censor = window / 2.0
    print(f"[E0c] receptive field = {rf0:.1f} px | window = {window} px "
          f"(ell censored at {censor:.1f} px) | hidden fraction = 1/{cfg.mask_variants}")
    if rf0 > ER.ELL_HEADROOM_RATIO * censor:
        raise RuntimeError(f"[E0c] rf {rf0:.0f} px exceeds {ER.ELL_HEADROOM_RATIO:.2f} x "
                           f"censor ({censor:.0f} px): 'ell > rf' would not be measurable. "
                           f"Lower --depth or raise --win-mult.")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[E0c] predictor: depth {cfg.depth}, width {cfg.width}, {n_params / 1e6:.2f} M params")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

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
                raise RuntimeError(f"[E0c] stale snapshot for {name}: delete {name}.* "
                                   f"and run this seed fresh.")
        print(f"[E0c] resumed at step {start_step}")

    snap_prev_path = os.path.join(outdir, f"{name}.snap_prev.npz")
    snap_step = max(1, cfg.e0_steps - cfg.e0_lam_gap)

    # Plateau fuse bookkeeping: mean loss over a trailing window at 5/10/20%.
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
        hidden_region, mask_px = _hidden_masks_inv(bidx, cfg, G, device, variant=variant)
        hidden_region = hidden_region.to(device)
        for pg in opt.param_groups:
            pg["lr"] = ER._lr_at(step, cfg, cfg.e0_steps)
        with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
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
            print(f"[E0c] loss at {100 * step / N:.0f}% of budget: {fuse[marks[step]]:.5f}")
            if marks[step] == "l20" and "l10" in fuse and not args.smoke:
                gain = (fuse["l10"] - fuse["l20"]) / max(fuse["l10"], 1e-12)
                run.log(step, plateau_gain_10_to_20=gain)
                if gain < PLATEAU_MIN_GAIN and not args.no_abort_on_plateau:
                    raise SystemExit(
                        f"[E0c] PLATEAU: loss gained {100 * gain:.2f}% between 10% and "
                        f"20% of the budget (< {100 * PLATEAU_MIN_GAIN:.0f}%). The predictor "
                        f"has stopped learning; its residual will not carry range "
                        f"information (E-002). Aborting to save GPU time; "
                        f"--no-abort-on-plateau overrides.")

        if step % cfg.log_every == 0:
            run.log(step, loss=lv, lr=ER._lr_at(step, cfg, cfg.e0_steps), seed=seed,
                    imgs_per_s=step * cfg.bs / max(1e-6, time.time() - t0))
        if step == snap_step and snap_prev is None:
            snap_prev = e0c_diagnostic_sweep(model, pool, cfg, device, diag_idx)
            np.savez_compressed(snap_prev_path, **snap_prev)
            print(f"[E0c] first residual snapshot at step {step}")
        if step % cfg.ckpt_every == 0 or step == cfg.e0_steps:
            run.save_ckpt(step, model=model.state_dict(), opt=opt.state_dict(),
                          snap_prev_path=snap_prev_path if snap_prev is not None else None,
                          best_metric=lv)

    if snap_prev is None:
        snap_prev = e0c_diagnostic_sweep(model, pool, cfg, device, diag_idx)
        np.savez_compressed(snap_prev_path, **snap_prev)
    snap_now = e0c_diagnostic_sweep(model, pool, cfg, device, diag_idx)

    r_prev = snap_prev["res_region"].astype(np.float64)
    r_now = snap_now["res_region"].astype(np.float64)
    n = min(r_prev.shape[0], r_now.shape[0])
    r_prev, r_now, cells = r_prev[:n], r_now[:n], cells[:n]
    lam_raw = (r_prev - r_now) / (r_prev + cfg.eps_r)
    lam_drift = lam_raw - float(np.nanmean(lam_raw))

    res_pix = torch.from_numpy(snap_now["res_pix"][:n]).float().to(device)
    valid = torch.ones_like(res_pix)
    chunks, chunks_s, acf_rf = [], [], []
    k_smooth = max(3, int(round(rf0 / 2.0)) | 1)
    for a in range(0, n, 64):
        blk = res_pix[a:a + 64]
        chunks.append(ER.field_statistics(blk, valid[a:a + 64], cfg.patch,
                                          chunk=cfg.ell_chunk, win_mult=cfg.win_mult))
        chunks_s.append(ER.field_statistics(_smooth(blk, k_smooth), valid[a:a + 64],
                                            cfg.patch, chunk=cfg.ell_chunk,
                                            win_mult=cfg.win_mult))
        acf_rf.append(radial_acf_at(blk, cfg.patch, cfg.win_mult, radii=[rf0],
                                    chunk=cfg.ell_chunk)[float(rf0)])
    stats = {k: torch.cat([c[k] for c in chunks], 0).cpu().numpy() for k in chunks[0]}
    ell_smooth = torch.cat([c["ell_acf"] for c in chunks_s], 0).cpu().numpy()
    acf_at_rf = torch.cat(acf_rf, 0).cpu().numpy()

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
        "ell_acf_smooth": ell_smooth.reshape(-1),
        "acf_at_rf": acf_at_rf.reshape(-1),
        "cell_label": cells.reshape(-1).astype(np.int64),
    }
    table_path = os.path.join(outdir, f"{name}.regions.npz")
    np.savez_compressed(table_path, **table)

    lab = table["cell_label"]
    labeled_mask = lab >= 0
    aucs = {
        "auc_acf_at_rf": EB._labeled_auc(table["acf_at_rf"], lab),
        "auc_ell_acf": EB._labeled_auc(table["ell_acf"], lab),
        "auc_ell_freq": EB._labeled_auc(table["ell_freq"], lab),
        "auc_ell_acf_smooth": EB._labeled_auc(table["ell_acf_smooth"], lab),
        "auc_naive_acf_energy": EB._labeled_auc(table["acf_energy"], lab),
        "auc_residual_magnitude": EB._labeled_auc(table["r"], lab),
    }
    stuck = table["lam_raw"] < np.nanquantile(table["lam_raw"], 0.5)
    aucs["auc_acf_at_rf_stuck"] = EB._labeled_auc(table["acf_at_rf"], lab, stuck)
    aucs["auc_naive_stuck"] = EB._labeled_auc(table["acf_energy"], lab, stuck)

    ell_lab = table["ell_acf"][labeled_mask]
    frac_above = float(np.nanmean(ell_lab > rf0))
    summary = {
        "seed": seed, "receptive_field_px": rf0, "measurement_window_px": int(window),
        "mask_ratio": 1.0 / cfg.mask_variants,
        "censored_frac": float(np.mean(table["censored"][labeled_mask])),
        "n_cells_total": int(lab.size), "n_cells_labeled": int(labeled_mask.sum()),
        "frac_thing_cells": float((lab == CELL_THING).sum() / max(1, labeled_mask.sum())),
        "frac_excluded_cells": float((lab == CELL_EXCLUDED).sum() / lab.size),
        "frac_ell_above_rf": frac_above,
        "frac_ell_above_rf_thing": float(np.nanmean(table["ell_acf"][lab == CELL_THING] > rf0)),
        "frac_ell_above_rf_stuff": float(np.nanmean(table["ell_acf"][lab == CELL_STUFF] > rf0)),
        "mean_ell_thing": float(np.nanmean(table["ell_acf"][lab == CELL_THING])),
        "mean_ell_stuff": float(np.nanmean(table["ell_acf"][lab == CELL_STUFF])),
        "mean_acf_at_rf_thing": float(np.nanmean(table["acf_at_rf"][lab == CELL_THING])),
        "mean_acf_at_rf_stuff": float(np.nanmean(table["acf_at_rf"][lab == CELL_STUFF])),
        "loss_marks": fuse, "n_diag_images": int(n),
        "regions_table": os.path.basename(table_path),
        **aucs,
    }
    run.log(cfg.e0_steps, **{k: v for k, v in summary.items() if isinstance(v, (int, float))})
    run.finish(summary)
    print(f"[E0c seed {seed}] PRIMARY AUC(acf_at_rf)={aucs['auc_acf_at_rf']:.3f}  "
          f"AUC(naive)={aucs['auc_naive_acf_energy']:.3f}  |  secondaries: "
          f"ell_acf={aucs['auc_ell_acf']:.3f} ell_freq={aucs['auc_ell_freq']:.3f} "
          f"ell_smooth={aucs['auc_ell_acf_smooth']:.3f} r={aucs['auc_residual_magnitude']:.3f}")
    print(f"[E0c seed {seed}] regime: {100 * frac_above:.2f}% of labelled cells have "
          f"ell > rf ({100 * summary['frac_ell_above_rf_thing']:.2f}% thing, "
          f"{100 * summary['frac_ell_above_rf_stuff']:.2f}% stuff); floor {100 * REGIME_MIN_FRAC:.0f}%")
    return summary


# =============================================================================
# 5.  CLI
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--outdir", default=os.path.join(_ROOT, "results", "E0c"))
    p.add_argument("--data-root", default=os.path.join(_ROOT, "data", "cocostuff"))
    p.add_argument("--n-images", type=int, default=5000)
    p.add_argument("--e0-steps", type=int, default=10000)
    p.add_argument("--e0-lam-gap", type=int, default=2500)
    p.add_argument("--e0-diag-n", type=int, default=1024)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--patch", type=int, default=8)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=10, help="rf0 = 1 + 2*(1+depth); 10 -> 23 px")
    p.add_argument("--win-mult", type=int, default=8, help="window = win_mult*patch; 8 -> 64 px")
    p.add_argument("--mask-groups", type=int, default=3,
                   help="hidden fraction = 1/groups; 3 -> 33%% (at or under the 40%% ceiling)")
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
        args.size, args.patch, args.width, args.depth = 48, 8, 16, 2
        args.win_mult, args.bs, args.workers, args.mask_groups = 4, 16, 0, 3
        args.e0_steps, args.e0_lam_gap, args.e0_diag_n, args.n_images = 60, 20, 32, 64
        if args.seed is None:
            args.seed = 0
    seeds = [args.seed] if args.seed is not None else \
        [int(s) for s in str(args.seeds).split(",") if s != ""]
    device = torch.device("cpu") if args.cpu else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not args.cpu and not args.smoke:
        raise SystemExit("[E0c] no CUDA device found and --cpu was not passed. At 128 px "
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
    print(f"E0c (T4, region labels, seeing predictor)  |  device={device}  seeds={seeds}  "
          f"size={cfg.size} patch={cfg.patch} win={cfg.win_mult * cfg.patch}px  "
          f"hidden=1/{cfg.mask_variants}")
    print(f"outdir = {args.outdir}")
    print("=" * 78)

    summaries = [run_e0c(s, cfg, args, args.outdir, device, resume=args.resume)
                 for s in seeds]
    verdict = e0c_verdict(summaries)
    payload = {"prereg": PREREG_E0C,
               "config": {**cfg.to_dict(), "n_images": args.n_images, "purity": args.purity,
                          "thing_max_label": args.thing_max_label,
                          "mask_ratio": 1.0 / cfg.mask_variants, "smoke": bool(args.smoke)},
               "per_seed": summaries, "E0c_verdict": verdict}
    out = os.path.join(args.outdir, "E0c_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)
    print("-" * 78)
    print(f"E0c DECISION: {verdict['decision']} ({verdict['reason']})  "
          f"primary AUC={verdict['auc_mean']:.3f} +/- {verdict['auc_std']:.3f}, "
          f"naive={verdict['auc_naive_mean']:.3f}, margin={verdict['margin_over_naive']:.3f}, "
          f"regime={100 * verdict['regime_frac_ell_above_rf']:.1f}% (floor "
          f"{100 * REGIME_MIN_FRAC:.0f}%), n_seeds={verdict['n_seeds']}")
    if len(seeds) < 3:
        print("NOTE: fewer than 3 seeds -> PILOT reading, not the pre-registered verdict.")
    if args.smoke:
        print("SMOKE RUN: pipeline only, the numbers mean nothing.")
    print(f"summary -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
