#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xp_E0b_coco.py -- E0 at real resolution, with REGION-LEVEL ground truth.

WHY THIS FILE EXISTS
--------------------
E0 (xp_E_router.py, stage e0) asks the T4 question: does the correlation
length `ell` of the residual field separate texture from long structure,
better than the naive "residual is structured" statistic?  Its first run was
an instrument failure (JOURNAL A-002 era: single-origin diagnostic set, NaN
AUCs), and even fixed, it keeps two structural weaknesses:

  1. LABELS BY DATASET ORIGIN.  An image from the object pool counts as
     "structure" in EVERY region, but an STL-10 photo is mostly textured
     background; a DTD image can carry long structure.  The label is a proxy,
     and the AUC measures the proxy as much as the thesis.
  2. SCALE.  At size=64, patch=8, win_mult=4, `ell` is censored at 16 px and
     the level-0 receptive field is 9 px: the interval where "ell > receptive
     field" is even representable is a few pixels wide.

E0b fixes both, and changes NOTHING else:

  * COCO-Stuff pixel maps give a human-annotated THING / STUFF label per
    region of the SAME image ("things" = objects = long structure; "stuff" =
    grass, water, sky, wall = texture-like).  Cells that are mixed or mostly
    unlabeled are EXCLUDED, not guessed.
  * size=128, patch=8, win_mult=6: window 48 px, ell censored at 24 px,
    receptive field ~11 px at depth=4.  `ell` has room on both sides.
  * model, masking protocol, residual sweep, ell estimators, decision rule:
    imported VERBATIM from xp_E_router.  Same thresholds, same verdict code.

WHAT IT DOES NOT FIX (say it before a reviewer does): COCO "stuff" contains
smooth expanses (sky, plain walls) whose residual is near zero; a correlation
length on a near-empty field is noise.  Those cells are NOT removed from the
pre-registered AUC (that would be a knob); a secondary AUC excluding the
lowest-energy decile is reported alongside, labelled exploratory.

DATA LAYOUT (see --help for the download commands)
--------------------------------------------------
  <data-root>/val2017/*.jpg                          COCO val2017 images
  <data-root>/stuffthingmaps/val2017/*.png           COCO-Stuff label maps

Label encoding of stuffthingmaps PNGs (nightrome/cocostuff): one byte per
pixel, 255 = unlabeled, thing classes in the low block, stuff classes above.
The default boundary is --thing-max-label 90 (values <= 90 count as things).
THE BOUNDARY IS THE ONE FRAGILE ASSUMPTION OF THIS FILE: the startup banner
prints the pixel-count histogram of both sides on a sample of images so a
mis-set boundary is visible before any GPU time is spent, and the run aborts
if either class holds under 2 percent of labeled cells.

USAGE
-----
  python xp/xp_E0b_coco.py --smoke --cpu                   # 2-3 min, synthetic
  python xp/xp_E0b_coco.py --data-root ./data/cocostuff    # the real thing

Cost at the defaults (size 128, bs 192, 10000 steps): a 128 px step costs
about 4x a 64 px step, so budget ~1.5-2.5 h A100 PER SEED.  Run --seed 0
first and look at the AUCs before paying for seeds 1 and 2.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from lib.harness import Run, set_seed  # noqa: E402
import xp_E_router as ER               # noqa: E402  (model, masks, stats, verdict)

CELL_THING, CELL_STUFF, CELL_EXCLUDED = 1, 0, -1

PREREG_E0B = {
    "statement": "P0b: on COCO images at 128 px, the correlation length ell of "
                 "the residual field separates THING cells from STUFF cells "
                 "(human region labels, same image), better than the naive "
                 "'residual is structured' statistic",
    "decision_rule": "identical to E0 and evaluated by xp_E_router.e0_verdict: "
                     f"GO iff mean AUC >= {ER.GO_AUC} over >= 3 seeds AND "
                     f"margin over naive >= {ER.GO_MARGIN_OVER_NAIVE} AND "
                     f"mean - std >= {ER.NOGO_AUC}; NO-GO iff mean < {ER.NOGO_AUC}",
    "population": "all cells whose COCO-Stuff purity reaches --purity; mixed "
                  "and unlabeled cells are excluded before the AUC, and the "
                  "excluded fraction is reported",
    "declared_secondary": "auc_*_energetic (lowest-energy decile of cells "
                          "removed) is exploratory: smooth stuff (sky) has a "
                          "near-empty residual whose ell is noise",
    "registered": "2026-09-06, before the first run of this script",
}


# =============================================================================
# 1.  Data: COCO images + per-cell thing/stuff labels
# =============================================================================

def _center_crop_resize(img: "np.ndarray", size: int, nearest: bool) -> "np.ndarray":
    """Square center crop then resize.  Labels use NEAREST, images bilinear."""
    h, w = img.shape[:2]
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    img = img[top:top + side, left:left + side]
    t = torch.from_numpy(np.ascontiguousarray(img))
    if img.ndim == 2:
        t = t[None, None].float()
        mode = "nearest"
    else:
        t = t.permute(2, 0, 1)[None].float()
        mode = "nearest" if nearest else "bilinear"
    kw = {} if mode == "nearest" else {"align_corners": False}
    out = F.interpolate(t, size=(size, size), mode=mode, **kw)[0]
    return out.numpy() if img.ndim != 2 else out[0].numpy()


class CocoStuffCells(Dataset):
    """COCO val images with a lazily loaded [G,G] thing/stuff label per image.

    ``__getitem__`` never touches the label PNG: training only needs pixels,
    and the diagnostic fetches labels once, for its own indices, through
    ``cell_labels_for``.  Batches carry the same keys as the E0 pool
    ({"x","label","idx","origin"}) so every xp_E_router utility works unchanged.
    """

    def __init__(self, images_dir: str, labels_dir: str, size: int, patch: int,
                 n_images: int, seed: int, purity: float, thing_max_label: int):
        from PIL import Image  # local: torchvision pulls PIL anyway
        self._Image = Image
        self.size, self.patch = int(size), int(patch)
        self.G = self.size // self.patch
        self.purity = float(purity)
        self.thing_max = int(thing_max_label)
        self.labels_dir = labels_dir

        imgs = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
        pairs = []
        for ip in imgs:
            lp = os.path.join(labels_dir,
                              os.path.splitext(os.path.basename(ip))[0] + ".png")
            if os.path.exists(lp):
                pairs.append((ip, lp))
        if not pairs:
            raise FileNotFoundError(
                f"no (jpg, png) pair under {images_dir} + {labels_dir}. "
                f"Expected COCO val2017 images and stuffthingmaps PNGs; see the "
                f"module docstring for the download commands.")
        order = np.random.RandomState(seed).permutation(len(pairs))
        keep = order[:min(int(n_images), len(pairs))]
        self.pairs = [pairs[i] for i in keep]
        self._cell_cache: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> dict:
        ip, _ = self.pairs[int(i)]
        img = np.asarray(self._Image.open(ip).convert("RGB"), dtype=np.float32) / 255.0
        img = _center_crop_resize(img, self.size, nearest=False)
        x = torch.from_numpy(img).float()
        x = (x - 0.5) / 0.5
        return {"x": x, "label": 0, "idx": int(i), "origin": 0}

    def _cells_one(self, i: int) -> np.ndarray:
        _, lp = self.pairs[int(i)]
        lab = np.asarray(self._Image.open(lp), dtype=np.int64)
        lab = _center_crop_resize(lab, self.size, nearest=True).astype(np.int64)
        G, p = self.G, self.patch
        cells = lab.reshape(G, p, G, p).transpose(0, 2, 1, 3).reshape(G, G, p * p)
        labeled = cells != 255
        thing = labeled & (cells <= self.thing_max)
        stuff = labeled & (cells > self.thing_max)
        f_thing = thing.mean(axis=-1)
        f_stuff = stuff.mean(axis=-1)
        out = np.full((G, G), CELL_EXCLUDED, dtype=np.int8)
        out[f_thing >= self.purity] = CELL_THING
        out[f_stuff >= self.purity] = CELL_STUFF
        return out

    def cell_labels_for(self, indices: Sequence[int]) -> np.ndarray:
        out = np.zeros((len(indices), self.G, self.G), dtype=np.int8)
        for k, i in enumerate(indices):
            i = int(i)
            if i not in self._cell_cache:
                self._cell_cache[i] = self._cells_one(i)
            out[k] = self._cell_cache[i]
        return out

    def label_histogram(self, n_sample: int = 64) -> Dict[str, list]:
        """Pixel counts per raw label id, split at the boundary.  Eyeball this:
        a mis-set --thing-max-label shows up as absurd counts on one side."""
        counts: Dict[int, int] = {}
        for i in range(min(n_sample, len(self.pairs))):
            lab = np.asarray(self._Image.open(self.pairs[i][1]), dtype=np.int64)
            ids, c = np.unique(lab, return_counts=True)
            for v, n in zip(ids.tolist(), c.tolist()):
                counts[v] = counts.get(v, 0) + n
        counts.pop(255, None)
        thing = sorted(((v, n) for v, n in counts.items() if v <= self.thing_max),
                       key=lambda t: -t[1])[:8]
        stuff = sorted(((v, n) for v, n in counts.items() if v > self.thing_max),
                       key=lambda t: -t[1])[:8]
        return {"thing_side_top_ids": thing, "stuff_side_top_ids": stuff}


class SyntheticThingsStuff(Dataset):
    """Smoke fixture: textured background + one smooth blob, labels known.

    Background = white noise lightly blurred (short correlation).  Blob = a
    smooth low-frequency disc (long correlation).  The label map is the blob
    mask, so the whole cells pipeline runs end to end with no download.
    """

    def __init__(self, n: int, size: int, patch: int, purity: float, seed: int):
        self.n, self.size, self.patch = int(n), int(size), int(patch)
        self.G = self.size // self.patch
        self.purity = float(purity)
        self.seed = int(seed)
        self._cell_cache: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self.n

    def _make(self, i: int) -> Tuple[torch.Tensor, np.ndarray]:
        g = np.random.RandomState(self.seed * 100003 + i)
        s = self.size
        noise = g.randn(3, s, s).astype(np.float32)
        x = torch.from_numpy(noise)
        x = F.avg_pool2d(x[None], 3, 1, 1)[0]           # texture: short-range
        yy, xx = np.mgrid[0:s, 0:s]
        cy, cx = g.randint(s // 4, 3 * s // 4, size=2)
        r = g.randint(s // 5, s // 3)
        d2 = ((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float32)
        blob = np.exp(-d2 / (2 * (r / 1.5) ** 2))       # smooth: long-range
        mask = (d2 <= r * r)
        xb = torch.from_numpy((blob * 2.0).astype(np.float32))[None].repeat(3, 1, 1)
        m = torch.from_numpy(mask.astype(np.float32))[None]
        img = x * (1 - m) * 0.6 + xb * m
        return img.float(), mask

    def __getitem__(self, i: int) -> dict:
        img, _ = self._make(int(i))
        return {"x": img, "label": 0, "idx": int(i), "origin": 0}

    def cell_labels_for(self, indices: Sequence[int]) -> np.ndarray:
        out = np.zeros((len(indices), self.G, self.G), dtype=np.int8)
        p = self.patch
        for k, i in enumerate(indices):
            i = int(i)
            if i not in self._cell_cache:
                _, mask = self._make(i)
                cells = mask.reshape(self.G, p, self.G, p).transpose(0, 2, 1, 3)
                f = cells.reshape(self.G, self.G, p * p).mean(axis=-1)
                lab = np.full((self.G, self.G), CELL_EXCLUDED, dtype=np.int8)
                lab[f >= self.purity] = CELL_THING
                lab[(1.0 - f) >= self.purity] = CELL_STUFF
                self._cell_cache[i] = lab
            out[k] = self._cell_cache[i]
        return out

    def label_histogram(self, n_sample: int = 8) -> Dict[str, list]:
        return {"thing_side_top_ids": [("synthetic_blob", -1)],
                "stuff_side_top_ids": [("synthetic_texture", -1)]}


# =============================================================================
# 2.  The run: identical protocol to run_e0, per-cell labels at the end
# =============================================================================

def _labeled_auc(values: np.ndarray, cell_label: np.ndarray,
                 extra_mask: Optional[np.ndarray] = None) -> float:
    m = cell_label >= 0
    if extra_mask is not None:
        m = m & extra_mask
    return ER.rank_auc(np.where(m, values, np.nan),
                       cell_label == CELL_THING)


def run_e0b(seed: int, cfg: "ER.Cfg", args, outdir: str,
            device: torch.device, resume: bool = True) -> dict:
    set_seed(seed)
    name = f"E0b_coco_seed{seed}"
    run = Run(name=name, outdir=outdir,
              config={"stage": "E0b", "seed": seed, "n_images": args.n_images,
                      "purity": args.purity, "thing_max_label": args.thing_max_label,
                      "smoke": bool(args.smoke), **cfg.to_dict()},
              resume=resume, higher_is_better=False)

    if args.smoke:
        pool: Dataset = SyntheticThingsStuff(args.n_images, cfg.size, cfg.patch,
                                             args.purity, seed=0)
    else:
        pool = CocoStuffCells(os.path.join(args.data_root, "val2017"),
                              os.path.join(args.data_root, "stuffthingmaps", "val2017"),
                              cfg.size, cfg.patch, args.n_images, seed=0,
                              purity=args.purity,
                              thing_max_label=args.thing_max_label)
    G = cfg.size // cfg.patch

    # ---- diagnostic set + labels, BEFORE any GPU time -----------------------
    diag_idx = np.random.RandomState(seed).permutation(len(pool))[:cfg.e0_diag_n]
    diag_idx = np.sort(diag_idx).astype(np.int64)
    cells = pool.cell_labels_for(diag_idx)               # [n, G, G] in {-1,0,1}
    n_thing = int((cells == CELL_THING).sum())
    n_stuff = int((cells == CELL_STUFF).sum())
    n_excl = int((cells == CELL_EXCLUDED).sum())
    tot = cells.size
    hist = pool.label_histogram()
    print(f"[E0b] label histogram on a sample (id, pixel count), "
          f"boundary at <= {args.thing_max_label}:")
    print(f"      thing side: {hist['thing_side_top_ids']}")
    print(f"      stuff side: {hist['stuff_side_top_ids']}")
    print(f"[E0b] diagnostic cells: {tot} total | {n_thing} thing "
          f"({100 * n_thing / tot:.1f}%) | {n_stuff} stuff "
          f"({100 * n_stuff / tot:.1f}%) | {n_excl} excluded "
          f"({100 * n_excl / tot:.1f}%)")
    labeled = n_thing + n_stuff
    if labeled == 0 or min(n_thing, n_stuff) < 0.02 * labeled:
        raise RuntimeError(
            "[E0b] one of the two classes holds under 2% of the labeled cells. "
            "Either --thing-max-label is mis-set for this stuffthingmaps "
            "encoding (check the histogram above) or --purity is too strict. "
            "Fix it BEFORE spending GPU time.")

    # ---- model + window bookkeeping (win_mult passed EXPLICITLY: E0 printed
    # 3*patch while measuring on win_mult*patch; here the print and the
    # measurement are the same variable by construction) ----------------------
    model = ER._make_model(cfg, levels=1, device=device)
    rf0 = model.receptive_field_px[0]
    window = cfg.win_mult * cfg.patch
    print(f"[E0b] receptive field = {rf0:.1f} px | window = {window} px "
          f"(ell censored at {window / 2:.1f} px)")
    if rf0 >= window / 2:
        raise RuntimeError("[E0b] receptive field >= half the window: "
                           "'ell > receptive field' is not measurable. "
                           "Lower --depth or raise --win-mult.")

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
                raise RuntimeError(
                    f"[E0b] {os.path.basename(ck['snap_prev_path'])} was measured "
                    f"on a different image set. Delete {name}.* and run fresh.")
        print(f"[E0b] resumed at step {start_step}")

    snap_prev_path = os.path.join(outdir, f"{name}.snap_prev.npz")
    snap_step = max(1, cfg.e0_steps - cfg.e0_lam_gap)

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
        _hc, hidden_region, mask_px = ER._hidden_masks(bidx, cfg, G, G, device,
                                                       variant=variant)
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
        if step % cfg.log_every == 0:
            run.log(step, loss=float(loss.detach()),
                    lr=ER._lr_at(step, cfg, cfg.e0_steps), seed=seed,
                    imgs_per_s=step * cfg.bs / max(1e-6, time.time() - t0))
        if step == snap_step and snap_prev is None:
            snap_prev = ER.e0_diagnostic_sweep(model, pool, cfg, cfg.e0_diag_n,
                                               device, diag_idx)
            np.savez_compressed(snap_prev_path, **snap_prev)
            print(f"[E0b] first residual snapshot at step {step}")
        if step % cfg.ckpt_every == 0 or step == cfg.e0_steps:
            run.save_ckpt(step, model=model.state_dict(), opt=opt.state_dict(),
                          snap_prev_path=snap_prev_path if snap_prev is not None else None,
                          best_metric=float(loss.detach()))

    if snap_prev is None:
        snap_prev = ER.e0_diagnostic_sweep(model, pool, cfg, cfg.e0_diag_n,
                                           device, diag_idx)
        np.savez_compressed(snap_prev_path, **snap_prev)
    snap_now = ER.e0_diagnostic_sweep(model, pool, cfg, cfg.e0_diag_n,
                                      device, diag_idx)

    # ---- lam + ell, verbatim E0 arithmetic ----------------------------------
    r_prev = snap_prev["res_region"].astype(np.float64)
    r_now = snap_now["res_region"].astype(np.float64)
    n = min(r_prev.shape[0], r_now.shape[0])
    r_prev, r_now, cells = r_prev[:n], r_now[:n], cells[:n]
    lam_raw = (r_prev - r_now) / (r_prev + cfg.eps_r)
    lam_drift = lam_raw - float(np.nanmean(lam_raw))

    res_pix = torch.from_numpy(snap_now["res_pix"][:n]).float().to(device)
    valid = torch.ones_like(res_pix)
    chunks = []
    for a in range(0, n, 64):
        chunks.append(ER.field_statistics(res_pix[a:a + 64], valid[a:a + 64],
                                          cfg.patch, chunk=cfg.ell_chunk,
                                          win_mult=cfg.win_mult))
    stats = {k: torch.cat([c[k] for c in chunks], 0).cpu().numpy()
             for k in chunks[0]}

    table = {
        "image_id": np.repeat(np.arange(n)[:, None, None], G, 1).repeat(G, 2).reshape(-1),
        "pool_idx": np.repeat(diag_idx[:n][:, None, None], G, 1).repeat(G, 2).reshape(-1),
        "row": np.tile(np.arange(G)[:, None], (n, 1, G)).reshape(-1),
        "col": np.tile(np.arange(G)[None, :], (n, G, 1)).reshape(-1),
        "r": r_now.reshape(-1),
        "r_prev": r_prev.reshape(-1),
        "lam_raw": lam_raw.reshape(-1),
        "lam_drift": lam_drift.reshape(-1),
        "ell_acf": stats["ell_acf"].reshape(-1),
        "ell_freq": stats["ell_freq"].reshape(-1),
        "acf_energy": stats["acf_energy"].reshape(-1),
        "censored": stats["censored"].reshape(-1),
        "cell_label": cells.reshape(-1).astype(np.int64),
    }
    table_path = os.path.join(outdir, f"{name}.regions.npz")
    np.savez_compressed(table_path, **table)

    lab = table["cell_label"]
    aucs = {
        "auc_ell_acf": _labeled_auc(table["ell_acf"], lab),
        "auc_ell_freq": _labeled_auc(table["ell_freq"], lab),
        "auc_naive_acf_energy": _labeled_auc(table["acf_energy"], lab),
        "auc_residual_magnitude": _labeled_auc(table["r"], lab),
    }
    stuck = table["lam_raw"] < np.nanquantile(table["lam_raw"], 0.5)
    aucs["auc_ell_acf_stuck"] = _labeled_auc(table["ell_acf"], lab, stuck)
    aucs["auc_naive_stuck"] = _labeled_auc(table["acf_energy"], lab, stuck)
    # exploratory: drop the lowest-energy decile (smooth sky problem)
    energetic = table["r"] >= np.nanquantile(table["r"], 0.10)
    aucs["auc_ell_acf_energetic"] = _labeled_auc(table["ell_acf"], lab, energetic)
    aucs["auc_naive_energetic"] = _labeled_auc(table["acf_energy"], lab, energetic)

    labeled_mask = lab >= 0
    summary = {
        "seed": seed,
        "receptive_field_px": float(rf0),
        "measurement_window_px": int(window),
        "censored_frac": float(np.mean(table["censored"][labeled_mask])),
        "n_cells_total": int(lab.size),
        "n_cells_labeled": int(labeled_mask.sum()),
        "frac_thing_cells": float((lab == CELL_THING).sum() / max(1, labeled_mask.sum())),
        "frac_excluded_cells": float((lab == CELL_EXCLUDED).sum() / lab.size),
        "mean_ell_thing": float(np.nanmean(table["ell_acf"][lab == CELL_THING])),
        "mean_ell_stuff": float(np.nanmean(table["ell_acf"][lab == CELL_STUFF])),
        "n_diag_images": int(n),
        "regions_table": os.path.basename(table_path),
        **aucs,
    }
    run.log(cfg.e0_steps, **{k: v for k, v in summary.items()
                             if isinstance(v, (int, float))})
    run.finish(summary)
    print(f"[E0b seed {seed}] AUC(ell_acf)={aucs['auc_ell_acf']:.3f}  "
          f"AUC(ell_freq)={aucs['auc_ell_freq']:.3f}  "
          f"AUC(naive)={aucs['auc_naive_acf_energy']:.3f}  "
          f"[energetic: ell={aucs['auc_ell_acf_energetic']:.3f} "
          f"naive={aucs['auc_naive_energetic']:.3f}]")
    return summary


# =============================================================================
# 3.  CLI
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""download (Colab):
  mkdir -p /content/ubergang_local/data/cocostuff && cd /content/ubergang_local/data/cocostuff
  wget -q http://images.cocodataset.org/zips/val2017.zip && unzip -q val2017.zip
  wget -q http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip
  mkdir -p stuffthingmaps && unzip -q stuffthingmaps_trainval2017.zip 'val2017/*' -d stuffthingmaps
""")
    p.add_argument("--seed", type=int, default=None, help="single seed shortcut")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--outdir", default=os.path.join(_ROOT, "results", "E0b"))
    p.add_argument("--data-root", default=os.path.join(_ROOT, "data", "cocostuff"))
    p.add_argument("--n-images", type=int, default=5000,
                   help="images drawn from val2017 (5000 = all of it; the pool "
                        "is small for 10k steps x bs 192, ~380 epochs, declared)")
    p.add_argument("--e0-steps", type=int, default=10000)
    p.add_argument("--e0-lam-gap", type=int, default=2500)
    p.add_argument("--e0-diag-n", type=int, default=1024)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--patch", type=int, default=8)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=4,
                   help="rf0 = 1 + 2*(1+depth) px; default 4 -> 11 px")
    p.add_argument("--win-mult", type=int, default=6,
                   help="ell window = win_mult * patch px; censoring at half")
    p.add_argument("--purity", type=float, default=0.7,
                   help="min fraction of a cell's pixels on one side to label it")
    p.add_argument("--thing-max-label", type=int, default=90,
                   help="stuffthingmaps boundary: value <= this counts as THING. "
                        "CHECK THE STARTUP HISTOGRAM before trusting it.")
    p.add_argument("--bs", type=int, default=192)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="synthetic things/stuff fixture, 2-3 min on CPU")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.smoke:
        args.size, args.patch, args.width, args.depth = 48, 8, 16, 2
        args.win_mult, args.bs, args.workers = 4, 16, 0
        args.e0_steps, args.e0_lam_gap, args.e0_diag_n = 60, 20, 32
        args.n_images = 64
        if args.seed is None:
            args.seed = 0

    seeds = [args.seed] if args.seed is not None else \
        [int(s) for s in str(args.seeds).split(",") if s != ""]
    device = torch.device("cpu") if args.cpu else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not args.cpu and not args.smoke:
        # A silent CPU fallback turns a two-hour run into a two-day run
        # (2026-09-11: a Colab session without a GPU runtime started E0b on
        # CPU and only the banner said so).  Refuse unless the caller meant it.
        raise SystemExit(
            "[E0b] no CUDA device found and --cpu was not passed. At 128 px this "
            "run takes DAYS on CPU. Select a GPU runtime (Colab: Runtime > Change "
            "runtime type > GPU), remount Drive, re-download COCO on the new VM, "
            "then relaunch. Pass --cpu explicitly if you really mean CPU.")
    os.makedirs(args.outdir, exist_ok=True)

    cfg = ER.Cfg(dataset="coco_stuff", size=args.size, patch=args.patch,
                 width=args.width, depth=args.depth, levels=1,
                 bs=args.bs, lr=args.lr, workers=args.workers,
                 e0_steps=args.e0_steps, e0_lam_gap=args.e0_lam_gap,
                 e0_diag_n=args.e0_diag_n, win_mult=args.win_mult,
                 amp=not args.cpu)

    print("=" * 78)
    print(f"E0b (T4, region labels)  |  device={device}  seeds={seeds}  "
          f"size={cfg.size} patch={cfg.patch} win={cfg.win_mult * cfg.patch}px")
    print(f"outdir = {args.outdir}")
    print("=" * 78)

    summaries = []
    for seed in seeds:
        summaries.append(run_e0b(seed, cfg, args, args.outdir, device,
                                 resume=args.resume))

    verdict = ER.e0_verdict(summaries)
    payload = {"prereg": PREREG_E0B, "config": {**cfg.to_dict(),
               "n_images": args.n_images, "purity": args.purity,
               "thing_max_label": args.thing_max_label, "smoke": bool(args.smoke)},
               "per_seed": summaries, "E0b_verdict": verdict}
    out = os.path.join(args.outdir, "E0b_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=float)
    print("-" * 78)
    print(f"E0b DECISION: {verdict['decision']}  "
          f"(AUC={verdict['auc_mean']:.3f} +/- {verdict['auc_std']:.3f}, "
          f"naive={verdict['auc_naive_mean']:.3f}, "
          f"margin={verdict['margin_over_naive']:.3f}, "
          f"n_seeds={verdict['n_seeds']})")
    if len(seeds) < 3:
        print("NOTE: fewer than 3 seeds -> the decision above is a PILOT "
              "reading, not the pre-registered verdict.")
    if args.smoke:
        print("SMOKE RUN: pipeline only, the numbers mean nothing.")
    print(f"summary -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
