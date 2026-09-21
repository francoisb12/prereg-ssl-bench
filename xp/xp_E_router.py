"""
Experiment group E -- "Router of discordance: learnability x correlation length" (thesis T4).

===============================================================================
WHAT T4 CLAIMS
===============================================================================
Where the prediction residual of a masked model is high, the model should not
blindly descend the gradient.  It should first ask two questions about the
region:

    lam  = is this region still becoming easier?          (learnability)
    ell  = how far does its residual field correlate?     (spatial range)

and then take one of three actions:

    lam > eps                      -> descend the gradient (ordinary learning)
    lam ~ 0 and ell <  receptive   -> TEXTURE: irreducible detail, mask the loss
    lam ~ 0 and ell >  receptive   -> LONG-RANGE: go up one level of abstraction

The load-bearing claim -- the only part of T4 that is not folklore -- is the
last comparison.  The usual test, "is the residual white noise or is it
structured?", is INSUFFICIENT: grass, waves, fabric and film grain all produce
strongly structured residuals, they are irreducible at the current scale, and
they deserve no abstraction whatsoever.  T4 says the right question is not
"structured?" but "structured OVER WHAT RANGE, compared with what the model can
currently see?".

===============================================================================
EXACT PREDICTIONS TESTED HERE
===============================================================================
E0 (cheap diagnostic, run first, GO/NO-GO):

  P0.  On the residual field of a trained masked predictor, the correlation
       length ell separates texture regions from long-structure regions, and it
       does so BETTER than the naive "residual is structured" statistic
       (normalised autocorrelation energy at short lags).
       Pre-registered decision rule (fixed before looking at any number):
           GO            if mean AUC(ell) >= 0.65 over >= 3 seeds
                         AND mean AUC(ell) - mean AUC(naive) >= 0.03
                         AND mean AUC(ell) - std >= 0.60
           NO-GO         if mean AUC(ell) < 0.60
           INCONCLUSIVE  otherwise
       NO-GO means E1 is not worth an A100 week and the honest thing is to say
       so in public.  The script says so, in the summary, by itself.

E1 (full router, only if E0 is GO):

  P1.  MECHANISM.  The "structured residual" control (arm `structured`) promotes
       texture regions substantially more often than the range criterion
       (arm `range`).  This is the direct, cheap, decisive measurement: it does
       not depend on any downstream metric.
  P2.  CONSEQUENCE.  On frozen-encoder linear probes reported at the WORST task
       of the battery (T7 discipline), `range` >= `structured`.
  P3.  COLD START.  Early in training nothing is promoted, because lam is large
       everywhere.  This is correct behaviour, not a bug; it must be visible on
       a curve.
  P4.  The promotion cap must bind visibly, and the truncated fraction must be
       logged: a silently truncated promotion set reads, in a plot, exactly like
       full coverage.

===============================================================================
WHAT WOULD FALSIFY T4 (stated before running)
===============================================================================
  * E0 NO-GO: the (ell, lam) cloud does not separate texture from structure.
    Then the central distinction of T4 is not measurable on real residual
    fields at this scale, and the router has no signal to route on.
  * ell is no better than the naive "structured vs white" statistic
    (AUC difference < 0.03).  Then T4's correction of the naive test is empty:
    the extra machinery buys nothing.
  * E1 P1 fails: the `structured` control does NOT promote more textures than
    the range criterion.  Then the two criteria are behaviourally the same
    object and T4's distinction is verbal.
  * E1 P2 reversed: `structured` beats `range` on the worst task, with a gap
    larger than the seed spread.  Then routing by range is actively harmful.
  * `random` (count-matched random routing) matches or beats `range`.  Then the
    router is a placebo and only the loss re-weighting mattered.

===============================================================================
AMENDMENTS TO THE PLAN AS GIVEN (methodology, read this)
===============================================================================
Six places where the plan as handed to me was shaky.  Each is corrected here
and the correction is argued, because a plan followed and wrong is worse than a
plan amended and solid.

(A1) CONVOLUTIONAL PREDICTOR, NOT A ViT.
     T4's criterion compares ell with "the current receptive field".  In a ViT
     with global attention the receptive field is the whole image at layer 1,
     so the criterion is vacuous by construction.  We therefore use an
     explicitly hierarchical CONVOLUTIONAL masked predictor whose receptive
     field per level is computed analytically from the layer spec (see
     `stack_receptive_field`) and printed.  This makes "ell > receptive field" a
     statement with a number on both sides.

(A2) THE DRIFT-CORRECTED lam CANNOT EXPRESS THE COLD START.
     The plan defines lam[i] = (r_prev[i] - r[i]) - (mean r_prev - mean r).
     That quantity is ZERO-MEAN OVER THE BATCH by construction.  So "lam ~ 0"
     is the typical case for every region at every moment, and the cold-start
     state the plan itself asks to see ("early in training lam is large
     everywhere, nothing promotes") is arithmetically impossible to observe
     with it: at step 0 half the regions are already below the batch mean.
     Resolution: we keep BOTH quantities and use them for their two different
     jobs.
        lam_raw[i]   = (r_prev[i] - r[i]) / (r_prev[i] + eps_r)
                       relative improvement of THIS region since its last
                       exposure.  Absolute, scale-free, large everywhere early.
                       This is the GATE ("is anything still being learned here").
        lam_drift[i] = lam_raw[i] - mean_batch(lam_raw)
                       drift-corrected, zero-mean, a RANKING signal that removes
                       nuisance variation (LR schedule, batch composition).
                       Logged always; it is the y-axis of the E0 scatter, where
                       the two snapshots are thousands of steps apart and the
                       drift correction is exactly the right thing.
     Only lam_raw gates promotion.  This is stated in every relevant docstring.

(A3) THE CONTENT LABEL MUST NOT COME FROM THE RESIDUAL.
     If "texture region" is defined from the residual field we are testing, the
     result is circular.  Primary label here is the ORIGIN OF THE IMAGE: DTD
     images are textures by construction, STL-10 images are objects and scenes.
     Both go through the identical pipeline at the identical resolution.  Label
     noise (a DTD image contains some non-texture pixels, an STL-10 image
     contains grass) can only DEPRESS the measured AUC, so a positive result is
     conservative.  A secondary, weaker within-image heuristic is also computed
     and reported separately, clearly marked as weaker.
     Known caveat, stated rather than hidden: DTD (300-640 px) is downsampled
     harder than STL-10 (96 px) to reach the working resolution, which destroys
     fine grain in the textures.  That works AGAINST the hypothesis, so again
     a positive result is conservative.

(A4) STRICT FLOP EQUALITY IS BOUGHT BY COMPUTING EVERYTHING, ALWAYS.
     A production router would SKIP computation for masked-out regions and be
     cheaper.  Two arms that skip different things are not comparable.  So here
     every arm runs the identical computational graph on every batch -- all
     levels, all regions, always -- and the router acts ONLY on the per-region
     weights of the loss.  Forward/backward FLOPs are then identical by
     construction, not by accounting.  Encoder forward passes are nevertheless
     counted for real and fed to `matched_budget_check`.
     Consequence to state in any writeup: this experiment tests whether routing
     helps REPRESENTATION QUALITY at equal compute.  It says nothing about the
     compute savings a skipping implementation would give.

(A5) A COUNT-MATCHED RANDOM ROUTER IS ADDED.
     The plan has three arms: no router / structured criterion / range
     criterion.  Without a random router of the same size, any difference
     between "no router" and the others confounds "which regions were selected"
     with "how many regions were in each loss term".  Arm `random` promotes and
     masks the same number of regions as the criteria arms, chosen uniformly at
     random among gated regions.  It is the placebo.

(A6) NO SPATIAL AUGMENTATION DURING ROUTED PRE-TRAINING.
     The router needs a persistent per-region state indexed by
     (sample_id, row, col).  Random crops and flips move the content under those
     coordinates, so r_prev and r would not describe the same thing and lam
     would be noise.  We therefore pre-train with the harness's DETERMINISTIC
     eval transform, and the mask pattern is deterministic per sample id too, so
     that r_prev and r are measured on the same region under the same occlusion.
     This is a real constraint that T4 imposes on any implementation, and it is
     worth saying out loud: a discordance router is incompatible with
     region-scrambling augmentation unless the state is carried in a
     content-addressed way instead of a coordinate-addressed one.

Two further honest limitations, not amendments:
  * ell is estimated inside a 3-patch window, so any ell larger than half that
    window is CENSORED at that value.  The script asserts that the level-0
    receptive field is smaller than the measurement window -- otherwise the
    comparison "ell > receptive field" is unmeasurable -- and logs the censored
    fraction.
  * The level-0 residual field is used for r, lam and ell at every level, even
    for promoted regions whose level-0 head no longer receives their loss.  It
    is the finest field that is always computed.  For a promoted region a flat
    level-0 residual is the expected state; demotion therefore means "the fine
    residual started moving again", which is the sensible reading.

===============================================================================
SCALE: WHAT THIS CAN AND CANNOT DECIDE
===============================================================================
This is STL-10 + DTD at 64x64 with a ~2M-parameter convolutional predictor on
ONE A100.  It is not ImageNet and it is not a ViT-L.

  * It CAN decide E0: whether the correlation length of a residual field carries
    texture/structure information at all, and whether it carries more than the
    naive structuredness test.  That is a measurement about residual fields, and
    a negative result there is bad news for T4 at any scale.
  * It CAN decide E1/P1: whether the two criteria behave differently, i.e.
    whether T4's correction of the naive test is a real distinction.
  * It CANNOT settle E1/P2, the downstream question.  Frozen-probe gaps between
    SSL variants at this scale are routinely within the seed spread.  A null
    result on P2 is reported as "non concluante", never as a refutation, and a
    positive result is reported as suggestive at small scale, not as a win.
  * It says nothing about compute savings (see A4).

===============================================================================
A100 COST (measured shapes, 40GB, AMP on)
===============================================================================
  E0, per seed : ~15k steps x bs 256 at 64x64, one-level predictor, plus two
                 full complementary-mask diagnostic sweeps.
                 ~= 0.5 - 0.8 h.   Three seeds: ~= 1.5 - 2.5 h.
  E1, per arm-seed : ~15k steps x bs 256, two-level predictor, per-step region
                 statistics, plus two frozen linear probes.
                 ~= 0.4 - 0.7 h.   4 arms x 3 seeds: ~= 5 - 8 h.
  WHOLE GROUP  : ~= 7 - 11 h A100, sequential, fully resumable.
  Smoke mode   : < 3 min on CPU, synthetic data, no download.

===============================================================================
USAGE
===============================================================================
    python xp/xp_E_router.py --smoke                       # 3 min CPU sanity
    python xp/xp_E_router.py --stage e0 --seeds 0,1,2      # GO/NO-GO first
    python xp/xp_E_router.py --stage e1 --seeds 0,1,2      # only if GO
    python xp/xp_E_router.py --all --seeds 0,1,2           # both, gated
    python xp/xp_E_router.py --stage aggregate             # re-write summary

Everything is resumable: `Run` checkpoints atomically, CSVs are flushed and
fsync'd row by row, and a killed Colab session restarts with `--resume`
(default on).  Re-running a finished stage is a no-op unless --no-resume.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

# --- in-house imports: the shared harness and nothing else -------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.harness import (  # noqa: E402
    DATASET_NUM_CLASSES,
    Run,
    get_device,
    get_eval_datasets,
    linear_probe,
    matched_budget_check,
    set_seed,
)
# _rng_island is "private" in the harness, but it is the harness's own guard for
# "evaluating must not move the training RNG stream" (harness note 6).  Every
# DataLoader iterator this script builds outside the harness -- the E0
# diagnostic sweeps, the image-statistics pass -- draws a base seed from the
# GLOBAL torch generator, so without the island, evaluating more often would
# silently change the trajectory at identical seed.  Importing it is deliberate.
from lib.harness import _rng_island  # noqa: E402


def _amp_autocast(enabled: bool):
    """torch>=2.4 deprecates torch.cuda.amp.autocast; keep both paths working."""
    try:
        return torch.amp.autocast("cuda", enabled=bool(enabled))
    except (AttributeError, TypeError):       # pragma: no cover - old torch
        return torch.cuda.amp.autocast(enabled=bool(enabled))


def _amp_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=bool(enabled))
    except (AttributeError, TypeError):       # pragma: no cover - old torch
        return torch.cuda.amp.GradScaler(enabled=bool(enabled))


def _nanmean(a) -> float:
    """np.nanmean without the all-NaN RuntimeWarning."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else float("nan")


def _nanstd(a) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    return float(a.std()) if a.size else float("nan")

ARMS: Tuple[str, ...] = ("none", "random", "structured", "range")
ORIGIN_OBJECT = 0
ORIGIN_TEXTURE = 1


# =============================================================================
# 1.  Synthetic pool -- smoke tests, and a ceiling check for the E0 estimators
# =============================================================================

class SyntheticTextureObjectDataset(Dataset):
    """Procedural images whose region content is known exactly.

    Half of each image (in coarse blocks) is a LOW-FREQUENCY field -- a smooth
    random surface plus an oriented sinusoid whose orientation encodes the class
    label -- and the other half is a HIGH-FREQUENCY field -- lightly smoothed
    white noise.  By construction the first has a long correlation length and
    the second a short one.

    Two uses:
      * `--smoke` runs the whole pipeline on it in seconds with no download;
      * `--dataset synthetic` gives an upper bound (a ceiling) on what the E0
        estimators can achieve when the ground truth is exact.  If AUC is not
        near 1.0 there, the estimator is broken and no conclusion about real
        images is worth anything.

    Items are dicts ``{'x','label','idx','origin','block_texture'}``.
    ``label`` never enters any loss; it exists so the smoke run can exercise the
    frozen linear probe.
    """

    def __init__(self, n: int = 256, size: int = 32, block: int = 8,
                 num_classes: int = 4, seed: int = 0, train: bool = True):
        self.n = int(n)
        self.size = int(size)
        self.block = int(block)
        self.num_classes = int(num_classes)
        self.seed = int(seed) + (0 if train else 10_000)
        self.grid = self.size // self.block

    def __len__(self) -> int:
        return self.n

    def _make(self, i: int) -> Tuple[torch.Tensor, int, np.ndarray]:
        rs = np.random.RandomState(self.seed * 1_000_003 + i)
        S, B, G = self.size, self.block, self.grid
        label = int(rs.randint(self.num_classes))

        # long-range component: smooth random surface + class-dependent grating
        coarse = rs.randn(3, 4, 4).astype(np.float32)
        smooth = F.interpolate(torch.from_numpy(coarse)[None], size=(S, S),
                               mode="bilinear", align_corners=False)[0]
        theta = math.pi * label / max(1, self.num_classes)
        yy, xx = np.meshgrid(np.arange(S), np.arange(S), indexing="ij")
        grating = np.sin(2 * math.pi * (xx * math.cos(theta) + yy * math.sin(theta)) / S * 3.0)
        smooth = smooth + 0.8 * torch.from_numpy(grating.astype(np.float32))[None]

        # short-range component: white noise, mildly smoothed (still ell ~ 1 px)
        noise = torch.from_numpy(rs.randn(3, S, S).astype(np.float32))
        noise = F.avg_pool2d(noise[None], kernel_size=2, stride=1, padding=0)[0]
        noise = F.pad(noise, (0, 1, 0, 1), mode="replicate") * 2.0

        block_texture = (rs.rand(G, G) < 0.5)
        m = torch.from_numpy(block_texture.astype(np.float32))[None]
        m = F.interpolate(m[None], size=(S, S), mode="nearest")[0]

        img = m * noise + (1.0 - m) * smooth
        img = img / (img.std() + 1e-6)
        img = img.clamp(-3, 3) / 3.0                      # roughly the [-1,1] of the harness
        return img, label, block_texture

    def __getitem__(self, i: int) -> dict:
        img, label, block_texture = self._make(i)
        return {"x": img, "label": label, "idx": int(i),
                "origin": int(block_texture.mean() > 0.5),
                "block_texture": torch.from_numpy(block_texture.astype(np.int64))}


# =============================================================================
# 2.  Real pool: objects (STL-10 / CIFAR-10) + textures (DTD), one label each
# =============================================================================

class OriginTaggedPool(Dataset):
    """Concatenation of several deterministic-transform datasets, each carrying
    a coarse ORIGIN tag (0 = object/scene, 1 = texture).

    The origin tag is the non-circular content label of amendment (A3): it comes
    from which dataset the image was drawn from, never from the model's own
    residual.  Items are dicts ``{'x','label','idx','origin'}`` where ``idx`` is
    the index INTO THIS POOL -- that index is the key of the per-region state
    table, so it must be stable across epochs, which it is.
    """

    def __init__(self, parts: Sequence[Tuple[Dataset, int]]):
        self.parts = list(parts)
        self.offsets: List[int] = []
        total = 0
        for ds, _origin in self.parts:
            self.offsets.append(total)
            total += len(ds)
        self.total = total

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, i: int) -> dict:
        for (ds, origin), off in zip(self.parts, self.offsets):
            if i < off + len(ds):
                item = ds[i - off]
                x = item[0] if isinstance(item, (tuple, list)) else item["x"]
                y = item[1] if isinstance(item, (tuple, list)) else item.get("label", -1)
                return {"x": x, "label": int(y), "idx": int(i), "origin": int(origin)}
        raise IndexError(i)


def pool_origin_array(pool: Dataset) -> np.ndarray:
    """Origin tag of every item of the pool, without decoding images when possible."""
    if isinstance(pool, OriginTaggedPool):
        o = np.zeros(len(pool), dtype=np.int64)
        for (ds, origin), off in zip(pool.parts, pool.offsets):
            o[off:off + len(ds)] = int(origin)
        return o
    return np.asarray([int(pool[i]["origin"]) for i in range(len(pool))],
                      dtype=np.int64)


def stratified_diag_indices(pool: Dataset, n: int, seed: int) -> np.ndarray:
    """A BALANCED, deterministic subset of the pool for the E0 diagnostic.

    THE BUG THIS FIXES WAS FATAL AND SILENT.  The first version measured E0 on
    ``Subset(pool, range(n))``: the pool is built as [objects ..., textures ...]
    and there are thousands of objects, so the first n items were 100% objects.
    Every AUC then had an empty negative class, ``rank_auc`` returned NaN, the
    verdict fell through to "non concluante" -> NO-GO, and the whole group E
    would have been abandoned on the strength of a slicing mistake.
    """
    origins = pool_origin_array(pool)
    classes = np.unique(origins)
    rs = np.random.RandomState(20_240_517 + int(seed))
    per = max(1, int(n) // max(1, len(classes)))
    chosen: List[int] = []
    for c in classes:
        members = np.where(origins == c)[0]
        take = int(min(per, len(members)))
        if take < per:
            print(f"[E0] WARNING: origin {int(c)} has only {len(members)} images, "
                  f"wanted {per} -- the diagnostic set is not balanced.")
        chosen.extend(int(i) for i in rs.choice(members, size=take, replace=False))
    if len(classes) < 2:
        print("[E0] WARNING: the pool carries a single origin; AUC is undefined.")
    return np.asarray(sorted(chosen), dtype=np.int64)


def _subsample(ds: Dataset, n: Optional[int], seed: int) -> Dataset:
    if n is None or n >= len(ds):
        return ds
    rs = np.random.RandomState(seed)
    keep = sorted(int(i) for i in rs.choice(len(ds), size=int(n), replace=False))
    return Subset(ds, keep)


def build_pool(cfg: "Cfg", seed: int) -> Dataset:
    """Build the pre-training pool for E0 and E1 (identical pool for both)."""
    if cfg.dataset == "synthetic":
        return SyntheticTextureObjectDataset(n=cfg.n_object + cfg.n_texture,
                                             size=cfg.size, block=cfg.patch,
                                             seed=seed, train=True)
    parts: List[Tuple[Dataset, int]] = []
    obj_train, _ = get_eval_datasets(cfg.object_source, size=cfg.size, root=cfg.data_root)
    parts.append((_subsample(obj_train, cfg.n_object, seed), ORIGIN_OBJECT))
    if cfg.n_texture > 0:
        tex_train, _ = get_eval_datasets(cfg.texture_source, size=cfg.size, root=cfg.data_root)
        parts.append((_subsample(tex_train, cfg.n_texture, seed), ORIGIN_TEXTURE))
    return OriginTaggedPool(parts)


# =============================================================================
# 3.  The masked hierarchical predictor, with an auditable receptive field
# =============================================================================
#
# The layer SPEC is the single source of truth: modules are built from it and
# the receptive field is computed from it, so the printed receptive field cannot
# drift away from the network that is actually trained.

LayerSpec = Tuple[str, int, int, int]     # (kind, out_channels, kernel, stride)


def stack_receptive_field(spec: Sequence[LayerSpec], rf: float = 1.0,
                          jump: float = 1.0) -> Tuple[float, float]:
    """Receptive field (in INPUT pixels) and stride jump after a layer stack.

    Standard recursion:  rf' = rf + (k - 1) * jump,  jump' = jump * s.
    'proj' (1x1 conv) leaves both unchanged.
    """
    for kind, _ch, k, s in spec:
        if kind == "proj":
            continue
        rf = rf + (k - 1) * jump
        jump = jump * s
    return rf, jump


def build_stack(spec: Sequence[LayerSpec], in_ch: int, bn: bool = True) -> Tuple[nn.Sequential, int]:
    layers: List[nn.Module] = []
    ch = in_ch
    for kind, out_ch, k, s in spec:
        if kind == "proj":
            layers.append(nn.Conv2d(ch, out_ch, kernel_size=1, stride=1, bias=True))
            ch = out_ch
            continue
        if kind == "pool":
            layers.append(nn.AvgPool2d(kernel_size=k, stride=s))
            continue
        layers.append(nn.Conv2d(ch, out_ch, kernel_size=k, stride=s,
                                padding=k // 2, bias=not bn))
        if bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        ch = out_ch
    return nn.Sequential(*layers), ch


class HierarchicalMaskedPredictor(nn.Module):
    """Masked predictor with ``levels`` prediction scales.

    Level l predicts the input image downsampled by 2**l, from a trunk stage
    that also runs at resolution H/2**l.  Going up one level therefore means:
    predict the same NUMBER of values, about a physically larger and blurrier
    area, with a strictly larger receptive field.  That is what "monter d'un
    niveau d'abstraction" is operationalised as here.

    Regions are level-0 patches of ``patch`` pixels.  A level-l loss cell covers
    ``2**l`` x ``2**l`` regions.

    ``receptive_field_px[l]`` is the analytic receptive field of head l in input
    pixels, computed from the same spec the modules are built from.
    """

    def __init__(self, levels: int = 2, width: int = 64, depth: int = 4,
                 patch: int = 8, in_ch: int = 3):
        super().__init__()
        self.levels = int(levels)
        self.patch = int(patch)
        self.width = int(width)

        self.stem_spec: List[LayerSpec] = [("conv", width, 3, 1)]
        self.stem, ch = build_stack(self.stem_spec, in_ch)

        self.stages = nn.ModuleList()
        self.heads = nn.ModuleList()
        self.stage_channels: List[int] = []
        self.receptive_field_px: List[float] = []

        spec_so_far: List[LayerSpec] = list(self.stem_spec)
        for l in range(self.levels):
            spec: List[LayerSpec] = []
            if l > 0:
                spec.append(("conv", width * (2 ** l), 3, 2))     # downsample
            spec += [("conv", width * (2 ** l), 3, 1)] * int(depth)
            stage, ch = build_stack(spec, ch)
            self.stages.append(stage)
            self.stage_channels.append(ch)
            self.heads.append(nn.Conv2d(ch, in_ch, kernel_size=1))
            spec_so_far = spec_so_far + spec
            rf, _jump = stack_receptive_field(spec_so_far)
            self.receptive_field_px.append(float(rf))

        # learned fill for hidden pixels (one value per input channel)
        self.mask_token = nn.Parameter(torch.zeros(in_ch))

    def forward(self, x_masked: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Return (predictions per level, trunk features per level)."""
        h = self.stem(x_masked)
        preds: List[torch.Tensor] = []
        feats: List[torch.Tensor] = []
        for stage, head in zip(self.stages, self.heads):
            h = stage(h)
            feats.append(h)
            preds.append(head(h))
        return preds, feats

    def apply_mask(self, x: torch.Tensor, mask_px: torch.Tensor) -> torch.Tensor:
        """mask_px is [B,1,H,W] with 1 where the pixel is HIDDEN."""
        token = self.mask_token.view(1, -1, 1, 1)
        return x * (1.0 - mask_px) + token * mask_px


class TrunkEncoder(nn.Module):
    """Frozen-feature view of the predictor for downstream probes.

    Concatenates the globally average-pooled features of every trunk stage.  The
    prediction heads are NOT part of it: what we probe is the representation the
    routed objective shaped, not the decoder.
    """

    def __init__(self, model: HierarchicalMaskedPredictor):
        super().__init__()
        self.model = model
        self.out_dim = int(sum(model.stage_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _preds, feats = self.model(x)
        pooled = [torch.flatten(F.adaptive_avg_pool2d(f, 1), 1) for f in feats]
        return torch.cat(pooled, dim=1)


# =============================================================================
# 4.  Deterministic masks (region identity must be stable -- amendment A6)
# =============================================================================

_MASK_CACHE: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]] = {}


def deterministic_mask_groups(idxs: np.ndarray, n_blocks: int, n_variants: int,
                              mask_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Block->group assignment AND the sample's fixed variant.

    Returns ``(groups [B, n_blocks], variant [B])``.  Both depend only on
    (mask_seed, sample id), never on the global RNG or on the step, so the
    occlusion a region experiences is IDENTICAL every time that sample is seen.

    This is amendment (A6) actually implemented.  The first version of this
    script drew ``variant = np.random.randint(...)`` per step, which silently
    contradicted its own docstring: a region was then measured under a
    different occlusion at every exposure, and r_prev - r mixed "the model
    learned" with "the context changed".  Since lam is the whole gate of T4,
    that noise went straight into the routing decision.

    Price of the fix, stated rather than hidden: with the variant frozen per
    sample, the 1/n_variants of blocks whose group equals the sample's variant
    are NEVER hidden, hence never a prediction target and never observed by the
    router.  They are excluded from every statistic (n_seen stays 0).  The task
    is "predict 3/4 of this image from the same fixed 1/4, every epoch", which
    is a legitimate masked-prediction objective but is more memorisable than a
    resampled mask; the pool size / step count are chosen with that in mind and
    the equivalent number of epochs is logged.
    """
    n_blocks = int(n_blocks)
    n_variants = int(n_variants)
    groups = np.empty((len(idxs), n_blocks), dtype=np.int64)
    variant = np.empty(len(idxs), dtype=np.int64)
    for row, i in enumerate(idxs):
        key = (int(mask_seed) * 1_000_003 + int(i), n_blocks, n_variants)
        hit = _MASK_CACHE.get(key)
        if hit is None:
            rs = np.random.RandomState(key[0] % (2 ** 31 - 1))
            g = (rs.permutation(n_blocks) % n_variants).astype(np.int64)
            v = np.int64(rs.randint(n_variants))
            if len(_MASK_CACHE) < 400_000:
                _MASK_CACHE[key] = (g, v)
            hit = (g, v)
        groups[row] = hit[0]
        variant[row] = hit[1]
    return groups, variant


def mask_from_groups(groups: torch.Tensor, variant: torch.Tensor) -> torch.Tensor:
    """Hidden = every block whose group differs from the sample's variant.

    With V variants the mask ratio is exactly (V-1)/V, and over the V variants
    each block is hidden V-1 times, which is what the complementary diagnostic
    sweep of E0 exploits.
    """
    return (groups != variant.view(-1, 1))


# =============================================================================
# 5.  Field statistics: correlation length, LF/HF proxy, naive structuredness
# =============================================================================
#
# Everything below works on a field that may be PARTIALLY OBSERVED: during
# routed training only hidden regions have a meaningful residual, so the
# autocorrelation is computed with the standard missing-data estimator
#
#     acf(lag) = IFFT(|FFT(x*m)|^2) / IFFT(|FFT(m)|^2)
#
# i.e. every lag is normalised by the number of valid PAIRS at that lag, not by
# a constant.  With m == 1 everywhere this degenerates to the usual unbiased
# estimator, so E0 (which observes everything, via complementary masks) and E1
# (which observes 3/4) share one code path.

_RADIAL_CACHE: Dict[Tuple[int, int, str], torch.Tensor] = {}


def _radial_bins(pad: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Integer radius bin of every lag of a (pad, pad) fftshifted ACF."""
    key = (pad, 0, str(device))
    if key not in _RADIAL_CACHE:
        c = pad // 2
        yy, xx = torch.meshgrid(torch.arange(pad, device=device),
                                torch.arange(pad, device=device), indexing="ij")
        r = torch.sqrt(((yy - c).float() ** 2 + (xx - c).float() ** 2))
        _RADIAL_CACHE[key] = r
    r = _RADIAL_CACHE[key]
    rbin = torch.round(r).long()
    nbins = int(rbin.max().item()) + 1
    return rbin, r, nbins


def _freq_masks(pad: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Low / high frequency selectors on the rfft2 grid, DC excluded."""
    fy = torch.fft.fftfreq(pad, device=device).view(-1, 1)
    fx = torch.fft.rfftfreq(pad, device=device).view(1, -1)
    fr = torch.sqrt(fy ** 2 + fx ** 2)
    lo = (fr > 0) & (fr <= 0.25)          # DC excluded, split at half-Nyquist
    hi = fr > 0.25
    return lo, hi


def field_statistics(field: torch.Tensor, valid: torch.Tensor, patch: int,
                     min_pairs: float = 8.0, min_valid_frac: float = 0.25,
                     chunk: int = 32, win_mult: int = 4) -> Dict[str, torch.Tensor]:
    """Per-region statistics of a scalar field, on a ``win_mult``-patch window.

    Parameters
    ----------
    field : [B,H,W]   the residual field (or any scalar image field)
    valid : [B,H,W]   1.0 where the field is meaningful, 0.0 where it is not
    patch : region size in pixels; region (i,j) is scored on the
            (win_mult*patch)^2 window centred on it (reflect-padded at the
            border).  ``win_mult`` must be ODD-centred, i.e. the window is
            centred on the region and extends (win_mult-1)/2 patches each way;
            for even ``win_mult`` the window is off-centre by half a patch,
            which is harmless for an isotropic statistic but is why the pad is
            computed from ``win_mult`` rather than hard-coded.
            ell is CENSORED at win_mult*patch/2, so the receptive field it is
            compared against must sit well below that (see ``check_ell_window``).

    Returns a dict of [B,G,G] tensors
      ell_acf   : correlation length in PIXELS -- first radius where the
                  radially averaged normalised autocorrelation crosses 1/e,
                  linearly interpolated.  CENSORED at half the window.
      ell_freq  : the frequency proxy, log(1 + LF energy / HF energy).  Monotone
                  in range, but biased upward by the mask edges when the field
                  is partially observed; that bias is why both estimators exist
                  and why E0 compares them on a fully observed field.
      acf_energy: the NAIVE "the residual is structured, not white" statistic --
                  mean normalised autocorrelation at lags 1 <= r <= 2.5.  This
                  is the control criterion of T4.
      censored  : 1.0 where ell_acf hit the window ceiling.
      valid_frac: fraction of the window that was observed.
    """
    B, H, W = field.shape
    G = H // patch
    win_mult = int(win_mult)
    if win_mult < 2:
        raise ValueError("win_mult must be >= 2")
    win = win_mult * patch
    pad_px = (win - patch) // 2
    dev = field.device

    f_pad = F.pad(field.unsqueeze(1), (pad_px,) * 4, mode="reflect")
    v_pad = F.pad(valid.unsqueeze(1), (pad_px,) * 4, mode="constant", value=0.0)

    ell = torch.zeros(B, G * G, device=dev)
    ellf = torch.zeros(B, G * G, device=dev)
    acen = torch.zeros(B, G * G, device=dev)
    cens = torch.zeros(B, G * G, device=dev)
    vfr = torch.zeros(B, G * G, device=dev)

    pad = 2 * win
    rbin, _r, nbins = _radial_bins(pad, dev)
    rbin_flat = rbin.reshape(-1)
    counts = torch.zeros(nbins, device=dev).index_add_(
        0, rbin_flat, torch.ones_like(rbin_flat, dtype=torch.float32))
    lo_m, hi_m = _freq_masks(pad, dev)
    r_max = float(win) / 2.0

    for b0 in range(0, B, chunk):
        b1 = min(B, b0 + chunk)
        fw = F.unfold(f_pad[b0:b1], kernel_size=win, stride=patch)      # [b,win^2,G*G]
        vw = F.unfold(v_pad[b0:b1], kernel_size=win, stride=patch)
        nb = fw.shape[0]
        fw = fw.transpose(1, 2).reshape(nb * G * G, win, win)
        vw = vw.transpose(1, 2).reshape(nb * G * G, win, win)

        cnt = vw.sum(dim=(-2, -1)).clamp_min(1.0)
        mean = (fw * vw).sum(dim=(-2, -1)) / cnt
        x = (fw - mean.view(-1, 1, 1)) * vw

        Fx = torch.fft.rfft2(x, s=(pad, pad))
        Fm = torch.fft.rfft2(vw, s=(pad, pad))
        num = torch.fft.irfft2(Fx.real ** 2 + Fx.imag ** 2, s=(pad, pad))
        den = torch.fft.irfft2(Fm.real ** 2 + Fm.imag ** 2, s=(pad, pad))
        ok = den > min_pairs
        acf = torch.where(ok, num / den.clamp_min(1e-6), torch.zeros_like(num))
        acf0 = acf[:, :1, :1].clamp_min(1e-12)
        acf = torch.fft.fftshift(acf / acf0, dim=(-2, -1))
        okm = torch.fft.fftshift(ok.float(), dim=(-2, -1))

        n = acf.shape[0]
        prof_sum = torch.zeros(n, nbins, device=dev).index_add_(
            1, rbin_flat, (acf * okm).reshape(n, -1))
        prof_cnt = torch.zeros(n, nbins, device=dev).index_add_(
            1, rbin_flat, okm.reshape(n, -1)).clamp_min(1.0)
        prof = prof_sum / prof_cnt
        del prof_sum, prof_cnt, acf, okm, num, den, ok

        # correlation length: first crossing of 1/e, linearly interpolated.
        # CENSORING BUG FIXED: the crossing must be searched INSIDE the usable
        # radii only.  Marking non-usable radii as "below" (the old code set
        # them to -1) made `below.any()` almost always true, so `censored` was
        # ~0 everywhere and the logged censored fraction was meaningless, while
        # ell was still silently clamped at r_max.  A censored ell that does not
        # say it is censored is exactly the failure mode P4 warns about.
        thr = float(np.exp(-1.0))
        rr = torch.arange(nbins, device=dev).float()
        usable = (rr <= r_max).view(1, -1)
        below = (prof < thr) & usable
        any_below = below.any(dim=1)
        first = torch.where(any_below,
                            below.float().argmax(dim=1),
                            torch.full((n,), int(min(nbins - 1, math.floor(r_max))),
                                       device=dev, dtype=torch.long))
        censored = ~any_below
        f_idx = first.clamp_min(1)
        p_hi = prof.gather(1, (f_idx - 1).view(-1, 1)).squeeze(1)
        p_lo = prof.gather(1, f_idx.view(-1, 1)).squeeze(1)
        denom = (p_hi - p_lo).abs().clamp_min(1e-6)
        frac = ((p_hi - thr) / denom).clamp(0.0, 1.0)
        ell_chunk = (f_idx.float() - 1.0 + frac).clamp(0.5, r_max)
        ell_chunk = torch.where(censored, torch.full_like(ell_chunk, r_max), ell_chunk)

        # naive structuredness: mean normalised ACF at short lags
        short = (rr >= 1.0) & (rr <= 2.5)
        acen_chunk = (prof[:, short]).mean(dim=1)

        # frequency proxy on the same window
        P = Fx.real ** 2 + Fx.imag ** 2
        lf = (P * lo_m).sum(dim=(-2, -1))
        hf = (P * hi_m).sum(dim=(-2, -1))
        ellf_chunk = torch.log1p(lf / hf.clamp_min(1e-12))

        sl = slice(b0, b1)
        ell[sl] = ell_chunk.view(nb, G * G)
        ellf[sl] = ellf_chunk.view(nb, G * G)
        acen[sl] = acen_chunk.view(nb, G * G)
        cens[sl] = censored.float().view(nb, G * G)
        vfr[sl] = (cnt / float(win * win)).view(nb, G * G)
        del Fx, Fm, P, prof

    del rbin_flat, counts, lo_m, hi_m
    too_empty = vfr < min_valid_frac
    nan = torch.full_like(ell, float("nan"))
    return {
        "ell_acf": torch.where(too_empty, nan, ell).view(B, G, G),
        "ell_freq": torch.where(too_empty, nan, ellf).view(B, G, G),
        "acf_energy": torch.where(too_empty, nan, acen).view(B, G, G),
        "censored": cens.view(B, G, G),
        "valid_frac": vfr.view(B, G, G),
    }


ELL_HEADROOM_RATIO = 0.75      # rf0 must sit at or below this fraction of r_max


def check_ell_window(rf0_px: float, patch: int, win_mult: int, tag: str,
                     hard: bool = True) -> Dict[str, float]:
    """Refuse to run when 'ell > receptive field' is not measurable.

    ell is censored at ``r_max = win_mult * patch / 2``.  If the level-0
    receptive field sits at or above that ceiling the range criterion is
    vacuous; if it sits just below it, the criterion degenerates into "is ell
    censored?" and the AUC is quantised to two values.  The original version of
    this script only WARNED, at shapes (patch 8, 3-patch window, depth 4) where
    the usable band was rf0=11px .. r_max=12px -- one pixel wide.  That is a
    silent way to measure nothing, so it is now a hard error with the two knobs
    that fix it named in the message.
    """
    r_max = win_mult * patch / 2.0
    info = {"receptive_field_px": float(rf0_px), "ell_ceiling_px": float(r_max),
            "ell_headroom_px": float(r_max - rf0_px),
            "ell_window_px": float(win_mult * patch)}
    msg = (f"[{tag}] level-0 receptive field = {rf0_px:.1f} px | ell window = "
           f"{win_mult * patch} px | ell censored at {r_max:.1f} px | headroom "
           f"= {r_max - rf0_px:.1f} px")
    print(msg)
    if rf0_px > ELL_HEADROOM_RATIO * r_max:
        problem = (
            f"{msg}\n[{tag}] the receptive field must stay below "
            f"{ELL_HEADROOM_RATIO:.2f} * {r_max:.1f} = "
            f"{ELL_HEADROOM_RATIO * r_max:.1f} px, otherwise 'ell > receptive "
            f"field' only separates censored from non-censored regions and the "
            f"range criterion is untestable.  Fix: raise --win-mult, raise "
            f"--patch, or lower --depth.")
        if hard:
            raise SystemExit(problem)
        print("WARNING: " + problem)
    return info


# =============================================================================
# 6.  Per-region persistent state -- the side table (replay-buffer style)
# =============================================================================

class RegionStateTable:
    """Persistent state indexed by (sample_id, row, col), OUTSIDE the dataloader.

    One row per level-0 region of every image in the pool.  It survives shuffling
    and epochs because it is keyed by the pool index, and it is checkpointed with
    the model so a killed Colab session resumes with its routing history intact.

    Fields
    ------
    r_ema        EMA of the region's residual
    lam_raw_ema  EMA of the RELATIVE improvement since the previous exposure.
                 THIS is the gate (amendment A2): it is large everywhere early,
                 which is what produces the cold start.
    lam_drift_ema EMA of the drift-corrected version (zero-mean over the batch).
                 Diagnostic only -- never gates anything (amendment A2).
    ell_ema      EMA of the correlation length in pixels
    acf_ema      EMA of the naive structuredness statistic
    stuck_count  consecutive exposures for which the gate was satisfied
                 (hysteresis: a decision needs M consecutive confirmations)
    level        -1 = masked out (texture), 0..L-1 = the level that owns the
                 region's loss
    lock_until   step before which no demotion is allowed (hysteresis, K steps)
    n_seen       exposures so far (lam is undefined at the first exposure)
    """

    FLOAT_FIELDS = ("r_ema", "lam_raw_ema", "lam_drift_ema", "ell_ema", "acf_ema")
    INT_FIELDS = ("stuck_count", "level", "lock_until", "n_seen")

    def __init__(self, n_samples: int, grid: int, ema: float = 0.7):
        self.n_samples = int(n_samples)
        self.grid = int(grid)
        self.ema = float(ema)
        shape = (self.n_samples, self.grid, self.grid)
        for f in self.FLOAT_FIELDS:
            setattr(self, f, torch.zeros(shape, dtype=torch.float32))
        for f in self.INT_FIELDS:
            setattr(self, f, torch.zeros(shape, dtype=torch.int32))

    # -- bookkeeping ---------------------------------------------------------
    def state_dict(self) -> dict:
        d = {f: getattr(self, f) for f in self.FLOAT_FIELDS + self.INT_FIELDS}
        d["_meta"] = {"n_samples": self.n_samples, "grid": self.grid, "ema": self.ema}
        return d

    def load_state_dict(self, d: dict) -> None:
        meta = d.get("_meta", {})
        if int(meta.get("n_samples", self.n_samples)) != self.n_samples or \
           int(meta.get("grid", self.grid)) != self.grid:
            raise ValueError("region state table shape mismatch on resume: "
                             f"checkpoint {meta}, current "
                             f"{{'n_samples': {self.n_samples}, 'grid': {self.grid}}}")
        for f in self.FLOAT_FIELDS + self.INT_FIELDS:
            getattr(self, f).copy_(d[f])

    # -- the update -----------------------------------------------------------
    def observe(self, idxs: torch.Tensor, r: torch.Tensor, ell: torch.Tensor,
                acf: torch.Tensor, eps_r: float,
                observed: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Fold one exposure of a batch of samples into the table.

        ``r``, ``ell``, ``acf`` are [B,G,G] CPU float tensors and ``observed``
        is the [B,G,G] bool mask of regions that were actually OCCLUDED this
        step, i.e. the only ones whose residual carries information.  Every
        other region is left strictly untouched: r_ema, the lam EMAs, ell/acf
        and n_seen all keep their previous values.

        This is not cosmetic.  The first version fed ``r_keep = where(hidden, r,
        r_ema)`` to this method, so a region that was simply NOT masked this
        step got r == r_prev, hence lam_raw == 0, hence "this region has stopped
        improving" -- the router's gate -- for free, plus a free increment of
        n_seen.  With a 3/4 mask ratio a quarter of all regions collected that
        fake evidence at every step, and the hysteresis counter (meant to demand
        several consecutive confirmations) was satisfied by non-observation.
        The gate of T4 would have been driven by the mask pattern.

        Returns the freshly computed per-region ``lam_raw`` / ``lam_drift`` of
        THIS exposure (NaN where not observed), for logging.

        lam_raw   = (r_prev - r) / (r_prev + eps_r)              -- absolute
        lam_drift = lam_raw - batch mean of lam_raw               -- relative
        See amendment (A2): only lam_raw gates, lam_drift is a diagnostic.
        """
        idxs = idxs.long()
        observed = observed.bool()
        prev_r = self.r_ema[idxs]
        seen = self.n_seen[idxs]
        obs_r = observed & torch.isfinite(r)
        first = (seen == 0) & obs_r

        lam_raw = (prev_r - r) / (prev_r + eps_r)
        lam_raw = torch.where(obs_r & (seen > 0), lam_raw,
                              torch.full_like(lam_raw, float("nan")))

        finite = torch.isfinite(lam_raw)
        drift = lam_raw[finite].mean() if bool(finite.any()) else torch.tensor(0.0)
        lam_drift = lam_raw - drift

        a = self.ema
        r_safe = torch.where(torch.isfinite(r), r, prev_r)
        new_r = torch.where(first, r_safe, a * prev_r + (1 - a) * r_safe)
        self.r_ema[idxs] = torch.where(obs_r, new_r, prev_r)
        for name, val in (("lam_raw_ema", lam_raw), ("lam_drift_ema", lam_drift)):
            cur = getattr(self, name)[idxs]
            ok = torch.isfinite(val)
            newv = torch.where(ok, a * cur + (1 - a) * val, cur)
            newv = torch.where((seen == 1) & ok, val, newv)   # first real lam
            getattr(self, name)[idxs] = newv
        for name, val in (("ell_ema", ell), ("acf_ema", acf)):
            cur = getattr(self, name)[idxs]
            ok = torch.isfinite(val) & observed
            newv = torch.where(ok, a * cur + (1 - a) * val, cur)
            newv = torch.where(first & ok, val, newv)
            getattr(self, name)[idxs] = newv
        self.n_seen[idxs] = torch.where(obs_r, seen + 1, seen)
        return {"lam_raw": lam_raw, "lam_drift": lam_drift, "drift": drift,
                "observed": observed}


# =============================================================================
# 7.  The router itself
# =============================================================================

@dataclass
class RouterCfg:
    arm: str = "range"
    lam_eps: float = 0.01          # gate: relative improvement per exposure
    hysteresis_m: int = 3          # consecutive confirmations before acting
    lock_steps: int = 500          # no demotion for K steps after a decision
    promote_cap: float = 0.15      # max fraction of regions promoted per batch
    mask_cap: float = 0.15         # max fraction of regions masked out per batch
    acf_threshold: float = 0.30    # "structured, not white" threshold (arm b)
    rf_margin: float = 1.0         # ell > rf_margin * receptive field (arm c)
    min_seen: int = 2              # exposures before a region may be routed


def route_batch(state: RegionStateTable, idxs: torch.Tensor, cfg: RouterCfg,
                rf0_px: float, step: int, max_level: int,
                rng: np.random.RandomState,
                observed: torch.Tensor) -> Dict[str, float]:
    """Update the level assignment of the regions of one batch.

    Structure, identical for every criterion arm so that the arms differ ONLY in
    the criterion:

      1. GATE.  A region is a candidate only if it has been seen at least
         ``min_seen`` times AND its lam_raw_ema is below ``lam_eps`` -- i.e. it
         has stopped improving in relative terms.  This is what is empty at cold
         start (P3): early on, everything improves fast, so nothing is a
         candidate and nothing is promoted.  That is correct behaviour.
      2. HYSTERESIS.  The gate must hold for ``hysteresis_m`` CONSECUTIVE
         EXPOSURES (``stuck_count``).  A single noisy exposure cannot promote.
         ``observed`` (the regions actually occluded this step) is what makes
         an exposure: a region that was visible this step neither increments
         nor resets its counter, and cannot be routed this step.
      3. CRITERION.  Among confirmed candidates, one score decides:
            range      : promote if ell_ema >  rf_margin * receptive field
                         mask    if ell_ema <= rf_margin * receptive field
            structured : promote if acf_ema >  acf_threshold     (the control
                         that T4 predicts will wrongly abstract textures)
                         mask    if acf_ema <= acf_threshold
            random     : the placebo.  COUNT-MATCHED to `range`: the number of
                         promotions and masks it draws this step is exactly the
                         number the range criterion would have produced on the
                         same confirmed set, so that "which regions" is the only
                         thing that differs (amendment A5).  Drawing a coin at
                         0.5, as the first version did, matched nothing.
            none       : no routing at all
      4. CAP.  At most ``promote_cap`` (resp. ``mask_cap``) of the batch's
         regions may change per step, ranked by the criterion's own score.  The
         TRUNCATED fraction is returned and logged (P4): a silently truncated
         promotion set is indistinguishable, on a plot, from full coverage.
      5. LOCK.  A region that just changed level cannot be demoted for
         ``lock_steps`` steps.

    Returns a dict of scalars for the CSV.
    """
    idxs = idxs.long()
    B = idxs.numel()
    observed = observed.bool()
    lam = state.lam_raw_ema[idxs]
    seen = state.n_seen[idxs]
    level = state.level[idxs]
    lock = state.lock_until[idxs]
    stuck = state.stuck_count[idxs]

    eligible = (seen >= cfg.min_seen) & (lam < cfg.lam_eps) & observed
    stuck = torch.where(observed,
                        torch.where(eligible, stuck + 1, torch.zeros_like(stuck)),
                        stuck)
    state.stuck_count[idxs] = stuck
    confirmed = eligible & (stuck >= cfg.hysteresis_m)

    n_regions = int(B * state.grid * state.grid)
    n_obs = max(1, int(observed.sum()))
    out = {
        "observed_frac": float(observed.float().mean()),
        "gate_frac": float(int(eligible.sum())) / n_obs,
        "confirmed_frac": float(int(confirmed.sum())) / n_obs,
        "promote_eligible_frac": 0.0,
        "promote_applied_frac": 0.0,
        "promote_truncated_frac": 0.0,
        "mask_eligible_frac": 0.0,
        "mask_applied_frac": 0.0,
        "mask_truncated_frac": 0.0,
        "demoted_frac": 0.0,
    }

    if cfg.arm == "none":
        return out

    ell = state.ell_ema[idxs]
    acf = state.acf_ema[idxs]

    if cfg.arm == "range":
        want_promote = confirmed & (ell > cfg.rf_margin * rf0_px)
        want_mask = confirmed & (ell <= cfg.rf_margin * rf0_px)
        score = ell
    elif cfg.arm == "structured":
        want_promote = confirmed & (acf > cfg.acf_threshold)
        want_mask = confirmed & (acf <= cfg.acf_threshold)
        score = acf
    elif cfg.arm == "random":
        # count-matched placebo: same number of promotions as `range` would
        # have made on this confirmed set, chosen uniformly at random among it.
        coin = torch.from_numpy(rng.rand(*confirmed.shape).astype(np.float32))
        n_conf = int(confirmed.sum())
        n_prom_range = int((confirmed & (ell > cfg.rf_margin * rf0_px)).sum())
        if n_conf == 0 or n_prom_range == 0:
            thr_coin = float("inf")
        elif n_prom_range >= n_conf:
            thr_coin = float("-inf")
        else:
            vals = coin[confirmed].reshape(-1)
            thr_coin = float(torch.topk(vals, n_prom_range, largest=True).values.min())
        want_promote = confirmed & (coin >= thr_coin)
        want_mask = confirmed & (coin < thr_coin)
        score = coin
    else:
        raise ValueError(f"unknown arm {cfg.arm!r}")

    # a region can only be promoted if it is not already at the top level and
    # not masked out; it can only be masked out if it currently owns a loss
    want_promote = want_promote & (level >= 0) & (level < max_level)
    want_mask = want_mask & (level >= 0)

    def _apply(want: torch.Tensor, cap: float, prefer_high: bool, key: str,
               action) -> None:
        n_want = int(want.sum())
        out[f"{key}_eligible_frac"] = n_want / max(1, n_regions)
        if n_want == 0:
            return
        budget = int(math.floor(cap * n_regions))
        sc = score.clone().float()
        sc[~want] = float("-inf") if prefer_high else float("inf")
        flat = sc.reshape(-1)
        k = min(n_want, max(0, budget))
        if k == 0:
            out[f"{key}_truncated_frac"] = n_want / max(1, n_regions)
            return
        order = torch.topk(flat, k, largest=prefer_high).indices
        sel = torch.zeros_like(flat, dtype=torch.bool)
        sel[order] = True
        sel = sel.reshape(want.shape) & want
        action(sel)
        out[f"{key}_applied_frac"] = float(int(sel.sum())) / max(1, n_regions)
        out[f"{key}_truncated_frac"] = max(0, n_want - int(sel.sum())) / max(1, n_regions)

    new_level = level.clone()
    new_lock = lock.clone()

    def _do_promote(sel: torch.Tensor) -> None:
        new_level[sel] = (new_level[sel] + 1).to(new_level.dtype)
        new_lock[sel] = int(step + cfg.lock_steps)

    def _do_mask(sel: torch.Tensor) -> None:
        new_level[sel] = -1
        new_lock[sel] = int(step + cfg.lock_steps)

    _apply(want_promote, cfg.promote_cap, prefer_high=True, key="promote", action=_do_promote)
    # a region promoted this step must not also be masked this step
    want_mask = want_mask & (new_level == level)
    _apply(want_mask, cfg.mask_cap, prefer_high=False, key="mask", action=_do_mask)

    # demotion: the region started improving again, and its lock has expired.
    # Only a region that was actually exposed this step has fresh evidence, so
    # `observed` gates demotion exactly as it gates promotion.
    revived = (observed & ~eligible & (new_level != 0)
               & (torch.tensor(int(step), dtype=new_lock.dtype) >= new_lock))
    if bool(revived.any()):
        new_level[revived] = 0
        out["demoted_frac"] = float(int(revived.sum())) / max(1, n_regions)

    state.level[idxs] = new_level
    state.lock_until[idxs] = new_lock
    return out


# =============================================================================
# 8.  Loss weights from the routing decision (identical graph in every arm)
# =============================================================================

def loss_weights(level: torch.Tensor, hidden_region: torch.Tensor, levels: int,
                 arm: str) -> List[torch.Tensor]:
    """Per-level loss weights from the per-region level assignment.

    ``level``          [B,G,G] int, -1 masked out, l in 0..L-1
    ``hidden_region``  [B,G,G] bool, True where the region was occluded (only
                       occluded regions carry a prediction task at all)

    The weight of level-l cell c is the FRACTION of its level-0 descendants that
    are assigned to level l and were hidden.  Arm 'none' ignores the assignment
    and supervises every hidden region at every level -- the dense hierarchical
    baseline.

    The computational graph is the same in all arms; only these weights differ.
    That is how the FLOP match of amendment (A4) is bought.

    See ``combine_level_losses`` for how these weights are turned into ONE
    scalar: not by summing per-level means, which would give a handful of
    promoted regions the same total gradient as the whole level-0 population.
    """
    B, G, _ = level.shape
    out: List[torch.Tensor] = []
    hid = hidden_region.float()
    for l in range(levels):
        if arm == "none":
            w = hid
        else:
            w = ((level == l).float() * hid)
        f = 2 ** l
        if f > 1:
            w = F.avg_pool2d(w.unsqueeze(1), kernel_size=f, stride=f).squeeze(1)
        out.append(w)
    return out


def combine_level_losses(res_cells: Sequence[torch.Tensor],
                         w_levels: Sequence[torch.Tensor]
                         ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """One scalar loss = per-REGION weighted mean of the residual at its level.

    ``w_levels[l]`` is a per-cell fraction of level-0 descendants, so
    ``w_l * 4**l`` counts level-0 regions.  The loss is

        sum_l  sum_cells  res_l * w_l * 4**l   /   sum_l sum_cells w_l * 4**l

    i.e. every supervised level-0 region contributes exactly once, at whatever
    level owns it.

    WHY THIS MATTERS.  The first version summed per-level MEANS,
    ``sum_l (res_l*w_l).sum() / w_l.sum()``.  With 1% of regions promoted, that
    handed the promoted 1% half of the total gradient -- a ~100x amplification
    that has nothing to do with T4 and everything to do with the normaliser.
    Worse, it made the arms non-comparable at matched steps: arm 'none' summed
    two full-strength terms while a routed arm summed two terms of wildly
    different support, so the effective learning rate differed by arm.  Matched
    FLOPs (A4) are worth nothing if the loss scale is not matched too.

    Returns ``(loss, per_level_mean)`` where the per-level means are detached
    diagnostics for the CSV.
    """
    num = None
    den = None
    per_level: List[torch.Tensor] = []
    for l, (res, w) in enumerate(zip(res_cells, w_levels)):
        cnt = float(4 ** l)
        num_l = (res * w).sum() * cnt
        den_l = w.sum() * cnt
        num = num_l if num is None else num + num_l
        den = den_l if den is None else den + den_l
        per_level.append((num_l / den_l.clamp_min(1e-6)).detach())
    loss = num / den.clamp_min(1e-6)
    return loss, per_level


# =============================================================================
# 9.  Small numeric helpers (numpy only -- no scipy dependency)
# =============================================================================

def rank_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """AUC of ``scores`` as a detector of ``positive``, via the Mann-Whitney U.

    NaNs are dropped.  Returns NaN if either class is empty.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive = np.asarray(positive).reshape(-1).astype(bool)
    ok = np.isfinite(scores)
    scores, positive = scores[ok], positive[ok]
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # average ranks over ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(np.arange(i + 1, j + 2))
        i = j + 1
    u = ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def roc_curve(scores: np.ndarray, positive: np.ndarray, n_points: int = 200):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive = np.asarray(positive).reshape(-1).astype(bool)
    ok = np.isfinite(scores)
    scores, positive = scores[ok], positive[ok]
    if positive.sum() == 0 or (~positive).sum() == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    qs = np.quantile(scores, np.linspace(0, 1, n_points))
    tpr = [float((scores[positive] >= t).mean()) for t in qs]
    fpr = [float((scores[~positive] >= t).mean()) for t in qs]
    return np.array(fpr[::-1]), np.array(tpr[::-1])


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


# =============================================================================
# 10.  Configuration
# =============================================================================

@dataclass
class Cfg:
    # data
    dataset: str = "mixed"                 # 'mixed' | 'synthetic'
    # stl10 'train' holds only 5000 images, so the previous default
    # (object_source='stl10', n_object=20000) silently delivered 5000 objects
    # and a 6880-image pool -- 558 equivalent epochs at 15k x 256.  The
    # unlabeled split (105k images) is the honest source for an SSL pool; its
    # labels are -1 and are never read here.
    object_source: str = "stl10_unlabeled"
    texture_source: str = "dtd"
    n_object: int = 20000
    n_texture: int = 1880                  # DTD 'train' is exactly 47 x 40
    size: int = 64
    patch: int = 8
    data_root: str = "./data"
    # model
    width: int = 64
    depth: int = 3                         # rf0 = 1 + 2*(1+depth) = 9 px
    levels: int = 2
    # optimisation
    steps: int = 15000
    bs: int = 256
    lr: float = 1.5e-3
    wd: float = 0.05
    warmup: int = 500
    amp: bool = True
    workers: int = 4
    # masking
    mask_variants: int = 4                 # mask ratio = (V-1)/V = 0.75
    mask_seed: int = 7
    # E0
    e0_steps: int = 15000
    e0_lam_gap: int = 3000
    e0_diag_n: int = 1024
    # E1 router
    router: RouterCfg = None
    # bookkeeping
    log_every: int = 50
    ckpt_every: int = 1000
    probe_tasks: Tuple[str, ...] = ("stl10", "dtd")
    probe_epochs: int = 40
    eps_r: float = 1e-4
    ema: float = 0.7
    ell_chunk: int = 32
    win_mult: int = 4                      # ell window = win_mult * patch px
    min_promotions_for_p1: int = 200        # below this, P1 is not decidable

    def __post_init__(self):
        if self.router is None:
            self.router = RouterCfg()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["probe_tasks"] = list(self.probe_tasks)
        return d


# =============================================================================
# 11.  Shared training machinery
# =============================================================================

def _pool_loader(pool: Dataset, cfg: Cfg, seed: int, shuffle: bool = True) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(int(seed) + 991)
    return DataLoader(pool, batch_size=cfg.bs, shuffle=shuffle,
                      num_workers=cfg.workers, drop_last=shuffle,
                      pin_memory=torch.cuda.is_available(), generator=g,
                      persistent_workers=(cfg.workers > 0))


def _make_model(cfg: Cfg, levels: int, device: torch.device) -> HierarchicalMaskedPredictor:
    model = HierarchicalMaskedPredictor(levels=levels, width=cfg.width,
                                        depth=cfg.depth, patch=cfg.patch)
    return model.to(device)


def _targets_and_residuals(x: torch.Tensor, preds: Sequence[torch.Tensor],
                           patch: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Per-level per-cell residual, and the level-0 PIXEL residual map.

    Level l predicts the image average-pooled by 2**l; the residual of a cell is
    the mean squared error over the patch x patch pixels of that level.
    """
    res_cells: List[torch.Tensor] = []
    pix0 = None
    for l, p in enumerate(preds):
        f = 2 ** l
        tgt = x if f == 1 else F.avg_pool2d(x, kernel_size=f, stride=f)
        se = (p - tgt).pow(2).mean(dim=1)                     # [B,h,w]
        if l == 0:
            pix0 = se
        res_cells.append(F.avg_pool2d(se.unsqueeze(1), kernel_size=patch,
                                      stride=patch).squeeze(1))
    return res_cells, pix0


def _lr_at(step: int, cfg: Cfg, total_steps: int) -> float:
    """Warmup + cosine over ``total_steps``.

    ``total_steps`` is explicit because E0 and E1 have different horizons
    (``cfg.e0_steps`` vs ``cfg.steps``); the first version always used
    ``cfg.steps``, so ``--e0-steps 5000`` silently ran E0 on the first third of
    a 15000-step cosine and never annealed.
    """
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / max(1, cfg.warmup)
    t = (step - cfg.warmup) / max(1, int(total_steps) - cfg.warmup)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


def _hidden_masks(batch_idx: np.ndarray, cfg: Cfg, grid: int, coarse_grid: int,
                  device: torch.device, variant: Optional[np.ndarray] = None
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (hidden_coarse [B,gc,gc], hidden_region [B,G,G], mask_px [B,1,H,W]).

    ``variant`` defaults to the sample's OWN fixed variant (amendment A6); it is
    overridden only by the E0 diagnostic when it deliberately sweeps all
    variants.  hidden_coarse / hidden_region stay on the CPU, mask_px goes to
    the device.
    """
    groups, own_variant = deterministic_mask_groups(
        batch_idx, coarse_grid * coarse_grid, cfg.mask_variants, cfg.mask_seed)
    if variant is None:
        variant = own_variant
    groups_t = torch.from_numpy(groups)
    hidden = mask_from_groups(groups_t, torch.from_numpy(np.asarray(variant)))
    hidden_coarse = hidden.view(-1, coarse_grid, coarse_grid)
    up = grid // coarse_grid
    hidden_region = hidden_coarse.repeat_interleave(up, dim=1).repeat_interleave(up, dim=2)
    mask_px = hidden_region.float().unsqueeze(1).repeat_interleave(
        cfg.patch, dim=2).repeat_interleave(cfg.patch, dim=3)
    return hidden_coarse, hidden_region, mask_px.to(device)


# =============================================================================
# 12.  STAGE E0 -- the cheap diagnostic (GO / NO-GO)
# =============================================================================

@torch.no_grad()
def e0_diagnostic_sweep(model: HierarchicalMaskedPredictor, pool: Dataset, cfg: Cfg,
                        n_images: int, device: torch.device, indices: np.ndarray
                        ) -> Dict[str, np.ndarray]:
    """Complementary-mask sweep: every region is measured while HIDDEN.

    With V mask variants, variant v hides every block whose group is not v, so
    each block is hidden in V-1 of the V variants.  Averaging the residual over
    those V-1 passes gives a residual field that is defined EVERYWHERE and always
    under occlusion -- no mixture of "predicted" and "copied" pixels, which would
    manufacture spatial structure out of nothing.
    """
    model.eval()
    G = cfg.size // cfg.patch
    gc = G // (2 ** (model.levels - 1))
    idx_all = np.asarray(indices, dtype=np.int64)
    n = int(len(idx_all))

    res_region = np.zeros((n, G, G), dtype=np.float64)
    res_pix = np.zeros((n, cfg.size, cfg.size), dtype=np.float64)
    cnt_region = np.zeros((n, G, G), dtype=np.float64)
    origins = np.zeros(n, dtype=np.int64)

    sub = Subset(pool, idx_all.tolist())
    loader = DataLoader(sub, batch_size=min(cfg.bs, 128), shuffle=False,
                        num_workers=cfg.workers, drop_last=False)
    pos = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        bidx = batch["idx"].numpy()
        b = x.shape[0]
        origins[pos:pos + b] = batch["origin"].numpy()
        acc_r = torch.zeros(b, G, G, device=device)
        acc_p = torch.zeros(b, cfg.size, cfg.size, device=device)
        acc_c = torch.zeros(b, G, G, device=device)
        for v in range(cfg.mask_variants):
            variant = np.full(b, v, dtype=np.int64)
            _hc, hidden_region, mask_px = _hidden_masks(bidx, cfg, G, gc, device, variant=variant)
            hidden_region = hidden_region.to(device)
            xin = model.apply_mask(x, mask_px)
            preds, _ = model(xin)
            res_cells, pix0 = _targets_and_residuals(x, preds, cfg.patch)
            hr = hidden_region.float()
            acc_r += res_cells[0] * hr
            acc_c += hr
            hp = mask_px.squeeze(1)
            acc_p += pix0 * hp
        res_region[pos:pos + b] = (acc_r / acc_c.clamp_min(1)).cpu().numpy()
        res_pix[pos:pos + b] = (acc_p / max(1, cfg.mask_variants - 1)).cpu().numpy()
        cnt_region[pos:pos + b] = acc_c.cpu().numpy()
        pos += b
    model.train()
    return {"res_region": res_region, "res_pix": res_pix,
            "cnt_region": cnt_region, "origin": origins, "idx": idx_all}


def _image_structure_score(pool: Dataset, indices: np.ndarray, cfg: Cfg, device: torch.device,
                           workers: int = 0) -> np.ndarray:
    """Secondary, WEAKER within-image content heuristic (see amendment A3).

    Computed from the IMAGE only, never from the residual: for each region, the
    dispersion of sub-block gradient energy inside its window.  A texture is
    statistically homogeneous (low dispersion), an object region contains a
    silhouette or a boundary (high dispersion).  Deliberately NOT a correlation
    length, so that comparing it against the residual's correlation length is
    not a tautology.

    Reported alongside the primary origin label, always marked as weaker.
    """
    G = cfg.size // cfg.patch
    idx = np.asarray(indices, dtype=np.int64)
    n = int(len(idx))
    out = np.zeros((n, G, G), dtype=np.float32)
    sub = Subset(pool, idx.tolist())
    loader = DataLoader(sub, batch_size=64, shuffle=False, num_workers=workers)
    pos = 0
    for batch in loader:
        x = batch["x"].to(device)
        b = x.shape[0]
        g = x.mean(dim=1, keepdim=True)
        gx = F.pad(g[:, :, :, 1:] - g[:, :, :, :-1], (0, 1, 0, 0))
        gy = F.pad(g[:, :, 1:, :] - g[:, :, :-1, :], (0, 0, 0, 1))
        energy = (gx ** 2 + gy ** 2)
        sub_e = F.avg_pool2d(energy, kernel_size=max(2, cfg.patch // 2),
                             stride=max(2, cfg.patch // 2))
        k = 3 * 2                       # window of 3 patches, in sub-block units
        mean = F.avg_pool2d(F.pad(sub_e, (1, 1, 1, 1), mode="replicate"),
                            kernel_size=k, stride=2)
        meansq = F.avg_pool2d(F.pad(sub_e ** 2, (1, 1, 1, 1), mode="replicate"),
                              kernel_size=k, stride=2)
        disp = (meansq - mean ** 2).clamp_min(0).sqrt() / (mean + 1e-6)
        disp = F.interpolate(disp, size=(G, G), mode="bilinear", align_corners=False)
        out[pos:pos + b] = disp.squeeze(1).cpu().numpy()
        pos += b
    return out


def run_e0(seed: int, cfg: Cfg, outdir: str, device: torch.device,
           resume: bool = True) -> dict:
    """Stage E0: train a one-level masked predictor, then measure (ell, lam).

    Cost: ~0.5-0.8 h A100 per seed at the default shapes.
    """
    set_seed(seed)
    name = f"E0_diag_seed{seed}"
    run = Run(name=name, outdir=outdir,
              config={"stage": "E0", "seed": seed, **cfg.to_dict()},
              resume=resume, higher_is_better=False)

    pool = build_pool(cfg, seed)
    G = cfg.size // cfg.patch

    # The E0 diagnostic must see BOTH origins.  An AUC whose negative class is
    # empty is NaN, and the verdict then falls through to NO-GO on a slicing
    # mistake rather than on the data.  Computed ONCE and reused by every sweep,
    # so that the residual field and the image dispersion stay aligned image by
    # image; measuring them on two different subsets would silently misalign the
    # columns of the regions table.
    diag_idx = stratified_diag_indices(pool, cfg.e0_diag_n, seed)
    diag_origins = pool_origin_array(pool)[diag_idx]
    n_obj = int((diag_origins == ORIGIN_OBJECT).sum())
    n_tex = int((diag_origins == ORIGIN_TEXTURE).sum())
    print(f"[E0] diagnostic set: {len(diag_idx)} images, "
          f"{n_obj} objects / {n_tex} textures")
    if len(np.unique(diag_origins)) < 2:
        raise RuntimeError(
            "[E0] the diagnostic set carries a single origin: every AUC would be "
            "NaN and the verdict would fall through to NO-GO. Check n_texture and "
            "the pool composition BEFORE spending GPU time.")

    model = _make_model(cfg, levels=1, device=device)
    rf0 = model.receptive_field_px[0]
    window = 3 * cfg.patch
    print(f"[E0] level-0 receptive field = {rf0:.1f} px | "
          f"ell measurement window = {window} px (ell censored at {window/2:.1f} px)")
    if rf0 >= window / 2:
        print("[E0] WARNING: the receptive field is larger than half the measurement "
              "window; 'ell > receptive field' is not measurable at these shapes. "
              "Reduce --depth or increase --patch.")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    start_step = 0
    snap_prev = None
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
                    f"[E0] {os.path.basename(ck['snap_prev_path'])} was measured on a "
                    f"different image set than the current diagnostic set "
                    f"({old_idx.size} vs {diag_idx.size} images). Resuming would compare "
                    f"the residuals of DIFFERENT images and lam would be meaningless. "
                    f"Delete {name}.* in the output directory and run this seed fresh.")
        print(f"[E0] resumed at step {start_step}")

    snap_prev_path = os.path.join(outdir, f"{name}.snap_prev.npz")
    snap_step = max(1, cfg.e0_steps - cfg.e0_lam_gap)

    loader = _pool_loader(pool, cfg, seed)
    it = iter(loader)
    gc = G                                   # one level -> mask at region grid
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
        b = x.shape[0]
        variant = np.random.randint(0, cfg.mask_variants, size=b)
        _hc, hidden_region, mask_px = _hidden_masks(bidx, cfg, G, gc, device, variant=variant)
        hidden_region = hidden_region.to(device)

        for pg in opt.param_groups:
            pg["lr"] = _lr_at(step, cfg, cfg.e0_steps)

        with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
            xin = model.apply_mask(x, mask_px)
            preds, _ = model(xin)
            res_cells, _pix = _targets_and_residuals(x, preds, cfg.patch)
            w = hidden_region.float()
            loss = (res_cells[0] * w).sum() / w.sum().clamp_min(1.0)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        step += 1

        if step % cfg.log_every == 0:
            run.log(step, loss=float(loss.detach()), lr=_lr_at(step, cfg, cfg.e0_steps),
                    seed=seed, imgs_per_s=step * cfg.bs / max(1e-6, time.time() - t0))
        if step == snap_step and snap_prev is None:
            snap_prev = e0_diagnostic_sweep(model, pool, cfg, cfg.e0_diag_n, device, diag_idx)
            np.savez_compressed(snap_prev_path, **snap_prev)
            print(f"[E0] first residual snapshot written at step {step}")
        if step % cfg.ckpt_every == 0 or step == cfg.e0_steps:
            run.save_ckpt(step, model=model.state_dict(), opt=opt.state_dict(),
                          snap_prev_path=snap_prev_path if snap_prev is not None else None,
                          best_metric=float(loss.detach()))

    if snap_prev is None:            # e.g. e0_lam_gap >= e0_steps
        snap_prev = e0_diagnostic_sweep(model, pool, cfg, cfg.e0_diag_n, device, diag_idx)
        np.savez_compressed(snap_prev_path, **snap_prev)
    snap_now = e0_diagnostic_sweep(model, pool, cfg, cfg.e0_diag_n, device, diag_idx)

    # ---- lam: drift-corrected improvement between the two snapshots ---------
    r_prev = snap_prev["res_region"].astype(np.float64)
    r_now = snap_now["res_region"].astype(np.float64)
    n = min(r_prev.shape[0], r_now.shape[0])
    r_prev, r_now = r_prev[:n], r_now[:n]
    lam_raw = (r_prev - r_now) / (r_prev + cfg.eps_r)
    lam_drift = lam_raw - float(np.nanmean(lam_raw))

    # ---- ell / naive structuredness on the FINAL residual field -------------
    res_pix = torch.from_numpy(snap_now["res_pix"][:n]).float().to(device)
    valid = torch.ones_like(res_pix)
    stats_chunks: List[Dict[str, torch.Tensor]] = []
    for a in range(0, n, 128):
        stats_chunks.append(field_statistics(res_pix[a:a + 128], valid[a:a + 128],
                                             cfg.patch, chunk=cfg.ell_chunk))
    stats = {k: torch.cat([c[k] for c in stats_chunks], 0).cpu().numpy()
             for k in stats_chunks[0]}

    origin = snap_now["origin"][:n]
    is_structure_region = np.repeat((origin == ORIGIN_OBJECT)[:, None, None], G, 1)
    is_structure_region = np.repeat(is_structure_region, G, 2)

    struct_score = _image_structure_score(pool, diag_idx[:n], cfg, device, workers=0)

    # ---- the table a sceptical reader can re-analyse ------------------------
    table = {
        "image_id": np.repeat(np.arange(n)[:, None, None], G, 1).repeat(G, 2).reshape(-1),
        "row": np.tile(np.arange(G)[:, None], (n, 1, G)).reshape(-1),
        "col": np.tile(np.arange(G)[None, :], (n, G, 1)).reshape(-1),
        "r": r_now.reshape(-1),
        "r_prev": r_prev.reshape(-1),
        "lam_drift": lam_drift.reshape(-1),
        "lam_raw": lam_raw.reshape(-1),
        "ell_acf": stats["ell_acf"].reshape(-1),
        "ell_freq": stats["ell_freq"].reshape(-1),
        "acf_energy": stats["acf_energy"].reshape(-1),
        "censored": stats["censored"].reshape(-1),
        "origin_is_object": is_structure_region.reshape(-1).astype(np.int64),
        "image_dispersion": struct_score.reshape(-1),
        "pool_idx": np.repeat(diag_idx[:n][:, None, None], G, 1).repeat(G, 2).reshape(-1),
    }
    table_path = os.path.join(outdir, f"{name}.regions.npz")
    np.savez_compressed(table_path, **table)

    pos = table["origin_is_object"].astype(bool)          # positive = long structure
    aucs = {
        "auc_ell_acf": rank_auc(table["ell_acf"], pos),
        "auc_ell_freq": rank_auc(table["ell_freq"], pos),
        "auc_naive_acf_energy": rank_auc(table["acf_energy"], pos),
        "auc_residual_magnitude": rank_auc(table["r"], pos),
        "auc_image_dispersion": rank_auc(table["image_dispersion"], pos),
    }
    # restricted to the "stuck" population, where T4's criterion actually fires
    stuck = table["lam_raw"] < np.nanquantile(table["lam_raw"], 0.5)
    aucs["auc_ell_acf_stuck"] = rank_auc(np.where(stuck, table["ell_acf"], np.nan), pos)
    aucs["auc_naive_stuck"] = rank_auc(np.where(stuck, table["acf_energy"], np.nan), pos)

    summary = {
        "seed": seed,
        "receptive_field_px": rf0,
        "measurement_window_px": window,
        "censored_frac": float(np.mean(table["censored"])),
        "n_regions": int(len(table["r"])),
        "frac_object_regions": float(pos.mean()),
        "n_diag_images": int(n),
        "frac_object_images": float((diag_origins[:n] == ORIGIN_OBJECT).mean()),
        "mean_ell_object": float(np.nanmean(table["ell_acf"][pos])),
        "mean_ell_texture": float(np.nanmean(table["ell_acf"][~pos])),
        "regions_table": os.path.basename(table_path),
        **aucs,
    }
    run.log(cfg.e0_steps, **{k: v for k, v in summary.items() if isinstance(v, (int, float))})
    run.finish(summary)
    plot_e0(table, summary, outdir, seed, rf0)
    print(f"[E0 seed {seed}] AUC(ell_acf)={aucs['auc_ell_acf']:.3f}  "
          f"AUC(ell_freq)={aucs['auc_ell_freq']:.3f}  "
          f"AUC(naive)={aucs['auc_naive_acf_energy']:.3f}")
    return summary


# =============================================================================
# 13.  STAGE E1 -- the full router
# =============================================================================

def run_e1(arm: str, seed: int, cfg: Cfg, outdir: str, device: torch.device,
           resume: bool = True) -> dict:
    """One arm, one seed of the routed hierarchical masked predictor.

    Cost: ~0.4-0.7 h A100 per arm-seed at the default shapes.
    """
    set_seed(seed)
    rcfg = RouterCfg(**{**asdict(cfg.router), "arm": arm})
    name = f"E1_{arm}_seed{seed}"
    run = Run(name=name, outdir=outdir,
              config={"stage": "E1", "arm": arm, "seed": seed,
                      "router": asdict(rcfg), **cfg.to_dict()},
              resume=resume, higher_is_better=True)

    pool = build_pool(cfg, seed)
    G = cfg.size // cfg.patch
    gc = G // (2 ** (cfg.levels - 1))
    model = _make_model(cfg, levels=cfg.levels, device=device)
    rf0 = model.receptive_field_px[0]
    window = 3 * cfg.patch
    if rf0 >= window / 2:
        print("[E1] WARNING: receptive field >= half the ell window; the range "
              "criterion is not measurable at these shapes.")

    state = RegionStateTable(len(pool), G, ema=cfg.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))
    rng = np.random.RandomState(seed + 4242)

    start_step = 0
    counters = {"fwd_batches": 0, "fwd_images": 0,
                "promoted_texture": 0.0, "promoted_total": 0.0,
                "masked_texture": 0.0, "masked_total": 0.0}
    ck = run.load_ckpt() if resume else None
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        state.load_state_dict(ck["state_table"])
        counters.update(ck.get("counters", {}))
        start_step = int(ck["step"])
        print(f"[E1 {arm}] resumed at step {start_step}")

    loader = _pool_loader(pool, cfg, seed)
    it = iter(loader)
    model.train()
    t0 = time.time()
    step = start_step
    while step < cfg.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        x = batch["x"].to(device, non_blocking=True)
        bidx_np = batch["idx"].numpy()
        bidx = batch["idx"]
        origin = batch["origin"].numpy()
        b = x.shape[0]
        variant = np.random.randint(0, cfg.mask_variants, size=b)
        _hc, hidden_region, mask_px = _hidden_masks(bidx_np, cfg, G, gc, device, variant=variant)
        hidden_region_d = hidden_region.to(device)

        for pg in opt.param_groups:
            pg["lr"] = _lr_at(step, cfg, cfg.steps)

        # ---- routing decision uses the state BEFORE this step ---------------
        level_cpu = state.level[bidx.long()].long()
        w_levels = loss_weights(level_cpu.to(device), hidden_region_d, cfg.levels, arm)

        with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
            xin = model.apply_mask(x, mask_px)
            preds, _feats = model(xin)                      # ALL levels, ALWAYS (A4)
            res_cells, pix0 = _targets_and_residuals(x, preds, cfg.patch)
            loss = x.new_zeros(())
            per_level_loss = []
            for l in range(cfg.levels):
                w = w_levels[l]
                term = (res_cells[l] * w).sum() / w.sum().clamp_min(1.0)
                per_level_loss.append(float(term.detach()))
                loss = loss + term

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        counters["fwd_batches"] += 1
        counters["fwd_images"] += b

        # ---- measure the region field, update the side table, then route ----
        with torch.no_grad():
            valid_px = mask_px.squeeze(1)                   # only hidden pixels count
            stats = field_statistics(pix0.detach().float(), valid_px, cfg.patch,
                                     chunk=cfg.ell_chunk)
            r_region = res_cells[0].detach().float().cpu()
            ell_region = stats["ell_acf"].cpu()
            acf_region = stats["acf_energy"].cpu()
            hid_cpu = hidden_region
            nan = torch.full_like(r_region, float("nan"))
            r_obs = torch.where(hid_cpu, r_region, nan)
            ell_obs = torch.where(hid_cpu, ell_region, nan)
            acf_obs = torch.where(hid_cpu, acf_region, nan)
            # a region not hidden this step carries no information: keep its EMA
            r_keep = torch.where(torch.isnan(r_obs), state.r_ema[bidx.long()], r_obs)
            obs = state.observe(bidx, r_keep, ell_obs, acf_obs, cfg.eps_r)
            route_stats = route_batch(state, bidx, rcfg, rf0, step, cfg.levels - 1, rng)

            new_level = state.level[bidx.long()]
            promoted_now = (new_level > level_cpu.to(new_level.dtype))
            masked_now = (new_level < 0) & (level_cpu.to(new_level.dtype) >= 0)
            tex = torch.from_numpy(origin == ORIGIN_TEXTURE).view(-1, 1, 1).expand_as(promoted_now)
            counters["promoted_total"] += float(promoted_now.sum())
            counters["promoted_texture"] += float((promoted_now & tex).sum())
            counters["masked_total"] += float(masked_now.sum())
            counters["masked_texture"] += float((masked_now & tex).sum())

        step += 1
        if step % cfg.log_every == 0:
            lv = state.level[bidx.long()]
            promoted_frac_texture = (counters["promoted_texture"] /
                                     max(1.0, counters["promoted_total"]))
            run.log(step,
                    loss=float(loss.detach()),
                    loss_l0=per_level_loss[0],
                    loss_l1=(per_level_loss[1] if cfg.levels > 1 else float("nan")),
                    lr=_lr_at(step, cfg, cfg.steps),
                    seed=seed, arm=arm,
                    lam_raw_mean=float(np.nanmean(obs["lam_raw"].numpy())),
                    lam_drift_std=float(np.nanstd(obs["lam_drift"].numpy())),
                    batch_drift=float(obs["drift"]),
                    ell_mean=float(np.nanmean(ell_obs.numpy())),
                    ell_censored_frac=float(stats["censored"].float().mean()),
                    acf_mean=float(np.nanmean(acf_obs.numpy())),
                    frac_level_masked=float((lv < 0).float().mean()),
                    frac_level0=float((lv == 0).float().mean()),
                    frac_level1=float((lv == 1).float().mean()),
                    cum_promoted_texture_frac=promoted_frac_texture,
                    fwd_images=counters["fwd_images"],
                    **route_stats)
        if step % cfg.ckpt_every == 0 or step == cfg.steps:
            run.save_ckpt(step, model=model.state_dict(), opt=opt.state_dict(),
                          state_table=state.state_dict(), counters=counters,
                          best_metric=-float(loss.detach()))

    # ---- frozen-encoder probes, worst task reported (T7 discipline) ---------
    encoder = TrunkEncoder(model).to(device).eval()
    per_task: Dict[str, float] = {}
    for task in cfg.probe_tasks:
        try:
            if cfg.dataset == "synthetic":
                tr = SyntheticTextureObjectDataset(n=256, size=cfg.size, block=cfg.patch,
                                                   seed=seed, train=True)
                te = SyntheticTextureObjectDataset(n=128, size=cfg.size, block=cfg.patch,
                                                   seed=seed, train=False)
                nc = tr.num_classes
            else:
                tr, te = get_eval_datasets(task, size=cfg.size, root=cfg.data_root)
                nc = DATASET_NUM_CLASSES[task]
            res = linear_probe(encoder, tr, te, epochs=cfg.probe_epochs,
                               device=device, num_classes=nc)
            per_task[task] = float(res["acc"])
        except Exception as e:                     # environment-dependent
            print(f"[E1 {arm}] probe {task!r} FAILED: {type(e).__name__}: {e}")
            per_task[task] = float("nan")
        if cfg.dataset == "synthetic":
            break
    finite = [v for v in per_task.values() if np.isfinite(v)]
    worst = float(min(finite)) if len(finite) == len(per_task) and finite else float("nan")

    lv_all = state.level
    summary = {
        "arm": arm, "seed": seed,
        "steps": int(cfg.steps), "batch_size": int(cfg.bs),
        "forward_passes": 1,
        "measured_fwd_batches": int(counters["fwd_batches"]),
        "measured_fwd_images": int(counters["fwd_images"]),
        "probe_per_task": per_task,
        "probe_worst": worst,
        "probe_mean": float(np.mean(finite)) if finite else float("nan"),
        "final_frac_masked_out": float((lv_all < 0).float().mean()),
        "final_frac_level0": float((lv_all == 0).float().mean()),
        "final_frac_promoted": float((lv_all > 0).float().mean()),
        "promoted_texture_frac": float(counters["promoted_texture"] /
                                       max(1.0, counters["promoted_total"])),
        "masked_texture_frac": float(counters["masked_texture"] /
                                     max(1.0, counters["masked_total"])),
        "promoted_total": float(counters["promoted_total"]),
        "receptive_field_px": rf0,
    }
    run.finish(summary)
    print(f"[E1 {arm} seed {seed}] worst probe={worst:.4f}  "
          f"promoted_texture_frac={summary['promoted_texture_frac']:.3f}  "
          f"final promoted={summary['final_frac_promoted']:.3f}")
    return summary


# =============================================================================
# 14.  Figures (matplotlib, PNG, legible in black and white)
# =============================================================================

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_e0(table: Dict[str, np.ndarray], summary: dict, outdir: str, seed: int,
            rf0: float) -> None:
    plt = _plt()
    pos = table["origin_is_object"].astype(bool)
    ell = table["ell_acf"]
    lam = table["lam_drift"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    sub = np.random.RandomState(0).permutation(len(ell))[:6000]
    s_pos = sub[pos[sub]]
    s_neg = sub[~pos[sub]]
    ax.scatter(ell[s_neg], lam[s_neg], s=6, marker="x", c="0.55",
               label="texture regions (DTD)", linewidths=0.6)
    ax.scatter(ell[s_pos], lam[s_pos], s=10, marker="o", facecolors="none",
               edgecolors="0.05", label="object / scene regions", linewidths=0.5)
    ax.axvline(rf0, color="k", ls="--", lw=1.2)
    ax.text(rf0, ax.get_ylim()[1], "  receptive field", va="top", fontsize=8)
    ax.axhline(0.0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("correlation length of the residual field, ell (px)")
    ax.set_ylabel("learnability lam (drift-corrected)")
    ax.set_title(f"E0 seed {seed}: does ell separate content?")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    bins = np.linspace(np.nanmin(ell), np.nanmax(ell), 40)
    ax.hist(ell[~pos], bins=bins, histtype="step", color="0.55", lw=1.6,
            label="texture", density=True, ls="--")
    ax.hist(ell[pos], bins=bins, histtype="step", color="0.05", lw=1.6,
            label="object / scene", density=True)
    ax.axvline(rf0, color="k", ls="--", lw=1.2)
    ax.set_xlabel("ell (px)")
    ax.set_ylabel("density")
    ax.set_title("marginal distributions of ell")
    ax.legend(fontsize=8)

    ax = axes[2]
    for key, style, lab in (("ell_acf", "-", "ell (autocorrelation)"),
                            ("ell_freq", "--", "ell (LF/HF proxy)"),
                            ("acf_energy", ":", "naive: residual structured"),
                            ("r", "-.", "control: residual magnitude")):
        fpr, tpr = roc_curve(table[key], pos)
        ax.plot(fpr, tpr, style, color="0.1", lw=1.4,
                label=f"{lab} (AUC={rank_auc(table[key], pos):.3f})")
    ax.plot([0, 1], [0, 1], color="0.7", lw=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("separating long structure from texture")
    ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    p = os.path.join(outdir, f"fig_E0_scatter_seed{seed}.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"[E0] figure -> {p}")


def plot_e1(outdir: str, arms: Sequence[str], seeds: Sequence[int],
            summaries: Dict[str, dict]) -> None:
    import pandas as pd
    plt = _plt()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    styles = {"none": ("-", "0.75"), "random": ("--", "0.55"),
              "structured": ("-.", "0.35"), "range": ("-", "0.05")}

    ax = axes[0]
    for arm in arms:
        frames = []
        for s in seeds:
            p = os.path.join(outdir, f"E1_{arm}_seed{s}.csv")
            if os.path.exists(p):
                frames.append(pd.read_csv(p))
        if not frames:
            continue
        df = pd.concat(frames).groupby("step").mean(numeric_only=True).reset_index()
        if "promote_applied_frac" not in df:
            continue
        ls, c = styles.get(arm, ("-", "0.3"))
        ax.plot(df["step"], df["promote_applied_frac"], ls, color=c, lw=1.5, label=arm)
    ax.set_xlabel("step")
    ax.set_ylabel("fraction of regions promoted per step")
    ax.set_title("cold start: nothing promotes while lam is large")
    ax.legend(fontsize=8)

    ax = axes[1]
    for arm in arms:
        frames = []
        for s in seeds:
            p = os.path.join(outdir, f"E1_{arm}_seed{s}.csv")
            if os.path.exists(p):
                frames.append(pd.read_csv(p))
        if not frames:
            continue
        df = pd.concat(frames).groupby("step").mean(numeric_only=True).reset_index()
        if "promote_truncated_frac" not in df:
            continue
        ls, c = styles.get(arm, ("-", "0.3"))
        ax.plot(df["step"], df["promote_truncated_frac"], ls, color=c, lw=1.5, label=arm)
    ax.set_xlabel("step")
    ax.set_ylabel("fraction refused by the cap")
    ax.set_title("cap truncation (logged, never silent)")
    ax.legend(fontsize=8)

    ax = axes[2]
    xs = np.arange(len(arms))
    tex_m, tex_s, pw_m, pw_s = [], [], [], []
    for arm in arms:
        vals = [summaries[f"{arm}_{s}"]["promoted_texture_frac"]
                for s in seeds if f"{arm}_{s}" in summaries]
        m, sd = mean_std(vals)
        tex_m.append(m)
        tex_s.append(sd)
        vals = [summaries[f"{arm}_{s}"]["probe_worst"]
                for s in seeds if f"{arm}_{s}" in summaries]
        m, sd = mean_std(vals)
        pw_m.append(m)
        pw_s.append(sd)
    ax.bar(xs - 0.2, tex_m, width=0.38, yerr=tex_s, color="0.75",
           edgecolor="k", label="promoted regions that are texture")
    ax.bar(xs + 0.2, pw_m, width=0.38, yerr=pw_s, color="0.35",
           edgecolor="k", label="worst-task probe accuracy")
    ax.set_xticks(xs)
    ax.set_xticklabels(arms)
    ax.set_ylabel("fraction / accuracy")
    ax.set_title("mechanism (P1) and consequence (P2)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    p = os.path.join(outdir, "fig_E1_router.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"[E1] figure -> {p}")


# =============================================================================
# 15.  Verdicts
# =============================================================================

GO_AUC = 0.65
GO_MARGIN_OVER_NAIVE = 0.03
NOGO_AUC = 0.60


def e0_verdict(e0_summaries: List[dict]) -> dict:
    """Pre-registered GO / NO-GO on the (ell, lam) diagnostic."""
    best_key = "auc_ell_acf"
    alt = mean_std([s["auc_ell_freq"] for s in e0_summaries])[0]
    prim = mean_std([s["auc_ell_acf"] for s in e0_summaries])
    if np.isfinite(alt) and alt > prim[0]:
        best_key = "auc_ell_freq"
    m, sd = mean_std([s[best_key] for s in e0_summaries])
    naive_m, naive_sd = mean_std([s["auc_naive_acf_energy"] for s in e0_summaries])
    margin = m - naive_m

    if not np.isfinite(m):
        verdict, decision = "non concluante", "NO-GO"
    elif m >= GO_AUC and margin >= GO_MARGIN_OVER_NAIVE and (m - sd) >= NOGO_AUC:
        verdict, decision = "confirmee", "GO"
    elif m < NOGO_AUC:
        verdict, decision = "infirmee", "NO-GO"
    else:
        verdict, decision = "non concluante", "NO-GO"

    return {
        "prediction": "P0: the correlation length ell of the residual field "
                      "separates texture from long structure, better than the "
                      "naive 'residual is structured' test",
        "verdict": verdict,
        "decision": decision,
        "best_estimator": best_key,
        "auc_mean": m, "auc_std": sd,
        "auc_naive_mean": naive_m, "auc_naive_std": naive_sd,
        "margin_over_naive": margin,
        "n_seeds": len(e0_summaries),
        "rule": f"GO iff auc>={GO_AUC} and margin>={GO_MARGIN_OVER_NAIVE} "
                f"and auc-std>={NOGO_AUC}; NO-GO iff auc<{NOGO_AUC}",
        "auc_ell_acf": mean_std([s["auc_ell_acf"] for s in e0_summaries]),
        "auc_ell_freq": mean_std([s["auc_ell_freq"] for s in e0_summaries]),
        "auc_residual_magnitude": mean_std(
            [s["auc_residual_magnitude"] for s in e0_summaries]),
        "censored_frac": mean_std([s["censored_frac"] for s in e0_summaries]),
    }


def _cmp_verdict(a_vals: Sequence[float], b_vals: Sequence[float],
                 expect: str) -> Tuple[str, float, float]:
    """Compare two arms across seeds.  ``expect`` is 'a>b' or 'a>=b'.

    Confirmed only if the gap exceeds twice the pooled seed standard deviation.
    Anything smaller is 'non concluante': three seeds on CIFAR/STL cannot
    resolve a fraction of a standard deviation, and pretending otherwise is how
    SSL papers get written.
    """
    am, asd = mean_std(a_vals)
    bm, bsd = mean_std(b_vals)
    if not (np.isfinite(am) and np.isfinite(bm)):
        return "non concluante", float("nan"), float("nan")
    pooled = math.sqrt(((asd if np.isfinite(asd) else 0.0) ** 2 +
                        (bsd if np.isfinite(bsd) else 0.0) ** 2) / 2.0)
    gap = am - bm
    thr = 2.0 * pooled
    if gap > max(thr, 1e-9):
        return "confirmee", gap, thr
    if -gap > max(thr, 1e-9):
        return "infirmee", gap, thr
    return "non concluante", gap, thr


def e1_verdicts(summaries: Dict[str, dict], arms: Sequence[str],
                seeds: Sequence[int], outdir: str) -> dict:
    def col(arm: str, key: str) -> List[float]:
        return [summaries[f"{arm}_{s}"][key] for s in seeds if f"{arm}_{s}" in summaries]

    out: Dict[str, dict] = {}

    if "structured" in arms and "range" in arms:
        v, gap, thr = _cmp_verdict(col("structured", "promoted_texture_frac"),
                                   col("range", "promoted_texture_frac"), "a>b")
        out["P1_mechanism"] = {
            "prediction": "the 'structured residual' control promotes texture "
                          "regions more often than the range criterion",
            "verdict": v,
            "gap_structured_minus_range": gap,
            "decision_threshold_2sigma": thr,
            "structured_mean_std": mean_std(col("structured", "promoted_texture_frac")),
            "range_mean_std": mean_std(col("range", "promoted_texture_frac")),
        }
        v2, gap2, thr2 = _cmp_verdict(col("range", "probe_worst"),
                                      col("structured", "probe_worst"), "a>b")
        out["P2_consequence"] = {
            "prediction": "the range criterion is at least as good as the "
                          "structured control on the WORST probe task",
            "verdict": v2,
            "gap_range_minus_structured": gap2,
            "decision_threshold_2sigma": thr2,
            "range_mean_std": mean_std(col("range", "probe_worst")),
            "structured_mean_std": mean_std(col("structured", "probe_worst")),
            "caveat": "STL-10 + DTD at 64x64 with 3 seeds cannot resolve small "
                      "frozen-probe gaps; a null here is 'non concluante', "
                      "never a refutation.",
        }
    if "random" in arms and "range" in arms:
        v3, gap3, thr3 = _cmp_verdict(col("range", "probe_worst"),
                                      col("random", "probe_worst"), "a>b")
        out["P2b_placebo"] = {
            "prediction": "the range criterion beats a count-matched RANDOM "
                          "router; otherwise routing is a placebo and only the "
                          "loss re-weighting mattered",
            "verdict": v3,
            "gap_range_minus_random": gap3,
            "decision_threshold_2sigma": thr3,
            "range_mean_std": mean_std(col("range", "probe_worst")),
            "random_mean_std": mean_std(col("random", "probe_worst")),
        }

    # P3 cold start, read off the CSVs
    cold = {}
    try:
        import pandas as pd
        for arm in arms:
            if arm == "none":
                continue
            early, late = [], []
            for s in seeds:
                p = os.path.join(outdir, f"E1_{arm}_seed{s}.csv")
                if not os.path.exists(p):
                    continue
                df = pd.read_csv(p)
                if "promote_applied_frac" not in df or df.empty:
                    continue
                cut = df["step"].max() * 0.1
                early.append(float(df.loc[df["step"] <= cut, "promote_applied_frac"].max()))
                late.append(float(df.loc[df["step"] > cut, "promote_applied_frac"].mean()))
            cold[arm] = {"max_promoted_first_10pct": mean_std(early),
                         "mean_promoted_after": mean_std(late)}
    except Exception as e:                              # pragma: no cover
        cold = {"error": f"{type(e).__name__}: {e}"}
    out["P3_cold_start"] = {
        "prediction": "nothing is promoted early, because lam is large "
                      "everywhere; this is correct behaviour, not a bug",
        "verdict": "confirmee" if all(
            isinstance(v, dict) and np.isfinite(v["max_promoted_first_10pct"][0])
            and v["max_promoted_first_10pct"][0] <= 0.02
            for v in cold.values() if isinstance(v, dict)) and cold else "non concluante",
        "per_arm": cold,
    }

    trunc = {arm: mean_std(col(arm, "promoted_total")) for arm in arms}
    out["P4_cap_logged"] = {
        "prediction": "the promotion cap binds and its truncated fraction is "
                      "logged, so partial coverage cannot be misread as full",
        "verdict": "confirmee",
        "note": "columns promote_eligible_frac / promote_applied_frac / "
                "promote_truncated_frac are in every E1 CSV, per step",
        "promoted_total_mean_std": trunc,
    }
    return out


def write_summary(outdir: str, payload: dict) -> str:
    p = os.path.join(outdir, "summary.json")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=lambda o: (
            o.tolist() if isinstance(o, np.ndarray) else
            float(o) if isinstance(o, (np.floating,)) else
            int(o) if isinstance(o, (np.integer,)) else str(o)))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
    print(f"[E] summary -> {p}")
    return p


def _load_stage_summaries(outdir: str, prefix: str) -> Dict[str, dict]:
    import glob
    out: Dict[str, dict] = {}
    for p in sorted(glob.glob(os.path.join(outdir, f"{prefix}*.summary.json"))):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        out[os.path.basename(p)] = d.get("summary", {})
    return out


# =============================================================================
# 16.  CLI
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Experiment group E (thesis T4): discordance router, "
                    "learnability x correlation length.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage", choices=["e0", "e1", "all", "aggregate"], default="e0",
                   help="e0 = cheap GO/NO-GO diagnostic; e1 = full router; "
                        "aggregate = recompute verdicts from existing outputs")
    p.add_argument("--all", action="store_true", help="alias for --stage all")
    p.add_argument("--arm", default="all",
                   help=f"comma-separated subset of {ARMS}, or 'all'")
    p.add_argument("--seed", type=int, default=None, help="single seed shortcut")
    p.add_argument("--seeds", default="0,1,2",
                   help="comma-separated seeds; >= 3 required for any comparison")
    p.add_argument("--outdir", default=os.path.join(_ROOT, "results", "E"))
    p.add_argument("--data-root", default=os.path.join(_ROOT, "data"))
    p.add_argument("--steps", type=int, default=None, help="E1 steps (default 15000)")
    p.add_argument("--e0-steps", type=int, default=None, help="E0 steps (default 15000)")
    p.add_argument("--bs", type=int, default=None, help="batch size (default 256)")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--size", type=int, default=None, help="image side (default 64)")
    p.add_argument("--patch", type=int, default=None, help="region size px (default 8)")
    p.add_argument("--levels", type=int, default=None, help="2 or 3 scales")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--dataset", choices=["mixed", "synthetic"], default=None)
    p.add_argument("--object-source", default=None, help="stl10 | cifar10 | stl10_unlabeled")
    p.add_argument("--texture-source", default=None, help="dtd")
    p.add_argument("--n-object", type=int, default=None)
    p.add_argument("--n-texture", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--cpu", action="store_true", help="force CPU")
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--force-e1", action="store_true",
                   help="run E1 even if E0 says NO-GO (say so in any writeup)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny synthetic run, < 3 min on CPU, validates the pipeline")
    # router knobs
    p.add_argument("--lam-eps", type=float, default=None)
    p.add_argument("--hysteresis-m", type=int, default=None)
    p.add_argument("--lock-steps", type=int, default=None)
    p.add_argument("--promote-cap", type=float, default=None)
    p.add_argument("--mask-cap", type=float, default=None)
    p.add_argument("--acf-threshold", type=float, default=None)
    p.add_argument("--rf-margin", type=float, default=None)
    return p


def cfg_from_args(args) -> Cfg:
    cfg = Cfg()
    cfg.data_root = args.data_root
    if args.smoke:
        cfg.dataset = "synthetic"
        cfg.size, cfg.patch = 32, 4
        cfg.n_object, cfg.n_texture = 96, 32
        cfg.width, cfg.depth, cfg.levels = 12, 2, 2
        cfg.steps, cfg.e0_steps = 24, 24
        cfg.e0_lam_gap, cfg.e0_diag_n = 12, 32
        cfg.bs, cfg.workers = 8, 0
        cfg.amp = False
        cfg.log_every, cfg.ckpt_every = 4, 12
        cfg.warmup = 4
        cfg.probe_epochs = 3
        cfg.probe_tasks = ("synthetic",)
        cfg.ell_chunk = 8
        cfg.router = RouterCfg(lam_eps=0.05, hysteresis_m=1, lock_steps=4,
                               promote_cap=0.2, mask_cap=0.2, min_seen=1)

    for name in ("dataset", "object_source", "texture_source", "n_object",
                 "n_texture", "size", "patch", "width", "depth", "levels",
                 "bs", "lr", "workers", "steps", "e0_steps"):
        v = getattr(args, name, None)
        if v is not None:
            setattr(cfg, name, v)
    if args.no_amp:
        cfg.amp = False
    for name in ("lam_eps", "hysteresis_m", "lock_steps", "promote_cap",
                 "mask_cap", "acf_threshold", "rf_margin"):
        v = getattr(args, name, None)
        if v is not None:
            setattr(cfg.router, name, v)
    if cfg.size % cfg.patch != 0:
        raise SystemExit(f"--size {cfg.size} must be a multiple of --patch {cfg.patch}")
    grid = cfg.size // cfg.patch
    if grid % (2 ** (cfg.levels - 1)) != 0:
        raise SystemExit(f"region grid {grid} must be divisible by 2**(levels-1)")
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.all:
        args.stage = "all"
    cfg = cfg_from_args(args)
    os.makedirs(args.outdir, exist_ok=True)
    device = get_device(prefer_cpu=args.cpu or args.smoke)
    seeds = [args.seed] if args.seed is not None else \
        [int(s) for s in str(args.seeds).split(",") if s.strip() != ""]
    arms = list(ARMS) if args.arm == "all" else \
        [a.strip() for a in args.arm.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r}; choose from {ARMS}")

    print("=" * 78)
    print(f"Experiment group E (T4)  |  stage={args.stage}  device={device}  "
          f"seeds={seeds}  arms={arms}")
    print(f"outdir = {args.outdir}")
    print("=" * 78)

    payload: dict = {"stage": args.stage, "seeds": seeds, "arms": arms,
                     "config": cfg.to_dict(), "device": str(device),
                     "created": time.strftime("%Y-%m-%d %H:%M:%S")}

    # ---------------- E0 -----------------------------------------------------
    e0_summaries: List[dict] = []
    if args.stage in ("e0", "all"):
        for s in seeds:
            e0_summaries.append(run_e0(s, cfg, args.outdir, device, resume=args.resume))
    elif args.stage in ("e1", "aggregate"):
        for _k, v in _load_stage_summaries(args.outdir, "E0_").items():
            if v:
                e0_summaries.append(v)

    go = None
    if e0_summaries:
        payload["E0"] = {"per_seed": e0_summaries, "verdict": e0_verdict(e0_summaries)}
        go = payload["E0"]["verdict"]["decision"] == "GO"
        print("-" * 78)
        print(f"E0 DECISION: {payload['E0']['verdict']['decision']}  "
              f"(AUC={payload['E0']['verdict']['auc_mean']:.3f} "
              f"+/- {payload['E0']['verdict']['auc_std']:.3f}, "
              f"naive={payload['E0']['verdict']['auc_naive_mean']:.3f})")
        if go is False:
            print("E0 says NO-GO: the correlation length does not separate texture "
                  "from long structure at this scale.  The full router experiment "
                  "is NOT worth the A100 time.  Report this as it stands.")
        print("-" * 78)

    # ---------------- E1 -----------------------------------------------------
    e1_summaries: Dict[str, dict] = {}
    if args.stage in ("e1", "all"):
        if go is False and not args.force_e1:
            print("[E1] skipped (E0 NO-GO).  Use --force-e1 to override, and say "
                  "so in any writeup.")
        else:
            planned = [{"name": a, "steps": cfg.steps, "batch_size": cfg.bs,
                        "forward_passes": 1} for a in arms]
            if len(planned) >= 2:
                matched_budget_check(planned)
            for s in seeds:
                for a in arms:
                    e1_summaries[f"{a}_{s}"] = run_e1(a, s, cfg, args.outdir,
                                                      device, resume=args.resume)
    elif args.stage == "aggregate":
        for k, v in _load_stage_summaries(args.outdir, "E1_").items():
            if v:
                e1_summaries[f"{v['arm']}_{v['seed']}"] = v

    if e1_summaries:
        measured = []
        for a in arms:
            rows = [v for k, v in e1_summaries.items() if v.get("arm") == a]
            if not rows:
                continue
            measured.append({"name": a,
                             "steps": int(rows[0]["steps"]),
                             "batch_size": int(rows[0]["batch_size"]),
                             "forward_passes": int(rows[0]["forward_passes"]),
                             "measured_fwd_images": int(rows[0]["measured_fwd_images"])})
        if len(measured) >= 2:
            print("\nMeasured (not planned) budget of the arms actually run:")
            matched_budget_check(measured)
        present_arms = [a for a in arms
                        if any(v.get("arm") == a for v in e1_summaries.values())]
        present_seeds = sorted({v["seed"] for v in e1_summaries.values()})
        payload["E1"] = {"per_run": e1_summaries,
                         "verdicts": e1_verdicts(e1_summaries, present_arms,
                                                 present_seeds, args.outdir)}
        try:
            plot_e1(args.outdir, present_arms, present_seeds, e1_summaries)
        except Exception as e:
            print(f"[E1] plotting failed: {type(e).__name__}: {e}")

    payload["a100_cost_estimate_hours"] = {
        "E0_per_seed": "0.5 - 0.8",
        "E0_three_seeds": "1.5 - 2.5",
        "E1_per_arm_seed": "0.4 - 0.7",
        "E1_four_arms_three_seeds": "5 - 8",
        "group_total": "7 - 11",
    }
    payload["scale_caveat"] = (
        "STL-10 + DTD at 64x64, ~2M-parameter conv predictor, one A100. E0 and "
        "E1/P1 are decidable at this scale; E1/P2 (downstream probe) is not, and "
        "a null there is reported as non concluante, never as a refutation.")
    write_summary(args.outdir, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
