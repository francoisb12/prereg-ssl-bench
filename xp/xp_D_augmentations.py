"""
xp_D_augmentations.py -- theses T5 and T6: the augmentation policy T.

===============================================================================
WHAT IS BEING TESTED
===============================================================================

D1 -- T5: "a LEARNED augmentation policy degenerates to the identity".

    Claim.  If the magnitudes m of the augmentation policy T_m are trained to
    minimise the SSL loss, then min_theta min_m L(theta, T_m) has a trivial
    minimum at T = identity: the two views coincide, the agreement term is
    exactly 0, and this holds for ANY encoder.  Therefore every scheme that
    learns T must add an adversarial sign plus a strength budget ||m|| <= c,
    and it is then the budget -- chosen by hand -- that fixes the equilibrium
    and becomes the real concept.  The prior did not disappear, it moved.

    Prediction P5.1 (degeneration).  Arm `learn-minimize`: ||m|| decays towards
    its floor, the agreement term decays towards 0, and the frozen linear probe
    decays towards the accuracy of an UNTRAINED encoder.
    Prediction P5.2 (the prior moved into the budget).  In the crossed plan
    (freeze the learned direction and sweep the budget c; freeze c and sweep the
    direction), the probe accuracy is moved much more by c than by the
    direction, and the learned direction is not the best direction of the grid.

    What would FALSIFY T5.
      * P5.1 is falsified if the minimising policy settles on a non-degenerate
        magnitude (final ||m|| > FALSIFY_MIN_KEEP_FRAC * initial ||m||) while its
        loss still decreases -- i.e. minimisation found a real augmentation.
      * P5.2 is falsified if spread(direction sweep) >= spread(budget sweep)
        AND the learned direction ranks first in the direction grid: the policy
        would then carry information that the scalar budget does not.
      * Note the asymmetry we accept: P5.2 confirmed is a statement about THIS
        objective, THIS parameterisation and THIS dataset.  It is evidence, not
        a theorem.
      * SELF-INFLICTED WEAKNESS, MADE VISIBLE.  The budget grid contains c = 0,
        i.e. no augmentation at all.  Including it inflates spread(budget) and
        would make P5.2 nearly unfalsifiable, so the verdict ALSO computes the
        spread over the strictly positive part of the grid and requires the
        stricter of the two ratios.  Both numbers are reported.

D2 -- T6: "the asymmetry of T; too little T is recoverable, too much is not".

    Claim.  Let G_task be the group under which the true label is invariant.
    T contained in G_task is safe (possibly insufficient).  T not contained in
    G_task destroys label information, and with a FROZEN encoder that loss is
    irrecoverable: no downstream head undoes an invariance engraved upstream.

    Domains, with G_task KNOWN by construction:
      * cifar10          -- standard; strong T is expected to win.
      * colored_shapes   -- the HUE is the label, shape/size/position are
                            nuisances.  Colour jitter + random grayscale destroy
                            the label.  G_task excludes colour transformations.
      * chirality        -- MNIST vs mirrored MNIST, binary "is it mirrored".
                            Horizontal flip destroys the label.  G_task excludes
                            the reflection.

    Prediction P6.1 (crossing).  acc(strong) > acc(conservative) on cifar10 and
    acc(strong) << acc(conservative) on the two adversarial domains: the curves
    cross when the domain changes, at identical budget.
    Prediction P6.2 (irrecoverability / amputation).  On the adversarial domains
    the damage measured with a FROZEN encoder is large, and fine-tuning the same
    checkpoint on the target task recovers a substantial part of it.  "Worse"
    then becomes "amputated": the information is still in the pixels, it is the
    frozen representation that no longer carries it.
    Prediction P6.3 (silence).  Neither the SSL loss nor its trend distinguishes
    the healthy domain from the amputated one: the failure mode gives no
    unsupervised warning.

    What would FALSIFY T6.
      * P6.1 is falsified if the ordering conservative/strong is the SAME in the
        three domains (no crossing) -- strong T would then simply be better or
        simply be worse, and the group argument would carry nothing.
      * P6.2 is falsified if damage_frozen ~ damage_finetuned (nothing was
        amputated, the features are merely worse everywhere), or if fine-tuning
        recovers nothing at all (the information really is gone from the pixels,
        which would mean the domain was badly built -- the pixel-probe ceiling
        below is there precisely to rule that out).
      * P6.3 is falsified if the final SSL loss ranks the arms in the same order
        as the probe: model selection would then be possible without labels.

===============================================================================
METHODOLOGICAL AMENDMENTS TO THE REQUESTED PLAN (read before reviewing)
===============================================================================

1.  GRADIENT ESTIMATOR: score function (REINFORCE) with a leave-one-out
    baseline, NOT a Gumbel relaxation.  Reason: a Gumbel/straight-through scheme
    would require differentiating the image operations with respect to their own
    magnitude, i.e. a differentiable re-implementation of crop / jitter / blur.
    That re-implementation is not the augmentation whose effect we claim to
    measure, and it introduces a confound exactly where the thesis lives.  The
    score-function estimator treats the pipeline as a black box, needs only the
    scalar loss, and is unbiased.  Its variance is handled by (a) sampling M
    magnitude vectors per step and (b) a leave-one-out baseline over those M
    samples, which is unbiased because the loss of the other chunks does not
    depend on m_j.
    TWO SOURCES OF BIAS ARE ACCEPTED AND NAMED, NOT HIDDEN.
      (i)  BatchNorm couples the chunks.  All M chunks go through ONE encoder
           forward, so chunk j's activations depend on batch statistics computed
           over the whole batch, hence on every m_k.  The leave-one-out baseline
           is therefore only approximately independent of m_j and credit
           assignment between chunks is slightly contaminated.  This affects how
           fast/where the policy moves, not the sign of P5.1, whose evidence is
           ||m|| -> floor with an unconstrained minimisation -- a statement about
           the location of the optimum, not about the estimator's efficiency.
      (ii) `--no-advantage-normalise` off by default divides the advantage by
           the batch reward std, which depends on m_j.  This is the standard
           REINFORCE variance reduction; it rescales, never flips, the sign.
    The estimator itself (score reaches mu only, no path through the sampled
    magnitudes) is checked numerically against a finite-difference gradient in
    `selfcheck_augment`, which runs before every experiment.

2.  ZERO FLOP OVERHEAD FOR THE POLICY.  The M magnitude samples are not M extra
    passes: the batch is split into M chunks, each chunk gets its own magnitude
    vector, and one forward/backward covers all of them.  A learned-policy arm
    and a fixed-policy arm therefore have IDENTICAL steps, batch size and
    encoder forward passes -- verified by matched_budget_check on the numbers
    READ BACK from the cell configs on disk after the run, not on the numbers
    the script intended to use (a resumed cell can carry a different --steps;
    the pre-run check cannot see that, the post-run audit can).  The per-chunk
    loss decomposition is used in the fixed arms too, so the encoder objective
    is bit-for-bit the same construction in every arm (a Gram loss over B/M rows
    is not the same object as a Gram loss over B rows; matching it matters).

3.  THE LEARNED MAGNITUDE VECTOR INCLUDES THE FLIP.  The brief listed four
    magnitudes (crop, colour, blur, gray).  With the flip probability pinned at
    0.5, T = identity is UNREACHABLE and the sharp prediction "the agreement
    term goes to exactly 0" cannot be tested.  All five magnitudes are learnable
    by default (`--d1-learn`), and m = 0 is the exact identity policy.

4.  AUGMENTATION IS DONE IN TENSOR SPACE, ON DEVICE, BY THIS FILE.  The harness
    builds its policies inside the torchvision dataset transform, which cannot
    be changed per optimisation step, let alone per chunk.  D1 requires exactly
    that.  `TensorAugment` below re-implements crop/flip/jitter/gray/blur on
    batched tensors, parameterised by a magnitude vector, and D2 uses the SAME
    augmenter with fixed magnitudes so that D1 and D2 are directly comparable.
    This is a deliberate, documented departure from `make_two_view_transform`.

5.  THE UNTRAINED-ENCODER FLOOR IS MEASURED.  "The probe collapses" is
    meaningless without the floor: a randomly initialised ResNet-18 already
    gives a far-from-chance linear probe on CIFAR-10.  D1 reports that floor.

6.  D2 CARRIES THREE CONTROLS THE BRIEF DID NOT ASK FOR, AND THEY MATTER.
    * `strong_no_color` / `strong_no_flip`: the strong policy with ONLY the
      offending operation removed.  Without them, "strong T loses" is consistent
      with "strong T is just too strong here", and the group argument is not
      tested.  With them, the damage is attributed to a named operation.
    * a from-scratch supervised baseline at the same fine-tuning budget, and a
      raw-pixel linear probe.  The pixel probe is the ceiling proving the label
      IS linearly present in the input, so any failure belongs to the encoder;
      the scratch baseline is what makes "fine-tuning recovers" interpretable.
    * THE G_task VIOLATION IS MEASURED, NOT ASSERTED (`verify_domain_violation`).
      For each domain a linear probe is trained and tested on PIXELS PUSHED
      THROUGH T itself.  "T leaves G_task" is then a number: the label survives
      T=conservative and does not survive T=strong, on the adversarial domains
      only.  Without this the whole of D2 rests on the author's claim that his
      own synthetic dataset does what he says, which is not a claim a reader
      should have to take on trust.  It runs before any pre-training, so a
      badly built domain costs seconds rather than a day of A100.

10. EVERY UNIT DIRECTION MUST BE REALISABLE AT THE SWEEP BUDGET.  Magnitudes
    live in the BOX [0,1]^5, so a budget c > 1 is unreachable for a
    concentrated direction (crop_only at c = 1.2 realises ||m|| = 1.0, not
    1.2) and "freeze the budget, vary the direction" would silently vary the
    budget too.  c = 1 is the largest budget at which EVERY unit direction is
    exactly realisable, so --c0 defaults to 1.0 and the realised norm of every
    cell is recorded and audited (`realised_mag_norm`); a direction-sweep cell
    that misses its budget is reported as such instead of being averaged in.

7.  HONEST LIMIT ON "IRRECOVERABLE".  With enough fine-tuning budget, nothing is
    strictly irrecoverable: the pixels still carry the label and the weights are
    not a hard bottleneck.  The defensible claim, and the one measured here, is
    that the destruction is irrecoverable UNDER THE FROZEN-ENCODER CONTRACT --
    which is the contract self-supervised pre-training actually sells.  The
    fine-tuning arm is reported as partial recovery at a fixed budget, never as
    "recoverable in principle".

8.  SCALE.  This is CIFAR-10, MNIST and two synthetic domains, ResNet-18, a few
    thousand steps.  It can decide questions of SIGN and ORDERING when the
    effects are large (they are, for T6).  It cannot decide small differences,
    it cannot be extrapolated to ImageNet-scale SSL, and a 2000-step
    pre-training is roughly 20 CIFAR epochs -- far short of the regime where
    published SSL numbers are produced.  Every claim below is a claim about
    this regime.

9.  STATISTICS.  Three seeds do not support a t-test.  Comparisons use a
    deliberately crude and visible rule -- a gap counts only if it exceeds the
    sum of the two seed standard deviations -- stated as such in the summary.

===============================================================================
COST
===============================================================================

Defaults (--steps 2000 --bs 512 --seeds 0 1 2), one A100 40GB, AMP bf16.
Cell counts are exact, not rounded: D1 has 2 phase-1 + 2 reference +
len(c_grid) sweep-c + (len(directions) - 1) sweep-direction cells per seed,
= (2 + 2 + 5 + 4) x 3 seeds = 39 pre-trainings.

    D1  39 pre-trainings  x ~2.5 min  ~1.6 h
        39 final probes + 3 untrained-encoder floors               ~0.3 h
    D2  36 pre-trainings  x ~2.5 min  ~1.5 h
        36 + 9 fine-tunings x ~1 min  ~0.8 h
        pixel probes, T-probes, final probes                       ~0.3 h
    ------------------------------------------------------------------------
    TOTAL                                                       ~4.5 - 7 h A100

    That range is wide on purpose: it is a FLOP estimate for a ResNet-18 with
    the CIFAR stem (stride-1, no maxpool: ~0.56 GFLOP/image at 32x32) at an
    assumed 80-120 achieved TFLOPS bf16, and the augmentation pipeline runs in
    fp32 on top.  Measure one cell before trusting the total.

    `--steps 6000` (a more respectable ~60 CIFAR epochs) costs ~3x, i.e.
    ~15-20 h A100, and is the setting to use before showing numbers to anyone.
    `--smoke` runs the whole pipeline on CPU in under 3 minutes with tiny
    tensors and a cheap encoder stem; it proves the plumbing, nothing else.

DISK.  Checkpoints are ~180 MB per cell (encoder + projector + AdamW state).
The step checkpoint is DELETED once a cell writes its summary, and `best.pt`
is never produced (the "best" of a per-batch SSL loss is meaningless -- we want
the LAST weights, and only D2 needs the encoder afterwards, so only D2 keeps
one).  Peak footprint is therefore a few hundred MB plus ~1.6 GB of D2
encoders, not the ~27 GB a naive keep-everything policy would write.

Everything is resumable: each cell writes its CSV row by row (flushed), keeps an
atomic checkpoint, and is skipped on restart once its summary.json exists.  A
skipped cell's stored config is CHECKED against the one being requested: a cell
pre-trained at --steps 2000 is refused as a stand-in for a --steps 6000 arm
instead of being silently averaged into an unmatched comparison.

Usage
-----
    python xp/xp_D_augmentations.py --smoke
    python xp/xp_D_augmentations.py --all
    python xp/xp_D_augmentations.py --exp D2 --seeds 0 1 2
    python xp/xp_D_augmentations.py --exp D1 --arm cross          # substring filter
    python xp/xp_D_augmentations.py --figures-only                # re-plot from disk
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

import pandas as pd

# --- in-house imports: the harness and nothing else -------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from lib.harness import (  # noqa: E402
    AugCfg,
    Run,
    aggregate_seeds,
    effective_rank,
    get_device,
    gram_loss,
    knn_probe,
    linear_probe,
    make_encoder,
    make_mlp,
    matched_budget_check,
    set_seed,
    vicreg_reg,
)
from lib.harness import NORM_MEAN, NORM_STD  # noqa: E402  (module constants, not in __all__)


# ===========================================================================
# 0.  Decision thresholds -- all of them, in one place, so a sceptic can see
#     exactly what "confirmed" was allowed to mean BEFORE the run.
# ===========================================================================

# P5.1: the minimising policy must shrink to at most this fraction of its
# initial magnitude norm to count as "degenerated to the identity".
CONFIRM_MIN_COLLAPSE_FRAC = 0.25
# ... and it is falsified if it keeps at least this fraction.
FALSIFY_MIN_KEEP_FRAC = 0.75
# P5.1: the agreement term must fall to at most this fraction of its initial value.
CONFIRM_AGREE_COLLAPSE_FRAC = 0.10

# P5.2: budget dominates the policy if the direction sweep moves the probe by at
# most this fraction of what the budget sweep moves it.  The ratio is computed
# TWICE -- over the full budget grid and over its strictly positive part -- and
# the LARGER (least favourable to T5) of the two must clear the threshold.
CONFIRM_DIR_OVER_C_RATIO = 0.50
# ... and T5's "the prior moved into the budget" is falsified above this ratio
# WHEN the learned direction is also the best of the grid.
FALSIFY_DIR_OVER_C_RATIO = 1.00
# A direction-sweep cell whose realised ||m|| misses the frozen budget c0 by
# more than this is not a constant-budget cell and is flagged in the verdict.
BUDGET_REALISATION_TOL = 1e-6

# P6.1: a domain counts as "strong T loses" if the drop exceeds this (absolute
# accuracy), and as "strong T wins" if the gain exceeds it.  A domain must ALSO
# clear the seed-noise rule below to be counted either way.
CONFIRM_CROSSING_MARGIN = 0.05
# P6.2: fine-tuning must recover at least this fraction of the frozen damage.
CONFIRM_RECOVERY_FRACTION = 0.30
# verify_domain_violation: T "destroys the label" if a probe on T(x) drops to
# within this of chance, and "preserves" it if it stays this far above chance.
VIOLATION_DESTROYS_WITHIN = 0.10
VIOLATION_PRESERVES_ABOVE = 0.20

VERDICT_CONFIRMED = "confirmee"
VERDICT_REFUTED = "infirmee"
VERDICT_INCONCLUSIVE = "non concluante"


def verdict(confirmed: bool, refuted: bool) -> str:
    """Three-valued verdict.  Both true is a bug in the caller's thresholds."""
    if confirmed and refuted:
        raise ValueError("contradictory verdict conditions")
    if confirmed:
        return VERDICT_CONFIRMED
    if refuted:
        return VERDICT_REFUTED
    return VERDICT_INCONCLUSIVE


def gap_is_meaningful(mean_a: float, std_a: float, mean_b: float, std_b: float) -> bool:
    """Crude, visible significance rule for 3 seeds: the gap must exceed the
    sum of the two standard deviations.  This is NOT a test; it is a guard
    against reporting a difference smaller than the seed noise."""
    if any(math.isnan(v) for v in (mean_a, std_a, mean_b, std_b)):
        return False
    return abs(mean_a - mean_b) > (abs(std_a) + abs(std_b))


# ===========================================================================
# 1.  Magnitude parameterisation of the augmentation policy
# ===========================================================================
#
# The policy is a point m in [0,1]^5.  m = 0 is EXACTLY the identity policy
# (resize only, no randomness at all), which is the degenerate optimum of T5.
# Each coordinate is mapped monotonically onto a physical parameter.

MAG_NAMES: Tuple[str, ...] = ("crop", "color", "gray", "blur", "flip")
N_MAG = len(MAG_NAMES)

CROP_MIN_AREA = 0.08          # m_crop = 1  ->  RandomResizedCrop scale (0.08, 1.0)
MAX_LOG_RATIO = math.log(4.0 / 3.0)
MAX_GRAY_P = 0.5              # m_gray  = 0.4 -> p(grayscale) = 0.2  (DINO/SimCLR value)
MAX_BLUR_P = 1.0              # m_blur  = 0.5 -> p(blur) = 0.5
MAX_FLIP_P = 0.5              # m_flip  = 1.0 -> p(hflip) = 0.5
COLOR_APPLY_P = 0.8           # SimCLR applies its jitter with probability 0.8
MAG_FLOOR = 1e-3              # logit parameterisation cannot reach 0 exactly


# Reference policies, expressed in the same magnitude space so that everything
# in this file -- learned or hand-set -- is one object of the same type.
REFERENCE_POLICIES: Dict[str, np.ndarray] = {
    #                       crop  color  gray  blur  flip
    "identity":  np.array([0.00, 0.00, 0.00, 0.00, 0.00], dtype=np.float64),
    "conservative": np.array([0.45, 0.00, 0.00, 0.00, 0.00], dtype=np.float64),
    "strong":    np.array([0.87, 1.00, 0.40, 0.50, 1.00], dtype=np.float64),
    # strong with ONLY the colour channel neutralised (grayscale destroys hue
    # too, so it must go as well) -- the control for the colored_shapes domain.
    "strong_no_color": np.array([0.87, 0.00, 0.00, 0.50, 1.00], dtype=np.float64),
    # strong with ONLY the reflection neutralised -- control for chirality.
    "strong_no_flip": np.array([0.87, 1.00, 0.40, 0.50, 0.00], dtype=np.float64),
}


def magnitudes_to_augcfg(m: Sequence[float], size: int) -> AugCfg:
    """Human-readable view of a magnitude vector, in the harness' own AugCfg.

    Used only for logging and for the config JSON: the actual augmentation is
    performed by TensorAugment (see amendment 4 in the module docstring).  The
    physical parameters are identical, so the AugCfg printed in the config is a
    faithful description of what was applied.
    """
    m = np.clip(np.asarray(m, dtype=np.float64), 0.0, 1.0)
    scale_min = 1.0 - (1.0 - CROP_MIN_AREA) * float(m[0])
    return AugCfg(
        crop_scale=(scale_min, 1.0),
        color_strength=float(m[1]),
        gray_p=MAX_GRAY_P * float(m[2]),
        blur_p=MAX_BLUR_P * float(m[3]),
        hflip_p=MAX_FLIP_P * float(m[4]),
        solarize_p=0.0,
        size=int(size),
    )


def clip_to_budget(m: torch.Tensor, budget: Optional[float]) -> torch.Tensor:
    """Project rows of ``m`` onto the L2 ball of radius ``budget``, then onto
    the box [0,1]^5.  ``budget=None`` means unconstrained.

    The projection is applied to every REALISED magnitude vector, so the budget
    constraint holds at every single step, not merely in expectation.  The
    score-function estimator remains correct: the rescaling is a deterministic
    map applied after sampling, exactly like the sigmoid.
    """
    if budget is None:
        return m.clamp(0.0, 1.0)
    norm = m.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = torch.clamp(float(budget) / norm, max=1.0)
    return (m * scale).clamp(0.0, 1.0)


def direction_vector(name: str, seed: int, learned: Optional[np.ndarray] = None) -> np.ndarray:
    """Unit vectors in magnitude space used by the D1 direction sweep.

    ``learned`` is the magnitude vector recovered from the adversarial arm; the
    other directions are deliberately very different from it, so that "the
    direction does not matter" is a strong statement if it comes out true.
    """
    if name == "learned":
        if learned is None:
            raise ValueError("the 'learned' direction requires a learned magnitude vector")
        v = np.asarray(learned, dtype=np.float64)
    elif name == "uniform":
        v = np.ones(N_MAG, dtype=np.float64)
    elif name == "crop_only":
        v = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    elif name == "color_only":
        v = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    elif name == "random":
        v = np.random.RandomState(90000 + int(seed)).uniform(0.1, 1.0, size=N_MAG)
    else:
        raise ValueError(f"unknown direction {name!r}")
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError(f"direction {name!r} is the zero vector")
    return v / n


# ===========================================================================
# 2.  TensorAugment -- the augmentation pipeline, on device, per-sample
#     magnitudes.  Input: float images in [0,1], [B,3,H,W].
#     Output: normalised views, [B,3,size,size].
# ===========================================================================

class TensorAugment(nn.Module):
    """Batched augmentation whose strength is a per-sample vector m in [0,1]^5.

    Operations, in this fixed order:
      1. random resized crop + horizontal flip  (single affine grid_sample)
      2. colour jitter: brightness, contrast, saturation, hue  (prob 0.8)
      3. random grayscale
      4. gaussian blur
      5. normalisation with the harness' NORM_MEAN / NORM_STD

    Two properties matter and are asserted in the self-check:
      * m = 0 gives the exact identity (a plain resize, no randomness);
      * every operation is monotone in its own magnitude.

    torchvision randomises the order of the four jitter operations.  We do not:
    a fixed order costs nothing conceptually and keeps this code readable.
    """

    def __init__(self, size: int, blur_kernel: Optional[int] = None):
        super().__init__()
        self.size = int(size)
        if blur_kernel is None:
            blur_kernel = max(3, int(0.1 * self.size))
            if blur_kernel % 2 == 0:
                blur_kernel += 1
        self.blur_kernel = int(blur_kernel)

        self.register_buffer("norm_mean", torch.tensor(NORM_MEAN).view(1, 3, 1, 1))
        self.register_buffer("norm_std", torch.tensor(NORM_STD).view(1, 3, 1, 1))
        self.register_buffer("luma", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        # RGB <-> YIQ, used for the hue rotation (rotation around the grey axis).
        self.register_buffer("to_yiq", torch.tensor([
            [0.299, 0.587, 0.114],
            [0.596, -0.274, -0.322],
            [0.211, -0.523, 0.312],
        ]))
        self.register_buffer("from_yiq", torch.tensor([
            [1.000, 0.956, 0.621],
            [1.000, -0.272, -0.647],
            [1.000, -1.106, 1.703],
        ]))
        blur_coords = torch.arange(self.blur_kernel).float() - (self.blur_kernel // 2)
        self.register_buffer("blur_coords", blur_coords)

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _u(shape, device) -> torch.Tensor:
        return torch.rand(shape, device=device)

    def _crop_flip(self, x: torch.Tensor, m_crop: torch.Tensor,
                   m_flip: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        dev = x.device
        scale_min = 1.0 - (1.0 - CROP_MIN_AREA) * m_crop            # [B]
        area = scale_min + (1.0 - scale_min) * self._u(B, dev)
        # The aspect-ratio jitter is scaled by m_crop as well, so that m_crop=0
        # gives w = h = 1 exactly, i.e. the identity crop.
        log_ratio = (2.0 * self._u(B, dev) - 1.0) * MAX_LOG_RATIO * m_crop
        ratio = torch.exp(log_ratio)
        w = torch.sqrt(area * ratio).clamp(max=1.0)
        h = torch.sqrt(area / ratio).clamp(max=1.0)
        tx = (1.0 - w) * (2.0 * self._u(B, dev) - 1.0)
        ty = (1.0 - h) * (2.0 * self._u(B, dev) - 1.0)
        flip = torch.where(self._u(B, dev) < MAX_FLIP_P * m_flip,
                           -torch.ones(B, device=dev), torch.ones(B, device=dev))

        theta = torch.zeros(B, 2, 3, device=dev, dtype=x.dtype)
        theta[:, 0, 0] = w * flip
        theta[:, 0, 2] = tx
        theta[:, 1, 1] = h
        theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, (B, x.shape[1], self.size, self.size),
                             align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection",
                             align_corners=False)

    def _color_jitter(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        dev = x.device
        apply = ((self._u(B, dev) < COLOR_APPLY_P) & (s > 0)).view(B, 1, 1, 1)
        s4 = s.view(B, 1, 1, 1)

        def _factor():
            return 1.0 + (2.0 * self._u(B, dev).view(B, 1, 1, 1) - 1.0) * 0.8 * s4

        y = x * _factor()                                            # brightness
        mean_gray = (y * self.luma).sum(dim=1, keepdim=True).mean(dim=(2, 3), keepdim=True)
        y = mean_gray + (y - mean_gray) * _factor()                  # contrast
        gray = (y * self.luma).sum(dim=1, keepdim=True)
        y = gray + (y - gray) * _factor()                            # saturation

        hue_max = torch.clamp(0.2 * s, max=0.5)                      # in turns
        angle = (2.0 * self._u(B, dev) - 1.0) * hue_max * (2.0 * math.pi)
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        rot = torch.zeros(B, 3, 3, device=dev, dtype=x.dtype)
        rot[:, 0, 0] = 1.0
        rot[:, 1, 1] = cos_a
        rot[:, 1, 2] = -sin_a
        rot[:, 2, 1] = sin_a
        rot[:, 2, 2] = cos_a
        mat = self.from_yiq.to(x.dtype) @ rot @ self.to_yiq.to(x.dtype)   # [B,3,3]
        y = torch.einsum("bij,bjhw->bihw", mat, y)                   # hue

        y = y.clamp(0.0, 1.0)
        return torch.where(apply, y, x)

    def _grayscale(self, x: torch.Tensor, m_gray: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        apply = (self._u(B, x.device) < MAX_GRAY_P * m_gray).view(B, 1, 1, 1)
        gray = (x * self.luma).sum(dim=1, keepdim=True).expand_as(x)
        return torch.where(apply, gray, x)

    def _blur(self, x: torch.Tensor, m_blur: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        dev = x.device
        apply = (self._u(B, dev) < MAX_BLUR_P * m_blur)
        if not bool(apply.any()):
            return x
        sigma = 0.1 + 1.9 * self._u(B, dev)                          # [B]
        coords = self.blur_coords.to(x.dtype)
        w = torch.exp(-(coords.view(1, -1) ** 2) / (2.0 * sigma.view(-1, 1) ** 2))
        w = w / w.sum(dim=1, keepdim=True)                           # [B,k]
        k = self.blur_kernel
        wc = w.repeat_interleave(C, dim=0)                           # [B*C,k]
        xf = x.reshape(1, B * C, H, W)
        xf = F.conv2d(xf, wc.view(B * C, 1, 1, k), groups=B * C, padding=(0, k // 2))
        xf = F.conv2d(xf, wc.view(B * C, 1, k, 1), groups=B * C, padding=(k // 2, 0))
        blurred = xf.reshape(B, C, H, W)
        return torch.where(apply.view(B, 1, 1, 1), blurred, x)

    # -- public ------------------------------------------------------------
    def forward(self, x01: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """``x01``: [B,3,H,W] in [0,1].  ``m``: [B,5] or [5]."""
        if m.dim() == 1:
            m = m.view(1, -1).expand(x01.shape[0], -1)
        if m.shape != (x01.shape[0], N_MAG):
            raise ValueError(f"m must be [B,{N_MAG}], got {tuple(m.shape)}")
        m = m.to(device=x01.device, dtype=x01.dtype).clamp(0.0, 1.0)

        y = self._crop_flip(x01, m[:, 0], m[:, 4])
        y = self._color_jitter(y, m[:, 1])
        y = self._grayscale(y, m[:, 2])
        y = self._blur(y, m[:, 3])
        return (y - self.norm_mean.to(y.dtype)) / self.norm_std.to(y.dtype)

    def normalise_only(self, x01: torch.Tensor) -> torch.Tensor:
        """Deterministic path: resize to ``size`` and normalise.  This is what
        the evaluation datasets use, so the probe sees the same input
        distribution as training does at m = 0."""
        if x01.shape[-1] != self.size or x01.shape[-2] != self.size:
            x01 = F.interpolate(x01, size=(self.size, self.size),
                                mode="bilinear", align_corners=False)
        return (x01 - self.norm_mean.to(x01.dtype)) / self.norm_std.to(x01.dtype)


# ===========================================================================
# 3.  The learnable policy: a diagonal Gaussian over logit-magnitudes
# ===========================================================================

class MagnitudePolicy(nn.Module):
    """pi(u) = N(mu, sigma^2 I) on R^5, magnitudes m = sigmoid(u) in (0,1)^5.

    Only ``mu`` is learned.  ``sigma`` is FIXED on purpose: if the policy could
    also shrink its own exploration noise, "the policy collapsed" would be
    ambiguous between "the magnitudes went to zero" (the T5 claim) and "the
    estimator ran out of signal" (an optimisation artefact).  With sigma fixed,
    a collapse can only be a collapse of the mean magnitudes.

    The score-function gradient uses log pi(u) with u sampled and detached; the
    deterministic maps applied afterwards (sigmoid, budget projection) do not
    enter the estimator, which is what makes it black-box valid.
    """

    def __init__(self, init_m: Sequence[float], sigma: float = 0.5,
                 learn_mask: Optional[Sequence[bool]] = None):
        super().__init__()
        m0 = np.clip(np.asarray(init_m, dtype=np.float64), MAG_FLOOR, 1.0 - MAG_FLOOR)
        u0 = np.log(m0 / (1.0 - m0))
        self.mu = nn.Parameter(torch.tensor(u0, dtype=torch.float32))
        self.register_buffer("sigma", torch.tensor(float(sigma)))
        mask = np.ones(N_MAG, dtype=np.float32) if learn_mask is None \
            else np.asarray(learn_mask, dtype=np.float32)
        self.register_buffer("learn_mask", torch.tensor(mask))

    def sample(self, n: int, budget: Optional[float]
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw ``n`` magnitude vectors.  Returns ``(m [n,5], logp [n])``.

        ``logp`` carries gradient with respect to ``mu`` only; ``m`` is a
        detached quantity, as it must be for a score-function estimator.
        """
        dev = self.mu.device
        eps = torch.randn(n, N_MAG, device=dev)
        u = (self.mu.detach().view(1, -1) + self.sigma * eps)
        # Components that are not being learned stay pinned at their mean.
        u = torch.where(self.learn_mask.view(1, -1) > 0, u,
                        self.mu.detach().view(1, -1).expand_as(u))
        logp = (-0.5 * ((u - self.mu.view(1, -1)) / self.sigma) ** 2
                - torch.log(self.sigma) - 0.5 * math.log(2 * math.pi))
        logp = (logp * self.learn_mask.view(1, -1)).sum(dim=1)
        m = torch.sigmoid(u).detach()
        return clip_to_budget(m, budget), logp

    @torch.no_grad()
    def mean_magnitudes(self, budget: Optional[float] = None) -> torch.Tensor:
        m = torch.sigmoid(self.mu).view(1, -1)
        return clip_to_budget(m, budget).view(-1)

    @torch.no_grad()
    def project_(self, budget: Optional[float]) -> None:
        """Projected gradient step: pull ``mu`` back inside the budget ball.

        This makes the optimisation a constrained one (biased with respect to
        the unconstrained estimator, as any projection is); that is the intended
        semantics -- the budget is a hard constraint of the experiment, not a
        penalty to be traded off.
        """
        if budget is None:
            return
        m = torch.sigmoid(self.mu)
        norm = float(m.norm())
        if norm <= budget:
            return
        m = (m * (budget / max(norm, 1e-12))).clamp(MAG_FLOOR, 1.0 - MAG_FLOOR)
        self.mu.copy_(torch.log(m / (1.0 - m)))

    @torch.no_grad()
    def clamp_(self) -> None:
        """Keep mu in a range where sigmoid is not numerically saturated."""
        lo = math.log(MAG_FLOOR / (1.0 - MAG_FLOOR))
        hi = -lo
        self.mu.clamp_(lo, hi)


# ===========================================================================
# 4.  Domains.  Everything is materialised once into uint8 CPU tensors, so the
#     training loop has no PIL in its hot path and is exactly reproducible.
# ===========================================================================

class ImageTensorDataset(torch.utils.data.Dataset):
    """uint8 [N,3,S,S] + int64 labels.

    ``normalise=True`` returns the deterministic evaluation view (float,
    normalised) as a plain ``(x, y)`` tuple, which the harness probes accept.
    ``normalise=False`` returns raw [0,1] floats for the SSL loop.
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, normalise: bool):
        assert images.dtype == torch.uint8 and images.dim() == 4
        self.images = images
        self.labels = labels.long()
        self.normalise = bool(normalise)
        self._mean = torch.tensor(NORM_MEAN).view(3, 1, 1)
        self._std = torch.tensor(NORM_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, i: int):
        x = self.images[i].float() / 255.0
        if self.normalise:
            x = (x - self._mean) / self._std
        return x, int(self.labels[i])


@dataclass
class Domain:
    """A domain, with its known invariance group stated explicitly."""
    name: str
    pool: torch.Tensor                  # uint8 [N,3,S,S], the SSL pre-training pool
    train_images: torch.Tensor
    train_labels: torch.Tensor
    test_images: torch.Tensor
    test_labels: torch.Tensor
    num_classes: int
    g_task_note: str                    # what the label IS invariant to
    offending_magnitude: Optional[str]  # which coordinate of m violates G_task

    def eval_datasets(self) -> Tuple[ImageTensorDataset, ImageTensorDataset]:
        return (ImageTensorDataset(self.train_images, self.train_labels, normalise=True),
                ImageTensorDataset(self.test_images, self.test_labels, normalise=True))


def _resize_u8(x: torch.Tensor, size: int) -> torch.Tensor:
    """uint8 [N,C,H,W] -> uint8 [N,C,size,size] (bilinear, done in float)."""
    if x.shape[-1] == size and x.shape[-2] == size:
        return x
    out = []
    for i in range(0, x.shape[0], 2048):
        chunk = x[i:i + 2048].float() / 255.0
        chunk = F.interpolate(chunk, size=(size, size), mode="bilinear",
                              align_corners=False)
        out.append((chunk.clamp(0, 1) * 255.0).round().to(torch.uint8))
    return torch.cat(out, 0)


# --- synthetic domains ------------------------------------------------------

def _shape_mask(kind: int, size: int, rng: np.random.RandomState) -> np.ndarray:
    """A random filled shape, position and size independent of any label."""
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    r = rng.uniform(0.18, 0.34) * size
    cy = rng.uniform(r, size - r)
    cx = rng.uniform(r, size - r)
    if kind == 0:                                   # disc
        return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
    if kind == 1:                                   # square
        return (np.abs(yy - cy) <= r) & (np.abs(xx - cx) <= r)
    if kind == 2:                                   # diamond
        return (np.abs(yy - cy) + np.abs(xx - cx)) <= 1.35 * r
    # cross
    t = 0.38 * r
    return ((np.abs(yy - cy) <= t) & (np.abs(xx - cx) <= r)) | \
           ((np.abs(xx - cx) <= t) & (np.abs(yy - cy) <= r))


def _chiral_glyph(size: int, rng: np.random.RandomState) -> np.ndarray:
    """An 'F'-like glyph: one vertical bar plus two arms on the SAME side.

    The glyph has no reflection symmetry, so mirroring it is detectable, and
    detecting it is exactly what a hflip-invariant encoder cannot do.
    """
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    h = rng.uniform(0.45, 0.70) * size
    w = rng.uniform(0.28, 0.45) * size
    t = max(1.0, rng.uniform(0.07, 0.12) * size)
    y0 = rng.uniform(0, size - h)
    x0 = rng.uniform(0, size - w)
    stem = (xx >= x0) & (xx <= x0 + t) & (yy >= y0) & (yy <= y0 + h)
    arm_top = (yy >= y0) & (yy <= y0 + t) & (xx >= x0) & (xx <= x0 + w)
    arm_mid = (yy >= y0 + 0.42 * h) & (yy <= y0 + 0.42 * h + t) & \
              (xx >= x0) & (xx <= x0 + 0.7 * w)
    return stem | arm_top | arm_mid


def _oriented_texture(size: int, cls: int, num_classes: int,
                      rng: np.random.RandomState) -> np.ndarray:
    """Class-dependent oriented grating; the colour is a nuisance.  Used only as
    the offline stand-in for CIFAR-10 in --smoke."""
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    theta = math.pi * cls / num_classes
    freq = rng.uniform(0.15, 0.30)
    phase = rng.uniform(0, 2 * math.pi)
    g = np.sin(2 * math.pi * freq * (xx * math.cos(theta) + yy * math.sin(theta)) + phase)
    return 0.5 + 0.35 * g


def make_synthetic(mode: str, n: int, size: int, num_classes: int,
                   seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Deterministic synthetic images.

    mode='hue'      : the label is the HUE bin.  Shape kind, size, position,
                      saturation, value and background are nuisances.  The hue
                      jitter inside each bin is +/- 0.35 bin widths, so the bins
                      never overlap: the task is exactly solvable from colour.
    mode='chiral'   : the label is "the glyph is mirrored".  Colour, position,
                      size and thickness are nuisances.
    mode='texture'  : the label is the orientation of a grating (CIFAR stand-in).
    """
    rng = np.random.RandomState(int(seed))
    imgs = np.empty((n, size, size, 3), dtype=np.float32)
    labels = np.empty(n, dtype=np.int64)

    for i in range(n):
        bg = rng.uniform(0.10, 0.50)
        img = np.full((size, size, 3), bg, dtype=np.float32)
        img += rng.normal(0.0, 0.05, size=(size, size, 3)).astype(np.float32)

        if mode == "hue":
            lab = int(rng.randint(num_classes))
            hue = (lab + rng.uniform(-0.35, 0.35)) / float(num_classes)
            rgb = hsv_to_rgb(np.array([hue % 1.0,
                                       rng.uniform(0.75, 1.0),
                                       rng.uniform(0.70, 1.0)]))
            mask = _shape_mask(int(rng.randint(4)), size, rng)
            img[mask] = rgb.astype(np.float32)
        elif mode == "chiral":
            lab = int(rng.randint(2))
            rgb = hsv_to_rgb(np.array([rng.uniform(0, 1), rng.uniform(0.2, 1.0),
                                       rng.uniform(0.70, 1.0)]))
            mask = _chiral_glyph(size, rng)
            if lab == 1:
                mask = mask[:, ::-1]
            img[mask] = rgb.astype(np.float32)
        elif mode == "texture":
            lab = int(rng.randint(num_classes))
            g = _oriented_texture(size, lab, num_classes, rng)
            tint = hsv_to_rgb(np.array([rng.uniform(0, 1), rng.uniform(0.1, 0.6), 1.0]))
            img = (g[..., None] * tint[None, None, :]).astype(np.float32)
            img += rng.normal(0.0, 0.03, size=(size, size, 3)).astype(np.float32)
        else:
            raise ValueError(f"unknown synthetic mode {mode!r}")

        imgs[i] = np.clip(img, 0.0, 1.0)
        labels[i] = lab

    x = torch.from_numpy((imgs * 255.0).round().astype(np.uint8)).permute(0, 3, 1, 2).contiguous()
    return x, torch.from_numpy(labels)


# --- real domains -----------------------------------------------------------

def _cache_path(root: str, key: str) -> str:
    d = os.path.join(root, "_ubergang_D_cache")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{key}.pt")


def _load_cifar10(root: str, size: int) -> Tuple[torch.Tensor, torch.Tensor,
                                                 torch.Tensor, torch.Tensor]:
    import torchvision
    cache = _cache_path(root, f"cifar10_{size}")
    if os.path.exists(cache):
        d = torch.load(cache, map_location="cpu", weights_only=False)
        return d["xtr"], d["ytr"], d["xte"], d["yte"]
    tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
    te = torchvision.datasets.CIFAR10(root, train=False, download=True)
    xtr = _resize_u8(torch.from_numpy(tr.data).permute(0, 3, 1, 2).contiguous(), size)
    xte = _resize_u8(torch.from_numpy(te.data).permute(0, 3, 1, 2).contiguous(), size)
    ytr = torch.tensor(tr.targets, dtype=torch.long)
    yte = torch.tensor(te.targets, dtype=torch.long)
    torch.save({"xtr": xtr, "ytr": ytr, "xte": xte, "yte": yte}, cache)
    return xtr, ytr, xte, yte


CHIRALITY_DIGITS = (2, 3, 4, 5, 6, 7, 9)
"""Digits kept for the chirality task.

0, 1 and 8 are close to mirror-symmetric, so 'is it mirrored' is undecidable on
them and they would only add an irreducible error floor that has nothing to do
with the thesis.  Dropping them is a modelling choice, stated here rather than
buried; --chirality-all-digits keeps them if you want to see the floor.
"""


def _load_mirror_mnist(root: str, size: int, seed: int, all_digits: bool
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    import torchvision
    key = f"mirrormnist_{size}_{seed}_{int(all_digits)}"
    cache = _cache_path(root, key)
    if os.path.exists(cache):
        d = torch.load(cache, map_location="cpu", weights_only=False)
        return d["xtr"], d["ytr"], d["xte"], d["yte"]

    out = []
    for train in (True, False):
        ds = torchvision.datasets.MNIST(root, train=train, download=True)
        x = ds.data.clone()                              # uint8 [N,28,28]
        y = ds.targets.clone().numpy()
        if not all_digits:
            keep = np.isin(y, np.array(CHIRALITY_DIGITS))
            x = x[torch.from_numpy(keep)]
        rng = np.random.RandomState(int(seed) + (0 if train else 777))
        mirrored = rng.rand(x.shape[0]) < 0.5
        idx = torch.from_numpy(np.where(mirrored)[0])
        x[idx] = torch.flip(x[idx], dims=[2])
        x3 = _resize_u8(x.unsqueeze(1).repeat(1, 3, 1, 1), size)
        out.append((x3, torch.from_numpy(mirrored.astype(np.int64))))
    (xtr, ytr), (xte, yte) = out
    torch.save({"xtr": xtr, "ytr": ytr, "xte": xte, "yte": yte}, cache)
    return xtr, ytr, xte, yte


D2_DOMAINS = ("cifar10", "colored_shapes", "chirality")


def build_domain(name: str, size: int, root: str, seed: int, synthetic: bool,
                 n_train: int, n_test: int, all_digits: bool = False) -> Domain:
    """Materialise a domain.  ``synthetic=True`` replaces every download by an
    offline generator; it is the --smoke path and proves nothing scientific."""
    if name == "cifar10":
        if synthetic:
            xtr, ytr = make_synthetic("texture", n_train, size, 10, seed)
            xte, yte = make_synthetic("texture", n_test, size, 10, seed + 5000)
        else:
            xtr, ytr, xte, yte = _load_cifar10(root, size)
        return Domain("cifar10", xtr, xtr, ytr, xte, yte, 10,
                      g_task_note=("natural object classes; label invariant to crop, "
                                   "flip, colour and blur -- T is contained in G_task"),
                      offending_magnitude=None)

    if name == "colored_shapes":
        n_cls = 8
        xtr, ytr = make_synthetic("hue", n_train, size, n_cls, seed)
        xte, yte = make_synthetic("hue", n_test, size, n_cls, seed + 5000)
        return Domain("colored_shapes", xtr, xtr, ytr, xte, yte, n_cls,
                      g_task_note=("the label IS the hue; invariant to shape, "
                                   "position, size, crop, flip -- NOT to colour"),
                      offending_magnitude="color")

    if name == "chirality":
        if synthetic:
            xtr, ytr = make_synthetic("chiral", n_train, size, 2, seed)
            xte, yte = make_synthetic("chiral", n_test, size, 2, seed + 5000)
        else:
            xtr, ytr, xte, yte = _load_mirror_mnist(root, size, seed, all_digits)
        return Domain("chirality", xtr, xtr, ytr, xte, yte, 2,
                      g_task_note=("the label IS the chirality; invariant to colour, "
                                   "crop, blur -- NOT to horizontal reflection"),
                      offending_magnitude="flip")

    raise ValueError(f"unknown domain {name!r}")


# ===========================================================================
# 5.  Training configuration and the shared SSL loop
# ===========================================================================

@dataclass
class TrainCfg:
    steps: int = 2000
    bs: int = 512
    chunks: int = 8                   # M: magnitude samples per step (see amendment 2)
    lr: float = 1e-3
    wd: float = 1e-6
    warmup_frac: float = 0.05
    policy_lr: float = 0.05
    policy_sigma: float = 0.5
    advantage_normalise: bool = True
    policy_objective: str = "full"    # 'full' (thesis wording) or 'agreement'
    size: int = 32
    arch: str = "resnet18"
    stem: str = "cifar"
    proj_hidden: int = 1024
    proj_dim: int = 256
    gram_w: float = 25.0
    var_w: float = 25.0
    cov_w: float = 1.0
    amp: bool = True
    log_every: int = 25
    eval_every: int = 500
    ckpt_every: int = 200
    probe_epochs: int = 100
    probe_lr: float = 1e-2
    finetune_epochs: int = 20
    finetune_lr: float = 1e-3
    knn_subsample: int = 10000        # cap for the mid-run kNN diagnostic
    eval_loss_batches: int = 8        # fixed-augmentation batches used to score the
                                      # SSL loss at a common reference T. Without a
                                      # COMMON T the arms' losses are not comparable:
                                      # arm (a) drives T->identity, which lowers its own
                                      # loss by changing the measuring stick.

    def forward_passes(self) -> int:
        return 2                      # two views, always; the policy is free


def build_model(cfg: TrainCfg, device: torch.device) -> Tuple[nn.Module, nn.Module, int]:
    encoder, feat_dim = make_encoder(cfg.arch, cfg.stem)
    projector = make_mlp(feat_dim, cfg.proj_hidden, cfg.proj_dim, n_layers=2, bn=True)
    return encoder.to(device), projector.to(device), feat_dim


def cosine_lr(step: int, total: int, base_lr: float, warmup_frac: float) -> float:
    warm = max(1, int(warmup_frac * total))
    if step < warm:
        return base_lr * (step + 1) / warm
    t = (step - warm) / max(1, total - warm)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


def _rng_state() -> dict:
    st = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st: dict) -> None:
    torch.set_rng_state(st["torch"])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    if torch.cuda.is_available() and "cuda" in st:
        try:
            torch.cuda.set_rng_state_all(st["cuda"])
        except Exception as e:                                   # pragma: no cover
            print(f"[resume] could not restore the CUDA RNG state ({e}); continuing.")


def per_chunk_losses(z1: torch.Tensor, z2: torch.Tensor, chunks: int, cfg: TrainCfg
                     ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Split the batch into ``chunks`` and compute the SSL loss on each.

    Returns ``(total_per_chunk [M], agreement_per_chunk [M], diagnostics)``.

    The objective is the project's own: a relational Gram term (T2) plus the
    VICReg legality term on the UNNORMALISED projector output.  Note that the
    Gram term is exactly 0 when the two views coincide, for ANY encoder -- which
    is precisely why T = identity is the trivial optimum of T5.  Nothing here
    prevents that; the anti-collapse regulariser constrains the encoder, not T.
    """
    M = int(chunks)
    n = z1.shape[0]
    if n % M != 0:
        raise ValueError(f"batch {n} not divisible by chunks {M}")
    per = n // M
    totals, agrees, vars_, covs = [], [], [], []
    for j in range(M):
        a = z1[j * per:(j + 1) * per]
        b = z2[j * per:(j + 1) * per]
        # Symmetrised: gram_loss detaches its second argument, so a single call
        # would give gradient to one branch only.
        agree = 0.5 * (gram_loss(a, b) + gram_loss(b, a))
        reg, parts = vicreg_reg(torch.cat([a, b], dim=0),
                                var_w=cfg.var_w, cov_w=cfg.cov_w)
        totals.append(cfg.gram_w * agree + reg)
        agrees.append(agree)
        vars_.append(parts["var"])
        covs.append(parts["cov"])
    diag = {"var": float(np.mean(vars_)), "cov": float(np.mean(covs))}
    return torch.stack(totals), torch.stack(agrees), diag


# Fields of the stored config that MUST agree for a cached cell to be reused as
# a member of a budget-matched comparison.  A cell trained at other values is a
# different experiment wearing the same file name.
_RESUME_CRITICAL_TRAIN_KEYS = ("steps", "bs", "chunks", "size", "arch", "stem",
                               "proj_hidden", "proj_dim", "gram_w", "var_w",
                               "cov_w", "lr", "policy_lr", "policy_sigma",
                               "policy_objective", "advantage_normalise")


def _check_cached_cell(blob: dict, cell_name: str, cfg: TrainCfg,
                       policy_mode: str, init_m: np.ndarray,
                       budget: Optional[float]) -> None:
    """Refuse a cached cell that was produced by a different configuration.

    Skipping a finished cell is the Colab resume contract; skipping a cell that
    was trained with OTHER hyper-parameters and then averaging it with freshly
    trained arms is an unmatched-budget comparison that nothing downstream can
    detect.  The stored config is the only evidence available, so it is used.
    """
    stored = (blob or {}).get("config", {}) or {}
    st = stored.get("train", {}) or {}
    cur = asdict(cfg)
    bad = [k for k in _RESUME_CRITICAL_TRAIN_KEYS
           if k in st and k in cur and st[k] != cur[k]]
    if stored.get("policy_mode", policy_mode) != policy_mode:
        bad.append("policy_mode")
    stored_init = stored.get("init_magnitudes", None)
    if stored_init is not None:
        want = {k: float(v) for k, v in zip(MAG_NAMES, init_m)}
        if any(abs(float(stored_init.get(k, np.nan)) - want[k]) > 1e-9 for k in want):
            bad.append("init_magnitudes")
    sb, cb = stored.get("budget", budget), budget
    if (sb is None) != (cb is None) or (sb is not None and cb is not None
                                        and abs(float(sb) - float(cb)) > 1e-9):
        bad.append("budget")
    if bad:
        raise RuntimeError(
            f"cached cell {cell_name!r} was produced with a DIFFERENT configuration "
            f"({', '.join(sorted(set(bad)))}) and would silently unmatch the "
            f"comparison.  Delete it, point --outdir elsewhere, or pass --no-resume."
        )


def _state_hash(state: dict) -> str:
    """Order-independent-in-content hash of a state_dict, for the frozen-encoder
    contract: after a FROZEN evaluation the weights must be bit-identical."""
    import hashlib
    h = hashlib.sha256()
    for k in sorted(state):
        v = state[k]
        h.update(k.encode("utf-8"))
        t = v.detach().cpu().contiguous() if isinstance(v, torch.Tensor) else torch.as_tensor(v)
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()[:16]


@torch.no_grad()
def evaluate_objective(encoder: nn.Module, projector: nn.Module,
                       augment: "TensorAugment", domain: Domain, cfg: TrainCfg,
                       m: torch.Tensor, device: torch.device,
                       n_batches: int = 16, eval_seed: int = 12345
                       ) -> Tuple[float, float]:
    """Deterministic end-of-training value of (total loss, agreement term).

    Why not the last training batch.  The summary used to report the loss of ONE
    training minibatch; P6.3 ("the SSL loss cannot tell the healthy domain from
    the amputated one") then ranked arms by a quantity whose seed-to-seed noise
    is comparable to the effect.  Here the objective is averaged over
    ``n_batches`` batches drawn from a generator seeded IDENTICALLY in every
    arm, with the encoder in ``eval()`` (BatchNorm running statistics, no
    batch-composition dependence).  Two arms of the same domain therefore see
    the same images; only their policies differ, which is precisely the
    comparison P6.3 makes.

    Runs inside an RNG island: it must not shift the training stream, otherwise
    "same seed" would depend on whether the evaluation happened.
    """
    was_enc, was_proj = encoder.training, projector.training
    encoder.eval()
    projector.eval()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(eval_seed))
    n_pool = int(domain.pool.shape[0])
    totals_, agrees_ = [], []
    t_state = torch.get_rng_state()
    c_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(int(eval_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(eval_seed))
        for _ in range(int(n_batches)):
            idx = torch.randint(0, n_pool, (cfg.bs,), generator=gen)
            x01 = domain.pool[idx].to(device).float() / 255.0
            ms = m.view(1, -1).expand(cfg.bs, -1)
            z1 = projector(encoder(augment(x01, ms))).float()
            z2 = projector(encoder(augment(x01, ms))).float()
            t, a, _ = per_chunk_losses(z1, z2, cfg.chunks, cfg)
            totals_.append(float(t.mean()))
            agrees_.append(float(a.mean()))
    finally:
        torch.set_rng_state(t_state)
        if c_state is not None:
            torch.cuda.set_rng_state_all(c_state)
        if was_enc:
            encoder.train()
        if was_proj:
            projector.train()
    return float(np.mean(totals_)), float(np.mean(agrees_))


def train_ssl(cell_name: str, outdir: str, domain: Domain, cfg: TrainCfg,
              seed: int, policy_mode: str, init_m: np.ndarray,
              budget: Optional[float], device: torch.device,
              resume: bool = True, learn_mask: Optional[Sequence[bool]] = None,
              quiet: bool = False, save_encoder: bool = False) -> dict:
    """One SSL pre-training cell.  ``policy_mode`` in {fixed, minimize, maximize}.

    Returns the cell summary (also written to ``<cell>.summary.json``).
    A cell whose summary already exists is NOT re-run -- that is the Colab
    resume contract at the experiment level, on top of the step-level
    checkpoint -- but its stored configuration is checked first, so a cached
    cell from another --steps/--bs/--arch cannot silently join a comparison.

    ``save_encoder`` keeps the final weights under ``<cell>.encoder.pt``.  Only
    D2 needs them (the fine-tuning arm); D1 does not, and 39 x 45 MB of dead
    weights on a Colab disk is not free.
    """
    summary_path = os.path.join(outdir, f"{cell_name}.summary.json")
    if resume and os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            done = json.load(f)
        _check_cached_cell(done, cell_name, cfg, policy_mode, init_m, budget)
        if save_encoder and not os.path.exists(
                os.path.join(outdir, f"{cell_name}.encoder.pt")):
            raise RuntimeError(
                f"cached cell {cell_name!r} has a summary but no encoder.pt, and "
                f"this experiment needs the weights (fine-tuning arm).  Delete "
                f"{summary_path} so the cell is re-trained."
            )
        if not quiet:
            print(f"[skip] {cell_name} (summary present)")
        return done["summary"]

    if policy_mode not in ("fixed", "minimize", "maximize"):
        raise ValueError(f"policy_mode={policy_mode!r}")

    set_seed(seed)
    config = {
        "cell": cell_name, "domain": domain.name, "seed": seed,
        "policy_mode": policy_mode,
        "init_magnitudes": {k: float(v) for k, v in zip(MAG_NAMES, init_m)},
        "budget": budget,
        "augcfg_at_init": magnitudes_to_augcfg(init_m, cfg.size).to_dict(),
        "g_task_note": domain.g_task_note,
        "train": asdict(cfg),
    }
    run = Run(cell_name, outdir, config, resume=resume, higher_is_better=False)

    encoder, projector, feat_dim = build_model(cfg, device)
    policy = MagnitudePolicy(init_m, sigma=cfg.policy_sigma,
                             learn_mask=learn_mask).to(device)
    augment = TensorAugment(cfg.size).to(device)

    params = list(encoder.parameters()) + list(projector.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    policy_opt = (torch.optim.Adam(policy.parameters(), lr=cfg.policy_lr)
                  if policy_mode != "fixed" else None)

    pool = domain.pool                                     # uint8 CPU
    n_pool = int(pool.shape[0])
    data_gen = torch.Generator(device="cpu")
    data_gen.manual_seed(1000 * int(seed) + 7)

    use_amp = bool(cfg.amp and device.type == "cuda")
    m_fixed = torch.tensor(np.clip(np.asarray(init_m, dtype=np.float32), 0.0, 1.0),
                           dtype=torch.float32, device=device)
    # The REALISED magnitude vector: the box [0,1]^5 can make a requested
    # ||m|| = c unreachable for a concentrated direction, and a "constant
    # budget" sweep that is not constant would be a silent confound.
    initial_norm = float(np.linalg.norm(np.clip(init_m, 0, 1)))

    def _current_magnitudes() -> np.ndarray:
        """What is ACTUALLY applied.  For a fixed arm this is ``init_m``, not
        ``sigmoid(logit(clip(init_m)))``: the policy object exists in fixed arms
        only to keep one code path, and reporting its (floor-clipped) mean would
        make the identity arm log ||m|| = 2.2e-3 instead of 0."""
        if policy_mode == "fixed":
            return m_fixed.detach().cpu().numpy().astype(np.float64)
        return policy.mean_magnitudes(budget).detach().cpu().numpy().astype(np.float64)

    start_step = 0
    first_agree: Optional[float] = None
    first_loss: Optional[float] = None
    ck = run.load_ckpt() if resume else None
    if ck is not None:
        encoder.load_state_dict(ck["encoder"])
        projector.load_state_dict(ck["projector"])
        policy.load_state_dict(ck["policy"])
        opt.load_state_dict(ck["opt"])
        if policy_opt is not None and ck.get("policy_opt") is not None:
            policy_opt.load_state_dict(ck["policy_opt"])
        data_gen.set_state(ck["data_gen"])
        _set_rng_state(ck["rng"])
        start_step = int(ck["step"])
        # first_agree MUST survive the resume.  It is the denominator of P5.1's
        # "the agreement term decayed to a fraction of its initial value"; taken
        # after a restart it would be the ALREADY-COLLAPSED value and the ratio
        # would come out at ~1, i.e. P5.1 would look refuted purely because the
        # Colab session was killed.  Silent, seed-stable, and wrong.
        fa = ck.get("first_agree", None)
        first_agree = None if fa is None else float(fa)
        fl = ck.get("first_loss", None)
        first_loss = None if fl is None else float(fl)
        if not quiet:
            print(f"[resume] {cell_name} from step {start_step}")

    t_start = time.time()

    encoder.train()
    projector.train()

    if first_agree is None:
        if start_step > 0:
            # Resumed from a checkpoint written before this field existed.
            # Measuring it now would time-stamp it AFTER the collapse, so it is
            # reported as missing instead of as a number that is quietly wrong.
            print(f"[warn] {cell_name}: resumed checkpoint carries no first_agree; "
                  f"the P5.1 agreement ratio will be NaN for this cell.")
            first_agree = float("nan")
            first_loss = float("nan")
        else:
            # Measured with the same deterministic estimator as the final value,
            # so the ratio final/first compares like with like.  Every arm pays
            # it, including the fixed ones, so no arm gains a budget advantage.
            first_loss, first_agree = evaluate_objective(
                encoder, projector, augment, domain, cfg,
                torch.tensor(_current_magnitudes(), dtype=torch.float32, device=device),
                device, n_batches=cfg.eval_loss_batches, eval_seed=777)

    for step in range(start_step, cfg.steps):
        lr_now = cosine_lr(step, cfg.steps, cfg.lr, cfg.warmup_frac)
        for g in opt.param_groups:
            g["lr"] = lr_now

        idx = torch.randint(0, n_pool, (cfg.bs,), generator=data_gen)
        x01 = (pool[idx].to(device, non_blocking=True).float() / 255.0)

        if policy_mode == "fixed":
            m_chunk = m_fixed.view(1, -1).expand(cfg.chunks, -1)
            logp = None
        else:
            m_chunk, logp = policy.sample(cfg.chunks, budget)

        per = cfg.bs // cfg.chunks
        m_sample = m_chunk.repeat_interleave(per, dim=0)         # [bs,5]
        v1 = augment(x01, m_sample)
        v2 = augment(x01, m_sample)

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            z1 = projector(encoder(v1))
            z2 = projector(encoder(v2))
        z1 = z1.float()
        z2 = z2.float()

        totals, agrees, diag = per_chunk_losses(z1, z2, cfg.chunks, cfg)
        loss = totals.mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # -- REINFORCE update of the policy ---------------------------------
        if policy_mode != "fixed":
            reward = (totals if cfg.policy_objective == "full" else agrees).detach()
            M = reward.shape[0]
            if M < 2:
                raise ValueError("REINFORCE with a leave-one-out baseline needs chunks >= 2")
            baseline = (reward.sum() - reward) / (M - 1)         # leave-one-out, unbiased
            advantage = reward - baseline
            if cfg.advantage_normalise:
                # A positive rescaling: it changes the step size, never the sign.
                advantage = advantage / (reward.std().clamp_min(1e-8))
            sign = 1.0 if policy_mode == "minimize" else -1.0
            policy_loss = sign * (advantage * logp).mean()
            policy_opt.zero_grad(set_to_none=True)
            policy_loss.backward()
            policy_opt.step()
            policy.clamp_()
            policy.project_(budget)

        # -- logging ---------------------------------------------------------
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            m_now = _current_magnitudes()
            row = {
                "seed": seed,
                "loss": float(loss.detach()),
                "agree": float(agrees.mean()),
                "var": diag["var"],
                "cov": diag["cov"],
                "erank": effective_rank(z1.detach()),
                "lr": lr_now,
                "mag_norm": float(np.linalg.norm(m_now)),
            }
            for k, name in enumerate(MAG_NAMES):
                row[f"mag_{name}"] = float(m_now[k])
            run.log(step, **row)

        if cfg.eval_every > 0 and step > 0 and step % cfg.eval_every == 0:
            acc = _quick_knn(encoder, domain, device, cfg)
            run.log(step, seed=seed, knn=acc)
            encoder.train()

        if step % cfg.ckpt_every == 0 or step == cfg.steps - 1:
            # No ``best_metric``: the harness would then also write a
            # <cell>.best.pt, doubling the disk cost for a "best" defined by the
            # lowest single-batch SSL loss -- a quantity nobody here uses and
            # that is not even monotone in representation quality.  The LAST
            # weights are the object of study.
            run.save_ckpt(step + 1, encoder=encoder.state_dict(),
                          projector=projector.state_dict(),
                          policy=policy.state_dict(), opt=opt.state_dict(),
                          policy_opt=(policy_opt.state_dict() if policy_opt else None),
                          data_gen=data_gen.get_state(), rng=_rng_state(),
                          first_agree=first_agree, first_loss=first_loss)

    # -- final evaluation ---------------------------------------------------
    m_final = _current_magnitudes()
    m_final_t = torch.tensor(m_final, dtype=torch.float32, device=device)
    final_loss, final_agree = evaluate_objective(
        encoder, projector, augment, domain, cfg, m_final_t, device,
        n_batches=cfg.eval_loss_batches, eval_seed=777)

    encoder.eval()
    # FROZEN-ENCODER CONTRACT, made operational rather than asserted: the probe
    # below must not touch a single weight or a single BatchNorm running
    # statistic.  The hash is over the whole state_dict, buffers included.
    hash_before = _state_hash(encoder.state_dict())
    train_ds, test_ds = domain.eval_datasets()
    probe = linear_probe(encoder, train_ds, test_ds, epochs=cfg.probe_epochs,
                         lr=cfg.probe_lr, device=device,
                         num_classes=domain.num_classes)
    hash_after = _state_hash(encoder.state_dict())
    if hash_before != hash_after:
        raise RuntimeError(
            f"{cell_name}: the encoder changed during the FROZEN linear probe "
            f"({hash_before} -> {hash_after}).  Every 'frozen' number in D2 "
            f"would be meaningless; refusing to record it."
        )

    # Fixed seeds so that the invariance numbers are comparable across arms:
    # left to the ambient RNG they would inherit whatever state training ended
    # in, i.e. a different measurement per arm.
    inv_strong = tensor_invariance_score(encoder, domain, REFERENCE_POLICIES["strong"],
                                         augment, device, eval_seed=4242)
    inv_own = tensor_invariance_score(encoder, domain, m_final, augment, device,
                                      eval_seed=4242)

    summary = {
        "cell": cell_name,
        "domain": domain.name,
        "seed": int(seed),
        "policy_mode": policy_mode,
        "budget": budget,
        "steps": int(cfg.steps),
        "bs": int(cfg.bs),
        "forward_passes": int(cfg.forward_passes()),
        "probe_acc": float(probe["acc"]),
        "probe_worst_class": float(np.nanmin(probe["per_class_acc"])),
        "final_loss": float(final_loss),
        "final_agree": float(final_agree),
        "first_agree": float(first_agree) if first_agree is not None else float("nan"),
        "first_loss": float(first_loss) if first_loss is not None else float("nan"),
        # None (not False) when the baseline is unavailable after a resume from an
        # older checkpoint: a missing measurement must not read as "did not decrease".
        "loss_decreased": (bool(final_loss < first_loss)
                           if first_loss is not None and first_loss == first_loss else None),
        "initial_mag_norm": initial_norm,
        "realised_mag_norm": initial_norm,
        "final_mag_norm": float(np.linalg.norm(m_final)),
        "final_magnitudes": {k: float(v) for k, v in zip(MAG_NAMES, m_final)},
        "invariance_under_strong": inv_strong,
        "invariance_under_own_policy": inv_own,
        "encoder_sha": hash_after,
        "wall_s": round(time.time() - t_start, 1),
    }
    if save_encoder:
        # Needed by the D2 fine-tuning arm.  Written atomically: a half-written
        # file that a later run happily loads would produce a "fine-tuned from
        # pre-training" number that came from garbage weights.
        enc_path = os.path.join(outdir, f"{cell_name}.encoder.pt")
        tmp = enc_path + ".tmp"
        with open(tmp, "wb") as f:
            torch.save({"encoder": encoder.state_dict(), "sha": hash_after}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, enc_path)
    run.finish(summary)
    # The summary is the resume token; once it exists the 180 MB step checkpoint
    # is dead weight on a Colab disk.
    for stale in (run.ckpt_path, run.best_path):
        if os.path.exists(stale):
            try:
                os.remove(stale)
            except OSError:                                       # pragma: no cover
                pass
    if not quiet:
        print(f"[done] {cell_name}: probe={summary['probe_acc']:.4f} "
              f"|m|={summary['final_mag_norm']:.3f} "
              f"agree={summary['final_agree']:.5f} ({summary['wall_s']:.0f}s)")
    return summary


@torch.no_grad()
def _quick_knn(encoder: nn.Module, domain: Domain, device: torch.device,
               cfg: TrainCfg) -> float:
    """Cheap mid-run diagnostic.  Subsampled so that evaluating often does not
    dominate the budget; the harness probe already runs in an RNG island, so it
    cannot perturb the training stream."""
    n = min(cfg.knn_subsample, int(domain.train_images.shape[0]))
    tr = ImageTensorDataset(domain.train_images[:n], domain.train_labels[:n], True)
    n_te = min(cfg.knn_subsample // 2, int(domain.test_images.shape[0]))
    te = ImageTensorDataset(domain.test_images[:n_te], domain.test_labels[:n_te], True)
    return knn_probe(encoder, tr, te, k=20, device=device)


@torch.no_grad()
def tensor_invariance_score(encoder: nn.Module, domain: Domain, m: Sequence[float],
                            augment: TensorAugment, device: torch.device,
                            n_views: int = 8, n_samples: int = 256,
                            eval_seed: Optional[int] = 777) -> float:
    """Worst-case augmentation invariance, in the harness' definition.

    For each image, ``n_views`` views are drawn from the policy ``m``, embedded
    and L2-normalised; the image's score is the MINIMUM pairwise cosine
    similarity, and the returned value averages those minima.  1.0 = perfectly
    invariant.

    We deliberately re-implement it on tensors instead of calling
    ``harness.invariance_score``: that function drives the PIL pipeline, while
    the encoders here were trained with TensorAugment.  Measuring invariance
    with a different implementation of the same nominal policy would confound
    the number with implementation differences.
    """
    # RNG ISLAND (see harness note 6).  This function draws n_views random
    # augmentations, so calling it consumes the global RNG stream.  Evaluating
    # more often would then change the TRAINING trajectory at an identical seed
    # and silently poison every matched comparison in this file.  We therefore
    # branch onto a private seed and restore the global state on the way out.
    _rng_state = None
    if eval_seed is not None:
        _rng_state = (torch.get_rng_state(), np.random.get_state(), random.getstate())
        torch.manual_seed(eval_seed)
        np.random.seed(eval_seed)
        random.seed(eval_seed)
    was_training = encoder.training
    encoder.eval()
    mt = torch.tensor(np.asarray(m, dtype=np.float32), device=device)
    n = min(n_samples, int(domain.test_images.shape[0]))
    x = domain.test_images[:n].to(device).float() / 255.0
    mins: List[float] = []
    chunk = 32
    for i in range(0, n, chunk):
        xb = x[i:i + chunk]
        b = xb.shape[0]
        rep = xb.repeat_interleave(n_views, dim=0)
        views = augment(rep, mt.view(1, -1).expand(rep.shape[0], -1))
        f = encoder(views).float()
        f = f / f.norm(dim=1, keepdim=True).clamp_min(1e-8)
        f = f.view(b, n_views, -1)
        sim = torch.bmm(f, f.transpose(1, 2))
        eye = torch.eye(n_views, device=device, dtype=torch.bool).unsqueeze(0)
        sim = sim.masked_fill(eye, float("inf"))
        mins.extend(sim.view(b, -1).min(dim=1).values.cpu().tolist())
    if was_training:
        encoder.train()
    if _rng_state is not None:
        torch.set_rng_state(_rng_state[0])
        np.random.set_state(_rng_state[1])
        random.setstate(_rng_state[2])
    return float(np.mean(mins)) if mins else float("nan")


# ===========================================================================
# 6.  Supervised fine-tuning (the irrecoverability test of T6)
# ===========================================================================

def finetune(cell_name: str, outdir: str, domain: Domain, cfg: TrainCfg, seed: int,
             encoder_state: Optional[dict], device: torch.device,
             resume: bool = True, quiet: bool = False) -> dict:
    """Fine-tune the whole encoder plus a linear head on the domain's own task.

    ``encoder_state=None`` gives the from-scratch supervised baseline at the
    same budget, which is the reference that makes "fine-tuning recovers" mean
    anything.

    The fine-tuning augmentation is the CONSERVATIVE policy in every arm
    (crop only, no colour, no flip): using each arm's own pre-training policy
    would re-introduce the label destruction at fine-tuning time and confound
    the measurement.  Identical recipe in every arm, by construction.

    A fine-tuning cell costs one to two minutes, so it is resumable at the CELL
    level only (skipped once its summary exists); adding step-level resume here
    would buy less than it costs in reader attention.
    """
    summary_path = os.path.join(outdir, f"{cell_name}.summary.json")
    if resume and os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            done = json.load(f)
        if not quiet:
            print(f"[skip] {cell_name} (summary present)")
        return done["summary"]

    set_seed(seed + 31337)
    run = Run(cell_name, outdir, {"cell": cell_name, "domain": domain.name,
                                  "seed": seed, "kind": "finetune",
                                  "from_pretrained": encoder_state is not None,
                                  "epochs": cfg.finetune_epochs,
                                  "lr": cfg.finetune_lr}, resume=False)

    encoder, _, feat_dim = build_model(cfg, device)
    if encoder_state is not None:
        encoder.load_state_dict(encoder_state)
    head = nn.Linear(feat_dim, domain.num_classes).to(device)
    augment = TensorAugment(cfg.size).to(device)
    m_ft = torch.tensor(REFERENCE_POLICIES["conservative"], dtype=torch.float32,
                        device=device)

    opt = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()),
                            lr=cfg.finetune_lr, weight_decay=1e-4)
    x_tr = domain.train_images
    y_tr = domain.train_labels
    n = int(x_tr.shape[0])
    steps_per_epoch = max(1, n // cfg.bs)
    total = cfg.finetune_epochs * steps_per_epoch
    use_amp = bool(cfg.amp and device.type == "cuda")
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 4242)

    step = 0
    encoder.train()
    for epoch in range(cfg.finetune_epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(steps_per_epoch):
            idx = perm[i * cfg.bs:(i + 1) * cfg.bs]
            xb = x_tr[idx].to(device).float() / 255.0
            yb = y_tr[idx].to(device)
            xb = augment(xb, m_ft.view(1, -1).expand(xb.shape[0], -1))
            for grp in opt.param_groups:
                grp["lr"] = cosine_lr(step, total, cfg.finetune_lr, 0.05)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = head(encoder(xb))
            loss = F.cross_entropy(logits.float(), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
        run.log(epoch, seed=seed, train_loss=float(loss.detach()))

    acc = _evaluate_supervised(encoder, head, domain, device, cfg)
    summary = {"cell": cell_name, "domain": domain.name, "seed": int(seed),
               "kind": "finetune", "from_pretrained": encoder_state is not None,
               "finetune_acc": float(acc)}
    run.finish(summary)
    if not quiet:
        print(f"[done] {cell_name}: finetune_acc={acc:.4f}")
    return summary


@torch.no_grad()
def _evaluate_supervised(encoder: nn.Module, head: nn.Module, domain: Domain,
                         device: torch.device, cfg: TrainCfg) -> float:
    encoder.eval()
    head.eval()
    augment = TensorAugment(cfg.size).to(device)
    x = domain.test_images
    y = domain.test_labels
    correct = 0
    for i in range(0, int(x.shape[0]), 512):
        xb = x[i:i + 512].to(device).float() / 255.0
        xb = augment.normalise_only(xb)
        pred = head(encoder(xb)).argmax(1).cpu()
        correct += int((pred == y[i:i + 512]).sum())
    encoder.train()
    head.train()
    return correct / float(x.shape[0])


def pixel_probe(domain: Domain, device: torch.device, cfg: TrainCfg) -> float:
    """Linear probe on raw normalised pixels.

    This is the CEILING that makes D2 interpretable: it proves the label is
    linearly decodable from the input itself, so any collapse observed on an
    encoder is the encoder's doing, not a badly built domain.
    """
    flat = nn.Flatten().to(device)
    tr, te = domain.eval_datasets()
    res = linear_probe(flat, tr, te, epochs=max(10, cfg.probe_epochs // 4),
                       lr=cfg.probe_lr, device=device,
                       num_classes=domain.num_classes)
    return float(res["acc"])


def random_encoder_probe(domain: Domain, cfg: TrainCfg, seed: int,
                         device: torch.device) -> float:
    """Linear probe on an UNTRAINED encoder -- the floor for D1.

    Without it, "the probe collapses" is unfalsifiable: a randomly initialised
    ResNet-18 is already far above chance on CIFAR-10.
    """
    set_seed(seed + 99)
    encoder, _, _ = build_model(cfg, device)
    encoder.eval()
    tr, te = domain.eval_datasets()
    res = linear_probe(encoder, tr, te, epochs=cfg.probe_epochs, lr=cfg.probe_lr,
                       device=device, num_classes=domain.num_classes)
    return float(res["acc"])


# ===========================================================================
# 7.  D1 -- the learned policy degenerates (T5)
# ===========================================================================

D1_DIRECTIONS = ("learned", "uniform", "crop_only", "color_only", "random")


def d1_grids(args) -> Tuple[List[float], List[str], float]:
    if args.smoke:
        return [0.0, 0.8, 1.6], ["learned", "uniform", "crop_only"], 0.8
    c_grid = [float(c) for c in args.c_grid]
    dirs = list(args.directions)
    c0 = float(args.c0)
    if c0 not in c_grid:
        raise ValueError(f"--c0 {c0} must be one of --c-grid {c_grid} "
                         f"(the crossed plan reuses that shared cell)")
    return c_grid, dirs, c0


def run_d1(args, cfg: TrainCfg, device: torch.device) -> dict:
    outdir = os.path.join(args.outdir, "D1")
    os.makedirs(outdir, exist_ok=True)
    c_grid, dir_names, c0 = d1_grids(args)
    seeds = list(args.seeds)

    domain = build_domain("cifar10", cfg.size, args.data_root, seed=0,
                          synthetic=args.synthetic, n_train=args.n_synth_train,
                          n_test=args.n_synth_test)
    learn_mask = [name in args.d1_learn for name in MAG_NAMES]
    if not any(learn_mask):
        raise ValueError("--d1-learn selected no magnitude")

    # ---- budget check over EVERY D1 cell, before spending anything --------
    cell_names: List[str] = ["learn-minimize", "learn-maximize", "fixed-strong",
                             "fixed-identity"]
    cell_names += [f"cross_c{c:.2f}_dir-learned" for c in c_grid]
    cell_names += [f"cross_c{c0:.2f}_dir-{d}" for d in dir_names if d != "learned"]
    matched_budget_check([{"name": n, "steps": cfg.steps, "batch_size": cfg.bs,
                           "forward_passes": cfg.forward_passes()} for n in cell_names])

    rows: List[dict] = []

    def _cell(name: str, mode: str, init_m: np.ndarray, budget: Optional[float],
              seed: int, extra: dict) -> dict:
        if args.arm and args.arm not in name:
            return {}
        s = train_ssl(f"D1_{name}_seed{seed}", outdir, domain, cfg, seed, mode,
                      init_m, budget, device, resume=args.resume,
                      learn_mask=learn_mask)
        if s:
            rows.append({**s, "arm": name, **extra})
        return s

    # ---- phase 1: learn the policy ---------------------------------------
    # (a) minimise the SSL loss: the degenerate direction of T5.
    # (b) maximise it under a budget: the only non-trivial way to learn T.
    init = REFERENCE_POLICIES["strong"].copy()
    learned_by_seed: Dict[int, np.ndarray] = {}
    for seed in seeds:
        _cell("learn-minimize", "minimize", init, None, seed,
              {"kind": "phase1", "c": float("nan"), "direction": "-"})
        s = _cell("learn-maximize", "maximize", init, c0, seed,
                  {"kind": "phase1", "c": c0, "direction": "learned"})
        if s:
            learned_by_seed[seed] = np.array([s["final_magnitudes"][k] for k in MAG_NAMES])
        else:
            # --arm filtered phase 1 out; recover the learned vector from disk
            p = os.path.join(outdir, f"D1_learn-maximize_seed{seed}.summary.json")
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    sm = json.load(f)["summary"]
                learned_by_seed[seed] = np.array([sm["final_magnitudes"][k]
                                                  for k in MAG_NAMES])

    # ---- references -------------------------------------------------------
    for seed in seeds:
        _cell("fixed-strong", "fixed", REFERENCE_POLICIES["strong"], None, seed,
              {"kind": "reference", "c": float(np.linalg.norm(REFERENCE_POLICIES["strong"])),
               "direction": "strong"})
        _cell("fixed-identity", "fixed", REFERENCE_POLICIES["identity"], None, seed,
              {"kind": "reference", "c": 0.0, "direction": "identity"})

    floor = {}
    for seed in seeds:
        floor[seed] = random_encoder_probe(domain, cfg, seed, device)
    print(f"[D1] untrained-encoder probe floor: "
          f"{np.mean(list(floor.values())):.4f} +/- {np.std(list(floor.values())):.4f}")

    # ---- phase 2: the crossed plan ---------------------------------------
    # Sweep A: learned DIRECTION frozen, budget c varied.
    # Sweep B: budget c0 frozen, DIRECTION varied.
    for seed in seeds:
        if seed not in learned_by_seed:
            print(f"[D1] no learned policy for seed {seed}; crossed plan skipped")
            continue
        d_learned = learned_by_seed[seed] / max(1e-9, float(np.linalg.norm(learned_by_seed[seed])))
        for c in c_grid:
            m = np.clip(c * d_learned, 0.0, 1.0)
            _cell(f"cross_c{c:.2f}_dir-learned", "fixed", m, None, seed,
                  {"kind": "sweep_c", "c": c, "direction": "learned",
                   "realised_norm": float(np.linalg.norm(m))})
        for dname in dir_names:
            if dname == "learned":
                continue                     # shared cell, already run at c0
            d = direction_vector(dname, seed, d_learned)
            m = np.clip(c0 * d, 0.0, 1.0)
            _cell(f"cross_c{c0:.2f}_dir-{dname}", "fixed", m, None, seed,
                  {"kind": "sweep_dir", "c": c0, "direction": dname,
                   "realised_norm": float(np.linalg.norm(m))})

    df = _collect_summaries(outdir, "D1_")
    df.to_csv(os.path.join(args.outdir, "D1_cells.csv"), index=False)
    verdicts = d1_verdicts(df, c_grid, c0, dir_names, floor)
    d1_figures(outdir, args.outdir, df, c_grid, c0, dir_names, seeds)
    return verdicts


def d1_verdicts(df: pd.DataFrame, c_grid: List[float], c0: float,
                dir_names: List[str], floor: Dict[int, float]) -> dict:
    out: dict = {"floor_untrained_encoder_probe": float(np.mean(list(floor.values())))
                 if floor else float("nan")}

    # ---- P5.1: the minimising policy degenerates --------------------------
    mn = df[df["arm"] == "learn-minimize"]
    if len(mn) == 0:
        out["P5.1"] = {"verdict": VERDICT_INCONCLUSIVE, "why": "arm not run"}
    else:
        keep_frac = float((mn["final_mag_norm"] / mn["initial_mag_norm"]).mean())
        agree_frac = float((mn["final_agree"] / mn["first_agree"].replace(0, np.nan)).mean())
        strong = df[df["arm"] == "fixed-strong"]
        probe_min = float(mn["probe_acc"].mean())
        probe_strong = float(strong["probe_acc"].mean()) if len(strong) else float("nan")
        fl = out["floor_untrained_encoder_probe"]
        confirmed = (keep_frac <= CONFIRM_MIN_COLLAPSE_FRAC
                     and agree_frac <= CONFIRM_AGREE_COLLAPSE_FRAC)
        refuted = keep_frac >= FALSIFY_MIN_KEEP_FRAC
        out["P5.1"] = {
            "claim": "a policy trained to minimise the SSL loss degenerates to T = identity",
            "verdict": verdict(confirmed, refuted),
            "final_over_initial_mag_norm": keep_frac,
            "final_over_initial_agreement": agree_frac,
            "probe_minimising_policy": probe_min,
            "probe_strong_policy": probe_strong,
            "probe_untrained_floor": fl,
            "probe_distance_to_floor": probe_min - fl,
            "thresholds": {"confirm_mag_frac<=": CONFIRM_MIN_COLLAPSE_FRAC,
                           "confirm_agree_frac<=": CONFIRM_AGREE_COLLAPSE_FRAC,
                           "refute_mag_frac>=": FALSIFY_MIN_KEEP_FRAC},
        }

    # ---- P5.2: the budget dominates the policy ----------------------------
    sweep_c = df[df["kind"] == "sweep_c"].groupby("c")["probe_acc"].agg(["mean", "std"])
    sweep_d = df[df["kind"].isin(["sweep_dir"])].groupby("direction")["probe_acc"].agg(["mean", "std"])
    # the shared cell (c0, learned) belongs to both sweeps
    shared = df[(df["kind"] == "sweep_c") & (np.isclose(df["c"], c0))]
    if len(shared):
        sweep_d.loc["learned"] = [float(shared["probe_acc"].mean()),
                                  float(shared["probe_acc"].std())]

    if len(sweep_c) < 2 or len(sweep_d) < 2:
        out["P5.2"] = {"verdict": VERDICT_INCONCLUSIVE, "why": "crossed plan incomplete"}
        return out

    spread_c = float(sweep_c["mean"].max() - sweep_c["mean"].min())
    spread_d = float(sweep_d["mean"].max() - sweep_d["mean"].min())
    ratio = spread_d / spread_c if spread_c > 1e-9 else float("inf")
    ranked = sweep_d["mean"].sort_values(ascending=False)
    rank_learned = int(list(ranked.index).index("learned")) + 1 if "learned" in ranked.index else -1

    confirmed = (ratio <= CONFIRM_DIR_OVER_C_RATIO)
    refuted = (ratio >= FALSIFY_DIR_OVER_C_RATIO and rank_learned == 1)
    out["P5.2"] = {
        "claim": ("with a learned T the decisive quantity is the strength budget c, "
                  "not the learned direction"),
        "verdict": verdict(confirmed, refuted),
        "spread_over_budget_sweep": spread_c,
        "spread_over_direction_sweep": spread_d,
        "ratio_direction_over_budget": ratio,
        "rank_of_learned_direction": rank_learned,
        "n_directions": int(len(sweep_d)),
        "probe_by_c": {f"{k:.2f}": float(v) for k, v in sweep_c["mean"].items()},
        "probe_by_direction": {str(k): float(v) for k, v in sweep_d["mean"].items()},
        "thresholds": {"confirm_ratio<=": CONFIRM_DIR_OVER_C_RATIO,
                       "refute_ratio>=_and_learned_is_best": FALSIFY_DIR_OVER_C_RATIO},
        "caveat": ("a confirmed verdict here is evidence for this objective, this "
                   "parameterisation and CIFAR-10 at this budget -- not a theorem"),
    }
    return out


# ===========================================================================
# 8.  D2 -- the asymmetry of T (T6)
# ===========================================================================

D2_POLICIES = ("conservative", "strong", "strong_no_color", "strong_no_flip")


def run_d2(args, cfg: TrainCfg, device: torch.device) -> dict:
    outdir = os.path.join(args.outdir, "D2")
    os.makedirs(outdir, exist_ok=True)
    seeds = list(args.seeds)
    policies = [p for p in D2_POLICIES if (not args.arm or args.arm in p)] or list(D2_POLICIES)

    # Budget matching is enforced WITHIN a domain: that is where the comparison
    # lives.  Across domains only the ORDERING of the policies is compared, never
    # the absolute accuracies (the tasks have different numbers of classes).
    matched_budget_check([{"name": p, "steps": cfg.steps, "batch_size": cfg.bs,
                           "forward_passes": cfg.forward_passes()} for p in D2_POLICIES])

    rows: List[dict] = []
    context: Dict[str, dict] = {}

    for dname in D2_DOMAINS:
        domain = build_domain(dname, cfg.size, args.data_root, seed=args.seeds[0],
                              synthetic=args.synthetic, n_train=args.n_synth_train,
                              n_test=args.n_synth_test,
                              all_digits=args.chirality_all_digits)
        ceiling = pixel_probe(domain, device, cfg)
        print(f"[D2] {dname}: raw-pixel linear probe ceiling = {ceiling:.4f} "
              f"(chance = {1.0 / domain.num_classes:.3f})")
        scratch_accs = []
        for seed in seeds:
            s = finetune(f"D2_{dname}_scratch_ft_seed{seed}", outdir, domain, cfg,
                         seed, None, device, resume=args.resume)
            scratch_accs.append(s["finetune_acc"])
        context[dname] = {
            "pixel_probe_ceiling": ceiling,
            "chance": 1.0 / domain.num_classes,
            "scratch_finetune_mean": float(np.mean(scratch_accs)),
            "scratch_finetune_std": float(np.std(scratch_accs, ddof=1)) if len(scratch_accs) > 1 else float("nan"),
            "g_task_note": domain.g_task_note,
            "offending_magnitude": domain.offending_magnitude,
        }

        for pol in policies:
            for seed in seeds:
                cell = f"D2_{dname}_{pol}_seed{seed}"
                s = train_ssl(cell, outdir, domain, cfg, seed, "fixed",
                              REFERENCE_POLICIES[pol], None, device,
                              resume=args.resume)
                enc_path = os.path.join(outdir, f"{cell}.encoder.pt")
                enc_state = torch.load(enc_path, map_location="cpu",
                                       weights_only=False)["encoder"] \
                    if os.path.exists(enc_path) else None
                ft = finetune(f"{cell}_ft", outdir, domain, cfg, seed, enc_state,
                              device, resume=args.resume)
                rows.append({**s, "policy": pol, "finetune_acc": ft["finetune_acc"]})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.outdir, "D2_cells.csv"), index=False)
    verdicts = d2_verdicts(df, context)
    d2_figures(args.outdir, df, context)
    return verdicts


def d2_verdicts(df: pd.DataFrame, context: Dict[str, dict]) -> dict:
    out: dict = {"context": context}
    if len(df) == 0:
        return {**out, "P6.1": {"verdict": VERDICT_INCONCLUSIVE, "why": "no cell"}}

    g = df.groupby(["domain", "policy"])
    stats = g.agg(probe_mean=("probe_acc", "mean"), probe_std=("probe_acc", "std"),
                  ft_mean=("finetune_acc", "mean"), ft_std=("finetune_acc", "std"),
                  loss_mean=("final_loss", "mean"),
                  inv_mean=("invariance_under_own_policy", "mean")).reset_index()
    out["per_cell"] = stats.to_dict(orient="records")

    def _get(dom: str, pol: str, col: str) -> float:
        r = stats[(stats["domain"] == dom) & (stats["policy"] == pol)]
        return float(r[col].iloc[0]) if len(r) else float("nan")

    # ---- P6.1: crossing ---------------------------------------------------
    deltas = {}
    for dom in df["domain"].unique():
        d = _get(dom, "strong", "probe_mean") - _get(dom, "conservative", "probe_mean")
        meaningful = gap_is_meaningful(_get(dom, "strong", "probe_mean"),
                                       _get(dom, "strong", "probe_std"),
                                       _get(dom, "conservative", "probe_mean"),
                                       _get(dom, "conservative", "probe_std"))
        deltas[dom] = {"strong_minus_conservative": d, "exceeds_seed_noise": meaningful}
    wins = [k for k, v in deltas.items() if v["strong_minus_conservative"] > CONFIRM_CROSSING_MARGIN]
    loses = [k for k, v in deltas.items() if v["strong_minus_conservative"] < -CONFIRM_CROSSING_MARGIN]
    crossing = bool(wins) and bool(loses)
    out["P6.1"] = {
        "claim": ("the ordering of a conservative and a strong T reverses when "
                  "T leaves G_task, at identical budget"),
        "verdict": verdict(crossing, len(wins) == 0 or len(loses) == 0),
        "delta_by_domain": deltas,
        "domains_where_strong_wins": wins,
        "domains_where_strong_loses": loses,
        "margin": CONFIRM_CROSSING_MARGIN,
    }

    # ---- attribution to the named operation -------------------------------
    attribution = {}
    for dom, control in (("colored_shapes", "strong_no_color"),
                         ("chirality", "strong_no_flip")):
        if dom not in set(df["domain"]):
            continue
        attribution[dom] = {
            "control_arm": control,
            "probe_strong": _get(dom, "strong", "probe_mean"),
            "probe_control": _get(dom, control, "probe_mean"),
            "probe_conservative": _get(dom, "conservative", "probe_mean"),
            "recovered_by_removing_the_offending_op":
                _get(dom, control, "probe_mean") - _get(dom, "strong", "probe_mean"),
        }
    out["attribution_to_the_offending_operation"] = attribution

    # ---- P6.2: irrecoverability under a frozen encoder --------------------
    recov = {}
    for dom in df["domain"].unique():
        dam_frozen = _get(dom, "conservative", "probe_mean") - _get(dom, "strong", "probe_mean")
        dam_ft = _get(dom, "conservative", "ft_mean") - _get(dom, "strong", "ft_mean")
        frac = float("nan") if abs(dam_frozen) < 1e-9 else 1.0 - dam_ft / dam_frozen
        recov[dom] = {"damage_frozen": dam_frozen, "damage_finetuned": dam_ft,
                      "recovery_fraction": frac,
                      "scratch_finetune": context.get(dom, {}).get("scratch_finetune_mean"),
                      "pixel_ceiling": context.get(dom, {}).get("pixel_probe_ceiling")}
    adversarial = [d for d in recov if context.get(d, {}).get("offending_magnitude")]
    conf = all(recov[d]["damage_frozen"] > CONFIRM_CROSSING_MARGIN
               and recov[d]["recovery_fraction"] > CONFIRM_RECOVERY_FRACTION
               for d in adversarial) if adversarial else False
    refu = all(recov[d]["recovery_fraction"] < 0.05 for d in adversarial) if adversarial else False
    out["P6.2"] = {
        "claim": ("what a violating T destroys is not recovered by a linear probe on a "
                  "frozen encoder, but is partly recovered by fine-tuning: amputation, "
                  "not mere degradation"),
        "verdict": verdict(conf, refu),
        "by_domain": recov,
        "threshold_recovery_fraction>": CONFIRM_RECOVERY_FRACTION,
        "honest_limit": ("'irrecoverable' is meant under the frozen-encoder contract. "
                         "The label is still in the pixels (see pixel_ceiling); with an "
                         "unbounded fine-tuning budget nothing is irrecoverable in principle."),
    }

    # ---- P6.3: the failure is silent --------------------------------------
    silence = {}
    for dom in df["domain"].unique():
        d_loss = _get(dom, "strong", "loss_mean") - _get(dom, "conservative", "loss_mean")
        d_probe = _get(dom, "strong", "probe_mean") - _get(dom, "conservative", "probe_mean")
        # An unsupervised model selection would pick the arm with the LOWER loss.
        picked = "strong" if d_loss < 0 else "conservative"
        best = "strong" if d_probe > 0 else "conservative"
        silence[dom] = {"delta_final_loss_strong_minus_cons": d_loss,
                        "delta_probe_strong_minus_cons": d_probe,
                        "loss_would_pick": picked, "probe_says_best": best,
                        "loss_misleads": picked != best}
    adversarial_mis = [d for d in adversarial if silence[d]["loss_misleads"]]
    out["P6.3"] = {
        "claim": "the SSL loss gives no warning that the label was destroyed",
        "verdict": verdict(bool(adversarial) and len(adversarial_mis) == len(adversarial),
                           bool(adversarial) and len(adversarial_mis) == 0),
        "by_domain": silence,
        "note": ("this tests the ranking given by the pre-training loss only; it does "
                 "not claim that no unsupervised criterion whatsoever could detect it"),
    }
    return out


# ===========================================================================
# 9.  Collection and figures
# ===========================================================================

def _collect_summaries(outdir: str, prefix: str) -> pd.DataFrame:
    """Rebuild the cell table from the summary files on disk.

    Going through disk rather than through in-memory results is deliberate: a
    partially resumed run must produce exactly the same table as a run that went
    through in one go.
    """
    rows = []
    for p in sorted(glob.glob(os.path.join(outdir, f"{prefix}*.summary.json"))):
        with open(p, "r", encoding="utf-8") as f:
            blob = json.load(f)
        s = dict(blob.get("summary", {}))
        cfgb = blob.get("config", {})
        if s.get("kind") == "finetune":
            continue
        name = s.get("cell", os.path.basename(p))
        arm = name[len(prefix):].rsplit("_seed", 1)[0] if name.startswith(prefix) else name
        s["arm"] = arm
        for k, v in (s.pop("final_magnitudes", {}) or {}).items():
            s[f"mag_{k}"] = v
        s["c"] = cfgb.get("config", {}).get("budget", float("nan"))
        rows.append(s)
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    # Re-derive the crossed-plan coordinates from the arm name: the name is the
    # single source of truth, so a resumed run cannot disagree with a fresh one.
    def _kind(a: str) -> str:
        if a.startswith("cross_") and a.endswith("dir-learned"):
            return "sweep_c"
        if a.startswith("cross_"):
            return "sweep_dir"
        if a.startswith("learn-"):
            return "phase1"
        return "reference"

    def _c(a: str) -> float:
        if a.startswith("cross_c"):
            return float(a.split("cross_c")[1].split("_")[0])
        return float("nan")

    def _dir(a: str) -> str:
        return a.split("dir-")[1] if "dir-" in a else "-"

    df["kind"] = df["arm"].map(_kind)
    df["c"] = df["arm"].map(_c)
    df["direction"] = df["arm"].map(_dir)
    return df


_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
_MARKERS = ["o", "s", "^", "D", "v", "P"]
_HATCHES = ["", "//", "\\\\", "xx", "..", "++"]


def d1_figures(cell_dir: str, outdir: str, df: pd.DataFrame, c_grid: List[float],
               c0: float, dir_names: List[str], seeds: List[int]) -> None:
    """Two figures, both readable in black and white."""
    # --- (1) magnitude trajectories of the two learned arms ---------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, arm, title in zip(axes, ["learn-minimize", "learn-maximize"],
                              ["(a) policy trained to MINIMISE the SSL loss",
                               "(b) policy trained to MAXIMISE it, under budget c"]):
        try:
            agg = aggregate_seeds(os.path.join(cell_dir, f"D1_{arm}_seed*.csv"), "step")
        except (FileNotFoundError, KeyError):
            ax.set_title(title + "\n(no data)")
            continue
        for i, name in enumerate(MAG_NAMES):
            col = f"mag_{name}_mean"
            if col not in agg:
                continue
            ax.plot(agg["step"], agg[col], linestyle=_LINESTYLES[i % len(_LINESTYLES)],
                    marker=_MARKERS[i % len(_MARKERS)], markevery=max(1, len(agg) // 8),
                    color="black", alpha=0.85, label=name)
        if "mag_norm_mean" in agg:
            ax.plot(agg["step"], agg["mag_norm_mean"], color="black", linewidth=2.5,
                    alpha=0.4, label="||m||")
        ax.set_xlabel("step")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("magnitude")
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("D1 / T5 -- trajectory of the learned augmentation magnitudes "
                 f"(mean over {len(seeds)} seeds)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "D1_policy_trajectory.png"), dpi=150)
    plt.close(fig)

    if len(df) == 0:
        return

    # --- (2) the crossed plan ---------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sc = df[df["kind"] == "sweep_c"].groupby("c")["probe_acc"].agg(["mean", "std"])
    if len(sc):
        axes[0].errorbar(sc.index, sc["mean"], yerr=sc["std"].fillna(0.0),
                         color="black", marker="o", capsize=3)
    axes[0].set_xlabel("strength budget c  (learned direction, frozen)")
    axes[0].set_ylabel("linear probe accuracy")
    axes[0].set_title("(A) freeze the policy, vary the budget", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    sd = df[df["kind"] == "sweep_dir"].groupby("direction")["probe_acc"].agg(["mean", "std"])
    shared = df[(df["kind"] == "sweep_c") & (np.isclose(df["c"], c0))]
    if len(shared):
        sd.loc["learned"] = [float(shared["probe_acc"].mean()),
                             float(shared["probe_acc"].std())]
    if len(sd):
        sd = sd.reindex([d for d in dir_names if d in sd.index])
        xs = np.arange(len(sd))
        axes[1].bar(xs, sd["mean"], yerr=sd["std"].fillna(0.0), color="white",
                    edgecolor="black", capsize=3,
                    hatch=[_HATCHES[i % len(_HATCHES)] for i in range(len(sd))])
        axes[1].set_xticks(xs)
        axes[1].set_xticklabels(sd.index, rotation=20, fontsize=8)
    axes[1].set_xlabel(f"policy direction  (budget frozen at c = {c0:g})")
    axes[1].set_title("(B) freeze the budget, vary the policy", fontsize=10)
    axes[1].grid(True, alpha=0.3, axis="y")

    # Same y-range on both panels: the whole point is the comparison of spreads.
    ymins, ymaxs = zip(*[ax.get_ylim() for ax in axes])
    for ax in axes:
        ax.set_ylim(min(ymins), max(ymaxs))
    fig.suptitle("D1 / T5 -- crossed plan: what actually moves the result, "
                 "the budget or the learned policy?", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "D1_crossed_plan.png"), dpi=150)
    plt.close(fig)


def d2_figures(outdir: str, df: pd.DataFrame, context: Dict[str, dict]) -> None:
    if len(df) == 0:
        return
    domains = [d for d in D2_DOMAINS if d in set(df["domain"])]
    policies = [p for p in D2_POLICIES if p in set(df["policy"])]

    # --- (1) the crossing, literally as crossing lines ---------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    xs = np.arange(len(domains))
    for i, pol in enumerate(policies):
        means, stds = [], []
        for dom in domains:
            sel = df[(df["domain"] == dom) & (df["policy"] == pol)]["probe_acc"]
            means.append(float(sel.mean()) if len(sel) else np.nan)
            stds.append(float(sel.std()) if len(sel) > 1 else 0.0)
        ax.errorbar(xs, means, yerr=stds, color="black",
                    linestyle=_LINESTYLES[i % len(_LINESTYLES)],
                    marker=_MARKERS[i % len(_MARKERS)], capsize=3, label=pol)
    for j, dom in enumerate(domains):
        ch = context.get(dom, {}).get("chance")
        if ch is not None:
            ax.hlines(ch, j - 0.25, j + 0.25, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{d}\n(G_task violated by: "
                        f"{context.get(d, {}).get('offending_magnitude') or 'nothing'})"
                        for d in domains], fontsize=8)
    ax.set_ylabel("linear probe accuracy (encoder FROZEN)")
    ax.set_title("D2 / T6 -- the same T, three domains: the curves cross\n"
                 "(short horizontal bars = chance level)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "D2_crossing.png"), dpi=150)
    plt.close(fig)

    # --- (2) frozen vs fine-tuned vs scratch -------------------------------
    fig, axes = plt.subplots(1, len(domains), figsize=(4.2 * len(domains), 4.2),
                             squeeze=False)
    for k, dom in enumerate(domains):
        ax = axes[0][k]
        xs = np.arange(len(policies))
        w = 0.38
        froz, ft = [], []
        for pol in policies:
            sel = df[(df["domain"] == dom) & (df["policy"] == pol)]
            froz.append(float(sel["probe_acc"].mean()) if len(sel) else np.nan)
            ft.append(float(sel["finetune_acc"].mean()) if len(sel) else np.nan)
        ax.bar(xs - w / 2, froz, w, color="white", edgecolor="black",
               hatch="", label="frozen + linear probe")
        ax.bar(xs + w / 2, ft, w, color="white", edgecolor="black",
               hatch="//", label="fine-tuned")
        c = context.get(dom, {})
        if c.get("scratch_finetune_mean") is not None:
            ax.axhline(c["scratch_finetune_mean"], color="black", linestyle="--",
                       linewidth=1.0, label="supervised from scratch")
        if c.get("pixel_probe_ceiling") is not None:
            ax.axhline(c["pixel_probe_ceiling"], color="black", linestyle=":",
                       linewidth=1.0, label="raw-pixel probe (ceiling)")
        ax.set_xticks(xs)
        ax.set_xticklabels(policies, rotation=25, fontsize=7)
        ax.set_title(dom, fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        if k == 0:
            ax.set_ylabel("accuracy")
            ax.legend(fontsize=7)
    fig.suptitle("D2 / T6 -- what a frozen encoder lost, and how much fine-tuning "
                 "gives back", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "D2_recovery.png"), dpi=150)
    plt.close(fig)


# ===========================================================================
# 10.  Self-check of the augmenter (cheap, always run before training)
# ===========================================================================

def selfcheck_augment(device: torch.device, size: int = 32) -> None:
    """Three properties the whole file rests on.  Failing any of them makes
    every downstream number meaningless, so this runs unconditionally."""
    aug = TensorAugment(size).to(device)
    torch.manual_seed(0)
    x = torch.rand(8, 3, size, size, device=device)

    # 1. m = 0 is the exact identity (up to the resize and the normalisation).
    m0 = torch.zeros(8, N_MAG, device=device)
    y = aug(x, m0)
    ref = aug.normalise_only(x)
    err = float((y - ref).abs().max())
    assert err < 1e-4, f"m=0 is not the identity policy (max abs error {err:.2e})"

    # 2. Strength is monotone: stronger m moves the image further from the source.
    d_prev = -1.0
    for s in (0.0, 0.25, 0.5, 1.0):
        torch.manual_seed(1)
        m = torch.full((8, N_MAG), float(s), device=device)
        d = float((aug(x, m) - ref).pow(2).mean())
        assert d >= d_prev - 1e-6, f"augmentation strength is not monotone at m={s}"
        d_prev = d

    # 3. The budget projection really bounds the realised magnitudes.
    m = torch.rand(64, N_MAG, device=device)
    mc = clip_to_budget(m, 0.7)
    assert float(mc.norm(dim=1).max()) <= 0.7 + 1e-5, "budget projection is leaky"

    # 4. The policy's score-function gradient reaches mu and nothing else.
    pol = MagnitudePolicy(REFERENCE_POLICIES["strong"], sigma=0.5).to(device)
    _, logp = pol.sample(4, None)
    logp.sum().backward()
    assert pol.mu.grad is not None and float(pol.mu.grad.abs().sum()) > 0, \
        "the policy receives no gradient"
    print("[selfcheck] TensorAugment + MagnitudePolicy: OK")


# ===========================================================================
# 11.  CLI
# ===========================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="T5 / T6: the learned augmentation policy degenerates, and T is asymmetric.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--exp", choices=["D1", "D2", "all"], default="all")
    p.add_argument("--all", action="store_true",
                   help="alias for --exp all (both experiment groups)")
    p.add_argument("--arm", type=str, default=None,
                   help="substring filter on cell / policy names, to run one arm only")
    p.add_argument("--outdir", type=str,
                   default=os.path.join(_PROJECT_ROOT, "results", "D"))
    p.add_argument("--data-root", type=str, default=os.path.join(_PROJECT_ROOT, "data"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="at least 3 seeds; a single-seed comparison decides nothing")
    p.add_argument("--seed", type=int, default=None,
                   help="shorthand for --seeds <seed> (single seed, diagnostics only)")

    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--chunks", type=int, default=8,
                   help="M: magnitude samples per step; also the loss decomposition "
                        "used by EVERY arm, so the objective stays identical")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--policy-lr", type=float, default=0.05)
    p.add_argument("--policy-sigma", type=float, default=0.5)
    p.add_argument("--policy-objective", choices=["full", "agreement"], default="full",
                   help="'full' follows the thesis wording (the policy minimises the "
                        "SSL loss); 'agreement' isolates the invariance term")
    p.add_argument("--no-advantage-normalise", action="store_true")
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--arch", type=str, default="resnet18")
    p.add_argument("--stem", choices=["cifar", "stl", "imagenet"], default="cifar")
    p.add_argument("--gram-w", type=float, default=25.0)
    p.add_argument("--var-w", type=float, default=25.0)
    p.add_argument("--cov-w", type=float, default=1.0)
    p.add_argument("--no-amp", action="store_true", help="disable bf16 autocast on CUDA")
    p.add_argument("--probe-epochs", type=int, default=100)
    p.add_argument("--finetune-epochs", type=int, default=20)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--ckpt-every", type=int, default=200)

    p.add_argument("--c-grid", type=float, nargs="+", default=[0.0, 0.4, 0.8, 1.2, 1.6],
                   help="D1 budget sweep")
    p.add_argument("--c0", type=float, default=1.2,
                   help="D1 budget at which the direction sweep is run; must be in --c-grid")
    p.add_argument("--directions", type=str, nargs="+", default=list(D1_DIRECTIONS))
    p.add_argument("--d1-learn", type=str, nargs="+", default=list(MAG_NAMES),
                   help="which magnitudes the D1 policy may learn (see amendment 3)")

    p.add_argument("--synthetic", action="store_true",
                   help="offline synthetic stand-ins for every domain (pipeline test only)")
    p.add_argument("--n-synth-train", type=int, default=4096)
    p.add_argument("--n-synth-test", type=int, default=1024)
    p.add_argument("--chirality-all-digits", action="store_true",
                   help="keep 0/1/8, which are nearly mirror-symmetric (adds an "
                        "irreducible error floor)")

    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--cpu", action="store_true", help="force CPU")
    p.add_argument("--smoke", action="store_true",
                   help="tiny offline end-to-end run, < 3 min on CPU; proves the "
                        "plumbing and nothing else")
    p.add_argument("--figures-only", action="store_true",
                   help="rebuild figures and verdicts from the summaries on disk")
    return p


def apply_smoke(args) -> None:
    """Tiny everything.  The encoder stem is switched to the ImageNet stem
    (stride-2 conv + maxpool), which is ~30x cheaper at 32x32 than the CIFAR
    stem: --smoke tests the pipeline, not the science."""
    args.steps = 6
    args.bs = 32
    args.chunks = 4
    args.size = 32
    args.stem = "imagenet"
    args.seeds = [0]
    args.probe_epochs = 3
    args.finetune_epochs = 1
    args.log_every = 2
    args.eval_every = 0
    args.ckpt_every = 3
    args.synthetic = True
    args.n_synth_train = 192
    args.n_synth_test = 96
    args.c_grid = [0.0, 0.8, 1.6]
    args.c0 = 0.8
    args.directions = ["learned", "uniform", "crop_only"]
    args.no_amp = True


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.all:
        args.exp = "all"
    if args.seed is not None:
        args.seeds = [args.seed]
    if args.smoke:
        apply_smoke(args)
    if len(args.seeds) < 3 and not args.smoke:
        print(f"[warning] {len(args.seeds)} seed(s): per the project rules a "
              f"comparison on fewer than 3 seeds decides nothing.")

    os.makedirs(args.outdir, exist_ok=True)
    device = get_device(prefer_cpu=args.cpu)
    print(f"[env] device={device}  torch={torch.__version__}  outdir={args.outdir}")

    cfg = TrainCfg(
        steps=args.steps, bs=args.bs, chunks=args.chunks, lr=args.lr,
        policy_lr=args.policy_lr, policy_sigma=args.policy_sigma,
        advantage_normalise=not args.no_advantage_normalise,
        policy_objective=args.policy_objective,
        size=args.size, arch=args.arch, stem=args.stem,
        gram_w=args.gram_w, var_w=args.var_w, cov_w=args.cov_w,
        amp=not args.no_amp, log_every=args.log_every, eval_every=args.eval_every,
        ckpt_every=args.ckpt_every, probe_epochs=args.probe_epochs,
        finetune_epochs=args.finetune_epochs,
    )
    if cfg.bs % cfg.chunks != 0:
        raise SystemExit(f"--bs {cfg.bs} must be divisible by --chunks {cfg.chunks}")
    if cfg.bs // cfg.chunks < 4:
        raise SystemExit("a chunk must hold at least 4 samples (Gram and variance "
                         "estimates on fewer rows are noise)")

    selfcheck_augment(device, cfg.size)

    t0 = time.time()
    results: dict = {
        "script": os.path.basename(__file__),
        "theses": ["T5 -- a learned T degenerates to the identity; the prior moves "
                   "into the strength budget",
                   "T6 -- T outside G_task destroys label information, "
                   "irrecoverably under a frozen encoder"],
        "config": {k: v for k, v in vars(args).items()},
        "scale_caveat": ("CIFAR-10 / MNIST / two synthetic domains, ResNet-18, "
                         f"{cfg.steps} steps at batch {cfg.bs} (~"
                         f"{cfg.steps * cfg.bs // 50000} CIFAR epochs). "
                         "Decides signs and orderings for large effects only; "
                         "says nothing about ImageNet-scale SSL."),
        "significance_rule": ("a gap is reported as meaningful only if it exceeds the "
                              "sum of the two seed standard deviations; with 3 seeds "
                              "this is a guard, not a test"),
    }

    if args.figures_only:
        d1_dir = os.path.join(args.outdir, "D1")
        if os.path.isdir(d1_dir):
            df1 = _collect_summaries(d1_dir, "D1_")
            c_grid, dirs, c0 = d1_grids(args)
            results["D1"] = d1_verdicts(df1, c_grid, c0, dirs, {})
            d1_figures(d1_dir, args.outdir, df1, c_grid, c0, dirs, args.seeds)
        p2 = os.path.join(args.outdir, "D2_cells.csv")
        if os.path.exists(p2):
            df2 = pd.read_csv(p2)
            ctx = {}
            results["D2"] = d2_verdicts(df2, ctx)
            d2_figures(args.outdir, df2, ctx)
    else:
        if args.exp in ("D1", "all"):
            print("\n" + "=" * 72 + "\nD1 -- T5: does a learned T degenerate?\n" + "=" * 72)
            results["D1"] = run_d1(args, cfg, device)
        if args.exp in ("D2", "all"):
            print("\n" + "=" * 72 + "\nD2 -- T6: the asymmetry of T\n" + "=" * 72)
            results["D2"] = run_d2(args, cfg, device)

    results["elapsed_s"] = round(time.time() - t0, 1)
    out_path = os.path.join(args.outdir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())

    print("\n" + "=" * 72)
    print(f"VERDICTS  ({out_path})")
    print("=" * 72)
    for group in ("D1", "D2"):
        for key, val in (results.get(group) or {}).items():
            if isinstance(val, dict) and "verdict" in val:
                print(f"  {group}.{key:6s} {val['verdict']:<16s} {val.get('claim', '')[:70]}")
    print(f"\nelapsed {results['elapsed_s']:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
