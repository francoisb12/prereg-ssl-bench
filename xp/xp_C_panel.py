#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xp_C_panel.py -- Thesis T3: "worst case over a panel of heads", smoothed, and the
                 self-sabotage failure of a TRAINABLE panel.

=============================================================================
1. WHAT IS BEING TESTED
=============================================================================
T3 says the agreement objective should be taken at the WORST CASE over K
projection heads rather than on average:

    L = max_k  d_k(theta)          instead of      L = mean_k d_k(theta)

and makes two claims about it.

(a) SMOOTHNESS.  The hard max is non-smooth: only the argmax head receives
    gradient, the K-1 others are starved, and the dynamics oscillate.  The
    proposed fix is a tempered log-sum-exp,

        L_beta = (1/beta) * log sum_k exp(beta * d_k) .

(b) SELF-SABOTAGE.  min_theta max_k has a trivial solution: make the K heads
    converge to the SAME function.  Then max_k d_k == mean_k d_k and the whole
    construction is vacuous.  The proposed fix is to FREEZE the heads (random
    projections drawn at init, never updated).

    Testable, decisive prediction: with TRAINABLE heads under a max/LSE
    objective, panel diversity collapses during training; with FROZEN heads it
    does not.

Honest limit, stated by the project itself and repeated here because it is the
weakest joint of T3: NO THEOREM connects a worst case over projection heads to
a worst case over downstream tasks.  The panel is a heuristic.  C2 below
measures whether the heuristic buys anything on transfer; a null result there
refutes the USEFULNESS of the panel, not the diversity-collapse claim of C1,
and vice versa.  The two must be reported separately.

=============================================================================
2. THREE EXPERIMENTS
=============================================================================
C1 -- PANEL COLLAPSE (the decisive, cheap one).
     10 arms = {mean, max, LSE(beta=1), LSE(beta=5), LSE(beta=20)}
             x {TRAINABLE heads, FROZEN heads},  K = 8, 3 seeds.
     Everything else is held fixed.  Measured along training:
       * cka_head_fn : mean pairwise linear CKA between the heads evaluated on
                       a FIXED isotropic reference input.  This is head
                       diversity in FUNCTION space, independent of the encoder.
       * cka_head_live: same CKA but on the heads' outputs for a fixed batch of
                       real images -- head diversity AS THE DATA SEES IT.
       * d_cv        : std_k(d_k) / mean_k(d_k), the dispersion of the panel.
       * gap_rel     : (max_k d_k - mean_k d_k) / mean_k d_k, i.e. the direct
                       measure of "the construction becomes empty".
       * eff_rank_z  : effective rank of the projector output (confound check,
                       see section 3).
       * w_entropy   : entropy of the LSE gradient weights softmax(beta*d),
                       normalised by log K -- 1.0 = every head gets gradient
                       (mean), 0.0 = a single head gets all of it (hard max).
                       This quantifies defect (a) directly.

C2 -- WORST-CASE TRANSFER.  Longer pretraining, then a linear probe on a
     HETEROGENEOUS battery (CIFAR-10, CIFAR-100, STL-10, DTD, EuroSAT,
     Flowers102).  Always report the WORST task and the full distribution,
     never the mean alone.

C3 -- BETA ABLATION.  Frozen heads, beta in {0 (=mean), 1, 5, 20, inf (=max)}.
     Shows the interpolation numerically and locates the best worst-task /
     mean-task compromise.

=============================================================================
3. WHERE THE PLAN AS GIVEN WAS AMENDED, AND WHY
=============================================================================
These are deliberate deviations.  A sceptical reader should check them first.

(A) THE LSE IS MEAN-NORMALISED.  The plain (1/beta) log sum_k exp(beta d_k)
    does NOT tend to mean_k d_k as beta -> 0: it tends to mean + log(K)/beta,
    which diverges.  C3's claim "beta -> 0 gives back mean" is therefore false
    for the plain form.  We use

        L_beta = (1/beta) * log( (1/K) sum_k exp(beta d_k) )

    which satisfies  mean_k d_k <= L_beta <= max_k d_k,  L_0 = mean, L_inf = max.
    The two forms differ by the ADDITIVE CONSTANT log(K)/beta, so they have
    IDENTICAL GRADIENTS and identical training dynamics: nothing is lost, only
    the reported number becomes interpretable.  (The plain form is the smooth
    UPPER bound on the max; the normalised one is a smooth interpolation.  Both
    are stated in `aggregate_panel`.)

(B) TRAINABLE HEADS ARE BIT-IDENTICAL TWINS OF THE FROZEN ONES.  The trainable
    panel is built by calling `frozen_random_projections(...)` and then
    re-registering the very same tensors as `nn.Parameter`.  Architecture,
    initialisation and RNG consumption are identical across the two conditions;
    the ONLY difference is `requires_grad`.  Otherwise the comparison would
    confound freezing with initialisation.

(C) HEADS READ THE WHOLE PROJECTOR OUTPUT BY DEFAULT (`--disjoint` off).
    `frozen_random_projections(disjoint=True)` gives each head a distinct input
    block, which is a SECOND, structural anti-redundancy device.  Turning it on
    for the frozen arm only would let the frozen arm win for two reasons at
    once, and would make the trainable arm structurally incapable of the very
    collapse we are trying to observe (heads that read disjoint coordinates
    cannot become the same function of z).  Default is therefore
    disjoint=False, i.e. all heads see all of z, which is the configuration in
    which T3's self-sabotage argument actually applies.  `--disjoint` runs the
    other configuration if wanted.

(D) THE CONFOUND: "PANEL COLLAPSE" vs "REPRESENTATION COLLAPSE".  CKA between
    head OUTPUTS can rise to 1 without the heads moving at all, simply because
    the representation z became low-rank -- every linear head of a rank-1
    signal is the same signal up to scale.  Reporting only output CKA would
    therefore not distinguish "the heads converged" (T3's claim) from "the
    encoder collapsed" (T2's failure mode).  Hence:
      * cka_head_fn on a fixed isotropic reference input isolates the heads;
      * eff_rank_z is logged everywhere as the control;
      * the frozen arm is a real control on cka_head_live (frozen heads CAN
        show rising live CKA if the representation collapses), while on
        cka_head_fn it is a tautology (constant by construction) and is used
        only as a correctness assertion.
    Read the verdicts with that distinction in mind; the summary keeps them
    separate.

(E) C2 GOT TWO EXTRA ARMS.  The three arms originally specified (single head /
    K heads + mean / K frozen heads + LSE) differ in THREE variables at once
    (K, aggregation, freezing), so no difference between them can be
    attributed.  C2 therefore runs a 2x2 factorial (freeze x aggregation) plus
    the single-head baseline:
        single, panel_mean_train, panel_lse5_train,
                panel_mean_frozen, panel_lse5_frozen.

(F) C2 AND C3 SHARE THEIR TRAINING RUNS.  `panel_mean_frozen` is exactly
    beta=0 and `panel_max_frozen` exactly beta=inf of the C3 grid; running them
    twice would burn GPU hours for nothing and, worse, would invite comparing
    two arms trained under different budgets.  One long-run pool serves both.

(G) THE PANEL IS SCALE-BLIND.  `gram_loss` L2-normalises its inputs, so a
    trainable head cannot lower d_k by shrinking its output.  If it lowers d_k
    it does so by changing the DIRECTION of its map -- which is what makes the
    collapse measurement meaningful.

=============================================================================
4. EXACT PREDICTIONS AND WHAT WOULD FALSIFY THEM
=============================================================================
Thresholds are pre-registered as module constants (CKA_COLLAPSE_DELTA, ...)
and are applied with a one-standard-deviation margin across the 3 seeds.

P1  TRAINABLE PANEL COLLAPSES.  For trainable heads with max or LSE(beta>=5),
    the mean pairwise function-space CKA rises by at least CKA_COLLAPSE_DELTA
    (=0.20) between the first and last diagnostic, and d_cv falls to at most
    CV_COLLAPSE_RATIO (=0.5) of its initial value.
    FALSIFIED IF: that CKA rise is under 0.05 (seed mean + std) while d_cv does
    not fall -- i.e. trainable heads stay as diverse as they started.  Then
    freezing the heads solves a non-problem and T3's remedy is unmotivated.

P2  THE CONSTRUCTION BECOMES EMPTY.  For the same arms, gap_rel falls to at
    most GAP_EMPTY_RATIO (=0.10) of its initial value: max becomes mean.
    FALSIFIED IF: gap_rel stays above half its initial value.

P3  FROZEN CONTROL.  Under the same aggregation, frozen heads keep a strictly
    larger d_cv and a strictly larger gap_rel at the end of training than
    trainable heads.
    FALSIFIED IF: frozen arms lose panel dispersion just as fast -- which would
    mean the dispersion loss comes from the representation, not from the heads,
    and T3's diagnosis of the mechanism is wrong even if the phenomenon is real.

P4  TRANSFER (the heuristic's usefulness -- NO theorem backs this one).
    We do not predict a direction for the mean; the claim under test is that
    the frozen-panel worst-case arm has a HIGHER WORST TASK than the
    single-head baseline, by more than seed noise.
    FALSIFIED IF: worst-task(panel_lse5_frozen) <= worst-task(single) across
    seeds.  Such a null refutes only the usefulness of the panel at this scale.

P5  BETA LIMITS.  L_beta -> mean as beta -> 0 and -> max as beta -> inf
    (deterministic numerical check, run even in --smoke), and the worst-task /
    mean-task compromise over beta is reported without a predicted optimum.

=============================================================================
5. WHAT THIS SCALE CAN AND CANNOT DECIDE
=============================================================================
Pretraining is ResNet-18 on CIFAR-10 at 32x32.  This is NOT ImageNet.

  * C1 (panel diversity) is an OPTIMISATION-DYNAMICS claim about an objective
    and a set of heads.  It does not depend on the dataset being large, and a
    collapse observed here is genuine evidence.  It remains possible that the
    dynamics differ at ImageNet scale; nothing here excludes that.
  * C2/C3 (transfer) are the weak part.  Probes on DTD / Flowers102 / EuroSAT
    are run at 32x32 because that is the pretraining resolution, so absolute
    accuracies are far below the published numbers for those datasets.  They
    are COMPARABLE ACROSS ARMS (identical protocol, identical budget) and are
    NOT absolute.  Any claim of the form "method X transfers better" must be
    read as "at CIFAR scale and 32x32 probes".
  * Three seeds bound seed noise, they do not bound architecture or
    hyper-parameter sensitivity.  Only one learning rate is swept.

=============================================================================
6. COST (single A100 40GB, Colab Pro), at the defaults
=============================================================================
Measured unit: ResNet-18, 32x32, batch 256, two encoder forwards + one
backward, bf16 autocast ~ 0.075 s/step on an A100.

  C1  : 10 arms x 3 seeds x 8 000 steps   ~ 10 min/run  ->  ~5.0 A100-hours
  C2+C3 (shared pool): 8 arms x 3 seeds x 15 000 steps ~ 19 min/run -> ~7.6 h
  probe battery (6 tasks, cached features) ~ 3 min x 24 runs        -> ~1.2 h
  diagnostics + checkpointing overhead                              -> ~0.5 h
  ------------------------------------------------------------------------
  TOTAL                                                             ~ 14.5 A100-hours
  C1 ALONE, which is the decisive part, is ~5 hours and can be run first.

First run also downloads DTD (~600 MB), Flowers102 (~350 MB), EuroSAT
(~90 MB), STL-10 (~2.5 GB), CIFAR-10/100 (~350 MB): budget wall-clock for that,
it is not GPU time.

=============================================================================
7. USAGE
=============================================================================
    python xp/xp_C_panel.py --smoke                 # < 3 min, CPU, synthetic data
    python xp/xp_C_panel.py --exp c1                # the decisive experiment
    python xp/xp_C_panel.py --all                   # everything
    python xp/xp_C_panel.py --exp c1 --seed 0       # one seed
    python xp/xp_C_panel.py --arm c1_max_train      # one arm
    python xp/xp_C_panel.py --analyse-only          # figures + verdicts only

Outputs land in results/C: one CSV + config + checkpoint + summary per
(arm, seed), the figures fig_C1_*.png / fig_C2_*.png / fig_C3_*.png, and the
group verdict file summary.json.
"""

from __future__ import annotations

import argparse
import copy
import glob as _glob
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- in-house imports: lib.harness and nothing else --------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.harness import (  # noqa: E402
    AugCfg,
    DATASET_NUM_CLASSES,
    EMATeacher,
    Run,
    aggregate_seeds,
    effective_rank,
    frozen_random_projections,
    get_device,
    get_eval_datasets,
    get_ssl_dataset,
    gram_loss,
    knn_probe,
    linear_cka,
    linear_probe,
    make_encoder,
    make_eval_transform,
    make_mlp,
    matched_budget_check,
    probe_battery,
    set_seed,
    vicreg_reg,
)


# =============================================================================
# Pre-registered decision thresholds.  Changing these after seeing the results
# would be cheating; they live here so that a reader can check they were not.
# =============================================================================
CKA_COLLAPSE_DELTA = 0.20   # P1: minimum rise in pairwise function-space CKA
CKA_NULL_DELTA = 0.05       # P1: rise below this counts as "no collapse"
CV_COLLAPSE_RATIO = 0.50    # P1: final d_cv / initial d_cv at most this
GAP_EMPTY_RATIO = 0.10      # P2: final gap_rel / initial gap_rel at most this
GAP_ALIVE_RATIO = 0.50      # P2: above this, the construction is still alive
FROZEN_DRIFT_TOL = 1e-5     # sanity: a frozen head must not move at all

TRANSFER_BATTERY = ["cifar10", "cifar100", "stl10", "dtd", "eurosat", "flowers102"]

VERDICT_CONFIRMED = "confirmee"
VERDICT_REFUTED = "infirmee"
VERDICT_INCONCLUSIVE = "non concluante"


# =============================================================================
# 1. The panel aggregators
# =============================================================================

def aggregate_panel(d: torch.Tensor, mode: str, beta: float) -> torch.Tensor:
    """Aggregate the K per-head agreement losses ``d`` (shape ``[K]``).

    mode='mean' : mean_k d_k                        (the standard objective)
    mode='max'  : max_k d_k                         (T3, non-smooth)
    mode='lse'  : (1/beta) log( (1/K) sum_k exp(beta d_k) )

    The LSE form is MEAN-NORMALISED (the ``1/K``).  Consequences, spelled out
    because this is the one place where the write-up and the code could drift
    apart:

      * mean_k d_k  <=  L_beta  <=  max_k d_k, with equality at beta -> 0 and
        beta -> +inf respectively.  The unnormalised (1/beta) log sum exp is
        instead a smooth UPPER bound on the max and tends to mean + log(K)/beta,
        which BLOWS UP as beta -> 0 -- so "beta -> 0 recovers the mean" is only
        true of the normalised form.
      * The two forms differ by the constant log(K)/beta, which does not depend
        on theta.  Their GRADIENTS are identical, hence so is training.  The
        normalisation changes the reported number, never the experiment.

    The gradient of L_beta w.r.t. the panel is a softmax-weighted combination,
    dL/dd_k = softmax(beta*d)_k, which is exactly what makes the hard max
    pathological: at beta = inf a single head carries all the gradient and the
    other K-1 are starved.  ``panel_gradient_weights`` below logs that.
    """
    if d.dim() != 1:
        raise ValueError(f"aggregate_panel expects a [K] tensor, got {tuple(d.shape)}")
    if mode == "mean":
        return d.mean()
    if mode == "max":
        return d.max()
    if mode == "lse":
        beta = float(beta)
        if not (beta > 0):
            raise ValueError("lse aggregation needs beta > 0 (use mode='mean' for beta=0)")
        K = d.numel()
        return (torch.logsumexp(beta * d, dim=0) - math.log(K)) / beta
    raise ValueError(f"unknown aggregation mode {mode!r} (mean|max|lse)")


def panel_gradient_weights(d: torch.Tensor, mode: str, beta: float) -> torch.Tensor:
    """The weights with which each head's loss enters the gradient, ``[K]``.

    mean -> uniform 1/K ; max -> one-hot on the argmax ; lse -> softmax(beta d).
    Purely diagnostic (no grad).
    """
    with torch.no_grad():
        K = d.numel()
        if mode == "mean":
            return torch.full_like(d, 1.0 / K)
        if mode == "max":
            w = torch.zeros_like(d)
            w[int(torch.argmax(d))] = 1.0
            return w
        return torch.softmax(float(beta) * d, dim=0)


def normalised_weight_entropy(w: torch.Tensor) -> float:
    """H(w)/log K in [0,1].  1 = every head gets gradient, 0 = only one does."""
    K = w.numel()
    if K < 2:
        return float("nan")
    p = w.clamp_min(1e-12)
    h = float(-(p * p.log()).sum())
    return h / math.log(K)


def check_lse_limits(verbose: bool = True) -> Dict[str, float]:
    """Deterministic check of prediction P5 -- the beta limits.

    Run in every mode including --smoke: it costs microseconds and it is the
    one part of C3 that is a mathematical identity rather than an empirical
    claim, so it should never be allowed to silently break.
    """
    g = torch.Generator().manual_seed(12345)
    d = torch.rand(8, generator=g, dtype=torch.float64) * 0.5 + 0.1
    lo = float(aggregate_panel(d, "lse", 1e-4))
    hi = float(aggregate_panel(d, "lse", 1e5))
    err_mean = abs(lo - float(d.mean()))
    err_max = abs(hi - float(d.max()))
    monotone = True
    prev = float(d.mean())
    for b in (0.5, 1.0, 5.0, 20.0, 100.0, 1e4):
        cur = float(aggregate_panel(d, "lse", b))
        monotone = monotone and (cur >= prev - 1e-9)
        prev = cur
    monotone = monotone and (prev <= float(d.max()) + 1e-9)
    if verbose:
        print(f"[P5] LSE limits: |L(beta=1e-4) - mean| = {err_mean:.3e} ; "
              f"|L(beta=1e5) - max| = {err_max:.3e} ; monotone in beta: {monotone}")
    return {"err_to_mean": err_mean, "err_to_max": err_max,
            "monotone": bool(monotone)}


# =============================================================================
# 2. The panel of heads: frozen vs trainable, bit-identical at init
# =============================================================================

class TrainableProjection(nn.Module):
    """A trainable clone of ``lib.harness.FrozenProjection``.

    Identical forward, identical initial values; the weight and bias are
    ``nn.Parameter`` instead of buffers, and the (optional) input index mask
    stays a buffer.  Built only from an existing frozen head, so the two
    conditions of C1 cannot differ by initialisation.
    """

    def __init__(self, frozen_head: nn.Module):
        super().__init__()
        self.in_dim = int(frozen_head.in_dim)
        self.out_dim = int(frozen_head.out_dim)
        idx = getattr(frozen_head, "idx", None)
        if idx is None:
            self.register_buffer("idx", None)
        else:
            self.register_buffer("idx", idx.clone().long())
        self.weight = nn.Parameter(frozen_head.weight.detach().clone())
        self.bias = nn.Parameter(frozen_head.bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"TrainableProjection expects [B,d], got {tuple(x.shape)}")
        if self.idx is not None:
            x = x.index_select(1, self.idx)
        return torch.nn.functional.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        sub = self.in_dim if self.idx is None else int(self.idx.numel())
        return f"in={self.in_dim}, sub={sub}, out={self.out_dim}, frozen=False"


def build_panel(proj_dim: int, head_dim: int, k: int, seed: int,
                trainable: bool, disjoint: bool) -> nn.ModuleList:
    """Build the K-head panel.

    Both conditions start from ``frozen_random_projections`` with the same seed,
    so the trainable panel is the frozen panel with ``requires_grad=True``.
    That is the whole experimental contrast of C1 and nothing else changes.
    """
    frozen = frozen_random_projections(proj_dim, head_dim, k, seed=seed,
                                       disjoint=disjoint)
    if not trainable:
        return nn.ModuleList(frozen)
    return nn.ModuleList([TrainableProjection(h) for h in frozen])


def panel_is_trainable(panel: nn.ModuleList) -> bool:
    return sum(p.numel() for p in panel.parameters()) > 0


def panel_weight_signature(panel: nn.ModuleList) -> torch.Tensor:
    """Flat copy of every head weight, used to assert frozen heads never move."""
    parts = [h.weight.detach().float().reshape(-1).cpu() for h in panel]
    return torch.cat(parts)


# =============================================================================
# 3. The model: encoder -> projector -> z (unnormalised) -> K heads
# =============================================================================

class Trunk(nn.Module):
    """Encoder + projector.  ``forward`` returns the UNNORMALISED z.

    Unnormalised on purpose: ``vicreg_reg`` (the T2 legality term that is the
    only thing standing between this objective and total collapse) refuses an
    L2-normalised input, and rightly so.
    """

    def __init__(self, encoder: nn.Module, projector: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.encoder(x)
        if f.dim() > 2:
            f = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(f, 1), 1)
        return self.projector(f)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        f = self.encoder(x)
        if f.dim() > 2:
            f = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(f, 1), 1)
        return f


def per_head_losses(panel: nn.ModuleList, z_student: torch.Tensor,
                    z_target: torch.Tensor) -> torch.Tensor:
    """d_k = gram_loss( head_k(z_student), head_k(z_target) ) for every head.

    ``gram_loss`` L2-normalises internally and detaches its target argument, so
    (i) a head cannot lower its loss by rescaling its output, and (ii) the
    target branch is a genuine stop-gradient.  Returns a ``[K]`` tensor that
    still carries grad w.r.t. the trunk and (if trainable) the heads.
    """
    return torch.stack([gram_loss(h(z_student), h(z_target)) for h in panel])


# =============================================================================
# 4. Diagnostics: panel diversity, with the representation-collapse control
# =============================================================================

class PanelDiagnostics:
    """Everything measured along training, computed on FIXED inputs.

    Two probes, deliberately distinct (see docstring section 3(D)):

    * ``cka_head_fn``  -- heads applied to a fixed isotropic Gaussian reference
      matrix ``Z_ref``.  Depends on the head parameters ONLY.  For frozen heads
      it is constant by construction and is checked to be so; for trainable
      heads it is the direct measurement of "the K heads became one function".
      The reference is drawn from a constant seed, identical across arms and
      seeds, so the numbers are comparable everywhere.

    * ``cka_head_live`` -- heads applied to the projector output of a fixed
      batch of real images.  This one CAN move even for frozen heads, because
      it also reflects the rank of the representation.  It is reported next to
      ``eff_rank_z`` so the two causes stay separable.
    """

    REF_SEED = 2718  # constant on purpose: the reference must not vary by arm

    def __init__(self, proj_dim: int, n_ref: int, images: torch.Tensor,
                 device: torch.device):
        g = torch.Generator().manual_seed(self.REF_SEED)
        self.z_ref = torch.randn(n_ref, proj_dim, generator=g).to(device)
        self.images = images.to(device)

    @staticmethod
    def _mean_pairwise_cka(mats: Sequence[torch.Tensor]) -> float:
        k = len(mats)
        if k < 2:
            return float("nan")
        vals = []
        for i in range(k):
            for j in range(i + 1, k):
                vals.append(linear_cka(mats[i], mats[j]))
        vals = [v for v in vals if not math.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    @torch.no_grad()
    def compute(self, trunk: Trunk, panel: nn.ModuleList) -> Dict[str, float]:
        was_training = trunk.training
        trunk.eval()
        panel.eval()

        # function-space diversity: heads only, encoder irrelevant
        ref_out = [h(self.z_ref).float() for h in panel]
        cka_fn = self._mean_pairwise_cka(ref_out)

        # live diversity + representation rank (the confound control)
        f = trunk.features(self.images).float()
        z = trunk.projector(f).float()
        live_out = [h(z).float() for h in panel]
        cka_live = self._mean_pairwise_cka(live_out)

        out = {
            "cka_head_fn": cka_fn,
            "cka_head_live": cka_live,
            "eff_rank_f": effective_rank(f),
            "eff_rank_z": effective_rank(z),
            "z_norm_mean": float(z.norm(dim=1).mean()),
        }
        if was_training:
            trunk.train()
        panel.train()
        return out


def panel_dispersion_stats(d: torch.Tensor) -> Dict[str, float]:
    """Dispersion of the per-head losses: the "is max still different from mean"
    family of numbers.

    ``gap_abs`` alone is NOT a valid measure of "the construction became empty":
    every d_k shrinks during training, so gap_abs shrinks even if the panel
    stays perfectly dispersed in relative terms.  ``gap_rel`` and ``d_cv``
    (both scale-free) are the quantities the verdicts use; gap_abs is logged
    for completeness only.
    """
    dd = d.detach().float()
    m = float(dd.mean())
    sd = float(dd.std(unbiased=True)) if dd.numel() > 1 else 0.0
    mx = float(dd.max())
    denom = max(abs(m), 1e-12)
    return {
        "d_mean": m,
        "d_max": mx,
        "d_min": float(dd.min()),
        "d_std": sd,
        "d_cv": sd / denom,
        "gap_abs": mx - m,
        "gap_rel": (mx - m) / denom,
    }


# =============================================================================
# 5. Synthetic data for --smoke (no download, no network)
# =============================================================================

class SyntheticTwoView(Dataset):
    """Two-view dataset with the same item contract as harness' SSLDataset.

    Each sample is a fixed class-dependent low-frequency pattern plus per-view
    noise, so a linear probe on it is neither trivial nor impossible.  Used only
    by --smoke: it exercises the whole training/diagnostic/checkpoint path
    without touching the network.
    """

    def __init__(self, n: int = 128, size: int = 32, num_classes: int = 4,
                 seed: int = 0, noise: float = 0.35):
        g = torch.Generator().manual_seed(int(seed))
        self.labels = torch.randint(0, num_classes, (n,), generator=g)
        base = torch.randn(num_classes, 3, size, size, generator=g)
        self.base = base
        self.jitter = torch.randn(n, 3, size, size, generator=g) * 0.2
        self.noise = float(noise)
        self.n = int(n)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.n

    def _view(self, i: int, v: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + i * 17 + v)
        y = int(self.labels[i])
        return self.base[y] + self.jitter[i] + torch.randn(
            self.base[y].shape, generator=g) * self.noise

    def __getitem__(self, i: int) -> dict:
        return {"v1": self._view(i, 0), "v2": self._view(i, 1),
                "idx": int(i), "label": int(self.labels[i])}


class SyntheticEval(Dataset):
    """(x, y) view of a SyntheticTwoView, deterministic, for the smoke probe."""

    def __init__(self, src: SyntheticTwoView, split: str = "train"):
        self.src = src
        self.offset = 0 if split == "train" else 7

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, i: int):
        y = int(self.src.labels[i])
        x = self.src.base[y] + self.src.jitter[i]
        if self.offset:
            g = torch.Generator().manual_seed(i * 31 + self.offset)
            x = x + torch.randn(x.shape, generator=g) * 0.1
        return x, y


# =============================================================================
# 6. Arm specification and the experiment registry
# =============================================================================

@dataclass
class ArmSpec:
    name: str
    agg: str                 # 'mean' | 'max' | 'lse'
    beta: float              # 0.0 for mean, inf for max, >0 for lse
    k_heads: int
    trainable_heads: bool
    steps: int
    group: str               # 'c1' (short, diversity) | 'long' (transfer)
    run_battery: bool
    disjoint: bool = False

    def beta_axis(self) -> float:
        """Position on the beta axis used by the C3 figure (0 = mean, inf = max)."""
        if self.agg == "mean":
            return 0.0
        if self.agg == "max":
            return float("inf")
        return float(self.beta)

    def label(self) -> str:
        agg = {"mean": "mean", "max": "max"}.get(self.agg, f"LSE b={self.beta:g}")
        head = "trainable" if self.trainable_heads else "frozen"
        return f"{agg} / {head} (K={self.k_heads})"


C1_BETAS = [("mean", 0.0), ("lse", 1.0), ("lse", 5.0), ("lse", 20.0), ("max", float("inf"))]


def build_registry(args) -> Dict[str, ArmSpec]:
    """All arms of the group, keyed by name.  Deterministic, no side effects."""
    reg: Dict[str, ArmSpec] = {}

    def add(a: ArmSpec) -> None:
        reg[a.name] = a

    # ---- C1: 5 aggregations x {trainable, frozen}, short runs, no battery ----
    for agg, beta in C1_BETAS:
        tag = agg if agg != "lse" else f"lse{beta:g}"
        for trainable in (True, False):
            suffix = "train" if trainable else "frozen"
            add(ArmSpec(name=f"c1_{tag}_{suffix}", agg=agg, beta=beta,
                        k_heads=args.k_heads, trainable_heads=trainable,
                        steps=args.c1_steps, group="c1", run_battery=False,
                        disjoint=args.disjoint))

    # ---- long pool, shared by C2 and C3 -------------------------------------
    # C2 = single-head baseline + the 2x2 factorial (freeze x aggregation).
    add(ArmSpec(name="L_single", agg="mean", beta=0.0, k_heads=1,
                trainable_heads=True, steps=args.long_steps, group="long",
                run_battery=True, disjoint=False))
    add(ArmSpec(name="L_panel_mean_train", agg="mean", beta=0.0, k_heads=args.k_heads,
                trainable_heads=True, steps=args.long_steps, group="long",
                run_battery=True, disjoint=args.disjoint))
    add(ArmSpec(name="L_panel_lse5_train", agg="lse", beta=5.0, k_heads=args.k_heads,
                trainable_heads=True, steps=args.long_steps, group="long",
                run_battery=True, disjoint=args.disjoint))
    # C3 = the frozen beta grid; its beta=0 and beta=5 members are also C2 arms.
    for agg, beta in C1_BETAS:
        tag = agg if agg != "lse" else f"lse{beta:g}"
        add(ArmSpec(name=f"L_panel_{tag}_frozen", agg=agg, beta=beta,
                    k_heads=args.k_heads, trainable_heads=False,
                    steps=args.long_steps, group="long", run_battery=True,
                    disjoint=args.disjoint))
    return reg


C2_ARMS = ["L_single", "L_panel_mean_train", "L_panel_lse5_train",
           "L_panel_mean_frozen", "L_panel_lse5_frozen"]
C3_ARMS = ["L_panel_mean_frozen", "L_panel_lse1_frozen", "L_panel_lse5_frozen",
           "L_panel_lse20_frozen", "L_panel_max_frozen"]
C1_ARMS = [f"c1_{a if a != 'lse' else 'lse%g' % b}_{s}"
           for a, b in C1_BETAS for s in ("train", "frozen")]


def select_arms(args, reg: Dict[str, ArmSpec]) -> List[ArmSpec]:
    if args.arm:
        missing = [a for a in args.arm if a not in reg]
        if missing:
            raise SystemExit(f"unknown arm(s) {missing}; available: {sorted(reg)}")
        return [reg[a] for a in args.arm]
    wanted: List[str] = []
    exps = ["c1", "c2", "c3"] if (args.all or args.exp == "all") else [args.exp]
    for e in exps:
        wanted += {"c1": C1_ARMS, "c2": C2_ARMS, "c3": C3_ARMS}[e]
    seen, out = set(), []
    for n in wanted:
        if n not in seen:
            seen.add(n)
            out.append(reg[n])
    return out


# =============================================================================
# 7. Training one arm at one seed
# =============================================================================

def cosine_lr(step: int, total: int, base_lr: float, warmup: int) -> float:
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    t = min(max(t, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


def rng_state() -> dict:
    return {"torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate()}


def restore_rng(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def infinite_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def build_datasets(args, seed: int):
    """Return (ssl_dataset, diagnostic_images, eval_pair_or_None)."""
    if args.smoke:
        ds = SyntheticTwoView(n=args.smoke_n, size=args.size, num_classes=4, seed=seed)
        diag = torch.stack([ds[i]["v1"] for i in range(min(args.diag_n, len(ds)))])
        ev = (SyntheticEval(ds, "train"), SyntheticEval(ds, "test"))
        return ds, diag, ev

    cfg = AugCfg.strong(size=args.size) if args.aug == "strong" else AugCfg.weak(size=args.size)
    ds = get_ssl_dataset(args.dataset, cfg, None, imbalance_gamma=None,
                         seed=seed, root=args.data_root)
    # Diagnostics use the deterministic EVAL transform on the test split: no
    # augmentation randomness, identical images for every arm and every step.
    _, test_ds = get_eval_datasets(args.dataset, size=args.size, root=args.data_root)
    idx = list(range(min(args.diag_n, len(test_ds))))
    diag = torch.stack([test_ds[i][0] for i in idx])
    return ds, diag, None


def train_arm(spec: ArmSpec, seed: int, args, device: torch.device) -> dict:
    """Train one (arm, seed).  Fully resumable; one CSV row per diagnostic."""
    run_name = f"{spec.name}_s{seed}"
    config = {"arm": asdict(spec), "seed": seed, "dataset": args.dataset,
              "bs": args.bs, "lr": args.lr, "wd": args.wd, "size": args.size,
              "proj_dim": args.proj_dim, "head_dim": args.head_dim,
              "var_w": args.var_w, "cov_w": args.cov_w, "arch": args.arch,
              "aug": args.aug, "smoke": bool(args.smoke),
              "ema_tau_base": args.ema_tau, "forward_passes": 2}
    run = Run(run_name, args.outdir, config, resume=args.resume,
              higher_is_better=False)

    set_seed(seed)
    ssl_ds, diag_images, smoke_eval = build_datasets(args, seed)

    encoder, feat_dim = make_encoder(args.arch, dataset=args.encoder_variant)
    projector = make_mlp(feat_dim, args.hidden, args.proj_dim,
                         n_layers=args.proj_layers, bn=True)
    trunk = Trunk(encoder, projector).to(device)
    panel = build_panel(args.proj_dim, args.head_dim, spec.k_heads,
                        seed=seed + 10_000, trainable=spec.trainable_heads,
                        disjoint=spec.disjoint).to(device)

    trainable_panel = panel_is_trainable(panel)
    if trainable_panel != spec.trainable_heads:
        raise RuntimeError(
            f"panel trainability mismatch for {spec.name}: asked "
            f"{spec.trainable_heads}, got {trainable_panel}")

    params = list(trunk.parameters()) + list(panel.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    ema = EMATeacher(trunk, tau_base=args.ema_tau, tau_final=1.0,
                     total_steps=spec.steps)
    ema.to(device)

    diag = PanelDiagnostics(args.proj_dim, args.n_ref, diag_images, device)
    frozen_signature = None if trainable_panel else panel_weight_signature(panel)

    start_step = 0
    ck = run.load_ckpt() if args.resume else None
    if ck is not None:
        trunk.load_state_dict(ck["trunk"])
        panel.load_state_dict(ck["panel"])
        opt.load_state_dict(ck["opt"])
        ema.load_state_dict(ck["ema"])
        ema.to(device)
        restore_rng(ck["rng"])
        start_step = int(ck["step"]) + 1
        print(f"  [resume] {run_name}: restarting at step {start_step}")
    if start_step >= spec.steps:
        print(f"  [skip] {run_name}: already complete ({spec.steps} steps)")
        return json.load(open(run.summary_path, encoding="utf-8"))["summary"] \
            if os.path.exists(run.summary_path) else {}

    loader = DataLoader(ssl_ds, batch_size=args.bs, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        pin_memory=(device.type == "cuda"),
                        persistent_workers=(args.workers > 0))
    stream = infinite_loader(loader)

    use_amp = (device.type == "cuda") and not args.no_amp
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_amp \
        else (lambda: torch.autocast("cpu", enabled=False))

    # running accumulators, reset at each diagnostic: they turn the per-batch
    # noise of d_k into a low-variance estimate at zero extra cost.
    acc_d = torch.zeros(spec.k_heads)
    acc_extra = {"loss": 0.0, "agg": 0.0, "vic_var": 0.0, "vic_cov": 0.0,
                 "w_entropy": 0.0}
    acc_n = 0
    first_row: Optional[dict] = None
    t_start = time.time()

    trunk.train()
    panel.train()
    for step in range(start_step, spec.steps):
        lr = cosine_lr(step, spec.steps, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr

        batch = next(stream)
        v1 = batch["v1"].to(device, non_blocking=True)
        v2 = batch["v2"].to(device, non_blocking=True)

        with amp_ctx():
            z1 = trunk(v1)
            with torch.no_grad():
                z2 = ema.target(v2)
            d = per_head_losses(panel, z1, z2.detach())
            agreement = aggregate_panel(d, spec.agg, spec.beta)
        # legality term (T2) in fp32 on the UNNORMALISED z: this is the only
        # thing preventing the global-minimum collapse of the relational target.
        legality, vic_parts = vicreg_reg(z1.float(), var_w=args.var_w,
                                         cov_w=args.cov_w, gamma=1.0)
        loss = agreement.float() + legality

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.clip)
        opt.step()
        ema.update(step)

        w = panel_gradient_weights(d.detach(), spec.agg, spec.beta)
        acc_d += d.detach().float().cpu()
        acc_extra["loss"] += float(loss.detach())
        acc_extra["agg"] += float(agreement.detach())
        acc_extra["vic_var"] += vic_parts["var"]
        acc_extra["vic_cov"] += vic_parts["cov"]
        acc_extra["w_entropy"] += normalised_weight_entropy(w)
        acc_n += 1

        is_last = (step == spec.steps - 1)
        if (step % args.diag_every == 0) or is_last:
            mean_d = acc_d / max(acc_n, 1)
            row = dict(panel_dispersion_stats(mean_d))
            row.update({k: v / max(acc_n, 1) for k, v in acc_extra.items()})
            row.update(diag.compute(trunk, panel))
            row.update({"lr": lr, "ema_tau": ema.last_tau, "seed": seed,
                        "arm": spec.name})
            if frozen_signature is not None:
                drift = float((panel_weight_signature(panel) - frozen_signature)
                              .abs().max())
                row["frozen_head_drift"] = drift
                if drift > FROZEN_DRIFT_TOL:
                    raise RuntimeError(
                        f"{run_name}: a FROZEN head moved by {drift:.3e} at step "
                        f"{step}. The frozen control is void; fix before using "
                        f"any C1 number.")
            run.log(step, **row)
            if first_row is None:
                first_row = dict(row)
            acc_d.zero_()
            acc_n = 0
            for k in acc_extra:
                acc_extra[k] = 0.0

        if (step % args.ckpt_every == 0 and step > start_step) or is_last:
            run.save_ckpt(step, trunk=trunk.state_dict(), panel=panel.state_dict(),
                          opt=opt.state_dict(), ema=ema.state_dict(),
                          rng=rng_state(), best_metric=None)

    train_s = time.time() - t_start

    # ---- final evaluation ---------------------------------------------------
    summary: Dict[str, object] = {
        "arm": spec.name, "seed": seed, "steps": spec.steps,
        "train_seconds": round(train_s, 1),
        "sec_per_step": round(train_s / max(1, spec.steps - start_step), 4),
    }
    last = _read_last_row(run.csv_path)
    for k in ("cka_head_fn", "cka_head_live", "d_cv", "gap_rel", "gap_abs",
              "d_mean", "eff_rank_z", "eff_rank_f", "w_entropy"):
        summary[f"final_{k}"] = last.get(k, float("nan"))
    first = first_row or _read_first_row(run.csv_path)
    for k in ("cka_head_fn", "cka_head_live", "d_cv", "gap_rel"):
        summary[f"init_{k}"] = first.get(k, float("nan"))

    trunk.eval()
    if args.smoke:
        tr, te = smoke_eval
        probe = linear_probe(trunk.encoder, tr, te, epochs=args.probe_epochs,
                             bs=64, device=device, num_classes=4)
        summary["battery"] = {"per_task": {"synthetic": probe["acc"]},
                              "worst": probe["acc"], "mean": probe["acc"]}
        summary["knn_cifar10"] = float("nan")
    else:
        if spec.run_battery and not args.no_eval:
            bat = probe_battery(trunk.encoder, args.tasks, device=device,
                                root=args.data_root, size=args.size,
                                epochs=args.probe_epochs)
            summary["battery"] = bat
        if not args.no_eval:
            tr, te = get_eval_datasets(args.dataset, size=args.size, root=args.data_root)
            summary["knn_cifar10"] = knn_probe(trunk.encoder, tr, te, k=20, device=device)

    run.finish(summary)
    print(f"  [done] {run_name}: {train_s/60:.1f} min, "
          f"cka_fn {summary.get('init_cka_head_fn', float('nan')):.3f} -> "
          f"{summary.get('final_cka_head_fn', float('nan')):.3f}, "
          f"d_cv {summary.get('init_d_cv', float('nan')):.3f} -> "
          f"{summary.get('final_d_cv', float('nan')):.3f}")
    return summary


def _read_last_row(csv_path: str) -> dict:
    import pandas as pd
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    return {} if df.empty else df.iloc[-1].to_dict()


def _read_first_row(csv_path: str) -> dict:
    import pandas as pd
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    return {} if df.empty else df.iloc[0].to_dict()


# =============================================================================
# 8. Analysis: seed aggregation, figures, verdicts
# =============================================================================

def load_summaries(outdir: str, arm: str, seeds: Sequence[int]) -> List[dict]:
    out = []
    for s in seeds:
        p = os.path.join(outdir, f"{arm}_s{s}.summary.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out.append(json.load(f)["summary"])
    return out


def mean_std(values: Sequence[float]) -> Tuple[float, float, int]:
    v = [float(x) for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return float("nan"), float("nan"), 0
    m = float(np.mean(v))
    s = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return m, s, len(v)


def verdict_greater(mean: float, std: float, threshold: float,
                    null_threshold: Optional[float] = None) -> str:
    """One-sided verdict with a one-standard-deviation margin across seeds.

    confirmee      : mean - std  >  threshold
    infirmee       : mean + std  <  null_threshold (if given)
    non concluante : otherwise, including too few seeds
    """
    if math.isnan(mean) or math.isnan(std):
        return VERDICT_INCONCLUSIVE
    if mean - std > threshold:
        return VERDICT_CONFIRMED
    if null_threshold is not None and mean + std < null_threshold:
        return VERDICT_REFUTED
    return VERDICT_INCONCLUSIVE


def verdict_smaller(mean: float, std: float, threshold: float,
                    null_threshold: Optional[float] = None) -> str:
    if math.isnan(mean) or math.isnan(std):
        return VERDICT_INCONCLUSIVE
    if mean + std < threshold:
        return VERDICT_CONFIRMED
    if null_threshold is not None and mean - std > null_threshold:
        return VERDICT_REFUTED
    return VERDICT_INCONCLUSIVE


# ---- plotting helpers (grayscale-safe) --------------------------------------
AGG_STYLE = {
    "mean": {"color": "0.65", "marker": "o"},
    "lse1": {"color": "0.45", "marker": "s"},
    "lse5": {"color": "0.25", "marker": "^"},
    "lse20": {"color": "0.10", "marker": "D"},
    "max": {"color": "0.00", "marker": "v"},
}


def arm_tag(spec: ArmSpec) -> str:
    return spec.agg if spec.agg != "lse" else f"lse{spec.beta:g}"


def arm_style(spec: ArmSpec) -> dict:
    st = dict(AGG_STYLE.get(arm_tag(spec), {"color": "0.3", "marker": "o"}))
    st["linestyle"] = "-" if spec.trainable_heads else "--"
    st["markersize"] = 4
    st["markevery"] = 0.2
    st["linewidth"] = 1.6
    return st


def plot_metric_over_time(ax, outdir: str, specs: Sequence[ArmSpec], metric: str,
                          title: str, ylabel: str) -> int:
    plotted = 0
    for spec in specs:
        pattern = os.path.join(outdir, f"{spec.name}_s*.csv")
        if not _glob.glob(pattern):
            continue
        try:
            df = aggregate_seeds(pattern, "step")
        except Exception as e:
            print(f"  [plot] {spec.name}: {e}")
            continue
        mcol, scol = f"{metric}_mean", f"{metric}_std"
        if mcol not in df.columns:
            continue
        st = arm_style(spec)
        ax.plot(df["step"], df[mcol], label=spec.label(), **st)
        if scol in df.columns:
            lo = df[mcol] - df[scol].fillna(0.0)
            hi = df[mcol] + df[scol].fillna(0.0)
            ax.fill_between(df["step"], lo, hi, color=st["color"], alpha=0.15,
                            linewidth=0)
        plotted += 1
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    return plotted


def analyse_c1(args, reg: Dict[str, ArmSpec], seeds: Sequence[int]) -> dict:
    """Panel collapse.  Figures + verdicts P1, P2, P3."""
    specs = [reg[n] for n in C1_ARMS]
    present = [s for s in specs if _glob.glob(os.path.join(args.outdir, f"{s.name}_s*.csv"))]
    if not present:
        print("[C1] no run found; skipping analysis.")
        return {"status": "no data"}

    # --- matched budget: this is the whole point of comparing these arms -----
    arms_budget = [{"name": s.name, "steps": s.steps, "batch_size": args.bs,
                    "forward_passes": 2, "K": s.k_heads} for s in present]
    if len(arms_budget) >= 2:
        matched_budget_check(arms_budget)
        print("Note: head FLOPs are NOT part of the check because they are "
              "negligible here: K=%d linear maps %d->%d is ~%.2f%% of one "
              "ResNet-18 forward at %dx%d." %
              (present[0].k_heads, args.proj_dim, args.head_dim,
               100.0 * present[0].k_heads * args.proj_dim * args.head_dim / 5.55e8,
               args.size, args.size))

    # --- figure -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plot_metric_over_time(axes[0, 0], args.outdir, present, "cka_head_fn",
                          "(a) panel diversity in FUNCTION space\n"
                          "mean pairwise linear CKA on a fixed isotropic input",
                          "CKA (1 = heads identical)")
    plot_metric_over_time(axes[0, 1], args.outdir, present, "cka_head_live",
                          "(b) panel diversity ON REAL DATA\n"
                          "(also reflects the rank of z -- see (d))",
                          "CKA (1 = heads identical)")
    plot_metric_over_time(axes[1, 0], args.outdir, present, "d_cv",
                          "(c) dispersion of the per-head losses\n"
                          "std_k(d_k) / mean_k(d_k)", "coefficient of variation")
    plot_metric_over_time(axes[1, 1], args.outdir, present, "eff_rank_z",
                          "(d) CONTROL: effective rank of z\n"
                          "a drop here would explain (b) without any head collapse",
                          "effective rank")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("C1 -- does a TRAINABLE panel sabotage the worst-case objective?\n"
                 "solid = trainable heads, dashed = frozen heads", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    p1 = os.path.join(args.outdir, "fig_C1_panel_diversity.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    plot_metric_over_time(axes[0], args.outdir, present, "gap_rel",
                          "(a) |max - mean| / mean\n"
                          "-> 0 means the worst-case construction is EMPTY",
                          "relative gap")
    plot_metric_over_time(axes[1], args.outdir, present, "w_entropy",
                          "(b) gradient spread over the panel\n"
                          "H(softmax(beta d)) / log K", "normalised entropy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("C1 -- emptiness of the max, and gradient starvation", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    p2 = os.path.join(args.outdir, "fig_C1_gap_and_gradient.png")
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- verdicts -----------------------------------------------------------
    per_arm: Dict[str, dict] = {}
    for spec in specs:
        summ = load_summaries(args.outdir, spec.name, seeds)
        if not summ:
            continue
        d_cka = [s["final_cka_head_fn"] - s["init_cka_head_fn"] for s in summ]
        cv_ratio = [s["final_d_cv"] / max(abs(s["init_d_cv"]), 1e-12) for s in summ]
        gap_ratio = [s["final_gap_rel"] / max(abs(s["init_gap_rel"]), 1e-12) for s in summ]
        per_arm[spec.name] = {
            "n_seeds": len(summ),
            "delta_cka_fn": mean_std(d_cka)[:2],
            "final_cka_fn": mean_std([s["final_cka_head_fn"] for s in summ])[:2],
            "final_cka_live": mean_std([s["final_cka_head_live"] for s in summ])[:2],
            "cv_ratio": mean_std(cv_ratio)[:2],
            "final_d_cv": mean_std([s["final_d_cv"] for s in summ])[:2],
            "gap_ratio": mean_std(gap_ratio)[:2],
            "final_gap_rel": mean_std([s["final_gap_rel"] for s in summ])[:2],
            "final_eff_rank_z": mean_std([s["final_eff_rank_z"] for s in summ])[:2],
        }

    # P1: trainable heads under max / LSE(beta >= 5) collapse in function space
    p1_arms = ["c1_max_train", "c1_lse20_train", "c1_lse5_train"]
    p1_detail, p1_votes = {}, []
    for a in p1_arms:
        if a not in per_arm:
            continue
        dm, ds = per_arm[a]["delta_cka_fn"]
        cvm, cvs = per_arm[a]["cv_ratio"]
        v_cka = verdict_greater(dm, ds, CKA_COLLAPSE_DELTA, CKA_NULL_DELTA)
        v_cv = verdict_smaller(cvm, cvs, CV_COLLAPSE_RATIO, 1.0)
        joint = (VERDICT_CONFIRMED if (v_cka == VERDICT_CONFIRMED and v_cv == VERDICT_CONFIRMED)
                 else VERDICT_REFUTED if (v_cka == VERDICT_REFUTED and v_cv != VERDICT_CONFIRMED)
                 else VERDICT_INCONCLUSIVE)
        p1_detail[a] = {"delta_cka_fn": [dm, ds], "cv_ratio": [cvm, cvs],
                        "verdict_cka": v_cka, "verdict_cv": v_cv, "verdict": joint}
        p1_votes.append(joint)

    # P2: the max becomes the mean (relative gap vanishes)
    p2_detail, p2_votes = {}, []
    for a in p1_arms:
        if a not in per_arm:
            continue
        gm, gs = per_arm[a]["gap_ratio"]
        v = verdict_smaller(gm, gs, GAP_EMPTY_RATIO, GAP_ALIVE_RATIO)
        p2_detail[a] = {"gap_ratio": [gm, gs], "verdict": v}
        p2_votes.append(v)

    # P3: frozen control -- same aggregation, does freezing preserve dispersion?
    p3_detail, p3_votes = {}, []
    for tag in ("max", "lse20", "lse5", "lse1", "mean"):
        a_t, a_f = f"c1_{tag}_train", f"c1_{tag}_frozen"
        if a_t not in per_arm or a_f not in per_arm:
            continue
        st = load_summaries(args.outdir, a_t, seeds)
        sf = load_summaries(args.outdir, a_f, seeds)
        n = min(len(st), len(sf))
        diff_cv = [sf[i]["final_d_cv"] - st[i]["final_d_cv"] for i in range(n)]
        diff_gap = [sf[i]["final_gap_rel"] - st[i]["final_gap_rel"] for i in range(n)]
        m_cv, s_cv, _ = mean_std(diff_cv)
        m_gap, s_gap, _ = mean_std(diff_gap)
        v = verdict_greater(m_cv, s_cv, 0.0)
        p3_detail[tag] = {"d_cv_frozen_minus_trainable": [m_cv, s_cv],
                          "gap_rel_frozen_minus_trainable": [m_gap, s_gap],
                          "verdict": v}
        if tag in ("max", "lse20", "lse5"):
            p3_votes.append(v)

    def majority(votes: List[str]) -> str:
        if not votes:
            return VERDICT_INCONCLUSIVE
        if all(v == VERDICT_CONFIRMED for v in votes):
            return VERDICT_CONFIRMED
        if all(v == VERDICT_REFUTED for v in votes):
            return VERDICT_REFUTED
        return VERDICT_INCONCLUSIVE

    return {
        "figures": [p1, p2],
        "per_arm": per_arm,
        "P1_trainable_panel_collapses": {
            "statement": "trainable heads under max/LSE(beta>=5): function-space "
                         "pairwise CKA rises by >= %.2f AND d_cv falls to <= %.2f "
                         "of its initial value" % (CKA_COLLAPSE_DELTA, CV_COLLAPSE_RATIO),
            "falsified_if": "CKA rise < %.2f and d_cv does not fall" % CKA_NULL_DELTA,
            "per_arm": p1_detail,
            "verdict": majority(p1_votes),
        },
        "P2_worst_case_becomes_empty": {
            "statement": "final gap_rel <= %.2f x initial gap_rel" % GAP_EMPTY_RATIO,
            "falsified_if": "final gap_rel > %.2f x initial" % GAP_ALIVE_RATIO,
            "per_arm": p2_detail,
            "verdict": majority(p2_votes),
        },
        "P3_frozen_control": {
            "statement": "at equal aggregation, frozen heads end with a larger "
                         "d_cv than trainable heads",
            "falsified_if": "frozen heads lose dispersion as fast as trainable ones "
                            "(then the loss of dispersion comes from the "
                            "representation, not from the heads)",
            "per_aggregation": p3_detail,
            "verdict": majority(p3_votes),
        },
    }


def _battery_frame(args, arms: Sequence[str], seeds: Sequence[int]):
    """{arm: {'worst': (m,s), 'mean': (m,s), 'per_task': {t: (m,s)}}}"""
    out: Dict[str, dict] = {}
    for a in arms:
        summ = [s for s in load_summaries(args.outdir, a, seeds) if "battery" in s]
        if not summ:
            continue
        tasks = sorted({t for s in summ for t in s["battery"]["per_task"]})
        out[a] = {
            "n_seeds": len(summ),
            "worst": mean_std([s["battery"]["worst"] for s in summ])[:2],
            "mean": mean_std([s["battery"]["mean"] for s in summ])[:2],
            "per_task": {t: mean_std([s["battery"]["per_task"].get(t, float("nan"))
                                      for s in summ])[:2] for t in tasks},
            "raw_worst": [s["battery"]["worst"] for s in summ],
        }
    return out


def analyse_c2(args, reg: Dict[str, ArmSpec], seeds: Sequence[int]) -> dict:
    """Worst-case transfer.  Reports the full per-task distribution."""
    arms = [a for a in C2_ARMS if a in reg]
    bat = _battery_frame(args, arms, seeds)
    if not bat:
        print("[C2] no battery result found; skipping analysis.")
        return {"status": "no data"}

    budget = [{"name": a, "steps": reg[a].steps, "batch_size": args.bs,
               "forward_passes": 2, "K": reg[a].k_heads} for a in bat]
    if len(budget) >= 2:
        matched_budget_check(budget)

    tasks = sorted({t for v in bat.values() for t in v["per_task"]})
    names = list(bat)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5),
                             gridspec_kw={"width_ratios": [1, 2]})

    x = np.arange(len(names))
    worst_m = [bat[n]["worst"][0] for n in names]
    worst_s = [bat[n]["worst"][1] for n in names]
    mean_m = [bat[n]["mean"][0] for n in names]
    mean_s = [bat[n]["mean"][1] for n in names]
    axes[0].bar(x - 0.2, worst_m, 0.4, yerr=worst_s, capsize=3, color="0.25",
                label="WORST task")
    axes[0].bar(x + 0.2, mean_m, 0.4, yerr=mean_s, capsize=3, color="0.75",
                edgecolor="0.2", hatch="//", label="mean over tasks")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([n.replace("L_", "") for n in names], rotation=30,
                            ha="right", fontsize=8)
    axes[0].set_ylabel("linear-probe accuracy")
    axes[0].set_title("(a) worst task vs mean task\n(the worst is the one that counts)",
                      fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].grid(True, axis="y", alpha=0.3, linewidth=0.5)

    width = 0.8 / max(1, len(names))
    xt = np.arange(len(tasks))
    grays = np.linspace(0.15, 0.8, len(names))
    for i, n in enumerate(names):
        vals = [bat[n]["per_task"].get(t, (float("nan"), 0.0))[0] for t in tasks]
        errs = [bat[n]["per_task"].get(t, (float("nan"), 0.0))[1] for t in tasks]
        axes[1].bar(xt + i * width - 0.4 + width / 2, vals, width, yerr=errs,
                    capsize=2, color=str(grays[i]), edgecolor="black",
                    linewidth=0.4, label=n.replace("L_", ""))
    axes[1].set_xticks(xt)
    axes[1].set_xticklabels(tasks, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("linear-probe accuracy")
    axes[1].set_title("(b) full distribution over the heterogeneous battery\n"
                      "32x32 probes: comparable across arms, NOT absolute numbers",
                      fontsize=10)
    axes[1].legend(fontsize=7, frameon=False, ncol=2)
    axes[1].grid(True, axis="y", alpha=0.3, linewidth=0.5)

    fig.suptitle("C2 -- does the panel buy worst-case transfer? "
                 "(no theorem says it should)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(args.outdir, "fig_C2_worst_case_transfer.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # P4: frozen LSE panel vs single head, on the WORST task
    detail = {}
    votes = []
    ref = "L_single"
    for a in ("L_panel_lse5_frozen", "L_panel_mean_frozen", "L_panel_lse5_train",
              "L_panel_mean_train"):
        if a not in bat or ref not in bat:
            continue
        n = min(len(bat[a]["raw_worst"]), len(bat[ref]["raw_worst"]))
        diff = [bat[a]["raw_worst"][i] - bat[ref]["raw_worst"][i] for i in range(n)]
        m, s, _ = mean_std(diff)
        v = verdict_greater(m, s, 0.0, 0.0)
        detail[f"{a}_minus_{ref}"] = {"worst_task_delta": [m, s], "verdict": v}
        if a == "L_panel_lse5_frozen":
            votes.append(v)

    return {
        "figures": [p],
        "battery": bat,
        "P4_panel_helps_worst_task": {
            "statement": "worst-task accuracy of the FROZEN LSE(5) panel exceeds "
                         "that of the single-head baseline by more than one seed "
                         "standard deviation",
            "falsified_if": "the difference is <= 0 across seeds",
            "IMPORTANT": "no theorem connects a worst case over projection heads "
                         "to a worst case over downstream tasks. A null here "
                         "refutes only the USEFULNESS of the panel at CIFAR "
                         "scale; it says nothing about the diversity collapse "
                         "measured in C1, and C1 says nothing about this.",
            "detail": detail,
            "verdict": votes[0] if votes else VERDICT_INCONCLUSIVE,
        },
    }


def analyse_c3(args, reg: Dict[str, ArmSpec], seeds: Sequence[int]) -> dict:
    """Beta ablation: interpolation between mean and max, and the compromise."""
    limits = check_lse_limits(verbose=False)
    arms = [a for a in C3_ARMS if a in reg]
    bat = _battery_frame(args, arms, seeds)
    result: Dict[str, object] = {
        "P5_beta_limits": {
            "statement": "L_beta -> mean as beta -> 0 and -> max as beta -> inf "
                         "(mean-normalised LSE; the unnormalised form diverges "
                         "as beta -> 0, see aggregate_panel)",
            "numeric_check": limits,
            "verdict": (VERDICT_CONFIRMED
                        if (limits["err_to_mean"] < 1e-4 and limits["err_to_max"] < 1e-4
                            and limits["monotone"]) else VERDICT_REFUTED),
        }
    }
    if not bat:
        print("[C3] no battery result found; beta trade-off skipped.")
        result["beta_tradeoff"] = {"status": "no data"}
        return result

    budget = [{"name": a, "steps": reg[a].steps, "batch_size": args.bs,
               "forward_passes": 2, "K": reg[a].k_heads} for a in bat]
    if len(budget) >= 2:
        matched_budget_check(budget)

    order = [a for a in C3_ARMS if a in bat]
    betas = [reg[a].beta_axis() for a in order]
    labels = ["0\n(mean)" if b == 0 else ("inf\n(max)" if math.isinf(b) else f"{b:g}")
              for b in betas]
    xs = np.arange(len(order))
    wm = [bat[a]["worst"][0] for a in order]
    ws = [bat[a]["worst"][1] for a in order]
    mm = [bat[a]["mean"][0] for a in order]
    ms = [bat[a]["mean"][1] for a in order]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(xs, wm, yerr=ws, color="black", marker="o", linestyle="-",
                capsize=3, label="WORST task")
    ax.errorbar(xs, mm, yerr=ms, color="0.55", marker="s", linestyle="--",
                capsize=3, label="mean over tasks")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("aggregation temperature beta  (frozen heads, K=%d)" % reg[order[0]].k_heads)
    ax.set_ylabel("linear-probe accuracy")
    ax.set_title("C3 -- beta interpolates mean -> max.\n"
                 "Where is the worst-task / mean-task compromise?", fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    p = os.path.join(args.outdir, "fig_C3_beta_ablation.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)

    best_worst = order[int(np.nanargmax(wm))] if any(not math.isnan(v) for v in wm) else None
    best_mean = order[int(np.nanargmax(mm))] if any(not math.isnan(v) for v in mm) else None
    spread = float(np.nanmax(wm) - np.nanmin(wm)) if len(wm) > 1 else float("nan")
    seed_noise = float(np.nanmean(ws)) if ws else float("nan")
    result["figures"] = [p]
    result["beta_tradeoff"] = {
        "per_beta": {a: {"beta": reg[a].beta_axis(), "worst": bat[a]["worst"],
                         "mean": bat[a]["mean"]} for a in order},
        "best_worst_task_arm": best_worst,
        "best_mean_task_arm": best_mean,
        "worst_task_spread_over_beta": spread,
        "mean_seed_std": seed_noise,
        "verdict": (VERDICT_CONFIRMED if (not math.isnan(spread)
                                          and not math.isnan(seed_noise)
                                          and spread > 2 * seed_noise)
                    else VERDICT_INCONCLUSIVE),
        "reading": "if the spread over beta does not exceed twice the seed "
                   "standard deviation, beta simply does not matter at this "
                   "scale and no compromise point can be claimed.",
    }
    return result


# =============================================================================
# 9. CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="T3 -- smoothed worst case over a panel of heads, and the "
                    "diversity collapse of a trainable panel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # what to run
    p.add_argument("--exp", choices=["c1", "c2", "c3", "all"], default="c1",
                   help="c1 = panel collapse (decisive, cheap); c2 = worst-case "
                        "transfer; c3 = beta ablation (shares c2's runs)")
    p.add_argument("--all", action="store_true", help="same as --exp all")
    p.add_argument("--arm", action="append", default=None,
                   help="run only this arm (repeatable); overrides --exp")
    p.add_argument("--analyse-only", action="store_true",
                   help="no training, only figures and verdicts from existing runs")
    p.add_argument("--list-arms", action="store_true", help="print the registry and exit")

    # seeds
    p.add_argument("--seed", type=int, default=None,
                   help="run a single seed instead of --seeds")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="a comparison on one seed decides nothing: 3 by default")

    # budget (defaults = the real A100 configuration)
    p.add_argument("--steps", type=int, default=None,
                   help="override BOTH --c1-steps and --long-steps")
    p.add_argument("--c1-steps", type=int, default=8000)
    p.add_argument("--long-steps", type=int, default=15000)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-6)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--clip", type=float, default=3.0, help="grad-norm clip, 0 to disable")
    p.add_argument("--ema-tau", type=float, default=0.996)

    # model
    p.add_argument("--arch", default="resnet18")
    p.add_argument("--encoder-variant", default="cifar", choices=["cifar", "stl", "imagenet"])
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--proj-layers", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--k-heads", type=int, default=8)
    p.add_argument("--disjoint", action="store_true",
                   help="each head reads a distinct block of z. OFF by default: "
                        "see section 3(C) of the module docstring -- disjoint "
                        "inputs are a SECOND anti-redundancy device and would "
                        "confound the freezing variable of C1.")
    p.add_argument("--var-w", type=float, default=25.0)
    p.add_argument("--cov-w", type=float, default=1.0)

    # data
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--aug", choices=["weak", "strong"], default="strong")
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--tasks", nargs="+", default=TRANSFER_BATTERY)

    # diagnostics / eval
    p.add_argument("--diag-every", type=int, default=200)
    p.add_argument("--diag-n", type=int, default=512, help="fixed images for the live CKA")
    p.add_argument("--n-ref", type=int, default=1024, help="rows of the isotropic reference")
    p.add_argument("--probe-epochs", type=int, default=40)
    p.add_argument("--no-eval", action="store_true", help="skip all probes")

    # plumbing
    p.add_argument("--outdir", default=os.path.join(_ROOT, "results", "C"))
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="wipe existing CSV/checkpoints and start clean")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no-amp", action="store_true", help="disable bf16 autocast on CUDA")

    # smoke
    p.add_argument("--smoke", action="store_true",
                   help="tiny synthetic run, no download, < 3 min on CPU")
    p.add_argument("--smoke-n", type=int, default=96)
    return p


def apply_smoke(args) -> None:
    """Shrink everything.  Synthetic data: --smoke never touches the network."""
    args.cpu = True
    args.workers = 0
    args.bs = 8
    args.c1_steps = 6
    args.long_steps = 6
    args.warmup = 2
    args.diag_every = 2
    args.ckpt_every = 4
    args.diag_n = 16
    args.n_ref = 64
    args.proj_dim = 32
    args.hidden = 64
    args.head_dim = 8
    args.k_heads = 4
    args.probe_epochs = 3
    args.seeds = [0, 1] if args.seed is None else [args.seed]
    args.size = 32


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        apply_smoke(args)
    if args.steps is not None:
        args.c1_steps = args.steps
        args.long_steps = args.steps
    seeds = [args.seed] if args.seed is not None else list(args.seeds)
    os.makedirs(args.outdir, exist_ok=True)

    reg = build_registry(args)
    if args.list_arms:
        for name, spec in reg.items():
            print(f"{name:26s} {spec.label():34s} steps={spec.steps} "
                  f"battery={spec.run_battery}")
        return 0

    print("=" * 78)
    print("xp_C_panel -- T3: worst case over a panel of projection heads")
    print("=" * 78)
    print("SCALE CAVEAT: ResNet-18 / CIFAR-10 / 32x32 probes. C1 is a claim about "
          "optimisation dynamics and is decidable here; C2-C3 are transfer claims "
          "and are only COMPARATIVE at this scale, never absolute.")
    print("NO THEOREM links a worst case over heads to a worst case over tasks. "
          "C1 and C2 must be read as independent results.")
    check_lse_limits(verbose=True)

    device = get_device(prefer_cpu=args.cpu)
    print(f"device: {device} | seeds: {seeds} | outdir: {args.outdir}")

    if not args.analyse_only:
        specs = select_arms(args, reg)
        print(f"training {len(specs)} arm(s) x {len(seeds)} seed(s) = "
              f"{len(specs) * len(seeds)} runs")
        for spec in specs:
            for seed in seeds:
                print(f"[run] {spec.name} seed={seed} :: {spec.label()} "
                      f"steps={spec.steps}")
                train_arm(spec, seed, args, device)

    summary: Dict[str, object] = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "thesis": "T3 -- the agreement objective should be a smoothed worst case "
                  "over K projection heads; a TRAINABLE panel self-sabotages by "
                  "collapsing onto a single function, which makes max == mean.",
        "scale_caveat": "ResNet-18 on CIFAR-10 at 32x32; transfer probes are run "
                        "at 32x32 and are comparative only. Not ImageNet.",
        "honest_limit": "No theorem relates a worst case over projection heads to "
                        "a worst case over downstream tasks. C1 (dynamics) and C2 "
                        "(transfer) are independent; neither rescues the other.",
        "thresholds": {"CKA_COLLAPSE_DELTA": CKA_COLLAPSE_DELTA,
                       "CKA_NULL_DELTA": CKA_NULL_DELTA,
                       "CV_COLLAPSE_RATIO": CV_COLLAPSE_RATIO,
                       "GAP_EMPTY_RATIO": GAP_EMPTY_RATIO,
                       "GAP_ALIVE_RATIO": GAP_ALIVE_RATIO},
        "config": {k: v for k, v in vars(args).items() if k != "arm"},
        "seeds": seeds,
    }
    print("\n--- analysis -------------------------------------------------")
    summary["C1_panel_collapse"] = analyse_c1(args, reg, seeds)
    summary["C2_worst_case_transfer"] = analyse_c2(args, reg, seeds)
    summary["C3_beta_ablation"] = analyse_c3(args, reg, seeds)

    path = os.path.join(args.outdir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nsummary written to {path}")

    print("\nVERDICTS")
    for block, keys in (("C1_panel_collapse", ["P1_trainable_panel_collapses",
                                               "P2_worst_case_becomes_empty",
                                               "P3_frozen_control"]),
                        ("C2_worst_case_transfer", ["P4_panel_helps_worst_task"]),
                        ("C3_beta_ablation", ["P5_beta_limits", "beta_tradeoff"])):
        blk = summary.get(block, {})
        for k in keys:
            v = blk.get(k, {})
            if isinstance(v, dict) and "verdict" in v:
                print(f"  {k:34s} -> {v['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
