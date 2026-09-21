"""
xp_B_relational_vs_prototype.py -- thesis T1, main experiment.

======================================================================
1. THESIS UNDER TEST (T1)
======================================================================
A relational target  G = A A^T  (A = L2-normalised embeddings) is invariant
under every rotation Q of the latent space:  (AQ)(AQ)^T = A A^T.  A softmax
prototype head  softmax(<a, c_k>/tau)  is NOT: the prototypes c_k pin a
distinguished basis.  "Legislating over the object" therefore means "fixing a
frame".

The claim that follows, and the ONLY one this script tries to decide, is:

    a prototype head carries three extra knobs -- K, centering/Sinkhorn,
    sharpening -- that simply do not exist on the relational side, and it
    presupposes a near-uniform marginal over clusters, an assumption that the
    data violates under a long tail.

The claim is NOT "relational is more accurate" and NOT "relational is robust
to imbalance".  The careful statement is "no uniformity prior", and the code
below is written so that a reader can check we did not quietly upgrade it.

======================================================================
2. PRE-REGISTERED PREDICTIONS, THEIR NUMBERS, AND WHAT FALSIFIES THEM
======================================================================
Thresholds are fixed HERE, before any run, and are re-read from this docstring
by nobody: they are duplicated in PREREG below so that summary.json carries
them verbatim.  Changing them after seeing results is fraud; the git history is
the audit trail.

P2  O(d) INVARIANCE (analytic, but measured on trained embeddings).
    Take the final student/teacher embeddings of a real batch, draw a random
    orthogonal Q, and recompute both losses on (sQ, aQ).
    Predicted: gram_loss relative change < 1e-4 (floating point only);
               prototype_loss relative change > 1e-3.
    FALSIFIED IF: the prototype loss is also invariant (it would mean the head
    is not actually frame-fixing), or the Gram loss moves.
    This is the propositional content of T1 and it is cheap; everything else is
    a consequence that the world is free to refuse.

P0  IMPOSED UNIFORMITY (the mechanism, primary).
    Measure the cluster marginal of the DINO teacher, marg_k = mean_n p_t(k|n),
    and its entropy normalised by log K.  Measure in parallel the TRUE label
    entropy of the pre-training pool, normalised by log C.
    Predicted: going from gamma=1.0 to gamma=0.01 the true label entropy falls
    substantially while the prototype marginal entropy barely moves -- the head
    keeps asserting a uniform marginal that the data no longer has.
    CONFIRMED IF  |dH_proto| < 0.25 * |dH_true|  AND  H_proto(0.01) > 0.90.
    FALSIFIED IF  |dH_proto| > 0.75 * |dH_true|  (the marginal tracks the data;
    then the "uniformity prior" story is simply wrong).
    Otherwise: non concluante.

P1  HYPERPARAMETER SENSITIVITY (primary).
    Sweep the prototype arm over K in {256,1024,4096} x tau_teacher in
    {0.02,0.04,0.07} -- nine configurations.
    IMPORTANT AMENDMENT (see section 4): the relational arm is NOT knob-free in
    an absolute sense, only free of THESE knobs.  It has var_w and cov_w.  So we
    sweep it over a grid of the SAME CARDINALITY, var_w in {10,25,50} x cov_w in
    {0.5,1,2}, and compare spreads.  A one-sided sweep would be a rigged
    comparison and a reviewer would say so immediately.
    Statistic: sensitivity_ratio = std over configs of the seed-mean accuracy,
    divided by the mean over configs of the across-seed std.  It answers "is the
    configuration spread bigger than the noise I would get by re-seeding?".
    CONFIRMED IF  ratio_proto >= 2.0  AND  ratio_proto >= 2.0 * ratio_relational.
    FALSIFIED IF  ratio_relational >= ratio_proto.
    Otherwise: non concluante.

P3  BATCH COUPLING (self-critical control -- omitting it would void P1).
    The Gram target is a BATCH statistic: with B samples the per-sample gradient
    contribution scales like 1/B and the positive-to-negative ratio is about
    1/B.  If the relational arm is strongly B-dependent then B *is* its hidden
    knob and P1 must be reported as conditional.
    Sweep B in {128,256,512} for the relational arm at EQUAL SAMPLE BUDGET
    (steps * B held constant, learning rate linearly scaled), 3 seeds.
    Statistic: same sensitivity_ratio, across B instead of across configs.
    "coupling detected" if ratio_B >= 2.0.  Either outcome is reported; a
    detected coupling annotates P1 rather than killing it, because the P1 sweep
    holds B fixed by construction.
    Honest caveat: a null result here is conditional on the linear lr-scaling
    rule; absence of coupling under one rule is not absence of coupling.

P4  THE DINO KNOBS ARE LOAD-BEARING (sanity / strawman check).
    Removing centering or sharpening must hurt.  If it does not, our DINO
    implementation is not DINO and P0/P1 mean nothing.
    CONFIRMED IF each ablation loses >= 2.0 accuracy points w.r.t. proto_full,
    or collapses (effective rank of the projector output < 2.0).

P5  LONG-TAIL ACCURACY (exploratory, deliberately demoted).
    Degradation of tail_acc from gamma=1.0 to gamma=0.01, per arm.
    Reported with mean +/- std over seeds and NO verdict of the "robust" kind.
    Read section 1 again before quoting this number.

======================================================================
3. WHAT THIS SCALE CAN AND CANNOT DECIDE
======================================================================
This is CIFAR-100 at 32x32 with a ResNet-18, not ImageNet.  Consequences a
sceptical reader should hold us to:
  * K = 4096 prototypes on a 10.9k-image long-tail pool is 2.7 prototypes per
    image.  The K axis is therefore partly a "too many clusters for the data"
    axis; that is a real property of the knob, but its ImageNet behaviour may
    differ.
  * absolute accuracies here are far below published CIFAR SSL numbers because
    the budget is a few tens of epochs, not 800.  Only the DIFFERENCES between
    budget-matched arms are claimed to mean anything.
  * a negative result at this scale does not refute the thesis at scale; a
    positive result at this scale does not establish it at scale.  P2 is the
    only prediction here that is scale-free, and it is scale-free because it is
    an identity, not an experiment.

======================================================================
4. AMENDMENTS TO THE REQUESTED PLAN (and why)
======================================================================
(a) The relational arm is swept too.  Asked: "for the relational arm there is
    nothing to sweep -- that is precisely the result."  That is true of K /
    centering / sharpening and false of var_w / cov_w.  Reporting a spread for
    one family and no spread for the other because we only swept one of them
    would be circular.  We sweep both over 3x3 grids and compare.
(b) The linear probe is trained on the LONG-TAIL subset, not on the balanced
    train split, and evaluated on the balanced test split.  Probing on balanced
    labels would hand back at evaluation time the very balance the pre-training
    lacked, and would make head/tail accuracy nearly meaningless.  The balanced
    probe is still computed once per run as a secondary diagnostic
    (`probe_acc_balanced`), clearly separated.
(c) The B sweep cannot go through `matched_budget_check`: that function
    (correctly) refuses arms with different batch sizes.  The B study is
    therefore matched on SAMPLES SEEN (steps * B constant) and on encoder
    forward passes, and we call `matched_budget_check` on it anyway, inside a
    try/except, to print the refusal -- the control should be visible, not
    implied.
(d) All five arms share one encoder, one projector, one EMA target, one
    optimiser and one schedule; only the loss differs.  In particular the
    InfoNCE control is run asymmetrically against the EMA target (MoCo-like
    without a queue), not as symmetric SimCLR, so that it differs from the other
    arms in exactly one thing.  `--infonce-symmetric` restores the textbook
    version for anyone who objects.
(e) No per-arm learning-rate tuning.  Tuning one arm and not the others is the
    classic way to manufacture a result.  AdamW is used precisely because it is
    approximately invariant to a global rescaling of the loss, which the arms do
    not share (Gram MSE ~ 1e-1, cross-entropy ~ log K).  The relative weighting
    INSIDE an arm is a genuine hyperparameter, which is exactly why (a) exists.
(f) Epochs differ across gamma because compute is matched and the pool shrinks.
    That is the standard long-tail SSL protocol; matching epochs instead would
    confound imbalance with compute.

======================================================================
5. A100 BUDGET (single A100 40GB, Colab Pro, bf16)
======================================================================
Default configuration: ResNet-18 (CIFAR stem), 32x32, B=256, 10 000 steps
(= 5.1M samples = ~102 balanced CIFAR-100 epochs), 2 encoder forwards/step.
Measured proxy: ~0.40 h per run including the final probes.

    study "main"   5 arms x 3 gammas x 3 seeds                 = 45 runs ~ 18 h
    study "sweep"  9 configs x 2 families x 3 seeds (gamma=.01)= 54 runs ~ 22 h
    study "batch"  3 B x 2 gammas x 3 seeds (equal samples)    = 18 runs ~  7 h
    ------------------------------------------------------------------------
    TOTAL, default settings                                    ~ 47 A100-hours

Spread over several sessions; every run is independently resumable and finished
runs are skipped, so `--resume` after a Colab kill costs nothing.  `--steps 5000`
halves the bill and is a reasonable first pass; the sweep is the expensive part
and `--study main` alone is ~18 h.

======================================================================
6. USAGE
======================================================================
    python xp/xp_B_relational_vs_prototype.py --smoke          # < 3 min, CPU
    python xp/xp_B_relational_vs_prototype.py --all --study all
    python xp/xp_B_relational_vs_prototype.py --arm relational --arm proto_full
    python xp/xp_B_relational_vs_prototype.py --aggregate-only # figures + verdicts

Outputs under --outdir:
    runs/                 per-run training CSV, checkpoints, per-run summaries
    results_long.csv      one row per finished run (long format)
    figures/*.png         four figures, greyscale-legible
    summary.json          verdicts, numbers, thresholds, limitations
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- in-house dependency: the shared harness, and nothing else --------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.harness import (  # noqa: E402
    AugCfg,
    DATASET_NUM_CLASSES,
    EMATeacher,
    Run,
    effective_rank,
    get_device,
    get_eval_datasets,
    get_ssl_dataset,
    gram_loss,
    info_nce,
    knn_probe,
    linear_probe,
    longtail_indices,
    make_encoder,
    make_eval_transform,
    make_mlp,
    make_two_view_transform,
    matched_budget_check,
    prototype_loss,
    set_seed,
    uniformity_alignment,
    vicreg_reg,
)
# `_rng_island` is private in the harness but it is the ONLY correct way to run
# an evaluation block without shifting the training RNG stream (harness note 6).
# This script builds one DataLoader of its own (`diagnostics`), and torch draws a
# base seed from the GLOBAL generator every time a DataLoader iterator is
# created -- even with shuffle=False.  Without the island, "how often do I
# evaluate" silently becomes part of the seed, which is a silent-fatal bug.
from lib.harness import _rng_island  # noqa: E402


# ===========================================================================
# pre-registered constants
# ===========================================================================

PREREG = {
    "P0_imposed_uniformity": {
        "confirm": "abs(dH_proto) < 0.25 * abs(dH_true) and H_proto(gamma=0.01) > 0.90",
        "falsify": "abs(dH_proto) > 0.75 * abs(dH_true)",
    },
    "P1_hyperparameter_sensitivity": {
        "confirm": "ratio_proto >= 2.0 and ratio_proto >= 2.0 * ratio_relational",
        "falsify": "ratio_relational >= ratio_proto",
        "statistic": "std_over_configs(seed_mean_acc) / mean_over_configs(seed_std_acc)",
    },
    "P2_od_invariance": {
        "confirm": "max gram relative change < 1e-4 and median prototype relative change > 1e-3 "
                   "and max prototype-rotated relative change < 1e-4",
        "falsify": "prototype loss also invariant, or gram loss not invariant, or the "
                   "prototype loss STAYS non-invariant when the prototypes are rotated too "
                   "(that would mean the culprit is the softmax, not the frame)",
    },
    "P3_batch_coupling": {
        "detected": "ratio_over_B >= 2.0",
        "note": "conditional on the linear lr-scaling rule; a null result is not proof of absence",
    },
    "P4_dino_knobs_load_bearing": {
        "confirm": "each ablation loses >= 2.0 accuracy points, or effective_rank < 2.0",
        "falsify": "ablations match or beat proto_full",
    },
    "P5_longtail_accuracy": {
        "status": "exploratory; reported without a robustness verdict on purpose",
    },
    # ---------------------------------------------------------------------
    # PRE-REGISTERED 2026-08-22, BEFORE THIS SCRIPT WAS EVER RUN, and after
    # experiment A produced a result nobody predicted: the gram+VICReg arm
    # reached 4.8x the effective rank of gram-alone and scored 2.3 accuracy
    # points LOWER on kNN (36.4% -> 34.1%, 4.5 pooled std over 3 seeds).
    #
    # A read on the same data that produced it is not evidence, it is a
    # hypothesis.  The threshold below is therefore fixed HERE, on data that
    # does not exist yet, and is not to be touched once B has run.  If it is
    # ever edited after a run, that edit invalidates the prediction -- say so
    # rather than adjusting it.
    # ---------------------------------------------------------------------
    # P6b.  P6 above correlates `eff_rank`, measured on the PROJECTOR output
    # (StudentNet.forward = projector(backbone(x))), with `probe_acc`, measured
    # on the BACKBONE (BackboneOnly).  Two different spaces separated by a
    # trained MLP.  A2 measured on 2026-08-26 what that gap can do: from `gram`
    # to `gram_vicreg` the projector rank is multiplied by 4.83 while the
    # backbone rank moves by 0.04 sigma.  A near-zero correlation would then
    # CONFIRM P6 for a trivial reason, and its "why_it_matters" conclusion would
    # rest on an artefact.
    # P6 is NOT edited: its statement and its thresholds stay exactly as
    # registered on 2026-08-22.  P6b repeats it verbatim in the space the probe
    # actually reads.  Registered 2026-08-26, still before B's first run.
    # If P6 and P6b disagree, that disagreement IS the result.
    "P6b_backbone_rank_does_not_predict_transfer": {
        "statement": "across B's arms at fixed gamma and fixed batch size, the effective "
                     "rank of the BACKBONE features does not predict linear-probe accuracy",
        "confirm": "abs(spearman(eff_rank_h, probe_acc)) < 0.4 over >= 4 arms, or the "
                   "correlation is NEGATIVE at any gamma",
        "falsify": "spearman(eff_rank_h, probe_acc) > 0.7 at every gamma",
        "why_it_matters": "same link as P6, but measured in the space the probe reads. "
                          "P6 alone cannot break the anti-collapse-to-transfer link, "
                          "because a null correlation between two different spaces is "
                          "expected whatever the truth about rank and transfer.",
        "scale_caveat": "same as P6. Additionally: eff_rank is UNCENTRED and h is "
                        "post-ReLU, hence non-negative; eff_rank_h_centred is logged "
                        "alongside and must be inspected before quoting either.",
        "registered": "2026-08-26, before the first run of this script",
    },
    "P6_rank_does_not_predict_transfer": {
        "statement": "across B's arms at fixed gamma and fixed batch size, the effective "
                     "rank of the embedding does not predict linear-probe accuracy",
        "confirm": "abs(spearman(eff_rank, probe_acc)) < 0.4 over >= 4 arms, or the "
                   "correlation is NEGATIVE at any gamma",
        "falsify": "spearman(eff_rank, probe_acc) > 0.7 at every gamma",
        "why_it_matters": "the legality term is justified in this project as anti-collapse, "
                          "and anti-collapse is justified as a route to better transfer. If "
                          "rank and transfer are uncorrelated, that second link is broken and "
                          "the term needs a different justification, or none.",
        "scale_caveat": "CIFAR-100-LT, ResNet-18, one linear probe. A null correlation here "
                        "does not generalise to ImageNet or to fine-tuning.",
        "registered": "2026-08-22, before the first run of this script",
    },
}

# Arm registry.  `family` groups arms that share a loss family; it is the axis
# along which sensitivity is compared in P1.
ARMS: Dict[str, dict] = {
    "relational": dict(
        family="relational", uses_protos=False, uses_vicreg=True,
        use_center=None, use_sharpen=None,
    ),
    "proto_full": dict(
        family="prototype", uses_protos=True, uses_vicreg=False,
        use_center=True, use_sharpen=True,
    ),
    "proto_nocenter": dict(
        family="prototype", uses_protos=True, uses_vicreg=False,
        use_center=False, use_sharpen=True,
    ),
    "proto_nosharpen": dict(
        family="prototype", uses_protos=True, uses_vicreg=False,
        use_center=True, use_sharpen=False,
    ),
    "infonce": dict(
        family="infonce", uses_protos=False, uses_vicreg=True,
        use_center=None, use_sharpen=None,
    ),
}

ARM_ORDER = ["relational", "proto_full", "proto_nocenter", "proto_nosharpen", "infonce"]

# Greyscale-legible plotting style: black lines, distinct markers/linestyles.
ARM_STYLE = {
    "relational":      dict(marker="o", linestyle="-",   color="black"),
    "proto_full":      dict(marker="s", linestyle="--",  color="black"),
    "proto_nocenter":  dict(marker="^", linestyle=":",   color="dimgray"),
    "proto_nosharpen": dict(marker="v", linestyle="-.",  color="dimgray"),
    "infonce":         dict(marker="D", linestyle=(0, (3, 1, 1, 1)), color="gray"),
}

# P1 grids -- same cardinality on both sides (see amendment (a)).
PROTO_GRID_K = [256, 1024, 4096]
PROTO_GRID_TAU_T = [0.02, 0.04, 0.07]
REL_GRID_VAR_W = [10.0, 25.0, 50.0]
REL_GRID_COV_W = [0.5, 1.0, 2.0]

GAMMAS_MAIN = [1.0, 0.1, 0.01]
BATCH_SIZES = [128, 256, 512]

# P3 needs a CONTROL arm: "the relational arm moves with B" is worth nothing
# unless we also know whether the prototype arm moves with B at the same budget.
# Without it, generic batch-size effects and the Gram's 1/B statistics are
# indistinguishable.  (Amendment (g), added at adversarial review, before any
# run -- no result exists yet, so the pre-registration is still open.)
BATCH_ARMS = ["relational", "proto_full"]

RESULT_COLUMNS = [
    "run_name", "study", "arm", "family", "gamma", "seed",
    "bs", "steps", "samples_seen", "K", "tau_t", "tau_s", "var_w", "cov_w", "lr",
    "n_pretrain", "label_entropy_norm",
    "probe_acc", "probe_head_acc", "probe_tail_acc", "probe_acc_balanced",
    "knn_acc", "eff_rank", "uniformity", "alignment",
    "eff_rank_h", "eff_rank_h_centred", "eff_rank_centred",
    "proto_marg_entropy_norm", "proto_used_frac", "proto_kl_uniform",
    "proto_marg_entropy_cap",
    "rot_gram_reldiff", "rot_proto_reldiff", "rot_proto_rotated_reldiff",
    "final_loss", "wall_s",
]


# ===========================================================================
# configuration
# ===========================================================================

@dataclass
class ExpConfig:
    """Everything that is shared by every run of one invocation."""
    # Anchored on the repo root like every other script, so aggregate.py finds it.
    # "./out_xpB" was relative to the CWD: launching from elsewhere silently
    # scattered the results and the aggregator saw nothing.
    outdir: str = os.path.join(_ROOT, "results", "B")
    data_root: str = "./data"
    dataset: str = "cifar100"
    arch: str = "resnet18"
    image_size: int = 32
    proj_hidden: int = 2048
    proj_dim: int = 256
    proj_layers: int = 3

    steps: int = 10000
    bs: int = 256
    lr: float = 1.0e-3           # AdamW base lr at B=256
    weight_decay: float = 1.0e-6
    warmup_frac: float = 0.05

    ema_base: float = 0.996
    ema_final: float = 1.0

    tau_s: float = 0.1           # student temperature (DINO default)
    tau_t: float = 0.04          # teacher temperature (default point of the grid)
    K: int = 1024                # prototypes (default point of the grid)
    proto_freeze_steps: int = 300  # DINO: no prototype gradient for ~1 epoch

    var_w: float = 25.0
    cov_w: float = 1.0
    tau_nce: float = 0.2

    eval_every: int = 2500
    ckpt_every: int = 500
    probe_epochs: int = 100
    n_diag: int = 2048           # deterministic diagnostic pool size

    seeds: Tuple[int, ...] = (0, 1, 2)
    workers: int = -1            # -1 = auto
    prefer_cpu: bool = False
    amp: bool = True
    resume: bool = True
    smoke: bool = False
    synthetic: bool = False
    infonce_symmetric: bool = False
    lr_scaling: bool = True

    def runs_dir(self) -> str:
        return os.path.join(self.outdir, "runs")

    def figures_dir(self) -> str:
        return os.path.join(self.outdir, "figures")

    def results_csv(self) -> str:
        return os.path.join(self.outdir, "results_long.csv")


@dataclass
class RunSpec:
    """One training run.  The name encodes every field, so resume is safe."""
    study: str
    arm: str
    gamma: float
    seed: int
    bs: int
    steps: int
    K: int
    tau_t: float
    var_w: float
    cov_w: float
    lr: float

    @property
    def family(self) -> str:
        return ARMS[self.arm]["family"]

    @property
    def samples_seen(self) -> int:
        return int(self.steps) * int(self.bs)

    def name(self) -> str:
        def f(x: float) -> str:
            return ("%g" % x).replace(".", "p").replace("-", "m")
        return (f"{self.study}__{self.arm}__g{f(self.gamma)}__K{self.K}"
                f"__tt{f(self.tau_t)}__vw{f(self.var_w)}__cw{f(self.cov_w)}"
                f"__B{self.bs}__T{self.steps}__s{self.seed}")


# ===========================================================================
# data providers
# ===========================================================================

class _SyntheticSSLDataset(Dataset):
    """Two-view wrapper with the exact contract of harness.SSLDataset.

    Used only by --smoke/--synthetic so that the pipeline can be validated with
    no network access.  It carries the same dict keys and the same "labels are
    diagnostics only" policy.
    """

    def __init__(self, base: Dataset, two_view, indices: Optional[List[int]],
                 labels: np.ndarray):
        self.base = base
        self.two_view = two_view
        self.indices = None if indices is None else [int(i) for i in indices]
        self._labels = labels

    def __len__(self) -> int:
        return len(self.base) if self.indices is None else len(self.indices)

    def _map(self, i: int) -> int:
        return i if self.indices is None else self.indices[i]

    def __getitem__(self, i: int) -> dict:
        img, label = self.base[self._map(i)]
        v1, v2 = self.two_view(img)
        return {"v1": v1, "v2": v2, "idx": int(i), "label": int(label)}

    @property
    def labels(self) -> np.ndarray:
        return self._labels


class _SyntheticImages(Dataset):
    """Class-conditional coloured-blob images, as PIL, so that the REAL
    augmentation pipeline is exercised end to end."""

    def __init__(self, n_per_class: int, num_classes: int, size: int, seed: int,
                 transform=None):
        from PIL import Image
        rs = np.random.RandomState(seed)
        self.transform = transform
        self.images = []
        self.targets: List[int] = []
        centres = rs.randint(40, 215, size=(num_classes, 3))
        for c in range(num_classes):
            for _ in range(n_per_class):
                base = np.zeros((size, size, 3), dtype=np.float32)
                base += centres[c][None, None, :]
                yy, xx = np.mgrid[0:size, 0:size]
                phase = 2.0 * math.pi * (c + 1) / max(1, num_classes)
                base[..., 0] += 40.0 * np.sin(phase + xx / 3.0)
                base[..., 1] += 40.0 * np.cos(phase + yy / 3.0)
                base += rs.normal(0.0, 12.0, base.shape)
                arr = np.clip(base, 0, 255).astype(np.uint8)
                self.images.append(Image.fromarray(arr))
                self.targets.append(c)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        img = self.images[i]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[i]


class DataProvider:
    """Single place where "which images does an arm see" is decided.

    Contract:
      ssl_dataset(gamma, seed)      -> two-view dataset over the long-tail pool
      probe_datasets(gamma, seed)   -> (lt_train_eval, balanced_train_eval, test_eval)
      num_classes                   -> int

    The long-tail index set is drawn with `longtail_indices(..., seed)` using
    the SAME seed as the run, so (i) pre-training pool and probe pool are
    literally the same images, and (ii) the seed-to-seed spread we report
    includes the variability of the subsampling itself.  That is deliberate: a
    method that only works for one lucky draw of the tail is not a method.
    """

    def __init__(self, cfg: ExpConfig):
        self.cfg = cfg
        self.synthetic = bool(cfg.synthetic)
        self.num_classes = (DATASET_NUM_CLASSES[cfg.dataset] if not self.synthetic
                            else 20)
        self._syn_train = None
        self._syn_test = None
        self._eval_train = None
        self._eval_test = None
        self._eval_labels = None

    # -- augmentation policy shared by every arm ----------------------------
    def aug_cfg(self) -> AugCfg:
        return AugCfg.strong(size=self.cfg.image_size)

    # -- torchvision path ---------------------------------------------------
    def _eval_splits(self):
        if self._eval_train is None:
            if self.synthetic:
                tf = make_eval_transform(self.cfg.image_size)
                self._eval_train = _SyntheticImages(20, self.num_classes,
                                                    self.cfg.image_size, seed=7,
                                                    transform=tf)
                self._eval_test = _SyntheticImages(10, self.num_classes,
                                                   self.cfg.image_size, seed=8,
                                                   transform=tf)
            else:
                self._eval_train, self._eval_test = get_eval_datasets(
                    self.cfg.dataset, size=self.cfg.image_size,
                    root=self.cfg.data_root)
            self._eval_labels = np.asarray(self._eval_train.targets, dtype=np.int64)
        return self._eval_train, self._eval_test

    def _lt_indices(self, gamma: float, seed: int) -> List[int]:
        self._eval_splits()
        return longtail_indices(self._eval_labels, self.num_classes, gamma, seed)

    def ssl_dataset(self, gamma: float, seed: int) -> Dataset:
        cfg_a = self.aug_cfg()
        if not self.synthetic:
            return get_ssl_dataset(self.cfg.dataset, cfg_a, None,
                                   imbalance_gamma=gamma, seed=seed,
                                   root=self.cfg.data_root)
        # synthetic: mirror the same construction by hand
        if self._syn_train is None:
            self._syn_train = _SyntheticImages(20, self.num_classes,
                                               self.cfg.image_size, seed=7,
                                               transform=None)
        labels_full = np.asarray(self._syn_train.targets, dtype=np.int64)
        idx = longtail_indices(labels_full, self.num_classes, gamma, seed)
        two_view = make_two_view_transform(cfg_a, None)
        return _SyntheticSSLDataset(self._syn_train, two_view, idx,
                                    labels_full[np.asarray(idx)])

    def probe_datasets(self, gamma: float, seed: int):
        """(long-tail probe train, balanced probe train, balanced test).

        The primary probe trains on the long-tail pool: at evaluation time we do
        not get to pretend the labels were balanced.  The balanced probe is a
        secondary diagnostic that separates "the representation is bad" from
        "the probe itself is starved on tail classes".
        """
        train_eval, test_eval = self._eval_splits()
        idx = self._lt_indices(gamma, seed)
        lt_train = Subset(train_eval, idx)
        balanced_train = train_eval
        return lt_train, balanced_train, test_eval

    def diagnostic_dataset(self, gamma: float, seed: int, n: int) -> Dataset:
        """Deterministic held-in pool used for effective rank and the prototype
        marginal.  Same images for every arm at a given (gamma, seed)."""
        lt_train, _, _ = self.probe_datasets(gamma, seed)
        n = int(min(n, len(lt_train)))
        rs = np.random.RandomState(20240 + seed)
        sel = sorted(int(i) for i in rs.choice(len(lt_train), size=n, replace=False))
        return Subset(lt_train, sel)


# ===========================================================================
# model
# ===========================================================================

class SmokeEncoder(nn.Module):
    """Tiny CNN used ONLY by --smoke.

    The smoke test validates the PIPELINE (shapes, resume, CSV, verdicts), not
    the science; a ResNet-18 would blow the three-minute CPU budget.  The real
    `make_encoder` is still constructed and forwarded once at smoke start so
    that the harness path is exercised too.
    """

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, out_dim, 3, stride=2, padding=1), nn.BatchNorm2d(out_dim), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StudentNet(nn.Module):
    """backbone -> projector.  Output is UNNORMALISED on purpose: `vicreg_reg`
    refuses a normalised input (thesis T2), and `gram_loss` / `info_nce` /
    `prototype_loss` all normalise internally."""

    def __init__(self, backbone: nn.Module, projector: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.projector = projector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.backbone(x))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_student(cfg: ExpConfig) -> Tuple[StudentNet, int]:
    if cfg.smoke:
        backbone: nn.Module = SmokeEncoder(out_dim=64)
        feat_dim = 64
    else:
        backbone, feat_dim = make_encoder(cfg.arch, dataset="cifar")
    projector = make_mlp(feat_dim, cfg.proj_hidden, cfg.proj_dim,
                         n_layers=cfg.proj_layers, bn=True)
    return StudentNet(backbone, projector), feat_dim


class BackboneOnly(nn.Module):
    """Adapter so the probes see the BACKBONE, not the projector.

    Linear evaluation is done on backbone features, as in every SSL paper; the
    projector is discarded.  This matters here: an arm could look good on
    projector features and bad on backbone features, and the backbone is what
    one would actually transfer.
    """

    def __init__(self, student: StudentNet):
        super().__init__()
        self.student = student

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.student.features(x)


# ===========================================================================
# training
# ===========================================================================

def lr_at(step: int, total: int, base_lr: float, warmup_frac: float) -> float:
    """Linear warmup then cosine decay to zero.  Identical for every arm."""
    warm = max(1, int(warmup_frac * total))
    if step < warm:
        return base_lr * float(step + 1) / float(warm)
    t = (step - warm) / max(1, total - warm)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, t)))


def random_orthogonal(d: int, seed: int, device, dtype) -> torch.Tensor:
    """A random element of O(d).  Built in float64 and cast afterwards, so that
    Q^T Q = I holds to ~1e-15 and the rotation check measures the mathematics,
    not the QR's rounding.  Note this is O(d), not SO(d): det(Q) may be -1, and
    that is deliberate -- T1 claims invariance under the full orthogonal group.
    """
    g = torch.Generator().manual_seed(int(seed))
    a = torch.randn(d, d, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    sgn = torch.sign(torch.diagonal(r))
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)   # unique QR, no zero column
    q = q * sgn.unsqueeze(0)
    return q.to(device=device, dtype=dtype)


def label_entropy_normalised(labels: np.ndarray, num_classes: int) -> float:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    p = counts.astype(np.float64)
    p = p / max(1e-12, p.sum())
    nz = p[p > 1e-12]
    H = float(-(nz * np.log(nz)).sum())
    return H / math.log(num_classes)


@torch.no_grad()
def diagnostics(student: StudentNet, ema: EMATeacher, protos: Optional[torch.Tensor],
                center: Optional[torch.Tensor], spec: RunSpec, cfg: ExpConfig,
                diag_ds: Dataset, device: torch.device) -> Dict[str, float]:
    """Effective rank of the projector output and, for prototype arms, the
    cluster marginal of the TEACHER -- the quantity P0 is about.

    Two things this function is careful about, both of them silent-fatal if got
    wrong:

    * it builds a DataLoader, and creating a DataLoader iterator draws a base
      seed from the GLOBAL torch generator even when ``shuffle=False``.  Called
      every ``eval_every`` steps from inside the training loop, that would make
      the trajectory depend on how often we evaluate.  The whole body therefore
      runs inside the harness's RNG island;
    * the network it measures is the one the loss actually trains against: the
      EMA target for every arm that HAS an EMA target, and the student for the
      one arm that does not (``--infonce-symmetric`` never calls ``ema.update``,
      so its "teacher" is still the random initialisation and its effective
      rank would be meaningless).
    """
    arm = ARMS[spec.arm]
    no_ema = (spec.arm == "infonce" and cfg.infonce_symmetric)
    net = student if no_ema else ema.target
    was_training = student.training
    student.eval()
    zs: List[torch.Tensor] = []
    hs: List[torch.Tensor] = []
    marg = None
    n_seen = 0
    tau_t_eff = spec.tau_t if arm.get("use_sharpen") else cfg.tau_s
    with _rng_island(seed=None):
        loader = DataLoader(diag_ds, batch_size=min(256, max(2, cfg.bs)), shuffle=False,
                            num_workers=0, drop_last=False)
        for batch in loader:
            x = batch["v1"] if isinstance(batch, dict) else batch[0]
            x = x.to(device, non_blocking=True)
            z_t = net(x).float()
            zs.append(z_t.cpu())
            # The linear probe reads the BACKBONE (see BackboneOnly): a rank
            # measured only on the projector output correlates two DIFFERENT
            # spaces, separated by a trained MLP.  Measured in A2 on 2026-08-26:
            # gram -> gram_vicreg moves the projector rank x4.83 and the backbone
            # rank x1.001 (0.04 sigma).  Logging both is what makes P6b possible.
            if hasattr(net, "features"):
                hs.append(net.features(x).float().cpu())
            if arm["uses_protos"] and protos is not None:
                Pn = F.normalize(protos.float(), dim=1)
                logits = F.normalize(z_t, dim=1) @ Pn.t()
                if arm["use_center"] and center is not None:
                    logits = logits - center.float().to(logits.device)
                p = F.softmax(logits / tau_t_eff, dim=1)
                s = p.sum(0).cpu()
                marg = s if marg is None else marg + s
                n_seen += p.shape[0]
    if was_training:
        student.train()

    Z = torch.cat(zs, 0)
    out = {"eff_rank": float(effective_rank(Z))}
    if hs:
        Hf = torch.cat(hs, 0)
        # Centred variants: h is post-ReLU and therefore non-negative, so its
        # UNCENTRED spectrum is dominated by the shared mean direction and can
        # mask a real difference between arms.  Both are reported; neither
        # replaces the other.
        out["eff_rank_h"] = float(effective_rank(Hf))
        out["eff_rank_h_centred"] = float(effective_rank(Hf - Hf.mean(dim=0, keepdim=True)))
        out["eff_rank_centred"] = float(effective_rank(Z - Z.mean(dim=0, keepdim=True)))
    if marg is not None and n_seen > 0:
        m = (marg / n_seen).numpy().astype(np.float64)
        m = m / max(1e-12, m.sum())
        nz = m[m > 1e-12]
        H = float(-(nz * np.log(nz)).sum())
        K = int(m.size)
        out["proto_marg_entropy_norm"] = H / math.log(K)
        out["proto_kl_uniform"] = math.log(K) - H
        out["proto_used_frac"] = float((m > 1.0 / (10.0 * K)).mean())
        # Honest ceiling: the marginal is an average of n_seen softmax rows, so
        # at most n_seen prototypes can carry mass and the normalised entropy is
        # capped at log(min(n_seen, K)) / log K.  P0's 0.90 threshold must be
        # read against this number, not against 1.0.
        out["proto_marg_entropy_cap"] = math.log(min(n_seen, K)) / math.log(K)
    else:
        out["proto_marg_entropy_norm"] = float("nan")
        out["proto_kl_uniform"] = float("nan")
        out["proto_used_frac"] = float("nan")
        out["proto_marg_entropy_cap"] = float("nan")
    return out


@torch.no_grad()
def rotation_check(z_s: torch.Tensor, z_t: torch.Tensor, protos: Optional[torch.Tensor],
                   spec: RunSpec, cfg: ExpConfig) -> Dict[str, float]:
    """P2, measured on the final embeddings of a real batch.

    Three measurements, and the third one is the one that makes the claim
    non-trivial:

      rot_gram_reldiff          rotate both branches, Gram objective    -> ~0
      rot_proto_reldiff         rotate both branches, prototypes FIXED  -> > 0
      rot_proto_rotated_reldiff rotate both branches AND the prototypes -> ~0

    The third is the MANDATORY control of T1.  Without it, "the prototype loss
    moves under rotation" is compatible with "softmax cross-entropy is just a
    fragile objective".  With it, the loss is shown to be exactly as invariant
    as the Gram once the frame travels with the data: what breaks invariance is
    the CHOICE OF A FRAME (fixed c_k), not the softmax.

    Everything is computed in float64.  The prediction is an identity, and the
    pre-registered threshold is 1e-4 relative; in float32 the rounding floor of
    a B x B reduction can itself reach that order when the loss is small, which
    would let arithmetic noise "falsify" a theorem.  What training sees is still
    float32; this measurement is about the mathematics.
    """
    d = z_s.shape[1]
    Q = random_orthogonal(d, seed=1234 + spec.seed, device=z_s.device,
                          dtype=torch.float64)
    s = z_s.detach().double()
    a = z_t.detach().double()
    sQ, aQ = s @ Q, a @ Q

    g0 = float(gram_loss(s, a))
    g1 = float(gram_loss(sQ, aQ))
    rel_gram = abs(g1 - g0) / (abs(g0) + 1e-30)

    P = protos
    if P is None:  # arms without prototypes still get the comparison, on a fixed random head
        gen = torch.Generator().manual_seed(999)
        P = torch.randn(min(spec.K, 1024), d, generator=gen)
    P = P.detach().to(device=z_s.device, dtype=torch.float64)
    p0, _ = prototype_loss(s, a, P, cfg.tau_s, spec.tau_t, None, True, True)
    p1, _ = prototype_loss(sQ, aQ, P, cfg.tau_s, spec.tau_t, None, True, True)
    p2, _ = prototype_loss(sQ, aQ, P @ Q, cfg.tau_s, spec.tau_t, None, True, True)
    p0, p1, p2 = float(p0), float(p1), float(p2)
    rel_proto = abs(p1 - p0) / (abs(p0) + 1e-30)
    rel_proto_rot = abs(p2 - p0) / (abs(p0) + 1e-30)
    return {"rot_gram_reldiff": rel_gram, "rot_proto_reldiff": rel_proto,
            "rot_proto_rotated_reldiff": rel_proto_rot}


def objective(spec: RunSpec, cfg: ExpConfig, arm: dict, z_s: torch.Tensor,
              z_t: torch.Tensor, protos, center):
    """The ONLY place an arm's loss is defined.

    Factored out so that the training step and the resume-recovery path below
    cannot drift apart: a recovery path that recomputed a slightly different
    loss would corrupt `final_loss` for exactly the runs that were interrupted,
    i.e. silently and only sometimes.

    Returns ``(loss, logs, new_center)``.  ``new_center`` is meaningful only for
    the prototype arms; callers that are not taking an optimisation step must
    discard it (advancing the DINO centre outside training would be a leak of
    evaluation into state).
    """
    logs: Dict[str, float] = {}
    if arm["uses_protos"]:
        loss, center = prototype_loss(z_s, z_t, protos, cfg.tau_s, spec.tau_t,
                                      center, use_center=arm["use_center"],
                                      use_sharpen=arm["use_sharpen"])
        logs["loss_agree"] = float(loss.detach())
    elif spec.arm == "relational":
        # pos_weight=None -> the pure T2 form: the two diagonals are identically
        # 1 and contribute exactly zero, so the objective is carried by the
        # B*(B-1) off-diagonal entries and there is NO explicit positive term.
        # That is the point (and the reason the VICReg term is mandatory here).
        agree = gram_loss(z_s, z_t)
        reg, parts = vicreg_reg(z_s, var_w=spec.var_w, cov_w=spec.cov_w)
        loss = agree + reg
        logs.update(loss_agree=float(agree.detach()),
                    vic_var=parts["var"], vic_cov=parts["cov"])
    elif spec.arm == "infonce":
        agree = info_nce(z_s, z_t, tau=cfg.tau_nce)
        reg, parts = vicreg_reg(z_s, var_w=spec.var_w, cov_w=spec.cov_w)
        loss = agree + reg
        logs.update(loss_agree=float(agree.detach()),
                    vic_var=parts["var"], vic_cov=parts["cov"])
    else:  # pragma: no cover - ARMS is closed
        raise ValueError(f"unknown arm {spec.arm!r}")
    return loss, logs, center


def train_one_run(spec: RunSpec, cfg: ExpConfig, provider: DataProvider,
                  device: torch.device) -> Dict[str, float]:
    """Train one arm at one (gamma, seed, B, K, tau_t, var_w, cov_w).

    Resumable: state is checkpointed every `ckpt_every` steps and the CSV is
    truncated back to the checkpoint on restart (handled by `Run`).  A run whose
    `<name>.summary.json` already exists is skipped entirely.
    """
    name = spec.name()
    summary_path = os.path.join(cfg.runs_dir(), f"{name}.summary.json")
    if cfg.resume and os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            done = json.load(f)
        print(f"[skip] {name} (already finished)")
        return dict(done.get("summary", {}))

    arm = ARMS[spec.arm]
    set_seed(spec.seed)

    ssl_ds = provider.ssl_dataset(spec.gamma, spec.seed)
    n_pretrain = len(ssl_ds)
    if n_pretrain < spec.bs:
        raise ValueError(
            f"{name}: pre-training pool has {n_pretrain} images but B={spec.bs}; "
            f"drop_last would yield no batch.  Lower --bs or raise gamma."
        )
    lab_ent = label_entropy_normalised(np.asarray(ssl_ds.labels), provider.num_classes)

    student, _ = build_student(cfg)
    student.to(device)
    ema = EMATeacher(student, tau_base=cfg.ema_base, tau_final=cfg.ema_final,
                     total_steps=spec.steps)
    ema.to(device)

    protos: Optional[nn.Parameter] = None
    center: Optional[torch.Tensor] = None
    params = list(student.parameters())
    if arm["uses_protos"]:
        g = torch.Generator().manual_seed(spec.seed * 7919 + 13)
        protos = nn.Parameter(F.normalize(
            torch.randn(spec.K, cfg.proj_dim, generator=g), dim=1).to(device))
        params.append(protos)
        center = torch.zeros(spec.K, device=device)

    opt = torch.optim.AdamW(params, lr=spec.lr, weight_decay=cfg.weight_decay)

    run = Run(name=name, outdir=cfg.runs_dir(),
              config={**asdict(spec), "family": spec.family,
                      "n_pretrain": n_pretrain, "label_entropy_norm": lab_ent,
                      "exp": {k: v for k, v in asdict(cfg).items()
                              if k not in ("seeds",)}},
              # The only scalar this Run tracks is a LOSS.  We never pass
              # `best_metric` to save_ckpt (there is no meaningful "best" step
              # for an SSL pre-training: the last step is the one we probe), so
              # the flag is inert -- but it is set to False anyway, because the
              # harness contract says a run that logs a loss must minimise, and
              # a future edit that starts passing best_metric must not silently
              # keep the WORST checkpoint.
              resume=cfg.resume, higher_is_better=False)

    start_step = 0
    ck = run.load_ckpt() if cfg.resume else None
    if ck is not None:
        student.load_state_dict(ck["student"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        if protos is not None and ck.get("protos") is not None:
            with torch.no_grad():
                protos.copy_(ck["protos"].to(device))
            center = ck["center"].to(device) if ck.get("center") is not None else center
        start_step = int(ck["step"])
        try:
            torch.set_rng_state(ck["rng_torch"])
            np.random.set_state(ck["rng_numpy"])
            random.setstate(ck["rng_python"])
        except Exception:
            pass  # RNG restoration is best-effort; model state is what matters
        print(f"[resume] {name} from step {start_step}/{spec.steps}")

    workers = cfg.workers if cfg.workers >= 0 else (8 if device.type == "cuda" else 0)
    gen = torch.Generator()
    gen.manual_seed(spec.seed * 100003 + 7)
    loader = DataLoader(ssl_ds, batch_size=spec.bs, shuffle=True,
                        num_workers=workers, drop_last=True, generator=gen,
                        pin_memory=(device.type == "cuda"),
                        persistent_workers=(workers > 0))
    it = iter(loader)

    diag_ds = provider.diagnostic_dataset(spec.gamma, spec.seed, cfg.n_diag)
    use_amp = bool(cfg.amp) and device.type == "cuda"

    student.train()
    last_z_s = last_z_t = None
    t0 = time.time()
    for step in range(start_step, spec.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, spec.steps, spec.lr, cfg.warmup_frac)

        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        v1 = batch["v1"].to(device, non_blocking=True)
        v2 = batch["v2"].to(device, non_blocking=True)

        amp_ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if use_amp
                   else contextlib.nullcontext())
        with amp_ctx:
            z_s = student(v1)
            if spec.arm == "infonce" and cfg.infonce_symmetric:
                z_t = student(v2)                    # textbook SimCLR: both branches student
            else:
                with torch.no_grad():
                    z_t = ema.target(v2)
        # The losses are computed in float32: the Gram MSE and the VICReg
        # covariance are exactly the places where bf16 rounding would bite.
        z_s = z_s.float()
        z_t = z_t.float()

        loss, logs, center = objective(spec, cfg, arm, z_s, z_t, protos, center)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if protos is not None and step < cfg.proto_freeze_steps:
            # DINO freezes the last layer for the first epoch.  Applied with the
            # SAME value to every prototype configuration, so it is a constant of
            # the family, not a per-config tuning knob.  It is also a fourth knob
            # the relational arm does not have.
            #
            # `grad = None`, NOT `grad.zero_()`: AdamW skips a parameter whose
            # grad is None, but a zero grad still advances its Adam moments and
            # still applies decoupled weight decay.  "Frozen" must mean frozen.
            protos.grad = None
        opt.step()
        if not (spec.arm == "infonce" and cfg.infonce_symmetric):
            ema.update(step)

        last_z_s, last_z_t = z_s.detach(), z_t.detach()

        if (step + 1) % max(1, cfg.eval_every) == 0 or step == spec.steps - 1:
            unif, align = uniformity_alignment(last_z_s, last_z_t)
            d = diagnostics(student, ema, protos, center, spec, cfg, diag_ds, device)
            run.log(step + 1, loss=float(loss.detach()), lr=opt.param_groups[0]["lr"],
                    uniformity=unif, alignment=align, **logs, **d)
        elif (step + 1) % 50 == 0:
            run.log(step + 1, loss=float(loss.detach()),
                    lr=opt.param_groups[0]["lr"], **logs)

        if (step + 1) % max(1, cfg.ckpt_every) == 0 or step == spec.steps - 1:
            run.save_ckpt(step + 1, student=student.state_dict(), ema=ema.state_dict(),
                          opt=opt.state_dict(),
                          protos=(protos.detach().cpu() if protos is not None else None),
                          center=(center.detach().cpu() if center is not None else None),
                          rng_torch=torch.get_rng_state(),
                          rng_numpy=np.random.get_state(),
                          rng_python=random.getstate())

    final_loss = float(loss.detach()) if last_z_s is not None else float("nan")

    # ---- resume edge case: the loop ran ZERO iterations -------------------
    # A session killed between the last `save_ckpt` and `run.finish` leaves a
    # checkpoint at step == spec.steps and no summary.json.  On restart
    # `start_step == spec.steps`, the loop body never executes, and `loss`,
    # `last_z_s`, `last_z_t` are unbound -> UnboundLocalError, i.e. every
    # interrupted-at-the-very-end run crashes forever afterwards.  Recompute
    # them from one fresh batch, WITHOUT touching the optimiser, the EMA or the
    # DINO centre, inside an RNG island so the recovery does not depend on how
    # the crash happened to land.
    if last_z_s is None:
        print(f"[recover] {name}: resumed at the final step; recomputing the "
              f"final-batch diagnostics without taking an optimisation step.")
        with _rng_island(seed=spec.seed), torch.no_grad():
            rec_loader = DataLoader(ssl_ds, batch_size=spec.bs, shuffle=False,
                                    num_workers=0, drop_last=True)
            batch = next(iter(rec_loader))
            v1 = batch["v1"].to(device)
            v2 = batch["v2"].to(device)
            student.eval()
            z_s = student(v1).float()
            z_t = (student(v2) if (spec.arm == "infonce" and cfg.infonce_symmetric)
                   else ema.target(v2)).float()
            rec_loss, _, _ = objective(spec, cfg, arm, z_s, z_t, protos, center)
            final_loss = float(rec_loss)
            last_z_s, last_z_t = z_s.detach(), z_t.detach()
            student.train()

    # ---------------- final evaluation ------------------------------------
    student.eval()
    backbone = BackboneOnly(student).to(device)
    lt_train, bal_train, test_ds = provider.probe_datasets(spec.gamma, spec.seed)

    probe = linear_probe(backbone, lt_train, test_ds, epochs=cfg.probe_epochs,
                         device=device, num_classes=provider.num_classes,
                         probe_seed=spec.seed)
    # The shortcut below is only legitimate when the two pools are literally the
    # same images.  That holds for a class-balanced source at gamma=1.0
    # (longtail_indices truncates every class to n_0 = min count = the full
    # class), and NOT for a source that is already imbalanced.  Check it rather
    # than assume it: a wrong `probe_acc_balanced` would be invisible.
    if abs(spec.gamma - 1.0) < 1e-12 and len(lt_train) == len(bal_train):
        probe_bal_acc = probe["acc"]      # identical pools; do not pay twice
    else:
        probe_bal = linear_probe(backbone, bal_train, test_ds, epochs=cfg.probe_epochs,
                                 device=device, num_classes=provider.num_classes,
                                 probe_seed=spec.seed)
        probe_bal_acc = probe_bal["acc"]
    knn = knn_probe(backbone, lt_train, test_ds, k=min(20, max(1, len(lt_train) // 10)),
                    device=device)

    diag = diagnostics(student, ema, protos, center, spec, cfg, diag_ds, device)
    rot = rotation_check(last_z_s, last_z_t,
                         (protos.detach() if protos is not None else None), spec, cfg)
    unif, align = uniformity_alignment(last_z_s, last_z_t)

    summary = {
        "run_name": name, "study": spec.study, "arm": spec.arm, "family": spec.family,
        "gamma": spec.gamma, "seed": spec.seed, "bs": spec.bs, "steps": spec.steps,
        "samples_seen": spec.samples_seen, "K": spec.K, "tau_t": spec.tau_t,
        "tau_s": cfg.tau_s, "var_w": spec.var_w, "cov_w": spec.cov_w, "lr": spec.lr,
        "n_pretrain": n_pretrain, "label_entropy_norm": lab_ent,
        "probe_acc": probe["acc"], "probe_head_acc": probe["head_acc"],
        "probe_tail_acc": probe["tail_acc"], "probe_acc_balanced": probe_bal_acc,
        "knn_acc": knn, "uniformity": unif, "alignment": align,
        "final_loss": final_loss, "wall_s": round(time.time() - t0, 1),
        **diag, **rot,
    }
    run.finish(summary)
    print(f"[done] {name}  acc={probe['acc']:.4f} tail={probe['tail_acc']:.4f} "
          f"({summary['wall_s']:.0f}s)")
    return summary


# ===========================================================================
# study definitions
# ===========================================================================

def specs_main(cfg: ExpConfig, arms: List[str], gammas: List[float]) -> List[RunSpec]:
    out = []
    for gamma in gammas:
        for arm in arms:
            for seed in cfg.seeds:
                out.append(RunSpec("main", arm, gamma, seed, cfg.bs, cfg.steps,
                                   cfg.K, cfg.tau_t, cfg.var_w, cfg.cov_w, cfg.lr))
    return out


def specs_sweep(cfg: ExpConfig, gammas: List[float],
                proto_grid_k: List[int], proto_grid_tau: List[float],
                rel_grid_var: List[float], rel_grid_cov: List[float]) -> List[RunSpec]:
    """P1.  Two grids of equal cardinality, one per family (amendment (a))."""
    out = []
    for gamma in gammas:
        for seed in cfg.seeds:
            for K in proto_grid_k:
                for tt in proto_grid_tau:
                    out.append(RunSpec("sweep", "proto_full", gamma, seed, cfg.bs,
                                       cfg.steps, K, tt, cfg.var_w, cfg.cov_w, cfg.lr))
            for vw in rel_grid_var:
                for cw in rel_grid_cov:
                    out.append(RunSpec("sweep", "relational", gamma, seed, cfg.bs,
                                       cfg.steps, cfg.K, cfg.tau_t, vw, cw, cfg.lr))
    return out


def specs_batch(cfg: ExpConfig, gammas: List[float], batch_sizes: List[int],
                arms: List[str]) -> List[RunSpec]:
    """P3.  Equal SAMPLES SEEN, not equal steps: steps * B is held at the default
    budget so that the arms differ in the number of updates and in the Gram size,
    which is exactly the coupling we want to expose, and not in how much data
    they were shown."""
    base_samples = cfg.steps * cfg.bs
    out = []
    for gamma in gammas:
        for bs in batch_sizes:
            steps = max(1, base_samples // bs)
            # Anchor the linear rule at cfg.bs, NOT at a hard-coded 256: with
            # `--bs 128` the old form silently halved every learning rate in the
            # batch study relative to the main study, so B=cfg.bs would not
            # reproduce the main-study run it is supposed to be compared to.
            lr = cfg.lr * (bs / float(cfg.bs)) if cfg.lr_scaling else cfg.lr
            for arm in arms:
                for seed in cfg.seeds:
                    out.append(RunSpec("batch", arm, gamma, seed, bs, steps,
                                       cfg.K, cfg.tau_t, cfg.var_w, cfg.cov_w, lr))
    return out


def budget_report(specs: List[RunSpec], study: str) -> None:
    """Every comparison in this file goes through a budget check.

    For `main` and `sweep` the arms are strictly matched (same steps, same B,
    same forwards) and `matched_budget_check` must pass.  For `batch` it must
    FAIL by construction -- different B is the independent variable -- so we call
    it anyway and print the refusal, then report the sample-matched table.
    """
    if not specs:
        return
    groups: Dict[str, dict] = {}
    for s in specs:
        key = f"{s.arm}|B{s.bs}|T{s.steps}"
        groups[key] = dict(name=key, steps=s.steps, batch_size=s.bs,
                           forward_passes=2, samples_seen=s.samples_seen)
    arms = list(groups.values())
    print(f"\n### budget report for study {study!r} ({len(specs)} runs) ###")
    if len(arms) < 2:
        print("only one budget signature; nothing to check.")
        return
    if study == "batch":
        try:
            matched_budget_check(arms)
            print("UNEXPECTED: the batch study should NOT be step-matched.")
        except ValueError as e:
            print("matched_budget_check refused the batch study, as it must:")
            print("  " + str(e).splitlines()[0])
        samples = {a["name"]: a["steps"] * a["batch_size"] * 2 for a in arms}
        uniq = sorted(set(samples.values()))
        print("sample-matched budget (steps * B * forward_passes):")
        for k, v in samples.items():
            print(f"  {k:<28} {v}")
        if len(uniq) > 1:
            spread = (max(uniq) - min(uniq)) / max(1, max(uniq))
            if spread > 0.02:
                raise ValueError(
                    f"batch study is not sample-matched: {uniq} "
                    f"(relative spread {spread:.3f} > 0.02). Integer division of "
                    f"steps*bs by B left a remainder; adjust --steps/--bs.")
            print(f"OK: sample budgets agree to within {spread:.4f} "
                  f"(integer-division remainder only).")
        else:
            print("OK: sample budgets identical.")
    else:
        matched_budget_check(arms)


# ===========================================================================
# results table
# ===========================================================================

class ResultsTable:
    """Append-only long-format CSV, flushed per row so a Colab kill loses at
    most the run in flight (whose own checkpoint survives anyway)."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.seen = set()
        if os.path.exists(path):
            import csv as _csv
            with open(path, "r", newline="", encoding="utf-8") as f:
                rd = _csv.DictReader(f)
                header = list(rd.fieldnames or [])
                # An existing CSV written under a DIFFERENT schema would be
                # appended to with today's fieldnames and every later row would
                # be misaligned column-by-column -- unreadable data that still
                # parses.  Refuse instead, and say what to do.
                if header and header != list(RESULT_COLUMNS):
                    raise ValueError(
                        f"{path} was written with a different column set:\n"
                        f"  on disk : {header}\n"
                        f"  expected: {list(RESULT_COLUMNS)}\n"
                        f"Appending would silently misalign every new row. Move "
                        f"the old file aside (or use a fresh --outdir); the "
                        f"per-run summary.json files are the source of truth and "
                        f"the table rebuilds from them on the next pass."
                    )
                for row in rd:
                    self.seen.add(row.get("run_name", ""))
        else:
            import csv as _csv
            with open(path, "w", newline="", encoding="utf-8") as f:
                _csv.DictWriter(f, fieldnames=RESULT_COLUMNS).writeheader()

    def add(self, summary: Dict[str, float]) -> None:
        import csv as _csv
        name = summary.get("run_name", "")
        if name in self.seen:
            return
        row = {k: summary.get(k, "") for k in RESULT_COLUMNS}
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
            w.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        self.seen.add(name)


# ===========================================================================
# aggregation, figures, verdicts
# ===========================================================================

def _mean_std(values: List[float]) -> Tuple[float, float]:
    v = [float(x) for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return float("nan"), float("nan")
    m = float(np.mean(v))
    s = float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
    return m, s


def sensitivity_ratio(df, config_keys: List[str], metric: str = "probe_acc") -> dict:
    """std over configurations of the seed-mean, divided by the typical
    across-seed std.  > 1 means "changing this knob matters more than changing
    the seed"; that is the whole content of P1."""
    if len(df) == 0:
        return {"ratio": float("nan"), "config_std": float("nan"),
                "seed_noise": float("nan"), "range_pp": float("nan"), "n_configs": 0}
    g = df.groupby(config_keys)[metric]
    means = g.mean().values
    stds = g.std(ddof=1).values
    seed_noise = float(np.nanmean(stds)) if len(stds) else float("nan")
    config_std = float(np.std(means, ddof=1)) if len(means) > 1 else float("nan")
    ratio = (config_std / seed_noise) if (seed_noise and seed_noise > 1e-12
                                          and not math.isnan(seed_noise)) else float("nan")
    return {"ratio": ratio, "config_std": config_std, "seed_noise": seed_noise,
            "range_pp": float(100.0 * (np.max(means) - np.min(means))) if len(means) else float("nan"),
            "best_pp": float(100.0 * np.max(means)) if len(means) else float("nan"),
            "worst_pp": float(100.0 * np.min(means)) if len(means) else float("nan"),
            "n_configs": int(len(means))}


def make_figures(df, cfg: ExpConfig) -> List[str]:
    os.makedirs(cfg.figures_dir(), exist_ok=True)
    paths: List[str] = []

    # --- figure 1: accuracy / head / tail vs gamma -------------------------
    main = df[df["study"] == "main"]
    if len(main):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
        for ax, metric, title in zip(
                axes, ["probe_acc", "probe_head_acc", "probe_tail_acc"],
                ["overall accuracy", "head-class accuracy", "tail-class accuracy"]):
            for arm in ARM_ORDER:
                sub = main[main["arm"] == arm]
                if not len(sub):
                    continue
                gs = sorted(sub["gamma"].unique())
                mu = [sub[sub["gamma"] == g][metric].mean() for g in gs]
                sd = [sub[sub["gamma"] == g][metric].std(ddof=1) for g in gs]
                ax.errorbar(gs, mu, yerr=sd, capsize=3, label=arm, **ARM_STYLE[arm])
            ax.set_xscale("log")
            ax.set_xlabel("gamma (tail/head ratio; 1.0 = balanced)")
            ax.set_ylabel("linear probe accuracy")
            ax.set_title(title)
            ax.grid(True, linewidth=0.4, alpha=0.5)
        axes[0].legend(fontsize=8, loc="best")
        fig.suptitle("T1 main: CIFAR-100-LT pre-training, balanced test, "
                     "matched budget (mean +/- sd over seeds)")
        fig.tight_layout()
        p = os.path.join(cfg.figures_dir(), "fig1_accuracy_vs_gamma.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

    # --- figure 2: hyperparameter sensitivity ------------------------------
    sw = df[df["study"] == "sweep"]
    if len(sw):
        # One COLUMN per family, one ROW per gamma.  Pooling the gammas into one
        # panel would draw the gamma effect as configuration spread, which is
        # exactly the statistic P1 reports.
        fams = [f for f in ("prototype", "relational") if len(sw[sw["family"] == f])]
        sw_gammas = sorted(np.round(sw["gamma"].unique(), 6))
        nrow, ncol = max(1, len(sw_gammas)), max(1, len(fams))
        fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4 * nrow),
                                 squeeze=False)
        for i, gamma in enumerate(sw_gammas):
            for j, fam in enumerate(fams):
                ax = axes[i][j]
                sub = sw[(sw["family"] == fam) & np.isclose(sw["gamma"], gamma)]
                if not len(sub):
                    ax.set_axis_off()
                    continue
                keys = ["K", "tau_t"] if fam == "prototype" else ["var_w", "cov_w"]
                g = sub.groupby(keys)["probe_acc"]
                means = g.mean()
                stds = g.std(ddof=1)
                labels = [", ".join(f"{k}={v}" for k, v in
                                    zip(keys, idx if isinstance(idx, tuple) else (idx,)))
                          for idx in means.index]
                xs = np.arange(len(means))
                ax.errorbar(xs, means.values, yerr=stds.values, fmt="o", color="black",
                            capsize=3, linestyle="none")
                ax.set_xticks(xs)
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
                st = sensitivity_ratio(sub, keys)
                centre = float(np.nanmean(means.values))
                noise = st["seed_noise"]
                if not math.isnan(noise) and not math.isnan(centre):
                    ax.axhspan(centre - noise, centre + noise, color="0.85", zorder=0,
                               label=f"+/- seed noise ({100 * noise:.2f} pp)")
                    ax.legend(fontsize=8)
                ax.set_title(f"{fam}, gamma={gamma:g}: spread/noise = {st['ratio']:.2f}, "
                             f"range = {st['range_pp']:.2f} pp")
                ax.set_ylabel("linear probe accuracy")
                ax.grid(True, linewidth=0.4, alpha=0.5)
        fig.suptitle("P1: configuration spread against seed noise "
                     "(equal-cardinality grids, matched budget)")
        fig.tight_layout()
        p = os.path.join(cfg.figures_dir(), "fig2_hyperparameter_sensitivity.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

    # --- figure 3: batch coupling ------------------------------------------
    bt = df[df["study"] == "batch"]
    if len(bt):
        fig, ax = plt.subplots(figsize=(6, 4))
        for arm in ARM_ORDER:
            for gamma in sorted(bt["gamma"].unique()):
                sub = bt[(bt["arm"] == arm) & (bt["gamma"] == gamma)]
                if not len(sub):
                    continue
                bs = sorted(sub["bs"].unique())
                mu = [sub[sub["bs"] == b]["probe_acc"].mean() for b in bs]
                sd = [sub[sub["bs"] == b]["probe_acc"].std(ddof=1) for b in bs]
                style = dict(ARM_STYLE[arm])
                if gamma < 1.0:
                    style["linestyle"] = ":"
                ax.errorbar(bs, mu, yerr=sd, capsize=3,
                            label=f"{arm}, gamma={gamma:g}", **style)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("batch size B (Gram is a batch statistic)")
        ax.set_ylabel("linear probe accuracy")
        ax.set_title("P3 control: equal samples seen, lr scaled linearly with B")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(cfg.figures_dir(), "fig3_batch_coupling.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

    # --- figure 4: the mechanism (imposed uniform marginal) ----------------
    if len(main):
        fig, ax = plt.subplots(figsize=(6, 4))
        for arm in ["proto_full", "proto_nocenter", "proto_nosharpen"]:
            sub = main[main["arm"] == arm]
            if not len(sub):
                continue
            gs = sorted(sub["gamma"].unique())
            mu = [sub[sub["gamma"] == g]["proto_marg_entropy_norm"].mean() for g in gs]
            sd = [sub[sub["gamma"] == g]["proto_marg_entropy_norm"].std(ddof=1) for g in gs]
            ax.errorbar(gs, mu, yerr=sd, capsize=3, label=f"{arm} cluster marginal",
                        **ARM_STYLE[arm])
        gs = sorted(main["gamma"].unique())
        true_h = [main[main["gamma"] == g]["label_entropy_norm"].mean() for g in gs]
        ax.plot(gs, true_h, color="black", linestyle="-", marker="x", linewidth=1.2,
                label="true label entropy of the pool")
        ax.set_xscale("log")
        ax.set_xlabel("gamma (tail/head ratio)")
        ax.set_ylabel("normalised entropy (H / log K, H / log C)")
        ax.set_title("P0: the head keeps asserting a uniform marginal\n"
                     "that the data no longer has")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(cfg.figures_dir(), "fig4_cluster_marginal_entropy.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)

    return paths


def compute_verdicts(df, cfg: ExpConfig) -> dict:
    """One verdict per prediction, each with the number that justifies it.

    Verdicts are limited to {confirmee, infirmee, non concluante}; anything the
    data cannot settle stays "non concluante" rather than being rounded up.
    """
    out: Dict[str, dict] = {}
    main = df[df["study"] == "main"]
    sw = df[df["study"] == "sweep"]
    bt = df[df["study"] == "batch"]

    def n_seeds(sub) -> int:
        return int(sub["seed"].nunique()) if len(sub) else 0

    # ---- P2: O(d) invariance ---------------------------------------------
    v: Dict[str, object] = {"prediction": PREREG["P2_od_invariance"]}
    if len(df) and df["rot_gram_reldiff"].notna().any():
        gmax = float(df["rot_gram_reldiff"].max())
        pmed = float(df["rot_proto_reldiff"].median())
        cmax = (float(df["rot_proto_rotated_reldiff"].max())
                if "rot_proto_rotated_reldiff" in df.columns
                and df["rot_proto_rotated_reldiff"].notna().any() else float("nan"))
        v.update(gram_max_relative_change=gmax, proto_median_relative_change=pmed,
                 proto_rotated_control_max_relative_change=cmax)
        control_ok = (not math.isnan(cmax)) and cmax < 1e-4
        if gmax < 1e-4 and pmed > 1e-3 and control_ok:
            v["verdict"] = "confirmee"
        elif math.isnan(cmax):
            v["verdict"] = "non concluante"
        else:
            v["verdict"] = "infirmee"
        v["justification"] = (
            f"rotating both branches by a random orthogonal Q moves the Gram "
            f"objective by at most {gmax:.2e} (relative) and the prototype "
            f"objective by {pmed:.2e} (median).  CONTROL: rotating the "
            f"prototypes as well brings the prototype objective back to "
            f"{cmax:.2e}, so what breaks the invariance is the fixed FRAME, not "
            f"the softmax.  All three measured in float64 on the final batch.")
    else:
        v["verdict"] = "non concluante"
        v["justification"] = "no run produced a rotation measurement."
    out["P2_od_invariance"] = v

    # ---- P0: imposed uniformity ------------------------------------------
    v = {"prediction": PREREG["P0_imposed_uniformity"]}
    pf = main[main["arm"] == "proto_full"]
    if len(pf) and {1.0, 0.01}.issubset(set(np.round(pf["gamma"].unique(), 6))):
        hi = pf[np.isclose(pf["gamma"], 1.0)]
        lo = pf[np.isclose(pf["gamma"], 0.01)]
        H_hi, _ = _mean_std(hi["proto_marg_entropy_norm"].tolist())
        H_lo, _ = _mean_std(lo["proto_marg_entropy_norm"].tolist())
        T_hi, _ = _mean_std(hi["label_entropy_norm"].tolist())
        T_lo, _ = _mean_std(lo["label_entropy_norm"].tolist())
        dH, dT = abs(H_lo - H_hi), abs(T_lo - T_hi)
        v.update(H_proto_balanced=H_hi, H_proto_longtail=H_lo,
                 H_true_balanced=T_hi, H_true_longtail=T_lo,
                 delta_proto=dH, delta_true=dT, n_seeds=n_seeds(pf))
        if dT < 1e-6 or math.isnan(dH):
            v["verdict"] = "non concluante"
        elif dH < 0.25 * dT and H_lo > 0.90:
            v["verdict"] = "confirmee"
        elif dH > 0.75 * dT:
            v["verdict"] = "infirmee"
        else:
            v["verdict"] = "non concluante"
        v["justification"] = (
            f"true label entropy falls by {dT:.3f} (normalised) between "
            f"gamma=1.0 and gamma=0.01 while the DINO cluster marginal entropy "
            f"moves by {dH:.3f} and stays at {H_lo:.3f}.")
    else:
        v["verdict"] = "non concluante"
        v["justification"] = "proto_full was not run at both gamma=1.0 and gamma=0.01."
    out["P0_imposed_uniformity"] = v

    # ---- P1: hyperparameter sensitivity ----------------------------------
    # PER GAMMA, always.  A sweep run at two gammas and pooled would report the
    # gamma effect as "configuration spread" -- the classic way to manufacture a
    # sensitivity out of nothing.  If several gammas were swept we take the
    # CONSERVATIVE reading: confirmed only if confirmed at every gamma,
    # falsified as soon as it is falsified at one.
    v = {"prediction": PREREG["P1_hyperparameter_sensitivity"]}
    proto_sw = sw[sw["family"] == "prototype"]
    rel_sw = sw[sw["family"] == "relational"]
    if len(proto_sw) and len(rel_sw):
        gammas = sorted(set(np.round(proto_sw["gamma"].unique(), 6))
                        & set(np.round(rel_sw["gamma"].unique(), 6)))
        per_gamma, verdicts_g = {}, []
        for gamma in gammas:
            p_g = proto_sw[np.isclose(proto_sw["gamma"], gamma)]
            r_g = rel_sw[np.isclose(rel_sw["gamma"], gamma)]
            sp = sensitivity_ratio(p_g, ["K", "tau_t"])
            sr = sensitivity_ratio(r_g, ["var_w", "cov_w"])
            ns = min(n_seeds(p_g), n_seeds(r_g))
            rp, rr = sp["ratio"], sr["ratio"]
            if math.isnan(rp) or math.isnan(rr) or ns < 2:
                vg = "non concluante"
            elif rp >= 2.0 and rp >= 2.0 * rr:
                vg = "confirmee"
            elif rr >= rp:
                vg = "infirmee"
            else:
                vg = "non concluante"
            verdicts_g.append(vg)
            per_gamma[f"gamma={gamma:g}"] = {"prototype": sp, "relational": sr,
                                             "n_seeds": ns, "verdict": vg}
        v["per_gamma"] = per_gamma
        v["n_seeds"] = min((d["n_seeds"] for d in per_gamma.values()), default=0)
        if not verdicts_g:
            v["verdict"] = "non concluante"
        elif "infirmee" in verdicts_g:
            v["verdict"] = "infirmee"
        elif all(x == "confirmee" for x in verdicts_g):
            v["verdict"] = "confirmee"
        else:
            v["verdict"] = "non concluante"
        v["justification"] = "; ".join(
            f"{g}: prototype spans {d['prototype']['range_pp']:.2f} pp "
            f"(spread/noise {d['prototype']['ratio']:.2f}) over a "
            f"{d['prototype']['n_configs']}-point grid, relational spans "
            f"{d['relational']['range_pp']:.2f} pp "
            f"(spread/noise {d['relational']['ratio']:.2f}) over a "
            f"{d['relational']['n_configs']}-point grid of ITS own knobs "
            f"(var_w, cov_w) -> {d['verdict']}"
            for g, d in per_gamma.items()) or "no common gamma between the two grids."
        v["caveat"] = ("this compares the sensitivity of two different knob sets, "
                       "not the existence of knobs in the absolute; the relational "
                       "arm is not knob-free, it is free of K/centering/sharpening. "
                       "The reported statistic is the SPREAD ACROSS THE WHOLE GRID "
                       "(std of the per-config seed-means, and range_pp/best_pp/"
                       "worst_pp), never the best point of either family.")
    else:
        v["verdict"] = "non concluante"
        v["justification"] = "the sweep study was not run for both families."
    out["P1_hyperparameter_sensitivity"] = v

    # ---- P3: batch coupling ----------------------------------------------
    # Reported per (arm, gamma) and never pooled: B is the independent variable,
    # gamma is a different data regime, and the prototype arm is here as a
    # CONTROL.  "The relational arm moves with B" only means something relative
    # to how much an arm with no batch-level statistic moves at the same budget.
    v = {"prediction": PREREG["P3_batch_coupling"]}
    rel_bt = bt[bt["arm"] == "relational"]
    if len(rel_bt):
        per_cell: Dict[str, dict] = {}
        detected = False
        for arm in sorted(bt["arm"].unique()):
            for gamma in sorted(bt[bt["arm"] == arm]["gamma"].unique()):
                sub = bt[(bt["arm"] == arm) & np.isclose(bt["gamma"], gamma)]
                st = sensitivity_ratio(sub, ["bs"])
                st["n_seeds"] = n_seeds(sub)
                per_cell[f"{arm}@gamma={gamma:g}"] = st
                if arm == "relational" and not math.isnan(st["ratio"]) and st["ratio"] >= 2.0:
                    detected = True
        # differential reading: relational vs the prototype control, same gamma
        differential = {}
        for gamma in sorted(rel_bt["gamma"].unique()):
            kr = f"relational@gamma={gamma:g}"
            kc = f"proto_full@gamma={gamma:g}"
            if kr in per_cell and kc in per_cell:
                rr, rc = per_cell[kr]["ratio"], per_cell[kc]["ratio"]
                differential[f"gamma={gamma:g}"] = {
                    "ratio_relational": rr, "ratio_control": rc,
                    "specific_to_the_gram": bool(
                        not math.isnan(rr) and not math.isnan(rc)
                        and rr >= 2.0 and rr >= 2.0 * rc),
                }
        v.update(per_cell=per_cell, differential=differential,
                 coupling_detected=bool(detected), n_seeds=n_seeds(rel_bt),
                 control_arm_present=bool(differential))
        v["verdict"] = "confirmee" if detected else "non concluante"
        worst = max((s["range_pp"] for k, s in per_cell.items()
                     if k.startswith("relational@") and not math.isnan(s["range_pp"])),
                    default=float("nan"))
        v["justification"] = (
            f"at equal samples seen, varying B over {sorted(rel_bt['bs'].unique())} "
            f"moves the relational arm by up to {worst:.2f} accuracy points; "
            f"coupling {'IS' if detected else 'is NOT'} above the seed noise. "
            f"If it is, P1 holds only at fixed B -- which is how the P1 sweep was "
            f"run, but it must be said out loud. "
            + ("The prototype control was run at the same budgets, so the effect "
               "can be read as specific to the Gram statistic or not: "
               + json.dumps(differential, default=str)
               if differential else
               "NO CONTROL ARM IN THIS TABLE: a B effect here cannot be "
               "attributed to the Gram rather than to batch size in general."))
    else:
        v["verdict"] = "non concluante"
        v["justification"] = "the batch study was not run."
    out["P3_batch_coupling"] = v

    # ---- P4: the DINO knobs are load-bearing ------------------------------
    # PER GAMMA.  Pooling proto_full over gamma in {1.0, 0.1, 0.01} and comparing
    # it to an ablation pooled the same way averages three different data
    # regimes; if the arms were not run at exactly the same set of gammas the
    # comparison is not even between the same quantities.  The ablation must
    # hurt at every gamma where both were run.
    v = {"prediction": PREREG["P4_dino_knobs_load_bearing"]}
    full = main[main["arm"] == "proto_full"]
    if len(full):
        per_gamma: Dict[str, dict] = {}
        cell_ok: List[bool] = []
        for gamma in sorted(np.round(full["gamma"].unique(), 6)):
            ref_sub = full[np.isclose(full["gamma"], gamma)]
            ref, _ = _mean_std(ref_sub["probe_acc"].tolist())
            details = {}
            for abl in ("proto_nocenter", "proto_nosharpen"):
                sub = main[(main["arm"] == abl) & np.isclose(main["gamma"], gamma)]
                if not len(sub):
                    continue
                a, _ = _mean_std(sub["probe_acc"].tolist())
                er, _ = _mean_std(sub["eff_rank"].tolist())
                drop_pp = 100.0 * (ref - a)
                hurts = bool(drop_pp >= 2.0 or (not math.isnan(er) and er < 2.0))
                details[abl] = {"acc": a, "drop_pp": drop_pp, "eff_rank": er,
                                "load_bearing": hurts, "n_seeds": n_seeds(sub)}
                cell_ok.append(hurts)
            if details:
                per_gamma[f"gamma={gamma:g}"] = {"proto_full_acc": ref,
                                                 "ablations": details}
        v["per_gamma"] = per_gamma
        if not cell_ok:
            v["verdict"] = "non concluante"
        elif all(cell_ok):
            v["verdict"] = "confirmee"
        else:
            v["verdict"] = "infirmee"
        v["justification"] = "; ".join(
            f"{g} {k}: {d['drop_pp']:+.2f} pp vs proto_full, effective rank "
            f"{d['eff_rank']:.2f}"
            for g, cell in per_gamma.items() for k, d in cell["ablations"].items()
        ) or "ablation arms were not run at any gamma where proto_full was run."
    else:
        v["verdict"] = "non concluante"
        v["justification"] = "proto_full was not run."
    out["P4_dino_knobs_load_bearing"] = v

    # ---- P5: long-tail accuracy (exploratory, no robustness verdict) ------
    v = {"prediction": PREREG["P5_longtail_accuracy"], "verdict": "exploratoire"}
    deg = {}
    for arm in ARM_ORDER:
        sub = main[main["arm"] == arm]
        hi = sub[np.isclose(sub["gamma"], 1.0)]
        lo = sub[np.isclose(sub["gamma"], 0.01)]
        if not len(hi) or not len(lo):
            continue
        t_hi, s_hi = _mean_std(hi["probe_tail_acc"].tolist())
        t_lo, s_lo = _mean_std(lo["probe_tail_acc"].tolist())
        a_hi, _ = _mean_std(hi["probe_acc"].tolist())
        a_lo, _ = _mean_std(lo["probe_acc"].tolist())
        deg[arm] = {"tail_balanced": t_hi, "tail_balanced_sd": s_hi,
                    "tail_longtail": t_lo, "tail_longtail_sd": s_lo,
                    "tail_drop_pp": 100.0 * (t_hi - t_lo),
                    "acc_drop_pp": 100.0 * (a_hi - a_lo)}
    v["per_arm"] = deg
    v["justification"] = ("tail-class accuracy drop from gamma=1.0 to gamma=0.01, "
                          "per arm. Reported WITHOUT a robustness claim: T1 says "
                          "'no uniformity prior', not 'robust to imbalance'.")
    v["caveat"] = (
        "head/tail is the harness's TRAIN-frequency split. At gamma=1.0 the pool "
        "is balanced, so the split is arbitrary (classes are ranked by count with "
        "ties broken by class id -> head = classes 0..C/2). 'tail_drop_pp' "
        "therefore compares an ARBITRARY half at gamma=1.0 with the genuine tail "
        "at gamma=0.01: it is a within-gamma comparison across arms that is "
        "meaningful, not the across-gamma difference. Quote the per-arm numbers "
        "at a FIXED gamma.")
    out["P5_longtail_accuracy"] = v

    # ---- P6: does effective rank predict transfer? ------------------------
    # Pre-registered before this script was first run; see PREREG for the
    # threshold and for why it is not to be adjusted afterwards.
    # P6 and P6b: identical thresholds, two different spaces.  P6b was added on
    # 2026-08-26, before this script's first run; see PREREG.  P6's own logic is
    # unchanged, it is simply run twice with a different rank column.
    for _key, _rank_col in (("P6_rank_does_not_predict_transfer", "eff_rank"),
                            ("P6b_backbone_rank_does_not_predict_transfer", "eff_rank_h")):
        v = {"prediction": PREREG[_key]}
        try:
            import numpy as _np

            def _spearman(a, b):
                """Rank correlation, computed by hand: no scipy in requirements."""
                a, b = _np.asarray(a, float), _np.asarray(b, float)
                if len(a) < 3 or _np.all(a == a[0]) or _np.all(b == b[0]):
                    return float("nan")
                ra = _np.argsort(_np.argsort(a)).astype(float)
                rb = _np.argsort(_np.argsort(b)).astype(float)
                ra -= ra.mean()
                rb -= rb.mean()
                den = _np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
                return float((ra * rb).sum() / den) if den > 0 else float("nan")

            acc_col = "probe_acc_balanced" if "probe_acc_balanced" in main.columns else "probe_acc"
            per_gamma, rhos = {}, []
            if len(main) and _rank_col in main.columns and acc_col in main.columns:
                for g, sub in main.groupby("gamma"):
                    # One point per ARM: average over seeds first, so the correlation
                    # is across arms and not inflated by seed replicates.
                    agg = sub.groupby("arm")[[_rank_col, acc_col]].mean().dropna()
                    if len(agg) >= 4:
                        rho = _spearman(agg[_rank_col].values, agg[acc_col].values)
                        per_gamma[str(g)] = {"spearman": rho, "n_arms": int(len(agg)),
                                             "arms": list(agg.index)}
                        if rho == rho:
                            rhos.append(rho)
            if not rhos:
                v["verdict"] = "non concluante"
                v["evidence"] = {"reason": f"fewer than 4 arms with both {_rank_col} and "
                                           f"{acc_col} at any gamma", "per_gamma": per_gamma}
            else:
                worst_abs = max(abs(r) for r in rhos)
                any_negative = any(r < 0 for r in rhos)
                if worst_abs < 0.4 or any_negative:
                    v["verdict"] = "confirmee"
                elif all(r > 0.7 for r in rhos):
                    v["verdict"] = "infirmee"
                else:
                    v["verdict"] = "non concluante"
                v["evidence"] = {"per_gamma": per_gamma, "max_abs_spearman": worst_abs,
                                 "any_negative": any_negative, "accuracy_column": acc_col,
                                 "rank_column": _rank_col, "n_seeds": n_seeds(main)}
        except Exception as exc:                                   # noqa: BLE001
            # A verdict that cannot be computed must read as missing, never as a pass.
            v["verdict"] = "non concluante"
            v["evidence"] = {"error": f"{type(exc).__name__}: {exc}"}
        out[_key] = v

    return out


def aggregate(cfg: ExpConfig) -> dict:
    import pandas as pd
    if not os.path.exists(cfg.results_csv()):
        raise FileNotFoundError(f"no results at {cfg.results_csv()}; run the study first.")
    df = pd.read_csv(cfg.results_csv())
    numeric = [c for c in RESULT_COLUMNS if c not in ("run_name", "study", "arm", "family")]
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    figs = make_figures(df, cfg)
    verdicts = compute_verdicts(df, cfg)

    per_arm_gamma = {}
    main = df[df["study"] == "main"]
    for arm in ARM_ORDER:
        for gamma in sorted(main["gamma"].unique()) if len(main) else []:
            sub = main[(main["arm"] == arm) & (main["gamma"] == gamma)]
            if not len(sub):
                continue
            m, s = _mean_std(sub["probe_acc"].tolist())
            hm, hs = _mean_std(sub["probe_head_acc"].tolist())
            tm, ts = _mean_std(sub["probe_tail_acc"].tolist())
            per_arm_gamma[f"{arm}@gamma={gamma:g}"] = {
                "acc_mean": m, "acc_sd": s, "head_mean": hm, "head_sd": hs,
                "tail_mean": tm, "tail_sd": ts, "n_seeds": int(sub["seed"].nunique()),
            }

    summary = {
        "experiment": "xp_B_relational_vs_prototype",
        "thesis": "T1 -- a relational target is O(d)-invariant and knob-free where a "
                  "prototype head fixes a frame and carries K / centering / sharpening, "
                  "plus a near-uniform-marginal assumption.",
        "careful_claim": "no uniformity prior -- NOT 'robust to imbalance'.",
        "preregistered_thresholds": PREREG,
        "n_runs": int(len(df)),
        "seeds_per_cell": sorted(int(s) for s in df["seed"].unique()) if len(df) else [],
        "verdicts": verdicts,
        "main_table": per_arm_gamma,
        "figures": figs,
        "scale_limitations": [
            "CIFAR-100 at 32x32 with a ResNet-18 and a few tens of epochs; not ImageNet.",
            "K=4096 on a ~10.9k-image long-tail pool is 2.7 prototypes per image, so the "
            "K axis is partly a 'too many clusters for the data' axis.",
            "absolute accuracies are far below published CIFAR SSL numbers; only the "
            "differences between budget-matched arms are claimed to mean anything.",
            "the P3 null case, if it occurs, is conditional on the linear lr-scaling rule.",
            "no per-arm learning-rate tuning was done, by design; a reviewer who wants "
            "per-arm tuning should ask for it as a follow-up, not read it as done.",
        ],
        "known_confounds": [
            "the relational and InfoNCE arms carry a VICReg term the prototype arms do "
            "not; that is not an oversight, it is thesis T2 (the relational target is "
            "degenerate alone) -- but it means the arms differ in one more thing than "
            "the loss family, and the (var_w, cov_w) sweep is what bounds its effect.",
            "epochs differ across gamma because compute is matched and the pool shrinks.",
        ],
    }
    with open(os.path.join(cfg.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    return summary


# ===========================================================================
# driver
# ===========================================================================

def run_specs(specs: List[RunSpec], cfg: ExpConfig, provider: DataProvider,
              device: torch.device, table: ResultsTable) -> None:
    total = len(specs)
    for i, spec in enumerate(specs, 1):
        print(f"\n=== [{i}/{total}] {spec.name()} ===")
        summary = train_one_run(spec, cfg, provider, device)
        table.add(summary)


def apply_smoke(cfg: ExpConfig) -> ExpConfig:
    """Tiny everything.  Target: under 3 minutes on a CPU, no network.

    The smoke run validates the PIPELINE -- shapes, arms, resume, CSV schema,
    figures, verdict computation -- and nothing about the science.  Every number
    it produces is meaningless.
    """
    cfg.smoke = True
    cfg.synthetic = True
    cfg.steps = 6
    cfg.bs = 16
    cfg.proj_hidden = 64
    cfg.proj_dim = 32
    cfg.proj_layers = 2
    cfg.K = 32
    cfg.proto_freeze_steps = 2
    cfg.eval_every = 3
    cfg.ckpt_every = 3
    cfg.probe_epochs = 3
    cfg.n_diag = 64
    cfg.seeds = (0, 1)
    cfg.workers = 0
    cfg.amp = False
    cfg.prefer_cpu = True
    return cfg


def smoke_grids():
    """Reduced but structurally identical grids (still square, still two
    families, still >= 2 seeds so that std/verdict paths are exercised)."""
    return dict(
        gammas_main=[1.0, 0.01],
        proto_k=[8, 32],
        proto_tau=[0.04, 0.07],
        rel_var=[10.0, 25.0],
        rel_cov=[0.5, 1.0],
        batch_sizes=[8, 16],
        sweep_gammas=[0.01],
        batch_gammas=[1.0],
        batch_arms=BATCH_ARMS,
    )


def real_grids(args) -> dict:
    return dict(
        gammas_main=args.gamma or GAMMAS_MAIN,
        proto_k=PROTO_GRID_K,
        proto_tau=PROTO_GRID_TAU_T,
        rel_var=REL_GRID_VAR_W,
        rel_cov=REL_GRID_COV_W,
        batch_sizes=args.batch_sizes or BATCH_SIZES,
        sweep_gammas=args.sweep_gamma or [0.01],
        # Default to the SAME gamma as the sweep: P3 exists to annotate P1, and
        # an annotation measured in a different data regime annotates nothing.
        # Two arms (relational + prototype control) at one gamma costs exactly
        # what one arm at two gammas used to cost.
        batch_gammas=args.batch_gamma or [0.01],
        batch_arms=args.batch_arm or BATCH_ARMS,
    )


def check_harness_encoder(cfg: ExpConfig) -> None:
    """Exercise the real `make_encoder` once even in smoke mode, so that the
    harness path is not silently skipped by the tiny stand-in encoder."""
    net, dim = make_encoder(cfg.arch, dataset="cifar")
    net.eval()
    with torch.no_grad():
        y = net(torch.zeros(2, 3, cfg.image_size, cfg.image_size))
    assert y.shape == (2, dim), f"make_encoder returned {tuple(y.shape)}, expected (2,{dim})"
    print(f"[check] harness make_encoder({cfg.arch}) -> feat_dim {dim}: OK")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="T1: relational target vs prototype head, under a long tail, "
                    "at controlled batch size.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--outdir", default=os.path.join(_ROOT, "results", "B"))
    p.add_argument("--data-root", default="./data")
    p.add_argument("--study", default="all",
                   choices=["all", "main", "sweep", "batch"],
                   help="which study group to run")
    p.add_argument("--arm", action="append", default=None,
                   choices=list(ARMS.keys()),
                   help="restrict the main study to these arms (repeatable)")
    p.add_argument("--all", action="store_true",
                   help="run every arm (default when --arm is not given)")
    p.add_argument("--seed", action="append", type=int, default=None,
                   help="seed (repeatable); at least 3 are required to decide anything")
    p.add_argument("--gamma", action="append", type=float, default=None,
                   help="long-tail ratio for the main study (repeatable)")
    p.add_argument("--sweep-gamma", action="append", type=float, default=None)
    p.add_argument("--batch-gamma", action="append", type=float, default=None)
    p.add_argument("--batch-sizes", action="append", type=int, default=None)
    p.add_argument("--batch-arm", action="append", default=None,
                   choices=list(ARMS.keys()),
                   help="arms of the B study (repeatable); the default pairs the "
                        "relational arm with a prototype CONTROL, without which a "
                        "B effect cannot be attributed to the Gram statistic")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--K", type=int, default=1024)
    p.add_argument("--tau-t", type=float, default=0.04)
    p.add_argument("--tau-s", type=float, default=0.1)
    p.add_argument("--var-w", type=float, default=25.0)
    p.add_argument("--cov-w", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=2500)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--probe-epochs", type=int, default=100)
    p.add_argument("--workers", type=int, default=-1, help="-1 = auto")
    p.add_argument("--cpu", action="store_true", help="force CPU")
    p.add_argument("--no-amp", action="store_true", help="disable bf16 autocast on CUDA")
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--smoke", action="store_true",
                   help="tiny CPU pipeline test, < 3 min, no network")
    p.add_argument("--synthetic", action="store_true",
                   help="use synthetic images instead of torchvision (offline)")
    p.add_argument("--infonce-symmetric", action="store_true",
                   help="run the InfoNCE control as textbook symmetric SimCLR "
                        "(no EMA target) instead of asymmetric against the EMA")
    p.add_argument("--no-lr-scaling", dest="lr_scaling", action="store_false",
                   default=True, help="do not scale lr linearly with B in the batch study")
    p.add_argument("--aggregate-only", action="store_true",
                   help="skip training, rebuild figures and summary.json from results_long.csv")
    p.add_argument("--dry-run", action="store_true",
                   help="print the run list and the budget tables, train nothing")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    cfg = ExpConfig(
        outdir=args.outdir, data_root=args.data_root, steps=args.steps, bs=args.bs,
        lr=args.lr, proj_dim=args.proj_dim, K=args.K, tau_t=args.tau_t,
        tau_s=args.tau_s, var_w=args.var_w, cov_w=args.cov_w,
        eval_every=args.eval_every, ckpt_every=args.ckpt_every,
        probe_epochs=args.probe_epochs, workers=args.workers,
        prefer_cpu=args.cpu, amp=not args.no_amp, resume=args.resume,
        synthetic=args.synthetic, infonce_symmetric=args.infonce_symmetric,
        lr_scaling=args.lr_scaling,
    )
    if args.seed:
        cfg.seeds = tuple(args.seed)
    if args.smoke:
        cfg = apply_smoke(cfg)
        if args.seed:
            cfg.seeds = tuple(args.seed)
    grids = smoke_grids() if args.smoke else real_grids(args)

    os.makedirs(cfg.runs_dir(), exist_ok=True)
    os.makedirs(cfg.figures_dir(), exist_ok=True)

    if args.aggregate_only:
        s = aggregate(cfg)
        print(json.dumps({k: v.get("verdict") for k, v in s["verdicts"].items()}, indent=2))
        return 0

    arms = args.arm if args.arm else ARM_ORDER
    if args.all:
        arms = ARM_ORDER
    if len(cfg.seeds) < 3 and not args.smoke:
        print(f"WARNING: {len(cfg.seeds)} seed(s). Project rule: at least 3 seeds, "
              f"otherwise a comparison decides nothing.")

    specs: List[RunSpec] = []
    if args.study in ("all", "main"):
        specs += specs_main(cfg, arms, grids["gammas_main"])
    if args.study in ("all", "sweep"):
        specs += specs_sweep(cfg, grids["sweep_gammas"], grids["proto_k"],
                             grids["proto_tau"], grids["rel_var"], grids["rel_cov"])
    if args.study in ("all", "batch"):
        specs += specs_batch(cfg, grids["batch_gammas"], grids["batch_sizes"],
                             grids["batch_arms"])

    for study in ("main", "sweep", "batch"):
        budget_report([s for s in specs if s.study == study], study)

    print(f"\nTOTAL RUNS QUEUED: {len(specs)}")
    if args.dry_run:
        for s in specs:
            print("  " + s.name())
        return 0

    device = get_device(prefer_cpu=cfg.prefer_cpu)
    print(f"device: {device}   amp: {cfg.amp and device.type == 'cuda'}")
    check_harness_encoder(cfg)

    provider = DataProvider(cfg)
    table = ResultsTable(cfg.results_csv())
    t0 = time.time()
    run_specs(specs, cfg, provider, device, table)
    print(f"\nall runs finished in {(time.time() - t0) / 60.0:.1f} min")

    s = aggregate(cfg)
    print("\n================ VERDICTS ================")
    for k, v in s["verdicts"].items():
        print(f"{k:<34} {v.get('verdict','?'):<16} {v.get('justification','')}")
    print(f"\nsummary.json -> {os.path.join(cfg.outdir, 'summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
