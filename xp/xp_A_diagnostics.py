#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment group A -- "cheap decisive diagnostics" for the Uebergang-SSL theses.

Two short experiments, ~25 A100-minutes together, meant to be the OPENING
ARGUMENT of a discussion with an SSL researcher.  Everything here is either a
closed-form fact that the code merely verifies, or a small controlled training
run whose falsifier is written down before it is run.

===============================================================================
A1 -- O(d) INVARIANCE UNIT TEST                                     (thesis T1)
===============================================================================
Thesis
    A relational target G = A A^T is invariant under every orthogonal Q acting
    on the latent space, because (A Q)(A Q)^T = A A^T.  A softmax prototype head
    softmax(<a, c_k>/tau) is not: the prototypes c_k FIX A DISTINGUISHED BASIS.
    "Legislating over the object" is "choosing a frame".

Prediction (exact)
    P1.a  |L(Z) - L(ZQ)| is at machine precision for gram_loss and info_nce,
          for every Q, and SHRINKS BY ~1e-9 WHEN COMPUTED IN float64 -- i.e. the
          residual is numerical, not structural.
    P1.b  |L(Z) - L(ZQ)| is macroscopic (>> machine precision, and comparable to
          the batch-to-batch spread of the loss itself) for prototype_loss, with
          fixed random prototypes AND with prototypes fitted to the batch.
    P1.c  CONTROL (mandatory, the experiment is dishonest without it): rotating
          the prototypes together with the embeddings, protos -> protos Q,
          restores invariance to machine precision.  So the culprit is the
          CHOICE OF A FRAME, not the softmax.
    P1.d  CONTROL: permuting the ROWS of the prototype matrix leaves the loss
          unchanged.  The head is equivariant to relabelling of the K slots;
          what it is not equivariant to is a rotation of the d-dimensional
          space.  This separates "arbitrary indexing" from "arbitrary frame".

What would falsify T1 here
    - a non-shrinking residual for gram_loss / info_nce when moving float32 ->
      float64 (that would mean the invariance is only approximate);
    - a machine-precision residual for prototype_loss with fixed prototypes
      (that would mean prototype heads are frame-free after all);
    - control P1.c failing to restore invariance (that would mean the softmax
      itself, not the frame, breaks the symmetry -- a different thesis).

Honesty about what A1 is
    A1 is a UNIT TEST, not a discovery.  gram_loss and info_nce depend on Z only
    through inner products, so their O(d) invariance is a two-line proof; A1
    verifies the implementation and, more usefully, MEASURES how large the
    prototype violation is on a realistic batch, against the natural scale of
    the loss.  Nobody should be told this is an empirical result about SSL.
    What A1 does buy: it turns "prototypes fix a frame" from a slogan into a
    number, and the P1.c control forecloses the obvious counter-reading.

Cost: ~20 s on CPU, 0 A100-hour.

===============================================================================
A2 -- THE RELATIONAL TARGET COLLAPSES ON ITS OWN                    (thesis T2)
===============================================================================
Thesis
    gram_loss alone is degenerate: if all embeddings are identical then
    S S^T = sg[A A^T] = the all-ones matrix and the loss is EXACTLY 0.  Total
    collapse is a GLOBAL minimum, not a local trap.  Anti-collapse must come
    from the legality term (VICReg variance + covariance), and that term must
    act on the UNNORMALISED z: on the unit sphere each coordinate has std
    bounded by 1/sqrt(d), so the hinge relu(gamma - sigma) with gamma = 1 is
    permanently saturated.

Arms (identical in every respect except the regulariser; matched_budget_check
is called before anything runs)
    gram                  agreement only
    gram_vicreg           agreement + vicreg_reg(z)                <- correct
    gram_vicreg_sphere    agreement + vicreg on L2-NORMALISED z    <- the warning
    gram_vicreg_sphere_gamma   (optional, --all) same but with gamma = 1/sqrt(d),
                          i.e. the fix a defender of the sphere would propose.
                          Included so the argument is not a straw man.

Prediction (exact)
    P2.1  arm `gram`: the agreement loss goes to ~0 WHILE the effective rank of
          z falls towards ~1, and the k-NN accuracy falls to chance.  Loss and
          representation quality move in OPPOSITE directions -- that is the
          whole point.
    P2.2  arm `gram_vicreg`: no collapse.  Effective rank stays of the order of
          its value at initialisation and k-NN stays well above chance.
    P2.3a arm `gram_vicreg_sphere`: the variance hinge is SATURATED AT EVERY
          LOGGED STEP -- the fraction of coordinates with sigma < gamma is 1.0
          throughout and the mean per-coordinate std is <= 1/sqrt(d).  The term
          can never be satisfied, so it stops being a hinge and becomes a
          constant pressure that carries no information about whether the
          representation has collapsed.
    P2.3b (weaker, reported separately and honestly) does that arm actually
          collapse?  Note the hinge's gradient does NOT vanish -- relu(gamma-s)
          has slope -1 wherever it is active -- and the covariance term still
          operates on the sphere.  So `gram_vicreg_sphere` may well avoid TOTAL
          collapse while being clearly worse than `gram_vicreg`.  The project's
          claim that survives either way is P2.3a: the hinge no longer functions
          as a hinge.  P2.3b is measured, not assumed.  ** This is an amendment
          to the plan as given: writing "vicreg on normalised z does not save
          you" as a flat prediction would have been over-claimed. **

What would falsify T2 here
    - arm `gram` keeping a high effective rank and above-chance k-NN after the
      loss has gone to ~0 (no collapse => the degeneracy is not reachable by
      SGD, and the objection "collapse is only a theoretical global minimum"
      would be right);
    - arm `gram_vicreg` collapsing (the legality term would not be what
      prevents collapse);
    - hinge saturation fraction < 1.0 in arm `gram_vicreg_sphere` (P2.3a dead).

Deliberately NOT done in A2
    No EMA teacher, no predictor, no asymmetric stop-gradient beyond the one
    gram_loss already applies to its target.  A BYOL/SimSiam-style architectural
    asymmetry is a different anti-collapse mechanism; adding it would test a
    different claim.  The claim under test is about the OBJECTIVE: it does not
    forbid collapse.

Scale caveat -- read before quoting any number
    This is CIFAR-10, a ResNet-18 and ~1.5k optimisation steps, not ImageNet and
    not a 100-epoch schedule.  What that scale CAN decide: whether the pure
    relational objective drives the representation to a degenerate solution at
    all, and whether the hinge is saturated on the sphere (that one is
    arithmetic and scale-free).  What it CANNOT decide: the size of any accuracy
    gap between well-regularised methods, whether the same dynamics hold for
    long schedules or large models, or anything about downstream transfer.

Cost: 4 arms x 3 seeds x 1500 steps at batch 256, ResNet-18 on 32x32 with bf16
autocast.  Measured ~1.5-3 min per run on an A100 40GB (augmentation-bound as
much as GPU-bound), i.e. ~0.25 A100-hour for the 3 core arms x 3 seeds and
~0.35 A100-hour with the optional fourth arm.  The real elapsed time of every
run is written into its own summary.json, so the estimate above can be checked
rather than believed.

===============================================================================
USAGE
    python xp/xp_A_diagnostics.py --smoke          # < 3 min, CPU, offline
    python xp/xp_A_diagnostics.py --exp a1
    python xp/xp_A_diagnostics.py --all            # A1 + A2, 4 arms, 3 seeds
    python xp/xp_A_diagnostics.py --exp a2 --arm gram --arm gram_vicreg
Outputs (default <repo>/results/A):
    A1_invariance.csv / .png / .summary.json
    A2_<arm>_seed<k>.csv / .ckpt.pt / .summary.json
    A2_curves.png, A2_final_bars.png
    A_summary.json      <- one verdict per prediction: confirmee / infirmee /
                           non concluante, with the number that justifies it
Every run is resumable: CSV rows are flushed and fsync'd as they are produced,
checkpoints are atomic, and a run whose summary.json already exists is skipped
(so a killed Colab session is restarted with the same command line).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- in-house imports: the harness and nothing else ------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.harness import (  # noqa: E402
    AugCfg,
    DATASET_NUM_CLASSES,
    Run,
    SSLDataset,
    aggregate_seeds,
    effective_rank,
    get_device,
    get_eval_datasets,
    get_ssl_dataset,
    gram_loss,
    info_nce,
    knn_probe,
    linear_probe,
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
from lib.harness import _extract_features, _l2, _rng_island  # noqa: E402  (z-scored kNN, A2V3)

DEFAULT_OUTDIR = os.path.join(_REPO_ROOT, "results", "A")
DEFAULT_DATA_ROOT = os.path.join(_REPO_ROOT, "data")

VERDICT_CONFIRMED = "confirmee"
VERDICT_REFUTED = "infirmee"
VERDICT_INCONCLUSIVE = "non concluante"


# ===========================================================================
# shared utilities
# ===========================================================================

def random_orthogonal(d: int, generator: torch.Generator, dtype: torch.dtype) -> torch.Tensor:
    """Draw Q from the orthogonal group O(d) via QR of a Gaussian matrix.

    The sign correction Q <- Q * sign(diag(R)) makes the draw Haar-uniform
    (without it, QR's sign convention biases the distribution).  The caller is
    expected to check Q Q^T = I; ``orthogonality_error`` below does that.
    """
    g = torch.randn(d, d, generator=generator, dtype=torch.float64)
    Q, R = torch.linalg.qr(g)
    Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
    return Q.to(dtype)


def orthogonality_error(Q: torch.Tensor) -> float:
    """max |Q Q^T - I|, in float64 regardless of Q's dtype."""
    Q64 = Q.double()
    return float((Q64 @ Q64.t() - torch.eye(Q.shape[0], dtype=torch.float64)).abs().max())


class SyntheticShapes(Dataset):
    """Tiny offline stand-in for a real image dataset, used ONLY by --smoke.

    Class-dependent coloured rectangle on a noisy background, returned as PIL so
    that it goes through exactly the same torchvision transform pipeline as
    CIFAR-10.  It exists so that --smoke needs no network: if the real dataset
    is already downloaded, --smoke uses the real one instead (see
    ``build_ssl_data``) and this class is never touched.
    """

    def __init__(self, n: int = 192, size: int = 32, num_classes: int = 4,
                 seed: int = 0, transform=None):
        from PIL import Image  # torchvision already depends on Pillow
        self._Image = Image
        rng = np.random.RandomState(seed)
        self.transform = transform
        self.num_classes = int(num_classes)
        self.targets = rng.randint(0, num_classes, size=int(n)).astype(np.int64)
        imgs = rng.randint(60, 120, size=(int(n), int(size), int(size), 3)).astype(np.uint8)
        for i, c in enumerate(self.targets):
            x0 = 2 + (int(c) % 2) * (size // 2 - 2)
            y0 = 2 + (int(c) // 2) * (size // 2 - 2)
            colour = np.array([(int(c) * 67) % 256, (int(c) * 131) % 256,
                               (int(c) * 197) % 256], dtype=np.uint8)
            imgs[i, y0:y0 + size // 3, x0:x0 + size // 3, :] = colour
        self.images = imgs

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i: int):
        img = self._Image.fromarray(self.images[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.targets[i])


def real_dataset_present(name: str, root: str) -> bool:
    """True if ``name`` is already on disk, so --smoke can use it without any
    network access.  Conservative: unknown layouts return False."""
    markers = {
        "cifar10": ["cifar-10-batches-py"],
        "cifar100": ["cifar-100-python"],
        "stl10": ["stl10_binary"],
        "stl10_unlabeled": ["stl10_binary"],
        "mnist": ["MNIST"],
        "fashionmnist": ["FashionMNIST"],
    }
    for m in markers.get(str(name).lower(), []):
        if os.path.isdir(os.path.join(root, m)):
            return True
    return False


def build_ssl_data(args) -> Tuple[Dataset, int, str]:
    """Return ``(two_view_dataset, num_classes, source_tag)``.

    ``source_tag`` is 'real' or 'synthetic' and is printed loudly and written
    into every summary, because a smoke run on synthetic data proves that the
    pipeline runs, not that the thesis holds.
    """
    cfg = AugCfg.strong(size=args.image_size)
    if args.synthetic:
        base = SyntheticShapes(n=args.synthetic_train, size=args.image_size,
                               num_classes=args.synthetic_classes, seed=0)
        ds = SSLDataset(base=base, two_view=make_two_view_transform(cfg),
                        indices=None, labels=base.targets.copy(), name="synthetic")
        return ds, args.synthetic_classes, "synthetic"

    ds = get_ssl_dataset(args.dataset, cfg, root=args.data_root, seed=args.seed)
    if args.train_subset is not None and args.train_subset < len(ds):
        keep = np.random.RandomState(0).permutation(len(ds))[:args.train_subset]
        ds = SSLDataset(base=ds.base, two_view=ds.two_view,
                        indices=sorted(int(i) for i in keep), labels=None,
                        name=ds.name)
    return ds, DATASET_NUM_CLASSES[args.dataset], "real"


def _subset(ds: Dataset, n: Optional[int], seed: int) -> Dataset:
    """Deterministic random subset (used by --eval-subset / --smoke).

    Uses its OWN numpy RandomState rather than the global RNG, so that changing
    the probe size cannot shift the training stream of any arm.
    """
    if n is None or n >= len(ds):
        return ds
    idx = np.random.RandomState(seed).permutation(len(ds))[:int(n)]
    return torch.utils.data.Subset(ds, sorted(int(i) for i in idx))


def build_eval_data(args) -> Tuple[Dataset, Dataset, float]:
    """Return ``(train_ds, test_ds, chance_level)`` for the final k-NN probe.

    ``--eval-subset`` caps both splits.  --smoke sets it: without it, a smoke
    run on an already-downloaded CIFAR-10 would push 60 000 images through a
    ResNet-18 on CPU and blow the three-minute budget by an order of magnitude.
    The two splits are the dataset's official train/test splits, so the probe
    never sees a test image at fit time.
    """
    if args.synthetic:
        tf = make_eval_transform(args.image_size)
        tr = SyntheticShapes(n=args.synthetic_train, size=args.image_size,
                             num_classes=args.synthetic_classes, seed=0, transform=tf)
        te = SyntheticShapes(n=max(32, args.synthetic_train // 2), size=args.image_size,
                             num_classes=args.synthetic_classes, seed=1, transform=tf)
        return (_subset(tr, args.eval_subset, 11), _subset(te, args.eval_subset, 12),
                1.0 / args.synthetic_classes)
    tr, te = get_eval_datasets(args.dataset, args.image_size, root=args.data_root)
    return (_subset(tr, args.eval_subset, 11), _subset(te, args.eval_subset, 12),
            1.0 / DATASET_NUM_CLASSES[args.dataset])


# ---------------------------------------------------------------------------
# The forbidden variant of the legality term, kept local and clearly labelled.
# ---------------------------------------------------------------------------

def vicreg_on_unit_sphere(z_normalised: torch.Tensor, var_w: float = 25.0,
                          cov_w: float = 1.0, gamma: float = 1.0
                          ) -> Tuple[torch.Tensor, dict]:
    """VICReg variance+covariance computed on an L2-NORMALISED z ON PURPOSE.

    ``harness.vicreg_reg`` refuses normalised input -- correctly, that guard is
    the project's own warning.  This function is a verbatim copy of the harness
    arithmetic with the guard removed, and it exists for one reason: arm
    `gram_vicreg_sphere` has to actually make the mistake in order to show what
    the mistake does.  ``selfcheck_local_vicreg()`` asserts that this copy
    reproduces ``harness.vicreg_reg`` exactly on unnormalised input, so the
    comparison between the two arms is not confounded by an implementation
    difference.

    The returned dict additionally carries ``hinge_active_frac``: the fraction
    of coordinates whose std is below ``gamma``, i.e. the fraction of the hinge
    that is still switched on.  A value pinned at 1.0 for the whole run is the
    measurement behind prediction P2.3a.
    """
    if z_normalised.dim() != 2:
        raise ValueError(f"expected [B,d], got {tuple(z_normalised.shape)}")
    B, d = z_normalised.shape
    if B < 2:
        raise ValueError("need B >= 2 to estimate a variance")

    zc = z_normalised - z_normalised.mean(dim=0, keepdim=True)
    std = torch.sqrt(zc.var(dim=0, unbiased=True) + 1e-4)
    var_term = F.relu(float(gamma) - std).mean()

    cov = (zc.t() @ zc) / (B - 1)
    off = cov - torch.diag_embed(torch.diagonal(cov))
    cov_term = off.pow(2).sum() / d

    loss = float(var_w) * var_term + float(cov_w) * cov_term
    stats = {
        "var": float(var_term.detach()),
        "cov": float(cov_term.detach()),
        "hinge_active_frac": float((std.detach() < float(gamma)).float().mean()),
        "mean_std": float(std.detach().mean()),
    }
    return loss, stats


def selfcheck_local_vicreg(verbose: bool = True) -> float:
    """Assert the local copy == harness.vicreg_reg on UNNORMALISED input."""
    g = torch.Generator().manual_seed(1234)
    z = torch.randn(64, 32, generator=g) * 2.0 + 0.3
    ref, _ = vicreg_reg(z, var_w=25.0, cov_w=1.0, gamma=1.0)
    mine, _ = vicreg_on_unit_sphere(z, var_w=25.0, cov_w=1.0, gamma=1.0)
    delta = float((ref - mine).abs())
    if delta > 1e-6:
        raise AssertionError(
            f"local vicreg copy disagrees with harness.vicreg_reg by {delta:.3e}; "
            f"arm gram_vicreg_sphere would not be comparable to gram_vicreg."
        )
    if verbose:
        print(f"[check] local vicreg copy == harness.vicreg_reg (delta={delta:.2e})")
    return delta


# ===========================================================================
# A1 -- O(d) invariance unit test
# ===========================================================================

@dataclass
class A1Record:
    objective: str
    transform: str
    # NOTE: this field is deliberately NOT called "dtype".  It becomes a column
    # of a pandas DataFrame, and ``Series.dtype`` is a real attribute of pandas
    # Series, so ``row.dtype`` inside ``df.iterrows()`` silently returns the
    # numpy dtype of the row instead of the column.  That bug crashed the first
    # run of this script; renaming the column removes the trap for good.
    precision: str
    batch: int
    rotation: int
    loss_reference: float
    loss_transformed: float
    abs_delta: float
    rel_delta: float
    ortho_error: float


def fit_prototypes(z1: torch.Tensor, z2: torch.Tensor, n_protos: int, steps: int,
                   lr: float, tau_s: float, tau_t: float, seed: int) -> torch.Tensor:
    """Fit prototypes to a batch by gradient descent on ``prototype_loss``.

    This is the "learned prototypes" condition of P1.b: a head that has been
    trained carries a frame just as much as a random one -- arguably more, since
    training aligns the c_k with directions of the data.  Returned detached.
    """
    g = torch.Generator().manual_seed(seed)
    protos = torch.randn(n_protos, z1.shape[1], generator=g,
                         dtype=z1.dtype).requires_grad_(True)
    opt = torch.optim.Adam([protos], lr=lr)
    center = None
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        loss, center = prototype_loss(z1, z2, protos, tau_s=tau_s, tau_t=tau_t,
                                      center=center, use_center=True, use_sharpen=True)
        loss.backward()
        opt.step()
    return protos.detach().clone()


@torch.no_grad()
def a1_measure_batch(z1: torch.Tensor, z2: torch.Tensor, protos_fixed: torch.Tensor,
                     protos_learned: torch.Tensor, Q: torch.Tensor,
                     perm: torch.Tensor,
                     tau_s: float, tau_t: float) -> List[Tuple[str, str, float, float]]:
    """All six measurements on one (batch, rotation) pair.

    Returns a list of ``(objective, transform, loss_reference, loss_transformed)``.
    Every entry compares the SAME loss evaluated twice: once on (z1, z2, protos)
    and once on the transformed quantities.  Nothing else changes.
    """
    z1r, z2r = z1 @ Q, z2 @ Q
    out: List[Tuple[str, str, float, float]] = []

    # (a) relational target: invariant by construction (G = A A^T).
    out.append(("gram_loss", "rotate embeddings",
                float(gram_loss(z1, z2)), float(gram_loss(z1r, z2r))))

    # (b) contrastive: also depends on Z only through inner products.
    out.append(("info_nce", "rotate embeddings",
                float(info_nce(z1, z2)), float(info_nce(z1r, z2r))))

    def proto(zz1, zz2, pp):
        loss, _ = prototype_loss(zz1, zz2, pp, tau_s=tau_s, tau_t=tau_t,
                                 center=None, use_center=True, use_sharpen=True)
        return float(loss)

    # (c) prototype head, prototypes left in place -> the frame is broken.
    out.append(("prototype_fixed", "rotate embeddings",
                proto(z1, z2, protos_fixed), proto(z1r, z2r, protos_fixed)))
    out.append(("prototype_learned", "rotate embeddings",
                proto(z1, z2, protos_learned), proto(z1r, z2r, protos_learned)))

    # (P1.c) CONTROL: carry the prototypes along with the rotation.  If this is
    # invariant, the softmax is innocent and the frame is guilty.
    out.append(("prototype_learned_CONTROL_rotate_protos",
                "rotate embeddings AND prototypes",
                proto(z1, z2, protos_learned), proto(z1r, z2r, protos_learned @ Q)))

    # (P1.d) CONTROL: permute the K prototype rows.  Softmax is equivariant to
    # relabelling the slots, so this must be exactly invariant -- which shows
    # the broken symmetry is over the d-dimensional space, not over the K slots.
    # ``perm`` is drawn by the caller from the batch's own torch.Generator, not
    # from the global RNG: an in-function torch.randperm() would consume global
    # state and make the measurement depend on how often the loop is entered.
    out.append(("prototype_learned_CONTROL_permute_protos",
                "permute prototype rows",
                proto(z1, z2, protos_learned), proto(z1, z2, protos_learned[perm])))
    return out


def run_a1(args) -> dict:
    """Run A1 and return the numbers the verdicts are computed from."""
    print("\n" + "=" * 78)
    print("A1 -- O(d) invariance unit test (thesis T1)")
    print("=" * 78)

    name = "A1_invariance"
    config = {
        "experiment": "A1", "batch_size": args.a1_batch, "dim": args.a1_dim,
        "n_protos": args.a1_protos, "n_batches": args.a1_batches,
        "n_rotations": args.a1_rotations, "tau_s": args.a1_tau_s,
        "tau_t": args.a1_tau_t, "proto_fit_steps": args.a1_fit_steps,
        "seed": args.seed,
    }
    run = Run(name=name, outdir=args.outdir, config=config, resume=False,
              higher_is_better=False)

    # No matched_budget_check here, on purpose.  A1 has no encoder, so any
    # ``forward_passes`` figure would be invented rather than measured, and an
    # invented budget table is worse than none.  The budget is instead ASSERTED
    # FROM THE RECORDS below: every objective must have been evaluated the same
    # number of times, on the same batches, at each precision.

    d, B, K = args.a1_dim, args.a1_batch, args.a1_protos

    # ---- master quantities, drawn ONCE in float64 per batch ---------------
    # P1.a asks whether the residual SHRINKS when the same computation is
    # redone in float64.  That question only has meaning if float32 and float64
    # see the SAME batch, the SAME prototypes and the SAME rotations.  Drawing
    # them inside the dtype loop (as the first draft did) silently compared two
    # different batches -- torch.randn consumes a different number of raw RNG
    # words for float32 and float64 -- so the "shrinkage" would have mixed a
    # precision effect with a sampling effect.  Everything below is generated
    # in float64 and merely CAST.
    masters = []
    for b in range(args.a1_batches):
        set_seed(args.seed + b)
        g = torch.Generator().manual_seed(args.seed * 1_000_003 + b)

        # Two views of the same B samples: correlated, not independent, so
        # the losses sit in a realistic regime instead of a degenerate one.
        base = torch.randn(B, d, generator=g, dtype=torch.float64)
        z1 = base + 0.5 * torch.randn(B, d, generator=g, dtype=torch.float64)
        z2 = base + 0.5 * torch.randn(B, d, generator=g, dtype=torch.float64)

        protos_fixed = torch.randn(K, d, generator=g, dtype=torch.float64)
        # Distinct seed stream: with the old ``args.seed * 7 + b`` and the
        # default seed 0 the prototype generator and the data generator were
        # seeded IDENTICALLY, so the first B*d prototype entries were literally
        # the data.  That correlation had no business being in P1.b.
        protos_learned = fit_prototypes(z1, z2, K, args.a1_fit_steps,
                                        lr=args.a1_fit_lr, tau_s=args.a1_tau_s,
                                        tau_t=args.a1_tau_t,
                                        seed=args.seed * 7_919 + 104_729 + b)
        rots = [random_orthogonal(d, g, torch.float64) for _ in range(args.a1_rotations)]
        perms = [torch.randperm(K, generator=g) for _ in range(args.a1_rotations)]
        masters.append((z1, z2, protos_fixed, protos_learned, rots, perms))

    records: List[A1Record] = []
    row = 0
    for dtype, dtype_name in ((torch.float32, "float32"), (torch.float64, "float64")):
        for b in range(args.a1_batches):
            z1_64, z2_64, pf_64, pl_64, rots, perms = masters[b]
            z1, z2 = z1_64.to(dtype), z2_64.to(dtype)
            protos_fixed, protos_learned = pf_64.to(dtype), pl_64.to(dtype)

            for r in range(args.a1_rotations):
                Q = rots[r].to(dtype)
                oe = orthogonality_error(Q)
                tol = 1e-10 if dtype is torch.float64 else 1e-4
                if oe > tol:
                    raise AssertionError(f"Q is not orthogonal: max|QQ^T - I| = {oe:.3e}")

                for obj, tr, l_ref, l_new in a1_measure_batch(
                        z1, z2, protos_fixed, protos_learned, Q, perms[r],
                        args.a1_tau_s, args.a1_tau_t):
                    absd = abs(l_ref - l_new)
                    rec = A1Record(objective=obj, transform=tr, precision=dtype_name,
                                   batch=b, rotation=r, loss_reference=l_ref,
                                   loss_transformed=l_new, abs_delta=absd,
                                   rel_delta=absd / max(abs(l_ref), 1e-300),
                                   ortho_error=oe)
                    records.append(rec)
                    run.log(step=row, objective=obj, transform=tr,
                            precision=dtype_name,
                            batch=b, rotation=r, loss_reference=l_ref,
                            loss_transformed=l_new, abs_delta=absd,
                            rel_delta=rec.rel_delta, ortho_error=oe)
                    row += 1

    # ---- aggregate -------------------------------------------------------
    import pandas as pd
    df = pd.DataFrame([r.__dict__ for r in records])

    # MEASURED budget check (replaces the fabricated matched_budget_check):
    # every objective must have been evaluated the same number of times, at
    # each precision, on the same set of batches.
    counts = df.groupby(["objective", "precision"]).size()
    if counts.nunique() != 1:
        raise AssertionError(
            "A1 budget is not matched across objectives: "
            f"{counts.to_dict()}"
        )
    per_obj_batches = df.groupby("objective")["batch"].apply(lambda s: tuple(sorted(set(s))))
    if per_obj_batches.nunique() != 1:
        raise AssertionError("A1 objectives were not evaluated on the same batches")
    print(f"[A1] measured budget: {int(counts.iloc[0])} evaluations per objective "
          f"per precision, on batches {per_obj_batches.iloc[0]} -- identical for all "
          f"{counts.index.get_level_values(0).nunique()} objectives.")

    agg = (df.groupby(["objective", "precision"])
             .agg(mean_abs_delta=("abs_delta", "mean"),
                  max_abs_delta=("abs_delta", "max"),
                  mean_rel_delta=("rel_delta", "mean"),
                  mean_loss=("loss_reference", "mean"))
             .reset_index())

    # Natural scale of the loss: how much it moves between INDEPENDENT batches.
    # A rotation-induced delta must be compared to this, not to zero.
    scale = (df[df["precision"] == "float64"]
             .groupby(["objective", "batch"])["loss_reference"].first()
             .groupby("objective").std().rename("loss_std_across_batches").reset_index())
    agg = agg.merge(scale, on="objective", how="left")
    agg["delta_over_batch_scale"] = agg["mean_abs_delta"] / agg["loss_std_across_batches"]

    print("\n" + "-" * 110)
    print("A1 RESULTS -- |L(Z) - L(transform(Z))|, averaged over "
          f"{args.a1_batches} batches x {args.a1_rotations} rotations")
    print("-" * 110)
    hdr = (f"{'objective':46} {'precision':10} {'mean|dL|':>12} {'max|dL|':>12} "
           f"{'mean L':>10} {'|dL|/L':>10} {'|dL|/batch_std':>15}")
    print(hdr)
    print("-" * 110)
    for _, r in agg.iterrows():
        print(f"{str(r['objective']):46} {str(r['precision']):10} "
              f"{r['mean_abs_delta']:12.3e} "
              f"{r['max_abs_delta']:12.3e} {r['mean_loss']:10.4f} "
              f"{r['mean_rel_delta']:10.3e} "
              f"{r['delta_over_batch_scale']:15.3e}")
    print("-" * 110)

    agg.to_csv(os.path.join(args.outdir, "A1_summary_table.csv"), index=False)
    plot_a1(agg, args.outdir)

    def pick(obj: str, dt: str, col: str = "mean_abs_delta") -> float:
        sel = agg[(agg["objective"] == obj) & (agg["precision"] == dt)]
        return float(sel[col].iloc[0]) if len(sel) else float("nan")

    result = {
        "gram_f32": pick("gram_loss", "float32"),
        "gram_f64": pick("gram_loss", "float64"),
        "infonce_f32": pick("info_nce", "float32"),
        "infonce_f64": pick("info_nce", "float64"),
        "proto_fixed_f64": pick("prototype_fixed", "float64"),
        "proto_fixed_f64_rel": pick("prototype_fixed", "float64", "mean_rel_delta"),
        "proto_fixed_f64_over_scale": pick("prototype_fixed", "float64",
                                           "delta_over_batch_scale"),
        "proto_learned_f64": pick("prototype_learned", "float64"),
        "proto_learned_f64_rel": pick("prototype_learned", "float64", "mean_rel_delta"),
        "control_rotate_protos_f64": pick("prototype_learned_CONTROL_rotate_protos",
                                          "float64"),
        "control_permute_protos_f64": pick("prototype_learned_CONTROL_permute_protos",
                                           "float64"),
        "max_ortho_error": float(df["ortho_error"].max()),
        "n_evaluations_per_objective_per_precision": int(counts.iloc[0]),
    }
    run.finish(result)
    return result


def plot_a1(agg, outdir: str) -> str:
    """Grouped log-scale bar chart, readable in black and white."""
    objectives = ["gram_loss", "info_nce", "prototype_fixed", "prototype_learned",
                  "prototype_learned_CONTROL_rotate_protos",
                  "prototype_learned_CONTROL_permute_protos"]
    labels = ["gram_loss\n(relational)", "info_nce\n(contrastive)",
              "prototype\n(fixed c_k)", "prototype\n(fitted c_k)",
              "CONTROL\nrotate c_k too", "CONTROL\npermute c_k rows"]
    floor = 1e-18  # so that an exact 0.0 is still drawable on a log axis

    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(objectives))
    width = 0.38
    styles = [("float32", "0.75", "//"), ("float64", "0.35", "")]
    for i, (dt, colour, hatch) in enumerate(styles):
        vals = []
        for obj in objectives:
            sel = agg[(agg["objective"] == obj) & (agg["precision"] == dt)]
            v = float(sel["mean_abs_delta"].iloc[0]) if len(sel) else np.nan
            vals.append(max(v, floor))
        ax.bar(x + (i - 0.5) * width, vals, width, label=dt, color=colour,
               edgecolor="black", hatch=hatch)

    ax.axhline(np.finfo(np.float32).eps, color="black", linestyle="--", linewidth=1)
    ax.text(len(objectives) - 0.45, np.finfo(np.float32).eps * 1.3,
            "float32 eps", ha="right", fontsize=8)
    ax.axhline(np.finfo(np.float64).eps, color="black", linestyle=":", linewidth=1)
    ax.text(len(objectives) - 0.45, np.finfo(np.float64).eps * 1.3,
            "float64 eps", ha="right", fontsize=8)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("mean |L(Z) - L(transform(Z))|")
    ax.set_title("A1 (T1): who is invariant under an orthogonal change of latent frame?\n"
                 "bars at machine precision = invariant; the two CONTROLs isolate the "
                 "frame from the softmax")
    ax.legend(title="precision", loc="upper left")
    ax.grid(axis="y", linestyle=":", linewidth=0.6)
    fig.tight_layout()
    path = os.path.join(outdir, "A1_invariance.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[A1] figure -> {path}")
    return path


# ===========================================================================
# A2 -- collapse of the relational target
# ===========================================================================

A2_ARMS = {
    # name -> (regulariser kind, gamma policy)
    "gram": ("none", None),
    "gram_vicreg": ("unnormalised", 1.0),
    # Same loss, same weights, but the legality term's gradient NEVER reaches
    # the backbone: vicreg is applied to projector(sg[h]).  The projector still
    # trains under both terms; the trunk trains under the agreement term only.
    # This is the F2 stop-grad idiom transposed to A2: it separates "the term's
    # gradient into the trunk does the damage" from "the reshaped projector
    # changes what the agreement term asks of the trunk".
    "gram_vicreg_sg": ("unnormalised_sg", 1.0),
    "gram_vicreg_sphere": ("sphere", 1.0),
    "gram_vicreg_sphere_gamma": ("sphere", "inv_sqrt_d"),
    "gram_vicreg_sphere_gamma_half": ("sphere", "half_inv_sqrt_d"),
}
A2_CORE_ARMS = ["gram", "gram_vicreg", "gram_vicreg_sphere"]
A2_V2_ARMS = ["gram", "gram_vicreg", "gram_vicreg_sg",
              "gram_vicreg_sphere", "gram_vicreg_sphere_gamma"]

# --- A2V3 arms (PREREG_A2V3, registered 2026-09-21) -------------------------
# All five run the sphere regulariser with gamma = 1, a hinge that can never be
# satisfied (the std ceiling on the sphere is 1/sqrt(d)), and differ ONLY in how
# much of the VARIANCE term reaches the loss.  The covariance term is untouched
# in every arm, exactly as in the three legacy sphere arms.
A2_ARMS.update({
    "gram_vicreg_sphere_dose51": ("sphere", 1.0),
    "gram_vicreg_sphere_dose11": ("sphere", 1.0),
    "gram_vicreg_sphere_interm51": ("sphere", 1.0),
    "gram_vicreg_sphere_interm11": ("sphere", 1.0),
    "gram_vicreg_sphere_early250": ("sphere", 1.0),
})
A2_VAR_SCHEDULE = {
    # arm -> (kind, value)
    #   dose   : variance weight multiplied by `value` at every step
    #   interm : variance term applied on a seeded Bernoulli(value) subset of the
    #            steps and dropped on the others
    #   early  : variance term applied for the first `value` steps, then dropped
    "gram_vicreg_sphere_dose51": ("dose", 0.51),
    "gram_vicreg_sphere_dose11": ("dose", 0.11),
    "gram_vicreg_sphere_interm51": ("interm", 0.51),
    "gram_vicreg_sphere_interm11": ("interm", 0.11),
    "gram_vicreg_sphere_early250": ("early", 250),
}
A2_V3_ARMS = ["gram", "gram_vicreg", "gram_vicreg_sphere"] + sorted(A2_VAR_SCHEDULE)

# ---------------------------------------------------------------------------
# Five-arm pre-registration.  Registered 2026-09-05, BEFORE any run of the
# `gram_vicreg_sg` and `gram_vicreg_sphere_gamma` arms.  Thresholds are
# anchored on the 2026-08-26 clean-test measurements of the three legacy arms
# (JOURNAL A-002 and its reanalysis): they use OLD-arm data to set bars for
# NEW arms, which is legitimate; editing them after the five-arm run is not.
#
# WHY THE MEASUREMENT IS NAMED EXPLICITLY: A2's history is three instrument-
# placement failures in a row.  The same rank contrast moved from 0.04 sigma
# (augmented views, N=512, train pool) to 13 sigma (clean test split,
# N=10000) between 2026-08-26 measurements.  A prediction that does not name
# its space, its input distribution and its N is not a prediction.
# ---------------------------------------------------------------------------
PREREG_A2V2 = {
    "measurement": {
        "primary_rank_statistic": "eff_rank_h_clean_centred",
        "definition": "effective rank (Roy-Vetterli, uncentred svdvals of the "
                      "CENTRED matrix) of the backbone features h, model in "
                      "eval(), computed on the FULL clean test split of "
                      "args.dataset with the deterministic eval transform",
        "deconfound_controls": "the same statistic is also logged at N=512 on "
                               "the clean test split (eff_rank_h_clean_c512) "
                               "and at N=512 on the clean train split "
                               "(eff_rank_h_cleantrain_c512), so the "
                               "augmentation, split and N axes of the A-002 "
                               "reversal stay separable in every future run",
        "knn_statistic": "knn_acc, unchanged: DINO-protocol weighted kNN on "
                         "L2-normalised backbone features, k=20, T=0.07",
    },
    "Q1_gradient_leak": {
        "statement": "the -2.29 pp kNN cost of the legality term is carried by "
                     "its gradient into the backbone: routing that gradient to "
                     "the projector only (arm gram_vicreg_sg) recovers the kNN "
                     "of gram while keeping the projector rank high",
        "confirm": "mean knn(gram_vicreg_sg) - mean knn(gram) > -0.8 pp AND "
                   "mean eff_rank_z(gram_vicreg_sg) > 50",
        "falsify": "mean knn(gram_vicreg_sg) - mean knn(gram) < -1.8 pp "
                   "(the drop survives the cut; the damage travels through the "
                   "reshaped projector, not through the gradient)",
        "anchors": "se of a between-arm kNN difference at 3 seeds was ~0.4 pp "
                   "in A-001; -0.8 pp is ~2 se, -1.8 pp is ~4.5 se, and the "
                   "measured drop of gram_vicreg is -2.29 pp at 4.5 sigma",
        "registered": "2026-09-05, before the first run of this arm",
    },
    "Q2_saturated_vs_satisfiable": {
        "statement": "a legality constraint the head can satisfy stops at the "
                     "head, one it cannot satisfy reaches the trunk: with an "
                     "ACHIEVABLE variance target (gamma = 1/sqrt(d), arm "
                     "gram_vicreg_sphere_gamma) the sphere arm's backbone-rank "
                     "gain disappears",
        "confirm": "gain(sphere_gamma) < 30 AND gain(sphere) > 60, where "
                   "gain(a) = mean eff_rank_h_clean_centred(a) - same(gram)",
        "falsify": "gain(sphere_gamma) >= 60 (an achievable target reshapes "
                   "the trunk just as much: the saturation story is wrong)",
        "anchors": "on the 2026-08-26 clean-test measurements, "
                   "gain(sphere) = +94.2 with an se of ~12.5 and "
                   "gain(gram_vicreg) = -178.5; the 30/60 bars sit ~2.4 se "
                   "from the observed values on either side",
        "registered": "2026-09-05, before the first run of this arm",
    },
    "budget_note": "the sg arm adds 2 projector-only forward passes per step "
                   "(a 2-layer MLP, no encoder pass); matched_budget_check "
                   "counts encoder forwards and cannot see this, so it is "
                   "declared here and in every sg summary "
                   "(extra_projector_forwards_per_step). Projector BN running "
                   "stats are updated twice per step in this arm; they are "
                   "only consumed in eval() diagnostics.",
}


# ---------------------------------------------------------------------------
# A2V3 pre-registration.  Registered 2026-09-21, BEFORE any run of the five
# arms of A2_VAR_SCHEDULE and before any z-scored kNN was ever computed.
# Anchors are the A-003 / A-004 measurements (JOURNAL).  The reference arm
# gram_vicreg_sphere is RE-RUN inside the same invocation: two identical runs
# differ by about 1 pp of kNN (A-004, point 5), so no threshold below compares
# a new arm with a number from an older run.
# ---------------------------------------------------------------------------
PREREG_A2V3 = {
    "Q3_pressure_at_fixed_gamma": {
        "context": "in A-004 three sphere arms differ by the variance target "
                   "gamma alone; the hinge ends active on 100%, 51% and 11% of "
                   "the coordinates and kNN is 44.5 / 41.1 / 39.3. A lower "
                   "gamma is reached sooner, so 'gamma' and 'how much variance "
                   "pressure the run received' are one knob. These arms move "
                   "the pressure at FIXED gamma = 1.",
        "statement": "the transfer drop across the legacy sphere arms is carried "
                     "by the amount of variance pressure, not by the value of "
                     "gamma: at gamma = 1, cutting the variance term to 11% "
                     "(by weight, or by applying it on 11% of the steps) "
                     "reproduces the kNN drop, and 51% falls in between",
        "confirm": "mean over the two 11% arms of [knn(sphere) - knn(arm)] >= "
                   "3.0 pp AND knn(sphere) > knn(51%) > knn(11%) inside both "
                   "families (dose, interm)",
        "falsify": "both 11% arms end within 1.5 pp of knn(sphere): at fixed "
                   "gamma the pressure does not reproduce the drop, so gamma "
                   "itself was the lever in A-004",
        "anchors": "A-004: knn(sphere) - knn(sphere_gamma_half) = 5.2 pp; two "
                   "identical runs differ by ~1 pp; 3.0 pp is ~60% of the "
                   "legacy drop and 1.5 pp is within run-to-run noise plus one "
                   "se of a 3-seed difference",
        "exploratory": "dose vs interm at equal p (same integrated pressure, "
                       "different persistence), the early250 arm, and the "
                       "Spearman correlation across sphere-type arms between "
                       "kNN and (i) var_pressure_mean, a label-free quantity "
                       "read on the training trajectory, (ii) the backbone "
                       "rank. Reported, never gated.",
        "not_decided": "a dose-response along the knob that sets the pressure is "
                       "not a label-free PREDICTOR: that needs arms where the "
                       "pressure moves indirectly (lr, batch size, covariance "
                       "weight). One loss family, CIFAR-10, 6000 steps.",
        "registered": "2026-09-21, before the first run of these arms",
    },
    "Q4_probe_normalisation": {
        "context": "A-003: adding the variance/covariance term on unnormalised z "
                   "costs 1.94 pp of kNN and gains 4.11 pp of linear probe on "
                   "the same features. The linear probe standardises features "
                   "with train statistics; the kNN only L2-normalises them. "
                   "Marks et al. (arXiv 2407.12210) report that linear and kNN "
                   "probing agree (r = 0.99) once features are normalised, with "
                   "a z-score before the kNN.",
        "statement": "the sign disagreement is a normalisation artefact: with "
                     "features z-scored per dimension on train statistics before "
                     "the cosine kNN, knn(gram_vicreg) - knn(gram) becomes "
                     "positive, in agreement with the linear probe",
        "confirm": "z-scored kNN difference > +0.5 pp",
        "falsify": "z-scored kNN difference < -1.0 pp (the disagreement survives "
                   "normalisation)",
        "premise": "the raw kNN difference must reproduce below -1.0 pp in the "
                   "same run; otherwise the verdict is 'non concluante'",
        "registered": "2026-09-21, before any z-scored kNN was computed",
    },
    "artefacts": "every run now writes <name>.weights.pt (model state, fp32) so "
                 "that label-free metrics can be recomputed without retraining; "
                 "A-003 and results/A were lost for lack of it",
}


class EncoderProjector(nn.Module):
    """Backbone + projector.  ``forward`` returns ``(h, z)``.

    ``z`` is the UNNORMALISED projector output: gram_loss normalises internally
    and vicreg_reg requires the unnormalised tensor, so nothing here normalises.
    """

    def __init__(self, backbone: nn.Module, feat_dim: int, hidden: int, proj_dim: int):
        super().__init__()
        self.backbone = backbone
        self.projector = make_mlp(feat_dim, hidden, proj_dim, n_layers=2, bn=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return h, self.projector(h)


def cosine_lr(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * float(step + 1) / float(max(1, warmup))
    p = float(step - warmup) / float(max(1, total - warmup))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


def infinite_batches(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def make_fixed_diagnostic_views(ds: Dataset, n: int, seed: int
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Freeze ONE pair of augmented views of ``n`` images, once, for the whole run.

    Diagnostics measured on a fixed pair are comparable across steps, seeds and
    arms; measuring them on the current training batch would mix representation
    drift with batch-to-batch noise.  Built with its own seed before the model
    is created, so it consumes the same RNG in every arm.
    """
    set_seed(seed)
    idx = np.random.RandomState(seed).permutation(len(ds))[:n]
    v1, v2 = [], []
    for i in idx:
        item = ds[int(i)]
        v1.append(item["v1"])
        v2.append(item["v2"])
    return torch.stack(v1), torch.stack(v2)


@torch.no_grad()
def diagnose(model: EncoderProjector, v1: torch.Tensor, v2: torch.Tensor,
             device: torch.device, gamma: float, chunk: int = 256) -> dict:
    """All collapse diagnostics on the frozen diagnostic views.

    The model is put in eval() (BatchNorm running statistics), which is the same
    regime a frozen encoder would be evaluated in.  Everything is float32.
    """
    was_training = model.training
    model.eval()
    H1, Z1, Z2 = [], [], []
    for i in range(0, v1.shape[0], chunk):
        h1, z1 = model(v1[i:i + chunk].to(device))
        _, z2 = model(v2[i:i + chunk].to(device))
        H1.append(h1.float().cpu()); Z1.append(z1.float().cpu()); Z2.append(z2.float().cpu())
    if was_training:
        model.train()
    h1 = torch.cat(H1); z1 = torch.cat(Z1); z2 = torch.cat(Z2)

    n, d = z1.shape
    # A diverged run must be VISIBLE in the CSV, not a LinAlgError from
    # svdvals three frames down.  Non-finite activations are recorded as such
    # and the caller raises; they are never averaged into a verdict.
    if not (torch.isfinite(z1).all() and torch.isfinite(z2).all()
            and torch.isfinite(h1).all()):
        nan = float("nan")
        return {"nonfinite": 1.0, "agree_loss_diag": nan, "eff_rank_z": nan,
                "eff_rank_h": nan, "eff_rank_z_centred": nan,
                "eff_rank_h_centred": nan,
                "mean_std_z": nan, "mean_std_z_normalised": nan,
                "inv_sqrt_d": 1.0 / math.sqrt(d),
                "std_ceiling_normalised": math.sqrt(n / max(1, n - 1)) / math.sqrt(d),
                "n_diag_used": int(n),
                "hinge_active_frac_z": nan, "hinge_active_frac_z_normalised": nan,
                "mean_z_norm": nan, "alignment": nan, "uniformity": nan}

    zn = z1 / z1.norm(dim=1, keepdim=True).clamp_min(1e-8)
    std_z = z1.std(dim=0, unbiased=True)
    std_zn = zn.std(dim=0, unbiased=True)
    # ``uniformity_alignment`` returns (uniformity, alignment) IN THAT ORDER
    # (harness deviation #11).  Verified against lib/harness.py:1355.
    uniformity, alignment = uniformity_alignment(z1, z2)

    # Hard ceiling on the mean per-coordinate std of an L2-NORMALISED cloud.
    # sum_k Var[z_k] <= n/(n-1) * (E||z||^2 - ||zbar||^2) <= n/(n-1) since
    # ||z||=1, and by Jensen mean_k std_k <= sqrt(mean_k Var_k).  The Bessel
    # factor n/(n-1) is NOT negligible: with n=512 it is 1.001 and with the
    # smoke n=32 it is 1.016, so comparing the measured mean std against a bare
    # 1/sqrt(d) with a 1e-6 tolerance would have declared P2.3a REFUTED for
    # purely arithmetic reasons.
    ceiling_unbiased = math.sqrt(n / max(1, n - 1)) / math.sqrt(d)

    return {
        "nonfinite": 0.0,
        "agree_loss_diag": float(gram_loss(z1, z2)),
        "eff_rank_z": effective_rank(z1),
        "eff_rank_h": effective_rank(h1),
        # h is post-ReLU, hence non-negative: its UNCENTRED spectrum is
        # dominated by the shared mean direction (A-002's reanalysis).  Both
        # variants are logged; neither replaces the other.
        "eff_rank_z_centred": effective_rank(z1 - z1.mean(dim=0, keepdim=True)),
        "eff_rank_h_centred": effective_rank(h1 - h1.mean(dim=0, keepdim=True)),
        "mean_std_z": float(std_z.mean()),
        "mean_std_z_normalised": float(std_zn.mean()),
        "inv_sqrt_d": 1.0 / math.sqrt(d),
        "std_ceiling_normalised": ceiling_unbiased,
        "n_diag_used": int(n),
        # fraction of coordinates for which relu(gamma - sigma) is still active
        "hinge_active_frac_z": float((std_z < gamma).float().mean()),
        "hinge_active_frac_z_normalised": float((std_zn < gamma).float().mean()),
        "mean_z_norm": float(z1.norm(dim=1).mean()),
        "alignment": alignment,
        "uniformity": uniformity,
    }


@torch.no_grad()
def _backbone_features_clean(model: EncoderProjector, ds, device: torch.device,
                             n_max: Optional[int] = None, bs: int = 256) -> torch.Tensor:
    """Backbone features on a CLEAN eval dataset, model in eval(), fp32."""
    was_training = model.training
    model.eval()
    n = len(ds) if n_max is None else min(int(n_max), len(ds))
    H = []
    for a in range(0, n, bs):
        xs = []
        for i in range(a, min(a + bs, n)):
            item = ds[i]
            xs.append(item[0] if isinstance(item, (tuple, list)) else item["x"])
        H.append(model.backbone(torch.stack(xs).to(device)).float().cpu())
    if was_training:
        model.train()
    return torch.cat(H)


def _clean_rank_block(model: EncoderProjector, eval_tr, eval_te,
                      device: torch.device) -> Dict[str, float]:
    """The PREREG_A2V2 primary rank statistic plus its de-confound controls.

    A-002's reversal was confounded three ways at once (augmented vs clean,
    N=512 vs N=10000, train pool vs test split).  This block pins two of the
    axes inside every run: the same statistic at N=512 on the test split and
    at N=512 on the clean train split.  The augmented-views value lives in the
    CSV (diagnose logs eff_rank_h on the frozen diagnostic views), so all
    three contrasts are available per run without any post-hoc script.
    """
    Hte = _backbone_features_clean(model, eval_te, device)
    Htr512 = _backbone_features_clean(model, eval_tr, device, n_max=512)

    def cen(H: torch.Tensor) -> torch.Tensor:
        return H - H.mean(dim=0, keepdim=True)

    return {
        "eff_rank_h_clean_centred": effective_rank(cen(Hte)),
        "eff_rank_h_clean_uncentred": effective_rank(Hte),
        "eff_rank_h_clean_c512": effective_rank(cen(Hte[:512])),
        "eff_rank_h_cleantrain_c512": effective_rank(cen(Htr512)),
        "n_clean_eval": float(Hte.shape[0]),
    }


@torch.no_grad()
def _clean_embedding_ranks(model: EncoderProjector, eval_te, device: torch.device,
                           bs: int = 256) -> Dict[str, float]:
    """RankMe-style effective rank of the PROJECTOR output z on the clean test
    split, uncentred (the formula as written in RankMe) and centred, at full N
    and on the first 512 rows (nested, so the two differ by N alone)."""
    was_training = model.training
    model.eval()
    n = len(eval_te)
    Z = []
    for a in range(0, n, bs):
        xs = []
        for i in range(a, min(a + bs, n)):
            item = eval_te[i]
            xs.append(item[0] if isinstance(item, (tuple, list)) else item["x"])
        _h, z = model(torch.stack(xs).to(device))
        Z.append(z.float().cpu())
    if was_training:
        model.train()
    Zt = torch.cat(Z)

    def cen(M: torch.Tensor) -> torch.Tensor:
        return M - M.mean(dim=0, keepdim=True)

    return {
        "eff_rank_z_clean_uncentred": effective_rank(Zt),
        "eff_rank_z_clean_centred": effective_rank(cen(Zt)),
        "eff_rank_z_clean_u512": effective_rank(Zt[:512]),
        "eff_rank_z_clean_c512": effective_rank(cen(Zt[:512])),
        "n_clean_eval_z": float(Zt.shape[0]),
    }


def knn_probe_zscore(encoder, train_ds, test_ds, k: int = 20, device=None) -> float:
    """harness.knn_probe with ONE change: features are z-scored per dimension
    with TRAIN statistics before the L2 normalisation.  Same k, same T = 0.07,
    same cosine vote.  This is the normalisation the linear probe already
    applies, so the two probes finally read the same features (PREREG_A2V3 Q4)."""
    with _rng_island(seed=0):
        return _knn_probe_zscore_impl(encoder, train_ds, test_ds, k, device)


@torch.no_grad()
def _knn_probe_zscore_impl(encoder, train_ds, test_ds, k, device) -> float:
    device = device or get_device()
    Xtr, ytr = _extract_features(encoder, train_ds, device)
    Xte, yte = _extract_features(encoder, test_ds, device)
    num_classes = int(max(ytr.max().item(), yte.max().item())) + 1
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr = _l2((Xtr - mu) / sd).to(device)
    Xte = _l2((Xte - mu) / sd).to(device)
    ytr = ytr.to(device)
    yte = yte.to(device)
    k = int(min(k, Xtr.shape[0]))
    T = 0.07
    correct = 0
    chunk = 1024
    for i in range(0, Xte.shape[0], chunk):
        sim = Xte[i:i + chunk] @ Xtr.t()
        sim_k, idx_k = sim.topk(k, dim=1)
        w = (sim_k / T).exp()
        lab_k = ytr[idx_k]
        scores = torch.zeros(sim.shape[0], num_classes, device=device)
        scores.scatter_add_(1, lab_k, w)
        correct += int((scores.argmax(1) == yte[i:i + chunk]).sum())
    return float(correct) / float(Xte.shape[0])


def train_a2_arm(arm: str, seed: int, args, device: torch.device) -> dict:
    """Train one (arm, seed) and return its final diagnostics.

    Resumable: if ``<name>.summary.json`` exists the run is skipped and its
    summary returned; otherwise training restarts from ``<name>.ckpt.pt``.
    NOTE on resume fidelity: the data stream restarts from the beginning of a
    fresh shuffle rather than being fast-forwarded.  The samples are unlabelled
    and i.i.d.-shuffled, so this does not bias any arm, but a resumed run is not
    bit-identical to an uninterrupted one; it is recorded in the summary.
    """
    name = f"A2_{arm}_seed{seed}"
    summary_path = os.path.join(args.outdir, f"{name}.summary.json")

    reg_kind, gamma_policy = A2_ARMS[arm]
    if gamma_policy == "inv_sqrt_d":
        gamma = 1.0 / math.sqrt(args.proj_dim)
    elif gamma_policy == "half_inv_sqrt_d":
        # A-003 auto-critique: 1/sqrt(d) is the sphere's std CEILING, so the arm
        # named "satisfiable" was in practice near-saturated too. 0.5/sqrt(d) is
        # a target the coordinate std can actually reach, so this is the honest
        # satisfiable-hinge arm Q2 needed.
        gamma = 0.5 / math.sqrt(args.proj_dim)
    elif gamma_policy is not None:
        gamma = float(gamma_policy)
    else:
        gamma = 1.0

    # A2V3: how much of the VARIANCE term reaches the loss at each step.  The
    # mask of the intermittent arms depends on the seed alone, never on the
    # global RNG, so it is identical across resumes.
    sched = A2_VAR_SCHEDULE.get(arm)
    early_cut = 250 if args.steps >= 1000 else max(1, args.steps // 4)
    interm_mask = None
    if sched is not None and sched[0] == "interm":
        interm_mask = (np.random.RandomState(1_000_003 * (seed + 1) + 17)
                       .rand(int(args.steps)) < float(sched[1]))

    def var_scale_at(step: int) -> float:
        if sched is None:
            return 1.0
        kind, val = sched
        if kind == "dose":
            return float(val)
        if kind == "interm":
            return 1.0 if bool(interm_mask[step]) else 0.0
        if kind == "early":
            return 1.0 if step < early_cut else 0.0
        raise ValueError(f"unknown variance schedule {kind!r}")

    # The config is built BEFORE the resume check, because the resume check has
    # to compare it against the stored one.  Skipping a finished run without
    # that comparison silently returns a summary produced with different
    # hyper-parameters -- e.g. re-running with --steps 3000 would have reported
    # the 1500-step numbers as if they were the new ones.
    config = {
        "experiment": "A2", "arm": arm, "seed": seed, "dataset": args.dataset,
        "data_source": ("synthetic" if args.synthetic else "real"),
        "steps": args.steps, "batch_size": args.bs,
        "lr": args.lr, "wd": args.wd, "arch": args.arch, "proj_dim": args.proj_dim,
        "hidden": args.hidden, "gram_w": args.gram_w, "var_w": args.var_w,
        "cov_w": args.cov_w, "gamma": gamma, "regulariser": reg_kind,
        "image_size": args.image_size, "n_diag": args.n_diag,
        "clip": float(args.clip),
        "warmup": args.warmup, "train_subset": args.train_subset,
        "eval_subset": args.eval_subset, "probe": bool(args.probe),
        "amp": bool(args.amp and device.type == "cuda"),
    }
    # Only added for scheduled arms, so legacy summaries keep matching on resume.
    if sched is not None:
        config["var_schedule"] = (f"early:{early_cut}" if sched[0] == "early"
                                  else f"{sched[0]}:{sched[1]}")

    if args.resume and os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            done = json.load(f)
        stored = done.get("config", {})
        diff = {k: (stored.get(k, "<absent>"), v) for k, v in config.items()
                if stored.get(k, "<absent>") != v}
        if diff:
            print(f"[A2] {name}: a finished summary exists but its config differs "
                  f"from the requested one -> RETRAINING from scratch. "
                  f"differences (stored, requested): {diff}")
            for p in (summary_path,
                      os.path.join(args.outdir, f"{name}.csv"),
                      os.path.join(args.outdir, f"{name}.ckpt.pt"),
                      os.path.join(args.outdir, f"{name}.best.pt")):
                if os.path.exists(p):
                    os.remove(p)
        else:
            print(f"[A2] {name}: already finished with an identical config, "
                  f"skipping (--no-resume to redo)")
            return done["summary"]

    # ---- data (built first, with its own seed, identically in every arm) ---
    set_seed(seed)
    ds, num_classes, source = build_ssl_data(args)
    if source != config["data_source"]:
        raise AssertionError(
            f"data source mismatch: declared {config['data_source']!r}, "
            f"built {source!r}")
    diag_v1, diag_v2 = make_fixed_diagnostic_views(ds, args.n_diag, seed=seed)

    # ---- model (re-seeded, so initial weights are identical across arms) ---
    set_seed(seed)
    backbone, feat_dim = make_encoder(arch=args.arch, dataset=args.encoder_variant)
    model = EncoderProjector(backbone, feat_dim, args.hidden, args.proj_dim).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.wd)

    # Clean-distribution instrumentation (PREREG_A2V2): the eval pair is
    # built ONCE and reused at step 0, for the final ranks and for both
    # probes, so every number that says "clean" refers to the same images.
    eval_tr, eval_te, eval_chance = build_eval_data(args)

    loader = DataLoader(ds, batch_size=args.bs, shuffle=True, drop_last=True,
                        num_workers=args.workers,
                        pin_memory=(device.type == "cuda"),
                        persistent_workers=(args.workers > 0),
                        generator=torch.Generator().manual_seed(seed))

    # higher_is_better=False: the only scalar this run could ever hand to
    # save_ckpt as ``best_metric`` is a LOSS (harness deviation #1).
    run = Run(name=name, outdir=args.outdir, config=config, resume=args.resume,
              higher_is_better=False)

    start_step = 0
    pressure_sum, pressure_n = 0.0, 0
    ck = run.load_ckpt() if args.resume else None
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = int(ck["step"])
        pressure_sum = float(ck.get("pressure_sum", 0.0))
        pressure_n = int(ck.get("pressure_n", 0))
        print(f"[A2] {name}: resumed at step {start_step}")

    use_amp = bool(args.amp and device.type == "cuda")

    # Clean rank at initialisation.  Only meaningful on a fresh run: a resumed
    # model is not an initialisation, and pretending otherwise would poison the
    # *_at_init fields silently.
    clean_init: Dict[str, float] = {}
    if start_step == 0:
        clean_init = _clean_rank_block(model, eval_tr, eval_te, device)
        print(f"[A2] {name}: clean rank at init  "
              f"centred={clean_init['eff_rank_h_clean_centred']:.1f}  "
              f"uncentred={clean_init['eff_rank_h_clean_uncentred']:.1f}")
    batches = infinite_batches(loader)
    t0 = time.time()
    last_diag: dict = {}
    # MEASURED, not declared: counted by the loop and cross-checked against the
    # other arms in run_a2.  A budget table filled with argparse defaults proves
    # nothing about what actually ran.
    opt_steps_this_session = 0
    samples_seen_this_session = 0

    print(f"[A2] {name}: {args.steps} steps, bs={args.bs}, reg={reg_kind}, "
          f"gamma={gamma:.4f}, device={device}")

    for step in range(start_step, args.steps):
        # log the state BEFORE the first update, so rank ratios have a baseline
        if step % args.eval_every == 0:
            last_diag = diagnose(model, diag_v1, diag_v2, device, gamma)
            run.log(step=step, seed=seed, arm=arm, lr=cosine_lr(step, args.steps,
                    args.lr, args.warmup), **last_diag)
            if last_diag.get("nonfinite", 0.0):
                raise RuntimeError(
                    f"[A2] {name}: activations became non-finite at step {step}. "
                    f"The run DIVERGED; its numbers must not be averaged into a "
                    f"verdict. Lower --lr or raise --bs and re-run this arm.")

        batch = next(batches)
        x1 = batch["v1"].to(device, non_blocking=True)
        x2 = batch["v2"].to(device, non_blocking=True)

        for gparam in opt.param_groups:
            gparam["lr"] = cosine_lr(step, args.steps, args.lr, args.warmup)

        need_sg = (reg_kind == "unnormalised_sg")
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                h1, z1 = model(x1)
                h2, z2 = model(x2)
                if need_sg:
                    # Same projector WEIGHTS, detached INPUT: the legality
                    # term's gradient reaches the projector and never the
                    # backbone.  Two extra MLP forwards, zero encoder forwards
                    # (declared in PREREG_A2V2["budget_note"]).
                    z1v = model.projector(h1.detach())
                    z2v = model.projector(h2.detach())
            z1, z2 = z1.float(), z2.float()
            if need_sg:
                z1v, z2v = z1v.float(), z2v.float()
        else:
            h1, z1 = model(x1)
            h2, z2 = model(x2)
            if need_sg:
                z1v = model.projector(h1.detach())
                z2v = model.projector(h2.detach())

        # Agreement: symmetric so that neither branch is privileged.  gram_loss
        # detaches its second argument, so each half is sg[A A^T] as in T2.
        agree = 0.5 * (gram_loss(z1, z2) + gram_loss(z2, z1))
        loss = args.gram_w * agree
        var_raw = cov_raw = hinge_frac = float("nan")

        if reg_kind == "unnormalised":
            r1, s1 = vicreg_reg(z1, var_w=args.var_w, cov_w=args.cov_w, gamma=gamma)
            r2, s2 = vicreg_reg(z2, var_w=args.var_w, cov_w=args.cov_w, gamma=gamma)
            loss = loss + 0.5 * (r1 + r2)
            var_raw, cov_raw = 0.5 * (s1["var"] + s2["var"]), 0.5 * (s1["cov"] + s2["cov"])
        elif reg_kind == "unnormalised_sg":
            r1, s1 = vicreg_reg(z1v, var_w=args.var_w, cov_w=args.cov_w, gamma=gamma)
            r2, s2 = vicreg_reg(z2v, var_w=args.var_w, cov_w=args.cov_w, gamma=gamma)
            loss = loss + 0.5 * (r1 + r2)
            var_raw, cov_raw = 0.5 * (s1["var"] + s2["var"]), 0.5 * (s1["cov"] + s2["cov"])
        elif reg_kind == "sphere":
            n1 = z1 / z1.norm(dim=1, keepdim=True).clamp_min(1e-8)
            n2 = z2 / z2.norm(dim=1, keepdim=True).clamp_min(1e-8)
            vs = var_scale_at(step)
            r1, s1 = vicreg_on_unit_sphere(n1, args.var_w * vs, args.cov_w, gamma)
            r2, s2 = vicreg_on_unit_sphere(n2, args.var_w * vs, args.cov_w, gamma)
            loss = loss + 0.5 * (r1 + r2)
            var_raw, cov_raw = 0.5 * (s1["var"] + s2["var"]), 0.5 * (s1["cov"] + s2["cov"])
            hinge_frac = 0.5 * (s1["hinge_active_frac"] + s2["hinge_active_frac"])
            # label-free trajectory statistic: share of the full variance
            # pressure this step actually received (active coordinates x scale)
            pressure_sum += float(hinge_frac) * float(vs)
            pressure_n += 1

        if not torch.isfinite(loss):
            run.log(step=step + 1, seed=seed, arm=arm, train_loss=float(loss),
                    train_agree=float(agree), train_var_raw=var_raw,
                    train_cov_raw=cov_raw, train_hinge_active=hinge_frac,
                    diverged=1.0)
            raise RuntimeError(
                f"[A2] {name}: loss is not finite at step {step} "
                f"(loss={float(loss)}, agree={float(agree)}, var={var_raw}, "
                f"cov={cov_raw}). The run DIVERGED; refusing to continue so that "
                f"no NaN can reach a verdict. Lower --lr or raise --bs.")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        opt_steps_this_session += 1
        samples_seen_this_session += int(x1.shape[0])

        if (step + 1) % args.log_every == 0:
            run.log(step=step + 1, seed=seed, arm=arm,
                    train_loss=float(loss.detach()),
                    train_agree=float(agree.detach()), train_var_raw=var_raw,
                    train_cov_raw=cov_raw, train_hinge_active=hinge_frac)
        if (step + 1) % args.ckpt_every == 0 or (step + 1) == args.steps:
            run.save_ckpt(step + 1, model=model.state_dict(), opt=opt.state_dict(),
                          pressure_sum=pressure_sum, pressure_n=pressure_n)

    # ---- final measurement ----------------------------------------------
    final = diagnose(model, diag_v1, diag_v2, device, gamma)
    run.log(step=args.steps, seed=seed, arm=arm, **final)

    clean_final = _clean_rank_block(model, eval_tr, eval_te, device)
    print(f"[A2] {name}: clean rank final  "
          f"centred={clean_final['eff_rank_h_clean_centred']:.1f}  "
          f"uncentred={clean_final['eff_rank_h_clean_uncentred']:.1f}")

    knn = float("nan")
    chance = float("nan")
    probe_linear = float("nan")
    if args.probe:
        tr, te, chance = eval_tr, eval_te, eval_chance
        knn = knn_probe(model.backbone, tr, te, k=min(20, len(tr) - 1), device=device)
        print(f"[A2] {name}: kNN = {knn:.4f} (chance {chance:.4f})")
        if args.linear_probe_epochs > 0:
            n_cls = int(round(1.0 / max(float(chance), 1e-9)))
            pr = linear_probe(model.backbone, tr, te,
                              epochs=args.linear_probe_epochs,
                              device=device, num_classes=n_cls)
            probe_linear = float(pr.get("acc", float("nan")))
            print(f"[A2] {name}: linear probe = {probe_linear:.4f}")

    # A2V3 instrumentation: z-scored kNN (Q4), RankMe-style ranks of z on clean
    # images, and the model weights, so nothing has to be retrained to re-measure.
    knn_z = float("nan")
    if args.probe:
        knn_z = knn_probe_zscore(model.backbone, eval_tr, eval_te,
                                 k=min(20, len(eval_tr) - 1), device=device)
        print(f"[A2] {name}: kNN on z-scored features = {knn_z:.4f}")
    z_ranks = _clean_embedding_ranks(model, eval_te, device)
    weights_path = os.path.join(args.outdir, f"{name}.weights.pt")
    torch.save({"model": model.state_dict(), "config": config}, weights_path + ".tmp")
    os.replace(weights_path + ".tmp", weights_path)

    # step-0 baseline, read back from our own CSV so it survives a resume
    baseline = _read_first_row_value(run.csv_path, "eff_rank_z")

    summary = dict(final)
    summary.update({
        "arm": arm, "seed": seed, "steps": args.steps, "batch_size": args.bs,
        "knn_acc": knn, "knn_chance": chance,
        "probe_linear_acc": probe_linear,
        "knn_acc_zscore": knn_z,
        **z_ranks,
        "var_schedule": config.get("var_schedule", "full"),
        "var_pressure_mean": (pressure_sum / pressure_n) if pressure_n else float("nan"),
        "var_pressure_steps": int(pressure_n),
        "weights_file": os.path.basename(weights_path),
        **clean_final,
        **{f"{k}_at_init": v for k, v in clean_init.items()},
        "extra_projector_forwards_per_step": 2 if reg_kind == "unnormalised_sg" else 0,
        "eff_rank_z_at_init": baseline,
        "eff_rank_ratio": (final["eff_rank_z"] / baseline)
                          if (baseline and not math.isnan(baseline)) else float("nan"),
        "wall_s": round(time.time() - t0, 1),
        "resumed_from_step": start_step,
        # measured budget: total optimiser steps this run has ever paid for
        "optimiser_steps_total": int(start_step + opt_steps_this_session),
        "optimiser_steps_this_session": int(opt_steps_this_session),
        "samples_seen_this_session": int(samples_seen_this_session),
        "forward_passes_per_step": 2,
        "data_source": source,
    })
    run.finish(summary)
    return summary


def _read_first_row_value(csv_path: str, column: str) -> float:
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
        sub = df[df[column].notna()].sort_values("step")
        return float(sub[column].iloc[0])
    except Exception:
        return float("nan")


def run_a2(args, device: torch.device) -> dict:
    print("\n" + "=" * 78)
    print("A2 -- the relational target collapses on its own (thesis T2)")
    print("=" * 78)

    selfcheck_local_vicreg()

    # The analytic fact, checked numerically rather than asserted in prose:
    # identical embeddings give gram_loss == 0 exactly.  This is the reason the
    # collapse of arm `gram` is a global minimum and not an optimisation fluke.
    collapsed = torch.randn(1, args.proj_dim).repeat(64, 1)
    collapse_loss = float(gram_loss(collapsed, collapsed))
    print(f"[check] gram_loss on a totally collapsed batch = {collapse_loss:.3e} "
          f"(global minimum, T2)")

    seeds = [args.seed + i for i in range(args.n_seeds)]
    arms = args.arms
    # matched_budget_check needs >= 2 arms; a single-arm run is a debug run and
    # has nothing to be matched against.
    if len(arms) >= 2:
        matched_budget_check([
            {"name": a, "steps": args.steps, "batch_size": args.bs,
             "forward_passes": 2, "optimiser_steps": args.steps} for a in arms
        ])
    else:
        print(f"[A2] single arm {arms} -- no budget matching to do, and no "
              f"comparison is licensed from this run.")

    results: Dict[str, List[dict]] = {a: [] for a in arms}
    for arm in arms:
        for seed in seeds:
            results[arm].append(train_a2_arm(arm, seed, args, device))

    # ---- MEASURED budget check, after the fact ---------------------------
    # The pre-flight matched_budget_check above compares argparse numbers.  This
    # one compares what the loops actually counted, which is the only version
    # that can catch a resumed / truncated / mis-configured arm.
    measured = {a: sorted({int(s.get("optimiser_steps_total", -1))
                           for s in results[a]}) for a in arms}
    flat = {v for vals in measured.values() for v in vals}
    budget_ok = (len(flat) == 1 and -1 not in flat)
    print(f"[A2] measured optimiser steps per (arm, seed): {measured}")
    if not budget_ok:
        raise AssertionError(
            "UNMATCHED MEASURED BUDGET across A2 arms/seeds: "
            f"{measured}. The arms are not comparable; refusing to emit "
            "verdicts. Delete the offending runs and re-run.")
    print(f"[A2] OK: every (arm, seed) paid exactly {flat.pop()} optimiser steps "
          f"at batch {args.bs}, 2 encoder forwards per step.")

    plot_a2(args, arms, seeds)
    return {"per_arm": results, "collapse_loss_check": collapse_loss,
            "seeds": seeds, "arms": arms, "measured_budget": measured}


# ---------------------------------------------------------------------------
# A2 plotting
# ---------------------------------------------------------------------------

_ARM_STYLE = {
    "gram": ("0.0", "-", "o"),
    "gram_vicreg": ("0.45", "--", "s"),
    "gram_vicreg_sg": ("0.3", "--", "v"),
    "gram_vicreg_sphere": ("0.0", ":", "^"),
    "gram_vicreg_sphere_gamma": ("0.55", "-.", "d"),
    "gram_vicreg_sphere_gamma_half": ("0.7", "-.", "*"),
}


def _panel(ax, args, arms, column, title, ylabel, logy=False, hline=None,
           hline_label=None):
    for arm in arms:
        try:
            df = aggregate_seeds(os.path.join(args.outdir, f"A2_{arm}_seed*.csv"), "step")
        except (FileNotFoundError, KeyError, ValueError) as e:
            print(f"[A2] panel {column!r}: no usable CSV for arm {arm!r} ({e})")
            continue
        m, s = f"{column}_mean", f"{column}_std"
        if m not in df.columns:
            continue
        sub = df[df[m].notna()]
        if sub.empty:
            continue
        colour, ls, marker = _ARM_STYLE.get(arm, ("0.3", "-", "o"))
        x = sub["step"].to_numpy()
        y = sub[m].to_numpy()
        e = np.nan_to_num(sub[s].to_numpy()) if s in sub.columns else np.zeros_like(y)
        ax.plot(x, y, color=colour, linestyle=ls, marker=marker, markersize=3.5,
                markevery=max(1, len(x) // 12), linewidth=1.5, label=arm)
        ax.fill_between(x, y - e, y + e, color=colour, alpha=0.18, linewidth=0)
    if hline is not None:
        ax.axhline(hline, color="black", linewidth=1, linestyle=(0, (1, 3)))
        if hline_label:
            ax.text(0.99, hline, hline_label, transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=7)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("optimisation step")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(linestyle=":", linewidth=0.5)


def plot_a2(args, arms: List[str], seeds: List[int]) -> None:
    inv_sqrt_d = 1.0 / math.sqrt(args.proj_dim)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))

    _panel(axes[0, 0], args, arms, "agree_loss_diag",
           "agreement loss (gram) on frozen diagnostic views",
           "gram_loss", logy=True)
    _panel(axes[0, 1], args, arms, "eff_rank_z",
           "effective rank of the projector output z",
           "effective rank", logy=True, hline=1.0, hline_label="total collapse")
    _panel(axes[0, 2], args, arms, "eff_rank_h",
           "effective rank of the backbone features h",
           "effective rank", logy=True, hline=1.0, hline_label="total collapse")
    _panel(axes[1, 0], args, arms, "mean_std_z",
           "mean per-coordinate std of UNNORMALISED z",
           "sigma", logy=True, hline=1.0, hline_label="gamma = 1 (hinge target)")
    _panel(axes[1, 1], args, arms, "mean_std_z_normalised",
           "mean per-coordinate std of L2-NORMALISED z\n"
           "(this is what arm *_sphere regularises)",
           "sigma", logy=True, hline=inv_sqrt_d,
           hline_label=f"1/sqrt(d) = {inv_sqrt_d:.3f}  <= hard ceiling")
    _panel(axes[1, 2], args, arms, "alignment",
           "alignment  E||z1 - z2||^2  (0 = the two views coincide)",
           "alignment", logy=True)

    handles, labels = axes[0, 1].get_legend_handles_labels()
    if labels:  # ncol=0 is a hard matplotlib error
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   frameon=False)
    fig.suptitle(
        "A2 (T2): the relational target is degenerate on its own -- "
        f"{args.dataset}, {args.arch}, {args.steps} steps, bs={args.bs}, "
        f"{len(seeds)} seeds (mean +/- std)", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    p = os.path.join(args.outdir, "A2_curves.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[A2] figure -> {p}")

    # ---- final bar chart: effective rank and kNN, mean +/- std over seeds --
    rank_m, rank_s, knn_m, knn_s, chance = [], [], [], [], float("nan")
    for arm in arms:
        ranks, knns = [], []
        for seed in seeds:
            p_sum = os.path.join(args.outdir, f"A2_{arm}_seed{seed}.summary.json")
            if not os.path.exists(p_sum):
                continue
            with open(p_sum, "r", encoding="utf-8") as f:
                s = json.load(f)["summary"]
            ranks.append(s.get("eff_rank_z", np.nan))
            knns.append(s.get("knn_acc", np.nan))
            chance = s.get("knn_chance", chance)
        rank_m.append(np.nanmean(ranks) if ranks else np.nan)
        rank_s.append(np.nanstd(ranks, ddof=1) if len(ranks) > 1 else 0.0)
        knn_m.append(np.nanmean(knns) if knns else np.nan)
        knn_s.append(np.nanstd(knns, ddof=1) if len(knns) > 1 else 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(arms))
    greys = [_ARM_STYLE.get(a, ("0.3",))[0] for a in arms]
    axes[0].bar(x, rank_m, yerr=rank_s, color=greys, edgecolor="black", capsize=4)
    axes[0].set_xticks(x); axes[0].set_xticklabels(arms, rotation=12, fontsize=8)
    axes[0].set_ylabel("effective rank of z (final)")
    axes[0].set_title("collapse: effective rank at the end")
    axes[0].grid(axis="y", linestyle=":", linewidth=0.6)
    if np.all(np.isnan(knn_m)):
        axes[1].text(0.5, 0.5, "k-NN probe disabled (--no-probe)", ha="center",
                     va="center", transform=axes[1].transAxes)
    else:
        axes[1].bar(x, knn_m, yerr=knn_s, color=greys, edgecolor="black", capsize=4)
        if not math.isnan(chance):
            axes[1].axhline(chance, color="black", linestyle="--", linewidth=1)
            axes[1].text(len(arms) - 0.5, chance * 1.05, "chance", ha="right", fontsize=8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(arms, rotation=12, fontsize=8)
    axes[1].set_ylabel("k-NN accuracy (frozen backbone)")
    axes[1].set_title("does the collapse show up downstream?")
    axes[1].grid(axis="y", linestyle=":", linewidth=0.6)
    fig.tight_layout()
    p = os.path.join(args.outdir, "A2_final_bars.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[A2] figure -> {p}")


# ===========================================================================
# verdicts
# ===========================================================================

def _mean(vals: List[float]) -> float:
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(v)) if v else float("nan")


def _std(vals: List[float]) -> float:
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def verdicts_a1(res: dict, args) -> dict:
    eps32, eps64 = float(np.finfo(np.float32).eps), float(np.finfo(np.float64).eps)
    out = {}

    # P1.a -- relational / contrastive invariance, and the precision test that
    # distinguishes a numerical residual from a structural one.
    shrank = (res["gram_f64"] < res["gram_f32"] * 1e-3 or res["gram_f64"] <= 1e-14) and \
             (res["infonce_f64"] < res["infonce_f32"] * 1e-3 or res["infonce_f64"] <= 1e-14)
    tiny = res["gram_f32"] < 1e-4 and res["infonce_f32"] < 1e-4
    out["P1.a"] = {
        "thesis": "T1",
        "statement": "gram_loss and info_nce are invariant under Z -> ZQ, Q orthogonal",
        "falsifier": "a residual that does NOT shrink when moving float32 -> float64",
        "verdict": VERDICT_CONFIRMED if (shrank and tiny) else
                   (VERDICT_REFUTED if not tiny else VERDICT_INCONCLUSIVE),
        "evidence": {
            "gram_mean_abs_delta_float32": res["gram_f32"],
            "gram_mean_abs_delta_float64": res["gram_f64"],
            "info_nce_mean_abs_delta_float32": res["infonce_f32"],
            "info_nce_mean_abs_delta_float64": res["infonce_f64"],
            "float32_eps": eps32, "float64_eps": eps64,
            "max_orthogonality_error": res["max_ortho_error"],
        },
    }

    # P1.b -- the prototype head is not invariant, by a macroscopic amount.
    macroscopic = (res["proto_fixed_f64"] > 1e-6 and res["proto_learned_f64"] > 1e-6)
    out["P1.b"] = {
        "thesis": "T1",
        "statement": "prototype_loss is NOT invariant: the c_k fix a distinguished frame",
        "falsifier": "a machine-precision residual for prototype_loss with fixed c_k",
        "verdict": VERDICT_CONFIRMED if macroscopic else VERDICT_REFUTED,
        "evidence": {
            "prototype_fixed_mean_abs_delta_float64": res["proto_fixed_f64"],
            "prototype_fixed_relative_delta": res["proto_fixed_f64_rel"],
            "prototype_fitted_mean_abs_delta_float64": res["proto_learned_f64"],
            "prototype_fitted_relative_delta": res["proto_learned_f64_rel"],
            "delta_over_batch_to_batch_std": res["proto_fixed_f64_over_scale"],
            "ratio_to_gram_delta": (res["proto_fixed_f64"] / res["gram_f64"]
                                    if res["gram_f64"] > 0 else float("inf")),
        },
    }

    # P1.c/d -- the two controls.
    ctrl_ok = res["control_rotate_protos_f64"] < 1e-8
    out["P1.c"] = {
        "thesis": "T1",
        "statement": "rotating the prototypes with the embeddings restores invariance "
                     "-- the culprit is the frame, not the softmax",
        "falsifier": "the control staying non-invariant",
        "verdict": VERDICT_CONFIRMED if ctrl_ok else VERDICT_REFUTED,
        "evidence": {
            "control_rotate_protos_mean_abs_delta_float64":
                res["control_rotate_protos_f64"],
            "compare_uncontrolled": res["proto_learned_f64"],
        },
    }
    perm_ok = res["control_permute_protos_f64"] < 1e-8
    out["P1.d"] = {
        "thesis": "T1",
        "statement": "permuting the K prototype rows changes nothing: the broken "
                     "symmetry is over the d-dimensional frame, not over the K slots",
        "falsifier": "a non-zero residual under a pure row permutation",
        "verdict": VERDICT_CONFIRMED if perm_ok else VERDICT_REFUTED,
        "evidence": {"control_permute_protos_mean_abs_delta_float64":
                     res["control_permute_protos_f64"]},
    }
    return out


def verdicts_a2(res: dict, args) -> dict:
    per_arm = res["per_arm"]
    out = {}

    def agg(arm: str, key: str) -> Tuple[float, float]:
        vals = [s.get(key, float("nan")) for s in per_arm.get(arm, [])]
        return _mean(vals), _std(vals)

    def _n(arm: str) -> int:
        return len(per_arm.get(arm, []))

    # ---- Q1 / Q2 : the five-arm pre-registration (PREREG_A2V2) -------------
    # Thresholds registered 2026-09-05, before any sg / sphere_gamma run.
    if "gram" in per_arm and "gram_vicreg_sg" in per_arm:
        v = {"prediction": PREREG_A2V2["Q1_gradient_leak"],
             "statement": PREREG_A2V2["Q1_gradient_leak"]["statement"]}
        k_g, ks_g = agg("gram", "knn_acc")
        k_s, ks_s = agg("gram_vicreg_sg", "knn_acc")
        rz_s, rzs_s = agg("gram_vicreg_sg", "eff_rank_z")
        d_pp = (k_s - k_g) * 100.0
        ev = {"knn_gram": k_g, "knn_gram_std": ks_g,
              "knn_sg": k_s, "knn_sg_std": ks_s,
              "delta_knn_pp": d_pp,
              "eff_rank_z_sg": rz_s, "eff_rank_z_sg_std": rzs_s,
              "n_seeds": min(_n("gram"), _n("gram_vicreg_sg"))}
        if ev["n_seeds"] < 3 or d_pp != d_pp:
            v["verdict"] = VERDICT_INCONCLUSIVE
            ev["reason"] = "fewer than 3 seeds, or missing kNN"
        elif d_pp > -0.8 and rz_s > 50.0:
            v["verdict"] = VERDICT_CONFIRMED
        elif d_pp < -1.8:
            v["verdict"] = VERDICT_REFUTED
        else:
            v["verdict"] = VERDICT_INCONCLUSIVE
        v["evidence"] = ev
        out["Q1_gradient_leak"] = v

    if all(a in per_arm for a in ("gram", "gram_vicreg_sphere",
                                  "gram_vicreg_sphere_gamma")):
        v = {"prediction": PREREG_A2V2["Q2_saturated_vs_satisfiable"],
             "statement": PREREG_A2V2["Q2_saturated_vs_satisfiable"]["statement"]}
        key = PREREG_A2V2["measurement"]["primary_rank_statistic"]
        r_g, rs_g = agg("gram", key)
        r_s, rs_s = agg("gram_vicreg_sphere", key)
        r_y, rs_y = agg("gram_vicreg_sphere_gamma", key)
        gain_sphere, gain_gamma = r_s - r_g, r_y - r_g
        ev = {"rank_gram": r_g, "rank_sphere": r_s, "rank_sphere_gamma": r_y,
              "gain_sphere": gain_sphere, "gain_sphere_gamma": gain_gamma,
              "stds": {"gram": rs_g, "sphere": rs_s, "sphere_gamma": rs_y},
              "rank_statistic": key,
              "n_seeds": min(_n("gram"), _n("gram_vicreg_sphere"),
                             _n("gram_vicreg_sphere_gamma"))}
        if ev["n_seeds"] < 3 or any(x != x for x in (r_g, r_s, r_y)):
            v["verdict"] = VERDICT_INCONCLUSIVE
            ev["reason"] = ("fewer than 3 seeds, or the clean centred rank is "
                            "missing (legacy summaries do not carry it: the "
                            "five arms must run with this instrumentation)")
        elif gain_gamma < 30.0 and gain_sphere > 60.0:
            v["verdict"] = VERDICT_CONFIRMED
        elif gain_gamma >= 60.0:
            v["verdict"] = VERDICT_REFUTED
        else:
            v["verdict"] = VERDICT_INCONCLUSIVE
        v["evidence"] = ev
        out["Q2_saturated_vs_satisfiable"] = v

    # ---- Q3 / Q4 : PREREG_A2V3, registered 2026-09-21 ----------------------
    REF = "gram_vicreg_sphere"
    D11, D51 = "gram_vicreg_sphere_dose11", "gram_vicreg_sphere_dose51"
    I11, I51 = "gram_vicreg_sphere_interm11", "gram_vicreg_sphere_interm51"
    if all(a in per_arm for a in (REF, D11, I11)):
        v = {"prediction": PREREG_A2V3["Q3_pressure_at_fixed_gamma"],
             "statement": PREREG_A2V3["Q3_pressure_at_fixed_gamma"]["statement"]}
        k = {a: agg(a, "knn_acc")[0] for a in per_arm}
        drop_d = (k[REF] - k[D11]) * 100.0
        drop_i = (k[REF] - k[I11]) * 100.0
        mean_drop = 0.5 * (drop_d + drop_i)
        have51 = (D51 in per_arm and I51 in per_arm)
        ordered = bool(have51 and k[REF] > k[D51] > k[D11] and k[REF] > k[I51] > k[I11])
        sphere_like = [a for a in per_arm if a.startswith("gram_vicreg_sphere")]

        def _spearman(xs: List[float], ys: List[float]) -> float:
            pts = [(x, y) for x, y in zip(xs, ys) if x == x and y == y]
            if len(pts) < 3:
                return float("nan")
            def _avg_ranks(vals: List[float]) -> np.ndarray:
                a = np.asarray(vals, dtype=float)
                order = np.argsort(a, kind="mergesort")
                ranks = np.empty(len(a), dtype=float)
                ranks[order] = np.arange(1, len(a) + 1, dtype=float)
                for val in np.unique(a):          # ties share their mean rank
                    m = (a == val)
                    ranks[m] = ranks[m].mean()
                return ranks

            rx = _avg_ranks([p[0] for p in pts])
            ry = _avg_ranks([p[1] for p in pts])
            if rx.std() == 0 or ry.std() == 0:
                return float("nan")
            return float(np.corrcoef(rx, ry)[0, 1])

        ks = [k[a] for a in sphere_like]
        ev = {"knn": {a: k[a] for a in sphere_like},
              "knn_std": {a: agg(a, "knn_acc")[1] for a in sphere_like},
              "drop_dose11_pp": drop_d, "drop_interm11_pp": drop_i,
              "mean_drop_11_pp": mean_drop, "ordered_within_families": ordered,
              "exploratory": {
                  "var_pressure_mean": {a: agg(a, "var_pressure_mean")[0] for a in sphere_like},
                  "spearman_knn_vs_var_pressure": _spearman(
                      [agg(a, "var_pressure_mean")[0] for a in sphere_like], ks),
                  "spearman_knn_vs_backbone_rank": _spearman(
                      [agg(a, "eff_rank_h_clean_centred")[0] for a in sphere_like], ks),
                  "spearman_knn_vs_embedding_rank_uncentred": _spearman(
                      [agg(a, "eff_rank_z_clean_uncentred")[0] for a in sphere_like], ks),
                  "n_arms": len(sphere_like)},
              "n_seeds": min(_n(a) for a in (REF, D11, I11))}
        if ev["n_seeds"] < 3 or mean_drop != mean_drop:
            v["verdict"] = VERDICT_INCONCLUSIVE
            ev["reason"] = "fewer than 3 seeds, or missing kNN"
        elif mean_drop >= 3.0 and ordered:
            v["verdict"] = VERDICT_CONFIRMED
        elif drop_d < 1.5 and drop_i < 1.5:
            v["verdict"] = VERDICT_REFUTED
        else:
            v["verdict"] = VERDICT_INCONCLUSIVE
        v["evidence"] = ev
        out["Q3_pressure_at_fixed_gamma"] = v

    if "gram" in per_arm and "gram_vicreg" in per_arm:
        kz_g, _ = agg("gram", "knn_acc_zscore")
        kz_v, _ = agg("gram_vicreg", "knn_acc_zscore")
        if kz_g == kz_g and kz_v == kz_v:      # legacy summaries do not carry it
            v = {"prediction": PREREG_A2V3["Q4_probe_normalisation"],
                 "statement": PREREG_A2V3["Q4_probe_normalisation"]["statement"]}
            d_raw = (agg("gram_vicreg", "knn_acc")[0] - agg("gram", "knn_acc")[0]) * 100.0
            d_z = (kz_v - kz_g) * 100.0
            d_lin = (agg("gram_vicreg", "probe_linear_acc")[0]
                     - agg("gram", "probe_linear_acc")[0]) * 100.0
            ev = {"delta_knn_raw_pp": d_raw, "delta_knn_zscore_pp": d_z,
                  "delta_linear_pp": d_lin,
                  "knn_zscore_gram": kz_g, "knn_zscore_gram_vicreg": kz_v,
                  "n_seeds": min(_n("gram"), _n("gram_vicreg"))}
            if ev["n_seeds"] < 3:
                v["verdict"] = VERDICT_INCONCLUSIVE
                ev["reason"] = "fewer than 3 seeds"
            elif not d_raw < -1.0:
                v["verdict"] = VERDICT_INCONCLUSIVE
                ev["reason"] = ("premise not reproduced: the raw kNN difference is "
                                "not below -1.0 pp in this run")
            elif d_z > 0.5:
                v["verdict"] = VERDICT_CONFIRMED
            elif d_z < -1.0:
                v["verdict"] = VERDICT_REFUTED
            else:
                v["verdict"] = VERDICT_INCONCLUSIVE
            v["evidence"] = ev
            out["Q4_probe_normalisation"] = v

    # ---- P2.1 : the pure relational objective collapses --------------------
    if "gram" in per_arm:
        rank_m, rank_s = agg("gram", "eff_rank_z")
        ratio_m, _ = agg("gram", "eff_rank_ratio")
        loss_m, _ = agg("gram", "agree_loss_diag")
        knn_m, knn_s = agg("gram", "knn_acc")
        chance, _ = agg("gram", "knn_chance")
        collapsed = (ratio_m < 0.2 and rank_m < args.collapse_rank) and loss_m < args.collapse_loss
        not_collapsed = ratio_m > 0.7
        out["P2.1"] = {
            "thesis": "T2",
            "statement": "with gram_loss alone the agreement loss goes to ~0 while the "
                         "effective rank of z collapses towards 1",
            "falsifier": "a preserved effective rank (ratio > 0.7) once the loss is ~0",
            "verdict": VERDICT_CONFIRMED if collapsed else
                       (VERDICT_REFUTED if not_collapsed else VERDICT_INCONCLUSIVE),
            "evidence": {
                "eff_rank_z_final_mean": rank_m, "eff_rank_z_final_std": rank_s,
                "eff_rank_final_over_init": ratio_m,
                "eff_rank_z_at_init": agg("gram", "eff_rank_z_at_init")[0],
                "agreement_loss_final": loss_m,
                "knn_acc_mean": knn_m, "knn_acc_std": knn_s, "knn_chance": chance,
                "gram_loss_on_a_totally_collapsed_batch": res["collapse_loss_check"],
                "thresholds": {"rank_ratio<": 0.2, "rank<": args.collapse_rank,
                               "loss<": args.collapse_loss},
            },
        }

    # ---- P2.2 : the legality term on unnormalised z prevents collapse ------
    if "gram_vicreg" in per_arm and "gram" in per_arm:
        rank_v, rank_vs = agg("gram_vicreg", "eff_rank_z")
        rank_g, _ = agg("gram", "eff_rank_z")
        ratio_v, _ = agg("gram_vicreg", "eff_rank_ratio")
        knn_v, knn_vs = agg("gram_vicreg", "knn_acc")
        knn_g, _ = agg("gram", "knn_acc")
        saved = (rank_v > 3.0 * max(rank_g, 1e-9)) and ratio_v > 0.3
        out["P2.2"] = {
            "thesis": "T2",
            "statement": "the legality term (vicreg on UNNORMALISED z) is what prevents "
                         "collapse",
            "falsifier": "gram_vicreg collapsing too, or not beating gram on rank",
            "verdict": VERDICT_CONFIRMED if saved else VERDICT_REFUTED,
            "evidence": {
                "eff_rank_z_gram_vicreg": rank_v, "eff_rank_z_gram_vicreg_std": rank_vs,
                "eff_rank_z_gram": rank_g,
                "rank_ratio_gram_vicreg": ratio_v,
                "knn_gram_vicreg": knn_v, "knn_gram_vicreg_std": knn_vs,
                "knn_gram": knn_g,
            },
        }

    # ---- P2.3a : on the sphere the hinge is saturated ---------------------
    sphere_arms = [a for a in per_arm if a.startswith("gram_vicreg_sphere")]
    if sphere_arms:
        arm = "gram_vicreg_sphere" if "gram_vicreg_sphere" in sphere_arms else sphere_arms[0]
        # read the whole trajectory, not just the end point
        try:
            df = aggregate_seeds(os.path.join(args.outdir, f"A2_{arm}_seed*.csv"), "step")
            col = "hinge_active_frac_z_normalised_mean"
            traj = df[df[col].notna()][col].to_numpy() if col in df.columns else np.array([])
        except FileNotFoundError:
            traj = np.array([])
        always_saturated = bool(traj.size and float(traj.min()) >= 1.0 - 1e-9)
        std_zn, _ = agg(arm, "mean_std_z_normalised")
        ceiling, _ = agg(arm, "inv_sqrt_d")
        # Compare against the Bessel-corrected ceiling sqrt(n/(n-1))/sqrt(d),
        # not against a bare 1/sqrt(d): the diagnostic std is computed with
        # unbiased=True, so the bare bound is exceeded by a factor sqrt(n/(n-1))
        # even for a perfectly uniform cloud.  Older summaries have no such key,
        # hence the fallback.
        ceiling_ub, _ = agg(arm, "std_ceiling_normalised")
        if math.isnan(ceiling_ub):
            ceiling_ub = ceiling
        under_ceiling = bool(std_zn <= ceiling_ub * (1.0 + 1e-6))
        out["P2.3a"] = {
            "thesis": "T2",
            "statement": "on L2-normalised z the variance hinge relu(gamma - sigma) is "
                         "saturated at every step: it can never be satisfied, so it stops "
                         "being a hinge",
            "falsifier": "the active fraction of the hinge dropping below 1.0 at any "
                         "logged step",
            "verdict": VERDICT_CONFIRMED if (always_saturated and under_ceiling)
                       else VERDICT_REFUTED,
            "evidence": {
                "min_hinge_active_frac_over_training": (float(traj.min()) if traj.size
                                                        else float("nan")),
                "n_logged_points": int(traj.size),
                "mean_std_of_normalised_z_final": std_zn,
                "hard_ceiling_1_over_sqrt_d": ceiling,
                "hard_ceiling_bessel_corrected": ceiling_ub,
                "arm": arm,
            },
        }

        # ---- P2.3b : does that arm actually collapse? (measured, not assumed)
        rank_s_arm, rank_s_std = agg(arm, "eff_rank_z")
        ratio_s, _ = agg(arm, "eff_rank_ratio")
        rank_g, _ = agg("gram", "eff_rank_z") if "gram" in per_arm else (float("nan"), 0)
        rank_v, _ = agg("gram_vicreg", "eff_rank_z") if "gram_vicreg" in per_arm \
            else (float("nan"), 0)
        if math.isnan(rank_v):
            verdict = VERDICT_INCONCLUSIVE
        elif ratio_s < 0.2 and rank_s_arm < args.collapse_rank:
            verdict = VERDICT_CONFIRMED          # collapses like the unregularised arm
        elif rank_s_arm >= 0.8 * rank_v:
            verdict = VERDICT_REFUTED            # as good as the correct arm
        else:
            verdict = VERDICT_INCONCLUSIVE       # degraded but not collapsed
        out["P2.3b"] = {
            "thesis": "T2 (weaker, secondary claim)",
            "statement": "vicreg on L2-normalised z fails to prevent collapse the way it "
                         "does on unnormalised z. NOTE: the hinge gradient does not "
                         "vanish (slope -1 while active) and the covariance term still "
                         "acts on the sphere, so a partial rescue is possible and would "
                         "NOT refute P2.3a.",
            "falsifier": "the sphere arm reaching >= 80% of the effective rank of "
                         "gram_vicreg",
            "verdict": verdict,
            "evidence": {
                "eff_rank_z_sphere": rank_s_arm, "eff_rank_z_sphere_std": rank_s_std,
                "eff_rank_ratio_sphere": ratio_s,
                "eff_rank_z_gram_only": rank_g,
                "eff_rank_z_gram_vicreg": rank_v,
                "knn_sphere": agg(arm, "knn_acc")[0],
            },
        }

    # ---- project rule: no conclusion from fewer than 3 seeds --------------
    # Every A2 verdict is a comparison of noisy training runs.  With < 3 seeds
    # there is no error bar, so the verdict is downgraded rather than printed
    # as if it had been established.  The raw call is kept for the record.
    n_seeds = len(res.get("seeds", []))
    for block in out.values():
        block["evidence"]["n_seeds"] = n_seeds
        if n_seeds < 3:
            block["verdict_from_thresholds"] = block["verdict"]
            block["verdict"] = VERDICT_INCONCLUSIVE
            block["downgrade_reason"] = (
                f"only {n_seeds} seed(s); the project rule requires >= 3 for any "
                f"comparison, so this verdict is not licensed by the data.")
    return out


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Uebergang-SSL experiment group A: cheap decisive diagnostics "
                    "(A1 = O(d) invariance unit test, A2 = collapse of the relational "
                    "target). Defaults are the real A100 configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--exp", choices=["a1", "a2", "all"], default="all",
                   help="which experiment to run")
    p.add_argument("--all", action="store_true",
                   help="run everything: both experiments and all four A2 arms "
                        "(including the optional gamma=1/sqrt(d) defence arm)")
    p.add_argument("--arm", action="append", choices=sorted(A2_ARMS),
                   help="A2 arm(s) to run; repeatable. Default: the three core arms")

    p.add_argument("--linear-probe-epochs", type=int, default=25,
                   help="linear probe on the frozen backbone at the end of "
                        "each A2 arm (0 disables it)")
    p.add_argument("--seed", type=int, default=0, help="base seed")
    p.add_argument("--n-seeds", type=int, default=3,
                   help="number of seeds: base, base+1, ... (>=3 for any comparison)")

    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="wipe existing CSVs/checkpoints and start clean")
    p.set_defaults(resume=True)

    # --- A2 training ---
    p.add_argument("--dataset", default="cifar10", choices=sorted(DATASET_NUM_CLASSES))
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.06)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=0.0,
                   help="global grad-norm clip; 0 = off. Real runs default to off so "
                        "the measured dynamics are untouched; --smoke turns it on "
                        "because a tiny batch makes the VICReg variance hinge stiff.")
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    p.add_argument("--encoder-variant", default="cifar", choices=["cifar", "stl", "imagenet"])
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--gram-w", type=float, default=25.0,
                   help="weight of the agreement term (VICReg weights its invariance "
                        "term at 25 too; identical in every arm)")
    p.add_argument("--var-w", type=float, default=25.0)
    p.add_argument("--cov-w", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=50, help="diagnostics period")
    p.add_argument("--log-every", type=int, default=25, help="training-loss log period")
    p.add_argument("--ckpt-every", type=int, default=200)
    p.add_argument("--n-diag", type=int, default=512,
                   help="images in the frozen diagnostic view pair")
    p.add_argument("--train-subset", type=int, default=None,
                   help="limit the SSL training pool (debug)")
    p.add_argument("--eval-subset", type=int, default=None,
                   help="cap the k-NN probe splits to this many images each "
                        "(debug / smoke; None = full official splits)")
    p.add_argument("--workers", type=int, default=None,
                   help="DataLoader workers (default: 8 on cuda, 0 on cpu)")
    p.add_argument("--no-amp", dest="amp", action="store_false",
                   help="disable bfloat16 autocast on cuda")
    p.set_defaults(amp=True)
    p.add_argument("--no-probe", dest="probe", action="store_false",
                   help="skip the final k-NN probe")
    p.set_defaults(probe=True)
    p.add_argument("--collapse-rank", type=float, default=3.0,
                   help="effective rank below which we call it a collapse")
    p.add_argument("--collapse-loss", type=float, default=1e-3,
                   help="agreement loss below which we call it converged")

    # --- A1 ---
    p.add_argument("--a1-batch", type=int, default=256)
    p.add_argument("--a1-dim", type=int, default=128)
    p.add_argument("--a1-protos", type=int, default=1024)
    p.add_argument("--a1-batches", type=int, default=8)
    p.add_argument("--a1-rotations", type=int, default=16)
    p.add_argument("--a1-tau-s", type=float, default=0.1)
    p.add_argument("--a1-tau-t", type=float, default=0.04)
    p.add_argument("--a1-fit-steps", type=int, default=200)
    p.add_argument("--a1-fit-lr", type=float, default=1e-2)

    p.add_argument("--cpu", action="store_true", help="force CPU")
    p.add_argument("--smoke", action="store_true",
                   help="tiny end-to-end pipeline check, < 3 min on CPU, no network "
                        "(uses the real dataset only if it is already downloaded)")
    return p


def apply_smoke(args) -> None:
    """Shrink everything.  --smoke validates the PIPELINE, never a thesis."""
    args.cpu = True
    args.steps = 8
    # bs=8 makes the VICReg variance/covariance estimate degenerate (a 32x32
    # covariance from 8 samples is rank<=7) and, with the lr tuned for bs=256,
    # the run diverges inside the smoke test itself.  Smoke must exercise the
    # path, not reproduce a divergence.
    args.bs = 32
    args.lr = 0.06 * 32 / 256.0     # linear lr scaling from the bs=256 default
    args.clip = 1.0
    args.warmup = 2
    args.eval_every = 2
    args.log_every = 2
    args.ckpt_every = 4
    args.n_diag = 32
    args.n_seeds = 2
    args.proj_dim = 32
    args.hidden = 64
    args.workers = 0
    args.amp = False
    args.image_size = 32
    args.a1_batch = 64
    args.a1_dim = 32
    args.a1_protos = 64
    args.a1_batches = 2
    args.a1_rotations = 4
    args.a1_fit_steps = 25
    args.synthetic_train = 192
    args.synthetic_classes = 4
    args.eval_subset = 128
    # Exercise every arm, including the gamma=1/sqrt(d) branch, unless the user
    # asked for specific ones: --smoke must cover the whole main path.
    if not args.arm:
        args.arm = sorted(A2_ARMS)
    # use the real dataset only if it is already on disk: --smoke must not hit
    # the network, and must not silently download 170 MB on a laptop.
    args.synthetic = not real_dataset_present(args.dataset, args.data_root)
    if args.synthetic:
        print("[smoke] real dataset not found on disk -> synthetic images. "
              "This validates the pipeline only.")
    else:
        args.train_subset = 512
        print(f"[smoke] using the already-downloaded {args.dataset} (subset of 512).")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if not hasattr(args, "synthetic"):
        args.synthetic = False
        args.synthetic_train = 192
        args.synthetic_classes = 4
    if args.all:
        args.exp = "all"
        args.arm = sorted(A2_ARMS)
    if args.smoke:
        apply_smoke(args)
    args.arms = list(dict.fromkeys(args.arm)) if args.arm else list(A2_CORE_ARMS)
    if args.workers is None:
        args.workers = 0 if (args.cpu or not torch.cuda.is_available()) else 8

    os.makedirs(args.outdir, exist_ok=True)
    device = get_device(prefer_cpu=args.cpu)

    print("=" * 78)
    print("UEBERGANG-SSL -- EXPERIMENT GROUP A (cheap decisive diagnostics)")
    print("=" * 78)
    print(f"device={device}  outdir={args.outdir}")
    print(f"exp={args.exp}  arms={args.arms}  seeds="
          f"{[args.seed + i for i in range(args.n_seeds)]}")
    if args.n_seeds < 3 and not args.smoke:
        print("WARNING: fewer than 3 seeds. Per project rule, a comparison on one "
              "seed decides nothing; treat the verdicts as provisional.")

    t0 = time.time()
    summary = {
        "experiment_group": "A",
        "script": os.path.abspath(__file__),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "torch": torch.__version__,
        "config": {k: v for k, v in vars(args).items() if k != "arm"},
        "predictions": {},
        "scale_caveat": (
            "CIFAR-scale, ResNet-18, ~1.5k steps. Decides whether the pure relational "
            "objective reaches a degenerate solution and whether the variance hinge is "
            "saturated on the sphere (that one is arithmetic). Decides nothing about "
            "ImageNet-scale behaviour, long schedules, or downstream transfer gaps."
        ),
    }

    if args.exp in ("a1", "all"):
        res_a1 = run_a1(args)
        summary["predictions"].update(verdicts_a1(res_a1, args))
        summary["A1_raw"] = res_a1

    if args.exp in ("a2", "all"):
        res_a2 = run_a2(args, device)
        summary["predictions"].update(verdicts_a2(res_a2, args))
        summary["A2_raw"] = {
            "collapse_loss_check": res_a2["collapse_loss_check"],
            "per_arm": res_a2["per_arm"],
        }

    summary["elapsed_s"] = round(time.time() - t0, 1)
    out_path = os.path.join(args.outdir, "A_summary.json")
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)

    print("\n" + "=" * 78)
    print("VERDICTS")
    print("=" * 78)
    for key, block in summary["predictions"].items():
        print(f"[{key}] {block['verdict'].upper()}")
        print(f"     {block['statement']}")
        ev = block["evidence"]
        shown = {k: v for k, v in list(ev.items())[:4]}
        print(f"     evidence: {json.dumps(shown, default=str)}")
    print(f"\nsummary -> {out_path}")
    print(f"total elapsed: {summary['elapsed_s']} s")
    if args.smoke:
        print("\nSMOKE RUN: pipeline only. The verdicts above are NOT evidence about "
              "any thesis (8 steps, tiny model, possibly synthetic images).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
