#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xp_F_frozen_substrate.py -- "Frozen versus fine-tuned; and the lossless substrate".

Theses under test: T7 (frozen vs fine-tuned) and T8 (a latent with no loss of its
own transfers better than a latent shaped by a direct reconstruction).

=============================================================================
WHAT IS CLAIMED, WHAT WOULD FALSIFY IT
=============================================================================

F1 -- T7. FROZEN VERSUS FINE-TUNED
    Claim.       A frozen encoder loses on any single task taken alone and wins
                 on universality.  Universality is measured on the WORST task of
                 a heterogeneous battery and on the spread across tasks, never
                 on the mean.
    Prediction.  (P1) full fine-tuning beats the linear probe on essentially
                 every task taken in isolation;
                 (P2) the frozen artefact has the better WORST task once the
                 comparison is made between artefacts that must serve every
                 task (see the universality matrix below);
                 (P3) the frozen artefact has the smaller across-task standard
                 deviation.
    Falsifiers.  P1 fails if fine-tuning does not beat the probe on a clear
                 majority of tasks.  P2 fails if the universality worst case of
                 the fine-tuned specialists is >= that of the frozen encoder.
                 P3 fails if the frozen across-task sd is >= the fine-tuned one.
                 All three are decided with 3 seeds and the explicit rule in
                 ``verdict()`` (a separation rule, NOT a p-value -- 3 seeds do
                 not support a real test, and the script says so in the output).

    METHODOLOGICAL AMENDMENT (important, read before believing any number).
    The naive form of F1 -- "fine-tune on task j, report task j; probe on task
    j, report task j; then take the worst over j" -- compares the weakest
    SPECIALIST against the weakest read-out of ONE generalist.  Those are not
    the same kind of object: the fine-tuned column is produced by 7 different
    encoders, one per task, so its "worst case" is not the worst case of any
    artefact anyone could ship.  Reported alone it is close to meaningless.
    We therefore keep it (it is the honest way to state P1) and we ADD the
    measurement that actually instantiates T7:

        UNIVERSALITY MATRIX.  Row 0 is the SSL encoder, frozen.  Row j (j >= 1)
        is the encoder fine-tuned on task j, then frozen.  Every row is probed
        linearly on EVERY task.  ``worst`` of a row is the worst case of that
        single artefact over the whole battery.  T7 says: row j beats row 0 in
        column j (specialisation pays) and loses to row 0 in ``worst`` (it paid
        with universality).  Without this matrix the experiment decides nothing
        about universality, and P2 above is evaluated ON THE MATRIX.

    OPERATIONAL DISINTERESTEDNESS TEST (implemented literally, as requested).
    Before and after the whole downstream stage, every tensor of the frozen
    encoder's ``state_dict`` is SHA-256 hashed.  The hashes must be identical.
    Because a test that cannot fail is not a test, the same hash function is
    used as a POSITIVE CONTROL on the fine-tuning arm, where the hash MUST
    change, and on the source encoder from which the fine-tuned copies are
    deep-copied, where it MUST NOT (this catches deepcopy aliasing).  If the
    positive control does not fire, the frozen verdict is reported as "non
    concluante" rather than "confirmee".

F2 -- T8. THE LOSSLESS SUBSTRATE
    One trunk, one intermediate latent X (a "substrate"), three arms, strictly
    matched pre-training budget (same steps, same batch, same number of encoder
    forward passes, same data order, same seeds):
        (a) plain       -- X has no loss of its own and no decoder; it is only
                           constrained by the agreement of the branches that
                           traverse it;
        (b) decoder     -- a pixel decoder is attached to X, its reconstruction
                           loss back-propagates INTO X;
        (c) decoder_sg  -- same decoder, same parameters, same optimiser, same
                           reconstruction loss, but it reads ``X.detach()``.
    Prediction.  (P5) (a) transfers better than (b) on the battery and on the
                 worst task; (P6) X has a higher effective rank in (a) than in
                 (b).
    Falsifier.   (b) >= (a) on battery worst case, with the seed spread taken
                 into account.

    WHAT ARM (c) IS FOR, STATED PRECISELY.  With per-parameter optimiser state,
    no global gradient clipping, and a step count fixed in advance, a decoder
    that receives ``X.detach()`` cannot influence the encoder AT ALL: its
    gradients never reach a shared parameter, and it consumes no shared budget.
    Arm (c) is therefore PREDICTED TO BE NUMERICALLY IDENTICAL to arm (a), and
    the script verifies it (max absolute parameter difference + linear CKA
    between the two encoders' features).  That makes (c) a null control on the
    EXPERIMENT rather than on the thesis: if (c) differs from (a) by more than
    floating-point noise, then something in the setup leaks -- shared RNG
    consumption, coupled optimiser state, a global clip -- and the (a) vs (b)
    comparison must be thrown away.  This is stronger than the usual reading of
    (c) as a "budget control", and it is the reason the RNG stream is reseeded
    after module construction (see ``pretrain_arm``): otherwise the decoder's
    weight initialisation would consume random numbers and silently change the
    augmentations arm (c) sees.
    The genuine "budget" reading of (c) survives too: (b) vs (c) isolates "the
    decoder DEFORMS X" from "the decoder merely EXISTS and is trained".

=============================================================================
WHAT THIS SCALE CAN AND CANNOT DECIDE -- READ THIS BEFORE QUOTING A NUMBER
=============================================================================
This is CIFAR/DTD/Flowers/Pets/EuroSAT at 32x32 with a ResNet-18, on one A100.
It is not ImageNet, and nothing here transfers automatically to a 300M-image
pre-training regime.  Concretely:
  * absolute accuracies are low by publication standards and are NOT the point;
    only the arm-to-arm differences under a matched budget are;
  * 32x32 is a real handicap for DTD / Flowers102 / OxfordPets, which are
    fine-grained or texture tasks meant for >= 224px.  The handicap is IDENTICAL
    for every arm, so comparisons stay valid, but "DTD accuracy 0.2" must never
    be quoted as a DTD result;
  * the pre-training set (CIFAR-10 by default) is also a battery task.  It is
    flagged ``in_domain`` everywhere and the worst case is reported twice: over
    all tasks, and over out-of-domain tasks only;
  * 3 seeds give a spread, not a significance level.  The verdict rule is a
    separation rule stated in ``verdict()``;
  * T7's universality claim is tested against ONE family of downstream tasks
    (small-image classification).  A battery is not the space of all tasks.

=============================================================================
ESTIMATED COST (A100 40GB, Colab Pro), default settings, 3 seeds
=============================================================================
  pre-training      3 arms x 3 seeds x 15k steps x batch 256, 4 encoder
                    forwards/step, ResNet-18 @32px          ~  9.0 h
  F2 frozen battery 3 arms x 3 seeds x 7 tasks x 2 LRs      ~  2.5 h
                    (arm ``plain`` is computed once and reused as the F1
                     frozen arm -- same encoder, same protocol)
  F1 fine-tuning    3 seeds x 7 tasks x 2 LRs x 30 epochs   ~  4.5 h
  universality mat. 3 seeds x 8 rows x 7 columns            ~  2.5 h
                                                            ---------
                                                            ~ 18-20 h A100
  Everything is checkpointed and resumable at task granularity; a killed Colab
  session costs at most one task.  ``--smoke`` runs the entire pipeline,
  including the aggregation and the figures, in under 3 minutes on CPU with
  synthetic images and no download.

=============================================================================
OUTPUTS
=============================================================================
  <outdir>/*.csv                per-run metrics, flushed and fsync'd per row
  <outdir>/*.ckpt.pt            resumable checkpoints
  <outdir>/records/*.json       one record per (stage, arm/regime, seed)
  <outdir>/fig_F1_per_task.png  frozen vs fine-tuned, per task
  <outdir>/fig_F1_universality.png   universality matrix + worst case per row
  <outdir>/fig_F2_substrate.png transfer / effective rank / invariance per arm
  <outdir>/summary.json         one verdict per prediction, with its number

Usage
-----
    python xp/xp_F_frozen_substrate.py --smoke                  # 3 min, CPU
    python xp/xp_F_frozen_substrate.py --all                    # full grid
    python xp/xp_F_frozen_substrate.py --stage pretrain --seed 0
    python xp/xp_F_frozen_substrate.py --stage aggregate
"""

from __future__ import annotations

import argparse
import copy
import glob as _glob
import hashlib
import json
import math
import os
import random
import sys
import time
import zlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- the one in-house import, per the project contract -----------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.harness import (                                    # noqa: E402
    AugCfg,
    DATASET_NUM_CLASSES,
    DEFAULT_BATTERY,
    Run,
    aggregate_seeds,
    effective_rank,
    get_device,
    get_eval_datasets,
    get_ssl_dataset,
    gram_loss,
    invariance_score,
    linear_cka,
    make_encoder,
    make_mlp,
    make_eval_transform,
    make_single_view_transform,
    make_two_view_transform,
    matched_budget_check,
    set_seed,
    vicreg_reg,
    EMATeacher,
)

from PIL import Image                                        # noqa: E402


PRETRAIN_ARMS = ("plain", "decoder", "decoder_sg")
DOWNSTREAM_REGIMES = ("frozen", "finetune")
VERDICT_CONFIRMED = "confirmee"
VERDICT_REFUTED = "infirmee"
VERDICT_INCONCLUSIVE = "non concluante"


# =============================================================================
# 0.  Small utilities that a sceptical reader should be able to check in a minute
# =============================================================================

def stable_hash(text: str) -> int:
    """A hash that does NOT depend on ``PYTHONHASHSEED``.

    ``hash(str)`` is randomised per interpreter process since Python 3.3, so
    using it to derive a dataset seed makes the synthetic tasks change between
    two invocations of this script.  That would silently break resume, break
    ``--stage battery`` followed by ``--stage finetune``, and make the
    universality matrix compare encoders trained on different data.  CRC-32 of
    the UTF-8 bytes is stable across processes, machines and versions.
    """
    return zlib.crc32(str(text).encode("utf-8")) & 0x7FFFFFFF


def state_hash(module: nn.Module) -> str:
    """SHA-256 over every tensor of ``module.state_dict()``.

    Name, dtype, shape and raw bytes all enter the digest, in sorted key order.
    ``state_dict()`` carries PARAMETERS *and* BUFFERS, which is the whole point:
    a BatchNorm ``running_mean`` / ``running_var`` / ``num_batches_tracked``
    moves on any forward pass taken in ``train()`` mode, with no gradient and
    no optimiser involved.  That is the classic silent violation of "the frozen
    encoder is not touched", and a hash restricted to ``parameters()`` would
    miss it entirely.  ``_bn_positive_control`` below proves this channel is
    live, so the test cannot pass vacuously.

    Two modules have the same hash if and only if they hold bit-identical
    tensors.
    """
    h = hashlib.sha256()
    sd = module.state_dict()
    for key in sorted(sd.keys()):
        t = sd[key]
        h.update(key.encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def max_abs_state_diff(m1: nn.Module, m2: nn.Module) -> float:
    """Largest absolute difference between two modules' state dicts (inf if the
    key sets differ).  Used for the arm (a) == arm (c) setup check, where exact
    bit equality is too strict on GPU (cuDNN reductions are not deterministic
    by default) but a difference above ~1e-5 means a real leak."""
    s1, s2 = m1.state_dict(), m2.state_dict()
    if set(s1.keys()) != set(s2.keys()):
        return float("inf")
    worst = 0.0
    for k in s1:
        a = s1[k].detach().float().cpu()
        b = s2[k].detach().float().cpu()
        if a.shape != b.shape:
            return float("inf")
        if a.numel() == 0:
            continue
        worst = max(worst, float((a - b).abs().max()))
    return worst


def _bn_positive_control(encoder: nn.Module, sample: torch.Tensor) -> dict:
    """POSITIVE CONTROL for the normalisation-buffer channel of ``state_hash``.

    A "the frozen encoder is bit-identical" test is worthless unless we know it
    would FIRE on the failure it is supposed to catch.  The failure that does
    not involve gradients at all is this one: a forward pass taken while the
    module is in ``train()`` mode updates every BatchNorm running statistic.

    So we deep-copy the encoder, push one batch through it in ``train()`` mode
    under ``no_grad`` -- no optimiser, no backward, nothing but a forward -- and
    check that the hash CHANGED.  The original encoder is untouched (we work on
    the copy), and we check that too.  If this control does not fire, the
    encoder has no normalisation buffers, or ``state_hash`` is blind to them,
    and the frozen verdict must be reported as inconclusive.
    """
    before = state_hash(encoder)
    probe = copy.deepcopy(encoder)
    probe.train()
    with torch.no_grad():
        probe(sample)
    fired = (state_hash(probe) != before)
    return {"bn_control_fired": bool(fired),
            "original_untouched": bool(state_hash(encoder) == before)}


def rng_state_dict() -> dict:
    """Snapshot every RNG stream, so that a resumed run continues the SAME
    augmentation and shuffling sequence as an uninterrupted one."""
    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else None),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def load_rng_state_dict(d: Optional[dict]) -> None:
    if not d:
        return
    torch.set_rng_state(torch.as_tensor(d["torch"], dtype=torch.uint8))
    if d.get("torch_cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(d["torch_cuda"])
        except Exception as e:                       # pragma: no cover
            print(f"[rng] could not restore CUDA RNG state: {e}")
    np.random.set_state(d["numpy"])
    random.setstate(d["python"])


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    v = [float(x) for x in values if x is not None and not math.isnan(float(x))]
    if not v:
        return float("nan"), float("nan")
    m = float(np.mean(v))
    s = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return m, s


MIN_SEEDS_FOR_A_VERDICT = 2      # below this there is no spread at all
MIN_SEEDS_RECOMMENDED = 3        # project rule: 3 seeds minimum


def verdict(vals_a: Sequence[float], vals_b: Sequence[float],
            expect: str = "a_greater") -> dict:
    """Decide one prediction from two per-seed samples.

    RULE (stated so nobody mistakes it for a significance test).  With 3 seeds
    there is no honest p-value.  We use a separation rule: let ``delta`` be the
    difference of the means and ``band = sd_a + sd_b`` the sum of the seed
    standard deviations.  The prediction is
        confirmee        if delta has the expected sign and |delta| > band
        infirmee         if delta has the opposite sign and |delta| > band
        non concluante   otherwise (the seed spread swallows the effect)
    ``band`` is deliberately conservative: it is wider than any standard error,
    so "confirmee" means the arms are separated by more than their combined
    spread, not merely by more than the noise on their means.

    TWO HARD REFUSALS, because both failure modes produce confident nonsense:

    1. FEWER THAN 2 SEEDS.  With one seed ``sd`` is 0 by convention, so
       ``band == 0`` and ANY difference, however microscopic, would come out
       "confirmee".  We refuse outright.  With exactly 2 seeds we answer but
       stamp ``underpowered: True`` (the project rule is 3).
    2. A NaN ANYWHERE IN THE INPUT.  A NaN here means a battery task failed,
       and ``battery_stats`` deliberately propagates it into ``worst``.
       Averaging over the surviving seeds would convert "one arm crashed on one
       task" into "that arm looks fine", which is exactly the silent bug the
       harness NaN policy exists to prevent.  We refuse and say so.
    """
    va = [float(x) for x in vals_a] if vals_a is not None else []
    vb = [float(x) for x in vals_b] if vals_b is not None else []
    n_a, n_b = len(va), len(vb)
    base = {"expect": expect, "n_a": n_a, "n_b": n_b,
            "underpowered": bool(min(n_a, n_b) < MIN_SEEDS_RECOMMENDED)}

    nan_a = sum(1 for x in va if math.isnan(x))
    nan_b = sum(1 for x in vb if math.isnan(x))
    if nan_a or nan_b:
        ma, sa = mean_std(va)
        mb, sb = mean_std(vb)
        return {**base, "verdict": VERDICT_INCONCLUSIVE,
                "reason": (f"NaN in the input ({nan_a} of {n_a} / {nan_b} of "
                           f"{n_b}); a NaN marks a failed battery task and is "
                           f"NOT dropped"),
                "mean_a": ma, "sd_a": sa, "mean_b": mb, "sd_b": sb,
                "delta": float("nan"), "band": float("nan")}

    if min(n_a, n_b) < MIN_SEEDS_FOR_A_VERDICT:
        ma, sa = mean_std(va)
        mb, sb = mean_std(vb)
        return {**base, "verdict": VERDICT_INCONCLUSIVE,
                "reason": (f"fewer than {MIN_SEEDS_FOR_A_VERDICT} seeds "
                           f"(n_a={n_a}, n_b={n_b}): with no spread the "
                           f"separation rule degenerates and confirms anything"),
                "mean_a": ma, "sd_a": sa, "mean_b": mb, "sd_b": sb,
                "delta": (ma - mb) if not (math.isnan(ma) or math.isnan(mb))
                         else float("nan"),
                "band": float("nan")}

    ma, sa = mean_std(va)
    mb, sb = mean_std(vb)
    delta = ma - mb
    band = sa + sb
    want_positive = (expect == "a_greater")
    ok_sign = (delta > 0) if want_positive else (delta < 0)
    v = (VERDICT_CONFIRMED if ok_sign else VERDICT_REFUTED) \
        if abs(delta) > band else VERDICT_INCONCLUSIVE
    return {**base, "verdict": v, "mean_a": ma, "sd_a": sa,
            "mean_b": mb, "sd_b": sb, "delta": delta, "band": band}


def set_transform(ds: Dataset, transform: Callable) -> None:
    """Install ``transform`` on a torchvision dataset, recursing through the
    ``Subset`` wrappers the harness uses for EuroSAT."""
    target = ds
    while isinstance(target, Subset):
        target = target.dataset
    if not hasattr(target, "transform"):
        raise TypeError(f"{type(target).__name__} has no .transform attribute")
    target.transform = transform


def json_dump(path: str, obj: dict) -> None:
    """Atomic JSON write (Colab dies mid-write more often than one expects)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, AugCfg):
        return o.to_dict()
    return str(o)


def json_load(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:                                   # pragma: no cover
        print(f"[record] {path} unreadable ({e}); recomputing.")
        return None


# =============================================================================
# 1.  Data.  Real torchvision datasets, or synthetic images for --smoke
# =============================================================================

class SyntheticImages(Dataset):
    """Offline stand-in for a small image dataset, used ONLY by ``--smoke``.

    Each class gets a base hue and a stripe orientation; each sample adds
    noise.  The signal is weak but learnable, so the linear probe and the
    fine-tuning path both produce accuracies above chance and every code path
    (LR selection, early comparison, figures) is genuinely exercised without
    downloading 170 MB.  It mimics the torchvision interface: ``__getitem__``
    returns ``(PIL.Image, int)`` and the object carries a ``.transform``.
    """

    def __init__(self, n: int, size: int, num_classes: int, seed: int,
                 transform: Optional[Callable] = None):
        self.n = int(n)
        self.size = int(size)
        self.num_classes = int(num_classes)
        self.transform = transform
        rng = np.random.RandomState(int(seed))
        self.labels = rng.randint(0, self.num_classes, size=self.n)
        self._seeds = rng.randint(0, 2 ** 31 - 1, size=self.n)
        self._hues = rng.rand(self.num_classes, 3) * 0.6 + 0.2

    def __len__(self) -> int:
        return self.n

    def _render(self, i: int) -> Image.Image:
        y = int(self.labels[i])
        rng = np.random.RandomState(int(self._seeds[i]))
        s = self.size
        base = np.ones((s, s, 3), dtype=np.float32) * self._hues[y][None, None, :]
        coords = np.arange(s, dtype=np.float32)
        if y % 2 == 0:
            stripe = np.sin(coords * (0.4 + 0.1 * y))[:, None]
        else:
            stripe = np.sin(coords * (0.4 + 0.1 * y))[None, :]
        base += 0.15 * stripe[..., None]
        base += rng.normal(scale=0.10, size=base.shape).astype(np.float32)
        arr = np.clip(base, 0.0, 1.0) * 255.0
        return Image.fromarray(arr.astype(np.uint8), mode="RGB")

    def __getitem__(self, i: int):
        img = self._render(int(i))
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i])


class TwoViewWrapper(Dataset):
    """Two-view SSL view over any ``(image, label)`` dataset.

    Mirrors the harness ``SSLDataset`` contract -- ``{'v1','v2','idx','label'}``
    -- and exposes ``raw(i)`` so ``invariance_score`` can re-augment the same
    source image.  Only used for synthetic smoke data; real runs use the
    harness ``get_ssl_dataset``.  The label is diagnostic only and never enters
    a loss here either.
    """

    def __init__(self, base: Dataset, two_view: Callable):
        self.base = base
        self.two_view = two_view

    def __len__(self) -> int:
        return len(self.base)

    def raw(self, i: int):
        old = self.base.transform
        try:
            self.base.transform = None
            item = self.base[int(i)]
        finally:
            self.base.transform = old
        return item[0]

    def __getitem__(self, i: int) -> dict:
        img = self.raw(i)
        v1, v2 = self.two_view(img)
        return {"v1": v1, "v2": v2, "idx": int(i),
                "label": int(self.base.labels[int(i)])}


@dataclass
class TaskData:
    """Everything one downstream task needs, built once."""
    name: str
    num_classes: int
    fit_ds: Dataset          # training subset, augmented if --probe-aug weak
    val_ds: Dataset          # held-out subset of TRAIN, deterministic transform
    test_ds: Dataset         # official test split, deterministic transform
    n_fit: int


_TASK_CACHE: Dict[tuple, TaskData] = {}


def build_task(name: str, size: int, root: str, synthetic: bool,
               probe_aug: str, val_frac: float, split_seed: int,
               smoke_n_train: int = 48, smoke_n_test: int = 24) -> TaskData:
    """Build the (fit, val, test) triple for one downstream task.

    The validation split is carved out of the TRAIN split with a fixed
    permutation: the test split is never touched by model selection.  Both
    downstream regimes (frozen probe and fine-tuning) receive exactly this
    object, so they see the same images, the same augmentation and the same
    split -- the only difference between the regimes is which parameters sit in
    the optimiser.

    CACHED on the full argument tuple.  The universality matrix asks for the
    same task once per row (8 rows x 7 tasks), and rebuilding a torchvision
    dataset means re-unpickling CIFAR or re-walking the DTD directory tree
    every time.  Nothing downstream mutates a ``TaskData``, so sharing it is
    safe; the cache key includes ``probe_aug``, so the augmented and the
    deterministic variants never alias.
    """
    key = (name, int(size), str(root), bool(synthetic), str(probe_aug),
           float(val_frac), int(split_seed), int(smoke_n_train), int(smoke_n_test))
    if key in _TASK_CACHE:
        return _TASK_CACHE[key]

    if synthetic:
        # NOT `hash(name)`: str hashing is salted per process (PYTHONHASHSEED),
        # so the synthetic task would silently differ between two invocations.
        num_classes = 4 + (stable_hash(name) % 3)
        base_seed = 1000 + (stable_hash(name) % 997)
        train_aug = SyntheticImages(smoke_n_train, size, num_classes, base_seed)
        train_eval = SyntheticImages(smoke_n_train, size, num_classes, base_seed)
        test_eval = SyntheticImages(smoke_n_test, size, num_classes, base_seed + 1)
        train_eval.transform = make_eval_transform(size)
        test_eval.transform = make_eval_transform(size)
        train_aug.transform = (make_single_view_transform(AugCfg.weak(size))
                               if probe_aug == "weak" else make_eval_transform(size))
    else:
        num_classes = DATASET_NUM_CLASSES[name]
        train_eval, test_eval = get_eval_datasets(name, size=size, root=root)
        if probe_aug == "weak":
            train_aug, _ = get_eval_datasets(name, size=size, root=root)
            set_transform(train_aug, make_single_view_transform(AugCfg.weak(size)))
        else:
            train_aug = train_eval

    n = len(train_eval)
    perm = np.random.RandomState(int(split_seed)).permutation(n)
    n_val = max(1, int(round(val_frac * n)))
    val_idx = sorted(int(i) for i in perm[:n_val])
    fit_idx = sorted(int(i) for i in perm[n_val:])

    td = TaskData(
        name=name,
        num_classes=num_classes,
        fit_ds=Subset(train_aug, fit_idx),
        val_ds=Subset(train_eval, val_idx),
        test_ds=test_eval,
        n_fit=len(fit_idx),
    )
    _TASK_CACHE[key] = td
    return td


def build_ssl_dataset(name: str, size: int, root: str, synthetic: bool,
                      seed: int, smoke_n: int = 96) -> Dataset:
    """Two-view SSL dataset for pre-training (strong policy on both branches)."""
    cfg = AugCfg.strong(size)
    if synthetic:
        base = SyntheticImages(smoke_n, size, 5, seed=7)
        return TwoViewWrapper(base, make_two_view_transform(cfg))
    return get_ssl_dataset(name, cfg, cfg_b=None, imbalance_gamma=None,
                           seed=seed, root=root)


# =============================================================================
# 2.  Models.  Trunk = backbone -> substrate X -> projector z.  Decoder is
#     a SEPARATE module so that arms (a) and (c) share an identical online net.
# =============================================================================

class OnlineNet(nn.Module):
    """backbone -> substrate (X) -> projector (z).

    ``X`` is the object of thesis T8: the intermediate latent that either does
    or does not carry a loss of its own.  ``z`` is unnormalised on purpose --
    ``vicreg_reg`` refuses an L2-normalised input (T2), and ``gram_loss``
    normalises internally.
    """

    def __init__(self, arch: str, stem: str, substrate_dim: int,
                 proj_hidden: int, proj_dim: int):
        super().__init__()
        self.backbone, self.feat_dim = make_encoder(arch, dataset=stem)
        self.substrate = make_mlp(self.feat_dim, proj_hidden, substrate_dim,
                                  n_layers=2, bn=True)
        self.projector = make_mlp(substrate_dim, proj_hidden, proj_dim,
                                  n_layers=2, bn=True)
        self.substrate_dim = int(substrate_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        latent = self.substrate(h)
        z = self.projector(latent)
        return {"h": h, "x": latent, "z": z}


class SubstrateEncoder(nn.Module):
    """The shipped artefact: backbone + substrate, returning X.

    This is what gets frozen, fine-tuned, probed and hashed.  It SHARES its
    submodules with the ``OnlineNet`` it was built from (no copy), which is
    precisely what makes the bit-identity test meaningful.
    """

    def __init__(self, online: OnlineNet):
        super().__init__()
        self.backbone = online.backbone
        self.substrate = online.substrate
        self.out_dim = online.substrate_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.substrate(self.backbone(x))


class BackboneEncoder(nn.Module):
    """Secondary read-out point: the backbone features h, before the substrate.

    Reported for F2 only, and only under the frozen probe.  If the arms differ
    at X but not at h, then the reconstruction loss deformed the substrate and
    not the trunk -- which is a real and reportable outcome, not a failure.
    """

    def __init__(self, online: OnlineNet):
        super().__init__()
        self.backbone = online.backbone
        self.out_dim = online.feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class PixelDecoder(nn.Module):
    """Small transposed-convolution decoder: X -> image, trained with pixel MSE.

    Three stride-2 stages, so the spatial start is ``size // 8`` (4 for 32px,
    12 for 96px).  This is the deliberately DIRECT reconstruction of T8: no
    perceptual loss, no adversarial term, just pixels.
    """

    def __init__(self, in_dim: int, size: int, width: int = 128):
        super().__init__()
        if size % 8 != 0:
            raise ValueError(f"decoder needs size divisible by 8, got {size}")
        self.base = size // 8
        self.width = int(width)
        self.fc = nn.Linear(in_dim, self.width * self.base * self.base)
        self.net = nn.Sequential(
            nn.BatchNorm2d(self.width), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.width, self.width // 2, 4, 2, 1),
            nn.BatchNorm2d(self.width // 2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.width // 2, self.width // 4, 4, 2, 1),
            nn.BatchNorm2d(self.width // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.width // 4, 3, 4, 2, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        b = latent.shape[0]
        y = self.fc(latent).view(b, self.width, self.base, self.base)
        return self.net(y)


# =============================================================================
# 3.  Pre-training (shared by F1 and F2)
# =============================================================================

def ssl_step_loss(online: OnlineNet, teacher: EMATeacher,
                  decoder: Optional[PixelDecoder], arm: str,
                  v1: torch.Tensor, v2: torch.Tensor,
                  vicreg_w: float, recon_w: float
                  ) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    """One optimisation step's loss.

    Agreement is the project's own relational target: ``gram_loss`` between the
    student's z on one view and the EMA target's z on the other, symmetrised.
    Anti-collapse is ``vicreg_reg`` on the UNNORMALISED z -- T2 says the Gram
    objective alone has total collapse as a GLOBAL minimum, so the legality term
    is not a refinement here, it is the only thing standing between the run and
    a constant embedding.

    ``arm`` decides the decoder path:
      plain       -> no decoder at all
      decoder     -> decoder(X); the reconstruction gradient enters X
      decoder_sg  -> decoder(X.detach()); the decoder trains, X does not move
    """
    out1 = online(v1)
    out2 = online(v2)
    with torch.no_grad():
        tgt = teacher.target
        t1 = tgt(v1)["z"]
        t2 = tgt(v2)["z"]

    agree = 0.5 * (gram_loss(out1["z"], t2) + gram_loss(out2["z"], t1))

    reg1, parts1 = vicreg_reg(out1["z"])
    reg2, parts2 = vicreg_reg(out2["z"])
    reg = 0.5 * (reg1 + reg2)

    loss = agree + vicreg_w * reg
    recon_value = 0.0
    if decoder is not None:
        latent = out1["x"] if arm == "decoder" else out1["x"].detach()
        recon = F.mse_loss(decoder(latent), v1)
        loss = loss + recon_w * recon
        recon_value = float(recon.detach())

    logs = {
        "loss": float(loss.detach()),
        "agree": float(agree.detach()),
        "vicreg": float(reg.detach()),
        "var": 0.5 * (parts1["var"] + parts2["var"]),
        "cov": 0.5 * (parts1["cov"] + parts2["cov"]),
        "recon": recon_value,
    }
    return loss, logs, out1["x"].detach()


def pretrain_arm(arm: str, seed: int, args, device: torch.device) -> str:
    """Pre-train one arm for one seed.  Returns the checkpoint path.

    RNG discipline (this is what makes arm (a) == arm (c) an informative test):
    modules are built in a FIXED order with the decoder LAST, then the global
    RNG is reseeded immediately before the loop.  Consequently the augmentation
    and shuffling streams do not depend on whether a decoder was constructed.
    The DataLoader also gets its own generator, so worker seeding is decoupled
    from the training stream.
    """
    name = f"pretrain_{arm}_seed{seed}"
    config = {
        "stage": "pretrain", "arm": arm, "seed": seed,
        "dataset": args.pretrain_data, "size": args.size, "arch": args.arch,
        "steps": args.steps, "batch_size": args.bs, "lr": args.lr,
        "weight_decay": args.wd, "substrate_dim": args.substrate_dim,
        "proj_dim": args.proj_dim, "proj_hidden": args.proj_hidden,
        "vicreg_w": args.vicreg_w, "recon_w": args.recon_w,
        "ema_base": args.ema_base, "synthetic": args.synthetic,
        "forward_passes_per_step": 4,
    }
    run = Run(name, args.outdir, config, resume=args.resume, higher_is_better=False)

    set_seed(seed)
    dataset = build_ssl_dataset(args.pretrain_data, args.size, args.data_root,
                               args.synthetic, seed, smoke_n=args.smoke_ssl_n)
    loader_gen = torch.Generator().manual_seed(seed * 7919 + 11)
    loader = DataLoader(dataset, batch_size=args.bs, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        pin_memory=(device.type == "cuda"),
                        generator=loader_gen,
                        persistent_workers=(args.workers > 0))

    # --- module construction, fixed order, decoder LAST ---------------------
    online = OnlineNet(args.arch, args.stem, args.substrate_dim,
                       args.proj_hidden, args.proj_dim).to(device)
    teacher = EMATeacher(online, tau_base=args.ema_base, tau_final=1.0,
                         total_steps=args.steps)
    teacher.to(device)
    decoder = None
    if arm in ("decoder", "decoder_sg"):
        decoder = PixelDecoder(args.substrate_dim, args.size).to(device)

    params = list(online.parameters())
    if decoder is not None:
        params += list(decoder.parameters())
    # NO global gradient clipping: clipping over the union of the parameters
    # would couple the decoder to the encoder and destroy the (a) == (c) check.
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    warmup = max(1, int(0.05 * args.steps))

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / warmup
        t = (step - warmup) / max(1, args.steps - warmup)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))

    start_step = 0
    ck = run.load_ckpt() if args.resume else None
    if ck is not None:
        online.load_state_dict(ck["online"])
        teacher.load_state_dict(ck["teacher"])
        if decoder is not None and ck.get("decoder") is not None:
            decoder.load_state_dict(ck["decoder"])
        opt.load_state_dict(ck["opt"])
        load_rng_state_dict(ck.get("rng"))
        start_step = int(ck["step"])
        print(f"[{name}] resumed at step {start_step}/{args.steps}")
    else:
        # Reseed AFTER construction so the decoder's init draws cannot shift the
        # augmentation stream (see docstring).
        set_seed(seed + 100_000)

    online.train()
    it = iter(loader)
    t0 = time.time()
    for step in range(start_step, args.steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        v1 = batch["v1"].to(device, non_blocking=True)
        v2 = batch["v2"].to(device, non_blocking=True)

        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        loss, logs, latent = ssl_step_loss(online, teacher, decoder, arm, v1, v2,
                                           args.vicreg_w, args.recon_w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        teacher.update(step)

        if (step % args.log_every == 0) or (step == args.steps - 1):
            run.log(step, seed=seed, arm=arm, lr=lr_at(step),
                    tau=teacher.last_tau,
                    rank_x=effective_rank(latent.float().cpu()),
                    imgs_per_s=round((step - start_step + 1) * args.bs
                                     / max(1e-9, time.time() - t0), 1),
                    **logs)
        if ((step + 1) % args.ckpt_every == 0) or (step == args.steps - 1):
            # NO ``best_metric`` here, on purpose.  Passing one would make Run
            # ALSO write <name>.best.pt whenever the SSL loss improves, i.e.
            # almost every save: 200 MB x 2 files x 30 saves x 9 runs is ~100 GB
            # of writes and 3.6 GB of residue, for a "best" checkpoint nobody
            # wants -- the artefact we ship is the encoder at the END of the
            # matched budget, not the one that happened to have the lowest
            # relational loss.  Selecting an SSL checkpoint on its own training
            # loss would also break budget matching between the arms.
            run.save_ckpt(step + 1,
                          online=online.state_dict(),
                          teacher=teacher.state_dict(),
                          decoder=(decoder.state_dict() if decoder else None),
                          opt=opt.state_dict(),
                          rng=rng_state_dict())

    # --- end-of-training diagnostics on the substrate X ----------------------
    diag = pretrain_diagnostics(online, dataset, args, device)
    run.finish({"arm": arm, "seed": seed, **diag,
                "encoder_hash": state_hash(SubstrateEncoder(online))})
    json_dump(os.path.join(args.outdir, "records",
                           f"pretrain_{arm}_seed{seed}.json"),
              {"stage": "pretrain", "arm": arm, "seed": seed,
               "config": config, **diag,
               "encoder_hash": state_hash(SubstrateEncoder(online))})
    print(f"[{name}] done in {time.time() - t0:.1f}s  {diag}")
    return run.ckpt_path


@torch.no_grad()
def pretrain_diagnostics(online: OnlineNet, ssl_dataset: Dataset, args,
                         device: torch.device) -> dict:
    """Effective rank of X and of h, plus the worst-case invariance score.

    ``effective_rank`` is computed on a fixed number of samples with the
    DETERMINISTIC transform, so it measures the geometry of the representation
    and not the spread the augmentations inject.
    """
    online.eval()
    enc_x = SubstrateEncoder(online).to(device).eval()
    n = min(args.diag_samples, len(ssl_dataset))

    # deterministic features: re-render the raw images through the eval pipeline
    tf = make_eval_transform(args.size)
    xs, hs = [], []
    buf = []
    for i in range(n):
        raw = ssl_dataset.raw(i) if hasattr(ssl_dataset, "raw") else None
        if raw is None:                                      # pragma: no cover
            break
        buf.append(tf(raw))
        if len(buf) == 128 or i == n - 1:
            b = torch.stack(buf).to(device)
            out = online(b)
            xs.append(out["x"].float().cpu())
            hs.append(out["h"].float().cpu())
            buf = []
    X = torch.cat(xs, 0) if xs else torch.zeros(2, 2)
    H = torch.cat(hs, 0) if hs else torch.zeros(2, 2)

    inv = invariance_score(enc_x, ssl_dataset, AugCfg.strong(args.size),
                           n_views=args.inv_views, device=device,
                           n_samples=min(args.diag_samples, 128), seed=0)
    online.train()
    return {"rank_x": effective_rank(X), "rank_h": effective_rank(H),
            "invariance_x": float(inv), "diag_n": int(X.shape[0])}


def load_online(arm: str, seed: int, args, device: torch.device) -> OnlineNet:
    """Rebuild an ``OnlineNet`` and load a finished pre-training checkpoint."""
    path = os.path.join(args.outdir, f"pretrain_{arm}_seed{seed}.ckpt.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"missing pre-training checkpoint {path}. Run "
            f"`--stage pretrain --arm {arm} --seed {seed}` first."
        )
    ck = torch.load(path, map_location="cpu", weights_only=False)
    online = OnlineNet(args.arch, args.stem, args.substrate_dim,
                       args.proj_hidden, args.proj_dim)
    online.load_state_dict(ck["online"])
    return online.to(device).eval()


# =============================================================================
# 4.  Downstream.  ONE loop for both regimes; the only difference is which
#     parameters enter the optimiser.  A sceptic can check that in ten lines.
# =============================================================================

def train_downstream(encoder: nn.Module, task: TaskData, finetune: bool,
                     epochs: int, bs: int, lr: float, wd: float,
                     device: torch.device, seed: int, workers: int = 0) -> dict:
    """Train a linear head on ``encoder``'s output; fine-tune the encoder or not.

    The two regimes share: the data, the augmentation, the split, the batch
    size, the number of epochs, the optimiser family, the schedule and the RNG
    seed.  They differ in exactly two lines:

        params = head.parameters()  (+ encoder.parameters() if finetune)
        features = encoder(x)       under no_grad + eval() if not finetune

    Budget note, stated plainly because it favours the fine-tuning arm: both
    regimes perform ONE encoder forward pass per sample per step, so
    ``matched_budget_check`` passes on forward passes; fine-tuning additionally
    performs an encoder BACKWARD pass, which is roughly twice the cost again.
    We do not equalise that.  Giving the fine-tuning arm strictly more compute
    makes any frozen win harder to obtain and therefore more credible.

    The frozen regime asserts, at every step, that no encoder parameter is in
    the optimiser and that ``requires_grad`` is False everywhere -- the
    structural half of the disinterestedness test, complementing the hash.
    """
    set_seed(seed * 131 + (7 if finetune else 3))

    head = nn.Linear(_encoder_out_dim(encoder, task, device), task.num_classes).to(device)

    if finetune:
        encoder.train()
        for p in encoder.parameters():
            p.requires_grad_(True)
        params = list(encoder.parameters()) + list(head.parameters())
    else:
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)
        params = list(head.parameters())
        encoder_ids = {id(p) for p in encoder.parameters()}
        in_opt = encoder_ids & {id(p) for p in params}
        assert not in_opt, "FROZEN REGIME VIOLATION: encoder parameters are in the optimiser"

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    gen = torch.Generator().manual_seed(seed * 977 + int(finetune))
    loader = DataLoader(task.fit_ds, batch_size=bs, shuffle=True,
                        num_workers=workers, drop_last=False,
                        pin_memory=(device.type == "cuda"), generator=gen)
    steps_per_epoch = max(1, len(loader))
    total_steps = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps))

    step = 0
    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if finetune:
                feats = encoder(x)
            else:
                with torch.no_grad():
                    feats = encoder(x)
            logits = head(feats)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1

    if not finetune:
        leaked = [n for n, p in encoder.named_parameters() if p.grad is not None]
        assert not leaked, f"FROZEN REGIME VIOLATION: gradients on {leaked[:3]}"

    val_acc = evaluate(encoder, head, task.val_ds, bs, device, workers)
    test_acc = evaluate(encoder, head, task.test_ds, bs, device, workers)
    return {"val_acc": val_acc, "test_acc": test_acc, "lr": lr,
            "steps": total_steps, "steps_per_epoch": steps_per_epoch}


@torch.no_grad()
def _encoder_out_dim(encoder: nn.Module, task: TaskData, device: torch.device) -> int:
    """Probe the encoder's output width with one real sample (no hard-coding)."""
    was_training = encoder.training
    encoder.eval()
    x, _ = task.val_ds[0]
    out = encoder(x.unsqueeze(0).to(device))
    if was_training:
        encoder.train()
    return int(out.shape[1])


@torch.no_grad()
def evaluate(encoder: nn.Module, head: nn.Module, ds: Dataset, bs: int,
             device: torch.device, workers: int = 0) -> float:
    """Top-1 accuracy with the encoder in eval() mode and no gradient anywhere.

    ``encoder.eval()`` matters even in the fine-tuning regime: leaving BatchNorm
    in training mode during evaluation would let the test batch composition
    change the prediction, which is a silent leak.  The encoder's previous mode
    is restored so that the hash test is not disturbed by a mode flip (modes are
    not part of ``state_dict``, but running statistics are, and a forward in
    ``train()`` mode would move them).
    """
    was_training = encoder.training
    encoder.eval()
    head.eval()
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers,
                        pin_memory=(device.type == "cuda"))
    correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = head(encoder(x)).argmax(dim=1)
        correct += int((pred == y).sum())
        total += int(y.numel())
    head.train()
    if was_training:
        encoder.train()
    return correct / max(1, total)


def select_lr_and_train(encoder_factory: Callable[[], nn.Module], task: TaskData,
                        finetune: bool, epochs: int, bs: int, lrs: Sequence[float],
                        wd: float, device: torch.device, seed: int,
                        workers: int = 0) -> dict:
    """Run the downstream loop once per candidate LR, select on VALIDATION.

    Model selection never sees the test split.  The same grid discipline is
    applied to both regimes, so neither is advantaged by a lucky hyper-parameter
    -- the classic way a frozen/fine-tuned comparison gets rigged.
    ``encoder_factory`` returns the encoder to use for one LR: the frozen regime
    returns the SAME shared object every time (nothing is written to it), the
    fine-tuning regime returns a fresh deep copy.
    """
    per_lr = []
    best = None
    for lr in lrs:
        enc = encoder_factory()
        res = train_downstream(enc, task, finetune, epochs, bs, lr, wd,
                               device, seed, workers)
        res["encoder"] = enc if finetune else None
        per_lr.append({k: v for k, v in res.items() if k != "encoder"})
        if best is None or res["val_acc"] > best["val_acc"]:
            best = res
    out = {"task": task.name, "finetune": bool(finetune),
           "test_acc": best["test_acc"], "val_acc": best["val_acc"],
           "lr": best["lr"], "steps": best["steps"], "per_lr": per_lr}
    if finetune:
        out["encoder"] = best["encoder"]
    return out


def battery_stats(per_task: Dict[str, float], in_domain: Optional[str]) -> dict:
    """Mean / worst / spread over a battery, with the in-domain task isolated.

    ``worst`` is the headline number of T7.  ``worst_ood`` repeats it with the
    pre-training dataset removed, because that task is not a transfer task.
    A NaN anywhere makes ``worst`` NaN: dropping a failed task silently would
    turn the worst case into a best case of the survivors.
    """
    names = list(per_task.keys())
    vals = [per_task[n] for n in names]
    finite = [v for v in vals if not math.isnan(v)]
    has_nan = len(finite) < len(vals)
    ood = [per_task[n] for n in names if n != in_domain]
    ood_finite = [v for v in ood if not math.isnan(v)]
    return {
        "per_task": per_task,
        "mean": float(np.mean(finite)) if finite else float("nan"),
        "worst": float("nan") if (has_nan or not finite) else float(min(finite)),
        "std_across_tasks": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        "worst_ood": (float(min(ood_finite))
                      if ood_finite and len(ood_finite) == len(ood) else float("nan")),
        "mean_ood": float(np.mean(ood_finite)) if ood_finite else float("nan"),
        "n_tasks": len(names),
    }


# =============================================================================
# 5.  F2 -- frozen battery for one pre-training arm
# =============================================================================

def stage_battery(arm: str, seed: int, args, device: torch.device) -> dict:
    """Frozen linear battery for one pre-trained arm (F2, and the F1 frozen arm).

    Resumable at TASK granularity: each finished task is checkpointed, so a
    Colab kill costs one task at most.  The record is cached on disk, and the
    F1 frozen regime reuses the ``plain`` record instead of recomputing it --
    same encoder, same protocol, same numbers.
    """
    rec_path = os.path.join(args.outdir, "records", f"battery_{arm}_seed{seed}.json")
    cached = json_load(rec_path)
    if cached is not None and args.resume and cached.get("complete"):
        print(f"[battery_{arm}_seed{seed}] cached, skipping")
        return cached

    name = f"battery_{arm}_seed{seed}"
    config = {"stage": "battery", "arm": arm, "seed": seed,
              "tasks": args.tasks, "probe_epochs": args.probe_epochs,
              "probe_lrs": args.probe_lrs, "down_bs": args.down_bs,
              "probe_aug": args.probe_aug, "size": args.size,
              "synthetic": args.synthetic}
    run = Run(name, args.outdir, config, resume=args.resume)

    online = load_online(arm, seed, args, device)
    enc_x = SubstrateEncoder(online).to(device)
    enc_h = BackboneEncoder(online).to(device)

    hash_before = state_hash(enc_x)
    done: Dict[str, dict] = {}
    ck = run.load_ckpt() if args.resume else None
    if ck is not None:
        done = ck.get("done", {})
        print(f"[{name}] resumed with {len(done)}/{len(args.tasks)} tasks done")

    for i, tname in enumerate(args.tasks):
        if tname in done:
            continue
        task = build_task(tname, args.size, args.data_root, args.synthetic,
                          args.probe_aug, args.val_frac, split_seed=1234,
                          smoke_n_train=args.smoke_n_train,
                          smoke_n_test=args.smoke_n_test)
        res_x = select_lr_and_train(lambda: enc_x, task, False, args.probe_epochs,
                                    args.down_bs, args.probe_lrs, args.down_wd,
                                    device, seed, args.workers)
        res_h = select_lr_and_train(lambda: enc_h, task, False, args.probe_epochs,
                                    args.down_bs, args.probe_lrs, args.down_wd,
                                    device, seed, args.workers)
        done[tname] = {"acc_x": res_x["test_acc"], "acc_h": res_h["test_acc"],
                       "lr_x": res_x["lr"], "lr_h": res_h["lr"],
                       "steps": res_x["steps"]}
        run.log(i, seed=seed, arm=arm, task=tname,
                acc_x=res_x["test_acc"], acc_h=res_h["test_acc"],
                lr_x=res_x["lr"], lr_h=res_h["lr"])
        run.save_ckpt(i + 1, done=done, best_metric=res_x["test_acc"])
        print(f"[{name}] {tname}: X={res_x['test_acc']:.4f} h={res_h['test_acc']:.4f}")

    # --- disinterestedness: the hash AFTER the whole downstream stage --------
    hash_after = state_hash(enc_x)
    encoder_unchanged = bool(hash_before == hash_after)

    # ... and the positive control that proves the test could have failed.
    sample_task = build_task(args.tasks[0], args.size, args.data_root,
                             args.synthetic, args.probe_aug, args.val_frac,
                             split_seed=1234,
                             smoke_n_train=args.smoke_n_train,
                             smoke_n_test=args.smoke_n_test)
    sx, _ = sample_task.val_ds[0]
    bn_ctl = _bn_positive_control(enc_x, torch.stack([sx, sx]).to(device))

    # FAIL FAST, BEFORE anything is written with complete=True.  If the record
    # were persisted first, a resumed run would read `complete: True` from disk,
    # skip the whole stage and never re-raise -- a real leak would survive as a
    # cached result.  The assert therefore comes first and the record is only
    # written once it has passed.
    assert encoder_unchanged, (
        f"FROZEN ENCODER CHANGED during the linear-probe battery "
        f"(arm={arm}, seed={seed}): {hash_before[:16]} -> {hash_after[:16]}. "
        f"This is a real gradient/BatchNorm-statistics leak, not a formality; "
        f"T7's whole claim of disinterestedness is void until it is fixed.")
    assert bn_ctl["bn_control_fired"], (
        "the BatchNorm positive control did NOT fire: a forward pass in train() "
        "mode left the state_dict hash unchanged, so state_hash is blind to the "
        "normalisation buffers and the frozen test above proves nothing.")
    assert bn_ctl["original_untouched"], (
        "the BatchNorm positive control mutated the real encoder -- the deep "
        "copy aliased it.")

    per_task_x = {t: float(done[t]["acc_x"]) for t in args.tasks}
    per_task_h = {t: float(done[t]["acc_h"]) for t in args.tasks}
    in_domain = None if args.synthetic else args.pretrain_data

    pre = json_load(os.path.join(args.outdir, "records",
                                 f"pretrain_{arm}_seed{seed}.json")) or {}
    record = {
        "stage": "battery", "arm": arm, "seed": seed, "complete": True,
        "config": config,
        "substrate": battery_stats(per_task_x, in_domain),
        "backbone": battery_stats(per_task_h, in_domain),
        "encoder_unchanged": encoder_unchanged,
        "bn_positive_control": bn_ctl,
        "hash_before": hash_before, "hash_after": hash_after,
        "rank_x": pre.get("rank_x"), "rank_h": pre.get("rank_h"),
        "invariance_x": pre.get("invariance_x"),
        "steps_per_task": {t: done[t]["steps"] for t in args.tasks},
    }
    json_dump(rec_path, record)
    run.finish({k: v for k, v in record.items()
                if k not in ("config", "hash_before", "hash_after")})
    return record


# =============================================================================
# 6.  F1 -- fine-tuning arm, universality matrix, disinterestedness test
# =============================================================================

def stage_finetune(seed: int, args, device: torch.device) -> dict:
    """Full fine-tuning of the SSL encoder, one specialist per task (F1).

    Also runs the POSITIVE CONTROL of the hash test: each fine-tuned copy must
    hash DIFFERENTLY from the source, and the source must hash IDENTICALLY
    before and after the whole stage (a deepcopy that silently aliased a
    submodule would show up here and nowhere else).

    The best-LR fine-tuned encoder of each task is kept on disk for the
    universality matrix.
    """
    rec_path = os.path.join(args.outdir, "records", f"finetune_seed{seed}.json")
    cached = json_load(rec_path)
    if cached is not None and args.resume and cached.get("complete"):
        print(f"[finetune_seed{seed}] cached, skipping")
        return cached

    name = f"finetune_seed{seed}"
    config = {"stage": "finetune", "arm": args.f1_arm, "seed": seed,
              "tasks": args.tasks, "ft_epochs": args.ft_epochs,
              "ft_lrs": args.ft_lrs, "down_bs": args.down_bs,
              "probe_aug": args.probe_aug, "size": args.size,
              "synthetic": args.synthetic}
    run = Run(name, args.outdir, config, resume=args.resume)

    online = load_online(args.f1_arm, seed, args, device)
    source_encoder = SubstrateEncoder(online).to(device)
    source_hash = state_hash(source_encoder)

    enc_dir = os.path.join(args.outdir, "ft_encoders")
    os.makedirs(enc_dir, exist_ok=True)

    done: Dict[str, dict] = {}
    ck = run.load_ckpt() if args.resume else None
    if ck is not None:
        done = ck.get("done", {})
        print(f"[{name}] resumed with {len(done)}/{len(args.tasks)} tasks done")

    for i, tname in enumerate(args.tasks):
        if tname in done and os.path.exists(
                os.path.join(enc_dir, f"ft_{tname}_seed{seed}.pt")):
            continue
        task = build_task(tname, args.size, args.data_root, args.synthetic,
                          args.probe_aug, args.val_frac, split_seed=1234,
                          smoke_n_train=args.smoke_n_train,
                          smoke_n_test=args.smoke_n_test)

        def factory():
            return copy.deepcopy(source_encoder).to(device)

        res = select_lr_and_train(factory, task, True, args.ft_epochs,
                                  args.down_bs, args.ft_lrs, args.down_wd,
                                  device, seed, args.workers)
        ft_enc = res.pop("encoder")
        ft_hash = state_hash(ft_enc)
        torch.save({"state_dict": ft_enc.state_dict(), "task": tname,
                    "seed": seed, "hash": ft_hash},
                   os.path.join(enc_dir, f"ft_{tname}_seed{seed}.pt"))
        done[tname] = {"acc": res["test_acc"], "lr": res["lr"],
                       "steps": res["steps"], "hash": ft_hash,
                       "hash_changed": bool(ft_hash != source_hash)}
        run.log(i, seed=seed, task=tname, acc=res["test_acc"], lr=res["lr"],
                hash_changed=int(ft_hash != source_hash))
        run.save_ckpt(i + 1, done=done, best_metric=res["test_acc"])
        print(f"[{name}] {tname}: ft acc={res['test_acc']:.4f} lr={res['lr']}")

    source_hash_after = state_hash(source_encoder)
    # Fail before persisting complete=True, for the same reason as in
    # stage_battery: a cached record would hide the leak on the next resume.
    assert source_hash == source_hash_after, (
        f"fine-tuning modified the SOURCE encoder (seed={seed}): the deep copy "
        f"aliased it, so every 'fine-tuned' specialist after the first was "
        f"trained on top of the previous one and the whole F1 arm is void.")
    per_task = {t: float(done[t]["acc"]) for t in args.tasks}
    in_domain = None if args.synthetic else args.pretrain_data
    record = {
        "stage": "finetune", "seed": seed, "complete": True, "config": config,
        "specialists": battery_stats(per_task, in_domain),
        "positive_control_all_changed": all(done[t]["hash_changed"] for t in args.tasks),
        "source_unchanged_by_deepcopy": bool(source_hash == source_hash_after),
        "source_hash": source_hash,
        "steps_per_task": {t: done[t]["steps"] for t in args.tasks},
    }
    json_dump(rec_path, record)
    run.finish({k: v for k, v in record.items() if k != "config"})
    return record


def stage_universality(seed: int, args, device: torch.device) -> dict:
    """The universality matrix: every artefact probed on every task (F1c).

    Row ``ssl_frozen``  = the SSL encoder, frozen.
    Row ``ft_<task j>`` = the encoder fine-tuned on task j, then frozen.
    Column i            = linear probe accuracy on task i.

    Every cell uses the SAME frozen protocol, one fixed LR (``--univ-lr``) and
    ``--univ-epochs``, so the rows are comparable to each other.  Cells are not
    comparable to the F1 headline numbers, which use the full LR grid and more
    epochs; the matrix is an internal comparison and is labelled as such.

    Resumable cell by cell.
    """
    rec_path = os.path.join(args.outdir, "records", f"universality_seed{seed}.json")
    cached = json_load(rec_path)
    if cached is not None and args.resume and cached.get("complete"):
        print(f"[universality_seed{seed}] cached, skipping")
        return cached

    name = f"universality_seed{seed}"
    config = {"stage": "universality", "seed": seed, "tasks": args.tasks,
              "univ_epochs": args.univ_epochs, "univ_lr": args.univ_lr,
              "down_bs": args.down_bs, "size": args.size,
              "synthetic": args.synthetic}
    run = Run(name, args.outdir, config, resume=args.resume)

    online = load_online(args.f1_arm, seed, args, device)
    ssl_encoder = SubstrateEncoder(online).to(device)
    ssl_hash_before = state_hash(ssl_encoder)

    rows = ["ssl_frozen"] + [f"ft_{t}" for t in args.tasks]
    cells: Dict[str, Dict[str, float]] = {}
    ck = run.load_ckpt() if args.resume else None
    if ck is not None:
        cells = ck.get("cells", {})

    enc_dir = os.path.join(args.outdir, "ft_encoders")
    flat = 0
    for row in rows:
        cells.setdefault(row, {})
        if row == "ssl_frozen":
            encoder = ssl_encoder
        else:
            p = os.path.join(enc_dir, f"ft_{row[3:]}_seed{seed}.pt")
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"missing fine-tuned encoder {p}; run `--stage finetune` first")
            encoder = copy.deepcopy(ssl_encoder)
            encoder.load_state_dict(torch.load(p, map_location="cpu",
                                               weights_only=False)["state_dict"])
            encoder = encoder.to(device)
        for tname in args.tasks:
            flat += 1
            if tname in cells[row]:
                continue
            task = build_task(tname, args.size, args.data_root, args.synthetic,
                              args.probe_aug, args.val_frac, split_seed=1234,
                              smoke_n_train=args.smoke_n_train,
                              smoke_n_test=args.smoke_n_test)
            res = train_downstream(encoder, task, False, args.univ_epochs,
                                   args.down_bs, args.univ_lr, args.down_wd,
                                   device, seed, args.workers)
            cells[row][tname] = float(res["test_acc"])
            run.log(flat, seed=seed, row=row, task=tname, acc=res["test_acc"])
            run.save_ckpt(flat, cells=cells, best_metric=res["test_acc"])
        print(f"[{name}] row {row}: " +
              " ".join(f"{t}={cells[row][t]:.3f}" for t in args.tasks))

    ssl_hash_after = state_hash(ssl_encoder)
    assert ssl_hash_before == ssl_hash_after, (
        f"the frozen SSL encoder changed while building the universality "
        f"matrix (seed={seed}): {ssl_hash_before[:16]} -> {ssl_hash_after[:16]}")

    in_domain = None if args.synthetic else args.pretrain_data
    per_row = {r: battery_stats({t: cells[r][t] for t in args.tasks}, in_domain)
               for r in rows}
    specialists = [r for r in rows if r != "ssl_frozen"]
    spec_worst = [per_row[r]["worst"] for r in specialists]
    spec_std = [per_row[r]["std_across_tasks"] for r in specialists]

    def _nan_safe(fn, vals):
        """NaN in -> NaN out.  ``max``/``np.mean`` over a list containing NaN is
        order-dependent garbage for ``max`` and silently poisons ``mean``; a NaN
        here means a battery task failed and MUST stay visible (harness NaN
        policy).  Never drop it."""
        v = [float(x) for x in vals]
        if not v or any(math.isnan(x) for x in v):
            return float("nan")
        return float(fn(v))

    record = {
        "stage": "universality", "seed": seed, "complete": True, "config": config,
        "cells": cells, "rows": rows, "per_row": per_row,
        "ssl_worst": per_row["ssl_frozen"]["worst"],
        "ssl_mean": per_row["ssl_frozen"]["mean"],
        "ssl_std_across_tasks": per_row["ssl_frozen"]["std_across_tasks"],
        "ssl_encoder_unchanged": True,
        # A specialist that must serve the whole battery is judged on ITS worst
        # task.  We report the best such specialist -- the most favourable
        # reading for the fine-tuning side.
        "best_specialist_worst": _nan_safe(max, spec_worst),
        "mean_specialist_worst": _nan_safe(np.mean, spec_worst),
        "mean_specialist_std": _nan_safe(np.mean, spec_std),
        "specialisation_gain": _nan_safe(np.mean, [
            cells[f"ft_{t}"][t] - cells["ssl_frozen"][t] for t in args.tasks]),
    }
    json_dump(rec_path, record)
    run.finish({k: v for k, v in record.items() if k not in ("config", "cells")})
    return record


def stage_setup_check(seed: int, args, device: torch.device) -> dict:
    """Verify the F2 null control: arm (a) and arm (c) must be the SAME encoder.

    See the module docstring.  A difference above ``--identity-tol`` means the
    stop-gradient decoder somehow influenced the encoder -- shared RNG, coupled
    optimiser state, a global clip -- and the whole (a) vs (b) comparison is
    void.  We report both the max absolute parameter difference and the linear
    CKA between the two encoders' features on a fixed batch (CKA is the
    functional version of the question: identical weights are sufficient but a
    reader may prefer to see the representations compared directly).
    """
    try:
        online_a = load_online("plain", seed, args, device)
        online_c = load_online("decoder_sg", seed, args, device)
    except FileNotFoundError as e:
        print(f"[setup_check] skipped: {e}")
        return {"available": False}

    enc_a = SubstrateEncoder(online_a).to(device).eval()
    enc_c = SubstrateEncoder(online_c).to(device).eval()
    diff = max_abs_state_diff(enc_a, enc_c)

    ds = build_ssl_dataset(args.pretrain_data, args.size, args.data_root,
                           args.synthetic, seed, smoke_n=args.smoke_ssl_n)
    tf = make_eval_transform(args.size)
    n = min(args.diag_samples, len(ds), 256)
    batch = torch.stack([tf(ds.raw(i)) for i in range(n)]).to(device)
    with torch.no_grad():
        fa = enc_a(batch).float().cpu()
        fc = enc_c(batch).float().cpu()
    cka = linear_cka(fa, fc)

    rec = {"available": True, "seed": seed, "max_abs_param_diff": diff,
           "cka_a_vs_c": float(cka), "tol": args.identity_tol,
           "arms_identical": bool(diff <= args.identity_tol)}
    json_dump(os.path.join(args.outdir, "records", f"setupcheck_seed{seed}.json"), rec)
    print(f"[setup_check seed={seed}] max|dW|={diff:.3e} CKA={cka:.6f} "
          f"identical={rec['arms_identical']}")
    return rec


# =============================================================================
# 7.  Aggregation, figures, verdicts
# =============================================================================

def _collect(args, pattern: str) -> List[dict]:
    return [r for r in (json_load(p) for p in
                        sorted(_glob.glob(os.path.join(args.outdir, "records", pattern))))
            if r is not None]


def aggregate(args) -> dict:
    """Read every record, build the figures, write ``summary.json``."""
    batteries = {arm: _collect(args, f"battery_{arm}_seed*.json") for arm in PRETRAIN_ARMS}
    finetunes = _collect(args, "finetune_seed*.json")
    universals = _collect(args, "universality_seed*.json")
    setupchecks = _collect(args, "setupcheck_seed*.json")

    summary: dict = {
        "experiment": "F -- frozen vs fine-tuned (T7) and lossless substrate (T8)",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scale_caveat": ("CIFAR/DTD/Flowers/Pets/EuroSAT at 32x32, ResNet-18, "
                         "one A100. Arm-to-arm differences only; absolute "
                         "accuracies are not comparable to published numbers."),
        "verdict_rule": ("confirmee iff |mean difference| > (sd_a + sd_b) with "
                         "the predicted sign; this is a separation rule over "
                         "seeds, NOT a significance test."),
        "n_seeds": {f"battery_{args.f1_arm}": len(batteries[args.f1_arm]),
                    "finetune": len(finetunes),
                    "universality": len(universals)},
        "min_seeds_recommended": MIN_SEEDS_RECOMMENDED,
        "tasks": args.tasks,
        "f1_arm": args.f1_arm,
        "in_domain_task": None if args.synthetic else args.pretrain_data,
    }

    # ---------------- F1 -----------------------------------------------------
    f1: dict = {}
    # The F1 frozen arm MUST be the same pre-trained arm the fine-tuning arm
    # starts from, otherwise "frozen vs fine-tuned" silently becomes
    # "arm plain vs arm <f1_arm> fine-tuned" and measures T8, not T7.
    frozen_recs = batteries[args.f1_arm]
    if frozen_recs and finetunes:
        by_seed_frozen = {r["seed"]: r for r in frozen_recs}
        by_seed_ft = {r["seed"]: r for r in finetunes}
        seeds = sorted(set(by_seed_frozen) & set(by_seed_ft))
        f1["seeds"] = seeds

        per_task_frozen = {t: [by_seed_frozen[s]["substrate"]["per_task"][t]
                               for s in seeds] for t in args.tasks}
        per_task_ft = {t: [by_seed_ft[s]["specialists"]["per_task"][t]
                           for s in seeds] for t in args.tasks}
        f1["per_task"] = {
            t: {"frozen_mean": mean_std(per_task_frozen[t])[0],
                "frozen_sd": mean_std(per_task_frozen[t])[1],
                "finetune_mean": mean_std(per_task_ft[t])[0],
                "finetune_sd": mean_std(per_task_ft[t])[1]}
            for t in args.tasks}

        wins = [t for t in args.tasks
                if f1["per_task"][t]["finetune_mean"] > f1["per_task"][t]["frozen_mean"]]
        clear_wins = [t for t in args.tasks
                      if (f1["per_task"][t]["finetune_mean"]
                          - f1["per_task"][t]["frozen_mean"])
                      > (f1["per_task"][t]["finetune_sd"] + f1["per_task"][t]["frozen_sd"])]
        f1["P1_finetune_wins_per_task"] = {
            "n_tasks": len(args.tasks), "n_wins": len(wins),
            "n_clear_wins": len(clear_wins), "clear_wins": clear_wins,
            "verdict": (VERDICT_CONFIRMED if len(clear_wins) >= math.ceil(0.7 * len(args.tasks))
                        else (VERDICT_REFUTED if len(wins) <= len(args.tasks) / 2
                              else VERDICT_INCONCLUSIVE)),
            "note": ("clear win = mean gap wider than the summed seed sd; the "
                     "fine-tuning arm also gets an encoder BACKWARD pass the "
                     "frozen arm does not pay for.")}

        f1["naive_worst_frozen"] = [by_seed_frozen[s]["substrate"]["worst"] for s in seeds]
        f1["naive_worst_specialists"] = [by_seed_ft[s]["specialists"]["worst"] for s in seeds]
        f1["naive_worst_note"] = (
            "NOT a like-for-like comparison: the fine-tuned column is produced "
            "by one encoder per task. Read P2 on the universality matrix.")
        f1["std_across_tasks"] = {
            "frozen": [by_seed_frozen[s]["substrate"]["std_across_tasks"] for s in seeds],
            "finetune_specialists": [by_seed_ft[s]["specialists"]["std_across_tasks"]
                                     for s in seeds]}

        pc = all(r.get("positive_control_all_changed") for r in finetunes)
        src_ok = all(r.get("source_unchanged_by_deepcopy") for r in finetunes)
        frozen_ok = all(r.get("encoder_unchanged") for r in frozen_recs)
        f1["P4_disinterestedness"] = {
            "frozen_encoder_bit_identical": bool(frozen_ok),
            "positive_control_finetune_changed": bool(pc),
            "source_unchanged_by_deepcopy": bool(src_ok),
            "verdict": (VERDICT_CONFIRMED if (frozen_ok and pc and src_ok)
                        else (VERDICT_INCONCLUSIVE if not pc
                              else VERDICT_REFUTED)),
            "note": ("without the positive control a passing hash test proves "
                     "nothing about the hash function.")}

    if universals:
        by_seed_u = {r["seed"]: r for r in universals}
        useeds = sorted(by_seed_u)
        ssl_worst = [by_seed_u[s]["ssl_worst"] for s in useeds]
        best_spec_worst = [by_seed_u[s]["best_specialist_worst"] for s in useeds]
        mean_spec_worst = [by_seed_u[s]["mean_specialist_worst"] for s in useeds]
        f1["P2_universality_worst_case"] = {
            "ssl_frozen_worst": ssl_worst,
            "best_specialist_worst": best_spec_worst,
            "mean_specialist_worst": mean_spec_worst,
            "vs_best_specialist": verdict(ssl_worst, best_spec_worst, "a_greater"),
            "vs_mean_specialist": verdict(ssl_worst, mean_spec_worst, "a_greater"),
            "note": ("compares artefacts that must serve the whole battery; "
                     "the 'best specialist' reading is the most favourable one "
                     "for fine-tuning.")}
        f1["P2_verdict"] = f1["P2_universality_worst_case"]["vs_best_specialist"]["verdict"]
        f1["P3_spread"] = verdict(
            [by_seed_u[s]["mean_specialist_std"] for s in useeds],
            [by_seed_u[s]["ssl_std_across_tasks"] for s in useeds], "a_greater")
        f1["P3_verdict"] = f1["P3_spread"]["verdict"]
        f1["specialisation_gain"] = mean_std(
            [by_seed_u[s]["specialisation_gain"] for s in useeds])
    summary["F1"] = f1

    # ---------------- F2 -----------------------------------------------------
    f2: dict = {}
    have = {arm: {r["seed"]: r for r in batteries[arm]} for arm in PRETRAIN_ARMS}
    common = sorted(set.intersection(*[set(have[a]) for a in PRETRAIN_ARMS])) \
        if all(have[a] for a in PRETRAIN_ARMS) else []
    f2["seeds"] = common
    if common:
        def col(arm, path):
            out = []
            for s in common:
                r = have[arm][s]
                v = r
                for k in path:
                    v = v[k]
                out.append(float(v))
            return out

        for arm in PRETRAIN_ARMS:
            f2[arm] = {
                "battery_mean": col(arm, ("substrate", "mean")),
                "battery_worst": col(arm, ("substrate", "worst")),
                "battery_worst_ood": col(arm, ("substrate", "worst_ood")),
                "backbone_mean": col(arm, ("backbone", "mean")),
                "rank_x": [have[arm][s].get("rank_x") for s in common],
                "invariance_x": [have[arm][s].get("invariance_x") for s in common],
                "per_task_mean": {t: mean_std([have[arm][s]["substrate"]["per_task"][t]
                                               for s in common])[0]
                                  for t in args.tasks},
            }

        f2["P5_plain_beats_decoder_worst"] = verdict(
            f2["plain"]["battery_worst"], f2["decoder"]["battery_worst"], "a_greater")
        f2["P5_plain_beats_decoder_mean"] = verdict(
            f2["plain"]["battery_mean"], f2["decoder"]["battery_mean"], "a_greater")
        f2["P5_verdict"] = f2["P5_plain_beats_decoder_worst"]["verdict"]
        f2["P6_rank_x"] = verdict(
            [v for v in f2["plain"]["rank_x"] if v is not None],
            [v for v in f2["decoder"]["rank_x"] if v is not None], "a_greater")
        f2["decoder_vs_sg_worst"] = verdict(
            f2["decoder_sg"]["battery_worst"], f2["decoder"]["battery_worst"], "a_greater")
        f2["invariance_plain_vs_decoder"] = verdict(
            [v for v in f2["plain"]["invariance_x"] if v is not None],
            [v for v in f2["decoder"]["invariance_x"] if v is not None], "a_greater")

    # --- P7, the null control that GATES P5 ----------------------------------
    # The module docstring is explicit: a stop-gradient decoder cannot reach the
    # encoder, so arm (c) must equal arm (a) to floating-point noise.  If it
    # does not, something couples the arms and (a) vs (b) is void.  A MISSING
    # control is treated exactly like a failed one -- otherwise forgetting to
    # run `--stage setupcheck` silently upgrades an unvalidated P5 to a verdict.
    avail = [r for r in setupchecks if r.get("available")]
    if avail:
        diffs = [r["max_abs_param_diff"] for r in avail]
        ckas = [r["cka_a_vs_c"] for r in avail]
        ok = all(r.get("arms_identical") for r in avail)
        f2["P7_null_control_a_equals_c"] = {
            "max_abs_param_diff": diffs, "cka": ckas, "tol": args.identity_tol,
            "n_seeds": len(avail),
            "verdict": VERDICT_CONFIRMED if ok else VERDICT_REFUTED,
            "note": ("a stop-gradient decoder cannot reach the encoder; if this "
                     "fails, the budget matching leaks and F2 must be discarded.")}
        if not ok and f2.get("P5_verdict"):
            f2["P5_verdict"] = VERDICT_INCONCLUSIVE
            f2["P5_invalidated_by"] = "P7_null_control_a_equals_c failed"
    else:
        f2["P7_null_control_a_equals_c"] = {
            "verdict": VERDICT_INCONCLUSIVE, "n_seeds": 0,
            "note": ("no setupcheck record found; run `--stage setupcheck`. "
                     "Until it passes, P5 cannot be read.")}
        if f2.get("P5_verdict"):
            f2["P5_verdict"] = VERDICT_INCONCLUSIVE
            f2["P5_invalidated_by"] = "P7_null_control_a_equals_c not run"
    summary["F2"] = f2

    summary["verdicts"] = {
        "T7_P1_finetune_wins_each_task_alone":
            f1.get("P1_finetune_wins_per_task", {}).get("verdict", VERDICT_INCONCLUSIVE),
        "T7_P2_frozen_wins_universality_worst_case":
            f1.get("P2_verdict", VERDICT_INCONCLUSIVE),
        "T7_P3_frozen_smaller_across_task_spread":
            f1.get("P3_verdict", VERDICT_INCONCLUSIVE),
        "T7_P4_frozen_encoder_bit_identical":
            f1.get("P4_disinterestedness", {}).get("verdict", VERDICT_INCONCLUSIVE),
        "T8_P5_lossless_substrate_transfers_better":
            f2.get("P5_verdict", VERDICT_INCONCLUSIVE),
        "T8_P6_lossless_substrate_higher_effective_rank":
            f2.get("P6_rank_x", {}).get("verdict", VERDICT_INCONCLUSIVE),
        "SETUP_P7_stopgrad_decoder_is_a_no_op":
            f2.get("P7_null_control_a_equals_c", {}).get("verdict", VERDICT_INCONCLUSIVE),
    }

    json_dump(os.path.join(args.outdir, "summary.json"), summary)
    make_figures(args, summary)
    print(json.dumps(summary["verdicts"], indent=2, ensure_ascii=False))
    return summary


# ---------------------------------------------------------------------------
# figures: grayscale + hatches, readable when printed in black and white
# ---------------------------------------------------------------------------

_HATCH = ["", "///", "...", "xxx"]
_GRAY = ["0.25", "0.65", "0.45", "0.85"]


def _bar_pair(ax, labels, series, series_names, ylabel, title):
    n = len(series)
    width = 0.8 / max(1, n)
    xs = np.arange(len(labels))
    for i, (name, (means, sds)) in enumerate(zip(series_names, series)):
        ax.bar(xs + i * width - 0.4 + width / 2, means, width, yerr=sds,
               capsize=3, label=name, color=_GRAY[i % len(_GRAY)],
               hatch=_HATCH[i % len(_HATCH)], edgecolor="black", linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", linestyle=":", linewidth=0.5)
    ax.legend(fontsize=8)


def make_figures(args, summary: dict) -> None:
    f1 = summary.get("F1", {})
    f2 = summary.get("F2", {})

    # ---- F1: per task, frozen vs fine-tuned --------------------------------
    if f1.get("per_task"):
        tasks = args.tasks
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        fr = ([f1["per_task"][t]["frozen_mean"] for t in tasks],
              [f1["per_task"][t]["frozen_sd"] for t in tasks])
        ft = ([f1["per_task"][t]["finetune_mean"] for t in tasks],
              [f1["per_task"][t]["finetune_sd"] for t in tasks])
        _bar_pair(axes[0], tasks, [fr, ft], ["frozen (linear probe)", "fine-tuned"],
                  "top-1 accuracy", "F1 -- per task (each task in isolation)")
        labels, series_f, series_g = [], [], []
        if "P2_universality_worst_case" in f1:
            u = f1["P2_universality_worst_case"]
            labels = ["worst task\n(universality matrix)"]
            series_f = [mean_std(u["ssl_frozen_worst"])]
            series_g = [mean_std(u["best_specialist_worst"])]
        if "std_across_tasks" in f1:
            labels.append("sd across tasks")
            series_f.append(mean_std(f1["std_across_tasks"]["frozen"]))
            series_g.append(mean_std(f1["std_across_tasks"]["finetune_specialists"]))
        if labels:
            _bar_pair(axes[1], labels,
                      [([m for m, _ in series_f], [s for _, s in series_f]),
                       ([m for m, _ in series_g], [s for _, s in series_g])],
                      ["frozen SSL encoder", "fine-tuned specialists"],
                      "accuracy / sd", "F1 -- universality (worst case, spread)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "fig_F1_per_task.png"), dpi=150)
        plt.close(fig)

    # ---- F1c: universality matrix ------------------------------------------
    urecs = _collect(args, "universality_seed*.json")
    if urecs:
        rows = urecs[0]["rows"]
        tasks = args.tasks
        M = np.full((len(rows), len(tasks)), np.nan)
        for i, r in enumerate(rows):
            for j, t in enumerate(tasks):
                vals = [rec["cells"][r][t] for rec in urecs
                        if r in rec["cells"] and t in rec["cells"][r]]
                if vals:
                    M[i, j] = float(np.mean(vals))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6),
                                 gridspec_kw={"width_ratios": [1.4, 1.0]})
        im = axes[0].imshow(M, cmap="gray", aspect="auto")
        axes[0].set_xticks(range(len(tasks)))
        axes[0].set_xticklabels(tasks, rotation=30, ha="right", fontsize=8)
        axes[0].set_yticks(range(len(rows)))
        axes[0].set_yticklabels(rows, fontsize=8)
        axes[0].set_title("Universality matrix: artefact (row) probed on task (column)",
                          fontsize=10)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    axes[0].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                                 fontsize=7,
                                 color="white" if M[i, j] < np.nanmean(M) else "black")
        fig.colorbar(im, ax=axes[0], fraction=0.03)
        worsts = np.nanmin(M, axis=1)
        axes[1].barh(range(len(rows)), worsts, color=_GRAY[0], edgecolor="black")
        axes[1].set_yticks(range(len(rows)))
        axes[1].set_yticklabels(rows, fontsize=8)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("worst task of that artefact")
        axes[1].set_title("T7: universality = worst case of ONE artefact", fontsize=10)
        axes[1].grid(axis="x", linestyle=":", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "fig_F1_universality.png"), dpi=150)
        plt.close(fig)

    # ---- F2: three arms ----------------------------------------------------
    if f2.get("seeds") and all(a in f2 for a in PRETRAIN_ARMS):
        metrics = [("battery_mean", "battery mean acc"),
                   ("battery_worst", "battery worst acc"),
                   ("rank_x", "effective rank of X"),
                   ("invariance_x", "invariance score of X")]
        fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.8))
        for ax, (key, title) in zip(axes, metrics):
            means, sds = [], []
            for arm in PRETRAIN_ARMS:
                vals = [v for v in f2[arm].get(key, []) if v is not None]
                m, s = mean_std(vals)
                means.append(m)
                sds.append(s)
            ax.bar(range(len(PRETRAIN_ARMS)), means, yerr=sds, capsize=3,
                   color=[_GRAY[i] for i in range(len(PRETRAIN_ARMS))],
                   hatch=[_HATCH[i] for i in range(len(PRETRAIN_ARMS))],
                   edgecolor="black", linewidth=0.6)
            ax.set_xticks(range(len(PRETRAIN_ARMS)))
            ax.set_xticklabels(["(a) plain", "(b) decoder", "(c) decoder+sg"],
                               rotation=20, ha="right", fontsize=8)
            ax.set_title(title, fontsize=10)
            ax.grid(axis="y", linestyle=":", linewidth=0.5)
        fig.suptitle("F2 -- lossless substrate (T8); (c) is the null control: it must equal (a)",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "fig_F2_substrate.png"), dpi=150)
        plt.close(fig)

    # ---- pre-training curves (uses the harness seed aggregator) -------------
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, arm in enumerate(PRETRAIN_ARMS):
            df = aggregate_seeds(os.path.join(args.outdir,
                                              f"pretrain_{arm}_seed*.csv"), "step")
            if "agree_mean" not in df.columns:
                continue
            ax.plot(df["step"], df["agree_mean"], color=_GRAY[i], label=f"{arm} (agreement)")
            if "agree_std" in df.columns:
                lo = df["agree_mean"] - df["agree_std"].fillna(0)
                hi = df["agree_mean"] + df["agree_std"].fillna(0)
                ax.fill_between(df["step"], lo, hi, color=_GRAY[i], alpha=0.25)
        ax.set_xlabel("step")
        ax.set_ylabel("Gram agreement loss")
        ax.set_title("Pre-training, mean +/- sd over seeds", fontsize=10)
        ax.grid(linestyle=":", linewidth=0.5)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "fig_F_pretrain_curves.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[figures] pre-training curve skipped: {e}")


# =============================================================================
# 8.  CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="F -- frozen vs fine-tuned (T7) and the lossless substrate (T8)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--stage", default="all",
                   choices=["all", "pretrain", "battery", "finetune",
                            "universality", "setupcheck", "aggregate"],
                   help="which stage to run; 'all' runs the whole pipeline")
    p.add_argument("--arm", default="all",
                   choices=list(PRETRAIN_ARMS) + ["all"],
                   help="pre-training arm(s) for the pretrain/battery stages")
    p.add_argument("--all", action="store_true",
                   help="shorthand: every arm, every seed, every stage")
    p.add_argument("--seed", type=int, default=None, help="run a single seed")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="seeds to run (>=3 for any comparison to mean anything)")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="wipe csv/checkpoints and start clean")
    p.add_argument("--outdir",
                   default=os.path.join(_ROOT, "results", "F"))
    p.add_argument("--data-root", default=os.path.join(_ROOT, "data"))
    p.add_argument("--cpu", action="store_true", help="force CPU")
    p.add_argument("--workers", type=int, default=4)

    g = p.add_argument_group("pre-training (A100 defaults)")
    g.add_argument("--steps", type=int, default=15000)
    g.add_argument("--bs", type=int, default=256)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--wd", type=float, default=1e-4)
    g.add_argument("--arch", default="resnet18")
    g.add_argument("--stem", default="cifar", choices=["cifar", "stl", "imagenet"])
    g.add_argument("--pretrain-data", default="cifar10")
    g.add_argument("--size", type=int, default=32)
    g.add_argument("--substrate-dim", type=int, default=256)
    g.add_argument("--proj-dim", type=int, default=256)
    g.add_argument("--proj-hidden", type=int, default=1024)
    g.add_argument("--vicreg-w", type=float, default=1.0,
                   help="weight of the legality term (T2); it is the ONLY thing "
                        "preventing total collapse of the Gram objective")
    g.add_argument("--recon-w", type=float, default=1.0,
                   help="weight of the pixel reconstruction in arms (b) and (c)")
    g.add_argument("--ema-base", type=float, default=0.996)
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--ckpt-every", type=int, default=500)
    g.add_argument("--diag-samples", type=int, default=2048)
    g.add_argument("--inv-views", type=int, default=8)

    d = p.add_argument_group("downstream")
    d.add_argument("--tasks", nargs="+", default=list(DEFAULT_BATTERY))
    d.add_argument("--down-bs", type=int, default=256)
    d.add_argument("--down-wd", type=float, default=1e-4)
    d.add_argument("--probe-epochs", type=int, default=30)
    d.add_argument("--probe-lrs", type=float, nargs="+", default=[1e-2, 1e-3])
    d.add_argument("--ft-epochs", type=int, default=30)
    d.add_argument("--ft-lrs", type=float, nargs="+", default=[1e-3, 1e-4])
    d.add_argument("--probe-aug", default="weak", choices=["weak", "none"],
                   help="both regimes see the SAME input distribution")
    d.add_argument("--val-frac", type=float, default=0.1,
                   help="held-out fraction of TRAIN used for LR selection")
    d.add_argument("--univ-epochs", type=int, default=15)
    d.add_argument("--univ-lr", type=float, default=1e-2)
    d.add_argument("--f1-arm", default="plain", choices=list(PRETRAIN_ARMS),
                   help="which pre-trained arm F1 evaluates (the T8-(a) trunk)")
    d.add_argument("--identity-tol", type=float, default=1e-5,
                   help="max |dW| tolerated between arms (a) and (c)")

    s = p.add_argument_group("smoke")
    s.add_argument("--smoke", action="store_true",
                   help="tiny synthetic end-to-end run, < 3 min on CPU, no download")
    s.add_argument("--synthetic", action="store_true",
                   help="use synthetic images instead of torchvision datasets")
    s.add_argument("--smoke-real-data", action="store_true",
                   help="run --smoke against the real torchvision datasets "
                        "(downloads); off by default so the smoke test is offline")
    s.add_argument("--smoke-ssl-n", type=int, default=96)
    s.add_argument("--smoke-n-train", type=int, default=48)
    s.add_argument("--smoke-n-test", type=int, default=24)
    return p


def apply_smoke(args) -> None:
    """Shrink everything to a pipeline test.  Every code path still runs:
    three pre-training arms, the LR grid, the frozen battery, the fine-tuning
    arm with its hash controls, the universality matrix, the (a)==(c) null
    control, the aggregation, the figures and ``summary.json``."""
    args.steps = 4
    args.bs = 16
    args.down_bs = 16
    args.probe_epochs = 1
    args.ft_epochs = 1
    args.univ_epochs = 1
    args.probe_lrs = [1e-2, 1e-3]
    args.ft_lrs = [1e-3, 1e-4]
    args.seeds = [0, 1]
    args.workers = 0
    args.log_every = 1
    args.ckpt_every = 2
    args.diag_samples = 32
    args.inv_views = 3
    args.size = 32
    args.cpu = True
    args.val_frac = 0.2
    args.identity_tol = 1e-5
    if not args.smoke_real_data:
        args.synthetic = True
        args.tasks = ["synth_a", "synth_b"]
        args.pretrain_data = "synthetic"
    else:
        args.tasks = ["cifar10"]
        args.pretrain_data = "cifar10"
    if args.outdir.endswith(os.path.join("results", "F")):
        args.outdir = os.path.join(_ROOT, "results", "F_smoke")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke:
        apply_smoke(args)
    if args.all:
        args.stage = "all"
        args.arm = "all"
    if args.seed is not None:
        args.seeds = [args.seed]

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "records"), exist_ok=True)
    device = get_device(prefer_cpu=args.cpu)
    arms = list(PRETRAIN_ARMS) if args.arm == "all" else [args.arm]

    print("=" * 78)
    print("XP F -- frozen vs fine-tuned (T7) / lossless substrate (T8)")
    print(f"device={device}  stage={args.stage}  arms={arms}  seeds={args.seeds}")
    print(f"outdir={args.outdir}")
    if len(args.seeds) < 3 and args.stage in ("all", "aggregate"):
        print("WARNING: fewer than 3 seeds -- no comparison in this file decides "
              "anything at that point.")
    print("=" * 78)

    stage = args.stage

    # --- budget declarations, checked BEFORE anything is spent ---------------
    if stage in ("all", "pretrain") and len(arms) > 1:
        matched_budget_check([
            {"name": f"pretrain_{a}", "steps": args.steps, "batch_size": args.bs,
             "forward_passes": 4} for a in arms])
    if stage in ("all", "finetune", "battery"):
        print("Downstream budget: both regimes run the same number of epochs, "
              "the same batch size and ONE encoder forward per sample per step. "
              "Fine-tuning additionally pays an encoder BACKWARD pass; that "
              "surplus is left in its favour on purpose.")

    if stage in ("all", "pretrain"):
        for seed in args.seeds:
            for arm in arms:
                pretrain_arm(arm, seed, args, device)

    if stage in ("all", "battery"):
        for seed in args.seeds:
            for arm in arms:
                stage_battery(arm, seed, args, device)

    if stage in ("all", "finetune"):
        for seed in args.seeds:
            rec_f = stage_finetune(seed, args, device)
            rec_b = json_load(os.path.join(args.outdir, "records",
                                           f"battery_{args.f1_arm}_seed{seed}.json"))
            if rec_b is not None:
                steps_frozen = sum(rec_b["steps_per_task"].values())
                steps_ft = sum(rec_f["steps_per_task"].values())
                matched_budget_check([
                    {"name": "downstream_frozen", "steps": steps_frozen,
                     "batch_size": args.down_bs, "forward_passes": 1},
                    {"name": "downstream_finetune", "steps": steps_ft,
                     "batch_size": args.down_bs, "forward_passes": 1}])

    if stage in ("all", "universality"):
        for seed in args.seeds:
            stage_universality(seed, args, device)

    if stage in ("all", "setupcheck"):
        for seed in args.seeds:
            stage_setup_check(seed, args, device)

    if stage in ("all", "aggregate"):
        aggregate(args)

    print(f"\nDone. Artefacts in {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
