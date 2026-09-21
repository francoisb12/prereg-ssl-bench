"""
harness.py -- shared substrate for the Uebergang-SSL falsification experiments.

This module is the ONLY in-house dependency of the experiment scripts.  Every
signature below is NORMATIVE: the six experiment groups are written against it
in parallel, so nothing here may change name, order or meaning.

Design rules enforced in this file
----------------------------------
* Correctness before elegance.  A detach bug, a shape bug or a train/test leak
  costs A100-weeks, so the code is explicit and defensive rather than short.
* Labels are DIAGNOSTIC ONLY.  ``get_ssl_dataset`` returns them in a separate
  dict key so that no loss can accidentally consume them.
* Anything that is meant to be frozen is stored as a *buffer*, not a
  ``nn.Parameter``, so it can never be swept into an optimiser by accident.
* Everything that writes to disk is crash-safe (atomic replace, per-line flush)
  because Colab sessions get killed mid-run.

Run ``python harness.py`` for a <60 s CPU self-test of every public function.
Add ``--with-data`` to additionally exercise the torchvision download paths.
"""

from __future__ import annotations

import contextlib
import copy
import csv
import glob as _glob
import json
import math
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

try:  # torchvision is required for the data / encoder paths only.
    import torchvision
    from torchvision import transforms as TVT
    from PIL import Image

    _HAS_TV = True
    _TV_IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover - environment dependent
    torchvision = None
    TVT = None
    Image = None
    _HAS_TV = False
    _TV_IMPORT_ERROR = _e


__all__ = [
    "set_seed",
    "get_device",
    "AugCfg",
    "make_two_view_transform",
    "make_single_view_transform",
    "make_eval_transform",
    "get_ssl_dataset",
    "get_eval_datasets",
    "longtail_indices",
    "make_encoder",
    "make_mlp",
    "EMATeacher",
    "frozen_random_projections",
    "gram_loss",
    "prototype_loss",
    "vicreg_reg",
    "info_nce",
    "linear_probe",
    "knn_probe",
    "probe_battery",
    "effective_rank",
    "invariance_score",
    "linear_cka",
    "uniformity_alignment",
    "Run",
    "aggregate_seeds",
    "matched_budget_check",
    "DATASET_NUM_CLASSES",
    "DEFAULT_BATTERY",
]


def _require_tv(what: str) -> None:
    if not _HAS_TV:
        raise ImportError(
            f"{what} needs torchvision (and PIL), which failed to import: "
            f"{_TV_IMPORT_ERROR!r}.  Install with "
            f"`pip install torchvision` matching your torch build."
        )


# ---------------------------------------------------------------------------
# reproducibility / device
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU + all CUDA devices).

    Does NOT force ``torch.use_deterministic_algorithms``: full determinism
    would forbid several cudnn kernels and slow A100 training appreciably.
    Seeding is enough for the seed-averaged comparisons this project makes
    (we always report mean +/- std over >= 3 seeds).
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(prefer_cpu: bool = False) -> torch.device:
    """Return the compute device; ``prefer_cpu`` forces CPU (used by --smoke)."""
    if prefer_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@contextlib.contextmanager
def _rng_island(seed: Optional[int] = None):
    """Run a block without disturbing the caller's global RNG streams.

    Evaluation helpers (probes, invariance_score) draw random numbers.  Called
    periodically from inside a training loop they would otherwise shift -- or,
    if they reseeded, outright reset -- the training stream, so that "same
    seed" runs would silently diverge depending on how often one evaluates.
    Everything below saves and restores torch/numpy/python RNG state.
    """
    t_state = torch.get_rng_state()
    c_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    n_state = np.random.get_state()
    p_state = random.getstate()
    try:
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed) % (2 ** 32))
            random.seed(int(seed))
        yield
    finally:
        torch.set_rng_state(t_state)
        if c_state is not None:
            torch.cuda.set_rng_state_all(c_state)
        np.random.set_state(n_state)
        random.setstate(p_state)


# ---------------------------------------------------------------------------
# augmentations
# ---------------------------------------------------------------------------

# A single dataset-agnostic normalisation is used everywhere (mean = std = 0.5).
# Rationale: several experiments (T6, T7, T8) transfer one frozen encoder across
# heterogeneous datasets; per-dataset statistics would silently change the input
# distribution between pre-training and probing.
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)


class _ToRGB:
    """PIL -> 3-channel RGB.  A module-level class, not a lambda, so that the
    transform survives pickling by DataLoader workers (spawn on Windows)."""

    def __call__(self, img):
        return img.convert("RGB")

    def __repr__(self):
        return "_ToRGB()"


class _TwoViewTransform:
    """Callable returning ``(v1, v2)`` from one image."""

    def __init__(self, tf_a: Callable, tf_b: Callable):
        self.tf_a = tf_a
        self.tf_b = tf_b

    def __call__(self, img):
        return self.tf_a(img), self.tf_b(img)

    def __repr__(self):
        return f"_TwoViewTransform(a={self.tf_a}, b={self.tf_b})"


@dataclass
class AugCfg:
    """Augmentation policy.

    Attributes
    ----------
    crop_scale : (float, float)   area range of RandomResizedCrop
    color_strength : float        SimCLR ``s``; 0.0 disables colour jitter
    gray_p : float                p(random grayscale)
    blur_p : float                p(gaussian blur)
    hflip_p : float               p(horizontal flip)
    solarize_p : float            p(solarize), threshold 0.5 on [0,1] tensors
    size : int                    output spatial size

    ``crop_scale == (1.0, 1.0)`` together with all probabilities at 0 and
    ``color_strength == 0`` gives the deterministic identity policy, which is
    exactly the degenerate ``T = identity`` of thesis T5.
    """

    crop_scale: Tuple[float, float] = (0.2, 1.0)
    color_strength: float = 0.5
    gray_p: float = 0.2
    blur_p: float = 0.0
    hflip_p: float = 0.5
    solarize_p: float = 0.0
    size: int = 32

    # ---- presets -----------------------------------------------------------
    @classmethod
    def weak(cls, size: int = 32) -> "AugCfg":
        return cls(crop_scale=(0.6, 1.0), color_strength=0.2, gray_p=0.0,
                   blur_p=0.0, hflip_p=0.5, solarize_p=0.0, size=size)

    @classmethod
    def strong(cls, size: int = 32) -> "AugCfg":
        return cls(crop_scale=(0.08, 1.0), color_strength=1.0, gray_p=0.2,
                   blur_p=0.5, hflip_p=0.5, solarize_p=0.1, size=size)

    @classmethod
    def none(cls, size: int = 32) -> "AugCfg":
        """The identity policy: resize only, no stochasticity at all."""
        return cls(crop_scale=(1.0, 1.0), color_strength=0.0, gray_p=0.0,
                   blur_p=0.0, hflip_p=0.0, solarize_p=0.0, size=size)

    def is_identity(self) -> bool:
        return (tuple(self.crop_scale) == (1.0, 1.0) and self.color_strength == 0.0
                and self.gray_p == 0.0 and self.blur_p == 0.0
                and self.hflip_p == 0.0 and self.solarize_p == 0.0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["crop_scale"] = list(self.crop_scale)
        return d


def make_single_view_transform(cfg: AugCfg) -> Callable:
    """Build the one-view pipeline described by ``cfg``.

    Order: PIL geometric/photometric ops -> ToTensor -> solarize -> normalise.
    Solarize is applied on the tensor (threshold 0.5 in [0,1]) so it behaves
    identically for 8-bit and float sources.
    """
    _require_tv("make_single_view_transform")
    size = int(cfg.size)
    ops: List[Callable] = [_ToRGB()]

    lo, hi = float(cfg.crop_scale[0]), float(cfg.crop_scale[1])
    if (lo, hi) == (1.0, 1.0):
        ops.append(TVT.Resize((size, size), antialias=True))
    else:
        ops.append(TVT.RandomResizedCrop(size, scale=(lo, hi), antialias=True))

    if cfg.hflip_p > 0:
        ops.append(TVT.RandomHorizontalFlip(p=float(cfg.hflip_p)))

    if cfg.color_strength > 0:
        s = float(cfg.color_strength)
        jitter = TVT.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, min(0.5, 0.2 * s))
        ops.append(TVT.RandomApply([jitter], p=0.8))

    if cfg.gray_p > 0:
        ops.append(TVT.RandomGrayscale(p=float(cfg.gray_p)))

    if cfg.blur_p > 0:
        k = max(3, int(0.1 * size))
        if k % 2 == 0:
            k += 1
        blur = TVT.GaussianBlur(kernel_size=k, sigma=(0.1, 2.0))
        ops.append(TVT.RandomApply([blur], p=float(cfg.blur_p)))

    ops.append(TVT.ToTensor())

    if cfg.solarize_p > 0:
        ops.append(TVT.RandomSolarize(threshold=0.5, p=float(cfg.solarize_p)))

    ops.append(TVT.Normalize(NORM_MEAN, NORM_STD))
    return TVT.Compose(ops)


def make_two_view_transform(cfg_a: AugCfg, cfg_b: AugCfg = None) -> Callable:
    """Return a callable ``img -> (v1, v2)``.

    ``cfg_b=None`` means "same policy on both branches" (symmetric SimCLR-style
    sampling: the two views still differ because the ops are stochastic).
    Asymmetric policies (T6: strong on one branch only) pass an explicit
    ``cfg_b``.
    """
    tf_a = make_single_view_transform(cfg_a)
    tf_b = make_single_view_transform(cfg_b if cfg_b is not None else cfg_a)
    return _TwoViewTransform(tf_a, tf_b)


def make_eval_transform(size: int) -> Callable:
    """Deterministic evaluation pipeline: resize to (size,size), normalise."""
    _require_tv("make_eval_transform")
    return TVT.Compose([
        _ToRGB(),
        TVT.Resize((int(size), int(size)), antialias=True),
        TVT.ToTensor(),
        TVT.Normalize(NORM_MEAN, NORM_STD),
    ])


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------

DATASET_NUM_CLASSES: Dict[str, int] = {
    "cifar10": 10,
    "cifar100": 100,
    "stl10": 10,
    "stl10_unlabeled": 10,
    "mnist": 10,
    "fashionmnist": 10,
    "dtd": 47,
    "flowers102": 102,
    "oxfordpets": 37,
    "eurosat": 10,
}

# Heterogeneous battery for T7 ("always report the WORST task, never the mean").
DEFAULT_BATTERY = ["cifar10", "cifar100", "dtd", "flowers102", "oxfordpets",
                   "eurosat", "fashionmnist"]

_ALIASES = {
    "cifar-10": "cifar10", "cifar_10": "cifar10",
    "cifar-100": "cifar100", "cifar_100": "cifar100",
    "stl-10": "stl10", "stl": "stl10",
    "fashion": "fashionmnist", "fashion_mnist": "fashionmnist",
    "pets": "oxfordpets", "oxfordiiitpet": "oxfordpets",
    "flowers": "flowers102",
}


def _canon(name: str) -> str:
    n = str(name).strip().lower().replace(" ", "")
    n = _ALIASES.get(n, n)
    if n not in DATASET_NUM_CLASSES:
        raise ValueError(
            f"unknown dataset {name!r}; allowed: {sorted(DATASET_NUM_CLASSES)}"
        )
    return n


def _build_raw(name: str, train: bool, root: str, transform):
    """Instantiate a torchvision dataset with the given transform.

    Only account-free, auto-downloading datasets are allowed (project rule).
    Datasets without an official test split (EuroSAT) are split here with a
    FIXED seed so that train and test never overlap across calls.
    """
    _require_tv("get_ssl_dataset / get_eval_datasets")
    n = _canon(name)
    D = torchvision.datasets
    os.makedirs(root, exist_ok=True)

    if n == "cifar10":
        return D.CIFAR10(root, train=train, download=True, transform=transform)
    if n == "cifar100":
        return D.CIFAR100(root, train=train, download=True, transform=transform)
    if n == "stl10":
        split = "train" if train else "test"
        return D.STL10(root, split=split, download=True, transform=transform)
    if n == "stl10_unlabeled":
        # SSL pre-training pool.  Labels are -1 -> diagnostics are meaningless
        # here, which is fine because labels never enter a loss anyway.
        split = "train+unlabeled" if train else "test"
        return D.STL10(root, split=split, download=True, transform=transform)
    if n == "mnist":
        return D.MNIST(root, train=train, download=True, transform=transform)
    if n == "fashionmnist":
        return D.FashionMNIST(root, train=train, download=True, transform=transform)
    if n == "dtd":
        return D.DTD(root, split="train" if train else "test", download=True,
                     transform=transform)
    if n == "flowers102":
        return D.Flowers102(root, split="train" if train else "test",
                            download=True, transform=transform)
    if n == "oxfordpets":
        return D.OxfordIIITPet(root, split="trainval" if train else "test",
                               download=True, transform=transform)
    if n == "eurosat":
        full = D.EuroSAT(root, download=True, transform=transform)
        g = np.random.RandomState(12345)          # FIXED: identical every call
        perm = g.permutation(len(full))
        cut = int(0.8 * len(full))
        keep = perm[:cut] if train else perm[cut:]
        return Subset(full, sorted(int(i) for i in keep))
    raise ValueError(n)  # pragma: no cover


def _dataset_labels(ds) -> np.ndarray:
    """Best-effort label vector for a dataset, WITHOUT running its transform."""
    if isinstance(ds, Subset):
        base = _dataset_labels(ds.dataset)
        return base[np.asarray(ds.indices, dtype=np.int64)]
    for attr in ("targets", "labels", "_labels", "_samples"):
        if hasattr(ds, attr):
            v = getattr(ds, attr)
            if attr == "_samples":  # DTD/Flowers style list of (path, label)
                try:
                    return np.asarray([int(t[1]) for t in v], dtype=np.int64)
                except Exception:
                    continue
            if isinstance(v, torch.Tensor):
                return v.detach().cpu().numpy().astype(np.int64)
            try:
                return np.asarray(v, dtype=np.int64)
            except Exception:
                continue
    raise AttributeError(
        f"cannot recover labels from {type(ds).__name__} without decoding "
        f"images; pass an explicit label array."
    )


def longtail_indices(labels, num_classes: int, gamma: float, seed: int) -> List[int]:
    """Exponential long-tail profile.

    The classic profile (Cui et al. 2019, Cao et al. 2019) is

        n_c = n_0 * gamma ** (c / (C - 1)),    c = 0 .. C-1

    so class 0 keeps all its ``n_0`` samples and class C-1 keeps a fraction
    ``gamma`` of it.  ``gamma`` is therefore the *tail/head ratio*: gamma=0.01
    is the usual "imbalance factor 100".  ``gamma=1.0`` returns a balanced
    subset (every class truncated to the smallest class count).

    ``n_0`` is the smallest per-class count of the source dataset, so the
    profile is achievable for every class.  Sampling is uniform without
    replacement inside each class, driven by ``seed``.

    Returns a sorted list of indices into ``labels``.
    """
    labels = np.asarray(labels).astype(np.int64).reshape(-1)
    C = int(num_classes)
    gamma = float(gamma)
    if not (0.0 < gamma <= 1.0):
        raise ValueError(f"gamma must be in (0,1]; got {gamma}")
    rng = np.random.RandomState(int(seed))

    per_class = [np.where(labels == c)[0] for c in range(C)]
    counts = np.array([len(ix) for ix in per_class])
    if (counts == 0).any():
        missing = np.where(counts == 0)[0].tolist()
        raise ValueError(f"classes {missing} are empty in `labels`")
    n0 = int(counts.min())

    keep: List[int] = []
    denom = max(1, C - 1)
    for c in range(C):
        n_c = int(round(n0 * (gamma ** (c / denom))))
        n_c = max(1, min(n_c, len(per_class[c])))
        sel = rng.choice(per_class[c], size=n_c, replace=False)
        keep.extend(int(i) for i in sel)
    keep.sort()
    return keep


class SSLDataset(Dataset):
    """Two-view SSL dataset.

    ``__getitem__`` returns ``dict(v1=Tensor, v2=Tensor, idx=int, label=int)``.

    THE LABEL IS FOR DIAGNOSTICS ONLY.  It is returned under its own key,
    never concatenated with the tensors, so a training loop that only reads
    ``batch['v1']`` / ``batch['v2']`` cannot leak it into a loss.  Any use of
    ``batch['label']`` inside a loss is a project-level bug; it exists so that
    T1 (long-tail marginal) and T4 can plot class-conditional diagnostics.
    ``label == -1`` marks genuinely unlabelled samples (STL-10 unlabeled).
    """

    LABEL_IS_DIAGNOSTIC_ONLY = True

    def __init__(self, base: Dataset, two_view: Callable, indices: Optional[Sequence[int]],
                 labels: Optional[np.ndarray], name: str):
        self.base = base
        self.two_view = two_view
        self.indices = None if indices is None else [int(i) for i in indices]
        self._labels = labels
        self.name = name

    def __len__(self) -> int:
        return len(self.base) if self.indices is None else len(self.indices)

    def _map(self, i: int) -> int:
        return i if self.indices is None else self.indices[i]

    def raw(self, i: int):
        """Return the untransformed source image (PIL).  Used by
        ``invariance_score`` so it can re-augment the same image n times."""
        j = self._map(i)
        tgt = self.base
        old = getattr(tgt, "transform", None)
        try:
            if hasattr(tgt, "transform"):
                tgt.transform = None
            item = tgt[j]
        finally:
            if hasattr(tgt, "transform"):
                tgt.transform = old
        return item[0] if isinstance(item, (tuple, list)) else item

    def __getitem__(self, i: int) -> dict:
        j = self._map(i)
        img, label = self.base[j]
        v1, v2 = self.two_view(img)
        return {"v1": v1, "v2": v2, "idx": int(i), "label": int(label)}

    @property
    def labels(self) -> np.ndarray:
        """Diagnostic label vector aligned with __getitem__ order."""
        if self._labels is None:
            self._labels = _dataset_labels(self.base)
            if self.indices is not None:
                self._labels = self._labels[np.asarray(self.indices)]
        return self._labels


def get_ssl_dataset(name: str, cfg_a: AugCfg, cfg_b: AugCfg = None,
                    imbalance_gamma: float = None, seed: int = 0,
                    root: str = "./data") -> Dataset:
    """Two-view SSL dataset over the TRAIN split of ``name``.

    ``imbalance_gamma`` (if not None) subsamples the split to the exponential
    long-tail profile of ``longtail_indices`` -- this is the T1 stress test for
    the near-uniform-marginal assumption of prototype heads.

    Items are dicts ``{'v1','v2','idx','label'}``; see ``SSLDataset`` for the
    label policy (diagnostics only, never a loss input).
    """
    n = _canon(name)
    base = _build_raw(n, train=True, root=root, transform=None)
    two_view = make_two_view_transform(cfg_a, cfg_b)

    indices = None
    labels = None
    if imbalance_gamma is not None:
        labels_full = _dataset_labels(base)
        if (labels_full < 0).any():
            raise ValueError(
                f"{n} contains unlabelled samples; a long-tail profile cannot "
                f"be imposed on it."
            )
        indices = longtail_indices(labels_full, DATASET_NUM_CLASSES[n],
                                   float(imbalance_gamma), int(seed))
        labels = labels_full[np.asarray(indices)]
    return SSLDataset(base, two_view, indices, labels, n)


def get_eval_datasets(name: str, size: int, root: str = "./data") -> Tuple[Dataset, Dataset]:
    """Return ``(train_ds, test_ds)`` with the deterministic eval transform.

    The two splits are the dataset's own official splits wherever they exist
    (EuroSAT is split 80/20 with a hard-coded seed).  There is no overlap, and
    the transform carries no randomness, so probe numbers are reproducible.
    """
    n = _canon(name)
    tf = make_eval_transform(size)
    return (_build_raw(n, train=True, root=root, transform=tf),
            _build_raw(n, train=False, root=root, transform=tf))


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def make_encoder(arch: str = "resnet18", dataset: str = "cifar") -> Tuple[nn.Module, int]:
    """Build a backbone whose ``forward`` returns pooled features ``[B, feat_dim]``.

    ``dataset='cifar'``   -> 3x3 stride-1 stem, maxpool removed (32x32 inputs).
    ``dataset='stl'``     -> 3x3 stride-1 stem, maxpool KEPT (96x96 inputs).
    ``dataset='imagenet'``-> stock torchvision stem (7x7 stride 2 + maxpool).

    The classifier head is replaced by ``nn.Identity``.
    """
    _require_tv("make_encoder")
    arch = str(arch).lower()
    dataset = str(dataset).lower()
    if dataset in ("cifar10", "cifar100"):
        dataset = "cifar"
    if dataset not in ("cifar", "stl", "imagenet"):
        raise ValueError(f"dataset must be cifar|stl|imagenet, got {dataset!r}")

    factory = {
        "resnet18": torchvision.models.resnet18,
        "resnet34": torchvision.models.resnet34,
        "resnet50": torchvision.models.resnet50,
    }
    if arch not in factory:
        raise ValueError(f"arch must be one of {sorted(factory)}, got {arch!r}")

    net = factory[arch](weights=None)
    feat_dim = int(net.fc.in_features)
    net.fc = nn.Identity()

    if dataset in ("cifar", "stl"):
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(net.conv1.weight, mode="fan_out", nonlinearity="relu")
    if dataset == "cifar":
        net.maxpool = nn.Identity()
    return net, feat_dim


def make_mlp(in_dim: int, hidden: int, out_dim: int, n_layers: int = 2,
             bn: bool = True) -> nn.Module:
    """Projector / predictor MLP.

    ``n_layers`` counts the Linear layers.  n_layers=1 is a bare linear map;
    n_layers=k has k-1 hidden (Linear[-BN]-ReLU) blocks and a final Linear with
    NO normalisation and NO activation (the output must stay unbounded so that
    ``vicreg_reg`` can act on it -- see T2).
    """
    in_dim, hidden, out_dim = int(in_dim), int(hidden), int(out_dim)
    n_layers = int(n_layers)
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    if n_layers == 1:
        return nn.Sequential(nn.Linear(in_dim, out_dim))
    layers: List[nn.Module] = []
    d = in_dim
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(d, hidden, bias=not bn))
        if bn:
            layers.append(nn.BatchNorm1d(hidden))
        layers.append(nn.ReLU(inplace=True))
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class EMATeacher:
    """Exponential-moving-average target network (BYOL / DINO style).

    The target is a deep copy of the student with ``requires_grad_(False)``,
    permanently in ``eval()`` mode (so BatchNorm uses running statistics and the
    target branch does not depend on the current batch composition).

    ``tau`` follows the BYOL cosine schedule from ``tau_base`` to ``tau_final``
    over ``total_steps``; with ``total_steps <= 0`` the momentum is constant at
    ``tau_base``.
    """

    def __init__(self, student: nn.Module, tau_base: float = 0.996,
                 tau_final: float = 1.0, total_steps: int = 0):
        self.student = student
        self.tau_base = float(tau_base)
        self.tau_final = float(tau_final)
        self.total_steps = int(total_steps)
        self._target = copy.deepcopy(student)
        for p in self._target.parameters():
            p.requires_grad_(False)
        self._target.eval()
        self.last_tau = self.tau_base

    def tau_at(self, step: int) -> float:
        if self.total_steps <= 0:
            return self.tau_base
        t = min(max(int(step), 0), self.total_steps) / float(self.total_steps)
        return self.tau_final - (self.tau_final - self.tau_base) * (math.cos(math.pi * t) + 1.0) / 2.0

    @torch.no_grad()
    def update(self, step: int) -> None:
        tau = self.tau_at(step)
        self.last_tau = tau
        for p_t, p_s in zip(self._target.parameters(), self.student.parameters()):
            p_t.mul_(tau).add_(p_s.detach(), alpha=1.0 - tau)
        # Buffers (BN running stats, num_batches_tracked) are copied, not EMA'd:
        # EMA on integer counters is meaningless and BYOL copies them.
        for b_t, b_s in zip(self._target.buffers(), self.student.buffers()):
            b_t.copy_(b_s)

    @property
    def target(self) -> nn.Module:
        self._target.eval()
        return self._target

    def state_dict(self) -> dict:
        return {"target": self._target.state_dict(), "last_tau": self.last_tau}

    def load_state_dict(self, sd: dict) -> None:
        self._target.load_state_dict(sd["target"])
        self.last_tau = float(sd.get("last_tau", self.tau_base))
        for p in self._target.parameters():
            p.requires_grad_(False)
        self._target.eval()

    def to(self, device) -> "EMATeacher":
        self._target.to(device)
        return self


class FrozenProjection(nn.Module):
    """A random linear map that is structurally impossible to train.

    The weight, bias and (optional) input index mask are stored as BUFFERS, not
    ``nn.Parameter``.  Consequences:
      * ``.parameters()`` is empty  -> the head can never enter an optimiser,
        even if someone writes ``optim.SGD(model.parameters())`` over a module
        that contains it;
      * ``.state_dict()`` still carries it, so checkpoints round-trip;
      * ``.to(device)`` / ``.float()`` still work.

    This is the T3 "panel of frozen heads": the min-max objective has a trivial
    solution (all heads collapse to the same function, max == mean) as soon as
    the heads are trainable.
    """

    def __init__(self, in_dim: int, out_dim: int, idx: Optional[torch.Tensor],
                 weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        if idx is None:
            self.register_buffer("idx", None)
        else:
            self.register_buffer("idx", idx.long())
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"FrozenProjection expects [B,d], got {tuple(x.shape)}")
        if self.idx is not None:
            x = x.index_select(1, self.idx)
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        sub = self.in_dim if self.idx is None else int(self.idx.numel())
        return f"in={self.in_dim}, sub={sub}, out={self.out_dim}, frozen=True"


def frozen_random_projections(in_dim: int, out_dim: int, k: int, seed: int,
                              disjoint: bool = True) -> List[nn.Module]:
    """Panel of ``k`` frozen random projections (thesis T3).

    ``disjoint=True``: the input space is split by a single fixed permutation
    into ``k`` blocks of ``in_dim // k`` coordinates and head *i* reads block
    *i* only.  The heads then look at genuinely distinct subspaces, so the
    panel cannot become redundant through the input side either.

    ``disjoint=False``: every head reads all ``in_dim`` coordinates.

    Weights are N(0, 1/sqrt(fan_in)) so outputs start at roughly unit scale.
    All tensors are buffers -> ``sum(p.numel() for h in heads for p in
    h.parameters()) == 0``.
    """
    in_dim, out_dim, k = int(in_dim), int(out_dim), int(k)
    if k < 1:
        raise ValueError("k must be >= 1")
    g = torch.Generator().manual_seed(int(seed))

    heads: List[nn.Module] = []
    if disjoint:
        block = in_dim // k
        if block < 1:
            raise ValueError(
                f"disjoint panel needs in_dim >= k (got in_dim={in_dim}, k={k}); "
                f"use disjoint=False or a wider representation."
            )
        perm = torch.randperm(in_dim, generator=g)
    for i in range(k):
        if disjoint:
            idx = perm[i * block:(i + 1) * block].clone()
            fan_in = int(idx.numel())
        else:
            idx = None
            fan_in = in_dim
        w = torch.randn(out_dim, fan_in, generator=g) / math.sqrt(fan_in)
        b = torch.zeros(out_dim)
        head = FrozenProjection(in_dim, out_dim, idx, w, b)
        head.eval()
        heads.append(head)
    return heads


# ---------------------------------------------------------------------------
# objectives
# ---------------------------------------------------------------------------

def _l2(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def gram_loss(s: torch.Tensor, a: torch.Tensor, temperature: float = None,
              pos_weight: float = None) -> torch.Tensor:
    """Relational (Gram-matching) objective.

    ``s`` = student embeddings [B,d], ``a`` = target embeddings [B,d].  Both are
    L2-normalised INTERNALLY; ``a`` is detached, so the target Gram is a genuine
    stop-gradient ``sg[A A^T]``.

    Default (``pos_weight is None``) -- literally thesis T2::

        L = mean_{i,j} ( (S S^T)_ij - sg[(A A^T)]_ij )^2 ,   S = normalise(s)

    Note the two diagonals are identically 1, so the B diagonal entries
    contribute exactly zero error: the objective is carried entirely by the
    B*(B-1) off-diagonal (relational) entries.  That is precisely the project's
    own reserve -- there is no explicit positive-pair term at all -- and it is
    also why the objective is degenerate on its own: all-identical embeddings
    give S S^T = A A^T = ones and L = 0, a GLOBAL minimum.  Anti-collapse must
    come from ``vicreg_reg`` on the UNNORMALISED z.

    ``pos_weight`` (>= 0, not None) restores an explicit positive term and
    reweights it against the relational bulk::

        L = [ mean_{i != j} (S S^T - sg[A A^T])^2
              + pos_weight * mean_i ( <s_i, a_i> - 1 )^2 ] / (1 + pos_weight)

    The positive term is the diagonal of the CROSS Gram ``s_n a_n^T``, i.e. the
    agreement between the two views of the same sample, which the pure
    within-view form cannot see.  ``pos_weight=0`` gives the purely relational
    off-diagonal loss.

    ``temperature`` (if not None) divides BOTH similarity matrices before the
    MSE.  Be explicit about what this does: for a squared error it is an exact
    rescaling of the loss by 1/temperature^2, hence an effective learning-rate
    change, NOT a sharpening.  It exists so that a Gram arm can be swept over
    the same nominal temperature grid as a prototype arm; do not read it as the
    analogue of the DINO temperature.

    O(d) invariance (T1): rotating both s and a by the same orthogonal Q leaves
    every quantity above unchanged.  This is asserted in the self-test.
    """
    if s.dim() != 2 or a.dim() != 2:
        raise ValueError(f"gram_loss expects [B,d] tensors, got {tuple(s.shape)} / {tuple(a.shape)}")
    if s.shape != a.shape:
        raise ValueError(f"shape mismatch: s={tuple(s.shape)} a={tuple(a.shape)}")
    B = s.shape[0]
    if B < 2:
        raise ValueError("gram_loss needs B >= 2 (a Gram over one sample is trivial)")

    S = _l2(s)
    A = _l2(a.detach())

    G_s = S @ S.t()                      # [B,B], requires grad
    G_a = (A @ A.t()).detach()           # sg[A A^T]

    if temperature is not None:
        t = float(temperature)
        if t <= 0:
            raise ValueError("temperature must be > 0")
        G_s = G_s / t
        G_a = G_a / t

    if pos_weight is None:
        return (G_s - G_a).pow(2).mean()

    w = float(pos_weight)
    if w < 0:
        raise ValueError("pos_weight must be >= 0 or None")

    eye = torch.eye(B, device=s.device, dtype=torch.bool)
    off = (G_s - G_a).pow(2).masked_select(~eye).mean()

    pos_sim = (S * A).sum(dim=1)         # <s_i, a_i>, the cross-Gram diagonal
    target = torch.ones_like(pos_sim)
    if temperature is not None:
        pos_sim = pos_sim / float(temperature)
        target = target / float(temperature)
    pos = (pos_sim - target).pow(2).mean()

    return (off + w * pos) / (1.0 + w)


def prototype_loss(s: torch.Tensor, a: torch.Tensor, protos: nn.Parameter,
                   tau_s: float, tau_t: float, center: torch.Tensor = None,
                   use_center: bool = True, use_sharpen: bool = True
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """DINO-style prototype cross-entropy.  Returns ``(loss, new_center)``.

    ``protos`` is ``[K, d]`` (an ``nn.Parameter``; the caller owns it and puts
    it in the optimiser).  Features and prototypes are L2-normalised, which is
    the weight-normalised DINO head with unit norm.

        student logits : z_s = <s_n, c_k> / tau_s
        teacher logits : z_t = <a_n, c_k>            (a is DETACHED here)
        teacher target : p_t = softmax( (z_t - center) / tau_t )
        loss           : - sum_k p_t log softmax(z_s)

    Ablation switches (T1):
      ``use_center=False``  drops the EMA centering term;
      ``use_sharpen=False`` sets the teacher temperature to ``tau_s`` (no
      sharpening).  Note DINO needs BOTH centering and sharpening: centering
      alone collapses to uniform, sharpening alone collapses to one-hot.

    ``center`` is ``[K]``; pass ``None`` on the first step to initialise it to
    zeros.  The returned ``new_center`` is detached and must be fed back on the
    next call (EMA momentum 0.9, the DINO default).

    NOT O(d)-invariant: the ``c_k`` fix a distinguished basis, so rotating s and
    a while leaving protos alone changes the loss.  Asserted in the self-test.
    """
    if s.dim() != 2 or a.dim() != 2:
        raise ValueError("prototype_loss expects [B,d] tensors")
    if s.shape != a.shape:
        raise ValueError(f"shape mismatch: s={tuple(s.shape)} a={tuple(a.shape)}")
    P = protos if isinstance(protos, torch.Tensor) else protos.weight
    if P.dim() != 2 or P.shape[1] != s.shape[1]:
        raise ValueError(f"protos must be [K,d] with d={s.shape[1]}, got {tuple(P.shape)}")
    tau_s = float(tau_s)
    tau_t_eff = float(tau_t) if use_sharpen else tau_s
    if tau_s <= 0 or tau_t_eff <= 0:
        raise ValueError("temperatures must be > 0")

    Pn = _l2(P)
    zs = _l2(s) @ Pn.t() / tau_s                       # [B,K] with grad
    zt = (_l2(a.detach()) @ Pn.t()).detach()           # [B,K] no grad

    K = P.shape[0]
    if center is None:
        center = torch.zeros(K, device=s.device, dtype=zt.dtype)
    center = center.to(device=zt.device, dtype=zt.dtype).detach()

    zt_c = zt - center if use_center else zt
    p_t = F.softmax(zt_c / tau_t_eff, dim=1).detach()

    loss = -(p_t * F.log_softmax(zs, dim=1)).sum(dim=1).mean()

    m = 0.9
    new_center = (m * center + (1.0 - m) * zt.mean(dim=0)).detach()
    return loss, new_center


def vicreg_reg(z: torch.Tensor, var_w: float = 25.0, cov_w: float = 1.0,
               gamma: float = 1.0) -> Tuple[torch.Tensor, dict]:
    """VICReg variance + covariance regulariser -- the "legality" term of T2.

    MUST be called on UNNORMALISED ``z``.  This is enforced: if every row of
    ``z`` has unit norm to within 1e-3, a ValueError is raised.  Reason: on the
    unit sphere each coordinate has std bounded by 1/sqrt(d), so for any usual
    d the hinge ``relu(gamma - sigma)`` with gamma=1 is saturated at essentially
    ``gamma`` for every coordinate at every step.  The variance term then
    contributes a near-constant with a vanishing, uninformative gradient, and
    the anti-collapse guarantee is silently void -- exactly the failure mode T2
    warns about.

    Returns ``(loss, {'var': float, 'cov': float})`` where the dict holds the
    RAW (unweighted) terms, so logs stay comparable when the weights change.
    """
    if z.dim() != 2:
        raise ValueError(f"vicreg_reg expects [B,d], got {tuple(z.shape)}")
    B, d = z.shape
    if B < 2:
        raise ValueError("vicreg_reg needs B >= 2 to estimate a variance")

    norms = z.detach().norm(dim=1)
    if torch.all((norms - 1.0).abs() < 1e-3):
        raise ValueError(
            "vicreg_reg received an L2-NORMALISED z (all row norms == 1 +/- 1e-3). "
            "This is forbidden: on the unit sphere each coordinate's std is "
            f"bounded by 1/sqrt(d) = {1.0 / math.sqrt(d):.4f}, so the hinge "
            f"relu(gamma - sigma) with gamma={gamma} is permanently saturated "
            "and the variance term stops preventing collapse (thesis T2). "
            "Pass the UNNORMALISED projector output; normalise only inside the "
            "agreement term (gram_loss / info_nce do it themselves)."
        )

    zc = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(zc.var(dim=0, unbiased=True) + 1e-4)
    var_term = F.relu(float(gamma) - std).mean()

    cov = (zc.t() @ zc) / (B - 1)
    off = cov - torch.diag_embed(torch.diagonal(cov))
    cov_term = off.pow(2).sum() / d

    loss = float(var_w) * var_term + float(cov_w) * cov_term
    return loss, {"var": float(var_term.detach()), "cov": float(cov_term.detach())}


def info_nce(s: torch.Tensor, a: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    """Symmetric InfoNCE (SimCLR/NT-Xent restricted to cross-view pairs).

    ``s[i]`` and ``a[i]`` are the two views of sample i.  Logits are the cross
    similarity matrix ``s_n a_n^T / tau``; the positive is the diagonal.  Both
    branches receive gradient (no stop-gradient here, unlike ``gram_loss``);
    the loss is averaged over the two symmetric directions.
    """
    if s.shape != a.shape or s.dim() != 2:
        raise ValueError("info_nce expects two [B,d] tensors of equal shape")
    B = s.shape[0]
    if B < 2:
        raise ValueError("info_nce needs B >= 2")
    tau = float(tau)
    if tau <= 0:
        raise ValueError("tau must be > 0")
    logits = _l2(s) @ _l2(a).t() / tau
    labels = torch.arange(B, device=s.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def _unpack_batch(batch):
    """Accept (x,y) tuples and {'v1'/'x'/'image', 'label'} dicts."""
    if isinstance(batch, dict):
        x = batch.get("x", batch.get("image", batch.get("v1")))
        y = batch.get("label", batch.get("y"))
        if x is None:
            raise KeyError(f"cannot find images in batch keys {list(batch)}")
        return x, y
    if isinstance(batch, (tuple, list)):
        return batch[0], (batch[1] if len(batch) > 1 else None)
    return batch, None


@torch.no_grad()
def _extract_features(encoder: nn.Module, ds: Dataset, device: torch.device,
                      bs: int = 256, num_workers: int = 0
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward the whole dataset ONCE under no_grad with the encoder in eval().

    The encoder's train/eval mode is restored afterwards.  Returns
    ``(features [N,D] float32 on CPU, labels [N] int64 on CPU)``.
    """
    was_training = encoder.training
    encoder.eval()
    encoder.to(device)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=num_workers,
                        pin_memory=(device.type == "cuda"), drop_last=False)
    feats: List[torch.Tensor] = []
    labs: List[torch.Tensor] = []
    for batch in loader:
        x, y = _unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        f = encoder(x)
        if f.dim() > 2:
            f = torch.flatten(F.adaptive_avg_pool2d(f, 1), 1)
        feats.append(f.detach().float().cpu())
        if y is None:
            raise ValueError("evaluation datasets must yield labels")
        labs.append(torch.as_tensor(y).long().reshape(-1).cpu())
    if was_training:
        encoder.train()
    return torch.cat(feats, 0), torch.cat(labs, 0)


def _head_tail_split(train_labels: torch.Tensor, num_classes: int) -> Tuple[List[int], List[int]]:
    """Split classes into head / tail by TRAIN frequency.

    Classes are ranked by their training count (descending, ties broken by
    class id) and cut in half: the more frequent half is the head, the rest is
    the tail.  For a balanced dataset the split is arbitrary but harmless
    (head_acc ~ tail_acc); for the T1 long-tail runs it is the quantity of
    interest.
    """
    counts = np.bincount(train_labels.numpy(), minlength=num_classes)
    order = sorted(range(num_classes), key=lambda c: (-counts[c], c))
    cut = max(1, num_classes // 2)
    return order[:cut], order[cut:]


def linear_probe(encoder, train_ds, test_ds, epochs: int = 100, lr: float = 1e-2,
                 bs: int = 512, device=None, num_classes: int = None,
                 probe_seed: int = 0) -> dict:
    """Linear evaluation on FROZEN features.

    Protocol
    --------
    * The encoder is put in ``eval()`` and used strictly under ``torch.no_grad``;
      features for train and test are computed ONCE (no augmentation -- the eval
      datasets carry the deterministic transform).
    * Features are standardised per dimension using TRAIN statistics only
      (``(f - mu_train) / (sd_train + 1e-6)``).  The test set never contributes
      to mu/sd: no leakage.
    * A single ``nn.Linear`` is trained with AdamW + cosine schedule for
      ``epochs`` passes over the cached train features.
    * Splits are the dataset's own train/test; nothing is shared between them.
    * Head training runs inside an RNG island seeded by ``probe_seed``, so
      calling the probe mid-training neither reseeds nor consumes the training
      RNG stream (evaluating more often must not change the run).

    Returns ``{'acc', 'per_class_acc', 'head_acc', 'tail_acc'}`` where
    head/tail follow the TRAIN class-frequency split (see ``_head_tail_split``).
    """
    with _rng_island(seed=probe_seed):
        return _linear_probe_impl(encoder, train_ds, test_ds, epochs, lr, bs,
                                  device, num_classes)


def _linear_probe_impl(encoder, train_ds, test_ds, epochs, lr, bs, device, num_classes):
    device = device or get_device()
    Xtr, ytr = _extract_features(encoder, train_ds, device)
    Xte, yte = _extract_features(encoder, test_ds, device)

    if num_classes is None:
        num_classes = int(max(ytr.max().item(), yte.max().item())) + 1
    num_classes = int(num_classes)

    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True)
    Xtr = (Xtr - mu) / (sd + 1e-6)
    Xte = (Xte - mu) / (sd + 1e-6)          # TRAIN statistics only

    Xtr, ytr = Xtr.to(device), ytr.to(device)
    Xte, yte = Xte.to(device), yte.to(device)

    head = nn.Linear(Xtr.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=float(lr), weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(epochs)))

    N = Xtr.shape[0]
    for _ in range(int(epochs)):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, int(bs)):
            j = perm[i:i + int(bs)]
            loss = F.cross_entropy(head(Xtr[j]), ytr[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()

    with torch.no_grad():
        pred = head(Xte).argmax(1)
        correct = (pred == yte)
        acc = float(correct.float().mean())
        per_class: List[float] = []
        for c in range(num_classes):
            m = (yte == c)
            per_class.append(float(correct[m].float().mean()) if int(m.sum()) > 0 else float("nan"))

    head_cls, tail_cls = _head_tail_split(ytr.cpu(), num_classes)
    def _avg(cs):
        v = [per_class[c] for c in cs if not math.isnan(per_class[c])]
        return float(np.mean(v)) if v else float("nan")

    return {"acc": acc, "per_class_acc": per_class,
            "head_acc": _avg(head_cls), "tail_acc": _avg(tail_cls)}


def knn_probe(encoder, train_ds, test_ds, k: int = 20, device=None) -> float:
    """Weighted k-NN on L2-normalised frozen features (InstDisc/DINO protocol).

    Cosine similarity, temperature 0.07 on the vote weights, ties broken by
    argmax.  No training, so it is the cheap sanity metric during SSL runs.
    Runs in an RNG island: torch's DataLoader draws a base seed from the global
    generator, so an unguarded probe would shift the training stream.
    """
    with _rng_island(seed=0):
        return _knn_probe_impl(encoder, train_ds, test_ds, k, device)


@torch.no_grad()
def _knn_probe_impl(encoder, train_ds, test_ds, k, device) -> float:
    device = device or get_device()
    Xtr, ytr = _extract_features(encoder, train_ds, device)
    Xte, yte = _extract_features(encoder, test_ds, device)
    num_classes = int(max(ytr.max().item(), yte.max().item())) + 1

    Xtr = _l2(Xtr).to(device)
    Xte = _l2(Xte).to(device)
    ytr = ytr.to(device)
    yte = yte.to(device)

    k = int(min(k, Xtr.shape[0]))
    T = 0.07
    correct = 0
    chunk = 1024
    for i in range(0, Xte.shape[0], chunk):
        sim = Xte[i:i + chunk] @ Xtr.t()                 # [c, Ntr]
        sim_k, idx_k = sim.topk(k, dim=1)
        w = (sim_k / T).exp()
        lab_k = ytr[idx_k]                               # [c, k]
        scores = torch.zeros(sim.shape[0], num_classes, device=device)
        scores.scatter_add_(1, lab_k, w)
        correct += int((scores.argmax(1) == yte[i:i + chunk]).sum())
    return float(correct) / float(Xte.shape[0])


def probe_battery(encoder, task_names: list, device=None, root: str = "./data",
                  size: int = 32, epochs: int = 40, lr: float = 1e-2,
                  bs: int = 512) -> dict:
    """Linear probe on a heterogeneous battery of tasks (thesis T7).

    Returns ``{'per_task': {name: acc}, 'worst': float, 'mean': float}``.
    ALWAYS report ``worst``: the whole point of T7 is that a frozen encoder is
    judged on its weakest task, never on the average, which hides the failure.
    A task that raises (missing download, etc.) is recorded as NaN and excluded
    from ``mean`` but makes ``worst`` NaN too -- a silently dropped task would
    defeat the purpose.
    """
    device = device or get_device()
    per_task: Dict[str, float] = {}
    for name in task_names:
        try:
            tr, te = get_eval_datasets(name, size=size, root=root)
            res = linear_probe(encoder, tr, te, epochs=epochs, lr=lr, bs=bs,
                               device=device,
                               num_classes=DATASET_NUM_CLASSES[_canon(name)])
            per_task[name] = float(res["acc"])
        except Exception as e:  # pragma: no cover - environment dependent
            print(f"[probe_battery] task {name!r} FAILED: {type(e).__name__}: {e}")
            per_task[name] = float("nan")
    vals = list(per_task.values())
    finite = [v for v in vals if not math.isnan(v)]
    worst = float("nan") if len(finite) < len(vals) or not finite else float(min(finite))
    mean = float(np.mean(finite)) if finite else float("nan")
    return {"per_task": per_task, "worst": worst, "mean": mean}


def effective_rank(Z: torch.Tensor) -> float:
    """Effective rank (Roy & Vetterli 2007): exp of the Shannon entropy of the
    singular-value distribution ``p_i = sigma_i / sum_j sigma_j``.

    ``Z`` is ``[N, d]`` and is used AS GIVEN (not centred): for embeddings the
    mean direction is itself part of the collapse story, and centring would
    hide a rank-1 "everything equals the mean" solution.  Range [1, min(N,d)];
    1.0 means total collapse.
    """
    if Z.dim() != 2:
        raise ValueError(f"effective_rank expects [N,d], got {tuple(Z.shape)}")
    Zf = Z.detach().float()
    s = torch.linalg.svdvals(Zf)
    s = s.clamp_min(0)
    tot = s.sum()
    if float(tot) <= 0:
        return 1.0
    p = s / tot
    p = p[p > 1e-12]
    ent = -(p * p.log()).sum()
    return float(torch.exp(ent))


@torch.no_grad()
def invariance_score(encoder, dataset, cfg: AugCfg, n_views: int = 8, device=None,
                     n_samples: int = 256, seed: int = 0) -> float:
    """Worst-case augmentation invariance of ``encoder`` under policy ``cfg``.

    For each of ``n_samples`` images, ``n_views`` independent views are drawn
    from ``cfg``, embedded and L2-normalised; the score of the image is the
    MINIMUM pairwise cosine similarity over the ``n_views*(n_views-1)/2`` pairs
    (worst case, not mean -- same discipline as T3 and T7).  The returned value
    is the average of those minima.  1.0 = perfectly invariant.

    ``dataset`` may be an :class:`SSLDataset` (its ``raw(i)`` is used) or any
    torchvision dataset with a ``.transform`` attribute (temporarily swapped
    out to recover the source image, then restored).
    """
    _require_tv("invariance_score")
    with _rng_island(seed=seed):
        return _invariance_score_impl(encoder, dataset, cfg, n_views, device,
                                      n_samples, seed)


@torch.no_grad()
def _invariance_score_impl(encoder, dataset, cfg, n_views, device, n_samples, seed):
    device = device or get_device()
    was_training = encoder.training
    encoder.eval()
    encoder.to(device)
    tf = make_single_view_transform(cfg)

    n = min(int(n_samples), len(dataset))
    rng = np.random.RandomState(int(seed))
    pick = rng.choice(len(dataset), size=n, replace=False)

    def _raw(i: int):
        if hasattr(dataset, "raw"):
            return dataset.raw(int(i))
        old = getattr(dataset, "transform", "__missing__")
        if old == "__missing__":
            raise TypeError(
                "invariance_score needs a dataset exposing raw images "
                "(SSLDataset.raw or a torchvision `.transform` attribute)"
            )
        try:
            dataset.transform = None
            item = dataset[int(i)]
        finally:
            dataset.transform = old
        return item[0] if isinstance(item, (tuple, list)) else item

    torch.manual_seed(int(seed))
    mins: List[float] = []
    batch: List[torch.Tensor] = []
    owners: List[int] = []
    FLUSH = 64  # images per forward chunk

    def _flush():
        if not batch:
            return
        x = torch.stack(batch).to(device)                 # [m*n_views, C,H,W]
        f = encoder(x)
        if f.dim() > 2:
            f = torch.flatten(F.adaptive_avg_pool2d(f, 1), 1)
        f = _l2(f.float()).view(len(owners), int(n_views), -1)
        sim = torch.bmm(f, f.transpose(1, 2))             # [m, V, V]
        V = int(n_views)
        eye = torch.eye(V, device=sim.device, dtype=torch.bool).unsqueeze(0)
        sim = sim.masked_fill(eye, float("inf"))
        mins.extend(sim.view(len(owners), -1).min(dim=1).values.cpu().tolist())
        batch.clear()
        owners.clear()

    for i in pick:
        img = _raw(i)
        for _ in range(int(n_views)):
            batch.append(tf(img))
        owners.append(int(i))
        if len(owners) >= FLUSH:
            _flush()
    _flush()

    if was_training:
        encoder.train()
    return float(np.mean(mins)) if mins else float("nan")


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between two representations of the SAME ``N`` samples.

    ``X`` is [N, p], ``Y`` is [N, q]; both are column-centred, then

        CKA = ||Y^T X||_F^2 / ( ||X^T X||_F * ||Y^T Y||_F )

    Range [0,1]; 1.0 means the two representations are related by an invertible
    linear map (up to rotation/scale).  This is the T3 panel-diversity metric:
    if trainable heads collapse onto one another, pairwise CKA -> 1.
    """
    if X.dim() != 2 or Y.dim() != 2:
        raise ValueError("linear_cka expects [N,p] and [N,q]")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"row mismatch: {X.shape[0]} vs {Y.shape[0]}")
    Xf = X.detach().float()
    Yf = Y.detach().float()
    Xf = Xf - Xf.mean(0, keepdim=True)
    Yf = Yf - Yf.mean(0, keepdim=True)
    num = (Yf.t() @ Xf).norm(p="fro") ** 2
    den = (Xf.t() @ Xf).norm(p="fro") * (Yf.t() @ Yf).norm(p="fro")
    if float(den) <= 1e-12:
        return float("nan")
    return float(num / den)


def uniformity_alignment(z1: torch.Tensor, z2: torch.Tensor) -> Tuple[float, float]:
    """Wang & Isola (2020) diagnostics.  Returns ``(uniformity, alignment)``
    -- in the order of the function name, mind the order.

        alignment  = E_i || z1_i - z2_i ||^2                       (lower = more aligned)
        uniformity = log E_{i != j} exp( -2 || z1_i - z1_j ||^2 )  (lower = more uniform)

    Both are computed on L2-NORMALISED inputs (normalisation is done here), and
    uniformity uses the ``z1`` cloud only, following the original definition.
    """
    if z1.shape != z2.shape or z1.dim() != 2:
        raise ValueError("uniformity_alignment expects two [B,d] tensors of equal shape")
    B = z1.shape[0]
    if B < 2:
        raise ValueError("need B >= 2")
    a = _l2(z1.detach().float())
    b = _l2(z2.detach().float())
    alignment = float((a - b).pow(2).sum(dim=1).mean())
    sq = torch.cdist(a, a).pow(2)
    iu = torch.triu_indices(B, B, offset=1, device=a.device)
    uniformity = float(torch.log((-2.0 * sq[iu[0], iu[1]]).exp().mean()))
    return uniformity, alignment


# ---------------------------------------------------------------------------
# runs / budget / resume
# ---------------------------------------------------------------------------

class Run:
    """Crash-safe experiment bookkeeping.

    Files written under ``outdir``:
      ``<name>.csv``           one row per ``log()`` call, flushed + fsync'd
      ``<name>.config.json``   the config dict, written once
      ``<name>.ckpt.pt``       last checkpoint (atomic replace)
      ``<name>.best.pt``       best checkpoint (see ``save_ckpt``)
      ``<name>.summary.json``  written by ``finish()``

    ``resume=True`` (default) keeps any existing checkpoint and truncates the
    CSV back to the checkpointed step, so a killed Colab session restarts
    without duplicated or orphaned rows.  ``resume=False`` wipes csv/ckpts and
    starts clean.
    """

    def __init__(self, name: str, outdir: str, config: dict, resume: bool = True,
                 higher_is_better: bool = True):
        self.name = str(name)
        self.outdir = str(outdir)
        self.config = dict(config or {})
        self.resume = bool(resume)
        self.higher_is_better = bool(higher_is_better)
        os.makedirs(self.outdir, exist_ok=True)

        self.csv_path = os.path.join(self.outdir, f"{self.name}.csv")
        self.ckpt_path = os.path.join(self.outdir, f"{self.name}.ckpt.pt")
        self.best_path = os.path.join(self.outdir, f"{self.name}.best.pt")
        self.config_path = os.path.join(self.outdir, f"{self.name}.config.json")
        self.summary_path = os.path.join(self.outdir, f"{self.name}.summary.json")

        self._fields: List[str] = []
        self._t0 = time.time()
        self.best_metric: Optional[float] = None
        self.resumed_step: int = 0

        if not self.resume:
            for p in (self.csv_path, self.ckpt_path, self.best_path, self.summary_path):
                if os.path.exists(p):
                    os.remove(p)
        else:
            ck = self.load_ckpt()
            if ck is not None:
                self.resumed_step = int(ck.get("step", 0))
                bm = ck.get("best_metric", None)
                self.best_metric = None if bm is None else float(bm)
                self._truncate_csv_after(self.resumed_step)

        if os.path.exists(self.csv_path):
            with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                r = csv.reader(f)
                try:
                    self._fields = next(r)
                except StopIteration:
                    self._fields = []

        self._write_json(self.config_path, {"name": self.name, "config": self.config})

    # -- internals ----------------------------------------------------------
    @staticmethod
    def _atomic_write_bytes(path: str, write_fn: Callable[[str], None]) -> None:
        tmp = path + ".tmp"
        write_fn(tmp)
        os.replace(tmp, path)

    def _write_json(self, path: str, obj: dict) -> None:
        def _w(tmp):
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, default=_json_default)
                f.flush()
                os.fsync(f.fileno())
        self._atomic_write_bytes(path, _w)

    def _truncate_csv_after(self, step: int) -> None:
        """Drop logged rows with step > `step` (they are ahead of the ckpt)."""
        if not os.path.exists(self.csv_path):
            return
        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if not rows:
            return
        header, body = rows[0], rows[1:]
        if "step" not in header:
            return
        si = header.index("step")

        def _keep(r):
            try:
                return float(r[si]) <= float(step)
            except Exception:
                return True
        body = [r for r in body if _keep(r)]

        def _w(tmp):
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(body)
                f.flush()
                os.fsync(f.fileno())
        self._atomic_write_bytes(self.csv_path, _w)

    def _rewrite_with_fields(self, new_fields: List[str]) -> None:
        """Widen the CSV header when a later log() introduces new metrics."""
        rows: List[dict] = []
        if os.path.exists(self.csv_path):
            with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        def _w(tmp):
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=new_fields, extrasaction="ignore")
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in new_fields})
                f.flush()
                os.fsync(f.fileno())
        self._atomic_write_bytes(self.csv_path, _w)
        self._fields = list(new_fields)

    # -- public -------------------------------------------------------------
    def log(self, step: int, **metrics) -> None:
        """Append one row to ``<name>.csv`` and flush it to disk immediately."""
        row = {"step": int(step), "wall_s": round(time.time() - self._t0, 3)}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().float().item() if v.numel() == 1 else v.detach().float().mean().item()
            elif isinstance(v, np.generic):
                v = v.item()
            row[k] = v

        if not self._fields:
            self._fields = list(row.keys())
            self._rewrite_with_fields(self._fields)
        else:
            missing = [k for k in row if k not in self._fields]
            if missing:
                self._rewrite_with_fields(self._fields + missing)

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._fields, extrasaction="ignore")
            w.writerow(row)
            f.flush()
            os.fsync(f.fileno())

    def save_ckpt(self, step: int, **state) -> None:
        """Atomically write ``<name>.ckpt.pt`` (write .tmp, then ``os.replace``).

        If ``state`` contains a float key ``best_metric``, the checkpoint is
        ALSO copied to ``<name>.best.pt`` whenever that metric improves
        (higher is better unless ``Run(..., higher_is_better=False)``).
        """
        payload = dict(state)
        payload["step"] = int(step)
        payload["_config"] = self.config
        payload["_saved_at"] = time.time()

        bm = state.get("best_metric", None)
        improved = False
        if bm is not None:
            bm = float(bm)
            if self.best_metric is None:
                improved = True
            elif self.higher_is_better and bm > self.best_metric:
                improved = True
            elif (not self.higher_is_better) and bm < self.best_metric:
                improved = True
            if improved:
                self.best_metric = bm
        payload["best_metric"] = self.best_metric

        def _w(tmp):
            with open(tmp, "wb") as f:
                torch.save(payload, f)
                f.flush()
                os.fsync(f.fileno())
        self._atomic_write_bytes(self.ckpt_path, _w)
        if improved:
            self._atomic_write_bytes(self.best_path, _w)

    def load_ckpt(self) -> Optional[dict]:
        """Return the last checkpoint dict, or None if there is none/it is corrupt."""
        if not os.path.exists(self.ckpt_path):
            return None
        try:
            return torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        except Exception as e:  # pragma: no cover
            print(f"[Run] checkpoint {self.ckpt_path} unreadable ({e}); ignoring.")
            return None

    def finish(self, summary: dict) -> None:
        out = {"name": self.name, "config": self.config,
               "elapsed_s": round(time.time() - self._t0, 3),
               "summary": dict(summary or {})}
        self._write_json(self.summary_path, out)


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


def aggregate_seeds(csv_glob: str, key: str):
    """Mean/std across seeds of every numeric column, indexed by ``key``.

    ``csv_glob`` matches the per-seed CSVs written by :class:`Run` (e.g.
    ``out/t1_gram_seed*.csv``); ``key`` is the alignment column, normally
    ``'step'`` or ``'epoch'``.  Returns a DataFrame with columns
    ``<m>_mean``, ``<m>_std`` (population-free ddof=1) and ``n_seeds``.

    A single-seed glob is accepted but the std column will be NaN -- and per
    project rules a single-seed comparison decides nothing.
    """
    import pandas as pd

    paths = sorted(_glob.glob(csv_glob))
    if not paths:
        raise FileNotFoundError(f"no CSV matched {csv_glob!r}")
    frames = []
    for i, p in enumerate(paths):
        df = pd.read_csv(p)
        if key not in df.columns:
            raise KeyError(f"{p} has no column {key!r} (columns: {list(df.columns)})")
        df["_source"] = os.path.basename(p)
        if "seed" not in df.columns:
            df["seed"] = i
        frames.append(df)
    allf = pd.concat(frames, ignore_index=True)

    num = allf.select_dtypes(include=[np.number])
    if key not in num.columns:
        num = num.join(allf[[key]])
    drop = {c for c in ("seed",) if c in num.columns}
    value_cols = [c for c in num.columns if c != key and c not in drop]

    g = num.groupby(key)
    out = g[value_cols].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_seeds"] = allf.groupby(key)["_source"].nunique()
    return out.reset_index()


def matched_budget_check(arms: list) -> None:
    """Guard-rail: refuse to compare arms that did not pay the same price.

    Each arm is a dict with at least::

        {'name': str, 'steps': int, 'batch_size': int, 'forward_passes': int}

    ``forward_passes`` is the number of encoder forward passes per optimisation
    step (2 for a plain two-view method, 2*K if K heads each re-encode, etc.).
    Optional extra keys are printed but not enforced.

    Prints a table, then raises ValueError if ``steps``, ``batch_size`` or
    ``forward_passes`` differ across arms.  This is not decoration: T1 compares
    a Gram target -- whose signal scales with B -- against a prototype head, and
    T4 compares a router against a control at strictly equal FLOPs.  An
    unmatched comparison there is worth nothing.
    """
    required = ("steps", "batch_size", "forward_passes")
    if not arms or len(arms) < 2:
        raise ValueError("matched_budget_check needs at least 2 arms")

    norm = []
    for i, a in enumerate(arms):
        if not isinstance(a, dict):
            raise TypeError(f"arm {i} is {type(a).__name__}, expected dict")
        miss = [k for k in required if k not in a]
        if miss:
            raise ValueError(
                f"arm {a.get('name', i)!r} is missing {miss}; every arm must "
                f"declare {list(required)}"
            )
        d = dict(a)
        d.setdefault("name", f"arm{i}")
        d["samples_seen"] = int(d["steps"]) * int(d["batch_size"])
        d["encoder_fwd"] = d["samples_seen"] * int(d["forward_passes"])
        norm.append(d)

    cols = ["name", "steps", "batch_size", "forward_passes", "samples_seen", "encoder_fwd"]
    widths = {c: max(len(c), max(len(str(d[c])) for d in norm)) + 2 for c in cols}
    line = "".join(c.ljust(widths[c]) for c in cols)
    print("-" * len(line))
    print("MATCHED BUDGET CHECK")
    print("-" * len(line))
    print(line)
    for d in norm:
        print("".join(str(d[c]).ljust(widths[c]) for c in cols))
    print("-" * len(line))

    bad = {}
    for k in required:
        vals = {d["name"]: d[k] for d in norm}
        if len(set(vals.values())) > 1:
            bad[k] = vals
    if bad:
        msg = ["UNMATCHED BUDGET -- these arms are not comparable:"]
        for k, vals in bad.items():
            msg.append(f"  {k}: " + ", ".join(f"{n}={v}" for n, v in vals.items()))
        msg.append("Equalise steps, batch_size and forward_passes before comparing.")
        print("\n".join(msg))
        raise ValueError("\n".join(msg))
    print(f"OK: {len(norm)} arms matched "
          f"({norm[0]['steps']} steps x {norm[0]['batch_size']} batch x "
          f"{norm[0]['forward_passes']} fwd).")


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

class _TinyImageDataset(Dataset):
    """In-memory PIL dataset with a swappable ``.transform`` (self-test only)."""

    def __init__(self, n=24, size=32, num_classes=4, seed=0, transform=None):
        rs = np.random.RandomState(seed)
        self.arrs = [rs.randint(0, 255, (size, size, 3), dtype=np.uint8) for _ in range(n)]
        self.targets = [int(i % num_classes) for i in range(n)]
        self.transform = transform

    def __len__(self):
        return len(self.arrs)

    def __getitem__(self, i):
        img = Image.fromarray(self.arrs[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[i]


class _TinyEncoder(nn.Module):
    """Fallback encoder when torchvision is unavailable."""

    def __init__(self, out=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, out, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)


def _ok(msg):
    print(f"  [ok] {msg}")


def _selftest(with_data: bool = False) -> int:
    t0 = time.time()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    print("=" * 72)
    print(f"harness self-test  (torch {torch.__version__}, torchvision "
          f"{'absent' if not _HAS_TV else torchvision.__version__})")
    print("=" * 72)

    # -- 1. seeding / device ------------------------------------------------
    set_seed(0)
    x1 = torch.randn(4)
    set_seed(0)
    assert torch.allclose(x1, torch.randn(4))
    assert get_device(prefer_cpu=True).type == "cpu"
    _ok("set_seed reproducible; get_device(prefer_cpu) -> cpu")

    dev = torch.device("cpu")

    # -- 2. augmentations ---------------------------------------------------
    for name, cfg in [("weak", AugCfg.weak()), ("strong", AugCfg.strong()),
                      ("none", AugCfg.none())]:
        assert cfg.size == 32
    assert AugCfg.none().is_identity() and not AugCfg.strong().is_identity()
    _ok("AugCfg presets weak/strong/none; none() is the identity policy (T5)")

    if _HAS_TV:
        img = Image.fromarray(np.random.RandomState(0).randint(0, 255, (40, 40, 3), dtype=np.uint8))
        tv = make_two_view_transform(AugCfg.strong(size=32), AugCfg.weak(size=32))
        v1, v2 = tv(img)
        assert v1.shape == (3, 32, 32) and v2.shape == (3, 32, 32)
        assert not torch.allclose(v1, v2)
        ev = make_eval_transform(32)
        e1, e2 = ev(img), ev(img)
        assert e1.shape == (3, 32, 32) and torch.allclose(e1, e2)
        gray = Image.fromarray(np.random.RandomState(1).randint(0, 255, (28, 28), dtype=np.uint8))
        assert make_eval_transform(32)(gray).shape == (3, 32, 32)
        idt = make_single_view_transform(AugCfg.none(size=32))
        assert torch.allclose(idt(img), idt(img))
        _ok("make_two_view_transform / make_eval_transform / make_single_view_transform "
            "(3xHxW, deterministic eval, grayscale->RGB, identity policy)")
    else:
        print("  [skip] transforms: torchvision missing")

    # -- 3. long tail -------------------------------------------------------
    labels = np.repeat(np.arange(5), 100)
    idx = longtail_indices(labels, 5, gamma=0.01, seed=0)
    cnt = np.bincount(labels[np.asarray(idx)], minlength=5)
    assert cnt[0] == 100 and cnt[4] == 1 and all(cnt[i] >= cnt[i + 1] for i in range(4)), cnt
    bal = longtail_indices(labels, 5, gamma=1.0, seed=0)
    assert len(set(np.bincount(labels[np.asarray(bal)], minlength=5))) == 1
    assert longtail_indices(labels, 5, 0.01, 3) == longtail_indices(labels, 5, 0.01, 3)
    _ok(f"longtail_indices exponential profile {cnt.tolist()} (gamma=0.01), "
        f"balanced at gamma=1, seed-deterministic")

    # -- 4. models ----------------------------------------------------------
    if _HAS_TV:
        enc, fd = make_encoder("resnet18", "cifar")
        assert fd == 512
        assert enc.conv1.kernel_size == (3, 3) and isinstance(enc.maxpool, nn.Identity)
        with torch.no_grad():
            assert enc(torch.randn(2, 3, 32, 32)).shape == (2, 512)
        enc_i, fdi = make_encoder("resnet18", "imagenet")
        assert enc_i.conv1.kernel_size == (7, 7) and not isinstance(enc_i.maxpool, nn.Identity)
        _ok("make_encoder resnet18 cifar stem 3x3/no-maxpool, feat_dim=512; imagenet stem intact")
        probe_enc = enc
        probe_dim = 512
    else:
        print("  [skip] make_encoder: torchvision missing")
        probe_enc = _TinyEncoder(32)
        probe_dim = 32

    mlp = make_mlp(16, 32, 8, n_layers=3, bn=True)
    assert mlp(torch.randn(5, 16)).shape == (5, 8)
    assert isinstance(mlp[-1], nn.Linear)  # unbounded output for vicreg (T2)
    assert make_mlp(16, 32, 8, n_layers=1)(torch.randn(5, 16)).shape == (5, 8)
    _ok("make_mlp n_layers=1/3, unbounded final Linear")

    student = make_mlp(8, 16, 8, n_layers=2)
    ema = EMATeacher(student, tau_base=0.9, tau_final=1.0, total_steps=10)
    tgt = ema.target
    assert all(not p.requires_grad for p in tgt.parameters()) and not tgt.training
    before = [p.clone() for p in tgt.parameters()]
    with torch.no_grad():
        for p in student.parameters():
            p.add_(1.0)
    ema.update(0)
    after = list(ema.target.parameters())
    moved = [float((b - a).abs().max()) for b, a in zip(before, after)]
    assert max(moved) > 0 and abs(ema.last_tau - 0.9) < 1e-6
    assert ema.tau_at(10) == 1.0 and ema.tau_at(0) == 0.9
    tgt.train()
    assert not ema.target.training  # property re-forces eval()
    _ok("EMATeacher frozen+eval, cosine tau 0.9->1.0, buffers copied, target re-evals")

    heads = frozen_random_projections(64, 8, k=4, seed=0, disjoint=True)
    assert len(heads) == 4
    assert sum(p.numel() for h in heads for p in h.parameters()) == 0
    masks = [set(h.idx.tolist()) for h in heads]
    assert all(len(m) == 16 for m in masks)
    for i in range(4):
        for j in range(i + 1, 4):
            assert not (masks[i] & masks[j])
    xz = torch.randn(6, 64)
    outs = [h(xz) for h in heads]
    assert all(o.shape == (6, 8) for o in outs)
    assert not any(o.requires_grad for o in outs)
    nd = frozen_random_projections(64, 8, k=3, seed=0, disjoint=False)
    assert nd[0].idx is None and nd[0](xz).shape == (6, 8)
    try:
        frozen_random_projections(4, 8, k=8, seed=0, disjoint=True)
        raise AssertionError("expected ValueError for in_dim < k")
    except ValueError:
        pass
    _ok("frozen_random_projections: 0 trainable params (buffers), disjoint masks, "
        "no-grad outputs, in_dim<k rejected")

    # -- 5. objectives ------------------------------------------------------
    set_seed(1)
    B, d = 16, 12
    s = torch.randn(B, d, requires_grad=True)
    a = torch.randn(B, d, requires_grad=True)

    L = gram_loss(s, a)
    L.backward()
    assert s.grad is not None and s.grad.abs().sum() > 0
    assert a.grad is None or float(a.grad.abs().sum()) == 0.0, "target must be stop-gradient"
    _ok("gram_loss: gradient flows to s only (sg on the target Gram)")

    # T1: O(d) invariance of the relational target
    Q, _ = torch.linalg.qr(torch.randn(d, d))
    with torch.no_grad():
        s0, a0 = torch.randn(B, d), torch.randn(B, d)
        base = float(gram_loss(s0, a0))
        rot = float(gram_loss(s0 @ Q, a0 @ Q))
        base_pw = float(gram_loss(s0, a0, temperature=0.5, pos_weight=1.0))
        rot_pw = float(gram_loss(s0 @ Q, a0 @ Q, temperature=0.5, pos_weight=1.0))
    assert abs(base - rot) < 1e-5, (base, rot)
    assert abs(base_pw - rot_pw) < 1e-5, (base_pw, rot_pw)
    _ok(f"T1 gram_loss is O(d)-invariant: {base:.6f} vs {rot:.6f} "
        f"(and with temperature+pos_weight: {base_pw:.6f} vs {rot_pw:.6f})")

    # T2: total collapse is a GLOBAL minimum of the relational target
    z = torch.randn(1, d).repeat(B, 1)
    assert float(gram_loss(z, z)) < 1e-10
    _ok("T2 gram_loss == 0 under total collapse (global minimum -> needs a legality term)")

    with torch.no_grad():
        assert float(gram_loss(s0, a0, pos_weight=0.0)) > 0
        assert float(gram_loss(s0, a0, temperature=1.0)) - base < 1e-6
    _ok("gram_loss knobs: pos_weight=0 -> pure off-diagonal, temperature=1 is a no-op")

    # prototype head
    K = 7
    protos = nn.Parameter(torch.randn(K, d))
    lp, c1 = prototype_loss(s0, a0, protos, tau_s=0.1, tau_t=0.04, center=None)
    assert c1.shape == (K,) and not c1.requires_grad
    lp2, c2 = prototype_loss(s0, a0, protos, tau_s=0.1, tau_t=0.04, center=c1)
    assert not torch.allclose(c1, c2)
    lrot, _ = prototype_loss(s0 @ Q, a0 @ Q, protos, tau_s=0.1, tau_t=0.04, center=None)
    assert abs(float(lp) - float(lrot)) > 1e-3, (float(lp), float(lrot))
    _ok(f"T1 prototype_loss is NOT O(d)-invariant: {float(lp):.6f} vs {float(lrot):.6f} "
        f"(prototypes fix a distinguished basis)")

    l_nc, _ = prototype_loss(s0, a0, protos, 0.1, 0.04, center=c1, use_center=False)
    l_ns, _ = prototype_loss(s0, a0, protos, 0.1, 0.04, center=c1, use_sharpen=False)
    assert abs(float(l_nc) - float(lp2)) > 1e-6 and abs(float(l_ns) - float(lp2)) > 1e-6
    sp = nn.Parameter(protos.detach().clone())
    lg, _ = prototype_loss(s0.detach().clone().requires_grad_(True), a0, sp, 0.1, 0.04)
    lg.backward()
    assert sp.grad is not None and float(sp.grad.abs().sum()) > 0
    _ok("prototype_loss: use_center/use_sharpen ablations change the loss; protos get gradient")

    # vicreg
    zu = torch.randn(32, 16, requires_grad=True) * 3.0
    lv, parts = vicreg_reg(zu, var_w=25.0, cov_w=1.0, gamma=1.0)
    assert set(parts) == {"var", "cov"} and lv.requires_grad
    lv.backward()
    coll = torch.randn(1, 16).repeat(32, 1) * 3.0
    lcoll, pc = vicreg_reg(coll)
    assert pc["var"] > 0.9, pc
    raised = False
    try:
        vicreg_reg(F.normalize(torch.randn(32, 16), dim=1))
    except ValueError as e:
        raised = "NORMALISED" in str(e) and "unit sphere" in str(e)
    assert raised, "vicreg_reg must reject L2-normalised z"
    _ok(f"vicreg_reg: parts={ {k: round(v, 4) for k, v in parts.items()} }, "
        f"collapse -> var={pc['var']:.3f}, and it RAISES on L2-normalised z (T2)")

    ln = info_nce(s0, a0, tau=0.2)
    assert float(ln) > 0
    perfect = F.normalize(torch.randn(B, d), dim=1) * 5.0
    assert float(info_nce(perfect, perfect, tau=0.01)) < 1e-3
    _ok(f"info_nce: random={float(ln):.4f}, identical views ~0")

    # -- 6. eval ------------------------------------------------------------
    N, dz = 200, 24
    Zc = torch.randn(1, dz).repeat(N, 1)
    Zf = torch.randn(N, dz)
    er_c, er_f = effective_rank(Zc), effective_rank(Zf)
    assert er_c < 1.01 and er_f > 0.5 * dz, (er_c, er_f)
    _ok(f"effective_rank: collapsed={er_c:.3f}, isotropic={er_f:.2f} (d={dz})")

    Xc = torch.randn(64, 10)
    assert abs(linear_cka(Xc, Xc) - 1.0) < 1e-4
    R, _ = torch.linalg.qr(torch.randn(10, 10))
    assert abs(linear_cka(Xc, Xc @ R * 3.0) - 1.0) < 1e-4
    cka_ind = linear_cka(Xc, torch.randn(64, 10))
    assert cka_ind < 0.9
    _ok(f"linear_cka: self=1.0, rotation+scale invariant=1.0, independent={cka_ind:.3f}")

    z1 = F.normalize(torch.randn(128, 16), dim=1)
    u_r, a_r = uniformity_alignment(z1, F.normalize(torch.randn(128, 16), dim=1))
    u_s, a_s = uniformity_alignment(z1, z1)
    assert a_s < 1e-6 and a_r > a_s
    zcol = F.normalize(torch.randn(1, 16), dim=1).repeat(128, 1)
    u_c, _ = uniformity_alignment(zcol, zcol)
    assert u_c > u_r, (u_c, u_r)
    _ok(f"uniformity_alignment -> (unif, align): random=({u_r:.3f}, {a_r:.3f}), "
        f"same=({u_s:.3f}, {a_s:.1e}), collapsed unif={u_c:.3f}")

    # probes on cached synthetic features (linearly separable by construction)
    set_seed(2)
    ncls = 4
    ntr, nte = 240, 120
    proto_v = torch.randn(ncls, 3, 16, 16) * 2.0
    ytr_ = torch.randint(0, ncls, (ntr,))
    yte_ = torch.randint(0, ncls, (nte,))
    Xtr_ = proto_v[ytr_] + 0.5 * torch.randn(ntr, 3, 16, 16)
    Xte_ = proto_v[yte_] + 0.5 * torch.randn(nte, 3, 16, 16)
    tr_ds, te_ds = TensorDataset(Xtr_, ytr_), TensorDataset(Xte_, yte_)
    small_enc = _TinyEncoder(32)
    res = linear_probe(small_enc, tr_ds, te_ds, epochs=30, lr=1e-2, bs=128,
                       device=dev, num_classes=ncls)
    assert set(res) == {"acc", "per_class_acc", "head_acc", "tail_acc"}
    assert len(res["per_class_acc"]) == ncls and 0.0 <= res["acc"] <= 1.0
    assert not math.isnan(res["head_acc"]) and not math.isnan(res["tail_acc"])
    knn = knn_probe(small_enc, tr_ds, te_ds, k=5, device=dev)
    assert 0.0 <= knn <= 1.0
    _ok(f"linear_probe acc={res['acc']:.3f} head={res['head_acc']:.3f} "
        f"tail={res['tail_acc']:.3f}; knn_probe acc={knn:.3f}")

    # encoder train/eval mode must be restored by the probes
    small_enc.train()
    linear_probe(small_enc, tr_ds, te_ds, epochs=1, bs=128, device=dev, num_classes=ncls)
    knn_probe(small_enc, tr_ds, te_ds, k=5, device=dev)
    assert small_enc.training, "probes must restore the encoder's train mode"
    small_enc.eval()
    Xa, ya = _extract_features(small_enc, tr_ds, dev)
    assert Xa.shape[0] == ntr and ya.shape[0] == ntr
    _ok("_extract_features / probes: single no_grad pass, encoder mode restored")

    # probes must NOT disturb the training RNG stream (they are called mid-run)
    set_seed(7)
    ref = torch.randn(3)
    set_seed(7)
    linear_probe(small_enc, tr_ds, te_ds, epochs=2, bs=128, device=dev, num_classes=ncls)
    knn_probe(small_enc, tr_ds, te_ds, k=5, device=dev)
    assert torch.allclose(ref, torch.randn(3)), "probe leaked into the global RNG"
    r1 = linear_probe(small_enc, tr_ds, te_ds, epochs=5, bs=128, device=dev,
                      num_classes=ncls, probe_seed=3)
    r2 = linear_probe(small_enc, tr_ds, te_ds, epochs=5, bs=128, device=dev,
                      num_classes=ncls, probe_seed=3)
    assert r1["acc"] == r2["acc"]
    _ok("probes run in an RNG island: global stream untouched, probe_seed reproducible")

    if _HAS_TV:
        inv_ds = _TinyImageDataset(n=8, size=32, transform=make_eval_transform(32))
        s_id = invariance_score(small_enc, inv_ds, AugCfg.none(size=32), n_views=4,
                                device=dev, n_samples=8)
        s_st = invariance_score(small_enc, inv_ds, AugCfg.strong(size=32), n_views=4,
                                device=dev, n_samples=8)
        assert s_id > 0.999, s_id
        assert s_st <= s_id + 1e-6
        assert getattr(inv_ds, "transform", None) is not None  # restored
        _ok(f"invariance_score: identity policy={s_id:.4f} (==1), strong={s_st:.4f}, "
            f"dataset.transform restored")
    else:
        print("  [skip] invariance_score: torchvision missing")

    # -- 7. runs / budget ---------------------------------------------------
    tmpd = tempfile.mkdtemp(prefix="harness_selftest_")
    r = Run("demo", tmpd, {"lr": 0.1, "aug": AugCfg.weak()}, resume=False)
    r.log(0, loss=1.0, acc=0.1)
    r.log(1, loss=torch.tensor(0.5), acc=0.2, extra=np.float32(3.0))  # widening header
    r.save_ckpt(1, model={"w": torch.zeros(2)}, best_metric=0.2)
    r.log(2, loss=0.4, acc=0.3)
    r.save_ckpt(2, model={"w": torch.ones(2)}, best_metric=0.3)
    r.log(3, loss=0.3, acc=0.4)          # ahead of the checkpoint on purpose
    ck = r.load_ckpt()
    assert ck["step"] == 2 and ck["best_metric"] == 0.3
    assert os.path.exists(r.best_path)
    r.finish({"final_acc": 0.4})
    assert json.load(open(r.summary_path))["summary"]["final_acc"] == 0.4

    r2 = Run("demo", tmpd, {"lr": 0.1}, resume=True)   # simulate a killed session
    assert r2.resumed_step == 2 and r2.best_metric == 0.3
    with open(r2.csv_path) as f:
        rows = list(csv.DictReader(f))
    assert [int(x["step"]) for x in rows] == [0, 1, 2], rows
    r2.log(3, loss=0.25, acc=0.45)
    _ok("Run: csv flush + header widening, atomic ckpt, best tracking, "
        "resume truncates rows ahead of the checkpoint")

    for seed in range(3):
        rs = Run(f"agg_seed{seed}", tmpd, {"seed": seed}, resume=False)
        for st in range(3):
            rs.log(st, seed=seed, acc=0.1 * st + 0.01 * seed)
        rs.finish({})
    agg = aggregate_seeds(os.path.join(tmpd, "agg_seed*.csv"), key="step")
    assert len(agg) == 3 and "acc_mean" in agg.columns and "acc_std" in agg.columns
    assert int(agg["n_seeds"].iloc[0]) == 3
    _ok(f"aggregate_seeds: {len(agg)} steps x 3 seeds, "
        f"acc_mean={agg['acc_mean'].tolist()}")

    matched_budget_check([
        {"name": "gram", "steps": 1000, "batch_size": 256, "forward_passes": 2},
        {"name": "proto", "steps": 1000, "batch_size": 256, "forward_passes": 2},
    ])
    raised = False
    try:
        matched_budget_check([
            {"name": "gram", "steps": 1000, "batch_size": 256, "forward_passes": 2},
            {"name": "proto", "steps": 1000, "batch_size": 512, "forward_passes": 2},
        ])
    except ValueError:
        raised = True
    assert raised
    raised = False
    try:
        matched_budget_check([{"name": "a", "steps": 1}, {"name": "b", "steps": 1}])
    except ValueError:
        raised = True
    assert raised, "missing keys must be rejected"
    _ok("matched_budget_check: passes matched arms, raises on mismatch and on missing keys")

    # -- 8. optional data path ---------------------------------------------
    if with_data and _HAS_TV:
        print("  ... downloading CIFAR-10 (--with-data)")
        ds = get_ssl_dataset("cifar10", AugCfg.strong(32), AugCfg.weak(32),
                             imbalance_gamma=0.1, seed=0, root="./data")
        it = ds[0]
        assert set(it) == {"v1", "v2", "idx", "label"}
        assert it["v1"].shape == (3, 32, 32)
        cnts = np.bincount(ds.labels, minlength=10)
        assert cnts[0] > cnts[9]
        tr, te = get_eval_datasets("cifar10", 32, root="./data")
        assert len(tr) == 50000 and len(te) == 10000
        enc, _ = make_encoder("resnet18", "cifar")
        bat = probe_battery(enc, ["cifar10"], device=dev, root="./data", size=32, epochs=1)
        assert "per_task" in bat and "worst" in bat and "mean" in bat
        _ok(f"get_ssl_dataset (long-tail {cnts.tolist()}), get_eval_datasets, probe_battery")
    else:
        print("  [skip] get_ssl_dataset / get_eval_datasets / probe_battery "
              "(pass --with-data to exercise the download paths)")

    dt = time.time() - t0
    print("=" * 72)
    print(f"SELF-TEST PASSED in {dt:.1f} s")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(_selftest(with_data="--with-data" in sys.argv))
