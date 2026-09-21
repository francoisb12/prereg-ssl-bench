#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run an ÜBERGANG experiment with the code on Drive and the I/O where it belongs.

Why this exists
---------------
Mounting Drive and running everything inside it looks like the obvious thing to
do, and it breaks in three different ways.  Each of the three kinds of file this
bench touches wants a different home:

  CODE      read once at import.  Drive FUSE is perfectly fine for that, and
            running the code from Drive means an edit there takes effect on the
            next run with nothing to copy.

  DATA      CIFAR / STL / DTD are read on EVERY epoch.  Through FUSE that is a
            throughput wall: the GPU waits on the network.  Re-downloading them
            to the VM costs a minute or two per session, which is far cheaper
            than paying FUSE on every batch for hours.  -> always local.

  RESULTS   the delicate one.  The harness flushes and fsyncs the CSV on EVERY
            logged row (so a session cut loses nothing) and writes checkpoints
            atomically with os.replace().  Per-row fsync over FUSE is slow
            enough to stall training, and os.replace() across a FUSE mount is
            not reliably atomic -- which defeats the whole point of writing it
            that way.  -> local while running, synced to Drive periodically and
            on exit.

What it does, in order
----------------------
 1. RESTORES the experiment's previous results from Drive into the VM.  This is
    what makes --resume work ACROSS a session cut: without it, every new Colab
    session starts from zero and the resume logic has nothing to resume from.
 2. Injects --data-root and --outdir unless you passed them yourself.
 3. Runs the command, streaming its output.
 4. Syncs results back to Drive every --sync-every seconds, AND on exit --
    including on Ctrl-C, on a crash, and on the runtime being reclaimed, as far
    as the process gets a chance to run its handler.

Usage
-----
    python colab_run.py -- python xp/xp_A_diagnostics.py --exp a1
    python colab_run.py --drive /content/drive/MyDrive/ubergang_xp \\
                        -- python xp/xp_C_panel.py --exp c1

Everything after ``--`` is the command, passed through untouched.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DRIVE = "/content/drive/MyDrive/ubergang_xp"
DEFAULT_LOCAL = "/content/ubergang_local"


def script_letter(cmd: List[str]) -> Optional[str]:
    """'xp/xp_C_panel.py' -> 'C'.  Used to pick the right results subdirectory."""
    for tok in cmd:
        m = re.search(r"xp_([A-F])_", os.path.basename(str(tok)))
        if m:
            return m.group(1)
    return None


def has_flag(cmd: List[str], flag: str) -> bool:
    return any(t == flag or str(t).startswith(flag + "=") for t in cmd)


def copy_tree(src: str, dst: str) -> int:
    """Copy src into dst, newer-only, returning the number of files copied."""
    if not os.path.isdir(src):
        return 0
    n = 0
    for root, _dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        out = os.path.join(dst, rel) if rel != "." else dst
        os.makedirs(out, exist_ok=True)
        for f in files:
            s, d = os.path.join(root, f), os.path.join(out, f)
            try:
                if os.path.exists(d) and os.path.getmtime(d) >= os.path.getmtime(s) \
                        and os.path.getsize(d) == os.path.getsize(s):
                    continue
                # Write to a sibling temp then replace, so a sync interrupted
                # halfway never leaves a truncated file behind on either side.
                tmp = d + ".part"
                shutil.copy2(s, tmp)
                os.replace(tmp, d)
                n += 1
            except OSError as exc:
                print(f"[sync] échec sur {s}: {exc}", file=sys.stderr)
    return n


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:
        cut = argv.index("--")
        own, cmd = argv[:cut], argv[cut + 1:]
    else:
        own, cmd = argv, []

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drive", default=DEFAULT_DRIVE,
                   help="racine du banc sur Drive (contient lib/, xp/, results/)")
    p.add_argument("--local", default=DEFAULT_LOCAL,
                   help="racine locale VM pour data/ et results/")
    p.add_argument("--sync-every", type=float, default=120.0,
                   help="secondes entre deux synchronisations vers Drive (0 = seulement à la fin)")
    p.add_argument("--no-restore", action="store_true",
                   help="ne PAS rapatrier les résultats de Drive au démarrage "
                        "(--resume ne pourra alors rien reprendre d'une session précédente)")
    args = p.parse_args(own)

    if not cmd:
        p.error("rien à lancer. Mets la commande après '--', "
                "ex: colab_run.py -- python xp/xp_A_diagnostics.py --exp a1")

    drive_results = os.path.join(args.drive, "results")
    local_data = os.path.join(args.local, "data")
    local_results = os.path.join(args.local, "results")
    os.makedirs(local_data, exist_ok=True)
    os.makedirs(local_results, exist_ok=True)

    letter = script_letter(cmd)
    if letter is None:
        print("[colab_run] aucun script xp_X_*.py reconnu dans la commande ; "
              "--outdir ne sera pas injecté.", file=sys.stderr)
    sub = letter or ""
    local_out = os.path.join(local_results, sub) if sub else local_results
    drive_out = os.path.join(drive_results, sub) if sub else drive_results

    # --- 1. restore ---------------------------------------------------------
    if not args.no_restore:
        if os.path.isdir(drive_out):
            t0 = time.time()
            n = copy_tree(drive_out, local_out)
            print(f"[colab_run] restauré {n} fichier(s) depuis {drive_out} "
                  f"en {time.time() - t0:.0f}s -> --resume peut repartir de là")
        else:
            print(f"[colab_run] rien à restaurer ({drive_out} n'existe pas encore)")

    # --- 2. inject paths ----------------------------------------------------
    full = list(cmd)
    if not has_flag(full, "--data-root"):
        full += ["--data-root", local_data]
    if letter and not has_flag(full, "--outdir"):
        full += ["--outdir", local_out]

    env = dict(os.environ)
    # Running the code from Drive would otherwise litter it with __pycache__,
    # which is slow over FUSE and pointless.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    print("[colab_run] code    :", args.drive)
    print("[colab_run] data    :", local_data, "(local, jamais Drive)")
    print("[colab_run] results :", local_out, "->", drive_out)
    print("[colab_run] cmd     :", " ".join(str(t) for t in full))
    print("-" * 72, flush=True)

    # --- 3. background sync -------------------------------------------------
    stop = threading.Event()

    def sync(tag: str) -> None:
        try:
            n = copy_tree(local_out, drive_out)
            print(f"\n[colab_run] sync {tag}: {n} fichier(s) -> Drive", flush=True)
        except Exception as exc:                       # noqa: BLE001
            print(f"\n[colab_run] sync {tag} a échoué: {exc}", file=sys.stderr, flush=True)

    def loop() -> None:
        while not stop.wait(args.sync_every):
            sync("périodique")

    thread = None
    if args.sync_every and args.sync_every > 0:
        thread = threading.Thread(target=loop, daemon=True)
        thread.start()

    rc = 1
    try:
        proc = subprocess.Popen(full, cwd=args.drive, env=env)
        # Forward Ctrl-C to the child so it can write its last checkpoint.
        def on_sigint(_sig, _frm):                     # noqa: ANN001
            proc.send_signal(signal.SIGINT)
        try:
            signal.signal(signal.SIGINT, on_sigint)
        except (ValueError, OSError):
            pass                                       # not in the main thread
        rc = proc.wait()
    except FileNotFoundError as exc:
        print(f"[colab_run] commande introuvable: {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\n[colab_run] interrompu", file=sys.stderr)
    finally:
        # The whole point: whatever happened, the results reach Drive.
        stop.set()
        if thread is not None:
            thread.join(timeout=1.0)
        sync("final")

    print(f"[colab_run] terminé, code de retour {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
