"""Shared substrate for the Uebergang-SSL falsification experiments.

Experiment scripts should import from here (or from ``lib.harness`` directly)
and from nothing else in-house::

    from lib.harness import AugCfg, get_ssl_dataset, gram_loss, Run

The API contract lives in ``harness.py`` and is normative: signatures there are
depended upon by six experiment groups written in parallel.
"""

from .harness import *  # noqa: F401,F403
from .harness import __all__  # noqa: F401

__version__ = "1.0.0"
