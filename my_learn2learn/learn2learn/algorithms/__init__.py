#!/usr/bin/env python3

r"""
A set of high-level algorithm implementations, with easy-to-use API.
"""

from .maml import MAML, maml_update
from .meta_sgd import MetaSGD, meta_sgd_update
from .gbml import GBML
from .maml_dp import MAML_DP, maml_update
from .lightning import (
    LightningEpisodicModule,
    LightningMAML,
    LightningANIL,
    LightningPrototypicalNetworks,
    LightningMetaOptNet,
)
