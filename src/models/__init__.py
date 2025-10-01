"""Neural network models for autoencoding.

This module contains the autoencoder and variational autoencoder models
used for patch encoding.
"""

from .ae import AE
from .vae import BetaVAE

__all__ = ['AE', 'BetaVAE']
