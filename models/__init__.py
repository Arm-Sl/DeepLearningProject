from .base import *
from .vanilla_vae import *
from .beta_vae import *


# Aliases
VAE = VanillaVAE
GaussianVAE = VanillaVAE

vae_models = {
              'BetaVAE':BetaVAE,
              'VanillaVAE':VanillaVAE,
            }
