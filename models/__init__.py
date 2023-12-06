from .base import *
from .vanilla_vae import *
from .gamma_vae import *
from .beta_vae import *
from .cvae import *
from .cat_vae import *
from .info_vae import *
from .betatc_vae import *


# Aliases
VAE = VanillaVAE
GaussianVAE = VanillaVAE
CVAE = ConditionalVAE
GumbelVAE = CategoricalVAE

vae_models = {
              'BetaVAE':BetaVAE,
              'InfoVAE':InfoVAE,
              'GammaVAE':GammaVAE,
              'BetaTCVAE':BetaTCVAE,
              'VanillaVAE':VanillaVAE,
              'ConditionalVAE':ConditionalVAE,
              'CategoricalVAE':CategoricalVAE}
