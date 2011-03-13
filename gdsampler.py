import numpy as np

class GaussDirichletSampler(object):
    """
    Gibbs sampler for a model similar to LDA but with the
    words replaced by vectors that have a Gaussian
    distribution.
    """

    def __init__(self, songs, K, alpha, mu0, sigma0):
        self.songs = songs
        self.K = K
        self.alpha = alpha
        self.mu0 = mu0
        self.sigma0 = sigma0
