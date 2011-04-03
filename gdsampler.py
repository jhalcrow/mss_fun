from collections import defaultdict
from itertools import izip
from bisect import bisect

import numpy as np


class GaussDirichletSampler(object):
    """
    Gibbs sampler for a model similar to LDA but with the
    words replaced by vectors that have a Gaussian
    distribution.
    """

    def __init__(self, songs, K, alpha, mu0, sigma0, genres=None):
        self.songs = songs
        self.K = K
        self.alpha = np.array(alpha)
        self.mu0 = mu0
        self.sigma0 = sigma0

        self.N = len(self.songs)
        self.T = len(self.songs[0][0])
        self.genres = self.randomize_state() if genres is None else genres
        self.update_statistics()

    def randomize_state(self):
        self.genres = [np.random.random_integers(0, self.K - 1, self.N)
                            for n in xrange(self.N)]

    def update_statistics(self):
        self.timbre_totals = np.zeros((self.K, self.T))
        self.genre_count = np.zeros(self.K, dtype=np.uint16)
        self.song_genre_count = np.zeros((self.N, self.K), dtype=np.uint16)

        for (song, genr, song_cts) in izip(self.songs,
                                          self.genres,
                                          self.song_genre_count):
            for (segment, k) in izip(song, genr):
                self.timbre_totals[k] += segment
                self.genre_count[k] += 1
                song_cts[k] += 1

    def genre_timbre_posterior(self):
        '''
        Returns the posterior distribution for each genre's timbre
        distribution as the tuple (mu, cov) with
        mu: list of genre-timbre means
        cov: list of genre_timbre covariance matrices
        '''
        inv = np.linalg.inv

        mu_emp = np.transpose(self.timbre_totals.T / self.genre_count)
        cov_emp = self.genre_timbre_cov()
        cov_emp_inv = [inv(np.matrix(cov)) for cov in cov_emp]

        cov_post = [np.matrix(inv(1. / self.sigma0 +
                                  self.genre_count[k] * cov_emp_inv[k]))
                    if self.genre_count[k] > 0
                    else np.eye(self.K) * self.sigma0
                    for k in xrange(self.K)
                   ]
        mu_post = [cov_post[k] *
                   np.matrix(self.mu0 / self.sigma0 +
                             self.genre_count[k] * cov_emp_inv[k] * mu_emp[k].reshape(self.T, 1))
                   if self.genre_count[k] > 0
                   else self.mu0[k]
                   for k in xrange(self.K)
                  ]
        return (mu_post, cov_post)

    def genre_timbre_cov(self):
        '''
        Computes the covariance of the timbre for each genre.
        If there are no observed instances of a genre, None is returned.
        TODO: This is a painfully slow way to do this.
        Do incrementally?
        '''
        genre_insts = [list() for i in xrange(self.K)]
        for (song, genr) in izip(self.songs, self.genres):
            for (t, k) in izip(song, genr):
                genre_insts[k].append(t)

        cov = [np.cov(np.transpose(g)) if len(g) > 0 else None for g in genre_insts]
        return cov

    def sample_song(self, j):
        genres = self.genres[j]
        song = self.songs[j]
        sgc = self.song_genre_count[j]

        cdf = np.zeros(self.K)
        pdf = np.zeros(self.K)

        for i in xrange(len(song)):
            # Hold out this segment
            k_old = genres[i]
            self.timbre_totals[k_old] -= song[i]
            self.genre_count[k_old] -= 1
            sgc[k_old] -= 1

            # Compute p(k_ji | everything else)
            timbre_mu = self.genre_timbre_mean()
            pdf = (sgc + self.alpha) / (len(song) + self.K * self.alpha)
            (mu, sigma) = self.genre_timbre_posterior()
            # TODO: Improve the prior
            for k in xrange(self.K):
                dev = mu[k] - song[i]
                suf_stat = dev * np.inv(sigma) * dev.reshape(self.T, 1)
                pdf[k] *= (np.abs(sigma) ** -0.5) * np.exp(-suf_stat)

            # Sample new genre
            np.cumsum(pdf, out=cdf)
            k_new = bisect(cdf, np.random.random() * cdf[-1])

            self.timbre_totals[k_new] += song[i]
            self.genre_count[k_new] += 1
            sgc[k_new] += 1

    def iterate(self, iters, verbose=True):
        for i in xrange(iters):
            if verbose and i % 10 == 0:
                print 'Iteration %d' % i
            for song in xrange(self.N):
                self.sample_song(song)


def generate_synthetic_data(N, L, T, K, mu0, sigma0, alpha):

    alpha = alpha if hasattr(alpha, '__iter__') else [alpha] * K
    songs = np.empty((N, L, T))
    theta = np.random.dirichlet(alpha, N)
    genres = np.empty((N, L), dtype=np.uint32)
    genre_mean = np.random.multivariate_normal(mu0, sigma0 * np.eye(T), K)

    for n in xrange(N):
        cdf = np.cumsum(theta[n])
        for l in xrange(L):
            genres[n, l] = bisect(cdf, np.random.random())
            k = genres[n, l]
            assert(k < K)
            songs[n, l] = np.random.multivariate_normal(
                genre_mean[k],
                np.eye(T) * sigma0
            )

    return (genre_mean,
            theta,
            GaussDirichletSampler(songs, K, alpha, mu0, sigma0, genres)
           )


def load_timbre():
    pass
