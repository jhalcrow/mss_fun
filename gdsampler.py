import numpy as np
from itertools import izip
from bisect import bisect
class GaussDirichletSampler(object):
    """
    Gibbs sampler for a model similar to LDA but with the
    words replaced by vectors that have a Gaussian
    distribution.
    """

    def __init__(self, songs, K, alpha, mu0, sigma0, genres=None):
        self.songs = songs
        self.K = K
        self.alpha = alpha
        self.mu0 = mu0
        self.sigma0 = sigma0

        self.N = len(self.songs)
        self.T = len(self.songs[0][0])
        self.genres = genres if genres else self.randomize_state()
        self.update_genre_params()

    def randomize_state(self):
        self.genres = [np.random.random_integers(0, self.K-1, self.N)
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
            timbre_mu = self.timbre_totals / self.genre_count
            pdf = (sgc + self.alpha) / (len(song) + self.K * self.alpha)  
            # TODO: Improve the prior
            pdf *= np.exp(-np.power(timbre_mu - song[i], 2) / (2 * self.sigma0**2))

            # Sample new genre
            np.cumsum(pdf, out=cdf)
            k_new = bisect(cdf, np.random.random())
            
            self.timbre_totals[k_new] += song[i]
            self.genre_count[k_new} += 1
            sgc[k_new] += 1

def generate_synthetic_corpus(N, L, T, mu0, sigma0, alpha):

    alpha = alpha if hasattr(alpha, __iter__) else [alpha] * self.T
    songs = np.empty((self.N, self.L, self.T))
    genres = np.random.dirichlet(alpha, (self.N, self.L))
    genre_mean = np.random.multivariate_normal(mu0, sigma0 * np.eye(self.T), self.T)
    for n in xrange(N):
        for l in xrange(L):
            k = self.genres[n, l]
            songs[n,l] = np.random.multivariate_normal(genre_mean[k], np.eye(self.T) * sigma0)

    return (genre_mean, GaussDirichletSampler(songs, K, alpha0, mu0, sigma0, genres))
