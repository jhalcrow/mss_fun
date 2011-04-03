from gdsampler import *


#TODO
def test_synthetic_data():
    N = 50
    L = 20
    T = 12
    K = 4
    mu0 = np.zeros(T)
    sigma0 = 20
    alpha = 0.5

    samples = 100
    mu_total = np.zeros(T)
    for i in xrange(samples):
        (mu, theta, sampler) = generate_synthetic_data(N, L, T, K, mu0,
                                                       sigma0, alpha)
        mu_est = sampler.genre_timbre_mean()
        assert(np.max(np.abs(mu_est - mu)) < 3 * np.sqrt(sigma0))
        mu_total += mu.sum(axis=0)

    trials = samples * K
    assert(np.max(mu_total / (samples * K) - mu0) < \
           3 * sigma0 / np.sqrt(samples * K))


#TODO
def test_sampler():
    N = 50
    L = 20
    T = 12
    K = 4
    mu0 = np.zeros(T)
    sigma0 = 20
    alpha = 0.5

    (mu, theta, sampler) = generate_synthetic_data(N, L, T, K, mu0,
                                                   sigma0, alpha)


#TODO
def test_optimize_alpha():
    pass
