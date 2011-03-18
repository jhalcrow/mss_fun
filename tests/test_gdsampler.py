from gdsampler import *

#TODO
def test_synthetic_data():
    N = 50
    L = 20
    T = 12
    mu0 = 0
    sigma0 = 20
    alpha = 0.5

    (mu, sampler) = generate_synthetic_data(N, L, T, mu0, sigma0, alpha)

#TODO
def test_sampler():
    pass

#TODO
def test_optimize_alpha():
    pass
