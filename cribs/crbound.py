import numpy as np


def cramer_rao_bound(wave, spectra, masses, sigma_contribution=True,
                     wlo=3800, whi=7000, snr=100):
    mu = np.dot(masses, spectra)
    unc = mu / snr

    g = (wave > wlo) & (wave < whi)

    # likelihood derivatives with respect to mu
    partial_mu = spectra[:, g] / unc[None, g]
    # likelihood derivatives with respect to Sigma
    partial_sigma = spectra[:, g] / mu[None, g]
    # Fisher information matrix including both terms
    fisher = np.einsum('ik,jk->ij', partial_mu, partial_mu)
    if sigma_contribution:
        fisher += 2 * np.einsum('ik,jk->ij', partial_sigma, partial_sigma)

    crb = np.linalg.pinv(fisher)
    return crb, mu
