import numpy as np


def cramer_rao_bound(wave, spectra, masses, sigma_contribution=True,
                     covariances=True, snr=100, **extras):
    """
    :param wave:
        Wavelength array, ndarray of shape (nw,)

    :param spectra:
        Spectral basis, ndarray of shape (nbasis, nw)
    """
    mu = np.dot(masses, spectra)
    unc = mu / snr

    # likelihood derivatives with respect to mu
    partial_mu = spectra / unc
    # likelihood derivatives with respect to Sigma
    partial_sigma = spectra / mu
    # Fisher information matrix including both terms
    fisher = np.einsum('ik,jk->ij', partial_mu, partial_mu)
    if sigma_contribution:
        fisher += 2 * np.einsum('ik,jk->ij', partial_sigma, partial_sigma)

    if covariances:
        crb = np.linalg.pinv(fisher)
    else:
        crb = np.diag(1./np.diag(fisher))
    return crb, mu


