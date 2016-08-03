import numpy as np


def fisher_matrix(spectra, masses, snr=100, transformation=None,
                  unc=None, sigma_contribution=False, **extras):
    """ Calculate the Fisher information Matrix.  Currently does not work for
    anything other than uncorrelated gaussian errors.
    """
    mu = np.dot(masses, spectra)
    if unc is None:
        unc = mu / snr
    # Inverse covariance matrix
    invSigma = np.diag(1./unc**2) # Uncorrelated errors

    partial_mu = spectra
    # Transform the amplitudes
    if transformation is not None:
        partial_mu *= transformation[:, None]
        
    fisher = np.dot(partial_mu, np.dot(invSigma, partial_mu.T))
    # Do it out explicitly?  I think this is the same as
    #for i in range(nage):
    #    for j in range(nage):
    #        fisher[i, j] = np.dot(partial_mu[i,:], np.dot(Sigma, partial_mu[j,:].T))

    if sigma_contribution:
        partial_sigma = 2 * unc * partial_mu / snr
        raise(NotImplementedError)

    return fisher, mu

def fisher_matrix_old(spectra, masses, snr=100, unc=None,
                      sigma_contribution=False, **extras):
    # Old style, using einsum
    mu = np.dot(masses, spectra)
    if unc is None:
        unc = mu / snr
    partial_mu = spectra / unc
    partial_sigma = spectra / mu
    fisher = np.einsum('ik,jk->ij', partial_mu, partial_mu)
    if sigma_contribution:
        fisher += 2 * np.einsum('ik,jk->ij', partial_sigma, partial_sigma)
    return fisher, mu


def cramer_rao_bound(spectra, masses, snr=100, covariances=True,
                     renormalize=False, regularize=False, **extras):
    """Calculate the Cramer-Rao Bound.

    :param masses:
        Wavelength array, ndarray of shape (nw,)

    :param spectra:
        Spectral basis, ndarray of shape (nbasis, nw)
    """

    fisher, mu = fisher_matrix(spectra, masses, snr=snr, **extras)

    if covariances:
        if renormalize:
            # Renormalize by the diagonal before trying to invert.
            # Probably useless
            Kinv = np.diag(1.0/np.sqrt(np.diag(fisher)))
            Kinv = np.diag(masses)
            V = np.dot(Kinv, np.dot(fisher, Kinv))
            ch = np.linalg.cholesky(V)
            Vinv = np.linalg.inv(V)
            crb = np.dot(Kinv, np.dot(Vinv, Kinv))
        elif regularize:
            # Add something to the diagonal to try and make it invertible.
            # Tends to give a huge overestimate....
            ev = np.linalg.eigvals(fisher)
            lam = ev.real.min()
            if lam < 0:
                lam = abs(lam)
            else:
                lam = 0.0
            crb = np.linalg.inv(fisher + np.diag(np.ones(fisher.shape[0]) + lam*1.1))
        else:
            # Just invert the damn thing
            try:
                ch = np.linalg.cholesky(fisher)
                crb = np.linalg.inv(fisher)
            except(np.linalg.LinAlgError):
                print("Fisher matrix not positive definite")
                crb = np.linalg.pinv(fisher)
    else:
        # Get the constraints without including covariances in the parameters.
        # Never do this.
        crb = np.diag(1./np.diag(fisher))

    return crb, mu
