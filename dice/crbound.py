import numpy as np


def fisher_matrix(spectra, masses, snr=100, transformation=None,
                  unc=None, sigma_contribution=False,
                  dust_curves=None, A_V=None, **extras):
    """ Calculate the Fisher information Matrix.  Currently does not work for
    anything other than uncorrelated gaussian errors.

    :param spectra:
        ndarray of shape (nbasis, nwave).  The stellar SSPs

    :param masses:
        fiducial masses used to normalize each components, ndarray of shape
        (nbasis,)

    :param transformation: (optional)
        Multiplies the spectra to transform the normalization parameter.
        ndarray of shape (nbasis,)
    """

    # Attenuate SSPs by dust
    if dust_curves is not None:
        R = np.atleast_2d(dust_curves) # nbasis x nwave or 1 x nwave
        att = np.exp(-np.atleast_2d(A_V).T * R) #nbasis x nwave or 1 x nwave
        fluxes = spectra * att # nbasis x nwave
    else:
        fluxes = spectra

    # Mock spectrum (Mean of gaussian from which data are drawn)
    mu = np.dot(masses, fluxes)
    if unc is None:
        unc = mu / snr
    # Inverse covariance matrix
    invSigma = np.diag(1./unc**2) # Uncorrelated errors

    # Derivatives with respect to mass/amplitude
    partial_mu = fluxes
    # Transformations of the mass parameter to some other amplitude
    if transformation is not None:
        partial_mu *= transformation[:, None]        

    # Add dust derivatives
    if dust_curves is not None:
        if att.shape[0] > 1:
            partial_mu_partial_A = R * masses[:, None] * fluxes # nbasis x nwave
        else:
            partial_mu_partial_A = R * np.atleast_2d(mu) # 1 x nwave
        partial_mu = np.vstack([partial_mu, partial_mu_partial_A])
        
    fisher = np.dot(partial_mu, np.dot(invSigma, partial_mu.T))
    # Do it out explicitly?  I think the above is the same as
    #for i in range(nage):
    #    for j in range(nage):
    #        fisher[i, j] = np.dot(partial_mu[i,:], np.dot(invSigma, partial_mu[j,:].T))

    if sigma_contribution:
        partial_sigma = 2 * unc * partial_mu / snr
        raise(NotImplementedError)

    return fisher, mu


def fisher_matrix_dust(spectra, masses, dust_curves=None, A_V=None):
    """
    :param dust_curves:
       Dust attenuation curves A_lambda/A_V, ndarray of shape (nbasis, nwave) or just (nwave,)

    :param A_V:
        Normalization of the dust curves. scalar or ndarray of shape (nbasis,).
    """
    R = np.atleast_2d(dust_curves) # nbasis x nwave or 1 x nwave
    att = np.exp(-np.atleast_2d(A_V).T * R) #nbasis x nwave or 1 x nwave
    fluxes = spectra * att # nbasis x nwave

    mu = np.dot(masses, fluxes) # nwave
    partial_mu_partial_m = fluxes # nbasis x nwave
    if att.shape[0] > 1:
        partial_mu_partial_A = R * masses[:, None] * fluxes # nbasis x nwave
    else:
        partial_mu_partial_A = R * np.atleast_2d(mu) # 1 x nwave

    partial_mu = np.vstack([partial_mu_partial_m, partial_mu_partial_A])

    fisher = np.dot(partial_mu, np.dot(invSigma, partial_mu.T))

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


def svd_comp(spectra, masses, snr=100, realizations=0):
    f = np.dot(masses, spectra)
    U, w, Vt = np.linalg.svd(spectra.T)

    norm_true = np.dot(U.T, f)
    if realizations > 0:
        noise = f /snr * np.random.normal(size=(realizations, len(f)))        
        norm_noise = np.dot(U, noise.T)
        return norm_true, f.mean() / snr, norm_noise
    
    return norm_true, f.mean() / snr
