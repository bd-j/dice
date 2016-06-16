import numpy as np
import matplotlib.pyplot as pl
from scipy.special import gammainc

from prospect.sources import StepSFHBasis 
sps = StepSFHBasis()

ages = [6.7, 7.5, 8.0, 8.5, 8.8, 9.0, 9.25, 9.6, 9.9, 10.15]
agebins = np.array([ages[:-1], ages[1:]]).T
mass = np.ones(len(ages) - 1)
logt = np.mean(agebins, axis=-1)


def get_spectral_basis(agebins, zbins=[0.]):
    """
    :param agebins:
        array giving lower and upper log(age) limits for each bin, shape nbin, 2
        
    """
    for z in zbins:
        sps.params['logzsol'] = z
        sps.params['agebins'] = agebins
        sps.params['mass'] = mass
        sps.params['mass_units'] = 'mformed'
        wght = sps.all_ssp_weights
        wght = sps._bin_weights
        w, spec_ssp = sps.ssp.get_spectrum(tage=0)
        spec = np.dot(wght[:, 1:], spec_ssp)
        try:
            np.vstack(spectra, spec)
        except:
            spectra = spec

    return spectra

def cramer_rau_bound(wave, spectra, masses, sigma_contribution=True,
                     wlo=3800, whi=7000, snr=100):
    mu = np.dot(mock_masses, spectra)
    unc = np.ones(len(w))
    unc = mu / snr

    g = (wave > wlo) & (wave < whi)

    partial_mu = spectra[:, g] / unc[None, g]
    partial_sigma = spectra[:, g] / mu[None, g]
    fisher = np.einsum('ik,jk->ij', partial_mu, partial_mu)
    if sigma_contribution:
        fisher += 2 * np.einsum('ik,jk->ij', partial_sigma, partial_sigma)

    crb = np.linalg.pinv(fisher)
    return crb


if __name__ == "__main__":
    snr = 100
    whi = 7000.0
    wlo = 3800.0

    spectra = get_spectral_basis(agebins)

    # constant SFR
    mock_masses = np.squeeze(np.diff(10**agebins, axis=-1))
    sfh='const'

    # exponential
    tau = 3
    sfh = 'tau={}'.format(tau)
    m = tau * gammainc(1, (10**(agebins.max()-9) - 10**(agebins-9)) / tau)
    mock_masses = np.squeeze(np.diff(m, axis=-1))

    # add a 5x burst
    burst = 4
    mock_masses[burst] *= 5
    sfh = 'tau={}_burst={}'.format(tau, burst)
    
    mock_masses /= mock_masses.sum()

    # MDF
    zbar = np.zeros(len(logt))
    

    crb = cramer_rao_bound(w, spectra, mock_masses,
                           wlo=wlo, whi=whi, snr=snr)
    
    cfig, caxes = pl.subplots(2, 1, sharex=True)
    cax, crax = caxes
    cax.plot(logt, np.sqrt(np.diag(crb)), 'o', label='C-R Bound')
    cax.plot(logt, mock_masses, '-o', label='input masses')
    cax.set_yscale('log')
    cax.set_ylabel('Stellar mass')
    cax.set_title(r'${:4.0f}<\lambda<{:4.0f}, SNR={}$'.format(wlo, whi, snr))
    cax.legend(loc=0)
    crax.plot(logt, mock_masses / np.sqrt(np.diag(crb)), '-o')
    crax.axhline(1.0, linestyle=':', color='k')
    crax.set_ylabel('$m/\delta(m)$')
    crax.set_xlabel('Bin Center Age (log years)')
    #cax.set_xticklabels([])

    cfig.savefig('{}_crb.pdf'.format(sfh))
    
    dia = 1/np.sqrt(np.diag(crb))
    corr = dia[:, None] * crb * dia[None, :]
    rfig, rax = pl.subplots()
    rcb = rax.imshow(corr, cmap='coolwarm')
    rfig.colorbar(rcb)
    rax.set_title(r'$\rho_{i,j}$')
    rfig.savefig('{}_corr.pdf'.format(sfh))
    
    pl.show()
    
