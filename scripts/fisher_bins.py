import sys
import numpy as np
import matplotlib.pyplot as pl
from scipy.special import gammainc

from prospect.sources import StepSFHBasis 

from crbound import cramer_rao_bound
from basis import get_binned_spectral_basis as get_basis

sps = StepSFHBasis()
w = sps.wavelengths
nage = len(sps.ssp.ssp_ages)
assert nage < 100
if len(w) > 1e4:
    lib = 'ckc'
    R = 3000
elif len(w) > 5e3:
    lib = 'miles'
    R = 2.54
else:
    lib = 'basel'
    R = 100


if __name__ == "__main__":
    snr = 100
    whi = 9000.0
    wlo = 3800.0

    
    ages = [6.7, 7.5, 8.0, 8.5, 8.8, 9.0, 9.25, 9.6, 9.9, 10.15]
    agebins = np.array([ages[:-1], ages[1:]]).T
    logt = np.mean(agebins, axis=-1)
    wave, spectra = get_basis(sps, agebins)

    # constant SFR
    mock_masses = np.squeeze(np.diff(10**agebins, axis=-1))
    sfh='const'

    # exponential
    tau = 3
    #sfh = 'tau={}'.format(tau)
    m = tau * gammainc(1, (10**(agebins.max()-9) - 10**(agebins-9)) / tau)
    #mock_masses = np.squeeze(np.diff(m, axis=-1))

    # add a 5x burst
    burst_bin = 4
    #mock_masses[burst_bin] *= 5
    #sfh = 'tau={}_burst={}'.format(tau, burst_time)
    
    mock_masses /= mock_masses.sum()

    # MDF
    zbar = np.zeros(len(logt))
    

    crb, mu = cramer_rao_bound(wave, spectra, mock_masses,
                               wlo=wlo, whi=whi, snr=snr)

    # Plotting
    cfig, caxes = pl.subplots(2, 1, sharex=True)
    cax, crax = caxes
    cax.plot(logt, np.sqrt(np.diag(crb)), 'o', label='C-R Bound, SNR/pix={:3.0e}'.format(snr))
    cax.plot(logt, mock_masses, '-o', label='input masses')
    cax.set_yscale('log')
    cax.set_ylabel('Stellar mass')
    cax.set_title(r'${:4.0f}<\lambda<{:4.0f}, R={}$'.format(wlo, whi, R))
    cax.legend(loc=0, prop={'size':10})
    crax.plot(logt, mock_masses / np.sqrt(np.diag(crb)), '-o')
    crax.axhline(1.0, linestyle=':', color='k')
    crax.set_ylabel('$m/\delta_m$')
    crax.set_xlabel('Bin Center Age (log years)')
    #cax.set_xticklabels([])

    cfig.savefig('../figures/{}_crb_binned_{}.pdf'.format(sfh, lib))
    sys.exit()
    
    dia = 1/np.sqrt(np.diag(crb))
    corr = dia[:, None] * crb * dia[None, :]
    rfig, rax = pl.subplots()
    rcb = rax.imshow(corr, cmap='coolwarm')
    rfig.colorbar(rcb)
    rax.set_title(r'$\rho_{i,j}$')
    rfig.savefig('{}_corr.pdf'.format(sfh))
    
    pl.show()
    
