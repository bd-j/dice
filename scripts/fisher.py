import sys
import numpy as np
import matplotlib.pyplot as pl

import fsps

from crbound import cramer_rao_bound
from basis import get_full_spectral_basis as full_basis

fwhm_to_sigma = 2.35

# Set up the SPS
sps = fsps.StellarPopulation(zcontinuous=0)
w = sps.wavelengths
nage = len(sps.ssp_ages)
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
    snrs = [1e2, 1e3, 1e4]
    whi = 9000.0
    wlo = 3800.0

    # Spectral basis
    good_age = sps.ssp_ages >= 6.8 
    t = 10**(sps.ssp_ages[good_age])
    wave, spectra = full_basis(sps, zlist=None, good_age=good_age)
    # Uncertainty: Constant SFH
    mock_masses =  np.exp(np.log(t/t.min()))

    # Set up figure
    cfig, caxes = pl.subplots(2, 1)
    cax, crax = caxes

    for snr in snrs:
        crb, mu = cramer_rao_bound(wave, spectra, mock_masses,
                                   wlo=wlo, whi=whi, snr=snr)
        cax.plot(sps.ssp_ages[good_age], np.sqrt(np.diag(crb)), 'o',
                 label='CRB, SNR/pix={:3.0e}'.format(snr))
        crax.plot(np.log10(t), mock_masses / np.sqrt(np.diag(crb)), '-o',
                  label='SNR/pix={:3.0e}'.format(snr))

    cax.plot(np.log10(t), mock_masses, '-o', label='Input masses')
    cax.set_yscale('log')
    cax.set_ylabel('Stellar mass of SSP')
    cax.set_title(r'${:4.0f}<\lambda<{:4.0f}, R={}$'.format(wlo, whi, R))
    cax.legend(loc=0, prop={'size':10})
    crax.axhline(1.0, linestyle=':', color='k')
    crax.set_ylabel('$m/\delta_m$')
    crax.set_xlabel('SSP Age (log years)')
    crax.set_yscale('log')
    crax.set_ylim(1e-2, 10)
    cfig.savefig('../figures/const_crb_full_{}.pdf'.format(lib))


    dia = 1/np.sqrt(np.diag(crb))
    corr = dia[:, None] * crb * dia[None, :]
    
    sys.exit()

    # s_i \dot s_j / (|s_i||s_j|)
    sdot = np.einsum('ik,jk->ij', spectra[:,g], spectra[:,g])
    dia = 1/np.sqrt(np.diag(fisher))
    sdot_norm = dia[:, None] * sdot * dia[None, :]
    rfig, rax = pl.subplots(figsize=(10.5, 8))
    rcb = rax.imshow(np.rad2deg(np.arccos(sdot_norm)))
    rax.set_title('Angle between adjacent flux vectors ($s_i \cdot s_j / (|s_i||s_j|)$, degrees)')
    rfig.colorbar(rcb)
    rfig.savefig('ssp_angles_{}.pdf'.format(lib))
    pl.show()
