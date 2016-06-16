import numpy as np
import matplotlib.pyplot as pl

import fsps
sps = fsps.StellarPopulation(zcontinuous=0)
nz = len(sps.zlegend)
zlist = [np.argmin(np.abs(sps.zlegend - 0.019)) + 1]
nage = len(sps.ssp_ages)
assert nage < 100

good_age = sps.ssp_ages >= 6.8 
t = 10**(sps.ssp_ages[good_age])

if __name__ == "__main__":
    snr = 10000
    whi = 7000.0
    wlo = 3800.0
    spectra = []
    for z in zlist:
        sps.params['sfh'] = 0
        sps.params['zmet'] = z+1
        w, s = sps.get_spectrum(tage=0, peraa=True)
        spectra += s[good_age, :].tolist()

    spectra = np.array(spectra)

    # uncertainty
    mock_masses =  np.exp(np.log(t/t.min()))
    mu = np.dot(mock_masses, spectra)
    unc = np.ones(len(w))
    unc = mu / snr

    wave = w
    g = (wave > wlo) & (wave < whi)

    # likelihood derivatives with respect to mu
    partial_mu = spectra[:, g] / unc[None, g]
    # likelihood derivatives with respect to Sigma
    partial_sigma = spectra[:, g] / mu[None, g]
    # Fisher information matrix including both terms
    fisher = np.einsum('ik,jk->ij', partial_mu, partial_mu)
    fisher += 2 * np.einsum('ik,jk->ij', partial_sigma, partial_sigma)
    #ffig, fax = pl.subplots()
    #cb = fax.imshow(np.log(fisher))
    #ffig.colorbar(cb)
    
    cfig, cax = pl.subplots()
    crb = np.linalg.pinv(fisher)
    cax.plot(sps.ssp_ages[good_age], np.sqrt(np.diag(crb)), 'o', label='C-R Bound')
    cax.plot(np.log10(t), mock_masses, '-o', label='Input masses')
    cax.set_yscale('log')
    cax.set_xlabel('SSP Age (log years)')
    cax.set_ylabel('Stellar mass')
    cax.set_title(r'${:4.0f}<\lambda<{:4.0f}, SNR={}$'.format(wlo, whi, snr))
    cax.legend(loc=0)
    

    dia = 1/np.sqrt(np.diag(crb))
    corr = dia[:, None] * crb * dia[None, :]

    # s_i \dot s_j / (|s_i||s_j|)
    sdot = np.einsum('ik,jk->ij', spectra[:,g], spectra[:,g])
    dia = 1/np.sqrt(np.diag(fisher))
    sdot_norm = dia[:, None] * sdot * dia[None, :]
    rfig, rax = pl.subplots(figsize=(10.5, 8))
    rcb = rax.imshow(np.rad2deg(np.arccos(sdot_norm)))
    rax.set_title('Angle between adjacent flux vectors ($s_i \cdot s_j / (|s_i||s_j|)$, degrees)')
    rfig.colorbar(rcb)
    rfig.savefig('ssp_angles.pdf')
    pl.show()
