import sys

import numpy as np
import matplotlib.pyplot as pl

from prospect.sources import StepSFHBasis 

#from getbins import possible_bins, native_bins
from basis import get_binned_spectral_basis as get_basis
from sfhs import constant, exponential

codename = 'Rasa'

filters = [
           'galex_FUV', 'galex_NUV',
           #'uvot_m2', 'uvot_w1', 'uvot_w2',
           'sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0',
           'twomass_J', 'twomass_H', 'twomass_Ks',
           'spitzer_irac_ch1', 'spitzer_irac_ch2', 'spitzer_irac_ch3'
           ]

if __name__ == "__main__":
    sps = StepSFHBasis()
    sps.params['add_nebular_emission'] = True
    w = sps.wavelengths
    binedges = sps.ssp.ssp_ages
    binedges = binedges[binedges > 6.6]

    params = {'sps': sps,
              'get_basis': get_basis,
              'sfh': exponential,
              'power': 2,
              'tau': 1.,
              'tage': 1., #None,
              'filters': filters,
              'wlow': 3800,
              'whigh': 7000,
              'snr': 20.0,
              'relative_precision': 0.5,
              'units': 'sfr',
              'sigma_contribution': False,
              'covariances': True,
              'renormalize': False,
              'regularize': False,
              'rebin': 4,
              }

    plabel = '\n$S/N=${snr:.0f}\n$tau$={tau}, $tage=${tage}'.format(**params)

    get_basis = params.pop('get_basis')
    sps = params.pop('sps')
    sfh = params.pop('sfh')

    allages = binedges[::params['rebin']]
    if allages[-1] < binedges[-1]:
        allages = np.append(allages, binedges[-1])
    allages = np.array([4.0, 7.5, 8.0, 8.5, 9.0, 9.5, 9.8, np.log10(13.6e9)])
    allages = np.array([4.0, 7.5, 8.3, 9.0, 9.5, 9.8, np.log10(13.6e9)])
    allages = np.array([0, 8.0, 8.5, 9.0, 9.5, np.log10(13.6e9)]) #Leja bins
    agebins = np.array([allages[:-1], allages[1:]]).T
    wave, spectra = get_basis(sps, agebins, **params)
    masses = sfh(agebins, **params)
    dt = np.squeeze(np.diff(10**agebins, axis=-1))
    sfr = masses / dt

    mu = np.dot(masses, spectra)
    unc = mu / params['snr']
    Sigma = np.diag(1./unc**2) # Uncorrelated errors

    if params['units'] == 'mfrac':
        # likelihood derivatives with respect to m/m_in
        partial_mu = spectra * masses[:,None] #/ unc
        unit = '$\Delta M/M$'
        norm = 1.0
        ulabel = None
    elif params['units'] == 'sfr':
        # likelihood derivates with respect to sfr
        partial_mu = spectra * dt[:, None] #/ unc
        unit = r'$SFR/ \langle SFR\rangle$'
        norm = sfr.mean()
        ulabel = 'Uncertainty'
    fisher = np.dot(partial_mu, np.dot(Sigma, partial_mu.T))
    try:
        ch = np.linalg.cholesky(fisher)
        crb = np.linalg.inv(fisher)
    except(np.linalg.LinAlgError):
        print('not positive definite!')
        crb = np.linalg.pinv(fisher)

    #pl.close('all')
    fig, ax = pl.subplots()
    #ax.plot((allages[1:] + allages[:-1])/2., np.sqrt(np.diag(crb)), '-o')
    punc = np.sqrt(np.diag(crb))
    ax.step(allages,  np.append(punc, 0) / norm, where='post', label=ulabel,
            linewidth=2)
    if params['units'] == 'sfr':
        ax.step(allages,  np.append(sfr, 0) / norm, where='post', label='Input',
                linewidth=2)
        ax.legend(loc=0)
    ax.axhline(1.0, linestyle=':', color='k', linewidth=1.5)
    ax.set_yscale('log')
    ax.set_title('Photometry ({} bands)'.format(len(filters)))
    ax.set_xlabel('lookback time (log yrs)')
    props = dict(boxstyle='round', facecolor='w', alpha=0.5)
    ax.text(0.05, 0.95, plabel,
            transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    ax.set_ylabel(unit)
    ax.set_xlim(max(allages.min(), 6.5), allages.max())
    ax.set_ylim(1e-2, 1e3)
    pl.show()
