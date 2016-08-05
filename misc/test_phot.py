import sys

import numpy as np
import matplotlib.pyplot as pl

from prospect.sources import StepSFHBasis 

#from getbins import possible_bins, native_bins
from dice.basis import get_binned_spectral_basis as get_basis
from dice.sfhs import constant, exponential
from dice.crbound import cramer_rao_bound
from dice.plotting import plot_sfh

codename = 'Dice'

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
              'tau': 5.,
              'tage': 10., #None,
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
    #allages = np.array([0, 8.0, 8.5, 9.0, 9.5, np.log10(13.6e9)]) #Leja bins
    agebins = np.array([allages[:-1], allages[1:]]).T
    
    wave, spectra = get_basis(sps, agebins, **params)
    masses = sfh(agebins, **params)
    dt = np.squeeze(np.diff(10**agebins, axis=-1))
    sfr = masses / dt

    ulabel = 'Uncertainty'
    if params['units'] == 'massratio':
        # likelihood derivatives with respect to m/m_in
        transform = masses
        unit = '$M/M_{input}$'
    elif params['units'] == 'sfr':
        # likelihood derivates with respect to sfr / <sfr>
        transform = dt * sfr.mean()
        unit = r'$SFR/ \langle SFR\rangle$'
    elif params['units'] == 'massfrac':
        # likelihood derivates with respect to mtot
        transform = np.ones(len(masses)) * masses.sum()
        unit = r'$M/ M_{total}$'

    crb, mu = cramer_rao_bound(spectra, masses, transformation=transform, **params)
        
    # Marginalized uncertainties
    fig, ax = pl.subplots()
    ax = plot_sfh(ax, allages, crb, masses/transform, unit=unit, **params)
    props = dict(boxstyle='round', facecolor='w', alpha=0.5)
    ax.text(0.05, 0.95, plabel,
            transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax.set_ylim(1e-2, 1e3)

    # Covariances

    
    pl.show()
