import sys

import numpy as np
import matplotlib.pyplot as pl
from astropy.cosmology import WMAP9 as cosmo

from prospect.sources import StepSFHBasis 

#from getbins import possible_bins, native_bins
from dice.basis import get_binned_spectral_basis as get_basis
from dice.sfhs import constant, exponential
from dice.crbound import cramer_rao_bound
from dice.plotting import plot_sfh, plot_covariances


wide = ['f070w', 'f090w', 'f115w', 'f150w', 'f200w', 'f277w', 'f356w', 'f444w']
bands = wide
filters = ['jwst_{}'.format(b) for b in bands]


if __name__ == "__main__":
    sps = StepSFHBasis()
    sps.params['add_nebular_emission'] = False
    w = sps.wavelengths
    binedges = sps.ssp.ssp_ages
    binedges = binedges[binedges > 6.6]

    params = {'sps': sps,
              'get_basis': get_basis,
              'sfh': constant,#exponential,
              'power': 2,
              'tau': 0.5,
              #'tage': 10., #None,
              'redshift': 9.0,
              'filters': filters,
              'snr': 10.0,
              'units': 'sfr',
              'sigma_contribution': False,
              'covariances': True,
              'renormalize': False,
              'regularize': False,
              'nbin': len(filters)-1,
              'minage': 6.0
              }

    tuniv = cosmo.age(params['redshift']).value * 1e9
    params['tage'] = tuniv / 1e9
    
    plabel = '\n$S/N=${snr:.0f}\n$tau$={tau}, $tage=${tage:1.2f}'.format(**params)

    get_basis = params.pop('get_basis')
    sps = params.pop('sps')
    sfh = params.pop('sfh')

    allages = np.linspace(params['minage'], np.log10(tuniv), params['nbin']+1)
    agebins = np.array([allages[:-1], allages[1:]]).T

    wave, spectra = get_basis(sps, agebins, **params)
    masses = sfh(agebins, **params)
    dt = np.squeeze(np.diff(10**agebins, axis=-1))
    sfr = masses / dt
    mean_sfr = masses.sum() / 10**agebins.max()

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
    ax.set_xlim(params['minage'], np.log10(tuniv))
    ax.legend(loc=0)
    ax.set_xlabel('lookback time (log years)')

    # Covariances
    pl.rcParams['contour.negative_linestyle'] = 'solid'
    nb = len(sfr)
    cfig, caxes = pl.subplots(nb, nb)
    caxes = plot_covariances(caxes, crb, masses/transform, unit=unit,
                             nbin=500, nsigma=5, ptiles=[0.683, 0.955])
    [ax.set_visible(False) for ax in caxes.flat if ax.has_data() is False]
    [ax.tick_params(axis='both', which='major', labelsize=7) for ax in caxes.flat]
    pl.show()
