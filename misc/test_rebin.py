import sys

import numpy as np
import matplotlib.pyplot as pl

from prospect.sources import StepSFHBasis 

from dice.getbins import build_rebin, rebin_sfh
from dice.basis import get_binned_spectral_basis as get_basis
from dice.sfhs import constant, exponential
from dice.crbound import cramer_rao_bound
#from dice.plotting import plot_sfh, plot_covariances


def plot_sfh(ax, allages, crb, truths=None, unit='', filters=None,
             plabel='Uncertainty', inlabel='Input', **kwargs):
    """Marginalized SFH uncertainties and input SFR.
    """
    punc = np.sqrt(np.diag(crb))
    if truths is not None:
        ax.step(allages,  np.append(truths, 0), where='post', label=inlabel,
                linewidth=2)
    ax.step(allages,  np.append(punc, 0), where='post', label=plabel,
            linewidth=2)
    
    #ax.axhline(1.0, linestyle=':', color='k', linewidth=1.5)
    ax.set_xlabel('lookback time (log yrs)')
    ax.set_xlim(max(allages.min(), 6.5), allages.max())
    
    ax.set_ylabel(unit)
    ax.set_yscale('log')
    

    return ax



filters = [
           ]


if __name__ == "__main__":
    sps = StepSFHBasis()
    sps.params['add_nebular_emission'] = False
    w = sps.wavelengths
    binedges = sps.ssp.ssp_ages
    binedges = binedges[binedges > 6.45]

    params = {'sps': sps,
              'get_basis': get_basis,
              'sfh': exponential,
              'power': 2,
              'tau': 5.,
              'tage': 10.0,
              'filters': None, #filters,
              'wlow': 3800,
              'whigh': 7000,
              'snr': 50000.0,
              'relative_precision': 0.5,
              'units': 'massfrac',
              'sigma_contribution': False,
              'covariances': True,
              'renormalize': False,
              'regularize': False,
              'rebin': 1,
              }
    params['outwave'] = np.arange(params['wlow'], params['whigh'], 2.54/2.0)
    plabel = '\n$S/N=${snr:.0f}/pix\n$tau$={tau}, $tage=${tage}'.format(**params)
        
    get_basis = params.pop('get_basis')
    sps = params.pop('sps')
    sfh = params.pop('sfh')

    allages = binedges[::params['rebin']]
    allages = np.insert(allages, 0, 4)
    if allages[-1] < binedges[-1]:
        allages = np.append(allages, binedges[-1])
    agebins = np.array([allages[:-1], allages[1:]]).T
    wave, spectra = get_basis(sps, agebins, **params)
    masses = sfh(agebins, **params)
    dt = np.squeeze(np.diff(10**agebins, axis=-1))
    sfr = masses / dt

    fig, axes = pl.subplots(3,1, sharex=False)
    props = dict(boxstyle='round', facecolor='w', alpha=0.5)
    
    #transform, unit = np.ones_like(dt), r'$M/ M_{total}$'
    transform, unit = 1 / (dt * masses.sum()), r'$SFR_i/M_{total}$'
    
    crb, mu = cramer_rao_bound(spectra, masses,  **params)
    tcrb = transform[:, None] * crb * transform[None, :]
    tmasses = masses * transform
    ax = plot_sfh(axes[0], allages, tcrb, tmasses, unit=unit,
                  plabel='SFH Uncertainty', inlabel='Input SFH', **params)
    ax.text(0.05, 0.95, 'Full resolution',
            transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    
    # --- now rebin two ways ----
    newages = np.array([4., 7.05, 8.1, 8.7, 9.35, 10.15])
    newagebins = np.array([newages[:-1], newages[1:]]).T
    dt = np.squeeze(np.diff(10**newagebins, axis=-1))
        
    # First way
    rmat = build_rebin(allages, newages)
    mu_new, crb_new = rebin_sfh(rmat, covar=crb, masses=masses)

    transform, unit = 1 / (dt * mu_new.sum()), r'$SFR_i/M_{total}$'
    tcrb = transform[:, None] * crb_new * transform[None, :]
    tmasses = mu_new * transform
    ax = plot_sfh(axes[1], newages, tcrb, tmasses, unit=unit,
                  plabel='Unc',
                  inlabel='SFH', **params)
    ax.text(0.05, 0.95, '5 bins\nMarginalized over intrabin SFH',
            transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    # second way
    wave, spectra_rebin = get_basis(sps, newagebins, **params)
    masses_rebin = sfh(newagebins, **params)
    crb_rebin, mu_rebin = cramer_rao_bound(spectra_rebin, masses_rebin, **params)

    transform, unit = 1 / (dt * masses_rebin.sum()), r'$SFR_i/M_{total}$'
    tcrb = transform[:, None] * crb_rebin * transform[None, :]
    tmasses = masses_rebin * transform

    ax = plot_sfh(axes[2], newages, tcrb, tmasses, unit=unit,
                  plabel='Unc',
                  inlabel='SFH', **params)
    ax.text(0.05, 0.65, '5 bins\nAssuming constant intrabin SFH',
            transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    # Marginalized uncertainties
    
    
    #ax = plot_sfh(axes[1], newages, crb_new, mu_new, unit=unit, ,**params)
    #ax = plot_sfh(axes[2], newages, crb_rebin, mu_new, unit=unit, plabel='Wide bins',**params)
    #props = dict(boxstyle='round', facecolor='w', alpha=0.5)
    #[ax.text(0.05, 0.95, plabel,
    #        transform=ax.transAxes, fontsize=14,
    #        verticalalignment='top', bbox=props) for ax in axes]

    [ax.set_ylim(1e-12, 1e-8) for ax in axes]
    axes[-1].set_ylim(1e-15, 1e-9)
    [ax.legend(loc='lower right') for ax in axes]
    axes[0].set_title('Spectroscopy ({wlow}-{whigh}$\AA$, R=2.5$\AA$, S/N/pix=$5\cdot 10^4$)'.format(**params))

    pl.show()
 
