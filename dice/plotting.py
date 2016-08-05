import matplotlib.pyplot as pl
import numpy as np


def plot_sfh(ax, allages, crb, invals, unit='', filters=None, plabel=None,
             **kwargs):
    """Marginalized SFH uncertainties and input SFR.
    """
    #fig, ax = pl.subplots()
    #ax.plot((allages[1:] + allages[:-1])/2., np.sqrt(np.diag(crb)), '-o')
    punc = np.sqrt(np.diag(crb))
    ax.step(allages,  np.append(punc, 0), where='post', label='Uncertainty',
            linewidth=2)
    ax.step(allages,  np.append(invals, 0), where='post', label='Input',
            linewidth=2)
    ax.legend(loc=0)
    
    ax.axhline(1.0, linestyle=':', color='k', linewidth=1.5)
    ax.set_xlabel('lookback time (log yrs)')
    ax.set_xlim(max(allages.min(), 6.5), allages.max())
    
    ax.set_ylabel(unit)
    ax.set_yscale('log')
    ax.set_ylim(1e-2, 1e3)
    
    if filters is not None:
        ax.set_title('Photometry ({} bands)'.format(len(filters)))
    else:
        ax.set_title('Spectroscopy ({wlow}-{whigh}$\AA$, R=2.5$\AA$)'.format(**kwargs))

    return ax

def plot_covariances(ax):
    pass
    # Covariances
