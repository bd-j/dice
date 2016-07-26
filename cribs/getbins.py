import sys
import numpy as np
import matplotlib.pyplot as pl
from scipy.special import gammainc

from crbound import cramer_rao_bound
from sfhs import *

def possible_bins(inbinedges, verbose=True, **kwargs):
    """Iteratively expand a bin until the desired output precision of the
    amplitude of the bin is reached.

    :param kwargs:
        All extra kwargs are passed to expand_bin
    """
    outbins, outprec, outnb = [], [], []
    ibin=0
    while ibin < len(inbinedges) - 1:
        ages, npb, pr = expand_bin(inbinedges.tolist(), ibin, **kwargs)
        blo, bhi = ibin, min(ibin+npb+1, len(inbinedges)-1)
        if verbose:
            print('{:4.3f} {:4.3f} {:4.3f} {}'.format(inbinedges[blo], inbinedges[bhi], pr, npb))
        outbins.append([inbinedges[blo], inbinedges[bhi]])
        outprec.append(pr)
        outnb.append(npb)
        ibin += npb+1
    return np.array(outbins), np.array(outprec), np.array(outnb)


def expand_bin(allages, ibin, relative_precision=1.0,
               snr=100, get_basis=None, sps=None, sfh=constant, **kwargs):
    """Given a sorted, ascending list of bin edges, expand the ``ibin``th bin to
    larger values until the desired relative precision is acheived, for a
    constant SFH.

    :param allages:
        The starting set of bin edges, as a sorted ascending list, units of log(yrs).

    :param ibin:
        The bin to expand.  That is, the new bin will go from allages[ibin] to
        allages[ibin+ncomb+1].

    :param relative_precision: (optional, default: 1.0)
        The desired relative precision for the new bin, defined here as
        value/dispersion.
        
    :returns ages:
        The bin edges with the ``ibin``th bin expanded.

    :returns ncomb:
        The number of bins that were combined to form the new bin

    :returns bin_precision:
        The actual upper limit to the acheivable precision for this bin.
    """
    bin_precision = 0
    unconstrained = True
    i = -1
    while (unconstrained):
        i += 1
        if ibin+i+1 >= len(allages):
            print('Could not reach desired precision')
            break
        ages = allages[:ibin+1] + allages[ibin+1+i:]
        agebins = np.array([ages[:-1], ages[1:]]).T
        wave, spectra = get_basis(sps, agebins, **kwargs)

        mock_masses = sfh(agebins, **kwargs)

        crb, mu = cramer_rao_bound(wave, spectra, mock_masses,
                                   snr=snr, **kwargs)
        sigma = np.sqrt(np.diag(crb))
        bin_precision = mock_masses[ibin] / sigma[ibin]
        unconstrained =  bin_precision < relative_precision

    print(spectra.shape, wave.shape)
    return ages, i, bin_precision


def native_bins(allages, relative_precision=1.0,
                snr=100, get_basis=None, sps=None, sfh=constant, **kwargs):
    agebins = np.array([allages[:-1], allages[1:]]).T
    wave, spectra = get_basis(sps, agebins, **kwargs)
    mock_masses = sfh(agebins, **kwargs)
    crb, mu = cramer_rao_bound(wave, spectra, mock_masses,
                               snr=snr, **kwargs)
    sigma = np.sqrt(np.diag(crb))
    bin_precision = mock_masses / sigma
    return mock_masses, bin_precision, crb, spectra, wave
