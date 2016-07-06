import sys
import numpy as np
import matplotlib.pyplot as pl
from scipy.special import gammainc

from prospect.sources import StepSFHBasis 

from .crbound import cramer_rao_bound
from basis import get_binned_spectral_basis as get_basis

sps = StepSFHBasis()
w = sps.wavelengths

sspages = sps.ssp.ssp_ages

def expand_bin(allages, ibin, relative_precision=1.0,
               snr=100):
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
    i = -1
    while (unconstrained):
        i += 1
        if ibin+i+1 >= len(allages):
            print('Could not reach desired precision')
            break
        ages = allages[:ibin+1] + allages[ibin+1+i:]
        agebins = np.array([ages[:-1], ages[1:]]).T
        wave, spectra = get_basis(sps, agebins)

        # constant SFR
        mock_masses = np.squeeze(np.diff(10**agebins, axis=-1))
        mock_masses /= mock_masses.sum()
        crb, mu = cramer_rao_bound(wave, spectra, mock_masses,
                                   wlo=wlo, whi=whi, snr=snr)
        sigma = np.sqrt(np.diag(crb))
        bin_precision = mu[ibin]/sigma[ibin]
        unconstrained =  bin_precision < relative_precision

    return ages, i, bin_precision
