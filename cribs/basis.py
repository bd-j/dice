import numpy as np


def get_binned_spectral_basis(sps, agebins, zbins=[0.]):
    """
    :param agebins:
        Array giving lower and upper log(age) limits for each bin, of shape
        (nbin, 2)
        
    """
    for z in zbins:
        sps.params['logzsol'] = z
        sps.params['agebins'] = agebins
        sps.params['mass'] = np.ones(len(agebins))
        sps.params['mass_units'] = 'mformed'
        wght = sps.all_ssp_weights
        wght = sps._bin_weights
        w, spec_ssp = sps.ssp.get_spectrum(tage=0)
        spec = np.dot(wght[:, 1:], spec_ssp)
        try:
            np.vstack(spectra, spec)
        except:
            spectra = spec

    return sps.wavelengths, spectra


def get_full_spectral_basis(sps, zlist=None,
                            good_age=slice(None)):
    """
    """
    spectra = []
    if zlist is None:
        # Find closest to solar
        zlist = [np.argmin(np.abs(sps.zlegend - 0.019)) + 1]
    for z in zlist:
        sps.params['sfh'] = 0
        sps.params['zmet'] = z+1
        w, s = sps.get_spectrum(tage=0, peraa=True)
        spectra += s[good_age, :].tolist()

    spectra = np.array(spectra)
    wave = w
    return wave, spectra
