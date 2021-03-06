import numpy as np
from scipy.interpolate import interp1d as interpolate
try:
    from sedpy import observate
except:
    pass


def rectify_basis(inwave, spectra, wlow=0, whigh=np.inf, redshift=0.0,
                  exclude=[], outwave=None, filters=None, **extras):
    """Mask a spectral basis using lists of include and exclude ranges
    """

    wave = inwave * (1 + redshift)
    
    if filters is not None:
        flist = observate.load_filters(filters)
        sed = observate.getSED(wave, spectra, filterlist=flist)
        return np.array([f.wave_effective for f in flist]), 10**(-0.4 * sed)

    if outwave is not None:
        onedinterp = interpolate(wave, spectra, axis=-1)
        spectra = onedinterp(outwave)
        wave = outwave

    g = (wave >= wlow) & (wave <= whigh)
    for (lo, hi) in exclude:
        g = g & ((wave < lo) | (wave > hi))
    return wave[g], spectra[:, g]


def get_binned_spectral_basis(sps, agebins, zbins=[0.], **kwargs):
    """
    :param agebins:
        Array giving lower and upper log(age) limits for each bin, of shape
        (nbin, 2)

    :returns spectra:
        Array of shape nbin, nspec giving the partial spectra of each bin (in
        L/formed_mass)
    """
    for z in zbins:
        sps.params['logzsol'] = z
        sps.params['agebins'] = agebins
        sps._ages = None
        sps.params['mass'] = np.ones(len(agebins))
        sps.params['mass_units'] = 'mformed'
        wght = sps.all_ssp_weights
        wght = sps._bin_weights
        w, spec_ssp = sps.ssp.get_spectrum(tage=0, peraa=True)
        spec = np.dot(wght[:, 1:], spec_ssp)
        try:
            np.vstack(spectra, spec)
        except:
            spectra = spec

    return rectify_basis(sps.ssp.wavelengths, spectra, **kwargs)


def get_full_spectral_basis(sps, zlist=None, good_age=slice(None), **kwargs):
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
    return rectify_basis(wave, spectra, **kwargs)
