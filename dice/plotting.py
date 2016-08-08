import matplotlib.pyplot as pl
import numpy as np
#import matplotlib.mlab as mlab
from scipy.stats import multivariate_normal
from scipy.linalg import cho_factor, cho_solve
from numpy.core.umath_tests import inner1d


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


def plot_covariances(axes, crb, invals, unit='', ptiles=[0.683, 0.955],
                     filters=None, plabel=None, color='red',
                     nbin=100, nsigma=4., **extras):

    n = crb.shape[0]
    for i in range(n):
        ax = axes[i,i]
        dx = nsigma * np.sqrt(crb[i,i])
        x = np.linspace(max(invals[i] - dx, 0), invals[i] + dx, nbin)
        ax.plot(x, np.exp(gauss(x, crb[i,i], mu=invals[i])))
        for j in range(i+1, n):
            ax = axes[j, i]
            dy = nsigma * np.sqrt(crb[j,j])
            y = np.linspace(max(invals[j] - dy, 0), invals[j] + dy, nbin)
            mean = np.array([invals[i], invals[j]])
            covar = np.array([[crb[i,i], crb[i,j]], [crb[j,i], crb[j,j]]])
            pdf = twod_gauss(x, y, covar, mu=mean)
            levels = -cdf_to_level(np.array(ptiles))
            ax.contour(pdf[0], pdf[1], pdf[2], levels=levels, colors=color)
    return axes
            

def multigaussian(X, Sigma, mu=0., intervals=True, **extras):
    """Values of a multivariate gaussian

    :param flatX:
         ndarray of shape (ndim, npoints)

    :param Sigma:
         Covariance matrix, ndarray of shape (ndim, ndim)

    :returns lnp:
         The ln of value of the gaussian at flatX

    :returns log_det:
         ln of the determinant
    """

    flatX = X - mu[:, None]
    n = Sigma.shape[0]
    icov = np.linalg.inv(Sigma)
    log_det =  np.log(np.linalg.det(Sigma))
    lnp = -0.5 * (inner1d(flatX.T, np.dot(icov, flatX).T))
    if not intervals:
        lnp -= 0.5 * (log_det + n * np.log(2.*np.pi))
    return lnp, log_det

def gauss(X, var, mu=0.):
    x = (X - mu) / var
    return -0.5 * x**2

def twod_gauss(x, y, covar, **kwargs):
    X, Y = np.meshgrid(x, y)
    pos = np.array([X,Y])
    dim = pos.shape
    flat_pos = pos.reshape(dim[0], dim[1]*dim[2])
    Z, logdet = multigaussian(flat_pos, covar, **kwargs)
    return X, Y, Z.reshape(dim[1], dim[2])


def cdf_to_level(ptile):
    """Convert between percentile and lnp value for a 2-d chi-square
    distribution
    """
    level = -2.0 * np.log(1 - ptile)
    return level
