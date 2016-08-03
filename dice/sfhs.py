import numpy as np
from scipy.special import gamma, gammainc

__all__ = ["constant", "exponential"]


def constant(agebins, **extras):
    mock_masses = np.squeeze(np.diff(10**agebins, axis=-1))
    mock_masses /= mock_masses.sum()
    return mock_masses


def exponential(agebins, tau=1.0, power=1, tage=None, **extras):
    if tage is None:
        tage = 10**(agebins.max()-9)
    normalized_times = np.clip(tage - 10**(agebins-9), 0, np.inf)
    m = tau * gammainc(power, normalized_times / tau)
    mock_masses = np.squeeze(np.diff(m, axis=-1))
    mock_masses /= mock_masses.sum()
    return mock_masses
