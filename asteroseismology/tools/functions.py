
#!/usr/bin/env/ python
# coding: utf-8


import numpy as np
from astropy.timeseries import LombScargle


__all__ = ["gaussian", "lorentzian", 
        "spectral_response", "standard_background_model"]

def gaussian(x, mu, sigma, height):
    '''
    Return the value of gaussian given parameters.

    Input:
    x: array-like[N,]
    mu, sigma, height: float

    Output:
    y: the dependent variable of the time series.

    '''
    return height * np.exp(-(x-mu)**2.0/(2*sigma**2.0))


def lorentzian(x, mu, gamma, height):
    '''
    Return the value of lorentzian given parameters.

    Input:
    x: array-like[N,]
    mu, gamma, height: float

    Output:
    y: the dependent variable of the time series.

    '''
    return height / (1 + (x-mu)**2.0/gamma**2.0)


def spectral_response(x, fnyq):
	sincfunctionarg = (np.pi/2.0)*x/fnyq
	response = (np.sin(sincfunctionarg)/sincfunctionarg)**2.0
	return response


def standard_background_model(x, params, fnyq, NHarvey=3, if_return_oscillation=True):
    '''
    Return the value of gaussian given parameters.

    Input:
    x: array-like[N,]
    params: flatNoiseLevel, heightOsc, numax, widthOsc, 
            ampHarvey1, freqHarvey1, powerHarvey1,
            (ampHarvey2, freqHarvey2, powerHarvery2,
            (ampHarvey3, freqHarvey3, powerHarvery3))
    fnyq: float, the nyquist frequency in unit of [x]
    NHarvey: int, the number of Harvey profiles
    if_return_oscillation: bool

    Output:
    y: array-like[N,]

    '''
    
    flatNoiseLevel, heightOsc, numax, widthOsc = params[0:4]
    power = np.zeros(len(x))

    zeta = 2.0*2.0**0.5/np.pi
    for iHarvey in range(NHarvey):
        ampHarvey, freqHarvey, powerHarvey = params[iHarvey*3+4:iHarvey*3+7]
        power += zeta*ampHarvey**2.0/(freqHarvey*(1+(x/freqHarvey)**powerHarvey))

    if if_return_oscillation:
        power += heightOsc * np.exp(-1.0*(numax-x)**2/(2.0*widthOsc**2.0))

    power *= spectral_response(x, fnyq)
    power += flatNoiseLevel
    return power


