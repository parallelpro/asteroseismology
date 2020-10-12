#!/usr/bin/env/ python
# coding: utf-8


import numpy as np
from astropy.timeseries import LombScargle
from .functions import gaussian

__all__ = ["a_correlate", "c_correlate", 
        "smoothWrapper", "powerSpectrumSmoothWrapper",
        "medianFilter", "psd", "arg_closest_node",
        "quantile"]

def a_correlate(x, y, ifInterpolate=True, samplingInterval=None): 

    '''
    Generate autocorrelation coefficient as a function of lag.

    Input:
        x: array-like[N,]
        y: array-like[N,]
        ifInterpolate: True or False. True is for unevenly spaced time series.
        samplinginterval: float. Default value: the median of intervals.

    Output:
        lag: time lag.
        rho: autocorrelation coeffecient.

    '''

    if len(x) != len(y): 
        raise ValueError("x and y must have equal size.")

    if ifInterpolate:
        samplingInterval = np.median(np.diff(x)) if (samplingInterval is None) else samplingInterval
        xp = np.arange(np.min(x),np.max(x),samplingInterval)
        yp = np.interp(xp, x, y)
        x, y = xp, yp

    new_y = y - np.mean(y)
    aco = np.correlate(new_y, new_y, mode='same')

    N = len(aco)
    lag = x[int(N/2):N] - x[int(N/2)]
    rho = aco[int(N/2):N] / np.var(y)
    rho = rho / np.max(rho)

    return lag, rho


def c_correlate(x, y1, y2, ifInterpolate=True, samplingInterval=None): 
    '''
    Generate autocorrelation coefficient as a function of lag.

    Input:
        x: array-like[N,]
        y1: array-like[N,]
        y2: array-like[N,]
        ifInterpolate: True or False. True is for unevenly spaced time series.
        samplinginterval: float. Default value: the median of intervals.

    Output:
        lag: time lag.
        rho: autocorrelation coeffecient.

    '''

    if (len(x) != len(y1)) or (len(x) != len(y2)): 
        raise ValueError("x and y1 and y2 must have equal size.")

    if ifInterpolate:
        samplingInterval = np.median(x[1:-1] - x[0:-2]) if (samplingInterval is None) else samplingInterval
        xp = np.arange(np.min(x),np.max(x),samplingInterval)
        yp1 = np.interp(xp, x, y1)
        yp2 = np.interp(xp, x, y2)
        x, y1, y2 = xp, yp1, yp2

    new_y1 = y1 - np.mean(y1)
    new_y2 = y2 - np.mean(y2)
    aco = np.correlate(new_y1, new_y2, mode='same')

    N = len(aco)
    lag = x[int(N/2):N] - x[int(N/2)]
    rho = aco[int(N/2):N] / (np.std(yp1)*np.std(yp2))
    rho = rho / np.max(rho)

    return lag, rho


def powerSpectrumSmoothWrapper(freq, power, windowSize=0.25, windowType='flat',
                                samplingInterval=None):
    '''
    Return the moving average of a power spectrum, with a changing width of the window

    Input:
    freq: array-like[N,] in muHz
    power: array-like[N,]
    windowSize: float, in unit of the p-mode large separation
    windowType: flat/hanning/hamming/bartlett/blackman/gaussian
    samplingInterval: the time between adjacent sampling points.
    '''

    if len(freq) != len(power): 
        raise ValueError("freq and power must have equal size.")
        
    if samplingInterval is None: samplingInterval = np.median(freq[1:-1] - freq[0:-2])

    freqp = np.arange(np.min(freq),np.max(freq),samplingInterval)
    powerp = np.interp(freqp, freq, power)
    powersp = np.zeros(powerp.shape)
    
    numax = np.logspace(1,4,20)
    delta_nu = 0.263*numax**0.772 # stello+09 relation
    numax[0] = 0.
    numax = np.append(numax,np.inf)
    for iw in range(len(numax)-1):
        idx = (freqp>=numax[iw]) & (freqp<=numax[iw+1])
        window_len = int(delta_nu[iw]*windowSize/samplingInterval)
        if window_len % 2 == 0:
            window_len = window_len + 1
        if np.sum(idx) <= window_len:
            powersp[idx] = powerp[idx]
        else:
            inputArray = np.concatenate((np.ones(window_len)*powerp[idx][window_len:0:-1], powerp[idx], np.ones(window_len)*powerp[idx][-1:-window_len-1:-1]))
            powersp[idx] = smooth(inputArray, window_len, window=windowType)[window_len:window_len+np.sum(idx)]

    powers = np.interp(freq, freqp, powersp)
    return powers


def smoothWrapper(x, y, windowSize, windowType, samplingInterval=None):
    '''
    Wrapping a sliding-average smooth function.

    Input:
    x: the independent variable of the time series.
    y: the dependent variable of the time series.
    windowSize: the period/width of the sliding window.
    windowType: flat/hanning/hamming/bartlett/blackman/gaussian
    samplingInterval: the time between adjacent sampling points.

    Output:
    yf: the smoothed time series with the exact same points as x.

    '''

    if len(x) != len(y): 
        raise ValueError("x and y must have equal size.")
        
    if samplingInterval is None: samplingInterval = np.median(x[1:-1] - x[0:-2])

    xp = np.arange(np.min(x),np.max(x),samplingInterval)
    yp = np.interp(xp, x, y)
    window_len = int(windowSize/samplingInterval)
    if window_len % 2 == 0:
        window_len = window_len + 1
    ys = smooth(yp, window_len, window = windowType)
    yf = np.interp(x, xp, ys)

    return yf


def smooth(x, window_len = 11, window = "hanning"):
    # stole from https://scipy.github.io/old-wiki/pages/Cookbook/SignalSmooth
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman", "gaussian"]:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = x #np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    if window == "flat":
        w = np.ones(window_len,"d")
    elif window == "gaussian":
        w = gaussian(np.arange(-window_len*3, window_len*3,1), 
                    0, window_len, 1./(np.sqrt(2*np.pi)*window_len))
    else:
        w = eval("np."+window+"(window_len)") 
    
    y = np.convolve(w/w.sum(),s,mode="same")
    return y


def medianFilter(x, y, period, yerr=None):
    if yerr==None: iferror=False
    binsize = np.median(x[1:-1]-x[0:-2])
    kernelsize = int(period/binsize)
    if kernelsize%2==0: kernelsize+=1
    from scipy.signal import medfilt
    yf = medfilt(y,kernel_size=kernelsize)
    ynew = y/yf #y-yf
    if iferror: yerrnew = yerr/yf

    if iferror:
        return ynew, yerrnew
    else:
        return ynew


def psd(x, y, oversampling=1, freqMin=None, freqMax=None, freq=None, return_val="power"):
    """
    Calculate the power spectrum density for a discrete time series.
    https://en.wikipedia.org/wiki/Spectral_density


    Input:
    x: array-like[N,]
        The time array.

    y: array-like[N,]
        The flux array.


    Optional input:
    oversampling: float, default: 1
        The oversampling factor to control the frequency grid.
        The larger the number, the denser the grid.

    freqMin: float, default: frequency resolution

    freqMax: float, default: nyquist frequency


    Output:
    freq: np.array
        The frequency, in unit of [x]^-1.

    psd: np.array
        The power spectrum density, in unit of [y]^2/[x].
        https://en.wikipedia.org/wiki/Spectral_density


    Examples:
    >>> ts = np.load("flux.npy")
    >>> t = ts["time_d"]   # the time in day
    >>> f = ts["flux_mf"]   # the relative flux fluctuated around 1
    >>> f = (f-1)*1e6   # units, from 1 to parts per million (ppm)

    >>> freq, psd = se.psd(t, f, return_val="psd_new")
    >>> freq = freq/(24*3600)*1e6   # c/d to muHz
    >>> psd = psd*(24*3600)*1e-6   # ppm^2/(c/d) to ppm^2/muHz

    """

    if not (return_val in ["psd_old", "periodogram", "power", "amplitude", "psd_new", "window"]):
        raise ValueError("return_val should be one of ['psd_old', 'periodogram', 'power', 'amplitude', 'psd_new', 'window'] ")

    Nx = len(x)
    dx = np.median(x[1:]-x[:-1]) 
    fs = 1.0/dx
    Tobs = dx*len(x)
    fnyq = 0.5*fs
    dfreq = fs/Nx

    if freqMin is None: freqMin = dfreq
    if freqMax is None: freqMax = fnyq

    if freq is None: freq = np.arange(freqMin, freqMax, dfreq/oversampling)
    
    if return_val == "psd_old":
        p = LombScargle(x, y).power(freq, normalization='psd')*dx*4.
    if return_val == "periodogram":
        p = LombScargle(x, y).power(freq, normalization='psd')
    if return_val == "power":
        p = LombScargle(x, y).power(freq, normalization='psd')/Nx*4.
    if return_val == "amplitude":
        p = np.sqrt(LombScargle(x, y).power(freq, normalization='psd')/Nx*4.)
    if return_val == "psd_new":
        nu = 0.5*(freqMin+freqMax)
        freq_window = np.arange(freqMin, freqMax, dfreq/10)
        power_window = LombScargle(x, np.sin(2*np.pi*nu*x)).power(freq_window, normalization="psd")/Nx*4.
        Tobs = 1.0/np.sum(np.median(freq_window[1:]-freq_window[:-1])*power_window)
        p = (LombScargle(x, y).power(freq, normalization='psd')/Nx*4.)*Tobs
    if return_val == "window":
        nu = 0.5*(freqMin+freqMax)
        freq_window = np.arange(freqMin, freqMax, dfreq/10)
        power_window = LombScargle(x, np.sin(2*np.pi*nu*x)).power(freq_window, normalization="psd")/Nx*4.
        freq, p = freq_window-nu, power_window

    return freq, p


def arg_closest_node(node, roads):
    assert len(roads.shape)==1, "roads should have dim=1."
    roads=roads.reshape((-1,1))
    assert len(node.shape)==1, "node should have dim=1."
    idx=np.argmin((roads-node)**2., axis=0)
    return idx


def quantile(x, q, weights=None):

    """
    Compute sample quantiles with support for weighted samples.
    Modified based on 'corner'.

    ----------
    Input:
    x: array-like[nsamples,]
        The samples.
    q: array-like[nquantiles,]
        The list of quantiles to compute. These should all be in the range
        '[0, 1]'.

    ----------
    Optional input:
    weights : Optional[array-like[nsamples,]]
        An optional weight corresponding to each sample.

    ----------
    Output:
    quantiles: array-like[nquantiles,]
        The sample quantiles computed at 'q'.
    
    """

    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if weights is None:
        return np.percentile(x, 100.0 * q)
    else:
        weights = np.atleast_1d(weights)
        idx = np.argsort(x, axis=0)

        res = []
        for i in range(x.shape[1]):
            sw = weights[idx[:,i]]
            cdf = np.cumsum(sw)[:-1]
            cdf /= cdf[-1]
            cdf = np.append(0, cdf)
            res.append(np.interp(q, cdf, x[idx[:,i],i]))
        return np.array(res).T