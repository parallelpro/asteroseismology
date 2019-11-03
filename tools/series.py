#!/usr/bin/env/ python
# coding: utf-8


import numpy as np
from astropy.timeseries import LombScargle


__all__ = ["auto_correlate", "a_correlate", "c_correlate", "smoothWrapper", "gaussian", 
		"lorentzian", "medianFilter", "psd", "arg_closest_node"]

def auto_correlate(x, y, need_interpolate = False, samplinginterval = None): 

    '''
    Generate autocorrelation coefficient as a function of lag.

    Input:

        x: np.array, the independent variable of the time series.
        y: np.array, the dependent variable of the time series.
        need_interpolate: True or False. True is for unevenly spaced time series.
        samplinginterval: float. Default value: the median of intervals.

    Output:

        lagn: time lag.
        rhon: autocorrelation coeffecient.

    '''

    if len(x) != len(y): 
        raise ValueError("x and y must have equal size.")

    if need_interpolate is not None:
        if samplinginterval is None:
            samplinginterval = np.median(x[1:-1] - x[0:-2])
        xp = np.arange(np.min(x),np.max(x),samplinginterval)
        yp = np.interp(xp, x, y)
        x = xp
        y = yp

    new_y = y - np.mean(y)
    aco = np.correlate(new_y, new_y, mode='same')

    N = len(aco)
    lagn = x[int(N/2):N] - x[int(N/2)]
    rhon = aco[int(N/2):N] / np.var(y)
    rhon = rhon / np.max(rhon)

    return lagn, rhon

def a_correlate(x: np.array, y: np.array): 
	'''
	Generate autocorrelation coefficient as a function of lag.
	See The Analysis Of Time Series by Chris Chatfield, 6th edition, equation 2.6

	Input:
	x: the independent variable of the time series.
	y: the dependent variable of the time series.

	Output:
	lagn: time lag.
	rhon: autocorrelation coeffecient.

	'''

	if len(x) != len(y): 
		raise ValueError("x and y must have equal size.")
		
	# first interpolate
	samplinginterval = np.median(x[1:-1] - x[0:-2])
	xp = np.arange(np.min(x),np.max(x),samplinginterval)
	yp = np.interp(xp, x, y)

	series = yp
	N = len(series)
	lag = np.arange(0,int(N/2))
	rho = np.zeros(len(lag))

	for i in range(0,len(lag)):
		h = lag[i]
		rho[i] = np.sum((series[h:N]-np.mean(series))*(series[0:N-h]-np.mean(series)))/np.sum((series-np.mean(series))**2)

	lagn = lag * samplinginterval
	rhon = rho

	return lagn, rhon

def c_correlate(x: np.array, y1: np.array, y2: np.array): 
	'''
	Generate sample cross-correlation function as a function of lag.
	See The Analysis Of Time Series by Chris Chatfield, 6th edition, equation 8.5

	Input:
	x: the independent variable of the time series.
	y1: the dependent variable of the time series 1.
	y2: the dependent variable of the time series 2.

	Output:
	lagn: time lag / displacement of y1 while y2 remains still.
	rhon: crosscorrelation coeffecient.

	'''

	if len(x) != len(y1): 
		raise ValueError("x and y1 must have equal size.")
	if len(x) != len(y2): 
		raise ValueError("x and y2 must have equal size.")
		
	# first interpolate
	samplinginterval= np.median(x[1:-1] - x[0:-2])
	xp = np.arange(np.min(x),np.max(x),samplinginterval)
	yp1 = np.interp(xp, x, y1)
	yp2 = np.interp(xp, x, y2)

	# change here!
	N = len(x)-1
	lag = np.arange(-(N-1),N-1,1)
	cross = np.zeros(len(lag))

	for i in range(0,len(lag)):
		h = lag[i]
		if h >= 0:
			cross[i] = np.sum((yp1[0:N-h]-yp1.mean())*(yp2[h:N]-yp2.mean()))/N
		else:
			cross[i] = np.sum((yp1[-h:N]-yp1.mean())*(yp2[0:N+h]-yp2.mean()))/N
		
	lagn = lag * samplinginterval
	rhon = cross/(yp1.std()*yp2.std())

	return lagn, rhon


def gaussian(x: np.array, mu: float, sigma: float, height: float):
	'''
	Return the value of gaussian given parameters.

	Input:
	x: the independent variable of the time series.
	mu
	sigma
	height

	Output:
	y: the dependent variable of the time series.

	'''
	return height * np.exp(-(x-mu)**2.0/(2*sigma**2.0))

def lorentzian(x: np.array, mu: float, gamma: float, height: float):
	'''
	Return the value of gaussian given parameters.

	Input:
	x: the independent variable of the time series.
	mu
	gamma
	height

	Output:
	y: the dependent variable of the time series.

	'''
	return height / (1 + (x-mu)**2.0/gamma**2.0)

def smoothWrapper(x: np.array, y: np.array, period: float, windowtype: str, samplinginterval: float = -999.0):
	'''
	Wrapping a sliding-average smooth function.

	Input:
	x: the independent variable of the time series.
	y: the dependent variable of the time series.
	period: the period/width of the sliding window.
	windowtype: flat/hanning/hamming/bartlett/blackman
	samplinginterveal: the time between adjacent sampling points.

	Output:
	yf: the smoothed time series with the exact same points as x.

	'''

	if len(x) != len(y): 
		raise ValueError("x and y must have equal size.")
		
	if samplinginterval <= 0.0: samplinginterval = np.median(x[1:-1] - x[0:-2])

	xp = np.arange(np.min(x),np.max(x),samplinginterval)
	yp = np.interp(xp, x, y)
	window_len = int(period/samplinginterval)
	if window_len % 2 == 0:
		window_len = window_len + 1
	ys = smooth(yp, window_len, window = windowtype)
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
	if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
		raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

	s = x #np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
	if window == "flat":
		w = np.ones(window_len,"d")
	else:
		w = eval("np."+window+"(window_len)") 
	
	y = np.convolve(w/w.sum(),s,mode="same")
	return y

def medianFilter(x, y, period, yerr=None):
	if yerr==None: iferror=False
	binsize = np.median(x[1:-1]-x[0:-2])
	kernelsize = int(period/binsize)
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

    if not (return_val in ["psd_old", "periodogram", "power", "amplitude", "psd_new"]):
        raise ValueError("return_val should be one of ['psd_old', 'periodogram', 'power', 'amplitude', 'psd_new'] ")

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
        p = LombScargle(x, y).power(freq, normalization='psd')*dx
    if return_val == "periodogram":
        p = LombScargle(x, y).power(freq, normalization='psd')
    if return_val == "power":
        p = LombScargle(x, y).power(freq, normalization='psd')/Nx*4.
    if return_val == "amplitude":
        p = np.sqrt(LombScargle(x, y).power(freq, normalization='psd')/Nx*4.)
    if return_val == "psd_new":
        nu = 0.5*(freqMin+freqMax)
        freq_window = np.arange(freqMin, freqMax, dfreq/10)
        power_window = LombScargle(x, np.cos(2*np.pi*nu*x)).power(freq, normalization="standard")
        Tobs = 1.0/np.sum(np.median(freq_window[1:]-freq_window[:-1])*power_window)
        p = (LombScargle(x, y).power(freq, normalization='psd')/Nx*4.)/4.*Tobs
		
    return freq, p

def arg_closest_node(node, roads):
	assert len(roads.shape)==1, "roads should have dim=1."
	roads=roads.reshape((-1,1))
	assert len(node.shape)==1, "node should have dim=1."
	idx=np.argmin((roads-node)**2., axis=0)
	return idx

