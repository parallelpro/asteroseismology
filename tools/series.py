#!/usr/bin/env/ python
# coding: utf-8


import numpy as np
from astropy.stats import LombScargle


__all__ = ["a_correlate", "c_correlate", "smoothWrapper", "gaussian", 
		"lorentzian", "medianFilter", "psd"]

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

def psd(time_d, f):
	"""
	Calculate the power spectrum density for a discrete time series.
	https://en.wikipedia.org/wiki/Spectral_density

	Input:
	time_d: time in day
	f: relative flux


	Output:
	freq: frequency in muHz
	psd: psd in [flux]^2/muHz

	"""

	time_d, f = time_d[index], f[index]
	# f=(f-1)*1e6

	Ntime = len(time_d)
	time_s = time_d*24.0*3600.0
	dt_d = np.median(time_d[1:]-time_d[:-1]) # day
	dt_s = dt_d*24.0*3600.0
	fs_hz = 1.0/dt_s
	fs_muhz = fs_hz * 1e6
	tobs_s = dt_s*len(time_s)

	fnyq_muhz = 0.5*fs_muhz

	Nfreq = int(len(time_d)/2.0+1.0)
	freq_hz = np.linspace(fnyq_muhz/1e6/Nfreq, fnyq_muhz/1e6, Nfreq)
	freq_muhz = freq_hz*1e6
	power = LombScargle(time_s, f).power(freq_hz,normalization='psd')
	psd = power*2*dt_s*1e-6

	return freq_muhz, psd