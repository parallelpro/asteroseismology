#!/usr/bin/env/ ipython
# coding: utf-8


import numpy as np

def SmoothWrapper(x: np.array, y: np.array, period: float, windowtype: str, samplinginterval: float = -999.0):
	'''
	Wrapping a sliding-average smooth function.

	Input:
	x: Independent variable of the time series.
	y: Dependent variable of the time series.
	period: the period/width of the sliding window.
	windowtype: flat/hanning/hamming/bartlett/blackman
	samplinginterveal: the time between adjacent sampling points.

	Output:
	yf: smoothed time series with the exact same points as x.

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

