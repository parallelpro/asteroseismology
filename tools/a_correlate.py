#!/usr/bin/env/ ipython
# coding: utf-8


import numpy as np

def a_correlate(x: np.array, y: np.array): 
	'''
	Generate autocorrelation coefficient as a function of lag.

	Input:
	x: Independent variable of the time series.
	y: Dependent variable of the time series.

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
	lag = np.arange(0,int(len(series)/2))
	
	rho = np.zeros(len(lag))
	for i in range(0,len(lag)):
		h = lag[i]
		N = len(series)
		rho[i] = np.sum((series[h:N]-np.mean(series))*(series[0:N-h]-np.mean(series)))/np.sum((series-np.mean(series))**2)

	lagn = lag * samplinginterval
	rhon = rho

	return lagn, rhon