#!/usr/bin/env/ ipython
# coding: utf-8


import numpy as np

def echelle(x: np.array, y: np.array, period: float, lowc: float, highc: float, echelletype: str="single"):
	'''
	Generate a z-map for echelle plotting.

	Input:
	x: frequency.
	y: power spectrum.
	period: delta_nu.
	lowc: lower boundary frequency.
	highc: higher boundary frequency.
	type: single or replicated

	Output:
	x: 1-d array.
	y: 1-d array.
	z: 2-d array.

	'''

	if not echelletype in ["single", "replicated"]:
		raise ValueError("echelletype is on of 'single', 'replicated'.")

	if len(x) != len(y): 
		raise ValueError("x and y must have equal size.")	

	if lowc <= 0.0:
		lowc = 0.0
	else:
		lowc = lowc - (lowc % period)

	# trim data
	index = np.intersect1d(np.where(x>=lowc)[0],np.where(x<=highc)[0])
	trimx = x[index]
	trimy = y[index]

	# first interpolate
	samplinginterval = np.median(trimx[1:-1] - trimx[0:-2])
	xp = np.arange(lowc,highc+period,samplinginterval)
	yp = np.interp(xp, x, y)

	n_stack = int((highc-lowc)/period)
	n_element = int(period/samplinginterval)
	#print(n_stack,n_element,len())

	morerow = 2
	arr = np.arange(1,n_stack) * period # + period/2.0
	arr2 = np.array([arr,arr])
	yn = np.reshape(arr2,len(arr)*2,order="F")
	yn = np.insert(yn,0,0.0)
	yn = np.append(yn,n_stack*period) + lowc

	if type == "single":
		xn = np.arange(1,n_element+1)/n_element * period
		z = np.zeros([n_stack*morerow,n_element])
		for i in range(n_stack):
			for j in range(i*morerow,(i+1)*morerow):
				z[j,:] = yp[n_element*(i):n_element*(i+1)]
	if type == "double":
		xn = np.arange(1,2*n_element+1)/n_element * period
		z = np.zeros([n_stack*morerow,2*n_element])
		for i in range(n_stack):
			for j in range(i*morerow,(i+1)*morerow):
				z[j,:] = np.concatenate([yp[n_element*(i):n_element*(i+1)],yp[n_element*(i+1):n_element*(i+2)]])

	return xn, yn, z