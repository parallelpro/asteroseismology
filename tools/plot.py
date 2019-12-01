#!/usr/bin/env/ python
# coding: utf-8


import numpy as np

__all__ = ["echelle"]

def echelle(x: np.array, y: np.array, period: float, 
	fmin: float=None, fmax: float=None, echelletype: str="single", offset: float=0.0):
	'''
	Generate a z-map for echelle plotting.

	Input:

	x: np.array
		the frequency.

	y: np.array
		the power spectrum.

	period: float
		the large separation.

	lowc: float
		the lower boundary frequency, in the same unit of x.

	highc: float
		the higher boundary frequency, in the same unit of x.


	Optional input:

	echelletype: str, default: "single"
		single or replicated.

	offset: float, default: 0
		the horizontal shift in the same unit of x.


	Output:

	x, y: 
		two 1-d arrays.
	z: 
		a 2-d array.

	Exemplary call:

	import matplotlib.pyplot as plt
	fig = plt.figure(figsize=(6,8))
	ax1 = fig.add_subplot(111)
	echx, echy, echz = echelle(tfreq,tpowers_o,dnu,numax-9.0*dnu,numax+9.0*dnu,echelletype="single",offset=offset)
	levels = np.linspace(np.min(echz),np.max(echz),500)
	ax1.contourf(echx,echy,echz,cmap="gray_r",levels=levels)
	ax1.axis([np.min(echx),np.max(echx),np.min(echy),np.max(echy)])
	if offset > 0.0:
		ax1.set_xlabel("(Frequency - "+str("{0:.2f}").format(offset)+ ") mod "+str("{0:.2f}").format(dnu) + " ($\mu$Hz)")
	if offset < 0.0:
		ax1.set_xlabel("(Frequency + "+str("{0:.2f}").format(np.abs(offset))+ ") mod "+str("{0:.2f}").format(dnu) + " ($\mu$Hz)")
	if offset == 0.0:
		ax1.set_xlabel("Frequency mod "+str("{0:.2f}").format(dnu) + " ($\mu$Hz)")
	plt.savefig("echelle.png")

	'''

	if not echelletype in ["single", "replicated"]:
		raise ValueError("echelletype is on of 'single', 'replicated'.")

	if len(x) != len(y): 
		raise ValueError("x and y must have equal size.")	

	if fmin is None: fmin=0.
	if fmax is None: fmax=np.nanmax(x)

	fmin = fmin - offset
	fmax = fmax - offset
	x = x - offset

	if fmin <= 0.0:
		fmin = 0.0
	else:
		fmin = fmin - (fmin % period)

	# first interpolate
	samplinginterval = np.median(x[1:-1] - x[0:-2]) * 0.1
	xp = np.arange(fmin,fmax+period,samplinginterval)
	yp = np.interp(xp, x, y)

	n_stack = int((fmax-fmin)/period)
	n_element = int(period/samplinginterval)
	#print(n_stack,n_element,len())

	morerow = 2
	arr = np.arange(1,n_stack) * period # + period/2.0
	arr2 = np.array([arr,arr])
	yn = np.reshape(arr2,len(arr)*2,order="F")
	yn = np.insert(yn,0,0.0)
	yn = np.append(yn,n_stack*period) + fmin #+ offset

	if echelletype == "single":
		xn = np.arange(1,n_element+1)/n_element * period
		z = np.zeros([n_stack*morerow,n_element])
		for i in range(n_stack):
			for j in range(i*morerow,(i+1)*morerow):
				z[j,:] = yp[n_element*(i):n_element*(i+1)]
	if echelletype == "replicated":
		xn = np.arange(1,2*n_element+1)/n_element * period
		z = np.zeros([n_stack*morerow,2*n_element])
		for i in range(n_stack):
			for j in range(i*morerow,(i+1)*morerow):
				z[j,:] = np.concatenate([yp[n_element*(i):n_element*(i+1)],yp[n_element*(i+1):n_element*(i+2)]])

	return xn, yn, z