#!/usr/bin/env/ ipython
# coding: utf-8


import numpy as np
from asteroseismology.tools.series import smoothWrapper, lorentzian, gaussian, c_correlate
from asteroseismology.tools.plot import echelle
from asteroseismology.peakbagging.modefit import modefitWrapper, h1testWrapper
from scipy.signal import find_peaks

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()
import os


__all__ = ["autoradialGuess", "autoradialFit"]

def autoradialGuess(freq: np.array, power: np.array, dnu: float, numax: float, filepath: str,
	 lowerbound: float=5.0, upperbound: float=5.0, ifeps: bool =False, teff: float=5777.0):
	'''
	Automatic initial guess for radial modes in [numax-5*dnu, numax+5*dnu]

	Input:
	freq: frequency in muHz.
	power: background divided power spectrum (so now is s/b instead).
	dnu: in unit of muHz.
	numax: in unit of muHz.
	filepath: file path to store outputs.
	lowerbound: the lower boundary of the power spectrum slice, in unit of dnu.
	upperbound: the upper boundary of the power spectrum slice, in unit of dnu.

	Output:
	Files containing necessary outputs.
	1. analysis plot autoradialGuess.png
	2. table frequencyGuess.csv

	'''

	# check
	if not os.path.exists(filepath):
		raise ValueError("Filepath does not exist.")

	if len(freq) != len(power):
		raise ValueError("len(freq) != len(power)")

	if lowerbound <= 0:
		raise ValueError("lowerbound <= 0")

	if upperbound <= 0:
		raise ValueError("upperbound <= 0")

	# set up plot
	fig = plt.figure(figsize=(10,10))

	# smooth the power spectrum
	period, samplinginterval = dnu/50.0, np.median(freq[1:-1] - freq[0:-2])
	powers = smoothWrapper(freq, power, period, "bartlett", samplinginterval)

	# slice the power spectrum
	index = np.all(np.array([freq >= numax - lowerbound*dnu, freq <= numax + upperbound*dnu]), axis=0)
	freq, power, powers = freq[index], power[index], powers[index]

	# collapse (c) the power spectrum
	freqc = np.arange(0.0, dnu, samplinginterval*0.1)
	powerc, numberc = np.zeros(len(freqc)), np.zeros(len(freqc))
	nl, nu = int(freq.min()/dnu), int(freq.max()/dnu)+1
	for i in range(nl, nu):
		tfreq = freqc + i*dnu
		tpowers = np.interp(tfreq, freq, powers)
		index = np.all(np.array([tfreq >= numax - lowerbound*dnu, tfreq <= numax + upperbound*dnu]), axis=0)
		tpowers[~index] = 0.0
		tnumber = np.array(index, dtype=int)
		powerc += tpowers
		numberc += tnumber 
	powerc /= numberc
	powerc /= powerc.max()
	freqc = np.concatenate((freqc, freqc+dnu))
	powerc = np.concatenate((powerc, powerc))

	### I use two methods to enhance the radial modes.
	### 1. calibrate epsilon as a function of temperature from observation, use it as a guide.
	###    this may not work every time. so a switcher ifeps is provided. 
	### 2. construct a template series containing two peaks separated by the small separation,
	###    then cross correlate it with the collapsed series.


	# now consider the first method
	eps_predict = 10.0**(0.32215*np.log(dnu) - 0.51461*np.log(teff) + 3.83118) # fit from White+2011
	if eps_predict > 1: eps_predict += -1.0
	if eps_predict < 0: eps_predict += 1.0
	prob_eps = np.zeros(len(freqc))
	for i in range(-3,4):
		prob_eps += gaussian(freqc, (eps_predict+i)*dnu, 0.3*dnu, 1.0)
	prob_eps /= prob_eps.max()

	powerc_enhance = powerc * prob_eps

	# now consider the second method
	dnu02 = 0.122*dnu + 0.05 # Bedding+2010, but only fit for dnu 8 - 20
	width, xinit = 0.01*dnu, 0.7*dnu
	xtemplate = freqc
	ytemplate = lorentzian(xtemplate, xinit, width, 1.0)
	ytemplate += lorentzian(xtemplate, xinit+dnu, width, 1.0)
	ytemplate += lorentzian(xtemplate, xinit-dnu02, width, 0.6)
	ytemplate += lorentzian(xtemplate, xinit-dnu02+dnu, width, 0.6)
	ytemplate /= ytemplate.max()
	if ifeps:
		lag, rho = c_correlate(freqc, powerc_enhance, ytemplate)
	else:
		lag, rho = c_correlate(freqc, powerc, ytemplate)

	# find the highest point in cross-correlation diagram and shift its location
	index_hp = np.where(rho == np.max(rho))[0][0]
	eps_cross = xinit/dnu-lag[index_hp]/dnu
	if eps_cross > 1: eps_cross += -1.0
	if eps_cross < 0: eps_cross += 1.0
	if eps_cross <= 0.2:
		offset = -0.2*dnu
	elif eps_cross >= 0.8:
		offset = 0.2*dnu
	else:
		offset = 0.0
	freqc += -offset
	freqc[freqc > 2.0*dnu] += -dnu
	freqc[freqc < 0.0] += dnu

	index = np.argsort(freqc)
	freqc, powerc, xtemplate = freqc[index], powerc[index], xtemplate[index],
	ytemplate, prob_eps, powerc_enhance = ytemplate[index], prob_eps[index], powerc_enhance[index]


	# slice power spectrum into blocks
	n_low, n_high = int((freq.min()-offset)/dnu), int((freq.max()-offset)/dnu)
	n_blocks = np.arange(n_low, n_high+1, 1)
	peaks = []
	for i, n_block in enumerate(n_blocks):
		freq_low, freq_high = offset+n_block*dnu, offset+(n_block+1)*dnu
		radial_freq_low, radial_freq_high = freq_low-offset+eps_cross*dnu-0.6*dnu02, freq_low-offset+eps_cross*dnu+1.0*dnu02
		#print(freq_low, freq_high, radial_freq_low, radial_freq_high, off, dnu02)
		index_norder = np.all(np.array([freq>=freq_low, freq<=freq_high]), axis=0)
		index_radial = np.all(np.array([freq>=radial_freq_low, freq<=radial_freq_high]), axis=0)

		tfreq, tpowers = freq[index_radial], powers[index_radial]
		# find the highest peak in this range as a guess for the radial mode
		index_peaks, properties = find_peaks(tpowers, height=(2.0,None), distance=int(dnu02/samplinginterval), prominence=(1.0,None))
		if len(index_peaks) != 0:
			index_maxpeak = index_peaks[properties["peak_heights"] == properties["peak_heights"].max()]
			peaks.append(tfreq[index_maxpeak[0]])

		### visulization (right)
		ax = fig.add_subplot(len(n_blocks),2,2*len(n_blocks)-2*i)
		ax.plot(freq[index_norder], power[index_norder], color="gray")
		ax.plot(freq[index_norder], powers[index_norder], color="C0")
		ax.plot(tfreq[index_peaks], tpowers[index_peaks], "x", color="C1")
		if len(index_peaks) != 0:
			ax.plot(tfreq[index_maxpeak], tpowers[index_maxpeak], "x", color="C3")
		ax.axvline(radial_freq_low, linestyle="--", color="C2")
		ax.axvline(radial_freq_high, linestyle="--", color="C2")
		ax.axis([freq_low, freq_high, power[index_norder].min(), power[index_norder].max()])
		### end of visulization


	### visulization (left) - plot echelle and collapsed echelle to locate peak
	ax1 = plt.subplot2grid((5,2), (0,0), rowspan=2)
	echx, echy, echz = echelle(freq, powers, dnu, freq.min(), freq.max(), 
		echelletype="replicated", offset=offset)
	levels = np.linspace(np.min(echz), np.max(echz), 500)
	ax1.contourf(echx, echy, echz, cmap="gray_r", levels=levels)
	ax1.axis([np.min(echx), np.max(echx), np.min(echy), np.max(echy)])
	ax1.axvline(dnu, color="C0")
	ax1.axvline(eps_cross*dnu-offset-0.6*dnu02, linestyle="--", color="C1")
	ax1.axvline(eps_cross*dnu-offset+1.0*dnu02, linestyle="--", color="C1")
	ax1.axvline(eps_cross*dnu-offset+dnu-0.6*dnu02, linestyle="--", color="C1")
	ax1.axvline(eps_cross*dnu-offset+dnu+1.0*dnu02, linestyle="--", color="C1")
	ax1.set_ylabel("Frequency [muHz]")

	l0_freq = np.array(peaks) - offset
	l0_rfreq =  l0_freq % dnu
	index = np.where(l0_rfreq < 0)[0]
	l0_rfreq[index] = l0_rfreq[index] + dnu
	index = np.where(l0_rfreq > dnu)[0]
	l0_rfreq[index] = l0_rfreq[index] - dnu
	l0_freq = l0_freq - ((l0_freq - offset) % dnu) + dnu/2.0
	ax1.plot(l0_rfreq, l0_freq, "x", color="C3")
	ax1.plot(l0_rfreq+dnu, l0_freq-dnu, "x", color="C3")

	ax2 = fig.add_subplot(5,2,5)
	ax2.plot(freqc, powerc, color='black')
	ax2.plot(freqc, prob_eps, color='C2')
	ax2.axis([freqc.min(), freqc.max(), powerc.min(), powerc.max()])

	ax3 = fig.add_subplot(5,2,7)
	ax3.plot(freqc, powerc_enhance, color='black')
	ax3.plot(freqc, ytemplate, color='C2')
	ax3.axis([freqc.min(), freqc.max(), powerc.min(), powerc.max()])
	if offset > 0.0:
		ax3.set_xlabel("(Frequency - "+str("{0:.2f}").format(offset)+ ") mod "+str("{0:.2f}").format(dnu) + " [muHz]")
	if offset < 0.0:
		ax3.set_xlabel("(Frequency + "+str("{0:.2f}").format(np.abs(offset))+ ") mod "+str("{0:.2f}").format(dnu) + " [muHz]")
	if offset == 0.0:
		ax3.set_xlabel("Frequency mod "+str("{0:.2f}").format(dnu) + " [muHz]")

	ax4 = fig.add_subplot(6,2,11)
	ax4.plot(lag, rho)
	ax4.plot(lag[index_hp], rho[index_hp], "x", color="C1")
	ax4.axis([-dnu, dnu, rho.min(), rho.max()+0.1])
	ax4.set_xlabel("Frequency lag [muHz]")
	### end of visulizaiton


	# save plot
	plt.savefig(filepath+"autoradialGuess.png")
	plt.close()

	# save a table
	table = np.array([np.arange(len(peaks)), np.zeros(len(peaks))+1, peaks]).T
	np.savetxt(filepath+"frequencyGuess.csv", table, delimiter=",", fmt=("%d","%d","%10.4f"), 
		header="ngroup, ifpkbg, freqGuess")

	return

def autoradialFit(freq: np.array, power: np.array, dnu: float, numax: float, filepath: str,
	 frequencyGuessFile: str):
	'''
	Automatic peakbagging for radial modes in [numax-5*dnu, numax+5*dnu]

	Input:
	freq: frequency in muHz.
	power: background divided power spectrum (so now is s/b instead).
	dnu: in unit of muHz.
	numax: in unit of muHz.
	filepath: file path to store outputs.
	frequencyGuessFile: input file which stores guessed resules.

	Output:
	Files containing necessary outputs.

	'''

	# check
	if not os.path.exists(filepath):
		raise ValueError("Filepath does not exist.")

	if len(freq) != len(power):
		raise ValueError("len(freq) != len(power)")

	# smooth the power spectrum
	period, samplinginterval = dnu/50.0, np.median(freq[1:-1] - freq[0:-2])
	powers = smoothWrapper(freq, power, period, "bartlett", samplinginterval)

	# read in table and cluster in group
	table = np.loadtxt(frequencyGuessFile, delimiter=",")
	index_ifpkbg = table[:,1] == 1
	table = table[index_ifpkbg]
	group_all = np.unique(table[:,0])
	ngroups = len(group_all)

	inclination, fnyq = 0.0, 283.2

	for i in range(1):#range(ngroups):
		group = group_all[i]
		tindex = table[:,0] == group
		ttable = table[tindex,:]
		mode_freq, mode_l = ttable[:,2], np.zeros(len(ttable), dtype=int)
		tfilepath = filepath + "{:0.0f}".format(group) + "/"
		if not os.path.exists(tfilepath): os.mkdir(tfilepath)

		# be careful! this program designs to fit one radial mode at a time, so each group
		# should only contain one mode.
		modefitWrapper(dnu, inclination, fnyq, mode_freq, mode_l, freq, power, powers, tfilepath)
		h1testWrapper(dnu, fnyq, mode_freq, mode_l, freq, power, powers, tfilepath)

	return
