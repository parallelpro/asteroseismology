#!/usr/bin/env/ python
# coding: utf-8


import numpy as np
from asteroseismology.tools.series import smoothWrapper, lorentzian, gaussian, c_correlate
from asteroseismology.tools.plot import echelle
from asteroseismology.peakbagging.modefit import modefitWrapper, h1testWrapper
from asteroseismology.globe import sep
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import os


__all__ = ["autoradialGuess", "autoradialFit", "autoradialSummarize"]

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
	freqc[freqc > 2.0*dnu] += -2.0*dnu
	freqc[freqc < 0.0] += 2.0*dnu

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

	mode_freq = np.array(peaks)
	l0_rfreq = (mode_freq-offset) % dnu
	l0_freq = (mode_freq-offset) - ((mode_freq-offset) % dnu) + dnu/2.0 + offset

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
	 frequencyGuessFile: str, fittype: str="ParallelTempering", ifmodefit: bool=True, 
	 ifh1test: bool=False, pthreads: int=1):
	'''
	Automatic peakbagging for radial modes in [numax-5*dnu, numax+5*dnu]

	Input:
	freq: frequency in muHz.
	power: background divided power spectrum (so now is s/b instead).
	dnu: in unit of muHz.
	numax: in unit of muHz.
	filepath: file path to store outputs.
	frequencyGuessFile: input file which stores guessed resules.
	fittype: one of ["ParallelTempering", "Ensemble", "LeastSquare"].
	ifmodefit: if fit modes.
	ifh1test: if perform h1 test.
	pthreads: the number of threads to use in parallel computing. 

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
	table = np.loadtxt(frequencyGuessFile, delimiter=",", ndmin=2)
	if len(table) != 0: 
		index_ifpkbg = table[:,1] == 1
		table = table[index_ifpkbg]
		group_all = np.unique(table[:,0])
		ngroups = len(group_all)

		inclination, fnyq = 0.0, 283.2 # only for radial modes, LC kepler data

		for i in range(ngroups): #range(1)
			group = group_all[i]
			tindex = table[:,0] == group
			ttable = table[tindex,:]
			mode_freq, mode_l = ttable[:,2], np.zeros(len(ttable), dtype=int)
			tfilepath = filepath + "{:0.0f}".format(group) + sep
			if not os.path.exists(tfilepath): os.mkdir(tfilepath)

			# be careful! this program designs to fit one radial mode at a time, so each group
			# should only contain one mode.
			if ifmodefit: modefitWrapper(dnu, inclination, fnyq, mode_freq, mode_l, freq, power, powers, tfilepath,
				fittype=fittype, pthreads=pthreads)
			if ifh1test: h1testWrapper(dnu, fnyq, mode_freq, mode_l, freq, power, powers, tfilepath, pthreads=pthreads)
	else:
		print("Void guessed frequency input.")
	return

def autoradialSummarize(frequencyGuessFile: str, fittype: str="ParallelTempering"):
	'''
	Summarize fitted mode parameters from the function autoradialFit.

	Input:
	frequencyGuessFile: input file which stores guessed resules.
	fittype: one of ["ParallelTempering", "Ensemble", "LeastSquare"].

	Output:
	A summary csv file located in the same directory as the frequencyGuessFile.

	'''

	# check
	if not os.path.exists(frequencyGuessFile):
		raise ValueError("frequencyGuessFile does not exist.")
	if not fittype in ["ParallelTempering", "Ensemble", "LeastSquare"]:
		raise ValueError("fittype should be one of ['ParallelTempering', 'Ensemble', 'LeastSquare']")

	filepath = sep.join(frequencyGuessFile.split(sep)[:-1]) + sep

	# read in table and cluster in group
	table = np.loadtxt(frequencyGuessFile, delimiter=",", ndmin=2)
	if len(table) != 0: 
		index_ifpkbg = table[:,1] == 1
		table = table[index_ifpkbg]
		group_all = np.unique(table[:,0])
		ngroups = len(group_all)

		if ngroups != 0: 
			# create lists to store results
			if fittype == "ParallelTempering":
				keys = ["ngroup", "PTamp", "PTamp_lc", "PTamp_uc", "PTlw", "PTlw_lc", "PTlw_uc", "PTfc", "PTfc_lc", "PTfc_uc", "PTlnK", "PTlnK_err"]
				fmt = ["%d", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f"]
			if fittype == "Ensemble":
				keys = ["ngroup", "ESamp", "ESamp_lc", "ESamp_uc", "ESlw", "ESlw_lc", "ESlw_uc", "ESfc", "ESfc_lc", "ESfc_uc"]
				fmt = ["%d", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f"]
			if fittype == "LeastSquare":
				keys = ["ngroup", "LSamp", "LSlw", "LSfc"]
				fmt = ["%d", "%10.4f", "%10.4f", "%10.4f"]
			data = [[] for i in range(len(keys))]

			# store pkbg results
			for i in range(ngroups):
				group = group_all[i]
				tfilepath = filepath + str(int(group)) + sep
				# tindex = table[:,0] == group
				# ttable = table[tindex,:]		

				if fittype == "ParallelTempering":
					ttable = np.loadtxt(tfilepath+"PTsummary.txt", delimiter=",", ndmin=2)
					for j in range(9):
						data[j+1].append(ttable[int(j/3), j%3])
					ttable1 = np.loadtxt(tfilepath+"PTevidence.txt", delimiter=",")
					ttable2 = np.loadtxt(tfilepath+"evidence_h1_0.txt", delimiter=",")
					data[10].append(ttable1[0]-ttable2[0])
					data[11].append((ttable1[1]**2.0+ttable2[1]**2.0)**0.5)
					data[0].append(group)

				if fittype == "Ensemble":
					ttable = np.loadtxt(tfilepath+"ESsummary.txt", delimiter=",", ndmin=2)
					for j in range(9):
						data[j+1].append(ttable[int(j/3), j%3])
					data[0].append(group)

				if fittype == "LeastSquare":
					ttable = np.loadtxt(tfilepath+"LSsummary.txt", delimiter=",")
					for j in range(3):
						data[j+1].append(ttable[j])
					data[0].append(group)


			# check if previous summary file exists
			ifexists = False
			if os.path.exists(filepath+"frequencySummary.csv"):
				olddata = np.loadtxt(filepath+"frequencySummary.csv", delimiter=",", ndmin=2)
				if len(olddata) != 0: ifexists = True

			if ifexists:
				# open old summary file and extract oldkeys and olddata
				# exclude the last column - "ifpublish"
				f = open(filepath+"frequencySummary.csv", "r")
				oldkeys = f.readline().replace(" ","").replace("#","").replace("\n","")
				oldkeys = np.array(oldkeys.split(","))#[:-1]
				f.close()
				olddata = np.loadtxt(filepath+"frequencySummary.csv", delimiter=",", ndmin=2)#[:,:-1] # exclude ifpublish
				# print("olddata", olddata)
				# print("oldkeys", oldkeys)
				add_keys = np.array(oldkeys[:-1])[~np.isin(oldkeys[:-1], keys)]
				# print("add_keys", add_keys)
				add_group = np.array(olddata[:,0])[~np.isin(olddata[:,0], data[0])]
				# print("add_group", add_group)
				newdata = np.array(data).T
				n_add_keys, n_add_group = len(add_keys), len(add_group)
				# print("N,N", n_add_keys, n_add_group)
				# assign -999.0 to new entries
				if n_add_keys != 0:
					newdata = np.concatenate([newdata, np.zeros((np.shape(newdata)[0], n_add_keys))-999.0], axis=1)
				if n_add_group != 0:
					newdata = np.concatenate([newdata, np.zeros((n_add_group, np.shape(newdata)[1]))-999.0], axis=0)
					newdata[-n_add_group:,0] = add_group

				# concatenate "ifpublish"
				# newkeys, newdata
				for j in range(n_add_keys):
					fmt.append("%10.4f")
				fmt.append("%d")
				n_add_keys += 1
				add_keys = np.append(add_keys, "ifpublish")
				newkeys = np.concatenate([keys, add_keys])
				newdata = np.concatenate([newdata, np.zeros((np.shape(newdata)[0], 1))+1], axis=1)


				nrows, ncols = np.shape(newdata)
				for j in range(0, nrows):
					for k in range(ncols - n_add_keys , ncols):
						tgroup, tkey = newdata[j,0], newkeys[k]
						index1 = np.where(olddata[:,0] == tgroup)[0]
						index2 = np.where(oldkeys == tkey)[0]
						if len(index1) != 0 and len(index2) != 0:
							newdata[j,k] = olddata[index1[0], index2[0]]

				for j in range(nrows - n_add_group , nrows):
					for k in range(0, ncols - n_add_keys):
						tgroup, tkey = newdata[j,0], newkeys[k]
						index1 = np.where(olddata[:,0] == tgroup)[0]
						index2 = np.where(oldkeys == tkey)[0]
						if len(index1) != 0 and len(index2) != 0:
							newdata[j,k] = olddata[index1[0], index2[0]]
				
			else:
				# not exists - create a new file
				newkeys, newdata = np.array(keys), np.array(data).T
				newkeys = np.append(keys, "ifpublish")
				fmt = np.append(fmt, "%d").tolist()
				newdata = np.concatenate([newdata, np.zeros((np.shape(newdata)[0], 1))+1], axis=1)
			
			# save table
			np.savetxt(filepath+"frequencySummary.csv", newdata, delimiter=",", fmt=fmt, header=", ".join(newkeys))

	return