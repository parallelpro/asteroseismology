#!/usr/bin/env/ python
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


__all__ = ["manualGuess", "manualFit", "manualSummarize"]

def manualGuess(freq: np.array, power: np.array, dnu: float, numax: float, filepath: str,
	 lowerbound: float=5.0, upperbound: float=5.0, eps=-999.0):
	'''
	Initial guess for all peaks in [numax-lowerbound*dnu, numax+upperbound*dnu], 
	with a user adjustable epsilon.

	Input:
	freq: frequency in muHz.
	power: background divided power spectrum (so now is s/b instead).
	dnu: in unit of muHz.
	numax: in unit of muHz.
	filepath: file path to store outputs.
	lowerbound: the lower boundary of the power spectrum slice, in unit of dnu.
	upperbound: the upper boundary of the power spectrum slice, in unit of dnu.
	eps: different from the physical epsilon. should be amid 0 and 1 and defines
		the ridge of radial modes.

	Output:
	Files containing necessary outputs.
	1. analysis plot manualGuess.png
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

	if eps!=-999.0 and (eps<0.0 or eps>1.0):
		raise ValueError("eps should be between 0 and 1.")

	# set up plot
	fig = plt.figure(figsize=(12,10))

	# smooth the power spectrum
	period, samplinginterval = dnu/30.0, np.median(freq[1:-1] - freq[0:-2])
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
	

	### Guess epsilon:
	dnu02 = 0.122*dnu + 0.05 # Bedding+2010, but only fit for dnu 8 - 20
	dnu01 = 0.520*dnu
	if eps == -999.0:
		### construct a template series containing two peaks separated by the small separation,
		###    then cross correlate it with the collapsed series.
		# now consider the second method

		width0, width1, width2, xinit = 0.03*dnu, 0.15*dnu, 0.03*dnu, 0.35*dnu
		xtemplate = freqc
		ytemplate = lorentzian(xtemplate, xinit, width0, 1.0)
		ytemplate += lorentzian(xtemplate, xinit+dnu, width0, 1.0)
		ytemplate += lorentzian(xtemplate, xinit-dnu, width0, 1.0)
		ytemplate += lorentzian(xtemplate, xinit-dnu02, width2, 0.8)
		ytemplate += lorentzian(xtemplate, xinit-dnu02+dnu, width2, 0.8)
		ytemplate += lorentzian(xtemplate, xinit-dnu02-dnu, width2, 0.8)
		ytemplate += lorentzian(xtemplate, xinit+dnu01, width1, 0.5)
		ytemplate += lorentzian(xtemplate, xinit+dnu01+dnu, width1, 0.5)
		ytemplate += lorentzian(xtemplate, xinit+dnu01-dnu, width1, 0.5)

		ytemplate /= ytemplate.max()

		lag, rho = c_correlate(freqc, powerc, ytemplate)

		# find the highest point in cross-correlation diagram and shift its location
		index_hp = np.where(rho == np.max(rho))[0][0]
		eps_cross = xinit/dnu-lag[index_hp]/dnu
		if eps_cross > 1: eps_cross += -1.0
		if eps_cross < 0: eps_cross += 1.0
	else:
		eps_cross = eps

	if eps_cross < 0.5:
		offset = eps_cross-dnu02-0.05*dnu
	elif eps_cross >= 0.5:
		offset = -eps_cross-dnu02-0.05*dnu

	freqc += -offset
	freqc[freqc > 2.0*dnu] += -2.0*dnu
	freqc[freqc < 0.0] += 2.0*dnu

	index = np.argsort(freqc)
	freqc, powerc, xtemplate = freqc[index], powerc[index], xtemplate[index]
	ytemplate = ytemplate[index]
	powerc = smoothWrapper(freqc, powerc, 0.02*dnu, "bartlett")
	powerc = powerc/np.max(powerc)


	# assign l=0,1,2,3 region to the power spectrum
	rfreq = freq/dnu % 1.0
	lowc = [-dnu02/dnu/2.0, +0.25, -dnu02/dnu-0.05, +0.10]
	highc = [+0.10, 1.0-dnu02/dnu-0.05, -dnu02/dnu/2.0, +0.25]
	index_l = []
	for l in range(4):
		dum1 = np.all(np.array([rfreq>=eps_cross+lowc[l], rfreq<eps_cross+highc[l]]),axis=0)
		dum2 = np.all(np.array([rfreq>=eps_cross+lowc[l]-1, rfreq<eps_cross+highc[l]-1]),axis=0)
		dum3 = np.all(np.array([rfreq>=eps_cross+lowc[l]+1, rfreq<eps_cross+highc[l]+1]),axis=0)
		index_l.append(np.any(np.array([dum1, dum2, dum3]),axis=0))


	# slice power spectrum into blocks
	n_blocks = int(lowerbound+upperbound)+1
	label_echx, label_echy, label_text = [[] for i in range(3)]
	rfreq_init = (numax/dnu)%1.0
	if rfreq_init-eps_cross < 0.0: freq_init = numax-dnu*lowerbound-dnu+np.abs(rfreq_init-eps_cross)*dnu-dnu02-0.05*dnu
	if rfreq_init-eps_cross >=0.0: freq_init = numax-dnu*lowerbound-np.abs(rfreq_init-eps_cross)*dnu-dnu02-0.05*dnu

	mode_l, mode_freq = [], []
	# find peaks in each dnu range
	for i in range(n_blocks):
		freq_low, freq_high = freq_init+i*dnu, freq_init+(i+1)*dnu
		index_norder = np.all(np.array([freq>=freq_low,freq<freq_high]),axis=0)
		if len(np.where(index_norder == True)[0])==0:
			continue
		
		# find peaks in each l range
		tindex_l, tmode_freq, tmode_l  = [], [], []
		for l in range(4):
			tindex_l.append(np.all(np.array([freq>=freq_low,freq<freq_high,index_l[l]]),axis=0))
			if len(freq[tindex_l[l]])==0: continue
			tfreq, tpower, tpowers = freq[tindex_l[l]], power[tindex_l[l]], powers[tindex_l[l]]
			meanlevel = np.median(tpowers)
			# find the highest peak in this range as a guess for the radial mode
			index_peaks, properties = find_peaks(tpowers, height=(2.0,None), 
				distance=int(dnu02/samplinginterval/5.0), prominence=(1.5,None))
			Npeaks = len(index_peaks)
			if Npeaks != 0:
				if l != 1:
					index_maxpeak = index_peaks[properties["peak_heights"] == properties["peak_heights"].max()]
					tmode_freq.append(tfreq[index_maxpeak[0]])
					tmode_l.append(l)
				else:
					for ipeak in range(Npeaks):
						tmode_freq.append(tfreq[index_peaks[ipeak]])
						tmode_l.append(l)
		tmode_freq, tmode_l = np.array(tmode_freq), np.array(tmode_l)
		mode_freq.append(tmode_freq)
		mode_l.append(tmode_l)

		### visulization (right)
		# ax1: the whole dnu range, ax2: only the l=1 range
		ax1 = fig.add_subplot(n_blocks,3,3*n_blocks-3*i-1)
		ax2 = fig.add_subplot(n_blocks,3,3*n_blocks-3*i)
		ax1.plot(freq[index_norder], power[index_norder], color="black")
		ax1.plot(freq[tindex_l[0]], powers[tindex_l[0]], color="C0", linewidth=3)
		ax1.plot(freq[tindex_l[1]], powers[tindex_l[1]], color="C3", linewidth=3)
		ax1.plot(freq[tindex_l[2]], powers[tindex_l[2]], color="C2", linewidth=3)
		ax1.plot(freq[tindex_l[3]], powers[tindex_l[3]], color="C1", linewidth=3)
		ax2.plot(freq[tindex_l[1]], power[tindex_l[1]], color="black")
		ax2.plot(freq[tindex_l[1]], powers[tindex_l[1]], color="C3", linewidth=3)
		ax2.text(1.1, 0.5, str(i), ha="center", va="center", transform=ax2.transAxes, 
			bbox=dict(facecolor='white', edgecolor="black"))

		# label the mode candidates
		colors=["C0","C3","C2","C1"]
		Npeaks = len(tmode_freq)
		for ipeak in range(Npeaks):
			ax1.axvline(tmode_freq[ipeak], linestyle="--", color=colors[tmode_l[ipeak]], linewidth=3)
		Npeaks1 = len(tmode_freq[tmode_l==1])
		for ipeak in range(Npeaks1):
			ax2.axvline(tmode_freq[tmode_l==1][ipeak], linestyle="--", color=colors[1], linewidth=3)


		# labels on the right side of the echelle
		label_text.append("{:0.0f}".format(i))
		label_echx.append(2.01*dnu)
		py = (freq_high-offset) - ((freq_high-offset) % dnu) + dnu/2.0 + offset - dnu
		label_echy.append(py)

		### end of visulization

	mode_freq = np.array([j for k in mode_freq for j in k])
	mode_l = np.array([j for k in mode_l for j in k])

	### visulization (left) - plot echelle and collapsed echelle to locate peak
	ax1 = plt.subplot2grid((5,3), (0,0), rowspan=2)
	echx, echy, echz = echelle(freq, powers, dnu, freq.min(), freq.max(), 
		echelletype="replicated", offset=offset)
	levels = np.linspace(np.min(echz), np.max(echz), 500)
	ax1.contourf(echx, echy, echz, cmap="gray_r", levels=levels)
	ax1.axis([np.min(echx), np.max(echx), np.min(echy), np.max(echy)])
	ax1.axvline(dnu, color="C0")
	# labels on the right side of the echelle
	for iblock in range(n_blocks):
		ax1.text(label_echx[iblock], label_echy[iblock], label_text[iblock], verticalalignment='center')
	ax1.set_ylabel("Frequency [muHz]")


	# mark mode candidates on the echelle
	px = (mode_freq-offset) % dnu
	py = (mode_freq-offset) - ((mode_freq-offset) % dnu) + dnu/2.0 + offset
	for l in range(4):
		if len(px[mode_l==l]) == 0: continue
		ax1.plot(px[mode_l==l], py[mode_l==l], "x", color=colors[l])
		ax1.plot(px[mode_l==l]+dnu, py[mode_l==l]-dnu, "x", color=colors[l])

	# collapsed echelle
	ax2 = fig.add_subplot(5,3,7)
	ax2.plot(freqc, powerc, color='black')
	ax2.axis([freqc.min(), freqc.max(), powerc.min(), powerc.max()])

	# template
	ax3 = fig.add_subplot(5,3,10)
	ax3.plot(freqc, ytemplate, color='C2')
	ax3.axis([freqc.min(), freqc.max(), powerc.min(), powerc.max()])
	if offset > 0.0:
		ax3.set_xlabel("(Frequency - "+str("{0:.2f}").format(offset)+ ") mod "+str("{0:.2f}").format(dnu) + " [muHz]")
	if offset < 0.0:
		ax3.set_xlabel("(Frequency + "+str("{0:.2f}").format(np.abs(offset))+ ") mod "+str("{0:.2f}").format(dnu) + " [muHz]")
	if offset == 0.0:
		ax3.set_xlabel("Frequency mod "+str("{0:.2f}").format(dnu) + " [muHz]")

	# cross correlation
	ax4 = fig.add_subplot(5,3,13)
	ax4.plot(lag, rho)
	ax4.plot(lag[index_hp], rho[index_hp], "x", color="C1")
	ax4.axis([-dnu, dnu, rho.min(), rho.max()+0.1])
	ax4.set_xlabel("Frequency lag [muHz]")
	### end of visulizaiton


	# save plot
	plt.savefig(filepath+"manualGuess.png")
	plt.close()

	# save a table
	# but first let's associate each mode with a group number
	mode_freq_group, mode_l_group, mode_group = [], [], np.array([])
	index = np.argsort(mode_freq)
	mode_freq, mode_l = mode_freq[index], mode_l[index]
	dist = mode_freq[1:] - mode_freq[:-1]
	group_index = np.where(dist>=0.2*dnu)[0] + 1 #each element the new group start from 
	Ngroups = len(group_index) + 1
	group_index = np.insert(group_index,0,0)
	group_index = np.append(group_index,len(mode_freq))

	# just sort a bit
	for igroup in range(Ngroups):
		tmode_freq = mode_freq[group_index[igroup]:group_index[igroup+1]]
		tmode_l = mode_l[group_index[igroup]:group_index[igroup+1]]

		index = np.argsort(tmode_l)
		tmode_freq, tmode_l = tmode_freq[index], tmode_l[index]

		mode_freq_group.append(tmode_freq)
		mode_l_group.append(tmode_l)
		elements = group_index[igroup+1] - group_index[igroup]
		for j in range(elements):
			mode_group = np.append(mode_group,igroup)

	mode_group = np.array(mode_group, dtype=int)
	mode_freq = np.concatenate(mode_freq_group)
	mode_l = np.concatenate(mode_l_group)

	table = np.array([np.arange(len(mode_freq)), np.zeros(len(mode_freq))+1, 
		mode_group, mode_l, mode_freq]).T
	np.savetxt(filepath+"frequencyGuess.csv", table, delimiter=",", fmt=("%d","%d","%d","%d","%10.4f"), 
		header="mode_id, ifpeakbagging, igroup, lGuess, freqGuess")

	return

def manualFit(freq: np.array, power: np.array, dnu: float, numax: float, filepath: str,
	 frequencyGuessFile: str, fittype: str="ParallelTempering", ifreadfromLS: bool=False,
	 igroup: int=None, para_guess: np.array=None):
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
	ifreadfromLS: set true to read from LeastSquare fitting results as the
		initial value, i.e. the mode_guess array passed to modefitWrapper.
	igroup: run the specific group to enable a truly manual fit. should be used 
		with para_guess

	Output:
	Files containing necessary outputs.

	'''

	# check
	if not os.path.exists(filepath):
		raise ValueError("Filepath does not exist.")

	if len(freq) != len(power):
		raise ValueError("len(freq) != len(power)")

	if igroup == None:
		automode=True
	else:
		automode=False

	# smooth the power spectrum
	period, samplinginterval = dnu/30.0, np.median(freq[1:-1] - freq[0:-2])
	powers = smoothWrapper(freq, power, period, "bartlett", samplinginterval)

	# read in table and cluster in group
	table = np.loadtxt(frequencyGuessFile, delimiter=",", ndmin=2)
	# columns: "mode_id, ifpeakbagging, igroup, lGuess, freqGuess"
	table = table[table[:,1]==1]
	if len(table) == 0: 
		print("Void guessed frequency input.")
	else:
		groups = np.unique(table[:,2])
		Ngroups = len(groups)

		inclination, fnyq = 0.0, 283.2 # only for radial modes, LC kepler data

		if automode: rstart, rend = 0, Ngroups
		if not automode: rstart, rend = igroup, igroup+1
		for i in range(rstart, rend): #range(1)
			igroup = groups[i]
			ttable = table[table[:,2]==igroup]
			mode_freq, mode_l = ttable[:,4], np.array(ttable[:,3], dtype=int)
			fitlowerbound, fitupperbound = dnu*0.1, dnu*0.1
			tfilepath = filepath + "{:0.0f}".format(igroup) + "/"
			if not os.path.exists(tfilepath): os.mkdir(tfilepath)

			# modefit
			modefitWrapper(dnu, inclination, fnyq, mode_freq, mode_l, 
				freq, power, powers, tfilepath, fittype=fittype, 
				para_guess=para_guess, fitlowerbound=fitlowerbound,
				fitupperbound=fitupperbound)
			# ifh1test temporarily deleted

	return

def manualSummarize(frequencyGuessFile: str, fittype: str="ParallelTempering"):
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
	if not fittype in ["Ensemble", "LeastSquare"]:
		raise ValueError("fittype should be one of ['Ensemble', 'LeastSquare']")

	filepath = "/".join(frequencyGuessFile.split("/")[:-1]) + "/"

	# read in table and cluster in group
	table = np.loadtxt(frequencyGuessFile, delimiter=",", ndmin=2)
	if len(table) != 0: 
		# columns: "mode_id, ifpeakbagging, igroup, lGuess, freqGuess"
		table = table[table[:,1] == 1]
		group_all = np.unique(table[:,2])

		if len(group_all) != 0: 
			# create lists to store results
			if fittype == "ParallelTempering":
				keys = ["igroup", "l", "mode_id", "PTamp", "PTamp_lc", "PTamp_uc", "PTlw", "PTlw_lc", "PTlw_uc", "PTfs", "PTfs_lc", "PTfs_uc", "PTfc", "PTfc_lc", "PTfc_uc", "PTlnK", "PTlnK_err"]
				fmt = ["%d", "%d", "%d", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f"]
			if fittype == "Ensemble":
				keys = ["igroup", "l", "mode_id", "ESamp", "ESamp_lc", "ESamp_uc", "ESlw", "ESlw_lc", "ESlw_uc", "ESfs", "ESfs_lc", "ESfs_uc", "ESfc", "ESfc_lc", "ESfc_uc"]
				fmt = ["%d", "%d", "%d", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f", "%10.4f"]
			if fittype == "LeastSquare":
				keys = ["igroup", "l", "mode_id", "LSamp", "LSlw", "LSfs", "LSfc"]
				fmt = ["%d", "%d", "%d", "%10.4f", "%10.4f", "%10.4f", "%10.4f"]
			data = [[] for i in range(len(keys))]

			# store pkbg results
			for igroup in group_all:
				tfilepath = filepath + str(int(igroup)) + "/"
				mode_l = table[table[:,2]==igroup][:,3]
				mode_id = table[table[:,2]==igroup][:,0]

				Nmodes = len(mode_l)
				# tindex = table[:,0] == group
				# ttable = table[tindex,:]		

				if fittype == "ParallelTempering":
					ttable = np.loadtxt(tfilepath+"PTsummary.txt", delimiter=",", ndmin=2)

					istart = 0
					for imode in range(Nmodes):
						if mode_l[imode] == 0:
							for j in range(0,6):
								data[j+3].append(ttable[istart+int(j/3), j%3])
							for j in range(6,9):
								data[j+3].append(0)
							for j in range(9,12):
								data[j+3].append(ttable[istart+int((j-3)/3), (j-3)%3])	
							istart += 3
						else:
							for j in range(12):
								data[j+3].append(ttable[istart+int(j/3), j%3])
							istart += 4
						data[0].append(igroup)
						data[1].append(mode_l[imode])
						data[2].append(mode_l[imode])
						ttable1 = np.loadtxt(tfilepath+"PTevidence.txt", delimiter=",")
						ttable2 = np.loadtxt(tfilepath+"evidence_h1_0.txt", delimiter=",")
						data[15].append(ttable1[0]-ttable2[0])
						data[16].append((ttable1[1]**2.0+ttable2[1]**2.0)**0.5)


				if fittype == "Ensemble":
					ttable = np.loadtxt(tfilepath+"ESsummary.txt", delimiter=",", ndmin=2)

					istart = 0
					for imode in range(Nmodes):
						if mode_l[imode] == 0:
							for j in range(0,6):
								data[j+3].append(ttable[istart+int(j/3), j%3])
							for j in range(6,9):
								data[j+3].append(0)
							for j in range(9,12):
								data[j+3].append(ttable[istart+int((j-3)/3), (j-3)%3])	
							istart += 3
						else:
							for j in range(12):
								data[j+3].append(ttable[istart+int(j/3), j%3])
							istart += 4
						data[0].append(igroup)
						data[1].append(mode_l[imode])
						data[2].append(mode_id[imode])

				if fittype == "LeastSquare":
					ttable = np.loadtxt(tfilepath+"LSsummary.txt", delimiter=",")
					istart = 0
					for imode in range(Nmodes):
						if mode_l[imode] == 0:
							for j in range(0,2):
								data[j+3].append(ttable[istart+j])
							for j in range(2,3):
								data[j+3].append(0)
							for j in range(3,4):
								data[j+3].append(ttable[istart+j-1])	
							istart += 3
						else:
							for j in range(4):
								data[j+3].append(ttable[istart+j])
							istart += 4
						data[0].append(igroup)
						data[1].append(mode_l[imode])
						data[2].append(mode_id[imode])


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
				add_group = np.array(olddata[:,0])[~np.isin(olddata[:,2], data[2])]  # match by mode_id
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