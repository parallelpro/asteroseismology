#!/usr/bin/env/ python
# coding: utf-8


import numpy as np
import sys
import emcee
from scipy.optimize import minimize

import matplotlib.pyplot as plt
import os
import corner

__all__ = ["modefitWrapper", "h1testWrapper"]

def ResponseFunction(freq, fnyq):
	sincfunctionarg = (np.pi/2.0)*freq/fnyq
	responsefunction = (np.sin(sincfunctionarg)/sincfunctionarg)**2.0
	return responsefunction

def LorentzianSplittingMixtureModel(freq, modelParameters, fnyq, mode_l):
	amplitude = modelParameters[0]
	linewidth = modelParameters[1]
	projectedSplittingFrequency = modelParameters[2]
	centralFrequency = modelParameters[3]
	inclination = modelParameters[4]
	splittingFrequency = projectedSplittingFrequency#/np.sin(inclination)
	height = amplitude**2.0/(np.pi*linewidth)

	sincFunctionArgument = (np.pi / 2.0) * freq / fnyq
	responseFunction = (np.sin(sincFunctionArgument) / sincFunctionArgument)**2.0

	power = np.zeros(len(freq))

	if mode_l == 0:
		power += height/(1.0 + (4.0*(freq-centralFrequency)**2.0/(linewidth**2.0)))
	if mode_l == 1:
		visibility_m0 = np.cos(inclination)**2.0
		visibility_m1 = np.sin(inclination)**2.0*0.5
		power += visibility_m0 * height/(1.0 + (4.0*(freq-centralFrequency)**2.0/(linewidth**2.0)))
		power += visibility_m1 * height/(1.0 + (4.0*(freq-centralFrequency-splittingFrequency)**2.0/(linewidth**2.0)))
		power += visibility_m1 * height/(1.0 + (4.0*(freq-centralFrequency+splittingFrequency)**2.0/(linewidth**2.0)))
	if mode_l == 2:
		visibility_m0 = (3.0*np.cos(inclination)**2.0-1)**2.0*0.25;
		visibility_m1 = np.sin(inclination*2.0)**2.0*3.0/8.0;
		visibility_m2 = np.sin(inclination)**4.0*3.0/8.0;
		power += visibility_m0 * height/(1.0 + (4.0*(freq-centralFrequency)**2.0/(linewidth**2.0)))
		power += visibility_m1 * height/(1.0 + (4.0*(freq-centralFrequency-splittingFrequency)**2.0/(linewidth**2.0)))
		power += visibility_m1 * height/(1.0 + (4.0*(freq-centralFrequency+splittingFrequency)**2.0/(linewidth**2.0)))
		power += visibility_m2 * height/(1.0 + (4.0*(freq-centralFrequency-2.0*splittingFrequency)**2.0/(linewidth**2.0)))
		power += visibility_m2 * height/(1.0 + (4.0*(freq-centralFrequency+2.0*splittingFrequency)**2.0/(linewidth**2.0)))
	if mode_l == 3:
		visibility_m0 = pow(-3.0*np.cos(inclination)+5.0*pow(np.cos(inclination),3.0),2.0)/4.0
		visibility_m1 = 3.0/16.0*pow(np.sin(inclination),2.0)*pow(-1.0+5.0*pow(np.cos(inclination),2.0),2.0)
		visibility_m2 = 15.0/8.0*pow(np.cos(inclination),2.0)*pow(-1.0+pow(np.cos(inclination),2.0),2.0)
		visibility_m3 = 5.0/16.0*pow(1.0-pow(np.cos(inclination),2.0),3.0)
		power += visibility_m0 * height/(1.0 + (4.0*(freq-centralFrequency)**2.0/(linewidth**2.0)))
		power += visibility_m1 * height/(1.0 + (4.0*(freq-centralFrequency-splittingFrequency)**2.0/(linewidth**2.0)))
		power += visibility_m1 * height/(1.0 + (4.0*(freq-centralFrequency+splittingFrequency)**2.0/(linewidth**2.0)))
		power += visibility_m2 * height/(1.0 + (4.0*(freq-centralFrequency-2.0*splittingFrequency)**2.0/(linewidth**2.0)))
		power += visibility_m2 * height/(1.0 + (4.0*(freq-centralFrequency+2.0*splittingFrequency)**2.0/(linewidth**2.0)))
		power += visibility_m3 * height/(1.0 + (4.0*(freq-centralFrequency-3.0*splittingFrequency)**2.0/(linewidth**2.0)))
		power += visibility_m3 * height/(1.0 + (4.0*(freq-centralFrequency+3.0*splittingFrequency)**2.0/(linewidth**2.0)))

	power *= responseFunction;

	return power

def SincModel(freq, modelParameters, fnyq, resolution):
	height = modelParameters[0]
	centralFrequency = modelParameters[1]

	sincFunctionArgument = (np.pi / 2.0) * freq / fnyq
	responseFunction = (np.sin(sincFunctionArgument) / sincFunctionArgument)**2.0

	unresolvedArgument = np.pi * (freq - centralFrequency) / resolution
	power = height * (np.sin(unresolvedArgument) / unresolvedArgument)**2.0
	power *= responseFunction;

	return power

def GuessLorentzianModelPriorForPeakbagging(mode_freq, mode_l, freq, power, powers, dnu,
	ifReturnSplitModelPrior = False, lowerbound=None, upperbound=None):
	dnu02 = 0.122*dnu + 0.05 # Bedding+2011 low luminosity RGB
	if lowerbound==None:
		lowerbound = mode_freq - 0.5*dnu02#0.04*dnu
	else:
		lowerbound = max(lowerbound, mode_freq - 0.5*dnu02)
	lowerbound = max(lowerbound, np.min(freq))
	if upperbound==None:
		upperbound = mode_freq + 0.5*dnu02#0.04*dnu
	else:
		upperbound = min(upperbound, mode_freq + 0.5*dnu02)
	upperbound = min(upperbound, np.max(freq))

	index = np.intersect1d(np.where(freq > lowerbound)[0],np.where(freq < upperbound)[0])
	freq, power, powers = freq[index], power[index], powers[index]

	height = np.max(powers)-1.0
	height = height if height>0 else np.max(powers)
	dfreq = np.median(freq[1:]-freq[:-1])
	area = np.sum(powers*dfreq-1.0*dfreq)
	lw = 2.0*area/height/np.pi if area>0 else 1.0
	amp = (height*np.pi*lw)**0.5

	# Flat priors
	centralFrequency = [lowerbound, upperbound]
	amplitude = [amp*0.2, amp*5.0]
	linewidth = [lw*0.2, lw*5.0]#[1e-8, dnu02*0.7]

	if ifReturnSplitModelPrior:
		projectedSplittingFrequency = [0.0, 12.0]
		prior2 = np.array([amplitude, linewidth, projectedSplittingFrequency, centralFrequency])
		return prior2
	else:
		prior1 = np.array([amplitude, linewidth, centralFrequency])
		return prior1


def GuessBestLorentzianModelForPeakbagging(mode_freq, mode_l, freq, power, powers, dnu, 
	ifReturnSplitModelPrior = False, lowerbound=None, upperbound=None):
	dnu02 = 0.122*dnu + 0.05 # Bedding+2011 low luminosity RGB
	if lowerbound==None:
		lowerbound = mode_freq - 0.5*dnu02#0.04*dnu
	else:
		lowerbound = max(lowerbound, mode_freq - 0.5*dnu02)
	lowerbound = max(lowerbound, np.min(freq))
	if upperbound==None:
		upperbound = mode_freq + 0.5*dnu02#0.04*dnu
	else:
		upperbound = min(upperbound, mode_freq + 0.5*dnu02)
	upperbound = min(upperbound, np.max(freq))

	index = np.intersect1d(np.where(freq > lowerbound)[0],np.where(freq < upperbound)[0])
	power = power[index]
	powers = powers[index]
	centralFrequency = mode_freq

	height = np.max(powers)-1.0
	height = height if height>0 else np.max(powers)
	dfreq = np.median(freq[1:]-freq[:-1])
	area = np.sum(powers*dfreq-1.0*dfreq)
	lw = 2.0*area/height/np.pi if area>0 else 1.0
	amp = (height*np.pi*lw)**0.5

	amplitude = amp
	linewidth = lw

	if ifReturnSplitModelPrior:
		projectedSplittingFrequency = 0.1
		prior2 = np.array([amplitude, linewidth, projectedSplittingFrequency, centralFrequency])
		return prior2
	else:
		prior1 = np.array([amplitude, linewidth, centralFrequency])
		return prior1

def lnprior_m1(theta, n_mode, n_mode_l0, flatPriors):
	pointer = True
	for j in range(0, 3*n_mode_l0 + (n_mode - n_mode_l0)*4):
		if not flatPriors[j][0] <= theta[j] <= flatPriors[j][1]:
			pointer = False
	if pointer == True:
		if n_mode - n_mode_l0 > 0:
			S, U, sigma = 0.0, 2.0, 1.0
			H = 1.0/(U + (2*np.pi)**0.5 * sigma/2.0 - S)
			for i in range(n_mode - n_mode_l0):
				if theta[3*n_mode_l0 + 4*i + 2] < U:
					lnfsprior = np.log(H)
				else:
					lnfsprior = np.log(H) - (theta[3*n_mode_l0 + 4*i + 2]-U)**2.0/(2*sigma**2.0)
		else:
			lnfsprior = 0.0
		lnfspriorbase = 0.0
		for k in range(0, 3*n_mode_l0 + (n_mode - n_mode_l0)*4):
			if not k in 3*n_mode_l0 + 4*np.arange(0,n_mode - n_mode_l0) + 2:
				lnfspriorbase = np.log(1.0/(flatPriors[j][1] - flatPriors[j][0]))
		return lnfspriorbase + lnfsprior
	else:
		return -np.inf

def lnlikelihood_m1(theta, freq, power, inclination, fnyq, mode_l, n_mode, n_mode_l0, ifresolved):
	model = np.zeros(len(freq))
	for j in range(0, n_mode_l0):
		if ifresolved[j]:
			model += LorentzianSplittingMixtureModel(freq, [theta[3*j], theta[3*j+1], 0.0, 
						theta[3*j+2], inclination], fnyq, 0)
		else:
			model += SincModel(freq, [theta[3*j], theta[3*j+2]], fnyq, resolution)
	for j in range(0, n_mode - n_mode_l0):
		if ifresolved[j]:
			model += LorentzianSplittingMixtureModel(freq, [theta[3*n_mode_l0+4*j], theta[3*n_mode_l0+4*j+1], 
						theta[3*n_mode_l0+4*j+2], theta[3*n_mode_l0+4*j+3], inclination], fnyq, mode_l[n_mode_l0+j])
		else:
			model += SincModel(freq, [theta[3*j], theta[3*j+2]], fnyq, resolution)
	model += 1.0
	return -np.sum(np.log(model) + power/model)


def lnpost_m1(theta, n_mode, n_mode_l0, flatPriors, freq, power, inclination, fnyq, mode_l, ifresolved):
	lp = lnprior_m1(theta, n_mode, n_mode_l0, flatPriors)
	if not np.isfinite(lp):
		return -np.inf
	else:
		return lp + lnlikelihood_m1(theta, freq, power, inclination, fnyq, mode_l, n_mode, n_mode_l0, ifresolved)

def lnprior_m0(theta, tpower):
	if tpower.min() <= theta[0] <= tpower.max():
		return np.log(1.0/(tpower.max() - tpower.min()))
	else:
		return -np.inf

def lnlikelihood_m0(theta, freq, power, fnyq):
	model = np.zeros(len(freq))
	model += theta[0]
	model *= ResponseFunction(freq, fnyq)
	return -np.sum(np.log(model) + power/model)


def modefitWrapper(dnu: float, inclination: float, fnyq: float, mode_freq: np.array, mode_l: np.array,
	freq: np.array, power: np.array, powers: np.array, filepath: str, fittype: str="ParallelTempering",
	ifoutputsamples: bool=False, para_guess: np.array=None, fitlowerbound: float=None,
	fitupperbound: float=None, nsteps: int=None, ifresolved: np.array=None, resolution: float=None):
	'''
	Provide a wrapper to fit mode defined in mode_freq.

	Input:
	dnu: float
		the large separation, in unit of muHz.

	inclination: float
		the inclination angle, in rad.

	fnyq: float
		the Nyquist frequency, in muHz.

	mode_freq: np.array
		the mode frequencies intend to fit, in muHz.

	mode_l: np.array
		the mode degree corresponding to mode_freq.
		now only support 0, 1, 2, and 3.

	freq: np.array
		frequency in muHz.

	power: np.array
		the background divided power spectrum (so now is s/b instead).

	powers: np.array
		the smoothed background divided power spectrum
		used to predict priors.

	filepath: str
		the file path to store outputs.


	Optional input:

	fittype: str, default: "ParallelTempering"
		one of ["ParallelTempering", "Ensemble", "LeastSquare"].

	ifoutputsamples: bool, default: False
		set True to output MCMC sampling points.
	
	para_guess: np.array, default: None
		the guessed initial parameters.

	fitlowerbound: float, default: None
		trim the data into [min(mode_freq)-fitlowerbound*dnu,
		max(mode_freq)+fitupperbound*dnu] for fit.

	fitupperbound: float, default: None
		trim the data into [min(mode_freq)-fitlowerbound*dnu,
		max(mode_freq)+fitupperbound*dnu] for fit.

	nsteps: int, default: 2000
		the number of steps to iterate for mcmc run.

	ifresolved: bool, default: True
		whether the modes are resolved. pass a 1-d array (len(mode_freq),)
		containing True/False.

	resolution: float, default: None
		the frequency spectra resolution. must be set when passing values
		from ``ifresolved''.


	Output:

	Data: acceptance fraction, bayesian evidence, 
		parameter estimation result, parameter initial guess.
	Plots: fitting results, posterior distribution, traces.

	'''

	# check
	if len(mode_freq) != len(mode_l):
		raise ValueError("len(mode_freq) != len(mode_l)")
	if not len(freq) == len(power) == len(powers):
		raise ValueError("not len(freq) == len(power) == len(powers)")
	if not fittype in ["ParallelTempering", "Ensemble", "LeastSquare"]:
		raise ValueError("fittype should be one of ['ParallelTempering', 'Ensemble', 'LeastSquare']")
	automode = True if isinstance(para_guess, type(None)) else False
	if fitlowerbound==None: fitlowerbound=0.5
	if fitupperbound==None: fitupperbound=0.5
	if ifresolved==None: 
		ifresolved=np.array(np.zeros(len(mode_freq))+1, dtype=bool)
		if resolution==None:
			raise ValueError("Resolution not set.")
	


	fitlowerbound *= dnu
	fitupperbound *= dnu

	# initilize
	n_mode = len(mode_l)
	n_mode_l0 = len( np.where(mode_l == 0 )[0] )

	# trim data into range we use
	# this is for plot
	index = np.all(np.array([freq >= np.min(mode_freq)-0.5*dnu, 
		freq <= np.max(mode_freq)+0.5*dnu]), axis=0)
	freq = freq[index]
	power = power[index]
	powers = powers[index]

	# this is for fit
	index = np.all(np.array([freq >= np.min(mode_freq)-fitlowerbound, 
		freq <= np.max(mode_freq)+fitupperbound]), axis=0)
	tfreq = freq[index]
	tpower = power[index]
	tpowers = powers[index]

	# defining likelihood and prior
	# model 1 - splitting model
	flatPriors = []
	if automode: para_guess = np.array([])


	for j in range(n_mode):
		if mode_freq[j] == np.min(mode_freq):
			lowerbound = None
		else:
			dummy = np.sort(mode_freq[mode_freq<mode_freq[j]])
			lowerbound = (dummy[-1]+mode_freq[j])/2.0 
		if mode_freq[j] == np.max(mode_freq):
			upperbound = None
		else:
			dummy = np.sort(mode_freq[mode_freq>mode_freq[j]])
			upperbound = (dummy[0]+mode_freq[j])/2.0
		# print(lowerbound, mode_freq[j], upperbound)

		ifsplit = True if mode_l[j]>0 else False
		prior = GuessLorentzianModelPriorForPeakbagging(mode_freq[j], mode_l[j], 
			tfreq, tpower, tpowers, dnu, ifsplit, lowerbound=lowerbound, upperbound=upperbound)
		guess = GuessBestLorentzianModelForPeakbagging(mode_freq[j], mode_l[j], 
			tfreq, tpower, tpowers, dnu, ifsplit, lowerbound=lowerbound, upperbound=upperbound)
		for k in range(len(prior)):
			flatPriors.append(prior[k])
			if automode: para_guess = np.append(para_guess, guess[k])

	
	# write guessed parameters;
	para_guess = para_guess

	if fittype in ["ParallelTempering", "Ensemble"]:

		if fittype == "ParallelTempering":

			# run mcmc with pt sampler
			ndim, nwalkers, ntemps = n_mode_l0 * 3 + (n_mode - n_mode_l0 ) * 4, 100, 20
			print("enabling ParallelTempering sampler.")
			print("ndimension: ", ndim, ", nwalkers: ", nwalkers, ", ntemps: ", ntemps)
			pos0 = [[para_guess + 1.0e-8*np.random.randn(ndim) for j in range(nwalkers)] for k in range(ntemps)]
			loglargs = [tfreq, tpower, inclination, fnyq, mode_l, n_mode, n_mode_l0, ifresolved]
			logpargs = [n_mode, n_mode_l0, flatPriors]
			sampler = emcee.PTSampler(ntemps, nwalkers, ndim, lnlikelihood_m1, lnprior_m1, loglargs=loglargs, logpargs=logpargs)

			# burn-in
			nburn, width = 1000, 30
			print("start burning in. nburn:", nburn)
			for j, result in enumerate(sampler.sample(pos0, iterations=nburn, thin=10)):
				n = int((width+1) * float(j) / nburn)
				sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
			sys.stdout.write("\n")
			pos, lnpost, lnlike = result
			sampler.reset()

			# actual iteration
			nsteps = 2000 if nsteps == None else nsteps
			width = 30 # 10000, 30
			print("start iterating. nsteps:", nsteps)
			for j, result in enumerate(sampler.sample(pos, iterations=nsteps, lnprob0=lnpost, lnlike0=lnlike)):
				#p, lnpost, lnlike = result
				n = int((width+1) * float(j) / nsteps)
				sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
			sys.stdout.write("\n")

			# modify samples
			samples = sampler.chain[0,:,:,:].reshape((-1,ndim))

			# save evidence
			evidence = sampler.thermodynamic_integration_log_evidence() 
			print("Bayesian evidence lnZ: {:0.5f}".format(evidence[0]))
			print("Bayesian evidence error dlnZ: {:0.5f}".format(evidence[1]))
			np.savetxt(filepath+"PTevidence.txt", evidence, delimiter=",", fmt=("%0.8f"), header="bayesian_evidence")

		if fittype == "Ensemble":

			# run mcmc with ensemble sampler
			ndim, nwalkers = n_mode_l0 * 3 + (n_mode - n_mode_l0 ) * 4, 100
			print("enabling Ensemble sampler.")
			print("ndimension: ", ndim, ", nwalkers: ", nwalkers)
			pos0 = [para_guess + 1.0e-8*np.random.randn(ndim) for j in range(nwalkers)]
			args = [n_mode, n_mode_l0, flatPriors, tfreq, tpower, inclination, fnyq, mode_l, ifresolved]
			sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost_m1, args=args)

			# burn-in
			nburn, width = 1000, 30
			print("start burning in. nburn:", nburn)
			for j, result in enumerate(sampler.sample(pos0, iterations=nburn, thin=10)):
				n = int((width+1) * float(j) / nburn)
				sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
			sys.stdout.write("\n")
			pos, lnpost, rstate = result
			sampler.reset()

			# actual iteration
			nsteps = 2000 if nsteps == None else nsteps
			width = 30 # 10000, 30
			print("start iterating. nsteps:", nsteps)
			for j, result in enumerate(sampler.sample(pos, iterations=nsteps, lnprob0=lnpost)):
				#pos, lnpost, rstate = result
				n = int((width+1) * float(j) / nsteps)
				sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
			sys.stdout.write("\n")

			# modify samples
			samples = sampler.chain[:,:,:].reshape((-1,ndim))


		# save samples if the switch is toggled on
		if ifoutputsamples: samples.save(filepath+"samples.npy")
		if fittype == "ParallelTempering": st = "PT"
		if fittype == "Ensemble": st = "ES"

		# save guessed parameters
		np.savetxt(filepath+st+"guess.txt", para_guess, delimiter=",", fmt=("%0.8f"), header="para_guess")

		# plot triangle and save
		tp = ["amp", "lw", "fc"]
		para1 = [tp[k]+str(j) for j in range(n_mode_l0) for k in range(3)]
		tp = ["amp", "lw", "fs", "fc"]
		para2 = [tp[k]+str(j) for j in range(n_mode_l0, n_mode) for k in range(4)]
		para = para1 + para2
		fig = corner.corner(samples, labels=para, quantiles=(0.16, 0.5, 0.84), truths=para_guess)
		fig.savefig(filepath+st+"triangle.png")
		plt.close()

		# save estimation result
		result = np.array(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
			zip(*np.percentile(samples, [16, 50, 84],axis=0)))))
		np.savetxt(filepath+st+"summary.txt", result, delimiter=",", fmt=("%0.8f", "%0.8f", "%0.8f"), header="parameter, upper uncertainty, lower uncertainty")
		para_fit = result[:,0]

		# save mean acceptance rate
		acceptance_fraction = np.array([np.mean(sampler.acceptance_fraction)])
		print("Mean acceptance fraction: {:0.3f}".format(acceptance_fraction[0]))
		np.savetxt(filepath+st+"acceptance_fraction.txt", acceptance_fraction, delimiter=",", fmt=("%0.8f"), header="acceptance_fraction")

		# plot traces and save
		fig = plt.figure(figsize=(5,ndim*3))
		for i in range(ndim):
			ax1=fig.add_subplot(ndim,1,i+1)
			evol=samples[:,i]
			Npoints=samples.shape[0]
			ax1.plot(np.arange(Npoints)/Npoints, evol, color="gray", lw=1, zorder=1)
			Nseries=int(len(evol)/15.0)
			evol_median=np.array([np.median(evol[i*Nseries:(i+1)*Nseries]) for i in range(0,15)])
			evol_std=np.array([np.std(evol[i*Nseries:(i+1)*Nseries]) for i in range(0,15)])
			evol_x=np.array([np.median(np.arange(Npoints)[i*Nseries:(i+1)*Nseries]/Npoints) for i in range(0,15)])
			ax1.errorbar(evol_x, evol_median, yerr=evol_std, color="C0", ecolor="C0", capsize=5)
			ax1.set_ylabel(para[i])
		plt.tight_layout()
		plt.savefig(filepath+st+'traces.png')
		plt.close()

	if fittype == "LeastSquare":
		
		st = "LS"
		# maximize likelihood function by scipy.optimize.minimize function
		function = lambda *arg: -lnlikelihood_m1(*arg)
		args = (tfreq, tpower, inclination, fnyq, mode_l, n_mode, n_mode_l0, ifresolved)
		result = minimize(function, para_guess, args=args, bounds=flatPriors)

		# save guessed parameters
		np.savetxt(filepath+st+"guess.txt", para_guess, delimiter=",", fmt=("%0.8f"), header="para_guess")

		# save estimation result
		np.savetxt(filepath+st+"summary.txt", result.x, delimiter=",", fmt=("%0.8f"), header="parameter")
		para_fit = result.x


	# plot fitting results and save
	power_fit, power_guess = np.zeros(len(freq)), np.zeros(len(freq))
	for j in range(0, n_mode_l0):
		power_fit += LorentzianSplittingMixtureModel(freq, [para_fit[3*j], para_fit[3*j+1], 0.0, 
							para_fit[3*j+2], inclination], fnyq, 0)
		power_guess += LorentzianSplittingMixtureModel(freq, [para_guess[3*j], para_guess[3*j+1], 0.0, 
							para_guess[3*j+2], inclination], fnyq, 0)
	for j in range(0, n_mode - n_mode_l0):
		power_fit += LorentzianSplittingMixtureModel(freq, [para_fit[3*n_mode_l0+4*j], para_fit[3*n_mode_l0+4*j+1], 
						para_fit[3*n_mode_l0+4*j+2], para_fit[3*n_mode_l0+4*j+3], inclination], fnyq, mode_l[n_mode_l0+j])
		power_guess += LorentzianSplittingMixtureModel(freq, [para_guess[3*n_mode_l0+4*j], para_guess[3*n_mode_l0+4*j+1], 
						para_guess[3*n_mode_l0+4*j+2], para_guess[3*n_mode_l0+4*j+3], inclination], fnyq, mode_l[n_mode_l0+j])
	power_fit += 1.0
	power_guess += 1.0
	fig = plt.figure(figsize=(6,5))
	ax = fig.add_subplot(1,1,1)
	ax.plot(freq, power, color="lightgray", label="power")
	ax.plot(freq, powers, color="black", label="smooth")
	ax.plot(freq, power_guess, color="blue", label="guess")
	ax.plot(freq, power_fit, color="orange", label="fit")
	ax.legend()
	a, b = np.min(mode_freq) - 0.5*dnu, np.max(mode_freq) + 0.5*dnu
	index = np.intersect1d(np.where(freq > a)[0], np.where(freq < b)[0])
	c, d = np.min(power[index]), np.max(power[index])
	color = ["blue", "red", "green", "purple"]
	marker = ["o", "^", "s", "v"]
	for j in range(n_mode):
		ax.scatter([mode_freq[j]],[c+(d-c)*0.8], c=color[mode_l[j]], marker=marker[mode_l[j]])
		if j<n_mode_l0: index=3*j+2
		if j>=n_mode_l0: index=3*n_mode_l0+4*(j-n_mode_l0)+3
		# print(j,flatPriors[index]-mode_freq[j])
		# print(np.abs(flatPriors[index]-mode_freq[j]))
		ax.errorbar([mode_freq[j]],[c+(d-c)*0.8], ecolor=color[mode_l[j]],
			 xerr=[[np.abs(flatPriors[index][0]-mode_freq[j])],
			 [np.abs(flatPriors[index][1]-mode_freq[j])]], capsize=5)
	ax.axis([a, b, c, d])
	ax.axvline(np.min(mode_freq)-fitlowerbound, linestyle="--", color="gray")
	ax.axvline(np.max(mode_freq)+fitupperbound, linestyle="--", color="gray")
	plt.savefig(filepath+st+"fit.png")
	plt.close()


	return

def h1testWrapper(dnu: float, fnyq: float, mode_freq: np.array, mode_l: np.array,
	freq: np.array, power: np.array, powers: np.array, filepath: str, fitlowerbound: float=None,
	fitupperbound: float=None):
	'''
	Provide a wrapper to perform H1 test.

	Input:
	dnu: float
		the large separation, in unit of muHz.

	inclination: float
		the inclination angle, in rad.

	fnyq: float
		the Nyquist frequency, in muHz.

	mode_freq: np.array
		the mode frequencies intend to fit, in muHz.

	mode_l: np.array
		the mode degree corresponding to mode_freq.
		now only support 0, 1, 2, and 3.

	freq: np.array
		frequency in muHz.

	power: np.array
		the background divided power spectrum (so now is s/b instead).

	powers: np.array
		the smoothed background divided power spectrum
		used to predict priors.

	filepath: str
		the file path to store outputs.

	Optional input:
	fitlowerbound: float, default: None
		trim the data into [min(mode_freq)-fitlowerbound,
		max(mode_freq)+fitupperbound] for fit.

	fitupperbound: float, default: None
		trim the data into [min(mode_freq)-fitlowerbound,
		max(mode_freq)+fitupperbound] for fit.

	Output:
	Data: acceptance fraction, bayesian evidence, 
		parameter estimation result, parameter initial guess.
	Plots: fitting results, posterior distribution, traces.


	'''

	# check
	if len(mode_freq) != len(mode_l):
		raise ValueError("len(mode_freq) != len(mode_l)")
	if not len(freq) == len(power) == len(powers):
		raise ValueError("not len(freq) == len(power) == len(powers)")
	if fitlowerbound==None: fitlowerbound=(0.122*dnu + 0.05)*0.6 # Bedding+2011 low luminosity RGB
	if fitupperbound==None: fitupperbound=(0.122*dnu + 0.05)*0.6

	# initilize
	n_mode = len(mode_freq)
	n_mode_l0 = len( np.where(mode_l == 0 )[0] )

	# trim data into range we use
	index = np.intersect1d(np.where(freq > np.min(mode_freq) - 8.0)[0],
		np.where(freq < np.max(mode_freq) + 8.0)[0])
	freq = freq[index]
	power = power[index]
	powers = powers[index]

	for i, tmode_freq in enumerate(mode_freq):
		dnu02 = 0.122*dnu + 0.05 # Bedding+2011 low luminosity RGB
		index = np.all(np.array([freq >= np.min(mode_freq)-fitlowerbound,
		 freq <= np.max(mode_freq)+fitupperbound]), axis=0)
		tfreq, tpower = freq[index], power[index]
		iden = str(i)

		# defining likelihood and prior
		# model 0 - nothing but a straight line

		# write guessed parameters
		para_guess = np.array([tpower.mean()])
		np.savetxt(filepath+"guess_h1_"+iden+".txt", para_guess, delimiter=",", fmt=("%0.8f"), header="para_guess")

		# run mcmc with pt sampler
		ndim, nwalkers, ntemps = 1, 100, 20
		print("ndimension: ", ndim, ", nwalkers: ", nwalkers, ", ntemps: ", ntemps)
		pos0 = [[para_guess + 1.0e-8*np.random.randn(ndim) for j in range(nwalkers)] for k in range(ntemps)]
		loglargs = [tfreq, tpower, fnyq]
		logpargs = [tpower]
		sampler = emcee.PTSampler(ntemps, nwalkers, ndim, lnlikelihood_m0, lnprior_m0, loglargs=loglargs, logpargs=logpargs)

		# burn-in
		nburn, width = 100, 30
		print("start burning in. nburn:", nburn)
		for j, result in enumerate(sampler.sample(pos0, iterations=nburn, thin=10)):
			n = int((width+1) * float(j) / nburn)
			sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
		sys.stdout.write("\n")
		pos, lnpost, lnlike = result
		sampler.reset()

		# actual iteration
		nsteps, width = 500, 30 # 10000, 30
		print("start iterating. nsteps:", nsteps)
		for j, result in enumerate(sampler.sample(pos, iterations=nsteps, lnprob0=lnpost, lnlike0=lnlike, thin=10)):
			#p, lnpost, lnlike = result
			n = int((width+1) * float(j) / nsteps)
			sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
		sys.stdout.write("\n")

		# modify samples
		samples = sampler.chain[0,:,:,:].reshape((-1,ndim))

		# save evidence
		evidence = sampler.thermodynamic_integration_log_evidence() 
		print("Bayesian evidence lnZ: {:0.5f}".format(evidence[0]))
		print("Bayesian evidence error dlnZ: {:0.5f}".format(evidence[1]))
		np.savetxt(filepath+"evidence_h1_"+iden+".txt", evidence, delimiter=",", fmt=("%0.8f"), header="bayesian_evidence")

		# plot triangle and save
		para = ["W"]
		fig = corner.corner(samples, labels=para, quantiles=(0.16, 0.5, 0.84), truths=para_guess)
		fig.savefig(filepath+"triangle_h1_"+iden+".png")
		plt.close()

		# save estimation result
		result = np.array(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
			zip(*np.percentile(samples, [16, 50, 84],axis=0)))))
		np.savetxt(filepath+"summary_h1_"+iden+".txt", result, delimiter=",", fmt=("%0.8f", "%0.8f", "%0.8f"), header="parameter, upper uncertainty, lower uncertainty")

		# plot fitting result and save
		para_fit = result[:,0]
		power_fit = np.zeros(len(freq)) + para_fit[0]
		fig = plt.figure(figsize=(6,5))
		ax = fig.add_subplot(1,1,1)
		ax.plot(freq, power, color="lightgray", label="power")
		ax.plot(freq, powers, color="black", label="smooth")
		ax.plot(freq, power_fit, color="orange", label="fit")
		ax.legend()
		a, b = np.min(mode_freq) - 8.0, np.max(mode_freq) + 8.0
		index = np.intersect1d(np.where(freq > a)[0],
			np.where(freq < b)[0] )
		c, d = np.min(power[index]), np.max(power[index])
		color = ["blue", "red", "green", "purple"]
		marker = ["o", "^", "s", "v"]
		for j in range(n_mode):
			ax.scatter([mode_freq[j]],[c+(d-c)*0.8], c=color[mode_l[j]], marker=marker[mode_l[j]])
		ax.axis([a, b, c, d])
		plt.savefig(filepath+"fit_h1_"+iden+".png")
		plt.close()

		# save mean acceptance rate
		acceptance_fraction = np.array([np.mean(sampler.acceptance_fraction)])
		print("Mean acceptance fraction: {:0.3f}".format(acceptance_fraction[0]))
		np.savetxt(filepath+"acceptance_fraction_h1_"+iden+".txt", acceptance_fraction, delimiter=",", fmt=("%0.8f"), header="acceptance_fraction")

		# plot traces and save
		fig = plt.figure(figsize=(5,ndim*3))
		for i in range(ndim):
			ax1=fig.add_subplot(ndim,1,i+1)
			ax1.plot(np.arange(samples.shape[0]), samples[:,i], color="black", lw=1, zorder=1)
			ax1.set_ylabel(para[i])
		plt.savefig(filepath+"traces_h1_"+iden+".png")
		plt.close()

	return
