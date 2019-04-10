#!/usr/bin/env/ python
# coding: utf-8


import numpy as np
import sys
import emcee
from scipy.optimize import minimize

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()
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

def GuessLorentzianModelPriorForPeakbagging(mode_freq, mode_l, freq, power, powers, dnu, ifReturnSplitModelPrior = False):
	dnu02 = 0.122*dnu + 0.05 # Bedding+2011 low luminosity RGB
	index = np.intersect1d(np.where(freq > mode_freq - 0.6*dnu02)[0],np.where(freq < mode_freq + 0.6*dnu02)[0])
	power = power[index]
	powers = powers[index]

	# Flat priors
	centralFrequency = [mode_freq-0.4*dnu02, mode_freq+0.4*dnu02]
	amplitude = [(np.max(powers)**0.5)*0.1, (np.max(powers)**0.5)*5.0]
	linewidth = [1e-8, dnu02*0.7]
	prior1 = np.array([amplitude, linewidth, centralFrequency])

	if ifReturnSplitModelPrior:
		# Flat decaying
		#projectedSplittingFrequency = [0.0, 3.0, 0.5]

		# Flat priors
		projectedSplittingFrequency = [0.0, 12.0]
		prior2 = np.array([amplitude, linewidth, projectedSplittingFrequency, centralFrequency])
		if mode_l >= 1:
			return prior1, prior2
		else:
			return prior1, prior1
	else:
		return prior1


def GuessBestLorentzianModelForPeakbagging(mode_freq, mode_l, freq, power, powers, dnu, ifReturnSplitModelPrior = False):
	dnu02 = 0.122*dnu + 0.05 # Bedding+2011 low luminosity RGB
	index = np.intersect1d(np.where(freq > mode_freq - 0.6*dnu02)[0],np.where(freq < mode_freq + 0.6*dnu02)[0])
	power = power[index]
	powers = powers[index]
	centralFrequency = mode_freq

	amplitude = np.max(powers)**0.5 * 2.0
	linewidth = 0.01*dnu

	prior1 = np.array([amplitude, linewidth, centralFrequency])

	if ifReturnSplitModelPrior:
		projectedSplittingFrequency = 0.1
		prior2 = np.array([amplitude, linewidth, projectedSplittingFrequency, centralFrequency])
		if mode_l >= 1:
			return prior1, prior2
		else:
			return prior1, prior1
	else:
		return prior1

def lnprior_m1(theta, n_mode, n_mode_l0, flatPriorGuess_split):
	pointer = True
	for j in range(0, 3*n_mode_l0 + (n_mode - n_mode_l0)*4):
		if not flatPriorGuess_split[j][0] <= theta[j] <= flatPriorGuess_split[j][1]:
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
				lnfspriorbase = np.log(1.0/(flatPriorGuess_split[j][1] - flatPriorGuess_split[j][0]))
		return lnfspriorbase + lnfsprior
	else:
		return -np.inf

def lnlikelihood_m1(theta, freq, power, inclination, fnyq, mode_l, n_mode, n_mode_l0):
	model = np.zeros(len(freq))
	for j in range(0, n_mode_l0):
		model += LorentzianSplittingMixtureModel(freq, [theta[3*j], theta[3*j+1], 0.0, 
						theta[3*j+2], inclination], fnyq, 0)
	for j in range(0, n_mode - n_mode_l0):
		model += LorentzianSplittingMixtureModel(freq, [theta[3*n_mode_l0+4*j], theta[3*n_mode_l0+4*j+1], 
						theta[3*n_mode_l0+4*j+2], theta[3*n_mode_l0+4*j+3], inclination], fnyq, mode_l[n_mode_l0+j])
	model += 1.0
	return -np.sum(np.log(model) + power/model)


def lnpost_m1(theta, n_mode, n_mode_l0, flatPriorGuess_split, freq, power, inclination, fnyq, mode_l):
	lp = lnprior_m1(theta, n_mode, n_mode_l0, flatPriorGuess_split)
	if not np.isfinite(lp):
		return -np.inf
	else:
		return lp + lnlikelihood_m1(theta, freq, power, inclination, fnyq, mode_l, n_mode, n_mode_l0)

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
	ifoutputsamples: bool=False, pthreads: int=1):
	'''
	Provide a wrapper to fit mode defined in mode_freq

	Input:
	dnu
	inclination: in rad
	fnyq: nyquist frequency, in muHz
	mode_freq: guessed mode frequencies, in muHz
	mode_l: 0, 1, 2 or 3
	freq: frequencies of the power spectrum, in muHz
	power: backgroud divided power spectrum, S/N
	powers: smoothed power, to predict amplitude
	filepath: path to store output files
	fittype: one of ["ParallelTempering", "Ensemble", "LeastSquare"]
	pthreads: the number of threads to use in parallel computing

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


	# initilize
	n_mode = len(mode_l)
	n_mode_l0 = len( np.where(mode_l == 0 )[0] )

	# trim data into range we use
	index = np.all(np.array([freq >= np.min(mode_freq)-8.0, freq <= np.max(mode_freq)+8.0]), axis=0)
	freq = freq[index]
	power = power[index]
	powers = powers[index]

	dnu02 = 0.122*dnu + 0.05 # Bedding+2011 low luminosity RGB
	index = np.all(np.array([freq >= np.min(mode_freq)-dnu02*0.6, freq <= np.max(mode_freq)+dnu02*0.6]), axis=0)
	tfreq = freq[index]
	tpower = power[index]
	tpowers = powers[index]

	# defining likelihood and prior
	# model 1 - splitting model
	flatPriorGuess_split = []
	para_guess = np.array([])

	for j in range(n_mode):
		para_prior1, para_prior2 = GuessLorentzianModelPriorForPeakbagging(mode_freq[j], mode_l[j], tfreq, tpower, tpowers, dnu, True)
		para_guess1, para_guess2 = GuessBestLorentzianModelForPeakbagging(mode_freq[j], mode_l[j], tfreq, tpower, tpowers, dnu, True)
		for k in range(len(para_prior2)):
			flatPriorGuess_split.append(para_prior2[k])
			para_guess = np.append(para_guess, para_guess2[k])

	
	# write guessed parameters
	para_best = para_guess
	np.savetxt(filepath+"guess.txt", para_best, delimiter=",", fmt=("%0.8f"), header="para_guess")

	if fittype in ["ParallelTempering", "Ensemble"]:

		if fittype == "ParallelTempering":

			# run mcmc with pt sampler
			ndim, nwalkers, ntemps = n_mode_l0 * 3 + (n_mode - n_mode_l0 ) * 4, 100, 20
			print("enabling ParallelTempering sampler.")
			print("ndimension: ", ndim, ", nwalkers: ", nwalkers, ", ntemps: ", ntemps)
			pos0 = [[para_best + 1.0e-8*np.random.randn(ndim) for j in range(nwalkers)] for k in range(ntemps)]
			loglargs = [tfreq, tpower, inclination, fnyq, mode_l, n_mode, n_mode_l0]
			logpargs = [n_mode, n_mode_l0, flatPriorGuess_split]
			sampler = emcee.PTSampler(ntemps, nwalkers, ndim, lnlikelihood_m1, lnprior_m1, loglargs=loglargs, logpargs=logpargs, threads=pthreads)

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
			nsteps, width = 2000, 30 # 10000, 30
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
			pos0 = [para_best + 1.0e-8*np.random.randn(ndim) for j in range(nwalkers)]
			args = [n_mode, n_mode_l0, flatPriorGuess_split, tfreq, tpower, inclination, fnyq, mode_l]
			sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost_m1, args=args, threads=pthreads)

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
			nsteps, width = 2000, 30 # 10000, 30
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

		# plot triangle and save
		tp = ["amp", "lw", "fc"]
		para1 = [tp[k]+str(j) for j in range(n_mode_l0) for k in range(3)]
		tp = ["amp", "lw", "fs", "fc"]
		para2 = [tp[k]+str(j) for j in range(n_mode_l0, n_mode) for k in range(4)]
		para = para1 + para2
		fig = corner.corner(samples, labels=para, quantiles=(0.16, 0.5, 0.84), truths=para_best)
		fig.savefig(filepath+st+"triangle.png")
		plt.close()

		# save estimation result
		result = np.array(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
			zip(*np.percentile(samples, [16, 50, 84],axis=0)))))
		np.savetxt(filepath+st+"summary.txt", result, delimiter=",", fmt=("%0.8f", "%0.8f", "%0.8f"), header="parameter, upper uncertainty, lower uncertainty")

		# plot fitting result and save
		para_plot = result[:,0]
		power_fit = np.zeros(len(freq))
		for j in range(0, n_mode_l0):
			power_fit += LorentzianSplittingMixtureModel(freq, [para_plot[3*j], para_plot[3*j+1], 0.0, 
								para_plot[3*j+2], inclination], fnyq, 0)
		for j in range(0, n_mode - n_mode_l0):
			power_fit += LorentzianSplittingMixtureModel(freq, [para_plot[3*n_mode_l0+4*j], para_plot[3*n_mode_l0+4*j+1], 
							para_plot[3*n_mode_l0+4*j+2], para_plot[3*n_mode_l0+4*j+3], inclination], fnyq, mode_l[n_mode_l0+j])
		power_fit += 1.0
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
		plt.savefig(filepath+st+"fit.png")
		plt.close()

		# save mean acceptance rate
		acceptance_fraction = np.array([np.mean(sampler.acceptance_fraction)])
		print("Mean acceptance fraction: {:0.3f}".format(acceptance_fraction[0]))
		np.savetxt(filepath+st+"acceptance_fraction.txt", acceptance_fraction, delimiter=",", fmt=("%0.8f"), header="acceptance_fraction")

		# plot traces and save
		fig = plt.figure(figsize=(5,ndim*3))
		for i in range(ndim):
			ax1=fig.add_subplot(ndim,1,i+1)
			ax1.plot(np.arange(samples.shape[0]), samples[:,i], color="black", lw=1, zorder=1)
			ax1.set_ylabel(para[i])
		plt.savefig(filepath+st+'traces.png')
		plt.close()

	if fittype == "LeastSquare":
		
		# maximize likelihood function by scipy.optimize.minimize function
		function = lambda *arg: -lnlikelihood_m1(*arg)
		args = (tfreq, tpower, inclination, fnyq, mode_l, n_mode, n_mode_l0)
		result = minimize(function, para_guess, args=args, bounds=flatPriorGuess_split)

		# save estimation result
		np.savetxt(filepath+"LSsummary.txt", result.x, delimiter=",", fmt=("%0.8f"), header="parameter")

		# plot fitting result and save
		para_plot = result.x
		power_fit = np.zeros(len(freq))
		for j in range(0, n_mode_l0):
			power_fit += LorentzianSplittingMixtureModel(freq, [para_plot[3*j], para_plot[3*j+1], 0.0, 
								para_plot[3*j+2], inclination], fnyq, 0)
		for j in range(0, n_mode - n_mode_l0):
			power_fit += LorentzianSplittingMixtureModel(freq, [para_plot[3*n_mode_l0+4*j], para_plot[3*n_mode_l0+4*j+1], 
							para_plot[3*n_mode_l0+4*j+2], para_plot[3*n_mode_l0+4*j+3], inclination], fnyq, mode_l[n_mode_l0+j])
		power_fit += 1.0
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
		plt.savefig(filepath+"LSfit.png")
		plt.close()

	return

def h1testWrapper(dnu: float, fnyq: float, mode_freq: np.array, mode_l: np.array,
	freq: np.array, power: np.array, powers: np.array, filepath: str, pthreads: int=1):
	'''
	Provide a wrapper to fit mode defined in mode_freq

	Input:
	dnu
	inclination: in rad
	fnyq: nyquist frequency, in muHz
	mode_freq: guessed mode frequencies, in muHz
	mode_l: 0, 1, 2 or 3
	freq: frequencies of the power spectrum, in muHz
	power: backgroud divided power spectrum, S/N
	powers: smoothed power, to predict amplitude
	filepath: path to store output files
	pthreads: the number of threads to use in parallel computing.

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
		index = np.all(np.array([freq >= np.min(mode_freq)-dnu02*0.6, freq <= np.max(mode_freq)+dnu02*0.6]), axis=0)
		tfreq, tpower = freq[index], power[index]
		iden = str(i)

		# defining likelihood and prior
		# model 0 - nothing but a straight line

		# write guessed parameters
		para_best = np.array([tpower.mean()])
		np.savetxt(filepath+"guess_h1_"+iden+".txt", para_best, delimiter=",", fmt=("%0.8f"), header="para_guess")

		# run mcmc with pt sampler
		ndim, nwalkers, ntemps = 1, 100, 20
		print("ndimension: ", ndim, ", nwalkers: ", nwalkers, ", ntemps: ", ntemps)
		pos0 = [[para_best + 1.0e-8*np.random.randn(ndim) for j in range(nwalkers)] for k in range(ntemps)]
		loglargs = [tfreq, tpower, fnyq]
		logpargs = [tpower]
		sampler = emcee.PTSampler(ntemps, nwalkers, ndim, lnlikelihood_m0, lnprior_m0, loglargs=loglargs, logpargs=logpargs, threads=pthreads)

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
		fig = corner.corner(samples, labels=para, quantiles=(0.16, 0.5, 0.84), truths=para_best)
		fig.savefig(filepath+"triangle_h1_"+iden+".png")
		plt.close()

		# save estimation result
		result = np.array(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
			zip(*np.percentile(samples, [16, 50, 84],axis=0)))))
		np.savetxt(filepath+"summary_h1_"+iden+".txt", result, delimiter=",", fmt=("%0.8f", "%0.8f", "%0.8f"), header="parameter, upper uncertainty, lower uncertainty")

		# plot fitting result and save
		para_plot = result[:,0]
		power_fit = np.zeros(len(freq)) + para_plot[0]
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
