#!/usr/bin/env/ python
# coding: utf-8


import numpy as np
import scipy
from scipy.optimize import minimize
import emcee
import sys
import corner
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool

__all__ = ["mfr"]

def mrf_minus_corr(theta, x, y, model, args=None, kwargs=None):
	ymodel = model(theta, x, *args, **kwargs)
	autocorr = scipy.signal.fftconvolve(y, ymodel, mode='full') 
	return -np.mean(np.abs(autocorr))

def mrf_corr(theta, x, y, model, args=None, kwargs=None):
	ymodel = model(theta, x, *args, **kwargs)
	autocorr = scipy.signal.fftconvolve(y, ymodel, mode='full') 
	return np.mean(autocorr)

def mrf_lnprior(theta, flatpriors):
	ndim = len(flatpriors)
	for idim in range(ndim):
		if flatpriors[idim][0] <= theta[idim] < flatpriors[idim][1]:
			return 0.0
		else:
			return -np.inf

def mrf_lnpost(theta, x, y, model, flatpriors, args, kwargs):
	# lp = mrf_prior(theta, flatpriors)
	# if not np.isfinite(lp): 
	# 	return -np.inf
	# else:
	# 	return lp + mrf_likelihood(theta, x, y, model, args=args, kwargs=kwargs)
	ndim = len(theta)
	pointer = True
	for idim in range(ndim):
		if not flatpriors[idim][0] < theta[idim] < flatpriors[idim][1]:
			pointer = False

	if pointer:
		return mrf_lnlikelihood(theta, x, y, model, args, kwargs)
	else:
		return -np.inf	


def mrf_lnlikelihood(theta, x, y, model, args, kwargs):
	ymodel = model(theta, x, *args, **kwargs)
	return np.log((np.sum(ymodel*y)/np.sum(ymodel)))*10.0

def mfr(x, y, model, para_guess, bounds, filepath, para_name, args=None, kwargs=None):
	'''
	Matched filter response.

	Input:
	x: np.array
		the independent variable of the time series.

	y: np.array
		the dependent variable of the time series.

	model: function
		passes the parameters and returns a series.

	para_guess: array-like[Ntheta,]

	bounds: array-like[Ntheta,2]

	para_name: array-like[Ntheta,]

	filepath: str

	Optional input:
	
	*args, **kwargs:
		any paramters paramters passed to ``model''.

	Output:
	

	'''

	# run mcmc with ensemble sampler
	ndim, nwalkers = len(para_guess), 100
	print("enabling Ensemble sampler.")
	print("ndimension: ", ndim, ", nwalkers: ", nwalkers)
	pos0 = [para_guess + 1.0e-8*np.random.randn(ndim) for j in range(nwalkers)]

	with Pool() as pool:
		sampler = emcee.EnsembleSampler(nwalkers, ndim, mrf_lnpost, 
			args=(x, y, model, bounds, args, kwargs), pool=pool)

		# burn-in
		nburn, width = 500, 60
		print("start burning in. nburn:", nburn)
		for j, result in enumerate(sampler.sample(pos0, iterations=nburn, thin=10)):
			n = int((width+1) * float(j) / nburn)
			sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
		sys.stdout.write("\n")
		pos, lnpost, rstate = result
		sampler.reset()

		# actual iteration
		nsteps = 1000
		width = 60 # 10000, 30
		print("start iterating. nsteps:", nsteps)
		for j, result in enumerate(sampler.sample(pos, iterations=nsteps, lnprob0=lnpost)):
			#pos, lnpost, rstate = result
			n = int((width+1) * float(j) / nsteps)
			sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
		sys.stdout.write("\n")

	# modify samples
	samples = sampler.chain[:,:,:].reshape((-1,ndim))
	np.save(filepath+"samples.npy", samples)
	
	st="ES"
	# save guessed parameters
	np.savetxt(filepath+st+"guess.txt", para_guess, delimiter=",", fmt=("%0.8f"), header="para_guess")

	# plot triangle and save
	para = para_name
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


	# # find the parameters that best reproduce y.
	# result = minimize(mrf_minus_corr, para_guess, 
	# 	args=(x, y, model, args, kwargs), bounds=bounds)

	# # # save guessed parameters
	# # np.savetxt(filepath+st+"guess.txt", para_guess, delimiter=",", fmt=("%0.8f"), header="para_guess")

	# # save estimation result
	# # np.savetxt(filepath+st+"summary.txt", result.x, delimiter=",", fmt=("%0.8f"), header="parameter")
	# para_fit = result.x

	return para_fit

