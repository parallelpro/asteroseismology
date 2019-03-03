#!/usr/bin/env/ ipython
# coding: utf-8


import numpy as np

__all__ = ["modefitWrapper"]

def modefitWrapper(filepath: str, ):


	si = str(i)

	# check
	if not os.path.exists(filepath):
		os.mkdir(filepath)

	

	# initilize
	index = np.where(mode_group == i)[0]
	tmode_l = mode_l[index]
	tmode_freq = mode_freq[index]
	tmode_id = mode_id[index]
	tn_mode = len(tmode_l)
	tn_mode_l0 = len( np.where(tmode_l == 0 )[0] )

	# trim data into range we use
	index = np.intersect1d(np.where(freq > np.min(tmode_freq) - 8.0)[0],
		np.where(freq < np.max(tmode_freq) + 8.0)[0])
	tfreq = freq[index]
	tpower = power_o[index]
	tpower_smooth = powers_o[index]
	tpower_bg = power_bg[index]

	# defining likelihood and prior
	# model 1 - splitting model
	flatPriorGuess_split = []
	para_guess = np.array([])

	for j in range(tn_mode):
		para_prior1, para_prior2 = GuessLorentzianModelPriorForPeakbagging(tfreq, tpower, tpower_smooth, tmode_freq[j], tmode_l[j], True)
		para_guess1, para_guess2 = GuessBestLorentzianModelForPeakbagging(tfreq, tpower, tpower_smooth, tmode_freq[j], tmode_l[j], True)
		for j in range(len(para_prior2)):
			flatPriorGuess_split.append(para_prior2[j])
			para_guess = np.append(para_guess, para_guess2[j])

	def lnprior(theta):
		pointer = True
		for j in range(0, 3*tn_mode_l0 + (tn_mode - tn_mode_l0)*4):
			if not flatPriorGuess_split[j][0] <= theta[j] <= flatPriorGuess_split[j][1]:
				pointer = False
		if pointer == True:
			if tn_mode - tn_mode_l0 > 0:
				S, U, sigma = 0.0, 2.0, 1.0
				H = 1.0/(U + (2*np.pi)**0.5 * sigma/2.0 - S)
				for i in range(tn_mode - tn_mode_l0):
					if theta[3*tn_mode_l0 + 4*i + 2] < U:
						lnfsprior = np.log(H)
					else:
						lnfsprior = np.log(H) - (theta[3*tn_mode_l0 + 4*i + 2]-U)**2.0/(2*sigma**2.0)
			else:
				lnfsprior = 0.0
			return 0.0 + lnfsprior
		else:
			return -np.inf

	def lnlikelihood(theta):
		model = np.zeros(len(tfreq))
		for j in range(0, tn_mode_l0):
			model += LorentzianSplittingMixtureModel(tfreq, [theta[3*j], theta[3*j+1], 0.0, 
							theta[3*j+2], inclination], 8496.35, 0)
		for j in range(0, tn_mode - tn_mode_l0):
			model += LorentzianSplittingMixtureModel(tfreq, [theta[3*tn_mode_l0+4*j], theta[3*tn_mode_l0+4*j+1], 
							theta[3*tn_mode_l0+4*j+2], theta[3*tn_mode_l0+4*j+3], inclination], 8496.35, tmode_l[tn_mode_l0+j])
		model += tpower_bg
		return -np.sum(np.log(model) + tpower/model)

	def lnprob(theta):
		lp = lnprior(theta)
		if not np.isfinite(lp):
			return -np.inf
		return lp + lnlikelihood(theta)

	
	# maximum likelihood estimation'
	nll = lambda *args: -lnlikelihood(*args)
	result = op.minimize(nll, para_guess)
	para_best = result["x"]

	# if one of the parameter is out of range, use the guessed one
	for j in range(len(para_best)):
		if not flatPriorGuess_split[j][0] <= para_best[j] <= flatPriorGuess_split[j][1]:
			para_best[j] = para_guess[j]

	##########################################
	#para_best[-2] = 0.01
	#para_best[-1] = 539.2
	
	#para_best = np.array([22.0,0.547,597.325,
	#			14.0,0.62699,0.0643,593.818])#,
				#1.26, 0.23, 0.1, 604.00])
	
	
	##########################################
	ascii.write(Table([para_best]), filepath+'guess.txt', format='no_header', delimiter=',', overwrite=True)

	
	# run mcmc with ensemble sampler
	ndim, nwalkers = tn_mode_l0 * 3 + (tn_mode - tn_mode_l0 ) * 4, 500 #100
	pos = [para_best + 1.0*np.random.randn(ndim) for j in range(nwalkers)]
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=1)
	nsteps = 2000
	width = 30
	for j, result in enumerate(sampler.sample(pos, iterations=nsteps)):
	    n = int((width+1) * float(j) / nsteps)
	    sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
	sys.stdout.write("\n")
	'''

	# run mcmc with parallel tempering
	ndim, nwalkers, ntemps = tn_mode_l0 * 3 + (tn_mode - tn_mode_l0 ) * 4, 100, 20 #100
	sampler = emcee.PTSampler(ntemps, nwalkers, ndim, lnlikelihood, lnprior)

	p0 = np.random.uniform(low=-1.0, high=1.0, size=(ntemps, nwalkers, ndim))
	for p, lnprob, lnlike in sampler.sample(p0, iterations=1000):
		pass
	sampler.reset()

	for p, lnprob, lnlike in sampler.sample(p, lnprob0=lnprob,
												lnlike0=lnlike,
												iterations=10000, thin=10):
		pass

	assert sampler.chain.shape == (ntemps, nwalkers, 1000, ndim)
	'''

	# modify samples
	samples = sampler.chain[:,100:,:].reshape((-1,ndim))
	#samples[:,-2] = np.pi/2.0 - np.abs(np.pi/2.0 - samples[:,-2])


	# plot triangle
	tp = ['amp', 'lw', 'fc']
	para1 = [tp[k]+str(j) for j in range(tn_mode_l0) for k in range(3)]
	tp = ['amp', 'lw', 'fs', 'fc']
	para2 = [tp[k]+str(j) for j in range(tn_mode_l0, tn_mode) for k in range(4)]
	para = para1 + para2
	#ascii.write(Table(samples), filepath+'samples.txt', format='csv', overwrite=True)
	fig = corner.corner(samples, labels=para, quantiles=(0.16, 0.5, 0.84), truths=para_best)
	fig.savefig(filepath+'triangle.png')
	plt.close()

	result = np.array(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
		zip(*np.percentile(samples, [16, 50, 84],axis=0)))))
	ascii.write(Table(result), filepath+'summary.txt', format='csv', overwrite=True)

	# plot mle result
	para_plot = result[:,0]
	tpower_fit = np.zeros(len(tfreq))
	for j in range(0, tn_mode_l0):
		tpower_fit += LorentzianSplittingMixtureModel(tfreq, [para_plot[3*j], para_plot[3*j+1], 0.0, 
						para_plot[3*j+2], inclination], 8496.35, 0)
	for j in range(0, tn_mode - tn_mode_l0):
		tpower_fit += LorentzianSplittingMixtureModel(tfreq, [para_plot[3*tn_mode_l0+4*j], para_plot[3*tn_mode_l0+4*j+1], 
						para_plot[3*tn_mode_l0+4*j+2], para_plot[3*tn_mode_l0+4*j+3], inclination], 8496.35, tmode_l[tn_mode_l0+j])
	tpower_fit += tpower_bg
	fig = plt.figure(figsize=(10,7))
	ax = fig.add_subplot(1,1,1)
	ax.plot(tfreq, tpower, color='lightgray', label='power')
	ax.plot(tfreq, tpower_smooth, color='black', label='1.0 smooth')
	ax.plot(tfreq, tpower_bg, color='black', linestyle='--', label='bg')
	ax.plot(tfreq, tpower_fit, color='orange', label='fit')
	a, b = np.min(tmode_freq) - 8.0, np.max(tmode_freq) + 8.0
	index = np.intersect1d(np.where(tfreq > a)[0],
		np.where(tfreq < b)[0] )
	c, d = np.min(tpower[index]), np.max(tpower[index])
	color = ['blue', 'red', 'green', 'purple']
	marker = ['o', '^', 's', 'v']
	for j in range(tn_mode):
		ax.scatter([tmode_freq[j]],[c+(d-c)*0.8], c=color[tmode_l[j]], marker=marker[tmode_l[j]])

	ax.axis([a, b, c, d])
	plt.savefig(filepath+'fit.png')
	plt.close()

	# mean acceptance rate
	print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
	ascii.write(Table([np.array([np.mean(sampler.acceptance_fraction)])]), filepath+'acceptance_fraction.txt', format='csv', overwrite=True)

	# plot traces
	fig = plt.figure(figsize=(5,ndim*3))
	for i in range(ndim):
		ax1=fig.add_subplot(ndim,1,i+1)
		ax1.plot(np.arange(samples.shape[0]), samples[:,i], color='black', lw=1, zorder=1)
		ax1.set_ylabel(para[i])
	plt.savefig(filepath+'traces.png')
	plt.close()

	print('Done for group '+si)

	return