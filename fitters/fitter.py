
#!/usr/bin/env/ python
# coding: utf-8

import numpy as np
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

# customized
from scipy.optimize import minimize, basinhopping
import emcee
import corner
import sys 


__all__ = ["ESSampler", "LSSampler"]

def _return_2dmap_axes(NSquareBlocks):

    # Some magic numbers for pretty axis layout.
    # stole from corner
    Kx = int(np.ceil(NSquareBlocks**0.5))
    Ky = Kx if (Kx**2-NSquareBlocks) < Kx else Kx-1

    factor = 2.0           # size of one side of one panel
    lbdim = 0.4 * factor   # size of left/bottom margin, default=0.2
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.30         # w/hspace size
    plotdimx = factor * Kx + factor * (Kx - 1.) * whspace
    plotdimy = factor * Ky + factor * (Ky - 1.) * whspace
    dimx = lbdim + plotdimx + trdim
    dimy = lbdim + plotdimy + trdim

    # Create a new figure if one wasn't provided.
    fig, axes = plt.subplots(Ky, Kx, figsize=(dimx, dimy), squeeze=False)

    # Format the figure.
    l = lbdim / dimx
    b = lbdim / dimy
    t = (lbdim + plotdimy) / dimy
    r = (lbdim + plotdimx) / dimx
    fig.subplots_adjust(left=l, bottom=b, right=r, top=t,
                        wspace=whspace, hspace=whspace)
    axes = np.concatenate(axes)

    return fig, axes

def _plot_mcmc_traces(Ndim, samples, paramsNames):

    fig, axes = _return_2dmap_axes(Ndim)

    for i in range(Ndim):
        ax = axes[i]
        evol = samples[:,i]
        Npoints = samples.shape[0]
        ax.plot(np.arange(Npoints)/Npoints, evol, color="gray", lw=1, zorder=1)
        Nseries = int(len(evol)/15.0)
        evol_median = np.array([np.median(evol[i*Nseries:(i+1)*Nseries]) for i in range(0,15)])
        evol_std = np.array([np.std(evol[i*Nseries:(i+1)*Nseries]) for i in range(0,15)])
        evol_x = np.array([np.median(np.arange(Npoints)[i*Nseries:(i+1)*Nseries]/Npoints) for i in range(0,15)])
        ax.errorbar(evol_x, evol_median, yerr=evol_std, color="C0", ecolor="C0", capsize=2)
        ax.set_ylabel(paramsNames[i])

    for ax in axes[i+1:]:
        fig.delaxes(ax)

    return fig



class ESSampler:
    def __init__(self, paramsInit, posterior, 
                    Nsteps=2000, Nburn=1000, Nwalkers=100,
                    paramsNames=None):
        self.paramsInit = paramsInit
        self.posterior = posterior

        self.Nburn=Nburn
        self.Nsteps=Nsteps
        self.Nwalkers=Nwalkers
        self.Ndim=len(paramsInit)
        if paramsNames is None:
            self.paramsNames = ['c{:0.0f}'.format(i) for i in range(self.Ndim)]
        else: 
            self.paramsNames = paramsNames
        return

    def _display_bar(self, j, Nburn, width=30):
        n = int((width+1) * float(j) / Nburn)
        sys.stdout.write("\r[{0}{1}]".format('#' * n, ' ' * (width - n)))
        return

    def run(self, verbose=False):
        # run mcmc with ensemble sampler
        if verbose: print("enabling Ensemble sampler.")
        pos0 = [self.paramsInit + 1.0e-8*np.random.randn(self.Ndim) for j in range(self.Nwalkers)]
        sampler = emcee.EnsembleSampler(self.Nwalkers, self.Ndim, self.posterior)

        # burn-in
        if verbose: print("start burning in. Nburn:", self.Nburn)
        for j, result in enumerate(sampler.sample(pos0, iterations=self.Nburn, thin=10)):
            if verbose: self._display_bar(j, self.Nburn)
        if verbose: sys.stdout.write("\n")
        pos, _, _ = result
        sampler.reset()

        # actual iteration
        if verbose: print("start iterating. Nsteps:", self.Nsteps)
        for j, result in enumerate(sampler.sample(pos, iterations=self.Nsteps)):
            if verbose: self._display_bar(j, self.Nsteps)
        if verbose: sys.stdout.write("\n")

        # modify samples
        self.samples = sampler.chain[:,:,:].reshape((-1,self.Ndim))

        # save estimation result
        # 16, 50, 84 quantiles
        result = np.array(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
            zip(*np.percentile(self.samples, [16, 50, 84],axis=0)))))
        self.paramsMedian = result[:,0]

        # maximum
        paramsMax = np.zeros(self.Ndim)
        for ipara in range(self.Ndim):
            n, bins, _ = plt.hist(self.samples[:,ipara], bins=80)
            idx = np.where(n == n.max())[0][0]
            paramsMax[ipara] = bins[idx:idx+1].mean()
        self.paramsMax = paramsMax

        self.result = np.concatenate([result, paramsMax.reshape(self.Ndim,1)], axis=1)

        # save acceptance fraction
        self.acceptanceFraction = np.mean(sampler.acceptance_fraction)

        self.diagnostics = {'paramsInit':self.paramsInit,
                        'paramsMedian':self.paramsMedian,
                        'paramsMax':self.paramsMax,
                        'paramsNames':self.paramsNames,
                        'result':self.result,
                        'acceptanceFraction':self.acceptanceFraction,
                        'samples':self.samples,
                        'Ndim':self.Ndim, 'Nwalkers':self.Nwalkers,
                        'Nburn':self.Nburn, 'Nsteps':self.Nsteps,
                        'posterior':self.posterior}
        return self.diagnostics

    def to_data(self, filepath):
        np.save(filepath+'data', self.diagnostics)
        return

    def to_plot(self, filepath):
        # plot triangle and save
        fig = corner.corner(self.samples, labels=self.paramsNames, quantiles=(0.16, 0.5, 0.84), truths=self.paramsMax)
        fig.savefig(filepath+"triangle.png")
        plt.close()

        # plot traces and save
        fig = _plot_mcmc_traces(self.Ndim, self.samples, self.paramsNames)
        plt.savefig(filepath+'traces.png')
        plt.close()
        return


class LSSampler:
    def __init__(self, chi2, paramsInit, paramsBounds,
                    paramsNames=None):
        self.paramsInit = paramsInit
        self.chi2 = chi2
        self.paramsBounds = paramsBounds 

        self.Ndim=len(paramsInit)
        if paramsNames is None:
            self.paramsNames = ['c{:0.0f}'.format(i) for i in range(self.Ndim)]
        else: 
            self.paramsNames = paramsNames
        return

    def run(self, wrapper='basinhopping'):
        # maximize likelihood function by scipy.optimize.minimize function
        minimizer_kwargs={"bounds":self.paramsBounds}
        if wrapper is 'minimize':
            self.result = minimize(self.chi2, self.paramsInit, **minimizer_kwargs)
        if wrapper is 'basinhopping':
            self.result = basinhopping(self.chi2, self.paramsInit, minimizer_kwargs=minimizer_kwargs)
        self.paramsMax = self.result.x
        self.diagnostics = {'paramsInit':self.paramsInit,
                        'paramsMax':self.paramsMax,
                        'paramsBounds':self.paramsBounds,
                        'paramsNames':self.paramsNames,
                        'result':self.result,
                        'Ndim':self.Ndim}
        return self.diagnostics

    def to_data(self, filepath):
        np.save(filepath+'data', self.diagnostics)
        return
