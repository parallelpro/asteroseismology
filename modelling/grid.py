import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.io import ascii
from astropy.table import Table, Column
import corner
import h5py

import multiprocessing
from functools import partial
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.special import logsumexp

from ..tools import return_2dmap_axes, quantile


__all__ = ['grid']


class grid:
    """
    
    Estimate stellar parameters from evolutionary grid models.

    """

    def __init__(self, read_models, tracks, estimates, outdir, starname, 
                colAge='star_age'):

        """
        Initialize a parameter estimation class.

        ----------
        Input:
        read_models: function reference
            The function takes one argument 'atrack' which is the path of
            a track, and returns a structured array, or a table-like object 
            (like the one defined in 'astropy'). The track column of an 
            observable/estimate 'i' defined in self.setup should be able 
            to be called with 'atrack[i]'.
        tracks: array-like[Ntrack,]
            A list containing the paths of all tracks.
        estimates: array-like[Nestimate,]
            The parameter names to estimate. Make sure each string is 
            callable from the returning table of the 'read_models function.
        outdir: str
            The directory to store output figures and tables.
        starname: list of str
            The starnames to create folders.

        ----------
        Optional input:
        colAge: str, default: 'star_age' as in MESA
            The age column in a track can be called using 'atrack[colAge].
        #colLogLum: str, default: 'log_L' as in MESA
        #colLogTeff: str, default: 'log_Teff' as in MESA
        """

        self.read_models = read_models
        self.tracks = tracks
        self.estimates = estimates
        self.Nestimate = len(self.estimates)
        if not os.path.exists(outdir): os.mkdir(outdir)
        self.outdir = outdir
        self.starname = starname
        self.colAge = colAge
        # self.colLogLum = colLogLum
        # self.colLogTeff = colLogTeff

        self.ifSetup = False
        self.ifSetupSeismology = False
        return 


    def setup(self, observables, stars_obs, stars_obserr):
        """
        Setup observables (to construct the chi2) and estimates (to estimate 
        parameters). 

        ----------
        Input:
        observables: array-like[Nobservable,]
            The observable names which are matched between models and observations.
            Make sure each string is callable from the returning table of the
            'read_models' function.
        stars_obs: array-like[Nstar, Nobservable]
            The array which stores all observed values.
        stars_obserr: array-like[Nstar, Nobservable]
            The array which stores all errors.

        """

        self.observables = observables
        self.stars_obs = stars_obs
        self.stars_obserr = stars_obserr

        self.ifSetup = True
        return self


    def setup_seismology(self, obs_freq, obs_efreq, obs_l, 
            colModeFreq='mode_freq', colModeDegree='mode_l', colModeInertia='mode_inertia',
            colAcFreq='acoustic_cutoff', weight_nonseis=1, weight_seis=1, ifCorrectSurface=True,
            obs_delta_nu=None, surface_correction_formula='cubic'):
        """
        Setup the matching of oscillation frequencies (to construct the chi2_seismo)
        In order to setup, you should make sure that 'atrack[colModeFreq]' returns a list  
        of arrays, each of which contains mode frequencies of that model. Same for 
        'atrack[colModeDegree]'.

        ----------
        Input:
        obs_freq: array-like[Nstar, Nmode] 
            The observed frequencies of each star.
            The number of modes of each star can be different.

        obs_efreq: array-like[Nstar, Nmode] 
            The observed frequency errors of each star.

        obs_l: array-like[Nstar, Nmode] 
            The mode degree of each star.

        ----------
        Optional input:
        colModeFreq: str, default 'mode_freq'
            The mode frequency column name retrived from atrack.

        colModeModeDegree: str, default 'mode_l'
            The mode degree column name retrived from atrack.    

        colModeInertia: str, default 'colModeInertia'
            The mode inertia column name retrived from atrack.  

        colAcFreq: str, default 'colAcFreq'
            The acoustic cutoff frequency column name retrived from atrack.  

        ifCorrectSurface: bool, default True
            if True, then correct model frequencies using the formula 
            of Ball & Gizon (2014) (the inverse and cubic term).
            Should extend more capabilities here!

        obs_delta_nu: float, [muHz]
            the large speration to make echelle plot for the best model.
            if not set, then the code estimates it with frequencies.
        """

        self.obs_freq = obs_freq
        self.obs_efreq = obs_efreq
        self.obs_l = obs_l
        self.colModeFreq = colModeFreq
        self.colModeDegree = colModeDegree
        self.colModeInertia = colModeInertia
        self.colAcFreq = colAcFreq
        self.weight_nonseis = weight_nonseis
        self.weight_seis = weight_seis
        self.ifCorrectSurface = ifCorrectSurface
        self.obs_delta_nu = obs_delta_nu
        self.surface_correction_formula = surface_correction_formula

        self.ifSetupSeismology=True
        return self


    def assign_n(self, obs_freq, obs_efreq, obs_l, mod_freq, mod_l, *modargs):
        # assign n_p or n_g based on the closenes of the frequencies
        new_obs_freq, new_obs_efreq, new_obs_l, new_mod_freq, new_mod_l = [np.array([]) for i in range(5)]
        # new_mod_freq = np.zeros(len(obs_freq)) 
        new_mod_args = [np.array([]) for i in range(len(modargs))]

        for l in np.sort(np.unique(obs_l)):
            obs_freq_l = obs_freq[obs_l==l]
            obs_efreq_l = obs_efreq[obs_l==l]
            mod_freq_l = mod_freq[mod_l==l]

            mod_args = [[] for iarg in range(len(modargs))]
            for iarg in range(len(modargs)):
                mod_args[iarg] = modargs[iarg][mod_l==l]

            # because we don't know n_p or n_g from observation (if we do that will save tons of effort here)
            # we need to assign each obs mode with a model mode
            # this can be seen as a linear sum assignment problem, also known as minimun weight matching in bipartite graphs
            cost = np.abs(obs_freq_l.reshape(-1,1) - mod_freq_l)
            row_ind, col_ind = linear_sum_assignment(cost)
            # obs_freq_l = obs_freq_l[row_ind]
            # obs_efreq_l = obs_efreq_l[row_ind]

            # mod_freq_l = mod_freq_l[col_ind]
            # mod_inertia_l = mod_inertia_l[col_ind]

            new_obs_freq = np.append(new_obs_freq, obs_freq_l[row_ind])
            new_obs_efreq = np.append(new_obs_efreq, obs_efreq_l[row_ind])
            new_obs_l = np.append(new_obs_l, obs_l[obs_l==l][row_ind])

            new_mod_freq = np.append(new_mod_freq, mod_freq_l[col_ind])
            new_mod_l = np.append(new_mod_l, mod_l[mod_l==l][col_ind])

            for iarg in range(len(modargs)):
                new_mod_args[iarg] = np.append(new_mod_args[iarg], mod_args[iarg][col_ind])

        return (new_obs_freq, new_obs_efreq, new_obs_l, new_mod_freq, new_mod_l, *new_mod_args)


    def get_surface_correction(self, obs_freq, obs_l, mod_freq, mod_l, mod_inertia, mod_acfreq, formula='cubic'):
        # formula is one of 'cubic', 'BG14'
        if not (formula in ['cubic', 'BG14']):
            raise ValueError('formula must be one of ``cubic`` and ``BG14``. ')

        if (np.sum(np.isin(mod_l, 0))) :
            # if correction is needed, first we use l=0 modes to derive correction factors
            # if obs_l don't have a 0, well I am not expecting this! 
            obs_freq_l0 = obs_freq[obs_l==0]
            mod_freq_l0 = mod_freq[mod_l==0]
            mod_inertia_l0 = mod_inertia[mod_l==0]

            # because we don't know n_p or n_g from observation (if we do that will save tons of effort here)
            # we need to assign each obs mode with a model mode
            # this can be seen as a linear sum assignment problem, also known as minimun weight matching in bipartite graphs
            cost = np.abs(obs_freq_l0.reshape(-1,1) - mod_freq_l0)
            row_ind, col_ind = linear_sum_assignment(cost)
            obs_freq_l0 = obs_freq_l0[row_ind]
            mod_freq_l0 = mod_freq_l0[col_ind]
            mod_inertia_l0 = mod_inertia_l0[col_ind]

            # regression
            b = obs_freq_l0-mod_freq_l0
            # # avoid selecting reversed models
            # if (np.abs(np.median(np.diff(np.sort(obs_freq_l0)))) > np.abs(np.median(np.diff(np.sort(mod_freq_l0)))) ) :
            #     return None

            if formula == 'BG14':
                A1 = (mod_freq_l0/mod_acfreq)**-1. / mod_inertia_l0
                A2 = (mod_freq_l0/mod_acfreq)**3. / mod_inertia_l0
                AT = np.array([A1, A2])
                A = AT.T
                b = b.reshape(-1,1)

                # apply corrections
                try:
                    coeff = np.dot(np.dot(np.linalg.inv(np.dot(AT,A)), AT), b)
                    delta_freq = (coeff[0]*(mod_freq/mod_acfreq)**-1.  + coeff[1]*(mod_freq/mod_acfreq)**3. ) / mod_inertia
                    mod_freq += delta_freq
                except:
                    print('An exception occurred when correcting surface effect.')
                    # pass

            if formula == 'cubic':
                A2 = (mod_freq_l0/mod_acfreq)**3. / mod_inertia_l0
                AT = np.array([A2])
                A = AT.T
                b = b.reshape(-1,1)

                # apply corrections
                try:
                    coeff = np.dot(np.dot(np.linalg.inv(np.dot(AT,A)), AT), b)
                    delta_freq = ( coeff[0]*(mod_freq/mod_acfreq)**3. ) / mod_inertia
                    mod_freq += delta_freq
                except:
                    print('An exception occurred when correcting surface effect.')
                    # pass

        return mod_freq


    def get_chi2_seismology(self, Nmodel, obs_freq, obs_efreq, obs_l, 
                mod_freq, mod_l, mod_inertia, mod_acfreq, 
                ifCorrectSurface=True):
        """
        Calculate chi2.
        ----------
        Input:
        Nmodel: int
            number of models
        obs_freq: array_like[Nmode_obs, ]
            observed mode frequency
        obs_efreq: array_like[Nmode_obs,]
            observed mode frequency
        obs_l: array_like[Nmode_obs, ]
            observed mode degree
        mod_freq: array_like[Nmodel, Nmode_mod]
            model's mode frequency
        mod_l: array_like[Nmodel, Nmode_mod]
            model's mode degree
        mod_inertia: array_like[Nmodel, Nmode_mod]
            model's mode inertia, used for surface correction
        mod_acfreq: array_like[Nmodel,]
            model's acoustic cutoff frequency, used for surface correction

        ----------
        Return:
        chi2: array_like[Nmodel, ]

        """

        chi2_seis = np.array([np.inf]*Nmodel)
        # chi2_best = np.inf
        for imod in range(Nmodel):
            if Nmodel == 1:
                tfreq, tl = mod_freq, mod_l
            else:
                tfreq, tl = mod_freq[imod], mod_l[imod]
        
            if ifCorrectSurface & (np.sum(np.isin(tl, 0))) :
                if Nmodel == 1:
                    tinertia, tacfreq = mod_inertia, mod_acfreq
                else:
                    tinertia, tacfreq = mod_inertia[imod], mod_acfreq[imod]

                tfreq = self.get_surface_correction(obs_freq, obs_l, tfreq, tl, tinertia, tacfreq, formula=self.surface_correction_formula)

            if (tfreq is None):
                chi2_seis[imod] = np.inf
            else:
                tobs_freq, tobs_efreq, _, tfreq, tl = self.assign_n(obs_freq, obs_efreq, obs_l, tfreq, tl)
                chi2_seis[imod] = np.mean((tobs_freq-tfreq)**2.0/(tobs_efreq**2.0))#/(Nobservable)

        return chi2_seis


    def get_chi2(self, obs, e_obs, mod):
        '''
        Calculate chi2.
        ----------
        Input:
        obs: array_like[Nobservable, ]
            observed value, e.g. Teff, luminosity, delta_nu, nu_max, logg, etc.
        e_obs: array_like[Nobservable, ]
            observed value errors, used for weighting
        mod: array_like[Nmodel, Nobservable]
            model's value

        ----------
        Return:
        chi2: array_like[Nmodel, ]

        '''
        # ndim = np.ndim(mod)
        # Nobservable = np.shape(mod)[-1]
            
        # chi2_nonseis = np.array([np.inf]*Nmodel)
        chi2_nonseis = np.sum((obs-mod)**2.0/(e_obs**2.0), axis=1)#/(Nobservable)

        return chi2_nonseis


    def get_chi2_combined(self, obs, e_obs, mod, 
                obs_freq, obs_efreq, obs_l, 
                mod_freq, mod_l, mod_inertia, mod_acfreq, 
                weight_nonseis=1.0, weight_seis=1.0, ifCorrectSurface=True):
        """
        Nonseismic and seismic combined. 

        """

        ndim = np.ndim(mod)
        if ndim == 1:
            Nmodel = 1
        else: 
            Nmodel = np.shape(mod)[0]
        
        chi2_nonseis = self.get_chi2(obs, e_obs, mod)
        chi2_seis = self.get_chi2_seismology(Nmodel, obs_freq, obs_efreq, obs_l, 
                mod_freq, mod_l, mod_inertia, mod_acfreq, 
                ifCorrectSurface=ifCorrectSurface)
        
        chi2 = chi2_nonseis * weight_nonseis + chi2_seis * weight_seis
        return chi2_nonseis, chi2_seis, chi2


    def assign_prob_to_models(self, tracks):

        Nestimate = len(self.estimates)
        Nstar = len(self.starname)
        Ntrack = len(tracks)
        Nseis = 4 if self.ifSetupSeismology else 0

        model_chi2 = [np.array([]) for istar in range(Nstar)]
        model_chi2_seis = [np.array([]) for istar in range(Nstar)]
        model_chi2_nonseis = [np.array([]) for istar in range(Nstar)]
        model_lnprob = [np.array([]) for istar in range(Nstar)]
        model_parameters = [[np.array([], dtype=object) for iestimate in range(Nestimate+Nseis)] for istar in range(Nstar)]

        for itrack in range(Ntrack): 

            # read in itrack
            atrack = self.read_models(tracks[itrack])
            Nmodel = len(atrack) - 2

            # calculate posterior
            for istar in range(Nstar):
                dt = (atrack[self.colAge][2:] - atrack[self.colAge][0:-2])/2.0
                lifespan = atrack[self.colAge][-1] - atrack[self.colAge][0]
                prior = dt/np.sum(lifespan)
                # lnprior = np.log(prior)
                # print(lnprior)

                if (self.ifSetup) & (~self.ifSetupSeismology):
                    # nonseis
                    obs, e_obs = self.stars_obs[istar], self.stars_obserr[istar]
                    mod = np.array([atrack[i][1:-1] for i in self.observables]).T.reshape(Nmodel,-1)#np.array(atrack[self.observables][1:-1]).view(np.float64).reshape(Nmodel + (-1,))

                    chi2_nonseis = self.get_chi2(obs, e_obs, mod)
                    chi2_seis = np.zeros(chi2_nonseis.shape)
                    chi2 = chi2_nonseis

                if (self.ifSetup) & (self.ifSetupSeismology):
                    # nonseis
                    # print(tracks[itrack])
                    obs, e_obs = self.stars_obs[istar], self.stars_obserr[istar]
                    mod = np.array([atrack[i][1:-1] for i in self.observables]).T.reshape(Nmodel,-1)#np.array(atrack[self.observables][1:-1]).view(np.float64).reshape(Nmodel + (-1,))

                    # seis
                    obs_freq, obs_efreq, obs_l = self.obs_freq[istar], self.obs_efreq[istar], self.obs_l[istar]
                    mod_freq = np.array(atrack[self.colModeFreq][1:-1])
                    mod_l = np.array(atrack[self.colModeDegree][1:-1])
                    if self.ifCorrectSurface:
                        mod_inertia = np.array(atrack[self.colModeInertia][1:-1])
                        mod_acfreq = np.array(atrack[self.colAcFreq][1:-1])
                    else: 
                        mod_inertia, mod_acfreq = None, None

                    chi2_nonseis, chi2_seis, chi2 = self.get_chi2_combined(obs, e_obs, mod, 
                        obs_freq, obs_efreq, obs_l, 
                        mod_freq, mod_l, mod_inertia, mod_acfreq, 
                        self.weight_nonseis, self.weight_seis, ifCorrectSurface=self.ifCorrectSurface)
                    
                if (~self.ifSetup) & (self.ifSetupSeismology):
                    # seis
                    obs_freq, obs_efreq, obs_l = self.obs_freq[istar], self.obs_efreq[istar], self.obs_l[istar]
                    mod_freq = np.array(atrack[self.colModeFreq][1:-1])
                    mod_l = np.array(atrack[self.colModeDegree][1:-1])
                    if self.ifCorrectSurface:
                        mod_inertia = np.array(atrack[self.colModeInertia][1:-1])
                        mod_acfreq = np.array(atrack[self.colAcFreq][1:-1])
                    else: 
                        mod_inertia, mod_acfreq = None, None

                    chi2_seis = self.get_chi2_seismology(Nmodel, obs_freq, obs_efreq, obs_l, 
                        mod_freq, mod_l, mod_inertia, mod_acfreq, ifCorrectSurface=self.ifCorrectSurface)

                    chi2_nonseis = np.zeros(chi2_seis.shape)
                    chi2 = chi2_seis

                lnlikelihood = -chi2/2.0 # proportionally speaking
                lnprior = np.log(prior)
                lnprob = lnprior + lnlikelihood
            
                # only save models with a large likelihood - otherwise not useful and quickly fill up memory
                fidx = chi2_nonseis < 23 # equal to likelihood<0.00001

                # estimates
                for iestimate in range(Nestimate):
                    model_parameters[istar][iestimate] = np.append(model_parameters[istar][iestimate], atrack[self.estimates[iestimate]][1:-1][fidx])
                
                if self.ifSetupSeismology:
                    for iseis, para in enumerate([self.colModeFreq, self.colModeDegree, self.colModeInertia, self.colAcFreq]):
                        model_parameters[istar][Nestimate+iseis] = np.append(model_parameters[istar][Nestimate+iseis], atrack[para][1:-1][fidx])

                # posterior
                model_chi2[istar] = np.append(model_chi2[istar], chi2[fidx])
                model_chi2_seis[istar] = np.append(model_chi2_seis[istar], chi2_seis[fidx])
                model_chi2_nonseis[istar] = np.append(model_chi2_nonseis[istar], chi2_nonseis[fidx])
                model_lnprob[istar] = np.append(model_lnprob[istar], lnprob[fidx])

        return model_lnprob, model_chi2, model_chi2_seis, model_chi2_nonseis, model_parameters


    def plot_parameter_distributions(self, samples, estimates, probs):
        # corner plot, overplotted with observation constraints

        # Ndim = samples.shape[1]
        fig = corner.corner(samples, labels=estimates, quantiles=(0.16, 0.5, 0.84), weights=probs)

        # if (not (estimates is None)) & (not (observables is None)) & (not (obs is None)) & (not (e_obs is None)):
        #     axes = np.array(fig.axes).reshape(Ndim, Ndim)
        #     for idim in range(Ndim):
        #         if estimates[idim] in observables:
        #             idx = observables == estimates[idim]

        #             axes[idim, idim].axvline(obs[idx][0], color='deepskyblue')
        #             axes[idim, idim].axvline(obs[idx][0]-e_obs[idx][0], color='lightskyblue')
        #             axes[idim, idim].axvline(obs[idx][0]+e_obs[idx][0], color='lightskyblue')
        #         else:
        #             pass
        return fig
    

    def plot_HR_diagrams(self, samples, estimates, zvals=None,
            Teff=['Teff', 'log_Teff'], lum=['luminosity', 'log_L', 'lum'], 
            delta_nu=['delta_nu', 'delta_nu_scaling', 'delta_nu_freq', 'dnu'], 
            nu_max=['nu_max', 'numax'], log_g=['log_g', 'logg']):
        
        if np.sum(np.array([iTeff in estimates for iTeff in Teff], dtype=bool))==0:
            return None 
        
        xstr = Teff[np.where(np.array([iTeff in estimates for iTeff in Teff], dtype=bool))[0][0]]
        
        numberofPlots = 0
        ystrs = []
        for val in lum+delta_nu+nu_max+log_g:
            if (val in estimates):
                numberofPlots += (val in estimates)
                ystrs.append(val)
        
        if len(ystrs) ==0:
            return None
        
        fig, axes = return_2dmap_axes(numberofPlots)

        for iax, ystr in enumerate(ystrs):
            x = samples[:,np.where(np.array(estimates) == xstr)[0][0]]
            y = samples[:,np.where(np.array(estimates) == ystr)[0][0]]
            im = axes[iax].scatter(x, y, marker='.', c=zvals, cmap='jet', s=1)
            axes[iax].set_xlabel(xstr)
            axes[iax].set_ylabel(ystr)
            axes[iax].set_xlim(quantile(x, (0.998, 0.002)).tolist())
            if ystr in delta_nu+nu_max+log_g:
                axes[iax].set_ylim(quantile(y, (0.998, 0.002)).tolist())
            else:
                axes[iax].set_ylim(quantile(y, (0.002, 0.998)).tolist())

        # fig..colorbar(im, ax=axes, orientation='vertical').set_label('Log(likelihood)')     
        # plt.tight_layout()

        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, orientation='vertical').set_label('Log(likelihood)')    

        return fig 


    def plot_seis_echelles(self, obs_freq, obs_efreq, obs_l, model_parameters, model_chi2):
        fig, axes = plt.subplots(figsize=(12,6), nrows=1, ncols=2, squeeze=False)
        axes = axes.reshape(-1)

        if (self.obs_delta_nu is None):
            delta_nu = np.abs(np.median(np.diff(np.sort(obs_freq[obs_l==0]))))  
        else:
            delta_nu = self.obs_delta_nu

        markers = ['o', '^', 's', 'v']
        colors = ['blue', 'red', 'green', 'orange']     

        # plot best model l=1
        for l in range(4):
            styles = {'marker':markers[l], 'color':colors[l], 'zorder':1}
            axes[0].scatter(obs_freq[obs_l==l] % delta_nu, obs_freq[obs_l==l], **styles)
            axes[0].scatter(obs_freq[obs_l==l] % delta_nu + delta_nu, obs_freq[obs_l==l], **styles)

        mod_freq, mod_l, mod_inertia, mod_acfreq = [model_parameters[i][np.nanargmin(model_chi2)] for i in range(len(model_parameters))]
        # _, _, _, mod_freq_uncor, mod_l_uncor = self.assign_n(obs_freq, obs_efreq, obs_l, mod_freq, mod_l)
        mod_freq_uncor, mod_l_uncor = mod_freq, mod_l
        if self.ifCorrectSurface:
            mod_freq_cor = self.get_surface_correction(obs_freq, obs_l, mod_freq, mod_l, mod_inertia, mod_acfreq, formula=self.surface_correction_formula)
            # if (mod_freq_cor is None): return fig
        else:
            mod_freq_cor = mod_freq

        # _, _, _, mod_freq_cor, mod_l_cor = self.assign_n(obs_freq, obs_efreq, obs_l, mod_freq_cor, mod_l)
        mod_freq_cor, mod_l_cor = mod_freq_cor, mod_l

        for l in [0,1]:
            styles = {'marker':markers[l], 'edgecolor':'gray', 'facecolor':'None', 'zorder':2}
            axes[0].scatter(mod_freq_uncor[mod_l_uncor==l] % delta_nu, mod_freq_uncor[mod_l_uncor==l], **styles)
            axes[0].scatter(mod_freq_uncor[mod_l_uncor==l] % delta_nu + delta_nu, mod_freq_uncor[mod_l_uncor==l], **styles)

            # surface corrected
            styles = {'marker':markers[l], 'edgecolor':'black', 'facecolor':'None', 'zorder':2}
            axes[0].scatter(mod_freq_cor[mod_l_cor==l] % delta_nu, mod_freq_cor[mod_l_cor==l], **styles)
            axes[0].scatter(mod_freq_cor[mod_l_cor==l] % delta_nu + delta_nu, mod_freq_cor[mod_l_cor==l], **styles)


        # plot best model l=2
        for l in range(4):
            styles = {'marker':markers[l], 'color':colors[l], 'zorder':1}
            axes[1].scatter(obs_freq[obs_l==l] % delta_nu, obs_freq[obs_l==l], **styles)
            axes[1].scatter(obs_freq[obs_l==l] % delta_nu + delta_nu, obs_freq[obs_l==l], **styles)

        mod_freq, mod_l, mod_inertia, mod_acfreq = [model_parameters[i][np.nanargmin(model_chi2)] for i in range(len(model_parameters))]
        # _, _, _, mod_freq_uncor, mod_l_uncor = self.assign_n(obs_freq, obs_efreq, obs_l, mod_freq, mod_l)
        mod_freq_uncor, mod_l_uncor = mod_freq, mod_l
        if self.ifCorrectSurface:
            mod_freq_cor = self.get_surface_correction(obs_freq, obs_l, mod_freq, mod_l, mod_inertia, mod_acfreq, formula=self.surface_correction_formula)
            # if (mod_freq_cor is None): return fig
        else:
            mod_freq_cor = mod_freq
        # _, _, _, mod_freq_cor, mod_l_cor = self.assign_n(obs_freq, obs_efreq, obs_l, mod_freq_cor, mod_l)
        mod_freq_cor, mod_l_cor = mod_freq_cor, mod_l

        for l in [0,2]:
            styles = {'marker':markers[l], 'edgecolor':'gray', 'facecolor':'None', 'zorder':2}
            axes[1].scatter(mod_freq_uncor[mod_l_uncor==l] % delta_nu, mod_freq_uncor[mod_l_uncor==l], **styles)
            axes[1].scatter(mod_freq_uncor[mod_l_uncor==l] % delta_nu + delta_nu, mod_freq_uncor[mod_l_uncor==l], **styles)

            # surface corrected
            styles = {'marker':markers[l], 'edgecolor':'black', 'facecolor':'None', 'zorder':2}
            axes[1].scatter(mod_freq_cor[mod_l_cor==l] % delta_nu, mod_freq_cor[mod_l_cor==l], **styles)
            axes[1].scatter(mod_freq_cor[mod_l_cor==l] % delta_nu + delta_nu, mod_freq_cor[mod_l_cor==l], **styles)


        # # plot top 10 models
        # for l in range(4):
        #     styles = {'marker':markers[l], 'color':colors[l], 'zorder':1}
        #     axes[1].scatter(obs_freq[obs_l==l] % delta_nu, obs_freq[obs_l==l], **styles)
        #     axes[1].scatter(obs_freq[obs_l==l] % delta_nu + delta_nu, obs_freq[obs_l==l] + delta_nu, **styles)

        # tmodel_parameters = [model_parameters[i][model_chi2 <= np.sort(model_chi2)[9]] for i in range(len(model_parameters))]
        # for imod in range(len(tmodel_parameters[0])):
        #     mod_freq, mod_l, mod_inertia, mod_acfreq = [tmodel_parameters[i][imod] for i in range(len(tmodel_parameters))]
        #     # _, _, _, mod_freq, mod_l = self.assign_n(obs_freq, obs_efreq, obs_l, mod_freq, mod_l)
        #     mod_freq_uncor, mod_l_uncor = mod_freq, mod_l
        #     mod_freq_cor = self.get_surface_correction(obs_freq, obs_l, mod_freq, mod_l, mod_inertia, mod_acfreq, formula=self.surface_correction_formula)
        #     if (mod_freq_cor is None): 
        #         continue
        #     # _, _, _, mod_freq_cor, mod_l_cor = self.assign_n(obs_freq, obs_efreq, obs_l, mod_freq_cor, mod_l)
        #     mod_freq_cor, mod_l_cor = mod_freq_cor, mod_l

        #     for l in range(4):
        #         # styles = {'marker':markers[l], 'edgecolor':'gray', 'facecolor':'None', 'zorder':2}
        #         # axes[1].scatter(mod_freq[mod_l==l] % delta_nu, mod_freq[mod_l==l], **styles)
        #         # axes[1].scatter(mod_freq[mod_l==l] % delta_nu + delta_nu, mod_freq[mod_l==l] + delta_nu, **styles)

        #         # surface corrected
        #         styles = {'marker':markers[l], 'edgecolor':'gray', 'facecolor':'None', 'zorder':2}
        #         axes[1].scatter(mod_freq_cor[mod_l_cor==l] % delta_nu, mod_freq_cor[mod_l_cor==l], **styles)
        #         axes[1].scatter(mod_freq_cor[mod_l_cor==l] % delta_nu + delta_nu, mod_freq_cor[mod_l_cor==l] + delta_nu, **styles)

        # # plot top 100 models
        # axes[2]

        return fig


    def output_results(self, model_prob, model_chi2, model_chi2_seis, model_chi2_nonseis, model_parameters, starnames, plot=False):

        Nstar = len(starnames)
        Nestimate = len(self.estimates)
        # Nseis = 4 if self.ifSetupSeismology else 0

        for istar in range(Nstar):
            toutdir = self.outdir + starnames[istar] + '/'
            if not os.path.exists(toutdir):
                os.mkdir(toutdir)
            
            weights = model_prob[istar]
            logweights = -model_chi2[istar]/2.
            index = (weights>=0.0) & np.isfinite(weights)
            weights, logweights = weights[index], logweights[index] # weights[~index] = 0.
            samples = (np.array(model_parameters[istar][0:Nestimate], dtype=float).T)[index,:]

            if plot:
                if samples.shape[0] <= samples.shape[1]:
                    f = open(toutdir+'log.txt', 'w')
                    f.write("Parameter estimation failed because samples.shape[0] <= samples.shape[1].")
                    f.close()
                else:
                    # plot prob distributions
                    # fig = corner.corner(samples, labels=self.estimates, quantiles=(0.16, 0.5, 0.84), weights=weights)
                    fig = self.plot_parameter_distributions(samples, self.estimates, weights)
                    fig.savefig(toutdir+"triangle.png")
                    plt.close()

                    # plot HR diagrams
                    fig = self.plot_HR_diagrams(samples, self.estimates, zvals=logweights)
                    if not (fig is None): 
                        fig.savefig(toutdir+"HR.png")
                        plt.close()

                    # plot echelle diagrams
                    if self.ifSetupSeismology:
                        fig = self.plot_seis_echelles(self.obs_freq[istar], self.obs_efreq[istar], self.obs_l[istar], 
                                model_parameters[istar][-4:], model_chi2[istar])
                        fig.savefig(toutdir+"echelles.png")
                        plt.close()

                    # write prob distribution summary file
                    results = quantile(samples, (0.16, 0.5, 0.84), weights=weights)
                    ascii.write(Table(results, names=self.estimates), toutdir+"summary.txt",format="csv", overwrite=True)

            # endofif

            # write related parameters to file
            with h5py.File(toutdir+'data.h5', 'w') as h5f:
                # classic parameters
                for i in range(len(self.estimates)):
                    h5f.create_dataset(self.estimates[i], data=np.array(model_parameters[istar][i],dtype=float))
                # seismic parameters
                if self.ifSetupSeismology:
                    for i, para in enumerate([self.colModeFreq, self.colModeDegree, self.colModeInertia]):
                        for j in range(len(model_parameters[istar][len(self.estimates)+i])):
                            h5f.create_dataset(para+'/{:0.0f}'.format(j), data=np.array(model_parameters[istar][len(self.estimates)+i][j], dtype=float))
                # chi2 parameters
                h5f.create_dataset('chi2', data=model_chi2[istar])
                h5f.create_dataset('chi2_seis', data=model_chi2_seis[istar])
                h5f.create_dataset('chi2_nonseis', data=model_chi2_nonseis[istar])
                h5f.create_dataset('prob', data=model_prob[istar])

        return 

    def estimate_parameters(self, Nthread=1, plot=False):
        """
        Estimate parameters. Magic function!
        ----------
        Optional input:
        Nthread: int
            The number of available threads to enable parallel computing.
        """

        Ntrack, Nestimate, Nstar = len(self.tracks), len(self.estimates), len(self.starname)

        if Nthread == 1:
            model_lnprob, model_chi2, model_chi2_seis, model_chi2_nonseis, model_parameters = self.assign_prob_to_models(self.tracks)
        else:
            # assign prob to models
            Ntrack_per_thread = int(Ntrack/Nthread)+1
            arglist = [self.tracks[ithread*Ntrack_per_thread:(ithread+1)*Ntrack_per_thread] for ithread in range(Nthread)]
            pool = multiprocessing.Pool(processes=Nthread)
            result_list = pool.map(self.assign_prob_to_models, arglist)
            pool.close()

            # merge probs from different threads
            model_lnprob, model_chi2, model_chi2_seis, model_chi2_nonseis = [[np.array([]) for istar in range(Nstar)] for i in range(4)]
            model_parameters = [[np.array([]) for iestimate in range(Nestimate)] for istar in range(Nstar)] 
            for ithread in range(Nthread):
                for istar in range(Nstar):
                    model_lnprob[istar] = np.append(model_lnprob[istar], result_list[ithread][0][istar])
                    model_chi2[istar] = np.append(model_chi2[istar], result_list[ithread][1][istar])
                    model_chi2_seis[istar] = np.append(model_chi2[istar], result_list[ithread][2][istar])
                    model_chi2_nonseis[istar] = np.append(model_chi2[istar], result_list[ithread][3][istar])
                    for iestimate in range(Nestimate):
                        model_parameters[istar][iestimate] = np.append(model_parameters[istar][iestimate], result_list[ithread][4][istar][iestimate])

        # normalize probs
        model_prob = model_lnprob
        for istar in range(Nstar):
            # model_prob[istar] /= np.nansum(model_lnprob[istar])
            # numerically more stable to handle extremely small probabilities
            prob = np.exp(model_lnprob[istar]-logsumexp(model_lnprob[istar][np.isfinite(model_lnprob[istar])])) #np.exp(lnprob)#
            prob[~np.isfinite(prob)] = 0.
            model_prob[istar] = prob

        # output results
        # assign prob to models
        if Nthread==1:
            self.output_results(model_prob, model_chi2, model_chi2_seis, model_chi2_nonseis, model_parameters, self.starname, plot=plot)
        else:
            Nstar_per_thread = int(Nstar/Nthread)+1
            arglist = [(model_prob[ithread*Nstar_per_thread:(ithread+1)*Nstar_per_thread], 
                    model_chi2[ithread*Nstar_per_thread:(ithread+1)*Nstar_per_thread], 
                    model_chi2_seis[ithread*Nstar_per_thread:(ithread+1)*Nstar_per_thread], 
                    model_chi2_nonseis[ithread*Nstar_per_thread:(ithread+1)*Nstar_per_thread], 
                    model_parameters[ithread*Nstar_per_thread:(ithread+1)*Nstar_per_thread], 
                    self.starname[ithread*Nstar_per_thread:(ithread+1)*Nstar_per_thread], plot) for ithread in range(Nthread)]

            pool = multiprocessing.Pool(processes=Nthread)
            pool.starmap(self.output_results, arglist)
            pool.close()

        return

    def tar_outputs(self):

        """
        Tar the output directory.
        """

        os.system("tar zcvf output.tar.gz " + self.outdir)
        return





if __name__ == '__main__':

    # ### test 1: read models
    # h = history('models/grid_mass/LOGS_data/m090feh000.history', ifReadProfileIndex=True)
    # s = sums('models/grid_mass/sums/m090feh000profile1.data.FGONG.sum')
    # p = profile('sample/bestModels/profiles/5607242/m112feh-045profile211.data')
    # m = modes('sample/bestModels/profiles/5607242/gyreMode00001.txt')
    

    # ### test 2: grid modelling
    # kepler = ascii.read('sample/KeplerSubgiants.csv')
    # modes = ascii.read('sample/mode_parameters-fs.csv')
    # kic = 10147635
    # modes = modes[(modes['keplerid']==kic) & (modes['lnk']>1)]

    # # define a read track function
    # def read_models(filepath):
    #     h = history(filepath, verbose=False, ifReadProfileIndex=True)
    #     atrack = h.track
    #     profileIndex = h.profileIndex
    #     idx = np.isin(atrack['model_number'], profileIndex['model_number'])
    #     atrack = Table(atrack[idx])
    #     atrack['luminosity'] = 10.0**atrack['log_L']
    #     atrack['radius'] = 10.0**atrack['log_R']
    #     atrack['Teff'] = 10.0**atrack['log_Teff']
    #     atrack['mode_freq'] = np.zeros(len(atrack), dtype=object)
    #     atrack['mode_l'] = np.zeros(len(atrack), dtype=object)
    #     atrack['mode_inertia'] = np.zeros(len(atrack), dtype=object)
        
    #     # read in sums
    #     for imodel, model in enumerate(atrack):
    #         model_number = model['model_number']
    #         idx = profileIndex['model_number']==model_number
    #         profile = profileIndex['profile_number'][idx][0]
    #         sumFilepath = filepath.replace('LOGS_data', 'sums').split('.history')[0] + 'profile{:0.0f}.data.FGONG.sum'.format(profile)
    #         s = sums(sumFilepath, verbose=False)
    #         s = s.modeSummary
    #         atrack['mode_freq'][imodel] = s['Refreq']
    #         atrack['mode_l'][imodel] = s['l']
    #         atrack['mode_inertia'][imodel] = s['E_norm']

    #     return atrack

    # outdir = 'sample/best_model/'
    # tracks = ['models/grid_mass/LOGS_data/'+f for f in os.listdir('models/grid_mass/LOGS_data/') if f.endswith('.history')]
    # estimates = ['star_mass', 'luminosity', 'radius', 'Teff', 
    #             'model_number', 'log_g', 'nu_max', 'delta_nu', 'acoustic_cutoff']

    # g = grid(read_models, tracks, estimates, outdir, [str(kic)])
    # g.setup_seismology([modes['fc']], [modes['fc_err']], [modes['l']])
    # g.estimate_parameters()

    # # atrack = read_models(tracks[0])
    pass