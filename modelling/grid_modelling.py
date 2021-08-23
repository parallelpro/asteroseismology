import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors
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

from asteroseismology.tools import return_2dmap_axes, quantile
from asteroseismology.modelling.surface_correction import get_surface_correction
from asteroseismology.modelling.model_Dnu import get_model_Dnu, get_obs_Dnu
from asteroseismology.modelling.results_container import stardata

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
            Return a 'None' object to skip this track. This is useful for
            cases like the data is not properly laoding.
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
        self.Nstar = len(self.starname)
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

        self.observables = np.array(observables)
        self.stars_obs = stars_obs
        self.stars_obserr = stars_obserr
        
        self.ifSetup = True
        return self


    def setup_seismology(self, obs_freq, obs_efreq, obs_l, Dnu, numax,
            colModeFreq='mode_freq', colModeDegree='mode_l', colModeInertia='mode_inertia',
            colAcFreq='acoustic_cutoff', colModeNode='mode_n', 
            weight_nonseis=1, weight_seis=1, weight_reg=1, ifCorrectSurface=True,
            surface_correction_formula='cubic', Nreg=0, rescale_percentile=10):
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

        Dnu: array-like[Nstar,] 
            The p-mode large separation in muHz. This is used to compare 
            the model Dnu calculated from radial mode frequencies, and 
            thus ensure the models in a reasonable Dnu range. 

        numax: array-like[Nstar,] 
            The frequency of maximum power in muHz. This is used to compare 
            the model Dnu calculated from radial mode frequencies, and 
            thus ensure the models in a reasonable Dnu range. 

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

        colModeNode: str, default 'mode_n'
            The mode radial node number retrieved from atrack.

        ifCorrectSurface: bool, default True
            if True, then correct model frequencies using the formula 
            of Ball & Gizon (2014) (the inverse and cubic term).
            Should extend more capabilities here!

        Nreg: int, default 0
            the number of low-frequency modes that do not apply surface effects.

        rescale_percentile: float, default 10, between 0 to 100
            In most cases, the frequencies of stellar models still can't match with 
            observations within the uncertainties. So we need to rescale the seismic
            chi2 (chi2_reg, chi2_seis) in order to avoid an extremely sharp posterior
            distribution. The variable ``rescale_percentile'' specifies the fraction of
            seismic models that will be thought as a reasonable agreement between models
            and observations (i.e. the difference will be treated as a systematic 
            uncertainty in models) in all seismic models. 

            In python language:
            seismic_chi2_unweighted[imod] = np.sum((obs_freq-mod_freq[imod])**2.0/(obs_efreq**2.0))
            mod_efreq = np.percentile(seismic_chi2_unweighted, rescale_percentile)**0.5
            seismic_chi2_weighted[imod] = np.sum((obs_freq-mod_freq[imod])**2.0/(obs_efreq**2.0+mod_efreq**2.0))

        """

        self.obs_freq = obs_freq
        self.obs_efreq = obs_efreq
        self.obs_l = obs_l
        self.obs_l_uniq = np.array([np.unique(l) for l in obs_l], dtype=object)
        self.obs_Nl = np.array([len(np.unique(l)) for l in obs_l])
        self.Dnu = Dnu
        self.numax = numax
        self.colModeFreq = colModeFreq
        self.colModeDegree = colModeDegree
        self.colModeInertia = colModeInertia
        self.colAcFreq = colAcFreq
        self.colModeNode = colModeNode
        self.weight_nonseis = weight_nonseis
        self.weight_seis = weight_seis
        self.weight_reg = weight_reg
        self.ifCorrectSurface = ifCorrectSurface
        self.surface_correction_formula = surface_correction_formula
        self.rescale_percentile = rescale_percentile

        if self.rescale_percentile==0.:
            self.ifRescale = False
        else:
            self.ifRescale = True
        
        self.ifSetupSeismology=True
        if Nreg>0:
            self.ifSetupRegularization=True
            self.Nreg=Nreg
        else:
            self.ifSetupRegularization=False

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


    def get_chi2_seismology(self, Nmodel, obs_freq, obs_efreq, obs_l, Dnu, numax,
                mod_freq, mod_l, mod_inertia, mod_acfreq, 
                ifCorrectSurface=True, mod_efreq_sys=None, mod_efreq_sys_reg=None):
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
        Dnu: float
            observed Dnu
        numax: float
            observed numax
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

        obs_l_unique = np.unique(obs_l)
        Nl = len(obs_l_unique)
        if (mod_efreq_sys is None) : mod_efreq_sys = [0. for l in range(Nl)]
        if (mod_efreq_sys_reg is None): mod_efreq_sys_reg = 0.

        chi2_seis = [np.array([np.inf]*Nmodel) for l in range(Nl)]
        chi2_reg = np.array([np.inf]*Nmodel)
        surface_parameters = np.array([np.nan]*Nmodel, dtype=object)

        for imod in range(Nmodel):
            if Nmodel == 1:
                mod_freq_imod, mod_l_imod = np.array(mod_freq), np.array(mod_l)
            else:
                mod_freq_imod, mod_l_imod = np.array(mod_freq[imod]), np.array(mod_l[imod])
        
            if ifCorrectSurface:
                corr_mod_freq_imod = None
                if (np.sum(np.isin(mod_l_imod, 0))) :
                    mod_Dnu = get_model_Dnu(mod_freq_imod, mod_l_imod, Dnu, numax)
                    if -0.10 < ((mod_Dnu-Dnu)/Dnu) < 0.20:
                        if Nmodel == 1:
                            mod_inertia_imod, mod_acfreq_imod = mod_inertia, mod_acfreq
                        else:
                            mod_inertia_imod, mod_acfreq_imod = mod_inertia[imod], mod_acfreq[imod]

                        corr_mod_freq_imod, surface_parameters_imod = get_surface_correction(obs_freq, obs_l, mod_freq_imod, mod_l_imod, 
                                                                            mod_inertia_imod, mod_acfreq_imod, 
                                                                            formula=self.surface_correction_formula,
                                                                            ifFullOutput=True)


                if (corr_mod_freq_imod is None):
                    for il, l in enumerate(obs_l_unique):
                        chi2_seis[il][imod] = np.inf
                else:
                    for il, l in enumerate(obs_l_unique):
                        oidx, midx = obs_l==l, mod_l_imod==l
                        if (l==0) & (self.ifSetupRegularization):
                            obs_freq_imod_il, obs_efreq_imod_il, _, corr_mod_freq_imod_il, _, mod_freq_imod_il = self.assign_n(obs_freq[oidx], obs_efreq[oidx], obs_l[oidx], corr_mod_freq_imod[midx], mod_l_imod[midx], mod_freq_imod[midx])
                            idx = np.argsort(obs_freq_imod_il)[:self.Nreg]
                            ridx = ~np.isin(np.arange(len(obs_freq_imod_il)), idx)
                            chi2_reg[imod] = np.sum((obs_freq_imod_il[idx]-mod_freq_imod_il[idx])**2.0/(obs_efreq_imod_il[idx]**2.0+mod_efreq_sys_reg**2.0))#/(self.Nreg-1)
                            chi2_seis[il][imod] = np.sum((obs_freq_imod_il[ridx]-corr_mod_freq_imod_il[ridx])**2.0/(obs_efreq_imod_il[ridx]**2.0+mod_efreq_sys[il]**2.0))#/(Nobservable
                        else:
                            obs_freq_imod_il, obs_efreq_imod_il, _, corr_mod_freq_imod_il, _ = self.assign_n(obs_freq[oidx], obs_efreq[oidx], obs_l[oidx], corr_mod_freq_imod[midx], mod_l_imod[midx])
                            chi2_seis[il][imod] = np.sum((obs_freq_imod_il-corr_mod_freq_imod_il)**2.0/(obs_efreq_imod_il**2.0+mod_efreq_sys[il]**2.0))#/(Nobservable)

                    surface_parameters[imod] = surface_parameters_imod
            
            else:
                mod_Dnu = get_model_Dnu(mod_freq_imod, mod_l_imod, Dnu, numax)
                if -0.15 < ((mod_Dnu-Dnu)/Dnu) < 0.15:
                    for il, l in enumerate(obs_l_unique):
                        oidx, midx = obs_l==l, mod_l_imod==l
                        if (l==0) & (self.ifSetupRegularization):
                            obs_freq_imod_il, obs_efreq_imod_il, _, mod_freq_imod_il, _,  = self.assign_n(obs_freq[oidx], obs_efreq[oidx], obs_l[oidx], mod_freq_imod[midx], mod_l_imod[midx])
                            idx = np.argsort(obs_freq_imod_il)[:self.Nreg]
                            ridx = ~np.isin(np.arange(len(obs_freq_imod_il)), idx)
                            chi2_reg[imod] = np.sum((obs_freq_imod_il[idx]-mod_freq_imod_il[idx])**2.0/(obs_efreq_imod_il[idx]**2.0+mod_efreq_sys_reg**2.0))#/(self.Nreg-1)
                            chi2_seis[il][imod] = np.sum((obs_freq_imod_il[ridx]-mod_freq_imod_il[ridx])**2.0/(obs_efreq_imod_il[ridx]**2.0+mod_efreq_sys[il]**2.0))#/(Nobservable
                        else:
                            obs_freq_imod_il, obs_efreq_imod_il, _, mod_freq_imod_il, _ = self.assign_n(obs_freq[oidx], obs_efreq[oidx], obs_l[oidx], mod_freq_imod[midx], mod_l_imod[midx])
                            chi2_seis[il][imod] = np.sum((obs_freq_imod_il-mod_freq_imod_il)**2.0/(obs_efreq_imod_il**2.0+mod_efreq_sys[il]**2.0))#/(Nobservable)

        return chi2_seis, chi2_reg, surface_parameters


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
                obs_freq, obs_efreq, obs_l, Dnu, numax,
                mod_freq, mod_l, mod_inertia, mod_acfreq, 
                weight_nonseis=1.0, weight_seis=1.0, weight_reg=1.0, 
                ifCorrectSurface=True, mod_efreq_sys=None, mod_efreq_sys_reg=None):
        """
        Nonseismic and seismic combined. 

        """
        
        ndim = np.ndim(mod)
        if ndim == 1:
            Nmodel = 1
        else: 
            Nmodel = np.shape(mod)[0]
        
        chi2_nonseis = self.get_chi2(obs, e_obs, mod)
        chi2_seis, chi2_reg, surface_parameters = self.get_chi2_seismology(Nmodel, obs_freq, obs_efreq, obs_l, Dnu, numax,
                mod_freq, mod_l, mod_inertia, mod_acfreq, 
                ifCorrectSurface=ifCorrectSurface, mod_efreq_sys=mod_efreq_sys, mod_efreq_sys_reg=mod_efreq_sys_reg)
        
        if self.ifSetupRegularization:
            chi2 = chi2_nonseis * weight_nonseis + np.sum(chi2_seis, axis=0) * weight_seis + chi2_reg * weight_reg
        else:
            chi2 = chi2_nonseis * weight_nonseis + np.sum(chi2_seis, axis=0) * weight_seis
        return chi2_nonseis, chi2_seis, chi2_reg, chi2, surface_parameters

    def find_chi2_unweighted_seis(self, tracks):
        
        Nstar = len(self.starname)
        Ntrack = len(tracks)
        
        starsdata = [stardata() for istar in range(Nstar)]

        for itrack in range(Ntrack): 

            # read in itrack
            atrack = self.read_models(tracks[itrack])
            if (atrack is None) : continue
            Nmodel = len(atrack) - 2

            # calculate posterior
            for istar in range(Nstar):
                # seis
                obs_freq, obs_efreq, obs_l = self.obs_freq[istar], np.ones(self.obs_freq[istar].shape), self.obs_l[istar]
                mod_freq = np.array(atrack[self.colModeFreq][1:-1])
                mod_l = np.array(atrack[self.colModeDegree][1:-1])
                if self.ifCorrectSurface:
                    mod_inertia = np.array(atrack[self.colModeInertia][1:-1])
                    mod_acfreq = np.array(atrack[self.colAcFreq][1:-1])
                else: 
                    mod_inertia, mod_acfreq = None, None

                chi2_unweighted_seis, chi2_unweighted_reg, _ = self.get_chi2_seismology(Nmodel, obs_freq, obs_efreq, obs_l, self.Dnu[istar], self.numax[istar],
                    mod_freq, mod_l, mod_inertia, mod_acfreq, ifCorrectSurface=self.ifCorrectSurface)

                for il, l in enumerate(self.obs_l_uniq[istar]):
                    starsdata[istar].append('chi2_unweighted_seis_l{:0.0f}'.format(l), chi2_unweighted_seis[il])

                if self.ifSetupRegularization:
                    starsdata[istar].append('chi2_unweighted_reg', chi2_unweighted_reg)

        return starsdata

    def assign_prob_to_models(self, tracks):
        
        Nestimate = self.Nestimate
        Nstar = len(self.starname)
        Ntrack = len(tracks)

        starsdata = [stardata() for istar in range(Nstar)]
        for itrack in range(Ntrack): 

            # read in itrack
            atrack = self.read_models(tracks[itrack])
            if (atrack is None) : continue
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
                    chi2 = np.array(chi2_nonseis)

                if (self.ifSetup) & (self.ifSetupSeismology):
                    # nonseis
                    # print(tracks[itrack])
                    obs, e_obs = self.stars_obs[istar], self.stars_obserr[istar]
                    mod = np.array([atrack[i][1:-1] for i in self.observables]).T.reshape(Nmodel,-1)#np.array(atrack[self.observables][1:-1]).view(np.float64).reshape(Nmodel + (-1,))

                    # seis
                    obs_freq, obs_efreq, obs_l = self.obs_freq[istar], self.obs_efreq[istar], self.obs_l[istar]
                    mod_freq = np.array(atrack[self.colModeFreq][1:-1])
                    mod_l = np.array(atrack[self.colModeDegree][1:-1])
                    mod_n = np.array(atrack[self.colModeNode][1:-1])
                    if self.ifCorrectSurface:
                        mod_inertia = np.array(atrack[self.colModeInertia][1:-1])
                        mod_acfreq = np.array(atrack[self.colAcFreq][1:-1])
                    else: 
                        mod_inertia, mod_acfreq = None, None

                    chi2_nonseis, chi2_seis, chi2_reg, chi2, surface_parameters = self.get_chi2_combined(obs, e_obs, mod, 
                        obs_freq, obs_efreq, obs_l, self.Dnu[istar], self.numax[istar],
                        mod_freq, mod_l, mod_inertia, mod_acfreq, 
                        self.weight_nonseis, self.weight_seis, self.weight_reg,
                        ifCorrectSurface=self.ifCorrectSurface, mod_efreq_sys=self.mod_efreq_sys[istar], mod_efreq_sys_reg=self.mod_efreq_sys_reg[istar])
                    
                if (~self.ifSetup) & (self.ifSetupSeismology):
                    # seis
                    obs_freq, obs_efreq, obs_l = self.obs_freq[istar], self.obs_efreq[istar], self.obs_l[istar]
                    mod_freq = np.array(atrack[self.colModeFreq][1:-1])
                    mod_l = np.array(atrack[self.colModeDegree][1:-1])
                    mod_n = np.array(atrack[self.colModeNode][1:-1])
                    if self.ifCorrectSurface:
                        mod_inertia = np.array(atrack[self.colModeInertia][1:-1])
                        mod_acfreq = np.array(atrack[self.colAcFreq][1:-1])
                    else: 
                        mod_inertia, mod_acfreq = None, None

                    chi2_seis, chi2_reg, surface_parameters = self.get_chi2_seismology(Nmodel, obs_freq, obs_efreq, obs_l, 
                        self.Dnu[istar], self.numax[istar],
                        mod_freq, mod_l, mod_inertia, mod_acfreq, 
                        ifCorrectSurface=self.ifCorrectSurface, mod_efreq_sys=self.mod_efreq_sys[istar], mod_efreq_sys_reg=self.mod_efreq_sys_reg[istar])
                    
                    if self.ifSetupRegularization:
                        chi2 = np.sum(chi2_seis, axis=0)*self.weight_seis + chi2_reg*self.weight_reg
                    else:
                        chi2 = np.sum(chi2_seis, axis=0)*self.weight_seis

                lnlikelihood = -chi2/2.0 # proportionally speaking
                lnprior = np.log(prior)
                lnprob = lnprior + lnlikelihood
                

                # only save models with a large likelihood - otherwise not useful and quickly fill up memory
                fidx = (chi2_nonseis < 23) if (self.ifSetup) else (chi2 < 23) # equal to likelihood<0.00001
                # print(istar, np.sum(fidx))

                # save estimates
                for iestimate in range(Nestimate):
                    starsdata[istar].append(self.estimates[iestimate], np.array(atrack[self.estimates[iestimate]][1:-1][fidx]), dtype=float)

                if self.ifSetup:
                    starsdata[istar].append('chi2_nonseis', chi2_nonseis[fidx], dtype=float)

                if self.ifSetupSeismology:
                    starsdata[istar].append(self.colModeFreq, np.array(atrack[self.colModeFreq][1:-1][fidx]), dtype=object)
                    starsdata[istar].append(self.colModeDegree, np.array(atrack[self.colModeDegree][1:-1][fidx]), dtype=object)
                    starsdata[istar].append(self.colModeNode, np.array(atrack[self.colModeNode][1:-1][fidx]), dtype=object)
                    if self.ifCorrectSurface:
                        starsdata[istar].append(self.colModeInertia, np.array(atrack[self.colModeInertia][1:-1][fidx]), dtype=object)
                        starsdata[istar].append('surface_parameters', surface_parameters[fidx], dtype=object)
                    for il, l in enumerate(self.obs_l_uniq[istar]):
                        starsdata[istar].append('chi2_seis_l{:0.0f}'.format(l), chi2_seis[il][fidx], dtype=float)
                
                    if self.ifSetupRegularization:
                        starsdata[istar].append('chi2_reg', chi2_reg[fidx], dtype=float)

                starsdata[istar].append('chi2', chi2[fidx], dtype=float)
                starsdata[istar].append('lnprior', lnprior[fidx], dtype=float)
                starsdata[istar].append('lnprob', lnprob[fidx], dtype=float)

        return starsdata


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


    def plot_seis_echelles(self, obs_freq, obs_efreq, obs_l, model_parameters, model_chi2, Dnu, 
                            ifCorrectSurface=True, surface_correction_formula='cubic'):
        
        if ifCorrectSurface:
            fig, axes = plt.subplots(figsize=(12,5), nrows=1, ncols=2, squeeze=False)
        else:
            fig, axes = plt.subplots(figsize=(6,5), nrows=1, ncols=1, squeeze=False)
        # axes = axes.reshape(-1) # 0: uncorrected, 1: corrected

        markers = ['o', '^', 's', 'v']
        colors = ['blue', 'red', 'green', 'orange']     

        # plot observation frequencies
        for l in range(4):
            styles = {'marker':markers[l], 'color':colors[l], 'zorder':1}
            axes[0,0].scatter(obs_freq[obs_l==l] % Dnu, obs_freq[obs_l==l], **styles)
            axes[0,0].scatter(obs_freq[obs_l==l] % Dnu + Dnu, obs_freq[obs_l==l], **styles)
            if ifCorrectSurface:
                axes[0,1].scatter(obs_freq[obs_l==l] % Dnu, obs_freq[obs_l==l], **styles)
                axes[0,1].scatter(obs_freq[obs_l==l] % Dnu + Dnu, obs_freq[obs_l==l], **styles)

        norm = matplotlib.colors.Normalize(vmin=np.min(model_chi2), vmax=np.max(model_chi2))
        cmap = plt.cm.get_cmap('gray')
        for imod in np.argsort(model_chi2)[::-1]:
            mod_freq_uncor, mod_l_uncor, mod_n = model_parameters['mode_freq'][imod], model_parameters['mode_l'][imod], model_parameters['mode_n'][imod]
            if ifCorrectSurface:
                mod_inertia, mod_acfreq = model_parameters['mode_inertia'][imod], model_parameters['acoustic_cutoff'][imod]
                if ifCorrectSurface:
                    mod_freq_cor = get_surface_correction(obs_freq, obs_l, np.array(mod_freq_uncor), np.array(mod_l_uncor), mod_inertia, mod_acfreq, formula=surface_correction_formula)
                    # if (mod_freq_cor is None): return fig
                else:
                    mod_freq_cor = np.array(mod_freq_uncor)
                mod_freq_cor, mod_l_cor = np.array(mod_freq_cor), np.array(mod_l_uncor)

            for l in np.array(np.unique(obs_l), dtype=int):
                # axes[0] plot uncorrected frequencies
                z = np.zeros(np.sum(mod_l_uncor==l))+model_chi2[imod]
                scatterstyles = {'marker':markers[l], 'edgecolors':cmap(norm(z)), 'c':'None', 'zorder':2}
                axes[0,0].scatter(mod_freq_uncor[mod_l_uncor==l] % Dnu, mod_freq_uncor[mod_l_uncor==l], **scatterstyles)
                axes[0,0].scatter(mod_freq_uncor[mod_l_uncor==l] % Dnu + Dnu, mod_freq_uncor[mod_l_uncor==l], **scatterstyles)
                if ifCorrectSurface:
                    # axes[1] plot surface corrected frequencies
                    axes[0,1].scatter(mod_freq_cor[mod_l_cor==l] % Dnu, mod_freq_cor[mod_l_cor==l], **scatterstyles)
                    axes[0,1].scatter(mod_freq_cor[mod_l_cor==l] % Dnu + Dnu, mod_freq_cor[mod_l_cor==l], **scatterstyles)

            # label the radial orders n for l=0 modes
            if (imod == np.argsort(model_chi2)[0]) & np.sum(mod_l_uncor==0):
                for idxn, n in enumerate(mod_n[mod_l_uncor==0]):
                    nstr = '{:0.0f}'.format(n)
                    # axes[0] plot uncorrected frequencies
                    textstyles = {'fontsize':12, 'ha':'center', 'va':'center', 'zorder':100, 'color':'purple'}
                    axes[0,0].text((mod_freq_uncor[mod_l_uncor==0][idxn]+0.05*Dnu) % Dnu, mod_freq_uncor[mod_l_uncor==0][idxn]+0.05*Dnu, nstr, **textstyles)
                    axes[0,0].text((mod_freq_uncor[mod_l_uncor==0][idxn]+0.05*Dnu) % Dnu + Dnu, mod_freq_uncor[mod_l_uncor==0][idxn]+0.05*Dnu, nstr, **textstyles)
                    if ifCorrectSurface:
                        # axes[1] plot surface corrected frequencies
                        axes[0,1].text((mod_freq_cor[mod_l_cor==0][idxn]+0.05*Dnu) % Dnu, mod_freq_cor[mod_l_cor==0][idxn]+0.05*Dnu, nstr, **textstyles)
                        axes[0,1].text((mod_freq_cor[mod_l_cor==0][idxn]+0.05*Dnu) % Dnu + Dnu, mod_freq_cor[mod_l_cor==0][idxn]+0.05*Dnu, nstr, **textstyles)

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap='gray'),cax=cbar_ax).set_label('chi2_seismic')

        for ax in axes.reshape(-1):
            ax.axis([0., Dnu*2, np.min(obs_freq)-Dnu*4, np.max(obs_freq)+Dnu*4])
            ax.set_ylabel('Frequency')
            ax.set_xlabel('Frequency mod Dnu {:0.3f}'.format(Dnu))
        axes[0,0].set_title('Before correction')
        if ifCorrectSurface:
            axes[0,1].set_title('After correction')

        return fig


    def output_results(self, starsdata, plot=False, thread_idx=None):
        
        if (thread_idx is None): thread_idx = slice(0,len(self.starname))
        starnames = self.starname[thread_idx]
        if (self.ifSetupSeismology):
            obs_freq = self.obs_freq[thread_idx]
            obs_efreq = self.obs_efreq[thread_idx]
            obs_l = self.obs_l[thread_idx]
            Dnu = self.Dnu[thread_idx]
            obs_l_uniq = self.obs_l_uniq[thread_idx]
        Nstar = len(starnames)


        for istar in range(Nstar):
            toutdir = self.outdir + starnames[istar] + '/'
            if not os.path.exists(toutdir):
                os.mkdir(toutdir)
            
            samples = []
            for estimate in self.estimates:
                samples.append(starsdata[istar][estimate])
            samples = np.transpose(np.array(samples))
            

            prob = np.exp(-starsdata[istar]['chi2']/2.)
            if (self.ifSetup):
                prob_nonseis = np.exp(-(starsdata[istar]['chi2_nonseis']*self.weight_nonseis)/2.)
            if (self.ifSetupSeismology):
                if (self.ifSetupRegularization):
                    chi2_seis = np.sum([starsdata[istar]['chi2_seis_l{:0.0f}'.format(l)] for l in obs_l_uniq[istar]],axis=0)*self.weight_seis + starsdata[istar]['chi2_reg']*self.weight_reg
                else:
                    chi2_seis = np.sum([starsdata[istar]['chi2_seis_l{:0.0f}'.format(l)] for l in obs_l_uniq[istar]],axis=0)*self.weight_seis
                prob_seis = np.exp(-(chi2_seis)/2.)
            if plot:
                if samples.shape[0] <= samples.shape[1]:
                    f = open(toutdir+'log.txt', 'w')
                    f.write("Parameter estimation failed because samples.shape[0] <= samples.shape[1].")
                    f.close()
                else:
                    # plot prob distributions
                    if (self.ifSetup):
                        fig = self.plot_parameter_distributions(samples, self.estimates, prob_nonseis)
                        fig.savefig(toutdir+"corner_prob_classic.png")
                        plt.close()

                    if (self.ifSetupSeismology):
                        fig = self.plot_parameter_distributions(samples, self.estimates, prob_seis)
                        fig.savefig(toutdir+"corner_prob_seismic.png")
                        plt.close()

                    # output the prob (prior included)
                    fig = self.plot_parameter_distributions(samples, self.estimates, prob)
                    fig.savefig(toutdir+"corner_prob.png")
                    plt.close()

                    # # plot HR diagrams
                    # fig = self.plot_HR_diagrams(samples, self.estimates, zvals=logweights)
                    # if not (fig is None): 
                    #     fig.savefig(toutdir+"HR.png")
                    #     plt.close()

                    # plot echelle diagrams
                    if self.ifSetupSeismology:
                        idx = np.argsort(chi2_seis, axis=0)[:10]
                        if self.ifCorrectSurface:
                            model_parameters = {'mode_freq': starsdata[istar][self.colModeFreq][idx], 
                                                'mode_l': starsdata[istar][self.colModeDegree][idx], 
                                                'mode_inertia': starsdata[istar][self.colModeInertia][idx], 
                                                'acoustic_cutoff': starsdata[istar][self.colAcFreq][idx], 
                                                'mode_n': starsdata[istar][self.colModeNode][idx]}
                        else:
                            model_parameters = {'mode_freq': starsdata[istar][self.colModeFreq][idx], 
                                                'mode_l': starsdata[istar][self.colModeDegree][idx], 
                                                'mode_n': starsdata[istar][self.colModeNode][idx]}
                        fig = self.plot_seis_echelles(obs_freq[istar], obs_efreq[istar], obs_l[istar], 
                                model_parameters, chi2_seis[idx], Dnu[istar], 
                                ifCorrectSurface=self.ifCorrectSurface, 
                                surface_correction_formula=self.surface_correction_formula)
                        fig.savefig(toutdir+"echelle_top10_prob_seismic.png")
                        plt.close()

                    # write prob distribution summary file
                    if (self.ifSetup):
                        results = quantile(samples, (0.16, 0.5, 0.84), weights=prob_nonseis)
                        ascii.write(Table(results, names=self.estimates), toutdir+"summary_prob_classic.txt",format="csv", overwrite=True)

                    if (self.ifSetupSeismology):
                        results = quantile(samples, (0.16, 0.5, 0.84), weights=prob_seis)
                        ascii.write(Table(results, names=self.estimates), toutdir+"summary_prob_seismic.txt",format="csv", overwrite=True)

                    # output the prob (prior included)
                    results = quantile(samples, (0.16, 0.5, 0.84), weights=prob)
                    ascii.write(Table(results, names=self.estimates), toutdir+"summary_prob.txt",format="csv", overwrite=True)

            # endofif

            # write related parameters to file
            with h5py.File(toutdir+'data.h5', 'w') as h5f:
                # classic parameters
                for key in starsdata[istar].keys:
                    if starsdata[istar][key].dtype == 'O':
                        for iobj in range(len(starsdata[istar][key])):
                            h5f.create_dataset(key+'/{:0.0f}'.format(iobj), data=np.array(starsdata[istar][key][iobj], dtype=float))
                    else:
                        h5f.create_dataset(key, data=np.array(starsdata[istar][key], dtype=float))

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

        # # step 1, find the systematic uncertainty in seismology models
        if (self.ifSetupSeismology):
            if (self.ifRescale):
                if Nthread == 1:
                    starsdata = self.find_chi2_unweighted_seis(self.tracks)
                else:
                    # multithreading
                    Ntrack_per_thread = int(Ntrack/Nthread)+1
                    arglist = [self.tracks[ithread*Ntrack_per_thread:(ithread+1)*Ntrack_per_thread] for ithread in range(Nthread)]

                    pool = multiprocessing.Pool(processes=Nthread)
                    result_list = pool.map(self.find_chi2_unweighted_seis, arglist)
                    pool.close()

                    starsdata = [0. for i in range(Nstar)]
                    for istar in range(Nstar):
                        starsdata[istar] = stardata([result_list[ithread][istar] for ithread in range(Nthread)])

                mod_efreq_sys = [0. for i in range(Nstar)]
                for istar in range(Nstar):
                    mod_efreq_sys_istar = [0. for i in range(len(self.obs_l_uniq[istar]))]
                    for il, l in enumerate(self.obs_l_uniq[istar]):
                        idx = np.isfinite(starsdata[istar]['chi2_unweighted_seis_l{:0.0f}'.format(l)])
                        if np.sum(idx) > 0:
                            sig = np.percentile(starsdata[istar]['chi2_unweighted_seis_l{:0.0f}'.format(l)][idx],self.rescale_percentile)**0.5
                        else:
                            sig = 0.
                        mod_efreq_sys_istar[il] = sig
                    mod_efreq_sys[istar] = mod_efreq_sys_istar
                
                mod_efreq_sys_reg = [0. for i in range(Nstar)]
                if self.ifSetupRegularization:
                    for istar in range(Nstar):
                        idx = np.isfinite(starsdata[istar]['chi2_unweighted_reg'])
                        if np.sum(idx) > 0:
                            sig = np.percentile(starsdata[istar]['chi2_unweighted_reg'][idx],self.rescale_percentile)**0.5
                        else:
                            sig = 0.
                        mod_efreq_sys_reg[istar] = sig
                    
                
                self.mod_efreq_sys = mod_efreq_sys
                self.mod_efreq_sys_reg = mod_efreq_sys_reg
        
            else: 
                self.mod_efreq_sys = [[0. for i in range(len(self.obs_l_uniq[istar]))] for istar in range(Nstar)]
                self.mod_efreq_sys_reg = [0. for i in range(Nstar)]

        # print(self.mod_efreq_sys)
        # print(self.mod_efreq_sys_reg)

        # # step 2, assign prob to models
        if Nthread == 1:
            starsdata = self.assign_prob_to_models(self.tracks)
        else:
            # multithreading
            Ntrack_per_thread = int(Ntrack/Nthread)+1
            arglist = [self.tracks[ithread*Ntrack_per_thread:(ithread+1)*Ntrack_per_thread] for ithread in range(Nthread)]

            pool = multiprocessing.Pool(processes=Nthread)
            result_list = pool.map(self.assign_prob_to_models, arglist)
            pool.close()

            # merge from different threads
            starsdata = [0. for i in range(Nstar)]
            for istar in range(Nstar):
                starsdata[istar] = stardata([result_list[ithread][istar] for ithread in range(Nthread)])


        # # step 3, normalize probs
        for istar in range(Nstar):
            # model_prob[istar] /= np.nansum(model_lnprob)
            # numerically more stable to handle extremely small probabilities
            model_lnprob = starsdata[istar]['lnprob']
            lnprob = np.zeros(len(model_lnprob))
            idx = (np.isfinite(model_lnprob)) & (model_lnprob>-50)
            if np.sum(idx)>0: lnprob[idx] = model_lnprob[idx]-logsumexp(model_lnprob[idx])
            lnprob[~idx] = -np.inf
            starsdata[istar]['prob'] = np.exp(lnprob)


        # # step 4, output results
        if Nthread==1:
            self.output_results(starsdata, plot=plot)
        else:
            Nstar_per_thread = int(Nstar/Nthread)+1
            thread_idxs = [slice(ithread*Nstar_per_thread, (ithread+1)*Nstar_per_thread) for ithread in range(Nthread)]
            arglist = [(np.array(starsdata)[thread_idx].tolist(), plot, thread_idx) for thread_idx in thread_idxs]

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

