#!/usr/bin/env/ python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt 

from .. import (powerSpectrumSmoothWrapper, 
smoothWrapper, a_correlate, 
standardBackgroundModel, 
LSSampler, ESSampler, echelle)

from wotan import flatten
import peakutils
from scipy.signal import detrend 

__all__ = ['solarlikeGlobalSeismo']

class solarlikeGlobalSeismo:
    '''
    Under development:
    # 1: p-mode asymptotic fitting
    # 2: light curve processing
    # 3: g-mode asymptotic fitting
    '''
    def __init__(self, freq, power, fnyq, starID='', filepath=''):
        idx = (freq>15.)
        self.freq, self.power = freq[idx], power[idx]
        self.fnyq = fnyq
        self.powerGlobalSmoothed = powerSpectrumSmoothWrapper(self.freq, self.power, windowSize=0.5)

        self.starID = starID if starID != '' else 'star'
        self.filepath = filepath if filepath != '' else './'
        return

    def run(self, verbose=True):
        if verbose: print('>>> Processing star {:s}.'.format(self.starID))

        if verbose: print('> Using 2d-acf to determine nu_max.')
        numax_diagnostics = self.get_numax()
        if verbose: print('  2d-acf numax: {:0.3f}'.format(numax_diagnostics['cacf_numax']))

        if verbose: print("> Using acf to determine Delta_nu.")
        dnu_diagnostics = self.get_dnu(numax_diagnostics['cacf_numax'])
        if verbose: print('  ACF dnu: {:0.3f}'.format(dnu_diagnostics['acf_dnu']))

        if verbose: print("> Fitting background to accurately determine nu_max.")
        bg_diagnostics = self.get_background(numax_diagnostics['cacf_numax'], verbose=verbose)
        if verbose: print('  NHarvey: {:0.0f}'.format(bg_diagnostics['NHarvey']))
        if verbose: print('  Fitted numax: {:0.3f}'.format(bg_diagnostics['paramsMax'][2]))

        # save
        if verbose: print('> Plotting.')
        self.to_plot(numax_diagnostics, dnu_diagnostics, bg_diagnostics, 1)

        if verbose: print('> Saving.')
        self.to_data(cacf_numax=numax_diagnostics, acf_dnu=dnu_diagnostics,
                     psfit_bg=bg_diagnostics)
        return

    # def run2(self, numax_diagnostics, dnu_diagnostics, bg_diagnostics, verbose=True):
    #     if verbose: print('>>> Processing star {:s}.'.format(self.starID))

    #     if verbose: print('> Fitting p-mode asymptotics.')
    #     pmode_diagnostics = self.get_pmode_asymp(numax_diagnostics['cacf_numax'])

    #     if verbose: print('> Plotting.')
    #     self.to_plot(numax_diagnostics, dnu_diagnostics, bg_diagnostics, pmode_diagnostics)

    #     # if verbose: print('> Saving.')
    #     # self.to_data(cacf_numax=numax_diagnostics, acf_dnu=dnu_diagnostics,
    #     #              psfit_bg=bg_diagnostics, pmode_asymp=pmode_diagnostics)
    #     return

    def get_numax(self):
        # use 2d ACF to get a rough estimation on numax
        freqRanges = np.logspace(min(np.log10(15.),np.log10(np.min(self.freq))), 
                                np.log10(np.max(self.freq)), 
                                250)
        # freqRanges = np.linspace(15., np.max(freq), 200.)
        freqCenters = (freqRanges[1:]+freqRanges[:-1])/2.
        spacings = np.diff(freqRanges)
        widths = 0.263*freqCenters**0.772 * 4.

        # # crude background estimation
        
        cacf = np.zeros(len(spacings))
        acf2d = np.zeros([140,len(spacings)])
        for isp, width in enumerate(widths):
            idx = (self.freq >= freqRanges[isp]) & (self.freq <= freqRanges[isp]+width)
            
            if np.sum(idx)>100:
                # powerbg = percentile_filter(self.power[idx], 15, size=int(width/np.median(np.diff(self.freq[idx]))))
                # powerbg = np.percentile(self.power[idx], 20)
                lag, rho = a_correlate(self.freq[idx], self.power[idx])  # return the acf at this freqRange
                acf = np.interp(np.arange(30,170)/200*np.max(lag), lag, rho) # switch to the scale of width
                acf2d[:, isp] = np.abs(detrend(acf))  # store the 2D acf
                cacf[isp] = np.sum(np.abs(detrend(acf)))  # store the max acf power (collapsed acf)
            else:
                acf2d[:, isp] = 0.
                cacf[isp] = np.nan

        # detrend the data to find the peak
        idx = np.isfinite(cacf)
        cacfDetrend = np.zeros(len(cacf))
        cacfDetrend[idx] = flatten(np.arange(np.sum(idx)), cacf[idx], 
                        method='biweight', window_length=50, edge_cutoff=10)

        # The highest value of the cacf (smoothed) corresponds to numax
        cacf_numax = freqCenters[np.nanargmax(cacfDetrend)]

        # create and return the object containing the result
        self.numax_diagnostics = {'cacf':cacf, 'acf2d':acf2d, 'freqCenters':freqCenters,
                    'spacings':spacings, 'widths':widths, 'freqRanges':freqRanges,
                    'cacfDetrend': cacfDetrend,
                    'cacf_numax':cacf_numax}
        return self.numax_diagnostics


    def _guessBackgroundParams(self, freq, powerSmoothed, numax):
        zeta = 2*2**0.5/np.pi
        flatNoiseLevel = np.median(powerSmoothed[int(len(self.freq)*0.9):]) 
        heightOsc = powerSmoothed[np.argmin(np.abs(freq-numax))] 
        widthOsc = 3.0 * (0.263*numax**0.772) 

        freqHarvey_solar = np.array([2440.5672465, 735.4653975, 24.298031575000003])
        numax_solar = 3050
      
        freqHarvey = numax/numax_solar*freqHarvey_solar
        powerHarvey = np.ones(3)*4.
        ampHarvey = np.zeros(3)
        for iHarvey in range(3):
            ampHarvey[iHarvey] = (powerSmoothed[np.argmin(np.abs(freq-freqHarvey[iHarvey]))]*2/zeta*freqHarvey[iHarvey])**0.5

        paramsInit = [flatNoiseLevel, heightOsc, numax, widthOsc,
                ampHarvey[0], freqHarvey[0], powerHarvey[0],
                ampHarvey[1], freqHarvey[1], powerHarvey[1],
                ampHarvey[2], freqHarvey[2], powerHarvey[2]]
        paramsBounds = [[flatNoiseLevel*0.1, flatNoiseLevel*1.1], 
                  [heightOsc*0.2, heightOsc*5.0],
                  [numax*0.8, numax*1.2],
                  [widthOsc*0.5, widthOsc*4.0],
                  [ampHarvey[0]*0.3, ampHarvey[0]*3.0],
                  [freqHarvey[0]*0.2, freqHarvey[0]*5.0],
                  [2.0, 8.0],
                  [ampHarvey[1]*0.3, ampHarvey[1]*3.0],
                  [freqHarvey[1]*0.2, freqHarvey[1]*5.0],
                  [2.0, 8.0],
                  [ampHarvey[2]*0.3, ampHarvey[2]*3.0],
                  [freqHarvey[2]*0.2, freqHarvey[2]*5.0],
                  [2.0, 8.0]]
        paramsNames = ["flatNoiseLevel", "heightOsc", "numax", "widthOsc", 
                'ampHarvey1', 'freqHarvey1', 'powerHarvey1',
                'ampHarvey2', 'freqHarvey2', 'powerHarvey2',
                'ampHarvey3', 'freqHarvey3', 'powerHarvey3']
        return paramsInit, paramsBounds, paramsNames


    def get_background(self, numax, verbose=True):
        # use background fit to get a precise estimation on numax
        # guess params
        paramsInit, paramsBounds, paramsNames = self._guessBackgroundParams(self.freq, self.powerGlobalSmoothed, numax)

        fitterOutput, fitterResidual = [[0,0,0] for i in range(2)]
        for NHarvey in range(1,4):
            def chi2(params):
                residual = np.sum((self.power-standardBackgroundModel(self.freq, 
                                params[:4+NHarvey*3], self.fnyq, NHarvey=NHarvey))**2.)
                return residual
            fitter = LSSampler(chi2, paramsInit[:4+NHarvey*3], paramsBounds[:4+NHarvey*3],
                        paramsNames=paramsNames[:4+NHarvey*3])
            fitterOutput[NHarvey-1] = fitter.run(wrapper='minimize')
            fitterResidual[NHarvey-1] = chi2(fitterOutput[NHarvey-1]['paramsMax'])
            if verbose: print('   No. of Harvey = {:0.0f}, chi2: {:0.5f}'.format(NHarvey, fitterResidual[NHarvey-1]))
        fitterDiagnostic = fitterOutput[np.nanargmin(fitterResidual)]

        NHarvey = int((len(fitterDiagnostic['paramsMax'])-4)/3)
        powerBackground = standardBackgroundModel(self.freq, fitterDiagnostic['paramsMax'], 
                                    self.fnyq, NHarvey=NHarvey, ifReturnOscillation=False)
        powerFit = standardBackgroundModel(self.freq, fitterDiagnostic['paramsMax'], 
                                    self.fnyq, NHarvey=NHarvey, ifReturnOscillation=True)
        powerSNR = self.power/powerBackground
        self.bg_diagnostics = {**fitterDiagnostic, 
                        'NHarvey':NHarvey, 
                        'powerBackground':powerBackground, 
                        'powerFit':powerFit, 'powerSNR':powerSNR}
        return self.bg_diagnostics


    def get_dnu(self, numax):
        # Determine dnu by acf
        dnu_guess = 0.263*numax**0.772
        idx = (self.freq>(numax-7*dnu_guess)) & (self.freq<(numax+7*dnu_guess))
        freq, power = self.freq[idx], self.power[idx]

        powers = smoothWrapper(freq, power, 0.1*dnu_guess, "bartlett")

        lag, acf = a_correlate(freq, powers) # acf
        acfs = smoothWrapper(lag, acf, 0.1*dnu_guess, "bartlett") # smooth acf

        # Use peak-finding algorithm to extract dnu in ACF
        idx = (lag>0.66*dnu_guess) & (lag<1.33*dnu_guess)
        index = peakutils.peak.indexes(acfs[idx], min_dist=int(dnu_guess/np.median(np.diff(freq))))

        if len(index) != 0:
            peaks_lags, peaks_amps = lag[idx][index], acfs[idx][index]
            acf_dnu = peaks_lags[np.nanargmax(peaks_amps)]
            acf_dnu_amp = peaks_amps[np.nanargmax(peaks_amps)]
        else:
            acf_dnu, acf_dnu_amp = np.nan, np.nan

        # create and return the object containing the result
        self.dnu_diagnostics = {'lag':lag, 'acf':acf, 'acfs':acfs,
                            'acf_dnu':acf_dnu,'acf_dnu_amp':acf_dnu_amp,
                            'dnu_guess':dnu_guess}

        return self.dnu_diagnostics


    def get_pmode_asymp(self, numax):
        
        dnu_guess = 0.263*numax**0.772
        idx = (self.freq>=numax-8*dnu_guess)&(self.freq<=numax+8*dnu_guess)
        x, y = self.freq[idx], self.power[idx]
        fs = np.median(np.diff(x))
        numax_j = np.nanargmin(np.abs(x-numax))

        def normal(theta, mu, sigma):
            return np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(theta-mu)**2/sigma**2

        def model(theta):
            delta_nu, dnu_01, dnu_02, A0, A1, A2, fwhm0, fwhm1, fwhm2, C, offset = theta
            Nbin = np.int(delta_nu/fs)
            yFoldObs = (y[numax_j-3*Nbin:numax_j-2*Nbin] + y[numax_j-2*Nbin:numax_j-Nbin] + 
                        y[numax_j-Nbin:numax_j] + y[numax_j:numax_j+Nbin] +
                        y[numax_j+Nbin:numax_j+Nbin*2] + y[numax_j+Nbin*2:numax_j+Nbin*3])
            yFoldObs /= np.max(yFoldObs)
            nu0 = offset*delta_nu 
            nu1 = nu0 - 0.5*delta_nu + dnu_01 
            nu2 = nu0 - dnu_02
            tx = np.linspace(0,1,Nbin)*delta_nu-delta_nu/2.
            y0 = A0/(1+(tx)**2./(fwhm0**2./4.))
            y0 = y0[np.argsort((tx+nu0)%delta_nu)]
            y1 = A1/(1+(tx)**2./(fwhm1**2./4.))
            y1 = y1[np.argsort((tx+nu1)%delta_nu)]
            y2 = A2/(1+(tx)**2./(fwhm2**2./4.))
            y2 = y2[np.argsort((tx+nu2)%delta_nu)]
            yFoldMod = y0+y1+y2+C 
            return yFoldObs, yFoldMod

        def posterior(theta):
            delta_nu, dnu_01, dnu_02, A0, A1, A2, fwhm0, fwhm1, fwhm2, C, offset = theta

            # priors for unkown model parameters
            boo = (0.7*dnu_guess<delta_nu<1.3*dnu_guess) & (0.035*dnu_guess<fwhm0<0.35*dnu_guess) \
                & (0.035*dnu_guess<fwhm1<0.35*dnu_guess) \
                & (0.035*dnu_guess<fwhm2<0.35*dnu_guess) \
                & (0.001<C<0.1) & (0.<offset<1.0) \
                & (A0>0) & (A1>0) & (A2>0)
            if boo:
                lnprior = 0.
            else:
                return -np.inf
            lnprior += normal(delta_nu, dnu_guess, 0.15*dnu_guess)
            lnprior += normal(dnu_01, -0.025*dnu_guess, 0.1*dnu_guess)
            lnprior += normal(dnu_02, 0.121*dnu_guess+0.047, 0.1*dnu_guess)
            lnprior += normal(A0, 1.0, 0.3)
            lnprior += normal(A1, 1.0, 0.3)
            lnprior += normal(A2, 0.8, 0.15)

            # expected value of outcome
            yFoldObs, yFoldMod = model(theta)

            # likelihood (sampling distribution) of observations
            lnlike = -np.sum(yFoldObs/yFoldMod+np.log(yFoldMod))*6.
            return lnprior + lnlike

        paramsInit = [dnu_guess, -0.025*dnu_guess, 0.121*dnu_guess+0.047,
                    1.0, 1.0, 0.8, 0.25*dnu_guess, 0.25*dnu_guess, 0.25*dnu_guess, 
                    0.05, 0.5]
        sampler = ESSampler(paramsInit, posterior, Nsteps=1000, Nburn=3000)
        diagnostics = sampler.run(verbose=True)
        yFoldObs, yFoldMod = model(diagnostics['paramsMax'])
        epsp = (numax/diagnostics['paramsMax'][0]+diagnostics['paramsMax'][-1]) % 1.

        self.pmode_diagnostics = {**diagnostics, 'model':model, 
            'yFoldObs':yFoldObs, 'yFoldMod':yFoldMod, 'epsp':epsp}
        return self.pmode_diagnostics


    def to_plot(self, numax_diagnostics, dnu_diagnostics, bg_diagnostics, pmode_diagnostics):
        _, axes = plt.subplots(figsize=(16,16), nrows=3, ncols=3, squeeze=False)

        # plot A, original flux
        # axes[0,0]

        # plot B, corrected flux
        # axes[1,0]

        # plot C, smoothed power spectra
        axes[2,0].plot(self.freq, self.power, color='gray')
        axes[2,0].plot(self.freq, self.powerGlobalSmoothed, color='red')
        axes[2,0].set_xlabel('$\\nu$ ($\\mu$Hz)')
        axes[2,0].set_ylabel('Power')
        axes[2,0].set_xlim(np.min(self.freq), np.max(self.freq))
        axes[2,0].set_xscale('log')
        axes[2,0].set_yscale('log')

        # plot D, 2d-acf
        axes[0,1].contourf(numax_diagnostics['freqCenters'], 
                            np.arange(0,numax_diagnostics['acf2d'].shape[0]), 
                            numax_diagnostics['acf2d'], cmap='gray_r')
        axes[0,1].set_xlabel('$\\nu$ ($\\mu$Hz)')
        axes[0,1].set_ylabel('Normalized lag')
        axes[0,1].set_xscale('log')

        # plot E, cacf
        # axes[1,1].plot(numax_diagnostics['freqCenters'], numax_diagnostics['cacf'], 'b-')
        axes[1,1].plot(numax_diagnostics['freqCenters'], numax_diagnostics['cacf'], color='C0')
        axes[1,1].plot(numax_diagnostics['freqCenters'], numax_diagnostics['cacfDetrend'], color='gray')
        axes[1,1].axvline(numax_diagnostics['cacf_numax'], color='r',linestyle='--')
        axes[1,1].set_xlabel('$\\nu$ ($\\mu$Hz)')
        axes[1,1].set_ylabel('Collapsed 2d-ACF')
        axes[1,1].set_xscale('log')
        axes[1,1].text(0.95,0.95,'2d-acf numax: {:0.3f}'.format(numax_diagnostics['cacf_numax']), 
                        transform=axes[1,1].transAxes, va='top', ha='right')


        # plot F, fitted power spectra
        powerBackground = standardBackgroundModel(self.freq, bg_diagnostics['paramsMax'], 
                            self.fnyq, NHarvey=bg_diagnostics['NHarvey'], ifReturnOscillation=False)
        powerFit = standardBackgroundModel(self.freq, bg_diagnostics['paramsMax'], 
                            self.fnyq, NHarvey=bg_diagnostics['NHarvey'], ifReturnOscillation=True)
        axes[2,1].plot(self.freq, self.power, color='gray')
        axes[2,1].plot(self.freq, self.powerGlobalSmoothed, color='black')
        axes[2,1].plot(self.freq, powerBackground, color='green')
        axes[2,1].plot(self.freq, powerFit, color='green', linestyle='--')
        axes[2,1].axhline(bg_diagnostics['paramsMax'][0], color='black', linestyle='--')
        axes[2,1].axvline(bg_diagnostics['paramsMax'][2], color='red', linestyle='--')
        axes[2,1].set_xlabel('$\\nu$ ($\\mu$Hz)')
        axes[2,1].set_ylabel('Power')
        axes[2,1].set_xlim(np.min(self.freq), np.max(self.freq))
        axes[2,1].set_xscale('log')
        axes[2,1].set_yscale('log')
        axes[2,1].text(0.05,0.05,'NHarvey: {:0.0f}'.format(bg_diagnostics['NHarvey']),
                        transform=axes[2,1].transAxes, va='bottom', ha='left')
        axes[2,1].text(0.05,0.10,'Fitted numax: {:0.3f}'.format(bg_diagnostics['paramsMax'][2]),
                        transform=axes[2,1].transAxes, va='bottom', ha='left')

        # plot G, acf
        axes[0,2].plot(dnu_diagnostics['lag'], dnu_diagnostics['acf'])
        axes[0,2].axvline(dnu_diagnostics['dnu_guess']*0.66, color='gray',linestyle='--')
        axes[0,2].axvline(dnu_diagnostics['dnu_guess']*1.33, color='gray',linestyle='--')
        axes[0,2].plot([dnu_diagnostics['acf_dnu']], [dnu_diagnostics['acf_dnu_amp']], 'rx')
        axes[0,2].set_xlabel('$\\nu$ ($\\mu$Hz)')
        axes[0,2].set_ylabel('ACF')
        axes[0,2].text(0.95,0.95,'ACF dnu: {:0.3f}'.format(dnu_diagnostics['acf_dnu']), 
                        transform=axes[0,2].transAxes, va='top', ha='right')

        # # plot H, asymptotic p fitting
        # x = np.linspace(0,1,len(pmode_diagnostics['yFoldObs']))*pmode_diagnostics['paramsMax'][0]
        # axes[1,2].plot(x, pmode_diagnostics['yFoldObs'], color='C0')
        # axes[1,2].plot(x, pmode_diagnostics['yFoldMod'], color='black')
        # axes[1,2].axvline(pmode_diagnostics['paramsMax'][0]*pmode_diagnostics['paramsMax'][-1], color='red', linestyle='--')
        # axes[1,2].text(0.95,0.95,'aysmp dnu: {:0.3f}'.format(pmode_diagnostics['paramsMax'][0]), 
        #                 transform=axes[1,2].transAxes, va='top', ha='right')
        # axes[1,2].text(0.95,0.90,'aysmp eps: {:0.3f}'.format(pmode_diagnostics['epsp']), 
        #                 transform=axes[1,2].transAxes, va='top', ha='right')


        # plot I, echelle
        # axes[2,2]
        numax = numax_diagnostics['cacf_numax']
        dnu = 0.263*numax**0.772
        powerSmoothed = smoothWrapper(self.freq, self.power, dnu*0.03, 'flat')
        echx, echy, echz = echelle(self.freq, powerSmoothed, 
                    dnu, numax-dnu*8, numax+dnu*8, echelletype="replicated")
        levels = np.linspace(np.min(echz), np.max(echz), 500)
        axes[2,2].contourf(echx, echy, echz, cmap="jet", levels=levels)
        axes[2,2].axis([np.min(echx), np.max(echx), np.min(echy), np.max(echy)])
        axes[2,2].set_xlabel("$\\nu$  mod {:0.2f} ($\\mu$Hz)".format(dnu))
        axes[2,2].set_ylabel('$\\nu$ ($\\mu$Hz)')
        # axes[2,2].axvline(dnu, color='black', linestyle='--')
        # axes[2,2].axhline(bg_diagnostics['paramsMax'][2], color='black', linestyle='--')

        # save
        plt.savefig(self.filepath+self.starID+'.png')
        plt.close() 
        return

    def to_data(self, **kwargs):
        data = kwargs
        np.save(self.filepath+self.starID+'', data)
        return