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
from geneticalgorithm import geneticalgorithm

from scipy.optimize import curve_fit

__all__ = ['solarlikeGlobalSeismo']


def model_cacf(x, *coeff, ifIncludeGauss=True):
    A, mu, sigma, k, c, b = coeff
    if ifIncludeGauss==True:
        return A*np.exp(-(x-mu)**2/(2.*sigma**2)) + k*x**c+ b
    else:
        return k*x**c+ b


class solarlikeGlobalSeismo:
    '''
    Under development:
    # 1: p-mode asymptotic fitting
    # 2: light curve processing
    # 3: g-mode asymptotic fitting
    '''
    def __init__(self, freq, power, fnyq, fmin=None, fmax=None, starID='', filepath=''):
        self.res = np.median(np.abs(np.diff(np.sort(freq))))
        self.fmin = self.res if (fmin is None) else fmin
        self.fmax = np.nanmax(freq) if (fmax is None) else fmax

        idx = (freq>self.fmin) & (freq<self.fmax)

        self.freq, self.power = freq[idx], power[idx]
        self.fnyq = fnyq
        self.powerGlobalSmoothed = powerSpectrumSmoothWrapper(self.freq, self.power, windowSize=0.5)

        self.starID = starID if starID != '' else 'star'
        self.filepath = filepath if filepath != '' else './'
        return

    def run(self, verbose=True, get_numax_kwargs={}, get_Dnu_kwargs={}, 
            get_background_kwargs={}, get_pmode_asymp_kwargs={},
            skip_numax=False, skip_Dnu=False, skip_background=False,
            skip_pmode_asymp=False, skip_save_data=False, skip_plot=False):
        if verbose: print('>>> Processing star {:s}.'.format(self.starID))

        if not skip_numax:
            if verbose: print('> Using 2d-acf to determine nu_max.')
            numax_diagnostics = self.get_numax(**get_numax_kwargs)
            if verbose: print('  2d-acf numax: {:0.3f}'.format(numax_diagnostics['cacf_numax']))
        else:
            numax_diagnostics = None

        if not skip_Dnu:
            if verbose: print("> Using acf to determine Delta_nu.")
            Dnu_diagnostics = self.get_Dnu(numax_diagnostics['cacf_numax'], 
                                        **get_Dnu_kwargs)
            if verbose: print('  ACF Dnu: {:0.3f}'.format(Dnu_diagnostics['acf_Dnu']))
            if verbose: print('  Collapse Dnu: {:0.3f}'.format(Dnu_diagnostics['collapse_Dnu']))
        else:
            Dnu_diagnostics = None

        if not skip_background:
            if verbose: print("> Fitting background to accurately determine nu_max.")
            bg_diagnostics = self.get_background(numax_diagnostics['cacf_numax'], 
                                                verbose=verbose,
                                                **get_background_kwargs)
            if verbose: print('  NHarvey: {:0.0f}'.format(bg_diagnostics['NHarvey']))
            if verbose: print('  Fitted numax: {:0.3f}'.format(bg_diagnostics['paramsMax'][2]))
        else:
            bg_diagnostics = None

        if not skip_pmode_asymp:
            if verbose: print('> Fitting p mode asymptotic parameters (epsp, alphap, Dnu).')
            pmode_diagnostics = self.get_pmode_asymp(bg_diagnostics['paramsMax'][1], # heightOsc
                                                    bg_diagnostics['paramsMax'][2], # numax
                                                    bg_diagnostics['paramsMax'][3], # widthOsc
                                                    Dnu_diagnostics['acf_Dnu'], # Dnu
                                                    **get_pmode_asymp_kwargs) 
            if verbose: print('  Fitted epsp: {:0.3f}'.format(pmode_diagnostics['epsp'])) 
            if verbose: print('  Fitted alphap: {:0.5f}'.format(pmode_diagnostics['alphap'])) 
            if verbose: print('  Fitted Dnu: {:0.3f}'.format(pmode_diagnostics['Dnu']))                                     
        else:
            pmode_diagnostics = None

        if not skip_plot:
            if verbose: print('> Plotting.')
            self.to_plot(numax_diagnostics=numax_diagnostics, 
                         Dnu_diagnostics=Dnu_diagnostics, 
                         bg_diagnostics=bg_diagnostics, 
                         pmode_diagnostics=pmode_diagnostics)

        if not skip_save_data:
            if verbose: print('> Saving.')
            self.to_data(cacf_numax=numax_diagnostics, acf_Dnu=Dnu_diagnostics,
                        psfit_bg=bg_diagnostics, pmode_asymp=pmode_diagnostics)
        return

    def run2(self, numax_diagnostics, Dnu_diagnostics, bg_diagnostics, verbose=True):
        if verbose: print('> Fitting p mode asymptotic parameters (epsp, alphap, Dnu).')
        pmode_diagnostics = self.get_pmode_asymp(bg_diagnostics['paramsMax'][1], # heightOsc
                                                bg_diagnostics['paramsMax'][2], # numax
                                                bg_diagnostics['paramsMax'][3], # widthOsc
                                                Dnu_diagnostics['acf_Dnu']) # Dnu
        if verbose: print('  Fitted epsp: {:0.3f}'.format(pmode_diagnostics['epsp'])) 
        if verbose: print('  Fitted alphap: {:0.5f}'.format(pmode_diagnostics['alphap'])) 
        if verbose: print('  Fitted Dnu: {:0.3f}'.format(pmode_diagnostics['Dnu']))       

        # save
        if verbose: print('> Plotting.')
        self.to_plot(numax_diagnostics, Dnu_diagnostics, bg_diagnostics, pmode_diagnostics)

        # if verbose: print('> Saving.')
        # self.to_data(cacf_numax=numax_diagnostics, acf_Dnu=Dnu_diagnostics,
        #              psfit_bg=bg_diagnostics, pmode_asymp=pmode_diagnostics)
        return

    def get_numax(self):
        fmin, fmax = np.min(self.freq), np.max(self.freq)
        res = self.res
        fmin = fmin if fmin>0 else self.res
        freq = np.concatenate([self.freq, self.freq+np.max(self.freq)])
        power = np.concatenate([self.power, np.interp(-self.freq+np.max(self.freq), self.freq, self.power)])

        # use 2d ACF to get a rough estimation on numax
        freq_ranges = np.logspace(np.log10(fmin), np.log10(fmax), 250)
        freq_centers = (freq_ranges[1:]+freq_ranges[:-1])/2.
        spacings = np.diff(freq_ranges)
        widths = 0.263*freq_centers**0.772 * 4.

        # # crude background estimation
        
        cacf = np.zeros(len(spacings))+np.nan
        freq_lag = np.arange(res, int(widths[-1]/res)+2)*res
        acf2d = np.zeros([len(freq_lag),len(spacings)])+np.nan

        for isp, width in enumerate(widths):
            idx = (freq >= freq_ranges[isp]) & (freq <= freq_ranges[isp]+width)
            
            if np.sum(idx)>10:
                lag, rho = a_correlate(freq[idx], power[idx])  # return the acf at this freqRange
                lagidx = (freq_lag>0.15*np.max(lag) ) & (freq_lag < 0.85*np.max(lag))
                acf = np.interp(freq_lag[lagidx], lag, rho) # switch to the scale of width
                acf2d[lagidx, isp] = np.abs(detrend(acf))  # store the 2D acf
                cacf[isp] = np.max(np.abs(detrend(acf)))  # store the max acf power (collapsed acf)

        # remove nan numbers
        ridx = np.nansum(acf2d, axis=1) != 0
        cidx = np.nansum(acf2d, axis=0) != 0
        acf2d = acf2d[ridx, :][:,cidx]
        freq_centers, widths, spacings, cacf = freq_centers[cidx], widths[cidx], spacings[cidx], cacf[cidx]
        freq_lag = freq_lag[ridx]

        # detrend the data to find the peak
        idx = np.isfinite(cacf)
        cacf_detrend = np.zeros(len(cacf))
        cacf_detrend[idx] = flatten(np.arange(np.sum(idx)), cacf[idx], 
                        method='biweight', window_length=50, edge_cutoff=10)

        # The highest value of the cacf (smoothed) corresponds to numax
        cacf_numax = freq_centers[np.nanargmax(cacf)] 

        # Gaussian fit 
        idx = np.isfinite(freq_centers) & np.isfinite(cacf)
        cacf_fitted = np.zeros(len(freq_centers)) + np.nan
        if np.sum(idx) > 7:
            freq_centers_finite, cacf_finite = freq_centers[idx], cacf[idx]

            p0 = [np.max(cacf_finite)-np.min(cacf_finite), #A
                  freq_centers_finite[np.argmax(cacf_finite)], #mu
                  0.263*freq_centers_finite[np.argmax(cacf_finite)]**0.772 * 4., #sigma
                  0, #k
                  -2, #c
                  np.min(cacf_finite)] #b

            cacf_fitted_coeff, _ = curve_fit(model_cacf, freq_centers_finite, cacf_finite, p0=p0)

            # Get the fitted curve
            cacf_fitted[idx] = model_cacf(freq_centers_finite, *cacf_fitted_coeff)

        else:
            cacf_fitted_coeff = np.zeros(6)+np.nan

        # create and return the object containing the result
        self.numax_diagnostics = {'cacf':cacf, 'acf2d':acf2d, 'freq_centers':freq_centers, 'freq_lag':freq_lag,
                    'spacings':spacings, 'widths':widths, 
                    'cacf_detrend': cacf_detrend,
                    'cacf_numax':cacf_numax,
                    'cacf_fitted':cacf_fitted,
                    'cacf_fitted_coeff':cacf_fitted_coeff}
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
                  [numax*0.5, numax*2.0],
                  [widthOsc*0.1, widthOsc*4.0],
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


    def get_background(self, numax, verbose=True, sampler='ensemble', Npower=None):
        # use background fit to get a precise estimation on numax
        # guess params
        freq, power, fnyq, powers = self.freq, self.power, self.fnyq, self.powerGlobalSmoothed
        paramsInit, paramsBounds, paramsNames = self._guessBackgroundParams(freq, powers, numax)
        
        filepath = self.filepath+'background_'

        fitters = []
        if (Npower is None):
            NHarveys = np.arange(1,3.1,1)
            fitter_output, fitter_residual = [[0,0,0] for i in range(2)]
        else:
            NHarveys = np.array([Npower])
            fitter_output, fitter_residual = [[0] for i in range(2)]

        for ibg, NHarvey in enumerate(NHarveys):
            def chi2(params):
                power_model = standardBackgroundModel(freq, params[:4+NHarvey*3], fnyq, NHarvey=NHarvey)
                residual = np.sum(np.log(power_model) + power/power_model)
                return residual

            def lnpost(params, paramsBounds):
                for ipara, para in enumerate(params):
                    if not (paramsBounds[ipara][0] <= para <= paramsBounds[ipara][1]):
                        return -np.inf
                power_model = standardBackgroundModel(freq, params[:4+NHarvey*3], fnyq, NHarvey=NHarvey)
                lnpost = -np.sum(np.log(power_model) + power/power_model)
                return lnpost

            if sampler=='ensemble':
                fitter = ESSampler(lnpost, paramsInit[:4+NHarvey*3], paramsBounds[:4+NHarvey*3], paramsNames=paramsNames[:4+NHarvey*3])
                fitters.append(fitter)
                fitter_output[ibg] = fitter.run(verbose=verbose)
                fitter_residual[ibg] = chi2(fitter_output[ibg]['paramsMax'])
            elif sampler=='basinhopping':
                fitter = LSSampler(chi2, paramsInit[:4+NHarvey*3], paramsBounds[:4+NHarvey*3],
                            paramsNames=paramsNames[:4+NHarvey*3])
                fitters.append(fitter)
                fitter_output[ibg] = fitter.run(wrapper='basinhopping')
                fitter_residual[ibg] = chi2(fitter_output[ibg]['paramsMax'])
            else: # sampler=='minimize':
                fitter = LSSampler(chi2, paramsInit[:4+NHarvey*3], paramsBounds[:4+NHarvey*3],
                            paramsNames=paramsNames[:4+NHarvey*3])
                fitters.append(fitter)
                fitter_output[ibg] = fitter.run(wrapper='minimize')
                fitter_residual[ibg] = chi2(fitter_output[ibg]['paramsMax'])

            if verbose: print('   No. of Harvey = {:0.0f}, chi2: {:0.5f}'.format(NHarvey, fitter_residual[ibg]))
        fitterDiagnostic = fitter_output[np.nanargmin(fitter_residual)]
        if sampler=='ensemble':
            fitters[np.nanargmin(fitter_residual)].to_plot(filepath)

        NHarvey = int((len(fitterDiagnostic['paramsMax'])-4)/3)
        power_background = standardBackgroundModel(freq, fitterDiagnostic['paramsMax'], 
                                    fnyq, NHarvey=NHarvey, ifReturnOscillation=False)
        power_fit = standardBackgroundModel(freq, fitterDiagnostic['paramsMax'], 
                                    fnyq, NHarvey=NHarvey, ifReturnOscillation=True)
        power_snr = power/power_background
        self.bg_diagnostics = {**fitterDiagnostic, 
                        'freq': freq, 'power': power, 'fnyq':fnyq, 'powers':powers,
                        'NHarvey':NHarvey, 
                        'power_background':power_background, 
                        'power_fit':power_fit, 'power_snr':power_snr}
        return self.bg_diagnostics

    def get_collapsed_spectrum(self, numax, freq, power, collapseLength):
        freqStart = int(numax/collapseLength-5+0.5)*collapseLength #49.0192
        freqGrid = np.arange(0., collapseLength, 0.001)
        collapse_power = np.zeros(len(freqGrid))
        for i in range(11):
            collapse_power += np.interp(freqStart+i*collapseLength+freqGrid, freq, power)

        freq_collapse = np.concatenate([freqGrid, freqGrid+collapseLength])
        power_collapse = np.concatenate([collapse_power, collapse_power])
        powers_collapse = smoothWrapper(freq_collapse, power_collapse, collapseLength/10., 'bartlett')

        return freq_collapse, power_collapse, powers_collapse

    def get_Dnu(self, numax):
        # method 1 - Determine Dnu by acf
        Dnu_guess = 0.263*numax**0.772
        idx = (self.freq>(numax-7*Dnu_guess)) & (self.freq<(numax+7*Dnu_guess))
        freq, power = self.freq[idx], self.power[idx]

        powers = smoothWrapper(freq, power, 0.1*Dnu_guess, "bartlett")

        lag, acf = a_correlate(freq, powers) # acf
        acfs = smoothWrapper(lag, acf, 0.1*Dnu_guess, "bartlett") # smooth acf

        # Use peak-finding algorithm to extract Dnu in ACF
        idx = (lag>0.66*Dnu_guess) & (lag<1.33*Dnu_guess)
        index = peakutils.peak.indexes(acfs[idx], min_dist=int(Dnu_guess/np.median(np.diff(freq))))

        if len(index) != 0:
            peaks_lags, peaks_amps = lag[idx][index], acfs[idx][index]
            acf_Dnu = peaks_lags[np.nanargmax(peaks_amps)]
            acf_Dnu_amp = peaks_amps[np.nanargmax(peaks_amps)]
        else:
            acf_Dnu, acf_Dnu_amp = np.nan, np.nan

        # method 2 - Determine Dnu by collapsing the power spectrum

        length = Dnu_guess + np.arange(-0.33,0.33,np.median(np.diff(freq))/Dnu_guess)*Dnu_guess
        maxPower = np.zeros(len(length)) 
        for iunit, unit in enumerate(length):
            # collapsed diagrams
            _, _, powers_collapse = self.get_collapsed_spectrum(numax, freq, power, unit)
            maxPower[iunit] = np.max(powers_collapse)

        collapse_Dnu = length[np.argmax(maxPower)]
        collapse_Dnu_power = maxPower[np.argmax(maxPower)]

        # collapsed spectrum for plotting
        freq_collapse, power_collapse, powers_collapse = self.get_collapsed_spectrum(numax, freq, power, collapse_Dnu)

        # create and return the object containing the result
        self.Dnu_diagnostics = {'Dnu_guess':Dnu_guess,
                            'lag':lag, 'acf':acf, 'acfs':acfs,
                            'acf_Dnu':acf_Dnu, 'acf_Dnu_amp':acf_Dnu_amp,
                            'length':length, 'maxPower':maxPower, 
                            'collapse_Dnu':collapse_Dnu, 'collapse_Dnu_power':collapse_Dnu_power,
                            'freq_collapse':freq_collapse, 'power_collapse':power_collapse, 'powers_collapse':powers_collapse}
        return self.Dnu_diagnostics


    # def get_pmode_asymp(self, numax):
        
    #     Dnu_guess = 0.263*numax**0.772
    #     idx = (self.freq>=numax-8*Dnu_guess)&(self.freq<=numax+8*Dnu_guess)
    #     x, y = self.freq[idx], self.power[idx]
    #     fs = np.median(np.diff(x))
    #     numax_j = np.nanargmin(np.abs(x-numax))

    #     def normal(theta, mu, sigma):
    #         return np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(theta-mu)**2/sigma**2

    #     def model(theta):
    #         delta_nu, dnu_01, dnu_02, A0, A1, A2, fwhm0, fwhm1, fwhm2, C, offset = theta
    #         Nbin = np.int(delta_nu/fs)
    #         yFoldObs = (y[numax_j-3*Nbin:numax_j-2*Nbin] + y[numax_j-2*Nbin:numax_j-Nbin] + 
    #                     y[numax_j-Nbin:numax_j] + y[numax_j:numax_j+Nbin] +
    #                     y[numax_j+Nbin:numax_j+Nbin*2] + y[numax_j+Nbin*2:numax_j+Nbin*3])
    #         yFoldObs /= np.max(yFoldObs)
    #         nu0 = offset*delta_nu 
    #         nu1 = nu0 - 0.5*delta_nu + dnu_01 
    #         nu2 = nu0 - dnu_02
    #         tx = np.linspace(0,1,Nbin)*delta_nu-delta_nu/2.
    #         y0 = A0/(1+(tx)**2./(fwhm0**2./4.))
    #         y0 = y0[np.argsort((tx+nu0)%delta_nu)]
    #         y1 = A1/(1+(tx)**2./(fwhm1**2./4.))
    #         y1 = y1[np.argsort((tx+nu1)%delta_nu)]
    #         y2 = A2/(1+(tx)**2./(fwhm2**2./4.))
    #         y2 = y2[np.argsort((tx+nu2)%delta_nu)]
    #         yFoldMod = y0+y1+y2+C 
    #         return yFoldObs, yFoldMod

    #     def posterior(theta):
    #         delta_nu, dnu_01, dnu_02, A0, A1, A2, fwhm0, fwhm1, fwhm2, C, offset = theta

    #         # priors for unkown model parameters
    #         boo = (0.7*Dnu_guess<delta_nu<1.3*Dnu_guess) & (0.035*Dnu_guess<fwhm0<0.35*Dnu_guess) \
    #             & (0.035*Dnu_guess<fwhm1<0.35*Dnu_guess) \
    #             & (0.035*Dnu_guess<fwhm2<0.35*Dnu_guess) \
    #             & (0.001<C<0.1) & (0.<offset<1.0) \
    #             & (A0>0) & (A1>0) & (A2>0)
    #         if boo:
    #             lnprior = 0.
    #         else:
    #             return -np.inf
    #         lnprior += normal(delta_nu, Dnu_guess, 0.15*Dnu_guess)
    #         lnprior += normal(dnu_01, -0.025*Dnu_guess, 0.1*Dnu_guess)
    #         lnprior += normal(dnu_02, 0.121*Dnu_guess+0.047, 0.1*Dnu_guess)
    #         lnprior += normal(A0, 1.0, 0.3)
    #         lnprior += normal(A1, 1.0, 0.3)
    #         lnprior += normal(A2, 0.8, 0.15)

    #         # expected value of outcome
    #         yFoldObs, yFoldMod = model(theta)

    #         # likelihood (sampling distribution) of observations
    #         lnlike = -np.sum(yFoldObs/yFoldMod+np.log(yFoldMod))*6.
    #         return lnprior + lnlike

    #     paramsInit = [Dnu_guess, -0.025*Dnu_guess, 0.121*Dnu_guess+0.047,
    #                 1.0, 1.0, 0.8, 0.25*Dnu_guess, 0.25*Dnu_guess, 0.25*Dnu_guess, 
    #                 0.05, 0.5]
    #     sampler = ESSampler(paramsInit, posterior, Nsteps=1000, Nburn=3000)
    #     diagnostics = sampler.run(verbose=True)
    #     yFoldObs, yFoldMod = model(diagnostics['paramsMax'])
    #     epsp = (numax/diagnostics['paramsMax'][0]+diagnostics['paramsMax'][-1]) % 1.

    #     self.pmode_diagnostics = {**diagnostics, 'model':model, 
    #         'yFoldObs':yFoldObs, 'yFoldMod':yFoldMod, 'epsp':epsp}
    #     return self.pmode_diagnostics

    def get_pmode_asymp(self, height, numax, width, Dnu):
        # Gaussian envolope corrected power spectrum
        freq, power = self.freq, self.power
        powers = smoothWrapper(self.freq, self.power, Dnu/30.0, 'bartlett')

        idx = (freq>=numax-2*width)&(freq<=numax+2*width)
        freq, power, powers = freq[idx], power[idx], powers[idx]
        weight = height*np.exp(-(freq-numax)**2.0/(2*width**2.))
        meanBGLevel = np.percentile(power, 20)

        power /= weight*meanBGLevel
        powers /= weight*meanBGLevel

        x, y, ys = freq, power, powers

        def merit(theta):
            tepsp, tDnu, talphap, td02 = theta
            ns = np.arange((numax-width*1.5)/tDnu, (numax+width*1.5)/tDnu, 1, dtype=int)
            A = talphap
            B = -2*numax*talphap-2*tDnu
            C = numax**2.*talphap + 2*tDnu**2.*(ns+tepsp)
            nu0s = (-B-(B**2-4*A*C)**0.5)/(2*A)
            nu2s = nu0s - td02*tDnu
            
            freqGrid = np.arange(-tDnu/20., tDnu/20., 0.001)
            powerCollapse = np.zeros(len(freqGrid))
            for nu0 in nu0s:
                powerCollapse += np.log(np.interp(freqGrid+nu0, x, ys))
            for nu2 in nu2s:
                powerCollapse += np.log(np.interp(freqGrid+nu2, x, ys))
            powerCollapse = np.exp(powerCollapse) * np.bartlett(len(freqGrid))
            metric = np.max(powerCollapse)
            
            return -metric

        bounds = np.array([[0., 1.0], #epsp
                         [Dnu*0.8, Dnu*1.2], # Dnu
                         [0., 0.008], # alphap
                         [0.05, 0.15]]) # d02

        model=geneticalgorithm(function=merit,dimension=4,variable_type='real',variable_boundaries=bounds)
        model.run()

        solution = model.output_dict['variable']

        epsp, Dnu, alphap, d02 = solution
        ns = np.arange((numax-width)/Dnu, (numax+width)/Dnu, 1, dtype=int)
        A = alphap
        B = -2*numax*alphap-2*Dnu
        C = numax**2.*alphap + 2*Dnu**2.*(ns+epsp)
        nu0s = (-B-(B**2-4*A*C)**0.5)/(2*A)
        nu2s = nu0s - d02*Dnu

        self.pmode_diagnostics = {'freq':freq, 'power':power, 'powers':powers,
                    'height':height, 'numax':numax, 'width':width,
                    'epsp':epsp, 'Dnu':Dnu, 
                    'alphap':alphap, 'd02':d02,
                    'nu0s':nu0s, 'nu2s':nu2s}
        return self.pmode_diagnostics


    def to_plot(self, numax_diagnostics=None, Dnu_diagnostics=None, bg_diagnostics=None, pmode_diagnostics=None):
        _, axes = plt.subplots(figsize=(16,16), nrows=3, ncols=3, squeeze=False)

        # plot A, flux in thirty days
        # axes[0,0]

        # plot B, corrected flux
        # axes[1,0]

        # col 1: numax
        # plot A, 2d-acf
        if not (numax_diagnostics is None):
            axes[0,0].contourf(numax_diagnostics['freq_centers'], 
                                np.arange(0,numax_diagnostics['acf2d'].shape[0]), 
                                numax_diagnostics['acf2d'], cmap='gray_r')
            axes[0,0].set_xlabel('$\\nu$ ($\\mu$Hz)')
            axes[0,0].set_ylabel('Normalized lag')
            axes[0,0].set_xscale('log')

            # plot B, cacf
            # axes[1,0].plot(numax_diagnostics['freq_centers'], numax_diagnostics['cacf'], 'b-')
            axes[1,0].plot(numax_diagnostics['freq_centers'], numax_diagnostics['cacf'], color='C0')
            # axes[1,0].plot(numax_diagnostics['freq_centers'], numax_diagnostics['cacf_detrend'], color='gray')
            axes[1,0].plot(numax_diagnostics['freq_centers'], numax_diagnostics['cacf_fitted'], 'r-')
            axes[1,0].axvline(numax_diagnostics['cacf_numax'], color='C0',linestyle='--')
            axes[1,0].set_xlabel('$\\nu$ ($\\mu$Hz)')
            axes[1,0].set_ylabel('Collapsed 2d-ACF')
            axes[1,0].set_xscale('log')
            axes[1,0].text(0.95,0.95,'2d-acf numax: {:0.3f}'.format(numax_diagnostics['cacf_numax']), 
                            transform=axes[1,0].transAxes, va='top', ha='right')

        # plot C, fitted power spectra
        if not (bg_diagnostics is None):
            power_background = standardBackgroundModel(bg_diagnostics['freq'], bg_diagnostics['paramsMax'], 
                                bg_diagnostics['fnyq'], NHarvey=bg_diagnostics['NHarvey'], ifReturnOscillation=False)
            power_fit = standardBackgroundModel(bg_diagnostics['freq'], bg_diagnostics['paramsMax'], 
                                bg_diagnostics['fnyq'], NHarvey=bg_diagnostics['NHarvey'], ifReturnOscillation=True)
            axes[2,0].plot(bg_diagnostics['freq'], bg_diagnostics['power'], color='gray')
            axes[2,0].plot(bg_diagnostics['freq'], bg_diagnostics['powers'], color='black')
            axes[2,0].plot(bg_diagnostics['freq'], bg_diagnostics['power_background'], color='green')
            axes[2,0].plot(bg_diagnostics['freq'], bg_diagnostics['power_fit'], color='green', linestyle='--')
            axes[2,0].axhline(bg_diagnostics['paramsMax'][0], color='black', linestyle='--')
            axes[2,0].axvline(bg_diagnostics['paramsMax'][2], color='red', linestyle='--')
            axes[2,0].set_xlabel('$\\nu$ ($\\mu$Hz)')
            axes[2,0].set_ylabel('Power')
            axes[2,0].set_xlim(np.min(bg_diagnostics['freq']), np.max(bg_diagnostics['freq']))
            axes[2,0].set_xscale('log')
            axes[2,0].set_yscale('log')
            axes[2,0].text(0.05,0.05,'NHarvey: {:0.0f}'.format(bg_diagnostics['NHarvey']),
                            transform=axes[2,0].transAxes, va='bottom', ha='left')
            axes[2,0].text(0.05,0.10,'Fitted numax: {:0.3f}'.format(bg_diagnostics['paramsMax'][2]),
                            transform=axes[2,0].transAxes, va='bottom', ha='left')

        # col 2: Dnu
        # plot D, acf Dnu
        if not (Dnu_diagnostics is None):
            axes[0,1].plot(Dnu_diagnostics['lag'], Dnu_diagnostics['acf'])
            axes[0,1].axvline(Dnu_diagnostics['Dnu_guess']*0.66, color='gray',linestyle='--')
            axes[0,1].axvline(Dnu_diagnostics['Dnu_guess']*1.33, color='gray',linestyle='--')
            axes[0,1].plot([Dnu_diagnostics['acf_Dnu']], [Dnu_diagnostics['acf_Dnu_amp']], 'rx')
            axes[0,1].set_xlabel('$\\nu$ ($\\mu$Hz)')
            axes[0,1].set_ylabel('ACF')
            axes[0,1].text(0.95,0.95,'ACF Dnu: {:0.3f}'.format(Dnu_diagnostics['acf_Dnu']), 
                            transform=axes[0,1].transAxes, va='top', ha='right')

            # plot E, collapse Dnu
            axes[1,1].plot(Dnu_diagnostics['length'], Dnu_diagnostics['maxPower'])
            axes[1,1].axvline(Dnu_diagnostics['collapse_Dnu'], color='red',linestyle='--')
            axes[1,1].set_xlabel('Length ($\\mu$Hz)')
            axes[1,1].set_ylabel('Maximum Collapsed Power')
            axes[1,1].text(0.95,0.95,'Collapse Dnu: {:0.3f}'.format(Dnu_diagnostics['collapse_Dnu']), 
                            transform=axes[1,1].transAxes, va='top', ha='right')
            axes[1,1].axvline(Dnu_diagnostics['collapse_Dnu'], color='r',linestyle='--')


            # plot F, collapsed power spectrum
            axes[2,1].plot(Dnu_diagnostics['freq_collapse'], Dnu_diagnostics['power_collapse'])
            axes[2,1].plot(Dnu_diagnostics['freq_collapse'], Dnu_diagnostics['powers_collapse'])
            axes[2,1].axvline(Dnu_diagnostics['collapse_Dnu'], color='gray',linestyle='--')
            axes[2,1].set_xlabel('Frequency ($\\mu$Hz)')
            axes[2,1].set_ylabel('Collapsed Power')

        # col 3: asymptotics
        # plot G, echelle, solution from p mode asymptotics
        if not (pmode_diagnostics is None):
            numax, Dnu = pmode_diagnostics['numax'], pmode_diagnostics['Dnu']
            epsp, alphap, width = pmode_diagnostics['epsp'], pmode_diagnostics['alphap'], pmode_diagnostics['width']
            nu0s, nu2s = pmode_diagnostics['nu0s'], pmode_diagnostics['nu2s']
            powerSmoothed = smoothWrapper(self.freq, self.power, Dnu*0.03, 'flat')
            echx, echy, echz = echelle(self.freq, powerSmoothed, 
                        Dnu, numax-width*4., numax+width*4., echelletype="replicated")
            levels = np.linspace(np.min(echz), np.max(echz), 500)
            axes[0,2].contourf(echx, echy, echz, cmap="gray_r", levels=levels)
            axes[0,2].scatter(nu0s%Dnu, nu0s-(nu0s%Dnu)+0.5*Dnu, marker='o', edgecolor='blue', facecolor='none')
            axes[0,2].scatter(nu0s%Dnu+Dnu, nu0s-(nu0s%Dnu)-0.5*Dnu, marker='o', edgecolor='blue', facecolor='none')
            axes[0,2].scatter(nu2s%Dnu, nu2s-(nu2s%Dnu)+0.5*Dnu, marker='s', edgecolor='green', facecolor='none')
            axes[0,2].scatter(nu2s%Dnu+Dnu, nu2s-(nu2s%Dnu)-0.5*Dnu, marker='s', edgecolor='green', facecolor='none')
            axes[0,2].axis([np.min(echx), np.max(echx), np.min(echy), np.max(echy)])
            axes[0,2].set_xlabel("$\\nu$  mod {:0.2f} ($\\mu$Hz)".format(Dnu))
            axes[0,2].set_ylabel('$\\nu$ ($\\mu$Hz)')
            axes[0,2].text(0.95,0.95,'Dnu: {:0.3f}'.format(Dnu), transform=axes[0,2].transAxes, va='top', ha='right')
            axes[0,2].text(0.95,0.90,'epsp: {:0.3f}'.format(epsp), transform=axes[0,2].transAxes, va='top', ha='right')
            axes[0,2].text(0.95,0.85,'alphap: {:0.5f}'.format(alphap), transform=axes[0,2].transAxes, va='top', ha='right')
            axes[0,2].axvline(Dnu, color='black', linestyle='--')
            # axes[2,2].axhline(bg_diagnostics['paramsMax'][2], color='black', linestyle='--')

            # # plot H, asymptotic p fitting
            # x = np.linspace(0,1,len(pmode_diagnostics['yFoldObs']))*pmode_diagnostics['paramsMax'][0]
            # axes[1,2].plot(x, pmode_diagnostics['yFoldObs'], color='C0')
            # axes[1,2].plot(x, pmode_diagnostics['yFoldMod'], color='black')
            # axes[1,2].axvline(pmode_diagnostics['paramsMax'][0]*pmode_diagnostics['paramsMax'][-1], color='red', linestyle='--')
            # axes[1,2].text(0.95,0.95,'aysmp Dnu: {:0.3f}'.format(pmode_diagnostics['paramsMax'][0]), 
            #                 transform=axes[1,2].transAxes, va='top', ha='right')
            # axes[1,2].text(0.95,0.90,'aysmp eps: {:0.3f}'.format(pmode_diagnostics['epsp']), 
            #                 transform=axes[1,2].transAxes, va='top', ha='right')

        # plot I, echelle, using Dnu from acf
        # axes[2,2]
        if (not (numax_diagnostics is None)) & (not (Dnu_diagnostics is None)):
            numax = numax_diagnostics['cacf_numax']
            Dnu = Dnu_diagnostics['acf_Dnu']
            powerSmoothed = smoothWrapper(self.freq, self.power, Dnu*0.03, 'flat')
            echx, echy, echz = echelle(self.freq, powerSmoothed, 
                        Dnu, numax-Dnu*8, numax+Dnu*8, echelletype="replicated")
            levels = np.linspace(np.min(echz), np.max(echz), 500)
            axes[2,2].contourf(echx, echy, echz, cmap="jet", levels=levels)
            axes[2,2].axis([np.min(echx), np.max(echx), np.min(echy), np.max(echy)])
            axes[2,2].set_xlabel("$\\nu$  mod {:0.2f} ($\\mu$Hz)".format(Dnu))
            axes[2,2].set_ylabel('$\\nu$ ($\\mu$Hz)')
            # axes[2,2].axvline(Dnu, color='black', linestyle='--')
            # axes[2,2].axhline(bg_diagnostics['paramsMax'][2], color='black', linestyle='--')

        # save
        plt.savefig(self.filepath+'global.png')
        plt.close() 
        return

    def to_data(self, **kwargs):
        data = kwargs
        np.save(self.filepath+'global', data)
        return