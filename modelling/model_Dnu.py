import numpy as np
from scipy.optimize import linear_sum_assignment, curve_fit

__all__ = ['get_model_Dnu', 'get_obs_Dnu']


def get_model_Dnu(mod_freq, mod_l, Dnu, numax, 
        obs_freq=None, obs_efreq=None, obs_l=None):
    
    """
    Calculate model Dnu around numax, by fitting freq vs n with 
    Gaussian envelope around numax weighted data points, or with
    observational uncertainty weighted data points (if obs_freq, 
    obs_efreq and obs_l are set).

    ----------
    Input:
    mod_freq: array_like[Nmode_mod]
        model's mode frequency
    mod_l: array_like[Nmode_mod]
        model's mode degree
    Dnu: float
        the p-mode large separation in muHz
    numax: float
        the frequency of maximum power in muHz

    ----------
    Optional input:
    obs_freq: array_like[Nmode_obs]
        observation's mode frequency
    obs_efreq: array_like[Nmode_obs]
        observation's mode frequency uncertainty
    obs_l: array_like[Nmode_obs]
        observation's mode degree

    ----------
    Return:
    mod_Dnu: float

    """

    if ((obs_freq is None) | (obs_efreq is None) | (obs_l is None)):
        ifUseModel = True 
    else:
        ifUseModel = False

    if ifUseModel:
        # width estimates based on Yu+2018, Lund+2017, Li+2020
        k, b = 0.9638, -1.7145
        width = np.exp(k*np.log(numax) + b)

        # assign n
        mod_freq_l0 = np.sort(mod_freq[mod_l==0])
        # mod_n = np.zeros(len(mod_freq_l0))
        # for imod in range(len(mod_n)-1):
        #     mod_n[(imod+1):] = mod_n[(imod+1):] + np.round((mod_freq_l0[imod+1]-mod_freq_l0[imod])/Dnu)
        mod_n = np.arange(len(mod_freq_l0))
        # sigma = 1/np.exp(-(mod_freq_l0-numax)**2./(2*width**2.))
        weight = np.exp(-(mod_freq_l0-numax)**2./(2*width**2.))
        idx = weight>1e-100
        if np.sum(idx)>2:
            p = np.polyfit(mod_n[idx], mod_freq_l0[idx], 1, w=weight[idx])
            mod_Dnu = p[0]
        else:
            mod_Dnu = np.nan

    else:
        # we need to assign each obs mode with a model mode
        # this can be seen as a linear sum assignment problem, also known as minimun weight matching in bipartite graphs
        obs_freq_l0, obs_efreq_l0, mod_freq_l0 = obs_freq[obs_l==0], obs_efreq[obs_l==0], mod_freq[mod_l==0]
        cost = np.abs(obs_freq_l0.reshape(-1,1) - mod_freq_l0)
        row_ind, col_ind = linear_sum_assignment(cost)
        obs_freq_l0, obs_efreq_l0 = obs_freq_l0[row_ind], obs_efreq_l0[row_ind]
        mod_freq_l0 = mod_freq_l0[col_ind]

        idx = np.argsort(mod_freq_l0)
        mod_freq_l0, obs_freq_l0, obs_efreq_l0 = mod_freq_l0[idx], obs_freq_l0[idx], obs_efreq_l0[idx]

        # assign n
        mod_freq_l0 = np.sort(mod_freq[mod_l==0])
        # mod_n = np.zeros(len(mod_freq_l0))
        # for imod in range(len(mod_n)-1):
        #     mod_n[(imod+1):] = mod_n[(imod+1):] + np.round((mod_freq_l0[imod+1]-mod_freq_l0[imod])/Dnu)
        mod_n = np.arange(len(mod_freq_l0))
        sigma = obs_efreq_l0
        p = np.polyfit(mod_n, mod_freq_l0, 1, w=1/sigma)  
        mod_Dnu = p[0]      

    return mod_Dnu


def get_obs_Dnu(obs_freq, obs_efreq=None, Dnu_guess=None):
    
    """
    Calculate observed Dnu from l=0 radial modes.

    ----------
    Input:
    obs_freq: array_like[Nmode_obs]
        radial mode frequency

    ----------
    Optional input:
    obs_efreq: array_like[Nmode_obs]
        mode frequency uncertainty

    Dnu_guess: float
        An initial guess for the Dnu

    ----------
    Return:
    obs_Dnu: float

    """
    if obs_efreq is None: obs_efreq = np.ones(len(obs_freq))
    if Dnu_guess is None: Dnu_guess = np.median(np.diff(np.sort(obs_freq)))

    Nmodes = len(obs_freq)
    
    mode_n = np.arange(Nmodes) + np.floor(obs_freq[0]/Dnu_guess)
    for im in range(len(mode_n)-1):
        add = np.round((obs_freq[im+1]-obs_freq[im])/Dnu_guess)-1
        mode_n[(im+1):] = mode_n[(im+1):] + add 

    # ## 1 - all modes without curvature
    def func1(xdata, Dnu, epsp):
        return (xdata + epsp )*Dnu
    # print(obs_freq, obs_efreq,mode_n)
    popt, pcov = curve_fit(func1, mode_n, obs_freq, sigma=obs_efreq)
    perr = np.diag(pcov)**0.5

    obs_Dnu, obs_eDnu = popt[0], perr[0]

    return obs_Dnu, obs_eDnu
