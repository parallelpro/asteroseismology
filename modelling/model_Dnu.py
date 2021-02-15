import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = ['get_model_Dnu']


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
    obs_freq: array_like[Nmode_mod]
        observation's mode frequency
    obs_efreq: array_like[Nmode_mod]
        observation's mode frequency uncertainty
    obs_l: array_like[Nmode_mod]
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
        mod_n = np.zeros(len(mod_freq_l0))
        for imod in range(len(mod_n)-1):
            mod_n[(imod+1):] = mod_n[(imod+1):] + np.round((mod_freq_l0[imod+1]-mod_freq_l0[imod])/Dnu)

        sigma = 1/np.exp(-(mod_freq_l0-numax)**2./(2*width**2.))
        
        p = np.polyfit(mod_n, mod_freq_l0, 1, w=1/sigma)
        mod_Dnu = p[0]

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
        mod_n = np.zeros(len(mod_freq_l0))
        for imod in range(len(mod_n)-1):
            mod_n[(imod+1):] = mod_n[(imod+1):] + np.round((mod_freq_l0[imod+1]-mod_freq_l0[imod])/Dnu)

        sigma = obs_efreq_l0
        p = np.polyfit(mod_n, mod_freq_l0, 1, w=1/sigma)  
        mod_Dnu = p[0]      

    return mod_Dnu