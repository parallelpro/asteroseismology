import numpy as np

__all__ = ['get_model_Dnu']


def get_model_Dnu(mod_freq, mod_l, Dnu, numax):
    
    """
    Calculate model Dnu around numax.
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
    Return:
    mod_Dnu: float

    """
    idx = (mod_l==0) & (mod_freq>(numax-4.3*Dnu)) & (mod_freq<(numax+4.3*Dnu))
    if np.sum(idx)>5:
        return np.median(np.diff(np.sort(mod_freq[idx])))
    else:
        return np.nan
        