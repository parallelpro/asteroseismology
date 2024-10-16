import numpy as np
from .series import fourier, smooth

__all__ = ['clean']


def clean(time, flux, e_flux=None, fmin=None, fmax=None, include_fcs=None, smooth_width=None):
    """
    Perform pre-whitening on input time-series data to identify the dominant frequency,
    fit a sinusoidal model, compute the residual flux, and calculate the signal-to-noise ratio (SNR).

    Parameters:
    ----------
    time : numpy.ndarray
        Array of time points.
    
    flux : numpy.ndarray
        Array of flux (signal) values corresponding to the time points.
    
    e_flux : numpy.ndarray, optional
        Array of uncertainties in the flux measurements. If not provided, assumes equal uncertainties.
    
    fmin : float, optional
        Minimum frequency to consider in the Fourier analysis. Defaults to 0 if not provided.
    
    fmax : float, optional
        Maximum frequency to consider in the Fourier analysis. Defaults to the Nyquist frequency if not provided.
    
    return_res_flux : bool, optional
        Whether to return the residual flux after removing the model. Default is True.
    
    include_fcs : list, optional
        List of additional frequencies to include in the model fitting. Default is None.
    
    smooth_width : float, optional
        Width for smoothing the power spectrum. If not provided, no smoothing is applied.
    

    Returns:
    -------
    f0 : float
        The frequency of the dominant component.
    amp : float
        The amplitude of the dominant frequency component.
    phi : float
        The phase of the dominant frequency component.
    snr : float
        The signal-to-noise ratio of the dominant frequency component.

    Notes:
    -----
    The function first computes the Fourier transform of the input time-series data to identify the dominant frequency component
    within the specified frequency range. It then fits a sinusoidal model to the data using linear regression, subtracts this model
    from the original data to obtain the residual flux, and computes the signal-to-noise ratio of the identified component.
    """
    
    # Validate inputs
    if not isinstance(time, np.ndarray) or not isinstance(flux, np.ndarray):
        raise ValueError("Time and flux must be numpy arrays")
    if len(time) != len(flux):
        raise ValueError("Time and flux arrays must have the same length")
    if e_flux is not None and len(e_flux) != len(flux):
        raise ValueError("e_flux array must have the same length as flux array")
    
    # Copy flux to avoid modifying the original array
    res_flux = np.copy(flux)
    
    # Perform Fourier transform
    freq, ps = fourier(time, flux, dy=e_flux, return_val='power', oversampling=10)

    # Apply smoothing if requested
    if smooth_width is not None:
        ps = smooth(freq, ps, smooth_width, 'bartlett')

    # Define frequency range if not provided
    if fmin is None:
        fmin = 0.0
    if fmax is None:
        fmax = 1 / (0.5 * np.median(np.diff(time)))

    # Find the dominant frequency in the specified range
    freq_mask = (freq > fmin) & (freq < fmax)
    fc = freq[freq_mask][np.argmax(ps[freq_mask])]
    max_power = np.max(ps[freq_mask])

    # Construct the model matrix
    if (include_fcs is not None) :
        # fcs = list(set([fc]) | set(include_fcs))  # avoid singular matrix
        fcs = [fc] + list(set(include_fcs)-set([fc]))  # avoid singular matrix
    else:
        fcs = [fc]
    fcs_expand = [wave for tfc in fcs for wave in [np.sin(2*np.pi*tfc*time), np.cos(2*np.pi*tfc*time)] ]
    X = np.array([*fcs_expand, np.ones(len(time))]).T
    y = res_flux
    w = np.diag(1 / e_flux**2) if e_flux is not None else np.eye(len(flux))

    # Perform weighted linear regression
    coeffs = np.linalg.inv(X.T @ w @ X) @ X.T @ w @ y

    # Compute power and phase
    power = coeffs[0]**2 + coeffs[1]**2
    phi = np.arctan2(coeffs[1], coeffs[0]) / (2 * np.pi)

    # Subtract the fitted model from the flux
    y_model = X @ coeffs
    res_flux -= y_model
    
    # Recalculate Fourier transform on residual flux
    freq, ps = fourier(time, res_flux, dy=e_flux, return_val='power', freq=freq[freq_mask])
    mean_ps = np.median(ps)
    local_snr = max_power / mean_ps

    return {'fc': fc,
            'power': power,
            'amp': power**0.5,
            'phi': phi,
            'local_snr': local_snr,
            'res_flux': res_flux}
