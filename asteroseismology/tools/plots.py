#!/usr/bin/env/ python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt

__all__ = ["echelle", "return_2dmap_axes"]

# def echelle(x, y, period, fmin=None, fmax=None, echelletype="single", offset=0.0):
#     '''
#     Generate a z-map for echelle plotting.

#     Input:

#     x: array-like[N,]
#     y: array-like[N,]
#     period: the large separation,
#     fmin: the lower boundary
#     fmax: the upper boundary
#     echelletype: single/replicated
#     offset: the horizontal shift

#     Output:

#     x, y: 
#         two 1-d arrays.
#     z: 
#         a 2-d array.

#     Exemplary call:

#     import matplotlib.pyplot as plt
#     fig = plt.figure(figsize=(6,8))
#     ax1 = fig.add_subplot(111)
#     echx, echy, echz = echelle(tfreq,tpowers_o,dnu,numax-9.0*dnu,numax+9.0*dnu,echelletype="single",offset=offset)
#     levels = np.linspace(np.min(echz),np.max(echz),500)
#     ax1.contourf(echx,echy,echz,cmap="gray_r",levels=levels)
#     ax1.axis([np.min(echx),np.max(echx),np.min(echy),np.max(echy)])
#     if offset > 0.0:
#         ax1.set_xlabel("(Frequency - "+str("{0:.2f}").format(offset)+ ") mod "+str("{0:.2f}").format(dnu) + " ($\mu$Hz)")
#     if offset < 0.0:
#         ax1.set_xlabel("(Frequency + "+str("{0:.2f}").format(np.abs(offset))+ ") mod "+str("{0:.2f}").format(dnu) + " ($\mu$Hz)")
#     if offset == 0.0:
#         ax1.set_xlabel("Frequency mod "+str("{0:.2f}").format(dnu) + " ($\mu$Hz)")
#     plt.savefig("echelle.png")

#     '''

#     if not echelletype in ["single", "replicated"]:
#         raise ValueError("echelletype is on of 'single', 'replicated'.")

#     if len(x) != len(y): 
#         raise ValueError("x and y must have equal size.")    

#     if fmin is None: fmin=0.
#     if fmax is None: fmax=np.nanmax(x)

#     fmin = fmin - offset
#     fmax = fmax - offset
#     x = x - offset

#     if fmin <= 0.0:
#         fmin = 0.0
#     else:
#         fmin = fmin - (fmin % period)

#     # first interpolate
#     samplinginterval = np.median(x[1:-1] - x[0:-2]) * 0.1
#     xp = np.arange(fmin,fmax+period,samplinginterval)
#     yp = np.interp(xp, x, y)

#     n_stack = int((fmax-fmin)/period)
#     n_element = int(period/samplinginterval)
#     #print(n_stack,n_element,len())

#     morerow = 2
#     arr = np.arange(1,n_stack) * period # + period/2.0
#     arr2 = np.array([arr,arr])
#     yn = np.reshape(arr2,len(arr)*2,order="F")
#     yn = np.insert(yn,0,0.0)
#     yn = np.append(yn,n_stack*period) + fmin #+ offset

#     if echelletype == "single":
#         xn = np.arange(1,n_element+1)/n_element * period
#         z = np.zeros([n_stack*morerow,n_element])
#         for i in range(n_stack):
#             for j in range(i*morerow,(i+1)*morerow):
#                 z[j,:] = yp[n_element*(i):n_element*(i+1)]
#     if echelletype == "replicated":
#         xn = np.arange(1,2*n_element+1)/n_element * period
#         z = np.zeros([n_stack*morerow,2*n_element])
#         for i in range(n_stack):
#             for j in range(i*morerow,(i+1)*morerow):
#                 z[j,:] = np.concatenate([yp[n_element*(i):n_element*(i+1)],yp[n_element*(i+1):n_element*(i+2)]])

#     return xn, yn, z

def echelle(freq, ps, Dnu, fmin=None, fmax=None, echelletype="single", offset=0.0):
    '''
    Make an echelle plot used in asteroseismology.
    
    Input parameters
    ----
    freq: 1d array-like, freq
    ps: 1d array-like, power spectrum
    Dnu: float, length of each vertical stack (Dnu in a frequency echelle)
    fmin: float, minimum frequency to be plotted
    fmax: float, maximum frequency to be plotted
    echelletype: str, `single` or `replicated`
    offset: float, an amount by which the diagram is shifted horizontally
    
    Return
    ----
    z: a 2d numpy.array, folded power spectrum
    extent: a list, edges (left, right, bottom, top) 
    x: a 1d numpy.array, horizontal axis
    y: a 1d numpy.array, vertical axis
    
    Users can create an echelle diagram with the following command:
    ----
    
    import matplotlib.pyplot as plt
    z, ext = echelle(freq, power, Dnu, fmin=numax-4*Dnu, fmax=numax+4*Dnu)
    plt.imshow(z, extent=ext, aspect='auto', interpolation='nearest')
    
    '''
    
    if fmin is None: fmin=0.
    if fmax is None: fmax=np.nanmax(freq)

    fmin -= offset
    fmax -= offset
    freq -= offset

    fmin = 1e-4 if fmin<Dnu else fmin - (fmin % Dnu)

    # define plotting elements
    resolution = np.median(np.diff(freq))
    # number of vertical stacks
    n_stack = int((fmax-fmin)/Dnu) 
    # number of point per stack
    n_element = int(Dnu/resolution) 

    fstart = fmin - (fmin % Dnu)
    
    z = np.zeros([n_stack, n_element])
    base = np.linspace(0, Dnu, n_element) if echelletype=='single' else np.linspace(0, 2*Dnu, n_element)
    for istack in range(n_stack):
        z[-istack-1,:] = np.interp(fstart+istack*Dnu+base, freq, ps)
    
    extent = (0, Dnu, fstart, fstart+n_stack*Dnu) if echelletype=='single' else (0, 2*Dnu, fstart, fstart+n_stack*Dnu)
    
    x = base
    y = fstart + np.arange(0, n_stack+1, 1)*Dnu + Dnu/2
    
    return z, extent, x, y


def period_echelle(period, ps, DPi, pmin=None, pmax=None, echelletype="single", offset=0.0, backwards=True):
    '''
    Make an echelle plot used in asteroseismology.
    
    Input parameters
    ----
    period: 1d array-like, s
    ps: 1d array-like, power spectrum
    DPi: float, length of each vertical stack (DPi in a period echelle)
    fmin: float, minimum frequency to be plotted
    fmax: float, maximum frequency to be plotted
    echelletype: str, `single` or `replicated`
    offset: float, an amount by which the diagram is shifted horizontally
    backwards: if the period array is descreasing
    
    Return
    ----
    z: a 2d numpy.array, folded power spectrum
    extent: a list, edges (left, right, bottom, top) 
    x: a 1d numpy.array, horizontal axis
    y: a 1d numpy.array, vertical axis
    
    Users can create an echelle diagram with the following command:
    ----
    
    import matplotlib.pyplot as plt
    z, ext = echelle(freq, power, Dnu, fmin=numax-4*Dnu, fmax=numax+4*Dnu)
    plt.imshow(z, extent=ext, aspect='auto', interpolation='nearest')
    
    '''
    
    if backwards: period, ps = period[::-1], ps[::-1]
    
    if pmin is None: pmin=np.nanmin(period)
    if pmax is None: pmax=np.nanmax(period)

    pmin -= offset
    pmax -= offset
    period -= offset

    # pmax = 1e-4 if pmax<1e-4 else pmin - (pmin % period)

    # define plotting elements
    resolution = np.median(np.abs(np.diff(period)))
    # number of vertical stacks
    n_stack = int((pmax-pmin)/DPi) 
    # number of point per stack
    n_element = int(DPi/resolution) 

    pstart = pmax + DPi - (pmax % DPi) if echelletype=='single' else pmax + 2*DPi - (pmax % DPi)
    
    
    z = np.zeros([n_stack, n_element])
    base = np.linspace(-DPi, 0, n_element) if echelletype=='single' else np.linspace(-2*DPi, 0, n_element)
    for istack in range(n_stack):
        z[-istack-1,:] = np.interp(pstart-istack*DPi+base, period, ps)

    extent = (-DPi, 0, pstart, pstart-n_stack*DPi) if echelletype=='single' else (-2*DPi, 0, pstart, pstart-n_stack*DPi)
    
    x = base
    y = pstart - np.arange(0, n_stack+1, 1)*DPi - DPi/2

    return z, extent, x, y



def return_2dmap_axes(numberOfSquareBlocks):

    # Some magic numbers for pretty axis layout.
    # stole from corner
    Kx = int(np.ceil(numberOfSquareBlocks**0.5))
    Ky = Kx if (Kx**2-numberOfSquareBlocks) < Kx else Kx-1

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

    