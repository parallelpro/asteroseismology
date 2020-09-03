#!/usr/bin/env/ python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt

__all__ = ["echelle", "return_2dmap_axes"]

def echelle(x, y, period, fmin=None, fmax=None, echelletype="single", offset=0.0):
    '''
    Generate a z-map for echelle plotting.

    Input:

    x: array-like[N,]
    y: array-like[N,]
    period: the large separation,
    fmin: the lower boundary
    fmax: the upper boundary
    echelletype: single/replicated
    offset: the horizontal shift

    Output:

    x, y: 
        two 1-d arrays.
    z: 
        a 2-d array.

    Exemplary call:

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6,8))
    ax1 = fig.add_subplot(111)
    echx, echy, echz = echelle(tfreq,tpowers_o,dnu,numax-9.0*dnu,numax+9.0*dnu,echelletype="single",offset=offset)
    levels = np.linspace(np.min(echz),np.max(echz),500)
    ax1.contourf(echx,echy,echz,cmap="gray_r",levels=levels)
    ax1.axis([np.min(echx),np.max(echx),np.min(echy),np.max(echy)])
    if offset > 0.0:
        ax1.set_xlabel("(Frequency - "+str("{0:.2f}").format(offset)+ ") mod "+str("{0:.2f}").format(dnu) + " ($\mu$Hz)")
    if offset < 0.0:
        ax1.set_xlabel("(Frequency + "+str("{0:.2f}").format(np.abs(offset))+ ") mod "+str("{0:.2f}").format(dnu) + " ($\mu$Hz)")
    if offset == 0.0:
        ax1.set_xlabel("Frequency mod "+str("{0:.2f}").format(dnu) + " ($\mu$Hz)")
    plt.savefig("echelle.png")

    '''

    if not echelletype in ["single", "replicated"]:
        raise ValueError("echelletype is on of 'single', 'replicated'.")

    if len(x) != len(y): 
        raise ValueError("x and y must have equal size.")    

    if fmin is None: fmin=0.
    if fmax is None: fmax=np.nanmax(x)

    fmin = fmin - offset
    fmax = fmax - offset
    x = x - offset

    if fmin <= 0.0:
        fmin = 0.0
    else:
        fmin = fmin - (fmin % period)

    # first interpolate
    samplinginterval = np.median(x[1:-1] - x[0:-2]) * 0.1
    xp = np.arange(fmin,fmax+period,samplinginterval)
    yp = np.interp(xp, x, y)

    n_stack = int((fmax-fmin)/period)
    n_element = int(period/samplinginterval)
    #print(n_stack,n_element,len())

    morerow = 2
    arr = np.arange(1,n_stack) * period # + period/2.0
    arr2 = np.array([arr,arr])
    yn = np.reshape(arr2,len(arr)*2,order="F")
    yn = np.insert(yn,0,0.0)
    yn = np.append(yn,n_stack*period) + fmin #+ offset

    if echelletype == "single":
        xn = np.arange(1,n_element+1)/n_element * period
        z = np.zeros([n_stack*morerow,n_element])
        for i in range(n_stack):
            for j in range(i*morerow,(i+1)*morerow):
                z[j,:] = yp[n_element*(i):n_element*(i+1)]
    if echelletype == "replicated":
        xn = np.arange(1,2*n_element+1)/n_element * period
        z = np.zeros([n_stack*morerow,2*n_element])
        for i in range(n_stack):
            for j in range(i*morerow,(i+1)*morerow):
                z[j,:] = np.concatenate([yp[n_element*(i):n_element*(i+1)],yp[n_element*(i+1):n_element*(i+2)]])

    return xn, yn, z



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

    