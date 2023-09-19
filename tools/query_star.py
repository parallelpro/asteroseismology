from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
import numpy as np

__all__ = ['get_plx', 'get_magnitudes']

def get_plx(starname):

    # # get RA and DEC for that source
    # result_table = Simbad.query_object(starname)

    # if len(result_table)>=1:
    #     ras, decs = result_table[0]['RA'] ,result_table[0]['DEC'] 
    #     l = [float(s) for s in ras.split()]
    #     ra = (l[0]+l[1]/60+l[2]/3600)/24*360
    #     l = [float(s) for s in decs.split()]
    #     dec = (l[0]+l[1]/60+l[2]/3600)

    # add a condition to drop if ra, dec is None

    # # get plx from Gaia
    v = Vizier(columns=["*"], catalog="I/355/gaiadr3")
    gaia_result_table = v.query_region(starname, radius="2s")
    if len(gaia_result_table)>=1:
        plx = gaia_result_table[0]['Plx'][0]
        e_plx = gaia_result_table[0]['e_Plx'][0]
    else:
        plx, e_plx = np.nan, np.nan

    return plx, e_plx
        

all_bands = ['Gmag', 'BPmag', 'RPmag', 'Jmag', 'Hmag', 'Kmag', 'W1mag', 'W2mag', 'W3mag', 'W4mag',
         'Vmag', 'Bmag', 'g_mag', 'r_mag', 'i_mag', 'FUVmag', 'NUVmag', "uPSF", "e_uPSF", "vPSF", "e_vPSF",
         "BTmag", "e_BTmag", "VTmag", "e_VTmag"]

def get_magnitudes(starname, bands=all_bands):

    # get magnitudes from photometric surveys

    stardata = {}

    ## Gaia
    # import astropy.units as u
    # from astropy.coordinates import SkyCoord
    # from astroquery.gaia import Gaia
    # Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source" # S
    # coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
    # radius = u.Quantity(cone, u.deg)
    # j = Gaia.cone_search_async(coord, radius)
    # gaia_result_table = j.get_results()
    
    if np.isin(['Gmag', 'BPmag', 'RPmag'], bands).sum() > 0:
        ## Gaia
        v = Vizier(columns=["*"], catalog="I/355/gaiadr3")
        gaia_result_table = v.query_region(starname, radius="2s")
        if len(gaia_result_table)>=1:
            stardata['Gmag'] = gaia_result_table[0]['Gmag'][0]
            stardata['BPmag'] = gaia_result_table[0]['BPmag'][0]
            stardata['RPmag'] = gaia_result_table[0]['RPmag'][0]

            stardata['e_Gmag'] = 2.5*np.log(10) * (gaia_result_table[0]['e_FG'][0]/gaia_result_table[0]['FG'][0])
            stardata['e_BPmag'] = 2.5*np.log(10) * (gaia_result_table[0]['e_FBP'][0]/gaia_result_table[0]['FBP'][0])
            stardata['e_RPmag'] = 2.5*np.log(10) * (gaia_result_table[0]['e_FRP'][0]/gaia_result_table[0]['FRP'][0])
        else:
            stardata['Gmag'], stardata['BPmag'], stardata['RPmag'] = [np.nan for i in range(3)]
            stardata['e_Gmag'], stardata['e_BPmag'], stardata['e_RPmag'] = [np.nan for i in range(3)]


    if np.isin(['Jmag', 'Hmag', 'Kmag'], bands).sum() > 0:
        ## 2MASS
        v = Vizier(columns=["*"], catalog="II/246/out")
        tmass_result_table = v.query_region(starname, radius="2s")
        cols = ['Jmag', 'Hmag', 'Kmag']
        if len(tmass_result_table)>=1:
            for col in cols:
                stardata[col] = tmass_result_table[0][col][0]
                stardata['e_'+col] = tmass_result_table[0]['e_'+col][0]
        else:
            for col in cols:
                stardata[col] = np.nan
                stardata['e_'+col] = np.nan

    if np.isin(['W1mag', 'W2mag', 'W3mag', 'W4mag'], bands).sum() > 0:
        ## WISE
        v = Vizier(columns=["*"], catalog="II/328/allwise")
        allwise_result_table = v.query_region(starname, radius="2s")
        cols = ['W1mag', 'W2mag', 'W3mag', 'W4mag']
        if len(allwise_result_table)>=1:
            for col in cols:
                stardata[col] = allwise_result_table[0][col][0]
                stardata['e_'+col] = allwise_result_table[0]['e_'+col][0]
        else:
            for col in cols:
                stardata[col] = np.nan
                stardata['e_'+col] = np.nan

    if np.isin(['Vmag', 'Bmag', 'g_mag', 'r_mag', 'i_mag'], bands).sum() > 0:
        ## APASS
        v = Vizier(columns=["*"], catalog="II/336")
        APASS_result_table = v.query_region(starname, radius="2s")
        cols = ['Vmag', 'Bmag', 'g_mag', 'r_mag', 'i_mag']
        if len(APASS_result_table)>=1:
            for col in cols:
                stardata[col] = APASS_result_table[0][col][0]
                stardata['e_'+col] = APASS_result_table[0]['e_'+col][0]
        else:
            for col in cols:
                stardata[col] = np.nan
                stardata['e_'+col] = np.nan

    if np.isin(['FUVmag', 'NUVmag'], bands).sum() > 0:
        ## GALEX
        v = Vizier(columns=["*"], catalog="II/335/galex_ais")
        GALEX_result_table = v.query_region(starname, radius="2s")
        cols = ['FUVmag', 'NUVmag']
        if len(GALEX_result_table)>=1:
            for col in cols:
                stardata[col] = GALEX_result_table[0][col][0]
                stardata['e_'+col] = GALEX_result_table[0]['e_'+col][0]
        else:
            for col in cols:
                stardata[col] = np.nan
                stardata['e_'+col] = np.nan
    
    if np.isin(["uPSF", "e_uPSF", "vPSF", "e_vPSF"], bands).sum() > 0:
        # skymapper
        v = Vizier(columns=["uPSF", "e_uPSF", "vPSF", "e_vPSF"], catalog="II/358/smss")
        skymapper_result_table = v.query_region(starname, radius="2s")
        if len(skymapper_result_table)>=1:
            for col in cols:
                stardata['umag'] = skymapper_result_table[0]['uPSF'][0]
                stardata['e_umag'] = skymapper_result_table[0]['e_uPSF'][0]
                stardata['vmag'] = skymapper_result_table[0]['vPSF'][0]
                stardata['e_vmag'] = skymapper_result_table[0]['e_vPSF'][0]
        else:
            for col in cols:
                stardata['umag'] = np.nan
                stardata['e_umag'] = np.nan
                stardata['vmag'] = np.nan
                stardata['e_vmag'] = np.nan
    
    if np.isin(["BTmag", "e_BTmag", "VTmag", "e_VTmag"], bands).sum() > 0:
        # tycho
        v = Vizier(columns=["BTmag", "e_BTmag", "VTmag", "e_VTmag"], catalog="I/259/tyc2")
        Tycho_result_table = v.query_region(starname, radius="2s")
        if len(Tycho_result_table)>=1:
            for col in cols:
                stardata['Bmag'] = Tycho_result_table[0]['BTmag'][0]
                stardata['e_Bmag'] = Tycho_result_table[0]['e_BTmag'][0]
                stardata['Vmag'] = Tycho_result_table[0]['VTmag'][0]
                stardata['e_Vmag'] = Tycho_result_table[0]['e_VTmag'][0]
        else:
            for col in cols:
                stardata['Bmag'] = np.nan
                stardata['e_Bmag'] = np.nan
                stardata['Vmag'] = np.nan
                stardata['e_Vmag'] = np.nan

    return stardata


# def get_flux():
    # # convert to flux density F_lambda [erg cm^-2 s^-1 AA^-1] with information here
    # # http://svo2.cab.inta-csic.es/svo/theory/fps/

    # # F_lambda [erg cm^-2 s^-1 AA^-1]
    # zpts = {'G': 2.5e-9, 'BP': 4.08e-9, 'RP': 1.27e-9, #gaia
    #        'J': 3.13e-10, 'H':1.13e-10, 'K':4.28e-11,  #2mass
    #        'W1': 8.18e-12, 'W2': 2.42e-12, 'W3': 6.52e-14, 'W4':5.09e-15, #wise
    #        'B':6.29e-9, 'V':3.57e-9, #generic->johnson
    #        'g_':5.45e-9, 'r_':2.5e-9, 'i_':1.39e-9,#sloan
    #        'FUV':6.72e-9, 'NUV':4.54e-9, #galex
    #        'u': 3.23e-9, 'v': 5.45e-9, # skymapper
    #        #'BT': 6.62e-9, 'VT': 3.94e-9, # tycho
    #        } 

    # # AA
    # lambdaeff = {'G': 5822.39, 'BP': 5035.75, 'RP': 7619.96, 
    #        'J': 12350, 'H':16620, 'K':21590, 
    #        'W1':33526, 'W2':46028, 'W3':115608, 'W4':220883,
    #        'B':4378.12, 'V':5466.11, #generic->johnson
    #        'g_':4671.78, 'r_':6141.12, 'i_':7457.89, #SLOAN
    #        'FUV':1548.85, 'NUV':2303.37, #galex
    #        'u': 3500.22, 'v': 3878.68,# skymapper
    #        #'BT': 4219.08, 'VT': 5258.48, # tycho
    #        }

    # # bands
    # bands = list(lambdaeff.keys())

    # mags = np.array([stardata[b+'mag'] for b in bands])
    # e_mags = np.array([stardata['e_'+b+'mag'] for b in bands])
    # wls = np.array([lambdaeff[b] for b in bands])*0.0001 #AA -> microm
    # f0s = np.array([zpts[b] for b in bands])
    # fs = 10.0**(-0.4*mags)*f0s
    # e_fs = np.abs(np.log(10)*10**(-0.4*mags)*(-0.4)*f0s * e_mags/mags * fs)
    # # magnitudes = -2.5*log10(F/F0)
    # # Gflx, BPflx, RPflx = Gflx