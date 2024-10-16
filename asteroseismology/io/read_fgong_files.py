#!/usr/bin/env/ python
# coding: utf-8


import numpy as np

__all__ = ['fgong']

class fgong:
    '''
    Read asteroseismology fgong files

    '''
    def __init__(self, filepath, verbose=True):
        '''
        Args:
            filepath: the path of .history file.
            verbose: whether print info to device. default: True.

        Attributes:
            header: a dictionary containing history file header
            track: a structure numpy array containing the evol track
            colnames: a tuple containing column names

        '''  
        self.filepath = filepath
        if verbose: 
            print('Processing :', self.filepath)
        self.header, self.mesh, self.colnames = self.readFile(self.filepath) #, self.colnames
        return
    
    def readFile(self, filepath, headerNameLine=1, headerDataLine=2, tableHeaderLine=6):
        '''
        Reads in a file.
        '''

        with open(self.filepath) as f:
            content = [line for line in f]
        # print(content[0:4])
        header = {}
        header['comment'] = ''.join(content[0:4])
        self.nn, self.iconst, self.ivar, self.ivers = [int(s) for s in content[4].split()]
        header['Nmesh'], header['Nglobal'], header['Nvar'], header['version'] = self.nn, self.iconst, self.ivar, self.ivers
        data = np.genfromtxt(self.filepath, skip_header=5).reshape(-1)
        
        # global data
        glob_names = ['M', 'R', 'L', 'Z', 'X0', 'alpha', 
                      'phi', 'eta', 'beta', 'lambda', 
                      '2nd_derivative_pressure_centre',
                      '2nd_derivative_density_centre',
                      'age', 'unused1', 'unused2']
        glob_data = data[0:self.iconst]
        for i in range(self.iconst):
            header[glob_names[i]] = glob_data[i]
        
        # mesh data
        mesh_names = ['r', 'ln(m/M)', 'T', 'p', 'rho', 'X', 
                      'L_r', 'kappa', 'eps', 'Gamma_1', 
                      'nabla_ad', 'delta',
                      'c_p', '1/mu_e', 'N^2/g_0', 'r_X',
                      'Z', 'R-r', 'eps_g', 'L_g', 'X(3He)',
                      'X(12C)', 'X(13C)', 'X(14N)', 'X(16O)',
                      'dlnGamma_1/dlnrho', 
                      'dlnGamma_1/dlnp',
                      'dlnGamma_1/dY',
                      'X(2H)', 'X(4He)', 'X(7Li)', 'X(7Be)',
                      'X(15N)', 'X(17O)', 'X(18O)', 'X(20Ne)',
                      'unused1', 'unused2', 'unused3', 'unused4' ]
        dtype = [(n, 'float64') for n in mesh_names]
        data = [tuple(n) for n in data[self.iconst:].reshape((self.nn, self.ivar))]

        mesh = np.array(data, dtype=dtype)
        
        colnames = mesh.dtype.names

        return header, mesh, colnames


