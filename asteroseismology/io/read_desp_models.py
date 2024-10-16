#!/usr/bin/env/ python
import numpy as np
import matplotlib.pyplot as plt

__all__ = ["DESP_ISO"]

class DESP_ISO:
    
    """
    
    Reads in MIST isochrone files.
    
    """
    
    def __init__(self, filename, verbose=True):
    
        """
        
        Args:
            filename: the name of .iso file.
        
        Usage:
            >> iso = DESP_ISO('MIST_v1.0_feh_p0.00_afe_p0.0_vvcrit0.4.iso')
            
        Attributes:
            header     Dictionary containing MIX_LEN, Y, Z, Zeff, [Fe/H], [a/Fe], age, eeps.
            iso        Data.
            
        """
        
        self.filename = filename
        if verbose:
            print('Reading in: ' + self.filename)
            
        self.header, self.iso = self.read_iso_file()
        
    def read_iso_file(self):

        """
        Reads in the isochrone file.
        
        Args:
            filename: the name of .iso file.
        
        """
        
        #open file and read it in
        with open(self.filename) as f:
            content = [line for line in f]
        header = {content[2][1:].split()[i]:float(content[3][1:].split()[i]) for i in range(0,6)}
        header['age'] = float(content[7].split()[0].split('AGE=')[-1])
        header['eeps'] = int(content[7].split('EEPS=')[-1])

        cols = content[8][1:].split()
        num_cols = len(cols)
        num_eeps = header['eeps']
        #read one block for each isochrone
        data = content[9:]
        formats = tuple([np.int32]+[np.float64 for i in range(num_cols-1)])
        iso = np.zeros((num_eeps),{'names':tuple(cols),'formats':tuple(formats)})
        #read through EEPs for each isochrone
        for eep in range(num_eeps):
            iso_chunk = data[eep].split()
            iso[eep] = tuple(iso_chunk)
        return header, iso 