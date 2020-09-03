#!/usr/bin/env/ python
# coding: utf-8


import numpy as np

__all__ = ['history', 'profile', 'sums', 'modes']

class readTable:
    '''
    A parent class to be wrapped by other class, in order to read in such as mesa history file.
    These files have very similar structures, typically header+table.

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
        return
    
    def readFile(self, filepath, headerNameLine=1, headerDataLine=2, tableHeaderLine=6):
        '''
        Reads in a file.
        '''

        with open(self.filepath) as f:
            content = [line.split() for line in f]
        header = {content[headerNameLine-1][i]:content[headerDataLine-1][i] for i in range(len(content[headerNameLine-1]))}
        table = np.genfromtxt(self.filepath, skip_header=tableHeaderLine-1, names=True)
        colnames = table.dtype.names

        return header, table, colnames


class history(readTable):
    '''

    A class to read mesa history files, store the data within, and offer useful routines (?).

    '''
    
    def __init__(self, filepath, verbose=True, ifReadProfileIndex=False):
        '''
        Args:
            filepath: the path of .history file.
            verbose: whether print info to device. default: True.

        Attributes:
            header: a dictionary containing history file header
            track: a structure numpy array containing the evol track
            colnames: a tuple containing column names
            profileIndex: a structured array containing the map between model_number and profile_number

        '''
        super().__init__(filepath, verbose)
        self.header, self.track, self.colnames = self.readFile(filepath, headerNameLine=2, headerDataLine=3, tableHeaderLine=6)
        
        if ifReadProfileIndex:
            self.profileIndex = self.read_profile_index()
        return
    
    def read_profile_index(self):
        '''
        Reads in the profile.index file
        '''
        filepath = self.filepath.split('.history')[0] + 'profile.index'
        profileIndex = np.genfromtxt(filepath, skip_header=1, names=('model_number', 'priority', 'profile_number'))
        return profileIndex


class profile(readTable):
    '''

    A class to read mesa history files, store the data within, and offer useful routines.

    '''

    
    def __init__(self, filepath, verbose=True):
        '''
        Args:
            filepath: the path of *profile*.data file.
            verbose: whether print info to device. default: True.

        Attributes:
            header: a dictionary containing history file header
            profile: a structure numpy array containing the structure profile
            colnames: a tuple containing column names

        '''
        super().__init__(filepath, verbose)
        self.header, self.profile, self.colnames = self.readFile(filepath, headerNameLine=2, headerDataLine=3, tableHeaderLine=6)

        return


class sums(readTable):
    '''

    A class to read gyre mode summary file, store the data within, and offer useful routines.

    '''

    
    def __init__(self, filepath, verbose=True):
        '''
        Args:
            filepath: the path of .sums file.
            verbose: whether print info to device. default: True.

        Attributes:
            header: a dictionary containing history file header
            sums: a structure numpy array containing the summary table
            colnames: a tuple containing column names

        '''
        super().__init__(filepath, verbose)
        self.header, self.sums, self.colnames = self.readFile(filepath, headerNameLine=3, headerDataLine=4, tableHeaderLine=6)

        return


class modes(readTable):
    '''

    A class to read gyre mode kernel file, store the data within, and offer useful routines.

    '''

    def __init__(self, filepath, verbose=True):
        '''
        Args:
            filepath: the path of .modes file.
            verbose: whether print info to device. default: True.

        Attributes:
            header: a dictionary containing history file header
            modes: a structure numpy array containing the mode profiles
            colnames: a tuple containing column names

        '''

        super().__init__(filepath, verbose)
        self.header, self.modes, self.colnames = self.readFile(filepath, headerNameLine=3, headerDataLine=4, tableHeaderLine=6)

        return


if __name__ == '__main__':

    ### test - read models
    h = history('models/grid_mass/LOGS_data/m090feh000.history', ifReadProfileIndex=True)
    s = sums('models/grid_mass/sums/m090feh000profile1.data.FGONG.sum')
    p = profile('sample/bestModels/profiles/5607242/m112feh-045profile211.data')
    m = modes('sample/bestModels/profiles/5607242/gyreMode00001.txt')
    