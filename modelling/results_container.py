import numpy as np 


class stardata():
    def __init__(self, initialize=None):
        '''
        This is a container for one dimensional star data.
        Each element must be a numpy array.
        
        Construct a stardata container:
        star = stardata()

        Construct a stardata container from existing containers:
        star0 = stardata()
        star1 = stardata()
        star = stardata([star0])
        star = stardata([star0, star1])

        Store values:
        star['Teff'] = np.arange(5300, 5400)

        Append values:
        star.append('Teff', np.arange(5400, 5500))

        View keys:
        star.Nkeys # number of keys
        star.keys # list of keys

        '''
        if (initialize is None):
            self.Nkeys = 0
            self.keys = []
        elif (isinstance(initialize, list)):
            Nsd = len(initialize)

            # check if have the same Nkeys and keys
            Nkeys = np.array([sd.Nkeys for sd in initialize])
            # keys = np.array([set(sd.keys) for sd in initialize])

            if np.all(np.isin(np.unique(Nkeys), [0, np.max(Nkeys)])) :
                self.Nkeys = 0
                self.keys = []
                for sd in initialize:
                    if sd.Nkeys == 0: continue
                    for key in sd.keys:
                        self.append(key, sd[key])
            else:
                self.Nkeys = 0
                self.keys = []

        else:
            self.Nkeys = 0
            self.keys = []

        return

    def __getitem__(self, key):
        return getattr(self,key)

    def __setitem__(self, key, value):
        if hasattr(self, key):
            pass
        else:
            self.keys.append(key)
            self.Nkeys += 1
        setattr(self, key, value)
        return 

    def add_key(self, key, dtype=float):
        if hasattr(self, key):
            pass
        else:
            setattr(self, key, np.array([],dtype=dtype))
            self.keys.append(key)
            self.Nkeys += 1
        return self

    def append(self, key, value, dtype=None):
        if hasattr(self, key):
            setattr(self, key, np.append(getattr(self,key), value))
        else:
            if dtype is None: dtype=type(value[0])
            self.add_key(key, dtype=dtype)
            setattr(self, key, np.append(getattr(self,key), value))
        return self


if __name__ == "__main__":
    # test 1 - construct keys and add values
    t1 = stardata()
    t1.add_key('Teff')
    t1.add_key('feh')
    t1.append('Teff', np.arange(5100, 5105))
    t1.append('feh', np.zeros(t1['Teff'].shape) )
    print(t1.Nkeys, t1.keys)
    print(t1['Teff'])
    print(t1['feh'])

    # test 2 - construct keys without claiming first
    t2 = stardata()
    t2.append('Teff', np.arange(5300, 5305, dtype=object))
    t2.append('feh', np.zeros(t2['Teff'].shape)+0.1 )
    print(t2.Nkeys, t2.keys)
    print(t2['Teff'])
    print(t2['feh'])

    # test 3 - construct keys without claiming first
    t3 = stardata()
    t3['Teff'] = np.arange(5500, 5505)
    t3.append('feh', np.zeros(t3['Teff'].shape)+0.5 )
    print(t3.Nkeys, t3.keys)
    print(t3['Teff'])
    print(t3['feh'])

    # test 4 - intialize a stardata class with existing class
    t4 = stardata([t1,t2,t3])
    print(t4.Nkeys, t4.keys)
    print(t4['Teff'])
    print(t4['feh'])