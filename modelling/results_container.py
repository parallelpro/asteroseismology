import numpy as np 


class stardata():
    def __init__(self):
        '''
        This is a container for one dimensional star data.
        Each element must be a numpy array.
        
        Construct a stardata container:
        star = stardata()

        Store values:
        star['Teff'] = np.arange(5300, 5400)

        Append values:
        star.append('Teff', np.arange(5400, 5500))

        View keys:
        star.Nkeys # number of keys
        star.keys # list of keys
        
        '''
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
        return

    def append(self, key, value):
        if hasattr(self, key):
            setattr(self, key, np.append(getattr(self,key), value))
        else:
            self.add_key(key, dtype=type(value[0]))
            setattr(self, key, np.append(getattr(self,key), value))
        return


if __name__ == "__main__":
    # test 1 - construct keys and add values
    t1 = stardata()
    t1.add_key('Teff')
    t1.add_key('[Fe/H]')
    t1.append('Teff', np.arange(5100, 5300))
    t1.append('[M/H]', np.zeros(t1['Teff'].shape) )
    print(t1.Nkeys, t1.keys)
    
    # test 2 - construct keys without claiming first
    t2 = stardata()
    t2.append('Teff', np.arange(5100, 5300, dtype=object))
    t2.append('[M/H]', np.zeros(t1['Teff'].shape) )
    print(t2.Nkeys, t2.keys)

    # test 3 - construct keys without claiming first
    t3 = stardata()
    t3['Teff'] = np.arange(0,100)
    print(t3.Nkeys, t3.keys)
