import numpy as np
from sobol_seq import i4_sobol

__all__ = ['get_sobol_sequence']

def get_sobol_sequence(ranges, N=2**12, skip=20000):
    # for reference & citation see https://github.com/earlbellinger/asteroseismology
    shift = ranges[:,0]
    scale = np.array([(b-a) for a,b in ranges])
    init_conds = []
    for i in range(skip, N+skip):
        vals = shift+np.array(i4_sobol(len(ranges), i)[0])*scale
        init_conds += [[tmp for tmp in vals]]
        for j, val in enumerate(vals):
            if np.isnan(vals[j]):
                vals[j] = 0
    return init_conds