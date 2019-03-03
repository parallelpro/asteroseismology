from .IO import *
from .peakbagging import *
from .tools import *

__all__ = IO.__all__
__all__.extend(peakbagging.__all__)
__all__.extend(tools.__all__)

'''
import inspect
currentfilepath = inspect.stack()[0][1]
currentdirpath = "/".join(currentfilepath.split("/")[0:-1]) # assume running unix-like system
import sys
sys.path.append(currentdirpath+"/")
print(currentdirpath)

from IO import read_yu_power_spectra
'''