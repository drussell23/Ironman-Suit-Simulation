import ctypes 
import numpy as np 
import os 
import sys

# Load your shared library (.so/.dll/.dylib)
def load_library():
    if sys.platform.startswith('linux'):
        libname = 'libaerodynamics_physics_plugin.so'
    elif sys.platform.startswith('darwin'):
        libname = 'libaerodynamics_physics_plugin.dylib'
    elif sys.platform.startswith('win'):
        libname = 'aerodynamics_physics_plugin.dll'
    else:
        raise RuntimeError("Unsupported platform")
    
    libpath = os.path.join(os.path.dirname(__file__), "..", "build", libname)
    
    if not os.path.isfile(libpath):
        raise FileNotFoundError(f"Library not found at '{libpath}'")
    
    return ctypes.CDLL(libpath)