import numpy as np
import scipy as sp
import matplotlib.pyplot as mpl

Initial_flux=5e13

x=np.linspace(0,1,100)
print(x)
def Intensity_profile(x):
    I= np.exp(x)
    return I

I=Intensity_profile(x)

mpl.plot(x,I)
mpl.show