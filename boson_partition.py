#!/usr/bin/env python3

### This codes can calculate the energy of a quantum harmonic oscilator for a given hw in finite temp
#IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

def e_k(bhw, k, d):
    import numpy as np
    #This function return the energy of a signle particle in a harmonic trap at T=1/(beta*k)
    #in units of hw0
    bhw = np.array(bhw)
    return 0.5*d*k*((1 + np.exp(-k*bhw))/(1-np.exp(-k*bhw)))
    
def z_k(bhw, k, d):
    import numpy as np
    #This function return the partition of a signle particle in a harmonic trap at T=1/(beta*k)
    bhw = np.array(bhw)
    return (np.exp(-k*bhw/2)/(1-np.exp(-k*bhw)))**d

def dz_k(bhw, k, d):
    import numpy as np
    bhw = np.array(bhw)
    return -0.5*d*k*z_k(bhw,k,d)*((1+np.exp(-k*bhw))/(1-np.exp(-k*bhw)))

def Z_n(bhw, n, d):
    
    if n == 0:
        return 1.0
    else:
        sig = 0.0
        for k in range(1,n+1):
#            print(k)
#            print(z_k(bhw,k,d)*Z_n(bhw,n-k,d))
            sig = sig + (z_k(bhw,k,d)*Z_n(bhw,n-k,d))
        return 1/n*sig        

def dZ_n(bhw, n, d):
    
    if n == 0:
        return 0.0
    else:
        sig = 0.0
        for k in range(1,n+1):
            sig = sig + (z_k(bhw,k,d)*dZ_n(bhw,n-k,d) +dz_k(bhw,k,d)*Z_n(bhw,n-k,d))
        return 1/n*sig     

def Zf_n(bhw, n, d):
    
    if n == 0:
        return 1.0
    else:
        sig = 0.0
        for k in range(1,n+1):
#            print(((-1)**(k-1))*(z_k(bhw,k,d)*Zf_n(bhw,n-k,d)))
            sig = sig + ((-1)**(k-1))*(z_k(bhw,k,d)*Zf_n(bhw,n-k,d))
        return 1/n*sig         

def dZf_n(bhw, n, d):
    
    if n == 0:
        return 0.0
    else:
        sig = 0.0
        for k in range(1,n+1):
            sig = sig + ((-1)**(k-1))*(z_k(bhw,k,d)*dZf_n(bhw,n-k,d) +dz_k(bhw,k,d)*Zf_n(bhw,n-k,d))
        return 1/n*sig     


#############################################################################################################


f, ((ax1,ax2)) = plt.subplots(2,1, figsize=(3.37,1.5*3.37),dpi=600,sharex='all',gridspec_kw={'hspace': 0})

#HARMONIC FORCE CONSTANT!
# omega = 0.003 #in eV
omega = 0.02673                     # [eV]
omega = omega/27.2114 #eV to Ha
omega_kJmol = omega/3.8088E-4 #Ha to kJmol

#NUMBER OF ATOMS
# Dimensions
d = 2
Natoms = 10
bhw_grid = [0.75, 1, 1.5, 2, 3]
e1 = e_k(bhw_grid, 1, d)
Ed_analytic2 = Natoms*e1
Z2 = Z_n(bhw_grid,Natoms,d)
dZ2_dbeta = dZ_n(bhw_grid, Natoms,d)
Eb_analytic3 = -1/Z2*dZ2_dbeta

Zf2 = Zf_n(bhw_grid,Natoms,d)
# dZf2_dbeta = dZf_n(bhw_grid,Natoms,d)
# Ef_analytic3 = -1/Zf2*dZf2_dbeta

#Results Harmonic N=10
Boson_Energy_N10_harmonic = [18.64948663, 13.02964836, 11.18361578, 10.4824691, 10.10549314]


fig = plt.figure(figsize=(10, 7))
tit = "Boson Energy - Harmonic"
plt.plot(bhw_grid, Eb_analytic3, '-o', color="blue", label="Analytical")
plt.plot(bhw_grid, Boson_Energy_N10_harmonic, '-o', color="red", label="NBS N=10")
plt.title(tit)
plt.xlabel("bhw")
plt.ylabel("Boson Energy")
plt.legend(loc='upper right')
plt.show()

pass

