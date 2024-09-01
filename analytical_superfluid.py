#IMPORTS
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
import math
import itertools
from scipy.ndimage import gaussian_filter1d
from functions_PIMD import read_lammps_table

# to know all premutations
list(itertools.permutations([1, 2, 3, 4, 5]))

d = 2   # dimension
N = 10
sigma = 1
convmeV_J =1.60217656535*10**-22
convJ_kj2mol = 6.022*10**20
hbar = 1.05457182e-34
hw = 26.73 * convmeV_J    # J
omega = hw / hbar
# T = 298
kb = 1.380649e-23                     # Boltzmann const in kJ/mol/K                             # K
# beta = 1/(kb*T)

            # kJ/mol /s
# bhw = beta*hbar*omega
#HARMONIC FORCE CONSTANT!

# This are the permutations for N quantum partilces. Below are for N = 3 N = 4 and N = 10
# permset = [[3, 0, 0], [1, 1, 0],  [0, 0, 1]]                       # N = 3
# permset = [[4, 0, 0, 0], [2, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1], [0, 2, 0, 0]]  # N =4
permset = \
[[10,  0,  0, 0, 0, 0,  0,  0,  0,  0] , [8, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[7, 0, 1, 0, 0, 0, 0, 0, 0, 0], [6, 2, 0, 0, 0, 0, 0, 0, 0, 0],
[6, 0, 0, 1, 0, 0, 0, 0, 0, 0], [5, 1, 1, 0, 0, 0, 0, 0, 0, 0],
[5, 0, 0, 0, 1, 0, 0, 0, 0, 0], [4, 3, 0, 0, 0, 0, 0, 0, 0, 0],
[4, 1, 0, 1, 0, 0, 0, 0, 0, 0], [4, 0, 2, 0, 0, 0, 0, 0, 0, 0],
[4, 0, 0, 0, 0, 1, 0, 0, 0, 0], [3, 2, 1, 0, 0, 0, 0, 0, 0, 0],
[3, 1, 0, 0, 1, 0, 0, 0, 0, 0], [3, 0, 1, 1, 0, 0, 0, 0, 0, 0],
[3, 0, 0, 0, 0, 0, 1, 0, 0, 0], [2, 4, 0, 0, 0, 0, 0, 0, 0, 0],
[2, 2, 0, 1, 0, 0, 0, 0, 0, 0], [2, 1, 2, 0, 0, 0, 0, 0, 0, 0],
[2, 1, 0, 0, 0, 1, 0, 0, 0, 0], [2, 0, 1, 0, 1, 0, 0, 0, 0, 0],
[2, 0, 0, 2, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 1, 0, 0],
[1, 3, 1, 0, 0, 0, 0, 0, 0, 0], [1, 2, 0, 0, 1, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
[1, 0, 3, 0, 0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
[1, 0, 0, 1, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
[0, 5, 0, 0, 0, 0, 0, 0, 0, 0], [0, 3, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 2, 2, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 1, 0, 0, 0, 0],
[0, 1, 1, 0, 1, 0, 0, 0, 0, 0], [0, 1, 0, 2, 0, 0, 0, 0, 0, 0],
[0, 1, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 2, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 1, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
[0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]

############################################################################
# The below equations are derived from the article " Path-integral Monte Carlo simulations of quantum dipole systems in
# traps: Superfuidity, quantum statistics, and structural properties"   section E. Ideal bosons and fermions
# by Tobias Dorngheim. You can calculate the Cq given an N and also the analytical solution for a harmonic potential
# for the local superfluid fraction.

def is_arr_in_list(myarr, list_arrays):
    return next((True for elem in list_arrays if np.array_equal(elem, myarr)), False)

def to_cycles(perm):
    pi = {i+1: perm[i] for i in range(len(perm))}
    cycles = []

    while pi:
        elem0 = next(iter(pi))
        this_elem = pi[elem0]
        next_item = pi[this_elem]

        cycle = []
        while True:
            cycle.append(this_elem)
            del pi[this_elem]
            this_elem = next_item
            if next_item in pi:
                next_item = pi[next_item]
            else:
                break

        cycles.append(cycle)

    return cycles

def get_cqs(N):
    perms = itertools.permutations(list(range(1, N + 1)))
    all_cqs = []
    for perm in list(perms):
        cycles = to_cycles(perm)  # Break the permutation into cycles
        cqs = np.zeros(N, int)  # Array that will hold all the cycle lengths for this permutation
        for cycle in cycles:
            cqs[len(cycle) - 1] += 1

        if not is_arr_in_list(cqs, all_cqs):
            all_cqs.append(cqs)

    degeneracy = []

    for cqs in all_cqs:
        denom = 1
        for q in range(1, N + 1):
            denom *= np.math.factorial(cqs[q - 1]) * (q ** (cqs[q - 1]))

        degeneracy.append(int(np.math.factorial(N) / denom))

    return [(all_cqs[i], degeneracy[i]) for i in range(len(all_cqs))]

print(get_cqs(3))


def get_sign(cq, boson=True):
    if boson:
        return 1
    sum = 0
    for q in range(len(cq)):
        sum += q * cq[q]
    return int((-1)**sum)


def partition_func(beta):
    return (np.exp(-beta / 2) / (1 - np.exp(-beta)))**2


def ancilla_1(q, beta, N, is_boson = True):
    result = 0
    all_cqs, degeneracies = zip(*get_cqs(N))
    for cq in all_cqs:
        sign = get_sign(cq, is_boson)
        term = 1
        for r in range(1, N + 1):
            c_r = cq[r - 1]
            term *= (partition_func(r*beta) ** c_r) / (np.math.factorial(c_r) * (r ** c_r))
        result += cq[q - 1] * sign * term
    return result


def get_superfluid_fraction(N, bosons, inv_beta):
    numer, denom = 0, 0
    beta = 1 / inv_beta

    for q in range(1, N + 1):
        exponent = np.exp(-q * beta)
        anc = ancilla_1(q, beta, N, bosons)
        numer += anc * (q**2) * exponent / (1 - exponent)**2
        denom += anc * q * (1 + exponent) / (1 - exponent)

    return 1 - 2 * beta * (numer / denom)

##################################################################################

def z_k(bhw, k, d):
    #This function return the partition of a signle particle in a harmonic trap at T=1/(beta*k)
    z_1k = (np.exp(-k*bhw/2) / (1 - np.exp(-k*bhw))) ** d
    return z_1k

def z_3(bhw):
    return (1/6)*(z_k(bhw, 1, 2)+3*z_k(bhw, 1, 2)*z_k(bhw, 2, 2)+2*z_k(bhw, 3, 2))


def ancilla(permset, Natoms, q, bhw):
    gamma = 0
    for j, set in enumerate(permset):
        gamma_tmp = 1   # for multiplication not to zero
        for r in range(1, Natoms+1):
            num = (z_k(bhw, r, d)**set[r-1] )
            denom = (math.factorial(set[r-1])*(r**set[r-1]))
            gamma_tmp *= num / denom
        gamma += set[q-1] * gamma_tmp
    return gamma


def superfluid_frac(beta, permset):
    sf_array = np.zeros(interval)
    for i, b in enumerate(beta):
        bhw = b * hbar * omega
        I = 0
        I_cl = 0
        for q in range(1, N+1):
            ancila_func = ancilla(permset, N, q, bhw)
            I += ancila_func * q**2 * ((np.exp(-q*bhw))/(1 - np.exp(-q*bhw))**2)
            I_cl += ancila_func * q * ((1.0 + np.exp(-q*bhw))/(1 - np.exp(-q*bhw)))
        const = 2 * b * hbar * omega
        sf_array[i] = 1 - const * (I/I_cl)
    return sf_array

interval = 100
T = np.linspace(3, 1500, interval)
beta = 1/(kb*T)
xh = np.array([5.17, 51.698, 103.396, 155.09, 206.79, 310.19, 413.584717393, 600, 1000, 1300])
xa = np.array([5.17, 51.698, 103.396, 155.09, 206.79, 310.19, 413.584717393, 600, 1000, 1300])
xa18 = np.array([51.698, 103.396, 155.09, 206.79, 310.19, 413.584717393, 600, 1000, 1300])
xa34 = np.array([51.698, 103.396, 155.09, 206.79, 310.19, 413.584717393, 600, 1000, 1300])
# xm = np.array([103.396, 155.09, 206.79, 310.19, 413.584717393, 600, 1000, 1300 ])
xm = np.array([103.396, 155.09, 206.79, 260.0, 310.19, 1000])
print(T)

sf = superfluid_frac(beta, permset)

# Dorenhaim N = 4 hw =3meV #####################################################
# temp = np.array([0.2, 0.5, 0.75, 1.0, 1.5])  # Dornheim hw = 3 mev N =4
# rhos_rho_4 = np.array([ 0.9775244107560579, 0.7251923876895442, 0.47414544316421153, 0.2707497709815957, 0.10367548884054434 ]) # Dornheim hw = 3 mev N =4
# rhos_rho_4_interaction = np.array([ 0.98338, 0.28387, 0.142395, 0.084930 ]) # Dornheim hw = 3 mev N =4 and coul interaction
# Dorenhaim N = 4 hw =3meV - COUL #####################################################
################################################################################
# this is how you choose  1/bhw
xbh = (kb*xh)/(hbar*omega)
xba = (kb*xa)/(hbar*omega)
xba18 = (kb*xa18)/(hbar*omega)
xba34 = (kb*xa34)/(hbar*omega)
xbm = (kb*xm)/(hbar*omega)
# y = [1.007, 1.021, 0.92, 0.53031]
harmonic_N10 = [1.05527506, 1.007338, 0.9619629, 0.865853112546111, 0.7359276572, 0.51489857586032, 0.28891, 0.08829169023, 0.013483266, 0.0073199980]
harmonic_N10_std = [ 0.12213937 , 0.01175,  0.01388263 ,0.025518990969505655, 0.025080029,  0.018693438,   0.0127278, 0.0068143998588, 0.00164700,  0.00995799]
anharmonic_N10 = [1.021645, 0.983672095, 0.96305898, 0.8800960, 0.81162318,  0.594937461691, 0.4321753, 0.23506402, 0.05263707, 0.0273645502]
anharmonic_N10_std =  [0.0201894, 0.01436139, 0.0099579,  0.015465642974, 0.027264963125, 0.0175176757, 0.00913743 , 0.04071257132 , 0.006200658, 0.001949865]
# moire_N10 = [0.9457072034, 0.872150,  0.6987139806334333, 0.5325468, 0.352789534, 0.55, 1.49923, 2.29501]
# moire_N10_std = [0.0090048089,  0.0329257710, 0.03488461717582711, 0.0191231407, 0.04939740, 0.31779 , 0.36468225  , 0.2800 ]


moire_N10 = [1.03071, 0.9430585, 0.814, 0.03958, 0.02028, 0.00281]
moire_N10_std = [0.0119, 0.0195684,  0.04150, 0.0131,  0.00346, 0.000889]

anharmonic_18 = [0.989440, 0.90920706, 0.7886045, 0.635538, 0.397040, 0.2351382, 0.0867554, 0.025893981413, 0.01480793]
anharmonic_18_std = [0.00627677, 0.016210345, 0.03408010, 0.0197705, 0.03256892,  0.0126134, 0.0135285048, 0.0022744160, 0.001254665]
anharmonic_34 = [1.00975258, 0.989243096, 0.95279155,0.8972929,  0.73453, 0.58874, 0.394430359, 0.102317, 0.05576935778]
anharmonic_34_std = [0.009872, 0.0070224, 0.028980641, 0.030998152, 0.03891400, 0.00673498, 0.061720, 0.0090784, 0.010713974437]

rho_155 = 'Temp:  155.09 rho:  0.9053143875883284 Area:  2.3234327426625244e-35 Icl:  1.851714080430781e-46 skip:  200003 nstart:  50000 Beads:  16'
rho_200 = 'Temp:  206.79 rho:  0.8177273485034163 Area:  1.6712746287940947e-35 Icl:  1.9661709728607863e-46 skip:  200003 nstart:  50000 Beads:  16'
rho_300 = 'Temp:  310.19 rho:  0.6059761198832206 Area:  9.624993915643148e-36 Icl:  2.2920189046961885e-46 skip:  200003 nstart:  50000 Beads:  16'
rho_400 = 'Temp:  413.58 rho:  0.4656925613453155 Area:  5.971312560604828e-36 Icl:  1.8503068013442685e-46 skip:  200003 nstart:  50000 Beads:  12'
fig = plt.figure()
q = (kb*T)/(hbar*omega)   # 1/bhw
p = sf
# plt.plot(q, p, ':', color="black", linewidth=3)
# plt.plot(temp, rhos_rho_4, 'o', color="blue", label="NBS N=4")
# plt.plot(xbh, harmonic_N10, '-o', color="blue", label="harmonic potential")
plt.errorbar(xbh, harmonic_N10, yerr=harmonic_N10_std , fmt="--o", color="C0", ecolor="C0", linewidth=2)
# plt.plot(xba, anharmonic_N10, '-o', color="red", label="anharmonic potential")
plt.errorbar(xba, anharmonic_N10, yerr=anharmonic_N10_std, fmt="-.o", color="C1", ecolor="C1", linewidth=2)
plt.errorbar(xba18, anharmonic_18, yerr=anharmonic_18_std, fmt="-.o", color="C2", ecolor="C2", linewidth=2)
plt.errorbar(xba34, anharmonic_34, yerr=anharmonic_34_std, fmt="-.o", color="C3", ecolor="C3", linewidth=2)
plt.plot(xbm, moire_N10, '-o', color="C4", label="Moire Potential", linewidth=2)
plt.errorbar(xbm, moire_N10, yerr=moire_N10_std, fmt="-o", color="C4", ecolor="C4",linewidth=2)
# plt.title("Local Superfluid Fraction: N = 10", fontsize=16)
# plt.ylabel(r'$ \gamma_sf $', fontsize=16)
plt.ylabel('superfluid fraction', fontsize=16)
plt.xticks(fontsize=16)
plt.xlabel(r'$ 1 / \beta\hbar\omega $', fontsize=16)
plt.xticks(fontsize=16)
plt.ylim(0, 1.2)
plt.yticks(fontsize=16)
# plt.legend()
plt.show()



###################################################################### Inflection points
# Create a fit and find a better estimate for the inflection points
# https://stackoverflow.com/questions/62537703/how-to-find-inflection-point-in-python

rawh =[1.05527506, 1.007338, 0.9619629, 0.865853112546111, 0.7359276572, 0.51489857586032, 0.28891, 0.08829169023, 0.013483266, 0.0073199980] # Harmonic
rawa = [1.021645, 0.983672095, 0.96305898, 0.8800960, 0.81162318,  0.594937461691, 0.4321753, 0.23506402, 0.05263707, 0.0273645502] # Anharmonic
rawa18 = [0.989440, 0.90920706, 0.7886045, 0.635538, 0.397040, 0.2351382, 0.0867554, 0.025893981413, 0.01480793]
rawa34 =   [1.00975258, 0.989243096, 0.95279155,0.8972929,  0.73453, 0.58874, 0.394430359, 0.102317, 0.05576935778]

smoothh = gaussian_filter1d(rawh, 1)
smootha = gaussian_filter1d(rawa, 1)
smootha18 = gaussian_filter1d(rawa18, 1)
smootha34 = gaussian_filter1d(rawa34, 1)
# compute second derivative
smooth_d2h = np.gradient(np.gradient(smoothh))
smooth_d2a = np.gradient(np.gradient(smootha))
smooth_d2a18 = np.gradient(np.gradient(smootha18))
smooth_d2a34 = np.gradient(np.gradient(smootha34))
# find switching points
inflsh = np.where(np.diff(np.sign(smooth_d2h)))[0]
inflsa = np.where(np.diff(np.sign(smooth_d2a)))[0]
inflsa18 = np.where(np.diff(np.sign(smooth_d2a18)))[0]
inflsa34 = np.where(np.diff(np.sign(smooth_d2a34)))[0]
# plot results
plt.plot(xbh, rawh, '-o', color='blue', label='Harmonic Potential')
plt.plot(xba, rawa, '-o', color='red', label='Anharmonic Potential')
plt.plot(xba18, rawa18, '-o', color='purple', label='Anharmonic Potential 18meV')
plt.plot(xba34, rawa34, '-o', color='yellow', label='Anharmonic Potential 34meV')
# plt.plot(xbh, smooth_d2h / np.max(smooth_d2h), label='Second Derivative Harmonic')
# plt.plot(xba, smooth_d2a / np.max(smooth_d2a),label='Second Derivative Anharmonic')
# plt.plot(xba18, smooth_d2a18 / np.max(smooth_d2a18),label='Second Derivative Anharmonic 18 ')
# plt.plot(xba34, smooth_d2a34 / np.max(smooth_d2a34),label='Second Derivative Anharmonic 34 ')
# for i, infl in enumerate(inflsh, 1):
#     plt.axvline(x=xbh[infl], color='b', label=f'Inflection Point Harmonic')
# for i, infl in enumerate(inflsa, 1):
#     plt.axvline(x=xba[infl], color='r', label=f'Inflection Point Anharmonic')
plt.axvline(x=1.069, color='b')
plt.axvline(x=1.255, color='r')
plt.axvline(x=0.8684, color='purple')
plt.axvline(x=1.652, color='y')
plt.ylabel(r'$ \gamma_sf $', fontsize=15)
plt.xlabel(r'$ 1 / \beta\hbar\omega $', fontsize=15)
plt.legend()
plt.show()

C = 1.0622 # the value for 1/bhw of the inflection point (x coordinate)
T = C * hbar*omega / kb

### Dornheim Dipole harmonic and Coulomb harmonic trap - only for benchmark ###
beta_myresult = [1, 0.5, 1/5]
sf_myresult = [0.07559, 0.1514, 0.53536]
beta_harmonic = [0.01666727, 0.16666626, 0.33333252, 0.49998589, 0.66665859, 1.00000401, 1.3333324 , 1.93430609, 3.22384348, 4.19099653]
beta_harmonic_D = [0.0965035, 0.1426573, 0.197203, 0.24755, 0.3314685, 0.4993007, 0.6671329, 0.9986014, 1.3321678]
sf_harmonic_D = [0.9958420, 0.983367, 0.9251559, 0.8191268, 0.61122661, 0.307692, 0.182952, 0.0914761, 0.0582120]
temp = np.array([0.2, 0.5, 0.75, 1.0, 1.5])  # Dornheim hw = 3 mev N =4
rhos_rho_4 = np.array([ 0.9775244107560579, 0.7251923876895442, 0.47414544316421153, 0.2707497709815957, 0.10367548884054434 ]) # Dornheim hw = 3 mev N =4
rhos_rho_4_interaction = np.array([ 0.98338, 0.28387, 0.142395, 0.084930 ,0]) # Dornheim hw = 3 mev N =4 and coul interaction
beta_harmonic_dipole = [0.0654, 0.1, 0.14073, 0.2, 0.333, 0.5, 0.666, 0.8, 1]
sf_harmonic_dipole = [0.99137,0.9913,  0.974137,0.9103448, 0.575862, 0.2793, 0.16034, 0.09655,0.08103448  ]

# plt.plot(temp, rhos_rho_4, '-o', color='green', label='My results of Dorn Harmonic Trap N = 4 ')
plt.plot(temp, rhos_rho_4_interaction, '-o', color='C0', label='NBS: Coulomb ')
plt.plot(beta_myresult, sf_myresult, '-o', color='C1', label='NBS: Dipole')
plt.plot(beta_harmonic_D, sf_harmonic_D, '-o', color='C2', label='Dornheim: Coulomb')
plt.plot(beta_harmonic_dipole, sf_harmonic_dipole, '-o', color='C3', label='Dornheim: Dipole ')
plt.ylabel('$ \gamma_sf $', fontsize=15)
plt.xlabel( r'$ 1 / \beta\hbar\omega $', fontsize=15)
plt.legend()
plt.title("NBS vs Dornherim results : N=4  $ \lambda = 3 $")
plt.show()
#################################################################################
# read_lammps_table()  # read table of potential



