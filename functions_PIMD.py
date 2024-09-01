# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
N = 3 # number of quantum particles
# Conversions
conv = 2625.499638   # [1 Hartree = 2625.499638 kJ/mol]    or the other way arround 3.8088E-4
conv_1 = 96.4853     # eV to kJ/mol
# Constants
part = 2                     # Quantum Particles
kb = 3.16683 * 10 ** (-6)    # Boltzman in Hartree/K
hbar, m = 1, 1               # hbar and mass

# For Exitons                                            from [ ... ]
# hw = 26.73meV   m=0.84m_e
k_exitons = 8.10542107e-07  # k = 0.84mw^2 spring constant Hartree/Bohr^2
w_exitons = math.sqrt(k_exitons / 0.84*m)   # Oscillator Frequency
hw_exitons = (hbar * w_exitons) / (1/conv)  # 2.5790520686098652 kJ/mol
#                                                       until here [ ... ]
# For Barak's Fermions Article                           from [ ... ]
#hw = 3meV
k_harmonic_fermions = 1.21647924 * 10 ** (-8)  # spring constant Hartree/Bohr^2
w = math.sqrt(k_harmonic_fermions / m)
hw = hbar*w  # 0.0001104536101718726   # [sqrt(Hartree /kg*K]
hw_2 = 0.2894557624503526   # 0.003 * conv_1  # 0.289458 kJ/mol    3meV as in the Article (to have same number as Barak)
#                                                       until here [ ... ]
# Cut the data until energy minimization
cut_data_exitons = 1000  # for data
cut_log_exitons = cut_data_exitons - 1       # for pimdb log the minus 1 is for the data to be with same index
cutoff_exitons = 100  # for block avg
perm_length = 3

# Cut the data until energy minimization  PIMD_F_B_S
cut_data = 1000  # for data
cut_log = cut_data - 1       # for pimdb log the minus 1 is for the data to be with same index
cutoff = 0  # for block avg

#################################################################################################################

def hw_to_k_const(hw_ev, mass_factor):
    '''
    :param hw: hbar omega of the harmonic oscillator in meV
    mass_factor: for example exitons are 0.84 of electron mass so : 0.84
    conv_1 = 96.4853  -> eV to kJ/mol
    conv = 2625.499638   [1 Hartree = 2625.499638 kJ/mol]
    :return: k - spring constant in Hartree / Bohr^2 (for LAMMPS unit: electron)
    '''
    hw_ev = hw_ev * 0.001  # from meV to eV
    w0 = hw_ev * conv_1 / conv
    k = mass_factor * w0**2
    return k


def temp_for_lammps(hw_ev, C):
    '''
    :param C: hw_ev in meV ; number of bhw desired (for example bhw = 6 = C)
    :return: Returns the temperature [K] to put in LAMMPS and beta for python code [[1 / kJ/mol]
    '''
    # Conversion from Hartree to kJ/mol
    hw_ev = hw_ev * 0.001   # from meV to eV
    hw_kj_mol = hw_ev * conv_1
    kb1 = kb * conv  # [kJ/mol/K]
    b = C / hw_kj_mol # [1 / kJ/mol]
    T = 1/(kb1*b)
    return T, b

print("hello")
def center_of_mass(beads, t, natoms):
    xcm = 0.0
    ycm = 0.0
    N_count = 0.0
    M = len(beads)
    for bead in range(M):
        for atm in range(natoms):  # 10 ntoms atm =  quantum atom
            N_count += 1
            xcm += beads[bead][t][atm][0]  # 0 is x axis
            ycm += beads[bead][t][atm][1] # 1 is y axis

    xcm = xcm / N_count
    ycm = ycm / N_count

    return xcm, ycm


def statistical_error_estimation(fermionic_energy, wj):
    '''
    :param fermionic_energy: list of the calculated energy of same simulation with different seed
    :param wj: list of the calculated Wj of same simulation with different seed
    :return: The average energy and stdv
    '''
    wj_2 = wj**2
    sum_wj = sum(wj)
    avg_energy = sum(fermionic_energy * wj) / sum_wj
    n = sum_wj**2 / sum(wj_2)
    numerator = 0
    for j in range(len(fermionic_energy)):
        numerator += wj[j] * (fermionic_energy[j] - avg_energy)**2
    stdv = (n/(n-1)) * (numerator / sum_wj)
    energy_error = np.sqrt(stdv) / np.sqrt(n)
    return avg_energy, energy_error


def block_averaging(cutoff, block_size, data):
    '''

    :param cutoff: Cut the cutoff amount of data from the beggining of the simulation since it has not achieved equilibrium.
    :param block_size: obtained from a STDV vs Blocksize simulation to get the corresponding STDV
    :param data: data that is evaluated
    :return: the average and stdv of the data.
    '''
    data_cut = data[cutoff:]
    data_len = len(data_cut)
    number_of_blocks = int(data_len/block_size)
    average_array = np.zeros(number_of_blocks)
    for i in range(number_of_blocks):
        average_array[i] = (data_cut[i * block_size:(i + 1) * block_size]).mean()
    averages = average_array.mean()
    stdv = np.std(average_array, ddof=1) / np.sqrt(number_of_blocks)

    return number_of_blocks, averages, stdv


def make_block_average(cutoff, data):
    '''
    :return: block averaging graph
    '''
    block_size_array = np.linspace(1, 1000, 500).astype(int)
    avg_array = np.zeros(len(block_size_array))
    stdv_array = np.zeros(len(block_size_array))
    number_of_blocks_array = np.zeros(len(block_size_array))

    for i, value in enumerate(block_size_array):
        number_of_blocks, avg, stdv = block_averaging(cutoff, block_size=value, data=data)
        avg_array[i] = avg
        stdv_array[i] = stdv
        number_of_blocks_array[i] = number_of_blocks


    figblock = plt.figure()
    plt.plot(block_size_array, stdv_array, label="stdv", color="red")
    plt.xlabel("Block Size")
    plt.ylabel("STDV")
    plt.legend()
    plt.show()
# To run block averaging
# potenital_energy, time_step, trap, newvir, trapvir = etot_b_2d_harmonic(number_of_files, path)
# make_block_average(cutoff, pot)


def read_file_pot(filename, cut, s_rows, s_footer):
    '''
    :param filename: input log.lammps.{} file
    :param cut: The first "cut" values will be eliminated.
    :param s_rows: skiprows in log.lammps file since there is no data
    :param s_footer: skipfooter in log.lammps file since there is no data
    :return: potenital_energy, time_step, trap, newvir, trapvir
    '''
    df = pd.read_csv(filename, sep='\s+', engine='python', skiprows=s_rows, skipfooter=s_footer) # delimiter='\s+'
    time_step = df['Time'][cut:]
    potenital_energy = df['PotEng'][cut:]
    trap = df['c_trap'][cut:]
    trapvir = df['c_trapvir'][cut:]
    newvir = df['v_newvir'][cut:]
    return potenital_energy, time_step, trap, newvir, trapvir


def avg_sign(filename, beta, cut):
    '''
    :param filename: pimdb.log file
    :param beta: beta in 1 / kJ/mol
    :param cut: he first "cut" values will be eliminated.
    :return: the average sign value
    '''
    # Here the pimdb.log file is in units of kJ/mol hence beta has to also be in same units
    df = pd.read_csv(filename, delimiter='\s+')
    df = df.iloc[cut:-1] # I added this and removed "cut from all [] example  e_1_1 = df.iloc[cut:, 0]
    v_n_column = df.iloc[:, -1]

    w_b = np.exp(- beta * v_n_column)         #  W(3)_Bosons
    print(len(w_b))
    e_1_1 = df.iloc[:, 0]
    e_2_2 = df.iloc[:, 1]
    e_2_1 = df.iloc[:, 2]
    e_3_3 = df.iloc[:, 3]
    e_3_2 = df.iloc[:, 4]
    e_3_1 = df.iloc[:, 5]
                                             # W(3) Fermions
    w_f_0 = 1
    w_f_1 = np.exp(- beta * e_1_1)
    w_f_2 = 0.5 * ((np.exp(- beta * e_2_1) * w_f_1) - (np.exp(- beta * e_2_2) * w_f_0))
    w_f_3 = (1/3) * ((np.exp(- beta * e_3_1) * w_f_2) - (np.exp(- beta * e_3_2) * w_f_1) + (np.exp(- beta * e_3_3) * w_f_0))

                                             #  W(3)_Bosons second way
    # w_b_0 = 1
    # w_b_1 = np.exp(- beta * e_1_1)
    # w_b_2 = 0.5 * ((np.exp(- beta * e_2_1) * w_b_1) + (np.exp(- beta * e_2_2) * w_b_0))
    # w_b_3 = (1/3) * ((np.exp(- beta * e_3_1) * w_b_2) + (np.exp(- beta * e_3_2) * w_b_1) +
    #                  (np.exp(- beta * e_3_3) * w_b_0))

    s = w_f_3 / w_b
    # s1 = w_f_3 / w_b_3
    return s


def etot_b_2d_harmonic(number_of_files, path):  # Bosons
    '''
    In the Boson Ensemble
    :param number_of_files: number of beads
    :param path: the file path
    :return: time_step, pot, avg_pot_exp, stdv_pot, trap, trapvir, newvir
    '''
    file = [path+'log.lammps.{}'.format(k) for k in range(0, number_of_files)]  # Makes a list of log.lammps.{}
    pot, trap, newvir, trapvir = 0, 0, 0, 0
    time_step = 0
    for i in range(0, number_of_files):
        try:
            potential, time_step, trap1, newvir1, trapvir1 = read_file_pot(file[i], cut_data, 142, 38)  # Harmonic (153,38) Auxiliary(167, 38)  sgnprob(145, 38)
        except:
            potential, time_step, trap1, newvir1, trapvir1 = read_file_pot(file[i], cut_data, 162, 38)
        pot += potential
        trap += trap1
        newvir += newvir1
        trapvir += trapvir1
    number_of_blocks, avg_pot_exp, stdv_pot = block_averaging(cutoff, block_size=900, data=pot)

    return time_step, pot*conv, avg_pot_exp, stdv_pot, trap*conv, trapvir*conv, newvir*conv


def etot_f_2d_harmonic(number_of_files, path, beta):   # Fermions
    '''
    :param number_of_files: number of beads
    :param path: location of path
    :param beta: beta in 1 / kJ/mol
    :return: Fermionic Energy for Harmonic Oscilator potentail
    '''
    # Fermions Sign Re-Weighting # since pimdb.log is bigger file than log.lammps then I reset indexes
    file_pimdb = path + 'pimdb.log'
    time_step, pot, avg_pot_exp, stdv_pot, trap, trapvir, newvir = etot_b_2d_harmonic(number_of_files, path)
    sign_array = avg_sign(file_pimdb, beta, cut_log).reset_index(drop=True)   # Sign average
    wj = sum(sign_array)    # Weights for sign average of each simulation

    e_s_num_h = (((trap.reset_index(drop=True) / number_of_files)) + newvir.reset_index(drop=True)) * sign_array    # (orange) for harmonic potential
    e_s_num_h = np.mean(e_s_num_h)
    s_denom = sign_array
    s_denom = np.mean(s_denom)

    avg_etot_f_h = e_s_num_h / s_denom

    return time_step,  avg_etot_f_h, sign_array, wj


def etot_f_2d_harmonic_aux(number_of_files, path, beta):   # Fermions
    '''
    :param number_of_files: number of beads
    :param path: location of path
    :param beta: beta in 1 / kJ/mol
    :return: Fermionic Energies for the Auxiliary system and the Bogoliubov ineq. term
    '''
    # Fermions Sign Re-Weighting # since pimdb.log is bigger file than log.lammps then I reset indexes
    file_pimdb = path + 'pimdb.log'
    time_step, pot, avg_pot_exp, stdv_pot, trap, trapvir, newvir = etot_b_2d_harmonic(number_of_files, path)
    sign_array = avg_sign(file_pimdb, beta, cut_log).reset_index(drop=True)               # Sign average
    wj = sum(sign_array)   # Weights for sign average of each simulation

                           # Below - a - Auxiliary: (grey) for sum of potentials that are position dependent
    e_s_num_a = ((pot.reset_index(drop=True) / number_of_files) + newvir.reset_index(drop=True)) * sign_array
    e_s_num_a_mean = np.mean(e_s_num_a)
                           # Below - b - bogoliubov: (purpule) <V_g>_H'   (needs to be subtracted! )
    e_s_num_b = (pot.reset_index(drop=True) - trap.reset_index(drop=True)) * sign_array
    e_s_num_b_mean = np.mean(e_s_num_b)

    s_denom = sign_array
    s_denom_mean = np.mean(s_denom)

                           # Below a is for <E>_H' and b for <V_g>_H'
    avg_etot_f_a = (e_s_num_a_mean ) / s_denom_mean   # Auxiliary
    avg_etot_f_b = (e_s_num_b_mean ) / s_denom_mean   # Bogoliubov



    return time_step, sign_array, wj, avg_etot_f_a, avg_etot_f_b


def ana_3_fermions(beta, hbarw):
    '''
    :return: Analytical result for three Fermions in Canonical Ensemble <E> in harmonic potential
    '''
    exp = np.exp(beta * hbarw)
    exp2 = np.exp(2 * beta * hbarw)
    exp3 = np.exp(3 * beta * hbarw)
    exp4 = np.exp(4 * beta * hbarw)
    exp5 = np.exp(5 * beta * hbarw)
    exp6 = np.exp(6 * beta * hbarw)
    exp7 = np.exp(7 * beta * hbarw)
    exp8 = np.exp(8 * beta * hbarw)
    num = hbarw*(5*exp6 + 31*exp5 + 47*exp4 + 50*exp3 + 47*exp2 + 31*exp + 5)
    denom = (exp-1)*(exp+1)*(exp2+exp+1)*(exp2+4*exp+1)
    etot_3_f = num/denom
    return etot_3_f

# def etot_f_2d_harmonic_aux(number_of_files, path, beta):   # Fermions
#     '''
#     :param number_of_files: number of beads
#     :param path: location of path
#     :param beta: beta in 1 / kJ/mol
#     :return: Fermionic Energies for the Auxiliary system and the Bogoliubov ineq. term
#     '''
#     # Fermions Sign Re-Weighting # since pimdb.log is bigger file than log.lammps then I reset indexes
#     file_pimdb = path + 'pimdb.log'
#     time_step, pot, avg_pot_exp, stdv_pot, trap, trapvir, newvir = etot_b_2d_harmonic(number_of_files, path)
#     sign_array = avg_sign(file_pimdb, beta, cut_log).reset_index(drop=True)               # Sign average
#     pot_array = pot.reset_index(drop=True)
#     wj = sum(sign_array)   # Weights for sign average of each simulation
#     # Below - a - Auxiliary: (grey) for sum of potentials that are position dependent
#     e_s_num_a = ((pot.reset_index(drop=True) / number_of_files) + newvir.reset_index(drop=True)) * sign_array
#     number_of_blocks, avg_pot_f_num_a, stdv_pot_f_num = block_averaging(cutoff, block_size=900, data=e_s_num_a)
#     # Below - b - bogoliubov: (purpule) <V_g>_H'   (needs to be subtracted! )
#     e_s_num_b = (pot.reset_index(drop=True) - trap.reset_index(drop=True)) * sign_array
#     number_of_blocks, avg_pot_f_num_b, stdv_pot_f_num = block_averaging(cutoff, block_size=900, data=e_s_num_b)
#
#     s_denom = sign_array
#     number_of_blocks, avg_sign_f_denom, stdv_pot_f_denom = block_averaging(cutoff, block_size=900, data=s_denom)
#     # I am multipling by the conversion from Hartree to kJ/mol since the potential comes from the lammps.log files
#     # and these are in units of Hartree.
#     # Below a is for <E>_H' and b for <V_g>_H'
#     avg_etot_f_a = (avg_pot_f_num_a ) / avg_sign_f_denom   # Auxiliary
#     avg_etot_f_b = (avg_pot_f_num_b ) / avg_sign_f_denom   # Bogoliubov
#
#     number_of_blocks, avg_trap_f_num, stdv_trap_f_num = block_averaging(cutoff, block_size=900, data=trap)
#     number_of_blocks, avg_trapvir_f_num, stdv_trapvir_f_num = block_averaging(cutoff, block_size=900, data=trapvir)
#     number_of_blocks, avg_newvir_f_num, stdv_newvir_f_num = block_averaging(cutoff, block_size=900, data=newvir)
#
#     return time_step, sign_array, wj, avg_etot_f_a, avg_etot_f_b, avg_trap_f_num, \
#            avg_trapvir_f_num, avg_newvir_f_num

#                                                                                # Exciton


def permutation_prob_3(filename, beta, cut, perm_length):
    '''
    :param filename: pimdb.log file
    :param beta: beta in 1 / kJ/mol
    :param cut: he first "cut" values will be eliminated.
    :return: length of array l and permutation praobability
    '''
    # Here the pimdb.log file is in units of kJ/mol hence beta has to also be in same units
    df = pd.read_csv(filename, delimiter='\s+')
    df = df.iloc[cut:-1] # I added this and removed "cut from all [] example  e_1_1 = df.iloc[cut:, 0]

    v_3_column = df.iloc[:, -1]
    v_2_column = df.iloc[:, -2]
    v_1_column = df.iloc[:, -3]
    v_0_column = df.iloc[:, -4]
    # e_1_1 = df.iloc[:, 0]
    # e_2_2 = df.iloc[:, 1]
    # e_2_1 = df.iloc[:, 2]
    e_3_3 = df.iloc[:, 3]
    e_3_2 = df.iloc[:, 4]
    e_3_1 = df.iloc[:, 5]

    v_3 = np.array([v_2_column, v_1_column, v_0_column])
    e_3 = np.array([e_3_1, e_3_2, e_3_3])
    permutation_probability = np.array([])
    length_array = len(e_3[0])
    p_l_denom = np.exp(- beta * (e_3_1 + v_2_column)) + np.exp(- beta * (e_3_2 + v_1_column)) \
                + np.exp(- beta * (e_3_3 + v_0_column))
    for j in range(0, perm_length):
        p_l_num = np.exp(- beta * (e_3[j] + v_3[j]))
        p_l = np.asarray(p_l_num / p_l_denom)
        # permutation_probability = np.append(permutation_probability, np.array(p_l))      #THIS
        permutation_probability = np.append(permutation_probability, np.mean(np.array(p_l)))

    l_array = np.arange(1, perm_length+1)  # array([1, 2, 3])

    # return l_array, permutation_probability.reshape((perm_length), length_array)         #THIS
    return l_array, permutation_probability


def permutation_prob_10(filename, beta, cut, perm_length):
    '''
    :param filename: pimdb.log file
    :param beta: beta in 1 / kJ/mol
    :param cut: he first "cut" values will be eliminated.
    :return: length of array l and permutation praobability
    '''
    # Here the pimdb.log file is in units of kJ/mol hence beta has to also be in same units
    df = pd.read_csv(filename, delimiter='\s+')
    df = df.iloc[cut:-1] # I added this and removed "cut from all [] example  e_1_1 = df.iloc[cut:, 0]

    v_10_column = df.iloc[:, -1]
    v_9_column = df.iloc[:, -2]
    v_8_column = df.iloc[:, -3]
    v_7_column = df.iloc[:, -4]
    v_6_column = df.iloc[:, -5]
    v_5_column = df.iloc[:, -6]
    v_4_column = df.iloc[:, -7]
    v_3_column = df.iloc[:, -8]
    v_2_column = df.iloc[:, -9]
    v_1_column = df.iloc[:, -10]
    v_0_column = df.iloc[:, -11]


    e_10_1 = df.iloc[:, -12]
    e_10_2 = df.iloc[:, -13]
    e_10_3 = df.iloc[:, -14]
    e_10_4 = df.iloc[:, -15]
    e_10_5 = df.iloc[:, -16]
    e_10_6 = df.iloc[:, -17]
    e_10_7 = df.iloc[:, -18]
    e_10_8 = df.iloc[:, -19]
    e_10_9 = df.iloc[:, -20]
    e_10_10 = df.iloc[:, -21]

    v_10 = np.array([v_9_column, v_8_column, v_7_column, v_6_column, v_5_column,
                     v_4_column, v_3_column, v_2_column, v_1_column, v_0_column])
    e_10 = np.array([e_10_1, e_10_2, e_10_3, e_10_4, e_10_5, e_10_6, e_10_7, e_10_8, e_10_9, e_10_10])

    permutation_probability = np.array([])
    length_array = len(e_10[0])
    p_l_denom = np.exp(- beta * (e_10_1 + v_9_column)) + np.exp(- beta * (e_10_2 + v_8_column)) + \
                np.exp(- beta * (e_10_3 + v_7_column)) + np.exp(- beta * (e_10_4 + v_6_column)) + \
                np.exp(- beta * (e_10_5 + v_5_column)) + np.exp(- beta * (e_10_6 + v_4_column)) + \
                np.exp(- beta * (e_10_7 + v_3_column)) + np.exp(- beta * (e_10_8 + v_2_column)) + \
                np.exp(- beta * (e_10_9 + v_1_column)) + np.exp(- beta * (e_10_10 + v_0_column))
    for j in range(0, perm_length):
        p_l_num = np.exp(- beta * (e_10[j] + v_10[j]))
        p_l = np.asarray(p_l_num / p_l_denom)
        # permutation_probability = np.append(permutation_probability, np.array(p_l))      #THIS
        permutation_probability = np.append(permutation_probability, np.mean(np.array(p_l)))

    l_array = np.arange(1, perm_length+1)  # array([1, 2, 3])

    # return l_array, permutation_probability.reshape((perm_length), length_array)          #THIS
    return l_array, permutation_probability


def etot_b_2d_exitons(number_of_files, path, beta):  # Bosons
    '''
    In the Boson Ensemble
    :param number_of_files: number of beads
    :param path: the file path
    :return: time_step, pot, avg_pot_exp, stdv_pot, trap, trapvir, newvir
    '''

    file = [path+'log.lammps.{}'.format(k) for k in range(0, number_of_files)]  # Makes a list of log.lammps.{}
    file_pimdb = path + 'pimdb.log'
    if N == 3:
        l_cond, perm_cond = permutation_prob_3(file_pimdb, beta, cut_log_exitons, 3)
    else:
        l_cond, perm_cond = permutation_prob_10(file_pimdb, beta, cut_log_exitons, 10)

    pot, trap, newvir, trapvir = 0, 0, 0, 0
    time_step = 0
    for i in range(0, number_of_files):
        try:
            potential, time_step, trap1, newvir1, trapvir1 = read_file_pot(file[i], cut_data_exitons, 148, 38)  #3bosons (153,38) 10boson (209,38) 100boson(929, 38)  148-moire
        except:
            potential, time_step, trap1, newvir1, trapvir1 = read_file_pot(file[i], cut_data_exitons, 134, 38)   # 107
        pot += potential
        trap += trap1
        newvir += newvir1
        trapvir += trapvir1
    number_of_blocks, avg_pot_exp, stdv_pot = block_averaging(cutoff, block_size=20, data=pot)
    number_of_blocks_trap, avg_trap_exp, stdv_trap = block_averaging(cutoff, block_size=20, data=trap)

    return time_step, pot*conv, avg_pot_exp*conv, stdv_pot*conv, \
           trap*conv, stdv_trap*conv, trapvir*conv, newvir*conv, l_cond, perm_cond


def ana_3_bosons(beta, hbarw):
    '''
    :return: Analytical result for three Bosons in Canonical Ensemble <E> in harmonic potential
    '''
    exp = np.exp(beta * hbarw)
    exp2 = np.exp(2 * beta * hbarw)
    exp3 = np.exp(3 * beta * hbarw)
    exp4 = np.exp(4 * beta * hbarw)
    exp5 = np.exp(5 * beta * hbarw)
    exp6 = np.exp(6 * beta * hbarw)
    exp7 = np.exp(7 * beta * hbarw)
    exp8 = np.exp(8 * beta * hbarw)
    exp9 = np.exp(9 * beta * hbarw)
    exp10 = np.exp(10 * beta * hbarw)
    num = hbarw*(12*exp10+28*exp9+79*exp8+169*exp7+241*exp6+238*exp5+241*exp4+169*exp3+79*exp2+28*exp+12)
    denom = (exp-1)*(exp+1)*(exp2+exp+1)*(4*exp6+2*exp5+7*exp4+10*exp3+7*exp2+2*exp+4)
    etot_3_b = num/denom
    return etot_3_b


def find_min_x_y(z):
    '''
    To find the (x,y) of all the minima of the Moire Potenital to write an xyz file
     of the boson's initial positions for LAMMPS simulation
    :param z: np array of function of the Moire potential
    :return: (x,y) coordinates
    '''
    z_min = min(z)
    x_cor, y_cor = 0, 0
    global x, y
    for index, value in enumerate(z):
        if value == z_min:
            x_cor = x[index]
            y_cor = y[index]
    return (x_cor, y_cor)


def f_minimum(x):
    '''
    gives the Minimum of a 2D function If this is called: fmin(f_minimum, np.array([0, 0]))
    :param x: numpy array of x and y coordinate of Initial Guess
    :return: the functions value at a certain point (finds minimum)
    '''
    global hbaromega, V_e, pi, moire_period, psi
    z = hbaromega + 2 * V_e * (np.cos((-2 * pi / (math.sqrt(3) * moire_period)) * x[0] - (2 * pi / moire_period) * x[1] - psi) +
                             np.cos((-2 * pi / (math.sqrt(3) * moire_period)) * x[0] + (2 * pi / moire_period) * x[1] - psi) +
                             np.cos(((4 * pi) / (math.sqrt(3) * moire_period)) * x[0] - psi))
    return z

# def myfmt(x, pos):
#     return '{0:.4f}'.format(x)

def myfmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

def moire_plot(units, x_moire_lims, hbaromega, moire_period, V_e, nmtom, mevtoJ, psi, pi):
    import matplotlib.ticker as ticker
    if units == 'meVmeter':
        X, Y = np.meshgrid(np.linspace(-x_moire_lims * nmtom, x_moire_lims * nmtom, 1024),
                           np.linspace(-x_moire_lims * nmtom, x_moire_lims * nmtom, 1024))  # * 10 ** -9  # nm

        Z = hbaromega + 2 * V_e * (
                    np.cos((-2 * pi / (math.sqrt(3) * moire_period)) * X - (2 * pi / moire_period) * Y - psi) +
                    np.cos((-2 * pi / (math.sqrt(3) * moire_period)) * X + (2 * pi / moire_period) * Y - psi) +
                    np.cos(((4 * pi) / (math.sqrt(3) * moire_period)) * X - psi))
        Z = Z / mevtoJ
        # # Plot Moire Potential 2D
        levels = np.linspace(Z.min(), Z.max(), 50)
        fig, ax = plt.subplots()
        plt.set_cmap('coolwarm')
        graph = ax.contourf(X, Y, Z, levels=levels)
        ax.set_title('Moire Potential (meV)')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.colorbar(graph)
        plt.show()
    elif units == 'hartreebohr':
        bohrconv = 5.2918e-11
        V_e = 0.0006614878
        moire_period = 359.048

        x_moire_lims = 1.2 * 10 ** 13
        X, Y = np.meshgrid(np.linspace(-x_moire_lims * bohrconv, x_moire_lims * bohrconv, 1024),
                           np.linspace(-x_moire_lims * bohrconv, x_moire_lims * bohrconv, 1024))  # * 10 ** -9  # nm

        # Z = hbaromega + 2 * V_e * (
        #             np.cos((-2 * pi / (math.sqrt(3) * moire_period)) * X - (2 * pi / moire_period) * Y - psi) +
        #             np.cos((-2 * pi / (math.sqrt(3) * moire_period)) * X + (2 * pi / moire_period) * Y - psi) +
        #             np.cos(((4 * pi) / (math.sqrt(3) * moire_period)) * X - psi))
        Z = hbaromega + 2 * V_e * (
                    np.cos((-2 * pi / (math.sqrt(3) * moire_period)) * Y - (2 * pi / moire_period) * X - psi) +
                    np.cos((-2 * pi / (math.sqrt(3) * moire_period)) * Y + (2 * pi / moire_period) * X - psi) +
                    np.cos(((4 * pi) / (math.sqrt(3) * moire_period)) * Y - psi))
        # # Plot Moire Potential 2D

        levels = np.linspace(Z.min(), Z.max(), 50)
        v = np.linspace(np.min(Z), np.max(Z),2, endpoint=True)
        fig, ax = plt.subplots()
        plt.set_cmap('coolwarm')
        graph = ax.contourf(X, Y, Z, levels=levels)
        # ax.plot([179.524], [0], 'wo', ms=7)
        # ax.plot([0, -269.286, 448.81, -179.524, -89.762, 538.572, 359.048, 493.691], [0, -155.4723446, -155.472344615, 0, 155.472344615, 155.472344615, 0, 0], 'ko', ms=5)   # plot points of lattice
        # ax.set_title('Moire Potential', fontsize=15)
        plt.xlabel('x [bohr]', fontsize=15)
        plt.xticks(fontsize=15)
        plt.ylabel('y [bohr]', fontsize=15)
        plt.yticks(fontsize=15)
        cbar =  plt.colorbar(graph, format=ticker.FuncFormatter(myfmt), ticks=v, orientation="horizontal")
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label(label='[Hartree]', size=15)
        plt.show()

def four_potential_slice(x_h, x_an, x_m, y, V_e, moire_period, pi, psi, lamb34, lamb26, lamb18, hbaromega):

    z = hbaromega + 2 * V_e * (np.cos((-2 * pi / (math.sqrt(3) * moire_period)) * x_m - (2 * pi / moire_period) * y - psi) +
                             np.cos((-2 * pi / (math.sqrt(3) * moire_period)) * x_m + (2 * pi / moire_period) * y - psi) +
                             np.cos(((4 * pi) / (math.sqrt(3) * moire_period)) * x_m - psi))

    z_harm = - 6 * V_e + ((16*np.pi**2*V_e) / (2*moire_period**2)) * x_h**2   # 108 is the minimum of z
    z_anharm34 = - 6 * V_e +  lamb34 * x_an**4
    z_anharm26 = - 6 * V_e + lamb26 * x_an ** 4
    z_anharm18 = - 6 * V_e + lamb18 * x_an ** 4

    # # Plot Moire Potential 2D
    plt.title('')
    plt.xlabel('position at x axis [Bohr]', fontsize=15)
    plt.ylabel('energy [Hartree]', fontsize=15)
    plt.plot(x_m, z, color="purple", label="Moire Potential")
    plt.plot(x_h, z_harm, color="C0", label="Harmonic $\hbar\omega$ = 26.73meV ")
    plt.plot(x_an, z_anharm34, color="C3", label="Anhramonic (34.73meV)")
    plt.plot(x_an, z_anharm26, color="C1", label="Anharmonic (26.73meV)")
    plt.plot(x_an, z_anharm18, color="C2", label="Anharmonic (18.73meV)")
    plt.title("Slice of Potentials at y = 0", fontsize=15)
    plt.legend(loc="lower left")
    plt.show()



def three_d_contour_potential(x_lim, lamb1, k, V, pi):
    X, Y = np.meshgrid(np.linspace(-x_lim, x_lim, 500), np.linspace(-x_lim, x_lim, 500))  # [nm]
    Z_anharm = -6 * V + lamb1 * (X ** 4 + Y ** 4) + 2 * lamb1 * (X ** 2 * Y ** 2)
    Z_harm = -6 * V + 0.5 * k * (X ** 2 + Y ** 2)
    # Z_moire = hbaromega + 2 * V * (np.cos((-2 * pi / (math.sqrt(3) * moire_period)) * X - (2 * pi / moire_period) * Y - psi) +
    #                          np.cos((-2 * pi / (math.sqrt(3) * moire_period)) * X + (2 * pi / moire_period) * Y - psi) +
    #                          np.cos(((4 * pi) / (math.sqrt(3) * moire_period)) * X - psi))

    fig3 = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z_anharm, 100, cmap='winter')  # Anharmonic (Blue)
    ax.contour3D(X, Y, Z_harm, 100, cmap='autumn')  # Harmonic (Red)
    # ax.contour3D(X, Y, Z_moire, 50, cmap='binary')  # Moire
    ax.set_zlim([-0.004, 0.001])
    ax.set_title('Blue - Anharmonic / Red - Harmonic', fontsize=15)
    ax.set_xlabel('x [bohr]', fontsize=15)
    ax.set_ylabel('y [bohr]', fontsize=15)
    plt.show()

#                                                                                # Sign Problem
def ana_2_bosons(beta, hbarw):
    '''
    :return: Analytical result for three Fermions in Canonical Ensemble <E> in harmonic potential
    '''
    exp = np.exp(beta * hbarw)
    exp2 = np.exp(2 * beta * hbarw)
    exp3 = np.exp(3 * beta * hbarw)
    exp4 = np.exp(4 * beta * hbarw)
    exp5 = np.exp(5 * beta * hbarw)
    exp6 = np.exp(6 * beta * hbarw)
    exp7 = np.exp(7 * beta * hbarw)
    exp8 = np.exp(8 * beta * hbarw)
    exp9 = np.exp(9 * beta * hbarw)
    exp10 = np.exp(10 * beta * hbarw)
    num = 2*hbarw*(exp4+exp3+4*exp2+1)
    denom = exp4-1
    # num = hbarw*(exp+exp2+2)
    # denom = exp2-1
    etot_2_b = num/denom
    return etot_2_b

def sgnprob_s1_extract_2b(filename, beta, cut):
    '''
    :param filename: pimdb.log file
    :param beta: beta in 1 / kJ/mol
    :param cut: he first "cut" values will be eliminated.
    :return: the average sign value
    '''
    # Here the pimdb.log file is in units of kJ/mol hence beta has to also be in same units
    df = pd.read_csv(filename, delimiter='\s+')
    df = df.iloc[cut:-1] # I added this and removed "cut from all [] example  e_1_1 = df.iloc[cut:, 0]
    e_1_1 = df.iloc[:, 0]
    e_2_2 = df.iloc[:, 1]
    e_2_1 = df.iloc[:, 2]

    s1 = e_2_2 - (e_2_1+e_1_1)

    return s1

def sgnprob_etot_b_2d_harmonic(number_of_files, path, beta):  # Bosons
    '''
    In the Boson Ensemble
    :param number_of_files: number of beads
    :param path: the file path
    :return: time_step, pot, avg_pot_exp, stdv_pot, trap, trapvir, newvir
    '''
    file = [path+'log.lammps.{}'.format(k) for k in range(0, number_of_files)]  # Makes a list of log.lammps.{}
    file_pimdb = path + 'pimdb.log'
    pot, trap, newvir, trapvir = 0, 0, 0, 0
    time_step = 0
    s1 = sgnprob_s1_extract_2b(file_pimdb, beta, cut_log)

    for i in range(0, number_of_files):
        try:
            potential, time_step, trap1, newvir1, trapvir1 = read_file_pot(file[i], cut_data, 76, 38)  # Harmonic (153,38) Auxiliary(167, 38)  sgnprob(145, 38)

        except:
            potential, time_step, trap1, newvir1, trapvir1 = read_file_pot(file[i], cut_data, 167, 38)
        pot += potential
        trap += trap1
        newvir += newvir1
        trapvir += trapvir1
    number_of_blocks, avg_pot_exp, stdv_pot = block_averaging(cutoff, block_size=900, data=pot)

    return time_step, pot*conv, avg_pot_exp, stdv_pot, trap*conv, trapvir*conv, newvir*conv, s1

def lambda_lammps(lamb, hw):
    hw_mev = hw / 1000
    e = -1.0
    hbar = 1
    m = 1 # mass of electron in amu
    w = hw_mev * conv_1 / conv
    k = m * w**2  # hartree / bohr^2
    kappa = np.sqrt(m*w) / (lamb * w)  # dielectric constant
    kappa_dip = (m*w)**(3/2) / (lamb * w)

    return k, kappa, 1/kappa, kappa_dip, 1/kappa_dip
print("hello")


def coul_pot(r, d):
    pot = d / r
    return pot


def read_lammps_table():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    path = '/home/netanelb2/Desktop/Netanel/Research/exiton/dornheim/table/'   # path to table
    file_name = 'coul_pot.table'   # name of table from lammps
    start = 0  # start point of talbe
    end = 10000000
    skp = 4  # skip first rows
    conv = 2625.499638  # [1 Hartree = 2625.499638 kJ/mol]    or the other way arround 3.8088E-4
    conv_1 = 96.4853  # eV to kJ/mol

    res = np.zeros(end - start)
    fopen = path + file_name
    f = pd.read_csv(fopen, sep='\s+', skiprows=skp, nrows=end + 1)
    f = f.iloc[start:end, :]

    ## HARTREE
    # hw_mev = 0.003  # meV
    # lamb = 3
    # e = -1
    # m = 1 # 0.00054858
    # w = hw_mev * conv_1 / conv
    # hbar = 1

    ## MKS
    hw_J = 4.80653e-22  # meV
    lamb = 3
    e = -1  # C
    m = 9.10938e-31  # 0.00054858
    hbar = 1.0545718e-34
    w = hw_J / hbar

    k1 = (e ** 2 * np.sqrt(m * w) / (hbar * w * np.sqrt(hbar))) * 1 / lamb
    k2 = 31.746329790939217
    print(k1, k2)

    r = np.linspace(0.1, 200, end)
    v = coul_pot(r, 1 / 31.7463297909)
    plt.plot(r, v, 'g', label="Analytical Result")
    plt.plot(f[str(end)], f['R'], '-o', color='blue', label='Harmonic Potential')
    # plt.plot(f['1000'], f['0.1'], '-o', color='red', label='Harmonic Potential')
    plt.ylabel(r'$ \gamma_sf $', fontsize=15)
    plt.xlabel(r'$ 1 / \beta\hbar\omega $', fontsize=15)
    plt.legend()
    plt.show()

    return None


def check_Area_single_particle():
    '''
    This function shows that a closed square of dimensions r has an area of 4
    :return:
    '''
    r = [[0, 0], [2, 0], [2, 2], [0, 2]]
    A = 0
    for atm in range(len(r)):
        print(r[atm])
        if atm == 3:
            A += np.cross(r[0], r[atm])
        else:
            A += np.cross(r[atm + 1], r[atm])
    A = A / 2
    return A