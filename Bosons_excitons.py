#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:12:40 2019
This code will allow analysing bosonic simulations.
@author: hirshb and updated by NBS
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from auxfunctions import CalcPhiEstimator, CalcPhiEstimator_from_PLUMED, CalcVirEstimator_from_PLUMED, \
    CalcUintEstimator_from_PLUMED, permutation_prob_3, calc_Wn,  permutation_prob_10, CalcmodeT
from functions_PIMD import make_block_average, block_averaging
correct = False


def read_saved_file(filename, cut, s_rows, s_footer):
    '''
    :param filename: input saved file file
    :param cut: The first "cut" values will be eliminated.
    :param s_rows: skiprows in log.lammps file since there is no data
    :param s_footer: skipfooter in log.lammps file since there is no data
    :return: potenital_energy, time_step, trap, newvir, trapvir
    '''
    df = pd.read_csv(filename, sep='\s+', engine='python', skiprows=s_rows, skipfooter=s_footer) # delimiter='\s+'
    EB = df['EB'][cut:]

    return EB


#Constants
kB = 0.0083144621 #Boltzmann const in kJ/mol/K
d = 2

#Constants that change
seeds = [1239451] # seeds = [98743501, 269451, 666472, 782943, 1239451]
gs = [0]  # Interaction between particles
perm_true = False  # If I want the permutation probability calculated TRUE
Natoms = 10
bhw_val = [60] # 0.31018876
step_start = 5000# line to start from colvar.
step_end = 200000# 300000  # 272745 #

Nbeads = [1, 16, 32, 64, 80, 96, 128]
Nbeads_ =  [1, 16, 32, 64, 80, 96, 128]
# Nbeads = [128]
# Nbeads_ =  [128]


skiprows_in_Phi = 199 #199#203 #199 #126 # 86 Dornheim # 134 moire_onewell_3    197 moire_onewell_10

#HARMONIC FORCE CONSTANT!
# omega = 0.003 #in eV
# omega = omega / 27.2114 #eV to Ha
# omega_kJmol = omega/3.8088E-4 #Ha to kJmol

#MOIRE FORCE CONSTANT!
omega = 0.02673  # [eV]
omega = omega/27.2114  # [Hartree]
omega_kJmol = omega/3.8088E-4  # [kJmol]

# bhw_energy_beta = np.array([1, 3, 4, 6, 10, 30, 60])
# beta_array = np.array([0.3877,  1.1632,  1.5510,  2.3264,  3.8774, 11.6322, 23.2644])
# bhw_val = [10, 30, 60]

#For BSE, Min block size, max block size, step and whether to plot convergence vs. block size or not.
Nblocks = 5
maxsize = int(np.floor((step_end-step_start)/Nblocks))
minsize = maxsize
step = 1  # dummy
plot = False
s = 1
count = 0
for g in gs:
    for ind, seed in enumerate(seeds):
        print("g: ", g, "seed: ", seed)
        for bhw in bhw_val:
            T = 1 / (kB*(bhw / omega_kJmol)) # [K]  FOR MOIRE POTENTIAL
            # T = (bhw * omega * 27.2114) / ((kB / 96.4853))   # for DORNHEIM SIMULATION # [K]
            beta = 1/(kB*T)   # kJ/mol/K
            if T > 45:     # for higher temperatures dt is smaller hence more iterations  45 Kelvin
                # step_end = 10*step_end
                step_end = step_end
                print("number of steps*10 because high temp (more itr lower dt)")
            if T < 15:
                Nbeads = Nbeads_

            Benergy = np.zeros(len(Nbeads))  # To zero the energy for the next temp loop
            save_data = pd.DataFrame(columns=['Nbeads', 'g', 'seed', 'bhw', 'EB', 'l', 'p_l'])
            for i, p in enumerate(Nbeads):
                # if p == 16:
                #     step_end = 4213583
                # else:
                #     step_end = 1958664
                path = '/hirshblab-storage/netanelb2/PIMD/exitons/harmonic/boson10/noninter/bhw60/sim1/{}beads/'.format(p)
                # path = '/home/netanelb2/Desktop/Netanel/Research/exiton/moire_10_1/sim1_bhw60_m/{}beads/'.format(p)
                path_ = '/hirshblab-storage/netanelb2/PIMD/exitons/harmonic/boson10/noninter/bhw60/'

                print("path: ", path, "Temp [K]:", round(T, 3), "beta [kJ/mol/K]:", round(beta), "omega_kjmol: [kJ/mol]", round(omega_kJmol, 3))

                #Get Pot energy
                Trap = CalcPhiEstimator_from_PLUMED(path, 'log.lammps', p, step_start, step_end, skip=skiprows_in_Phi, potlabel='c_trap')
                # NBS: I obtain Trap when I add all the TRAP from all logfiles.
                Phi = CalcPhiEstimator_from_PLUMED(path, 'log.lammps', p, step_start, step_end, skip=skiprows_in_Phi) # Pot Estimator
                # Get kin energy
                Vir = CalcVirEstimator_from_PLUMED(path, 'log.lammps', p, step_start, step_end, skip=skiprows_in_Phi) # Kinetic Estimator

                # try:
                #     modeT = CalcmodeT(path, 'log.lammps', p, step_start, step_end, skiprows_in_Phi, 'v_modeT')  # calculates Average Temperature
                #     modeT = np.mean(modeT)
                # except:
                #     continue

                Trap = Trap / 3.8088E-4  # Ha to kJmol
                Trap = Trap / p   # Needs to be divided by number of beads
                Phi = Phi / 3.8088E-4  # Ha to kJmol   # IT IS ALLREADY DIVIDED
                Vir = Vir / 3.8088E-4  # Ha to kJmol

                #Get RP energies
                fname = "pimdb.log"

                try:
                    data = pd.read_csv(path + fname, sep='\s+', dtype='float64', header=None)
                except:
                    print("IOError: Problem reading file " + path+fname)
                    raise
                if perm_true == True:
                    if p == max(Nbeads):  # Permutation Probability (Condensation)
                        if Natoms == 3:
                            l, p_l = permutation_prob_3(path + fname, beta, step_start, Natoms)
                        elif Natoms == 10:
                            l, p_l = permutation_prob_10(path+fname, beta, step_start, Natoms)
                    else:
                        l, p_l = 0, 0
                else:
                    l, p_l = 0, 0

                # This below create title for pimdb.log file ("data")
                all_labels_e = []
                save_label_n = []
                for n in range(1, Natoms+1):
                    save_label_e = []
                    for k in range(n, 0, -1):
                        lis = [ '{:02d}'.format(int(x)) for x in range(n-k+1,n+1)]
                        label=''.join(map(str, lis))
                        #        print(label)
                        if k==1:
                            index = str(lis[0])
                        else:
                            index = str(lis[0]) + '-' + str(lis[-1])

                        save_label_e.append(label)
                        all_labels_e.append(label)
                    save_label_n.append(str(n))
                data.columns = ['E' + x for x in all_labels_e] + ['VB' + str(x) for x in range(Natoms+1)] # gives name E01    E0102      E02   E010203  ...  VB0       VB1      VB2      VB3

                data = data.iloc[step_start:step_end] # Reads the pimdb.log file from step_start untill step_end

                vir_EB = (Phi + Vir)
                EB3 = np.mean(vir_EB)  # [kJ/mol]
                print('Boson Energy: ' + str(EB3), "kJ/mol")
                save_data.loc[count, 'Nbeads'] = p
                save_data.loc[count, 'EB'] = EB3
                save_data.loc[count,'seed'] = seed
                save_data.loc[count,'g'] = g
                save_data.loc[count,'bhw'] = bhw
                save_data.loc[count, 'l'] = l
                save_data.loc[count, 'p_l'] = p_l
                # save_data.loc[count, 'Temp'] = modeT


                # convert from kJ/mol to meV
                conv_1 = 96.4853  # kJ/mol to eV
                #Benergy = np.array(list(save_data['EB'])) * 1000 / conv_1  # meV
                Benergy = EB3 * 1000 / conv_1  / Natoms # meV                        ## ENERGY PER PARTICLE!!
                save_data.loc[count, 'EB'] = Benergy  # meV

                count += 1
#################################### PLOTS ###############################################
            array_energies = np.array(save_data['EB'])
            r_beta = round(beta)  # beta rounded
            np.savetxt(path_+'Figures/output_bosons{}bhw{}'.format(Natoms, np.round(bhw)), save_data, fmt="%s", header='Nbeads g seed bhw EB l p_l Temp', comments="")
            path_save_plot_betaEB = path_+'Figures/betaEB_bosons{}bhw{}Nbeads{}.png'.format(Natoms,  np.round(bhw), Nbeads[-1])
            # path_save_plot_cond = path_+'Figures/cond_bosons{}bhw{}Nbeads{}.png'.format(Natoms, bhw, Nbeads[-1])

            array_energies = [0.83958,12.125895,  18.821877,  23.3617015,24.144584,24.61398,  25.12197]

            fig = plt.figure(figsize=(10, 7))
            tit = "Beta{} - Energy vs Beads".format(r_beta)
            # plt.axhline(y=3, color='r', linestyle='-', label="Convergence") # 3.36
            # plt.axvline(x = 12, color='r', linestyle='-')
            plt.plot(Nbeads, array_energies, 'o', color="blue")
            plt.title(tit)
            plt.xlabel("Number of Beads")
            plt.ylabel("Boson Energy [meV] ")
            plt.legend(loc='lower right')
            plt.savefig(path_save_plot_betaEB)
            plt.show()

            # Plot for Condensation

            # fig1 = plt.figure(figsize=(10, 7))
            # plt.bar(l, p_l / (1 / Natoms))
            # plt.xlabel('Permutation Length', fontsize=20)
            # plt.ylabel('Normalized Permutation Probability', fontsize=20)
            # # plt.ylim(0.99, 1.01)
            # plt.savefig(path_save_plot_cond)
            # plt.show()

            print("All data:", save_data)  # ['g', 'seed', 'bhw', 'EB', 'l', 'p_l']            print("Beads: ", Nbeads, "Boson Energy: [meV]", Benergy)

        # meanB = np.mean(save_data['EB'])
        # BSEB = np.std(save_data['EB'])/np.sqrt(len(save_data['EB']))

## Mean of j independent simulations (read saved files)
path = '/home/netanelb2/Desktop/Netanel/Research/exiton/moire_10_1/conv_sim_bhw031_m/Figures/graph100M/'
eb_list = []
mean_eb = []
std_eb = []
for i in range (2,6):
    eb = read_saved_file(path + 'sim{}.0'.format(i), 0, 0, 0)
    eb_list.append(eb)
for j in range(len(eb_list[0])):
    eb_vals = []
    for i in range(len(eb_list)):
        eb_vals.append(eb_list[i][j])
    mean = np.mean(eb_vals)
    std = np.std(eb_vals)
    mean_eb.append(mean)
    std_eb.append(std)

fig = plt.figure(figsize=(10, 7))
plt.errorbar(Nbeads, mean_eb, yerr=std_eb, fmt="o", ecolor="black")
plt.ylabel('Energy [meV]', fontsize=16)
plt.xlabel('beads [#]', fontsize=16)
plt.title('Moire MoSe2/WSe2 at 1000K',fontsize=16)
plt.show()



# Results
bhw = [0.75, 1, 1.5, 2, 3]
Boson_Energy_N10_harmonic = [18.64948663, 13.02964836, 11.18361578, 10.4824691, 10.10549314]

fig1 = plt.figure(figsize=(10, 7))
# tit = "Permutation Probability - Temp {} K".format(77)
plt.bar(l, pl6 / (1 / 10))
plt.title(tit,  fontsize=20)
plt.xlabel('Permutation Length', fontsize=20)
plt.ylabel('Normalized Permutation Probability', fontsize=20)
plt.ylim(0.5, 1.01)
plt.show()

fig = plt.figure(figsize=(10, 7))
plt.plot(Nbeads, array_energies, 'o', color="blue")
plt.title(tit)
plt.xlabel("Number of Beads")
plt.ylabel("Boson Energy [meV] ")
plt.legend(loc='lower right')
plt.show()