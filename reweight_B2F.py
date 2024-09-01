#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:12:40 2019

@author: hirshb
"""
def getBSE(data, w, step_start, step_end, columnname = 'Phi', Nblocks_list=[5]):

    for Nblocks in Nblocks_list:
    
        delta = int((step_end - step_start)/Nblocks)
        block_res = pd.DataFrame(index=range(Nblocks),columns = [columnname])
        block_res = block_res.fillna(np.nan) #(0) # with 0s rather than NaNs
        init_step = step_start
        Wj = np.zeros(Nblocks)        
        Aj = np.zeros(Nblocks)
        for block in range(Nblocks):
            block_data = data[init_step:(init_step+delta)] 
            block_w = w[init_step:(init_step+delta)] 
            init_step = init_step + delta
            
            block_res.loc[block,columnname] = np.average(block_data*block_w)
#            Wj[block] = (np.sum(block_w))**2/np.sum(block_w**2)
            Wj[block] = np.sum(block_w)
            Aj[block] = np.sum(block_data*block_w)/np.sum(block_w)
            
        meanA = np.sum(Wj*Aj)/np.sum(Wj)
        V2 = np.sum(Wj**2)
        V1 = np.sum(Wj)
#        BSEA = np.sqrt( np.sum( np.multiply(Wj,(Aj-meanA)**2) )/(V1 - V2/V1) )/np.sqrt(Nblocks)
        Neff = np.sum(Wj)**2/np.sum(Wj**2)
        print("Nblocks is: " + str(Nblocks) + " and Neff is: " + str(Neff))
        # varA = np.sqrt( Neff*np.sum( save_data['Wj']*(save_data['EF']-meanA)**2 )/np.sum(save_data['Wj'])/np.sqrt(Neff-1) )
        varA = np.sqrt(Neff * np.sum(save_data['Wj'] * (save_data['EF'] - meanA) ** 2) / np.sum(save_data['Wj']) / np.sqrt(Neff))
        BSEA = varA/np.sqrt(Nblocks) #/np.sqrt(Nblocks)
        
        
#        BSEA = np.sqrt( np.sum( np.multiply(Wj,(Aj-meanA)**2) )/V1 )/np.sqrt(Nblocks)             
#        mean = block_res[columnname].mean()
#        BSE =  block_res[columnname].std(ddof=1)/np.sqrt(Nblocks)
        
    return (meanA,BSEA)

def calc_Wn(beta, data, n, save_label_e ,label='B'):
    import numpy as np
    ################################
    #PRINT Wn FOR BOSON OR FERMIONS#
    ################################            
    ib = 1/beta

    #Define function string. If I am running Bosons I am already calculating dlnWB/dbeta
    if label == 'B':
        save_label_n_new = ["+data['VB"+str(x)+"']" for x in range(1,n)]
        save_label_n_new = [''] + save_label_n_new
        sign = [1.0 for k in range(1,n+1)]
        func_string =  [str(z) + "*np.exp(-beta*(data['E" + x + "']" + y +"))" for x,y,z in zip(save_label_e, save_label_n_new, sign)]
        res = eval("1/n*(" + "+".join(func_string) + ")")  
    
    elif label == 'F':
        save_label_n_new = ["*data_F['WF"+str(x)+"']" for x in range(1,n)]
        save_label_n_new = [''] + save_label_n_new
        sign = [pow(-1.0,k-1) for k in range(n,0,-1)]
        func_string =  [str(z) + "*np.exp(-beta*(data['E" + x + "']))" + y for x,y,z in zip(save_label_e, save_label_n_new, sign)]      
        res = eval("1/n*(" + "+".join(func_string) + ")")  
    else:
        raise('IOError: Please use labels F or B')     

#    res = eval("data['E"+ save_label_e[-1] + "'] - ib*np.log(1/" + str(n) + "*(" + "+".join(func_string) + "))")  
    return res

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle 
from auxfunctions import ReadXYZofAllBeads
from auxfunctions import GetBSE
from auxfunctions import CalcPhiEstimator, CalcPhiEstimator_from_PLUMED, CalcVirEstimator_from_PLUMED, CalcUintEstimator_from_PLUMED
import math
import itertools
correct = False

Natoms = 3
Nbeads = 64 #[16,24,36,48,72,96]
gs = [0]
lam = 0.0
Msteps = 30

# seeds = [98743501, 269451, 666472, 782943, 1239451]
seeds = [1239451]

#User defined parameters
kB = 0.0083144621 #Boltzmann const in kJ/mol/K
bhw_val = [10] #[1.25, 1.5, 1.75, 2, 2.5]
temp_val = [31.019]
d = 2


# DIS BOSON FERMI
step_start = 10000 #line to start from colvar.
# step_end = 750000 #Final line from colvar
step_end = 300000

# step_start = 0 #line to start from colvar.
# step_end = 162 #Final line from colvar

skiprows_in_Phi = 134    #  Harmonic 153   /   Aux 167  # extion3boson 153

#HARMONIC FORCE CONSTANT!
# omega = 0.003 #in eV
# omega = omega/27.2114 #eV to Ha
# omega_kJmol = omega/3.8088E-4 #Ha to kJmol

#MOIRE FORCE CONSTANT!
omega = 0.02673 #in eV
omega = omega/27.2114 #eV to Ha
omega_kJmol = omega/3.8088E-4 #Ha to kJmol


#For BSE, Min block size, max block size, step and whether to plot convergence vs. block size or not.
Nblocks = 5
maxsize = int(np.floor((step_end-step_start)/Nblocks))
minsize = maxsize
step = 1 #dummy
plot = False

save_data = pd.DataFrame(columns = ['g','seed','EF', 'Err_EF', 'EB', 'Err_EB', 'sign', 'Err_sign', 'neff_reW'])
count=0
for g in gs:
    for ind, seed in enumerate(seeds):
        print(g, seed)
        
        for bhw in bhw_val:
            T = 31.019   # K
            beta = 1/(kB*T)   # kJ/mol/K
            # path_to_save = "/home/netanelb2/Desktop/Netanel/Research/PIMD/runs/fermions/five2/bhw1p25/"
            # path_to_save = '/home/netanelb2/Desktop/Netanel/Research/PIMD/runs/fermions/gauss/bogo/bhw6/'
            # path = '/home/netanelb2/Desktop/Netanel/Research/exiton/moire/3bosons/bhw30/32beads_1every_bh/'
            path = "/home/netanelb2/Desktop/Netanel/Research/exiton/moire_one_3p/boson3/bhw10/64beads/"
            # path = path_to_save + "/sim" + str(ind + 1) + "/"
            print("path: ", path, "temp: [K] ", T, "beta: [kJ/mol/K]", beta, "omega_kjmol: " , omega_kJmol)
            print("all energies are in kJ/mol")


            #Get Pot energy
            Trap = CalcPhiEstimator_from_PLUMED(path, '/log.lammps', Nbeads, step_start, step_end, skip=skiprows_in_Phi, potlabel='c_trap')
            # N: I obtain Trap when I add all the TRAP from all logfiles and divide by number of beads.
            Trap = Trap / 3.8088E-4 #Ha to kJmol
                
            Phi = CalcPhiEstimator_from_PLUMED(path, '/log.lammps',Nbeads, step_start, step_end, skip=skiprows_in_Phi) # Pot Estimator
            Vir = CalcVirEstimator_from_PLUMED(path, '/log.lammps',Nbeads, step_start, step_end, skip=skiprows_in_Phi) # Kinetic Estimator
            
            Phi = Phi / 3.8088E-4 #Ha to kJmol
            Vir = Vir / 3.8088E-4 #Ha to kJmol
           
            #Get RP energies
            fname = "pimdb.log"
            
            try:
                data = pd.read_csv(path + fname, sep='\s+', dtype='float64', header=None)
            except:
                print("IOError: Problem reading file " + path+fname)
                raise

            all_labels_e = []
            save_label_n = []
            for n in range(1, Natoms+1):
                save_label_e=[]
                #for k in range(1,n+1):
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

            data = data.iloc[step_start:step_end] # Reads the pimdb.log file from 10,001 until 750000

            
            #WF has to be calculated, not printed in LAMMPS since I am running Bosons
            Nenergies = int(Natoms*(Natoms+1)/2)
            data_F = data.iloc[:, 0:Nenergies];
            data_F = pd.concat([data_F, pd.DataFrame(1.0,index=np.arange(step_start,step_end), columns=['WF0'])], axis=1)  # Adds WF0 as 1 for the entire column at the end
                 
            for n in range(1, Natoms+1):
                save_label_e= []
                #for k in range(1,n+1):
                for k in range(n, 0, -1):
                    lis = [ '{:02d}'.format(int(x)) for x in range(n-k+1, n+1)]
                    label = ''.join(map(str, lis))
                    #        print(label)
                    if k==1:
                        index = str(lis[0])
                    else:
                        index = str(lis[0]) + '-' + str(lis[-1])
                
                    save_label_e.append(label)

                tmp = calc_Wn(beta, data_F, n, save_label_e, 'F')
                #this is required to give an expression as column name, done through dictionary
                col={"WF"+str(n): tmp.values}
                data_F = data_F.assign(**col)
                    
            #WB is simply obtained from LAMMPS
            # N: WB2 is e^(-beta * vN) : W(3)_Bosons
            WB2 = np.exp(-beta*data['VB' + str(Natoms)])    
        
            WF2 = data_F['WF' + str(Natoms)]
            
            neff_qsym = np.sum(WF2/WB2)**2/np.sum((WF2**2)/(WB2**2))
            print ('neff just for qsym: ' + str(neff_qsym) )
            save_data.loc[count,'neff_reW'] = neff_qsym
            save_data.loc[count,'seed'] = seed
            save_data.loc[count,'g'] = g
            save_data.loc[count,'bhw'] = bhw
            save_data.loc[count,'Wj'] = np.sum(WF2/WB2)
           
            if (g!=0 or correct):    

                vir_EF = (Phi + Vir)*WF2/WB2
                meansgn = np.mean(WF2/WB2)
                # EF2 = np.mean(vir_EF)/meansgn/omega_kJmol
                EF2 = np.mean(vir_EF) / meansgn
                print('int. F Ener: ' +str(EF2) )       
                save_data.loc[count,'EF'] = EF2
                
                (mean_EF2, BSE_EF2) = getBSE(Phi + Vir, WF2/WB2,  step_start = 1, step_end = len(Phi))        
                (meanw3, BSE_w3) = getBSE(WF2/WB2, np.ones(len(Phi)), step_start = 1, step_end = len(Phi))
                ERR = np.sqrt((BSE_w3/meanw3)**2 + (BSE_EF2/mean_EF2)**2)*EF2
                print('Err int. F is: ' + str(ERR))
                save_data.loc[count,'Err_EF'] = ERR
                
                vir_EF_corr = (Trap + Vir)*WF2/WB2
                # EF3 = np.mean(vir_EF_corr)/meansgn/omega_kJmol
                EF3 = np.mean(vir_EF_corr) / meansgn
                print('corrected int. F Ener: ' +str(EF3) )       
                save_data.loc[count,'EF_corr'] = EF3
                (mean_EF3, BSE_EF3) = getBSE(Trap + Vir, WF2/WB2,  step_start = 1, step_end = len(Phi))        
                ERR2 = np.sqrt((BSE_w3/meanw3)**2 + (BSE_EF3/mean_EF3)**2)*EF2
                print('Err corrected int. F is: ' + str(ERR2))
                save_data.loc[count,'Err_EF_corr'] = ERR2
                
                print('mean sign is: ' + str(meanw3) + ' error: ' + str(BSE_w3))
                save_data.loc[count,'sign'] = meanw3
                save_data.loc[count,'Err_sign'] = BSE_w3
                save_data.loc[count,'Wj'] = np.sum(WF2/WB2) 
                ##vir_EF2 = (Trap + TrapVir)*WF2/WB2*np.exp(beta*UB)
                ##meansgn = np.mean(WF2/WB2*np.exp(beta*UB))
        #        print('av meansgn is: ' + str(meansgn))
                ##EF3 = np.mean(vir_EF2)/meansgn/omega_kJmol
                ##print('nonint. F Ener: ' + str(EF3))           
                #    EB2_reW = np.mean(vir_EB*np.exp(beta*UB))/np.mean(np.exp(beta*UB))/omega_kJmol
                #    print(EB2_reW)
                
                vir_EB = (Phi + Vir)
                # EB3 = np.mean(vir_EB)/omega_kJmol
                EB3 = np.mean(vir_EB)
                print('int. B Ener: ' + str(EB3) ) 
                save_data.loc[count,'EB'] = EB3
                (mean_EB3, BSE_EB3) = getBSE(Phi + Vir, np.ones(len(Phi)),  step_start = 1, step_end = len(Phi))        
                print('Err int. B is: ' + str(BSE_EB3))
                save_data.loc[count,'Err_EB'] = BSE_EB3
            
                ##vir_EB2 = (Trap + TrapVir)*np.exp(beta*UB)
                ##EB2_reW = np.mean(vir_EB2)/np.mean(np.exp(beta*UB))/omega_kJmol
                ##print('nonint. B Ener: ' + str(EB2_reW) )
            else:
                vir_EF = (Phi + Vir)*WF2/WB2
                meansgn = np.mean(WF2/WB2)
                # EF2 = np.mean(vir_EF)/meansgn/omega_kJmol
                EF2 = np.mean(vir_EF) / meansgn
        #        print('av meansgn is: ' + str(meansgn))
        #         print('nonint. F Ener: ' + str(EF2) )
                save_data.loc[count,'EF'] = EF2
                
                (mean_EF2, BSE_EF2) = getBSE(Phi + Vir, WF2/WB2,  step_start = 1, step_end = len(Phi))        
                (meanw3, BSE_w3) = getBSE(WF2/WB2, np.ones(len(Phi)), step_start = 1, step_end = len(Phi))
                ERR = np.sqrt((BSE_w3/meanw3)**2 + (BSE_EF2/mean_EF2)**2)*EF2
                # print('Err nonint. F is: ' + str(ERR))
                save_data.loc[count,'Err_EF'] = ERR
                # print('mean sign is: ' + str(meanw3) + ' error: ' + str(BSE_w3))
                save_data.loc[count,'sign'] = meanw3
                save_data.loc[count,'Err_sign'] = BSE_w3
                
                vir_EB = (Phi + Vir)
                # EB3 = np.mean(vir_EB)/omega_kJmol
                EB3 = np.mean(vir_EB)
                print('nonint. B Ener (BOSON ENERGY): ' + str(EB3) )
                save_data.loc[count,'EB'] = EB3
                (mean_EB3, BSE_EB3) = getBSE(Phi + Vir, np.ones(len(Phi)),  step_start = 1, step_end = len(Phi))        
    #            (meanw3, BSE_w3) = getBSE(WF2/WB2, np.ones(len(Phi)), step_start = 1, step_end = len(Phi))
    #            ERR = np.sqrt((BSE_w3/meanw3)**2 + (BSE_EF2/mean_EF2)**2)*EF2
                print('Err nonint. B is: ' + str(BSE_EB3))
                save_data.loc[count,'Err_EB'] = BSE_EB3
            
            count+=1
                
            #WI = WF2
            #Wr = WB2*np.exp(beta*UB)           
            
        
#meanE = np.sum(np.multiply(save_neff,save_EF))/np.sum(save_neff)
#stdE = np.sqrt(len(save_EF)*np.sum(np.multiply(save_neff,np.multiply(save_EF-meanE,save_EF-meanE)))/np.sum(save_neff)/(len(save_EF)-1))
    
#print(meanE)
#print(stdE)
meanA = np.sum(save_data['Wj'] * save_data['EF'])/np.sum(save_data['Wj'])
Neff = np.sum(save_data['Wj'])**2/np.sum(save_data['Wj']**2)
# varA = np.sqrt( Neff*np.sum( save_data['Wj']*(save_data['EF']-meanA)**2 )/np.sum(save_data['Wj'])/np.sqrt(Neff-1) )
varA = np.sqrt( Neff*np.sum( save_data['Wj']*(save_data['EF']-meanA)**2 )/np.sum(save_data['Wj'])/np.sqrt(Neff))
BSEA = varA/np.sqrt(Neff)

# if (g!=0 or correct==True):
#    meanA = np.sum(save_data['Wj'] * save_data['EF_corr'])/np.sum(save_data['Wj'])
#    Neff = np.sum(save_data['Wj'])**2/np.sum(save_data['Wj']**2)
#    varA = np.sqrt( Neff*np.sum( save_data['Wj']*(save_data['EF_corr']-meanA)**2 )/np.sum(save_data['Wj'])/np.sqrt(Neff-1) )
#    BSEA = varA/np.sqrt(Neff)

meanB = np.mean(save_data['EB'])
BSEB = np.std(save_data['EB'])/np.sqrt(len(save_data['EB']))

# save_data.to_csv(path_to_save+"dat", index=False)

# if(g!=0):
#     meanA_corr = np.sum(save_data['Wj'] * save_data['EF_corr'])/np.sum(save_data['Wj'])
#     varA_corr = np.sqrt( Neff*np.sum( save_data['Wj']*(save_data['EF_corr']-meanA_corr)**2 )/np.sum(save_data['Wj'])/np.sqrt(Neff-1) )
#     BSEA_corr = varA_corr/np.sqrt(Neff)


        
