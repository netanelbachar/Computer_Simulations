from functions_PIMD import *
start = time.time()

# Harmonic Siumlation Constants
# kB = 0.0083144621 #Boltzmann const in kJ/mol/K     # Barak
# T = 34.8/bhw  # (bhw = 1.25)                       # Barak
# beta = 1/(kB*T)                                    # Barak
bhw = [1.25, 1.5, 1.75, 2.0, 2.5]

beta_list_5_v2 = [4.320127964728478, 5.184153557674173, 6.048179150619869, 6.912204743565564, 8.640255929456956]  # [1 / kJ/mol ]
temp = [27.8509, 23.2091, 19.8935, 17.4068, 13.9254]  # [K]
# Auxiliary and Bogoliubov Simulation Constants
bhw_g = [1, 3, 4, 5, 6]
temp_g = [34.813, 11.6, 8.7, 6.96, 5.8]  # [K]
beta_list_g = [3.454757702, 10.368307115348346, 13.824409487131128, 17.28051185891391, 20.73661423069669]


# Path of Folder and how many Beads (number of files) (For Harmonic num_of_files is 12 for Auxiliary is 72)
number_of_files = 72
# path = '/home/netanelb2/Desktop/Netanel/Research/PIMD/runs/fermions/five2/bhw1p25/sim5/'  # Harmonic
# path = '/home/netanelb2/Desktop/Netanel/Research/PIMD/runs/fermions/gauss/bogo/bhw6/sim5/'     # Auxiliary
# path = '/home/netanelb2/Desktop/Netanel/Research/PIMD/runs/bosons/harmonic_boson_2p/harmonic_boson_2p/bhw6/'
path = "/home/netanelb2/Desktop/Netanel/Research/exiton/test_lammps/test_boson_lammps_b_inter/bhw3/"
file_pimdb = path + 'pimdb.log'
beta = 3

#######################################################################################################################
energy_lammps = np.array([0.00044390700144266294, 0.00023494208877656474, 0.00022542832309916554, 0.00022278789487966625,  0.00022122040685063573])
energy_lammps_b = np.array([0.00044546168466307035, 0.00023353297503444983,  0.0002249183586180965, 0.00022288639594252382, 0.00022081155604605047])
#                                 ###      3 Fermions under Harmonic potential run         ###

# time_step,  avg_etot_f_h, sign_array, wj = etot_f_2d_harmonic(number_of_files, path, beta)
#
#
# print("<S>_B: ", np.mean(sign_array))  # [Kj/mol]
# print("W_j: ", wj)  # [Kj/mol]
#  #division here by hw_2 is only for graph purposes [Kj/mol]
# print(" Harmonic - Re-weighted Energy Fermions: ", avg_etot_f_h / hw_2)

# #                               ###     3 Fermions under Auxiliary potential run          ###

# Re-weighted results: <(pot_est+kin_est)*s> / <s>  = <(pot/P + newvir)*s>_B / <s>_B
# Bogoliubov Results: E_H' - <pot - harmonic(trap)>_H' = E_H' - <V_g>_H'>  >=  E_H

# time_step, sign_array, wj, avg_etot_f_a, avg_etot_f_b = etot_f_2d_harmonic_aux(number_of_files, path, beta)
# print("<S>_B: ", np.mean(sign_array))  # [Kj/mol]
# print("W_j: ", wj)  # [Kj/mol]
# print("Fer_Energy of Auxiliary (Re-weighted): ", avg_etot_f_a / hw_2)
# print("Auxiliary - Bogoliubov: ",  avg_etot_f_b / (number_of_files * hw_2))
# print("Fer_Energy of Bogoliubov (Re-weighted):", (avg_etot_f_a - (avg_etot_f_b / number_of_files)) / hw_2)  # E_H' - <V_g>_H'


#######################################################################################################################
# stop = time.time()
# duration = stop - start
# print("Time of execution:", duration)
#
#
# sign_avg_5_v2 = np.array([0.15955686287327467, 0.08724413765519949, 0.05048910486412432, 0.029095192602032777, 0.007946735077753436])
# etot_f_array_5_v2 = np.array([6.901289992143137, 6.383315530057243, 6.001863829786661, 5.711517982801404, 5.746343105765936])
# stdv_f_array_5_v2 = np.array([ 0.02798230127157498, 0.029509423362973766, 0.06765394958776051, 0.20068033218394465, 0.5243678741776379])
#
# sign_avg_a_b = np.array([0.2695437789672403, 0.13774697226402477, 0.07549642086584465, 0.0607452669175924])
# etot_f_array_a = np.array([5.701217136142354, 5.54200529256501, 5.513619834831573, 5.321186534155975])
# stdv_f_array_a = np.array([0.015204454921370483, 0.030931047708318646, 0.05903119391470891, 0.06799140877377802])
# etot_f_array_b = np.array([5.429256845174638, 5.296103954378438, 5.302963455074144, 4.999979651410427])
# stdv_f_array_b = np.array([0.024631012170872546, 0.05529193843597514, 0.10986761924841078, 0.10866600443487313])
#
# etot_f_array_5_barak = [6.93755744, 6.4148657, 6.074709, 5.9004776433, 5.6297522]
# stdv_f_array_5_barak = [0.01030199, 0.0094601, 0.05510046, 0.083725350, 0.40240035]
# sign_f_array_5_barak = [0.15715344, 0.08891504, 0.047802733, 0.026490030, 0.00804471243]
# etot_f_array_g_a_barak = [5.723669675071802, 5.48707494, 5.4201400039, 5.32728000]
# stdv_f_array_g_a_barak = [0.02653834, 0.0575122, 0.086961, 0.013275099]
# etot_f_array_g_b_barak = [ 5.459570480041557, 5.202135535081834, 5.124542912530602, 5.010820551140615]
# stdv_f_array_g_b_barak = [0.040907328079425515, 0.08633141649033521, 0.15185797821596392, 0.026682192651536943]
# sign_f_array_g_a_barak = [0.26612186003, 0.1506680643, 0.09444886968, 0.0521631974]
#
# fig, (ax1, ax2) = plt.subplots(2)
# q = np.linspace(0.8, 11, 1000)
# p = ana_3_fermions(q, 1)
# ax1.plot(q, p, 'g',  label="Analytical Result")
# ax1.errorbar(bhw, etot_f_array_5_v2, yerr=stdv_f_array_5_v2, fmt='^', color='orange', label='H_original')
# ax1.errorbar(bhw_g, etot_f_array_a, yerr=stdv_f_array_a, fmt='H', color='grey', label='H_aux')
# ax1.errorbar(bhw, etot_f_array_5_barak, yerr=stdv_f_array_5_barak, fmt='x', color='green', label='Baraks Harmonic and Aux results')
# ax1.errorbar(bhw_g, etot_f_array_b, yerr=stdv_f_array_b, fmt='X', color='purple', label='Bogoliubov')
# ax1.errorbar(bhw_g, etot_f_array_g_a_barak, yerr=stdv_f_array_g_a_barak, fmt='x', color='green')
# ax1.errorbar(bhw_g, etot_f_array_g_b_barak, yerr=stdv_f_array_g_b_barak, fmt='*', color='red', label='Baraks Bogoliubov results')
# ax1.set_ylabel('<E>_F')
# ax1.set_ylim([1, 8])
# ax1.set_xlim([1, 6.2])
# ax2.plot(bhw, sign_avg_5_v2, 'D', color='red')
# ax2.plot(bhw, sign_f_array_5_barak, 'D', color='green')
# ax2.plot(bhw_g, sign_avg_a_b, 'D',  color='red', label='<s>_B')
# ax2.plot(bhw_g, sign_f_array_g_a_barak, 'D',  color='green', label='Baraks - <s>_B')
# ax2.set_ylabel('<S>_B')
# ax2.set_xlabel('bhw')
# ax2.set_ylim([-0.01, 0.28])
# ax2.set_xlim([1, 6.2])
# ax1.legend()
# ax2.legend()
# plt.show()










#                                                         ### Bosons run ###

time_step, pot, avg_pot_exp, stdv_pot, trap, trapvir, newvir = etot_b_2d_harmonic(number_of_files, path)

print("Total Energy Bosons / P  [Hartree]: ", (2 * avg_pot_exp / (number_of_files)))  # [Hartree]
print("Total Energy Bosons / P /hw [meV]: ", (2 * avg_pot_exp / (number_of_files * hw)))  # [meV]

###RESULTS -comparason between lammps and lammps_b #####
bhws = [1, 3, 4, 5, 6]
energy_lammps_try1 = np.array([0.0004490309773257717, 0.00023428226587147733, 0.00022562850200136275, 0.0002235215784876358, 0.00022267794487525758])
energy_lammps_b_try1 = np.array([0.00045381136539992514,  0.00023418656303328368,  0.0002242815188440773,  0.00022211770255890827, 0.00022093636434442022])
energy_lammps = np.array([0.00044390700144266294, 0.00023494208877656474, 0.00022542832309916554, 0.00022278789487966625,  0.00022122040685063573])
energy_lammps_b = np.array([0.00044546168466307035, 0.00023353297503444983,  0.0002249183586180965, 0.00022288639594252382, 0.00022081155604605047])

###RESULTS -comparason between lammps and lammps_b #####
                                                         # Results #
# Results from Barak's artilce:
# Natoms = 3
# bhws = [1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5, 6]
# ##bhws = [3, 4, 5, 6]
# analytic = [6.89333779, 6.36220084, 6.01473798, 5.77315087, 5.46533286, 5.28532944, 5.10755903, 5.04008246, 5.01482427]
# g,#seed#,EF,Err_EF,EB,Err_EB,sign,Err_sign,neff_reW,bhw,Wj,EF_corr,Err_EF_corr
# 0.0,#98743501.0#,5.385968571985375,2.1132931542097952,3.216831456368679,0.0008102108362961339,0.009227708151542006,0.0034864803536090398,94.60365877312594,2.5,6002.020729958988,5.385968571985375,2.1132931542097952
# 0.0,#269451.0#,5.5266261470974625,1.4976033641370217,3.2127540136006094,0.001769823082885048,0.00868948612488037,0.0023017135189826454,83.85152951973014,2.5,5643.323090707206,5.5266261470974625,1.4976033641370217
# 0.0,#666472.0#,6.069622557267915,1.5428141445133405,3.2132450833359405,0.001427607025719007,0.006901265636481244,0.0014369515996920433,52.915969065364244,2.5,4485.547580785226,6.069622557267915,1.5428141445133405
# 0.0,#782943.0#,7.493737880005943,3.5688294211287,3.2141862512354002,0.001736622325001616,0.004491869374804708,0.0018347522174338323,22.364885294766964,2.5,2915.760388031396,7.493737880005943,3.5688294211287
# 0.0,#1239451.0#,4.873965362086076,1.6017255669153663,3.2146077683841368,0.0016959293541149814,0.010913233936965382,0.0033463418722573953,132.43399266019878,2.5,7095.686301233578,4.873965362086076,1.6017255669153663

# Bosons 2 Quantum Particles

# Total Energy vs number of Beads - FOR BHW = 3 find converagance number of beads (All data below is for 2M itr)

# beads_array = np.array([2, 4, 8, 16, 32, 64, 72])
# energy_array = np.array([1.7395061436968315, 1.9979284032744322, 2.0681892824853216, 2.1298897342298404, 2.0976698115269805, 2.124903807277683, 2.0974433474243304])
# stdv_energy_array = np.array([0.014384479212790001, 0.012366058535679911, 0.010588211817062711, 0.01120683054549245, 0.010863020424765689, 0.012543236748566056, 0.014649020291975198])

# plt.axhline(y=2.11, color='r', linestyle='-', label="Convergence (16,2.11)")
# plt.axvline(x = 16, color='r', linestyle='-')
# plt.plot(beads_array, energy_array, 'o', color="blue")
# plt.errorbar(beads_array, energy_array, yerr=stdv_energy_array, fmt="o", ecolor="black")
# plt.title("bhw=3 - Tot Energy vs Beads")
# plt.xlabel("beads")
# plt.ylabel("Total Energy")
# plt.legend(loc='lower right')
# plt.show()


# Fig7 Replication BOSONS partilces

w = 1
bhw = [2, 3, 4, 5, 6]
# energy_array_hw = [2.4096314412021793, 2.0974433474243304, 2.0392088383062266, 2.008478942194873, 1.9993265561758102]
# stdv_energy_array = [0.024837541563550877, 0.014649020291975198, 0.0034618631722831725, 0.014645367890929661, 0.009282708812675358]
bhw = [1, 3, 4, 5, 6]
energy_array_hw = np.array([0.00044390700144266294, 0.00023494208877656474, 0.00022542832309916554, 0.00022278789487966625,  0.00022122040685063573]) /hw
energy_array_hw_b = np.array([0.00044546168466307035, 0.00023353297503444983,  0.0002249183586180965, 0.00022288639594252382, 0.00022081155604605047]) /hw
figtotenergy = plt.figure()
plt.rcParams.update({'font.size': 13})
q = np.linspace(0.8, 6.5, 1000)
# p = part * hbar * w * (np.exp(hbar*w*q*4)+np.exp(hbar*w*q*3)+4*np.exp(hbar*w*q*2)+np.exp(hbar*w*q)+1)/(np.exp(hbar*w*q*4)-1)  # 3Bosons
w = 1 # since it is normalized by it
# p = part * hbar * w * (1 + (2*np.exp(-hbar*w*q))/(1-np.exp(-hbar*w*q)))  # 2 Bosons
p = 2*hbar*w * (np.exp(4*hbar*w*q)+np.exp(3*hbar*w*q)+4*np.exp(2*hbar*w*q)+np.exp(hbar*w*q)+1)/((np.exp(hbar*w*q)-1)*(np.exp(hbar*w*q)+1)*(np.exp(2*hbar*w*q)+1))
plt.plot(q, p, 'g',  label="Analytical Result")
plt.plot(bhw, energy_array_hw_b, '.', label="Simulation", color="blue")
plt.plot(bhw, energy_array_hw, '.', label="Simulation", color="red")
# plt.errorbar(bhw, energy_array_hw, yerr=stdv_energy_array, ecolor="black", fmt='o', markersize=3)
plt.title("Total Energy vs bhw")
plt.xlabel("bhw")
plt.ylabel("<E>/hw")
plt.legend(loc='upper right')
plt.show()

# Fig7 Replication Distinguishable partilces

# w=1
# bhw = [2, 3, 4, 5, 6]
# energy_array_hw = [2.616543754337255, 2.197529044650856, 2.072392746224948, 2.005743796543706, 1.9996729203280086]
# stdv_energy_array = [0.014703183110991508, 0.016797341507523517, 0.003752243273667098, 0.009754321389733773, 0.008868449634375707]
# figtotenergy = plt.figure()
# plt.rcParams.update({'font.size': 13})
# q = np.linspace(0.2, 11, 1000)
# p = part * hbar * w * (1 + ((2 * np.exp(- hbar * w * q))/(1 - np.exp(- hbar * w * q))))
# plt.plot(q, p, 'g',  label="Analytical Result")
# plt.plot(bhw, energy_array_hw, '.', label="Mean Total Energy", color="blue")
# plt.errorbar(bhw, energy_array_hw, yerr=stdv_energy_array, ecolor="black", fmt='o', markersize=3)
# plt.xlabel("bhw")
# plt.ylabel("<E>/hw")
# plt.legend(loc='upper right')
# plt.show()

# Figure 7 - BOTH DISTINGUISHABLE and BOSONS results - b for bosons   d for distinguishable

# bhw = [2, 3, 4, 5, 6]
# w=1
# figtotenergy = plt.figure()
# plt.rcParams.update({'font.size': 13})
# energy_array_hw_b = [2.4096314412021793, 2.0974433474243304, 2.0392088383062266, 2.008478942194873, 1.9993265561758102]
# stdv_energy_array_b = [0.024837541563550877, 0.014649020291975198, 0.0034618631722831725, 0.014645367890929661, 0.009282708812675358]
# energy_array_hw_d = [2.616543754337255, 2.197529044650856, 2.072392746224948, 2.005743796543706, 1.9996729203280086]
# stdv_energy_array_d = [0.014703183110991508, 0.016797341507523517, 0.003752243273667098, 0.009754321389733773, 0.008868449634375707]
# q = np.linspace(0.2, 11, 1000)
# p = part * hbar * w * (np.exp(hbar*w*q*4)+np.exp(hbar*w*q*3)+4*np.exp(hbar*w*q*2)+np.exp(hbar*w*q)+1)/(np.exp(hbar*w*q*4)-1)
# plt.plot(q, p, 'k',  label="Bosons - Analytical Result")
# plt.plot(bhw, energy_array_hw_b, '.', label="Boson - Simulation", color="blue")
# plt.errorbar(bhw, energy_array_hw_b, yerr=stdv_energy_array_b, ecolor="blue", fmt='o', markersize=3)
#
# x1 = np.linspace(0.2, 11, 1000)
# y1 = part * hbar * w * (1 + ((2 * np.exp(- hbar * w * x1))/(1 - np.exp(- hbar * w * x1))))
# plt.plot(x1, y1, 'r',  label="Distinguishable - Analytical Result", dashes=[6, 2])
# plt.plot(bhw, energy_array_hw_d, '.', label="Distinguishable - Simulation", color="green")
# plt.errorbar(bhw, energy_array_hw_d, yerr=stdv_energy_array_d, ecolor="red", fmt='o', markersize=3, color="green")
# plt.xlabel("bhw")
# plt.ylabel("<E>/hw")
# plt.legend(loc='upper right')
# plt.show()

# Block Averaging - STDV of Data vs Block Size

# block_size_array = np.linspace(1, 1000, 200).astype(int)
# avg_array = np.zeros(len(block_size_array))
# stdv_array = np.zeros(len(block_size_array))
# number_of_blocks_array = np.zeros(len(block_size_array))
#
# for i, value in enumerate(block_size_array):
#     number_of_blocks, avg, stdv = block_averaging(5000, block_size=value, data=pot)
#     avg_array[i] = avg
#     stdv_array[i] = stdv
#     number_of_blocks_array[i] = number_of_blocks
#
# figblock = plt.figure()
# plt.plot(block_size_array, stdv_array, label="stdv", color="red")
# plt.xlabel("Block Size")
# plt.ylabel("STDV")
# plt.legend()
# plt.show()
#
# number_of_blocks1, avg, stdv = block_averaging(cutoff, block_size=165, data=pot)
# print ("number of blocks", number_of_blocks1)

