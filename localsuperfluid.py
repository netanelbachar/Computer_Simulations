# Imports
import networkx as nx
from tqdm import tnrange, tqdm, tqdm_notebook
from IPython.display import Image
from numpy import linalg as LA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Iterable


#Functions:


def remove_nan(index, natoms, skp):
    if index > skp:
        if natoms == 3:  # for N = 3
            if index % 5 == 0 or index % 5 == 1:  # 5 for 3 particles
                return True
        elif natoms == 4:
            if index % 6 == 0 or index % 6 == 1:  # 6 for 4 particles
                return True
        elif natoms == 6:
            if index % 8 == 0 or index % 8 == 1:  # 8 for 6 particles
                return True
        else:          # for N = 10
            if index % 12 == 0 or index % 12 == 1:    #  12 for 10 particles
                return True
        return False
    else:
        return True


def read_system_xyz(files, skp, first_P=True):
    pos_list = []
    nbeads_files = len(files)
    for inx, file in enumerate(files):
        length = 0

        print('File: ', file)
        if first_P == True:
            pos1 = pd.read_table(file, skiprows=lambda x: remove_nan(x, natoms, skp), delim_whitespace=True,
                                            names=['x', 'y', 'z']).to_numpy()
            pos_list.append(pos1)
        else:

            pos2 = pd.read_table(file, skiprows=lambda x: remove_nan(x, natoms, skp), delim_whitespace=True,
                                      names=['x', 'y', 'z']).to_numpy()
            pos_list.append(pos2)
            print(inx)

    length = len(pos_list[0])

    positions = np.reshape(pos_list, (nbeads_files * length, 3))  # since there are 2 list that are appended

    ndata = int(positions.shape[0] / nbeads_files / natoms)
    natom_frame = int(ndata * natoms)
    beads = np.zeros((nbeads_files, natom_frame, 3))

    for i in range(nbeads_files):
        beads[i] = positions[i * natom_frame:(i + 1) * natom_frame, :]
    beads = np.reshape(beads, (nbeads_files, ndata, natoms, 3))

    return beads, ndata


def Permutation_W_A(beads, beads2, ndata, m):
    A_est = np.zeros(0) #### ADDED
    A2 = np.zeros(3)
    Icl = 0
    count_avg = 0
    for t in range(nstart, ndata):  # t goes fron nstart to all time steps
        count_avg += 1

        if t % 1000 == 0:
            print("t", t, "ndata", ndata)  # This is only to see progess of run.
            print("A2:", A2, "Icl: ", Icl)
        # gc = permutations_pimdb(beads2, t, ring_dist)  #beads2 is only 1 and P to find permutation
        gc = permutations_chang(beads2, t)
        ## I wrote here count_avg += 1 again! mistake!

        # W2, ww, cond, superfluid_sum, rhos_div_rho = Winding(t, D_p, W2,  ring_len_dist, ring_dist, gc)
        # return W2, ww, cond, superfluid_sum, rhos_div_rho

        A_2, I_cl = superfluid_area(gc, beads, t)  # beads is to calculate the area (all files)
        A2 += A_2
        Icl += I_cl
        A_est = np.append(A_est, sum(A_2)) #### ADDED
    print("A2:", A2, "Icl: ", Icl)
    Az = sum(A2) / count_avg
    Icl_m = Icl / count_avg
    return Az, Icl_m, A_est, count_avg


def superfluid_area(gconnected, beads, t):
    Atmp = np.zeros(3)
    Itmp = 0
    # xcm, ycm = center_of_mass(beads, t)  # center of mass of all beads in time step
    # r_beads_cm = beads[:, t] - [xcm, ycm, 0]  # coord change to COM only of the permutated beads
    r_beads_cm = beads[:, t] - [0, 0, 0]  # coord change to COM only of the permutated beads
    for inx, g in enumerate(gconnected):
        g = list(g)
        l = len(g)
        M = len(beads)
        for i, n in enumerate(g):  # which permutation
            # xcm, ycm = center_of_mass_per_perm(g, beads, t)  # center of mass PER PERMUTATION
            # r_beads_cm = beads[:, t] - [xcm, ycm, 0]  # coord change to COM only of the permutated beads
            if i != l-1:
                for j in range(M):  # which bead
                    if j != M-1:    # unitl bead P
                        Atmp += np.cross(beads[j][t][g[i]], beads[j+1][t][g[i]])  # from bead 1  to bead P
                        Itmp += np.dot(r_beads_cm[j][g[i]], r_beads_cm[j+1][g[i]])
                    else:
                        Atmp += np.cross(beads[-1][t][g[i]], beads[0][t][g[i+1]])
                        Itmp += np.dot(r_beads_cm[-1][g[i]], r_beads_cm[0][g[i+1]])
            else:
                for j in range(M):
                    if j != M-1:
                        Atmp += np.cross(beads[j][t][g[i]], beads[j+1][t][g[i]])
                        Itmp += np.dot(r_beads_cm[j][g[i]], r_beads_cm[j+1][g[i]])
                    else:
                        Atmp += np.cross(beads[-1][t][g[i]], beads[0][t][g[0]])
                        Itmp += np.dot(r_beads_cm[-1][g[i]], r_beads_cm[0][g[0]])

    Atmp = (1/2)*Atmp * conv_m_to_bohr**2
    A_squared = Atmp**2
    Itmp = m * Itmp * conv_m_to_bohr**2 #Itm = Itm bad use of words so i update it

    return A_squared, Itmp


def permutations_chang(beads, t):
    ring_dist = []
    pair = []
    for i in range(natoms):  # natoms
        delR = beads[0, t, :, :] - beads[-1, t, i, :]    # ensures file 1 and file P
        j = LA.norm(delR, axis=-1).argmin()
        # tmptmp = np.delete(LA.norm(delR, axis=-1), j)
        # jj = tmptmp.argmin()
        if (i != j):
            pair += [(i, j)]
        else:
            ring_dist += [1]
        for e in range(len(pair) - 1):   #my add
            if j == pair[e][1]:          #my add
                pair.pop(-1)             # my add
    # g_connected1, ring_len_dist, G1 = to_graph(pair)
    g_connected, ring_len_dist, G = to_graph_jacob(pair)
    ring_dist += ring_len_dist
        ##########adds the single configurations#####################################
    # bla = []
    # for i in a:
    #     print(i)
    #     for j in i:
    #         bla.append(j)
    g_connected1 = list(flatten(g_connected))
    for i in range(natoms):
        if i not in g_connected1:
            g_connected += [[i]]
#######################################################
    return g_connected


def flatten(lis):
    '''
    Flatens any nested list to a one -D list
    '''
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


def to_graph_jacob(pair):
    G = nx.DiGraph()
    G.add_edges_from(pair)
    connected_comp = nx.weakly_connected_components(G)
    g_connected, ring_len_list = [], []
    # Iterate through the connected components
    for comp in connected_comp:
        # Extract the corresponding sub-graph
        sub_g = G.subgraph(comp).copy()
        # If the graph is cyclic, turn it into an acyclic graph
        # by removing a single edge.
        if not nx.is_directed_acyclic_graph(sub_g):
            sub_g.remove_edge(*list(sub_g.edges())[0])
        # Sort the nodes of the resulting sub-graph topologically
        top_sort = list(nx.topological_sort(sub_g))
        g_connected.append(top_sort)
        ring_len_list.append(len(top_sort))

    return g_connected, ring_len_list, G



# Constants
bhw_list = [1, 3, 4, 6, 10, 30, 60]
conv_m_to_bohr = 5.29177249*10**-11
hbar = 1.05457182e-34                 # MKS
kb = 1.380649e-23                     # MKS
kB = 0.0083144621                     # Boltzmann const in kJ/mol/K
nbeads=2   #only read first and last bead
count = 0
winding_or_area = True   # True for Winding False for Area
files_range = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
first_bead = '01'
L = [718.096, 310.9446892, 3000]

# Constant that change from simulation to simulation ###########################################3
# m = 0.84*9.1093837e-31              # mass of exciton
m = 9.1093837e-31                     # mass of electron
# omega = 0.02673                     # [eV]
omega = 0.003                         # [eV]
omega = omega/27.2114         # [Hartree]                     # only relevant for moire
omega_kJmol = omega/3.8088E-4  # [kJmol]               # only relevant for moire
# ######################################per simulation ###############################################
nstart = 0 # 500000 # 600000#20000 #80000  # (yes53000 51K)  (yes103K125468) #(5K10K31K115000)
last_bead = 96
natoms = 4  #  10 for BOSONS  Number of Bosons in simulation  4 for N4
bhw = 1  #0.51698126 (600k),  0.31018876(1000K),  0.23860674(1300K)]
skp = 500003 # 20003 # #10000008 # 1200000 #200003 #2000003


T = 1 / (kB*(bhw / omega_kJmol))
# T = (bhw * omega * 27.2114) / ((kB / 96.4853))  # for DORNHEIM SIMULATION # [K]
print("Temperature:", T)
p = last_bead

s = 1 # # sim1 sim2 sim3 sim4 sim5

# path_ = '/home/netanelb2/Desktop/Netanel/Research/exiton/anharmonic/anharmonic18/'
# path = path_ + 'bhw{}/sim{}/{}beads/'.format(bhw,  s, p)

# path_ = '/home/netanelb2/Desktop/Netanel/Research/exiton/moire_10_1/conv_sim_bhw031_m/conv_sim/'
# path = path_ + 'sim{}/{}beads/'.format(s, p)

# path_ = '/home/netanelb2/Desktop/Netanel/Research/exiton/moire_10_1/conv_sim_bhw031_m/conv_sim/'
# path = path_ + 'sim{}/{}beads/'.format(s, p)

path_ = '/hirshblab-storage/netanelb2/PIMD/exitons/dornheim/dipole/N4/lambda285/table/bhw1_96/'
path = path_ + 'sim{}/'.format(s)

# ######################################READ FILE ###############################################
# files_range = ['1', '2', '3', '4', '5', '6', '7', '8']
# files_range = ['1', '2', '3', '4']
# # files_range = ['1', '2']
# first_bead = '1'

for i in range(10, last_bead+1):
    files_range.append(i)
files = [path+'system_{}.xyz'.format(k) for k in files_range]    # All system files

files2 = [path+'system_{}.xyz'.format(k) for k in [first_bead, last_bead]]   # Only file 1 and P for permutations

print ("Permutation files are being read")
beads2, ndata2 = read_system_xyz(files2, skp, first_P=winding_or_area)
print("Permutations can be calculated and now all system files read")
beads, ndata = read_system_xyz(files, skp, first_P=winding_or_area)
print("All files are ready")
A2_mean, Icl_mean, A_esti, count_t = Permutation_W_A(beads, beads2, ndata, m)
rho = 4*m**2*(A2_mean)/((1/(kb*T))*hbar**2*(Icl_mean/last_bead))
print("Temp: ", round(T, 2), "rho: ", rho, "Area: ", A2_mean, "Icl: ", Icl_mean, "skip: ", skp, "nstart: ", nstart, "Beads: ", last_bead, "sim: ", s)
print ("Hello")

# 10 Excitons Moire Potential
rho_5 = 'Temp:  5.17 rho:  0.9141359264894422 Area:  4.5282250820284325e-34 Icl:  7.148097541486153e-46 skip:  3000000 nstart:  35000'
rho_51 = 'Temp:  51.7 rho:  0.9494176599082494 Area:  8.148371428838692e-35 Icl:  8.256490666576819e-46 skip:  35280001 nstart:  50000'
rho_77 = 'Temp:  77.55 rho:  0.9622329231708723 Area:  5.553519025206186e-35 Icl:  8.328394589455852e-46 skip:  25000007 nstart:  200000'
rho_100 = 'Temp:  103.4 rho:  0.92595914063145 Area:  4.078971444503942e-35 Icl:  8.475607178048868e-46 skip:  32000003 nstart:  60000'
rho_300 = 'Temp:  310.19 rho:  0.0400226617114075 Area:  3.557510387024389e-35 Icl:  1.282665608717675e-44 skip:  49000007 nstart:  125000'

# 10 Excitons in harmonic Potential
rho_5 = 'Temp:  5.17 rho:  1.0070505313824765 Area:  4.4896661480776656e-34 Icl:  6.433332925968575e-46 skip:  100007 nstart:  5000'
rho_51 = 'Temp:  51.7 rho:  1.0218973514985046 Area:  7.250447654567001e-35 Icl:  3.412789758899014e-46 skip:  3000000 nstart:  35000  Beads: 32'
rho_100 = 'Temp:  103.4 rho:  0.9498188950987141 Area:  3.965546027786578e-35 Icl:  8.032933440205491e-46 skip:  2000003 nstart:  50000 Beads:  64'
rho_300 = 'Temp:  310.19 rho:  0.5303139973902821 Area:  9.731707358924307e-36 Icl:  2.6480683602059484e-46 skip:  16000007 nstart:  140000 Beads:  16_long'

# 10 Excitons in anharmonic Potential
rho_5 = 'Temp:  5.17 rho:  0.977821432655886 Area:  4.459755913442623e-34 Icl:  6.581498391594566e-46 skip:  3300000 nstart:  10000'
rho_51 ='Temp:  51.7 rho:  0.9086736426783152 Area:  6.407114778701302e-35 Icl:  3.3916151003385446e-46 skip:  35000003 nstart:  70000'
rho_100 = ''
rho_300 = 'Temp:  310.19 rho:  0.6263570803752564 Area:  9.699766068828375e-36 Icl:  2.234665432570465e-46 skip:  71500007 nstart:  25000'
####################################
###########DORNHEIM >#########################
####################################
##DORNHEIM DIPOLE DIPOLE lambda = 3 and HARMONIC 3meV  ( 31 ) ############################################################
rho_bhw_5 = 'Temp:  6.96 rho:  1.0019842621394717 Area:  6.021828551081763e-33 Icl:  1.2415006783519112e-44 skip:  200004 nstart:  100000 Beads:  72 sim:  1'
rho_bhw_2 = 'Temp:  17.41 rho:  0.5879569471639767 Area:  1.6127941808861017e-33 Icl:  1.4166184873813363e-44 skip:  30001 nstart:  0 Beads:  72 sim:  1'
rho_bhw_2 = 'Temp:  17.41 rho:  0.5936657581819048 Area:  1.6167373682276535e-33 Icl:  1.4064262381266904e-44 skip:  200004 nstart:  100000 Beads:  72 sim:  2'
rho_bhw_2 = 'Temp:  17.41 rho:  0.6345338991534536 Area:  1.7310076787918853e-33 Icl:  1.4088464092059985e-44 skip:  200004 nstart:  100000 Beads:  72 sim:  3'
rho_bhw_1 = 'Temp:  34.81 rho:  0.14632083401542265 Area:  2.675954466368798e-34 Icl:  1.8889568211413386e-44 skip:  200004 nstart:  100000 Beads:  72 sim:  3'
##DORNHEIM DIPOLE DIPOLE lambda = 3 ( 0.03) and HARMONIC 3meV ############################################################
rho_bhw_1 = 'Temp:  34.81 rho:  0.26515223172837316 Area:  3.013513961282384e-34 Icl:  1.1738899927697764e-44 skip:  100003 nstart:  10000 Beads:  72 sim:  1'
rho_bhw_1 = 'Temp:  34.81 rho:  0.25158732649816523 Area:  2.940036505494773e-34 Icl:  1.207017153662039e-44 skip:  100003 nstart:  10000 Beads:  72 sim:  2'
rho_bhw_1 = 'Temp:  34.81 rho:  0.25918654001980207 Area:  2.965120129729434e-34 Icl:  1.1816240679953356e-44 skip:  100003 nstart:  10000 Beads:  72 sim:  3'
##DORNHEIM DIPOLE DIPOLE lambda = 3 ( 285 ) and HARMONIC 3meV ############################################################
######### Lambda = 3  (20M itr)
rho_bhw_1 = 'Temp:  34.81 rho:  0.07450446640248616 Area:  1.9023160715719318e-34 Icl:  2.637241174780599e-44 skip:  120000 nstart:  0 Beads:  72 sim:  1'
rho_bhw_1 = 'Temp:  34.81 rho:  0.07508812372683552 Area:  1.904374498839715e-34 Icl:  2.619573474262984e-44 skip:  120000 nstart:  0 Beads:  72 sim:  2'
rho_bhw_1 = 'Temp:  34.81 rho:  0.07426216392296299 Area:  1.8902363128956215e-34 Icl:  2.6290447677639936e-44 skip:  120000 nstart:  0 Beads:  72 sim:  3'

rho_bhw_2 = 'Temp:  17.41 rho:  0.14909784666075215 Area:  6.337086776317075e-34 Icl:  2.1950163857125136e-44 skip:  10002 nstart:  0 Beads:  72 sim:  1'
rho_bhw_2 = 'Temp:  17.41 rho:  0.14927423359015654 Area:  6.3115902940679e-34 Icl:  2.183601747634727e-44 skip:  10002 nstart:  0 Beads:  72 sim:  2'
rho_bhw_2 = 'Temp:  17.41 rho:  0.2507591000459954 Area:  1.048639043903838e-33 Icl:  2.1596770084080177e-44 skip:  120001 nstart:  0 Beads:  72 sim:  3'

rho_bhw_5 = 'Temp:  6.96 rho:  0.3319919109742599 Area:  3.211096008164227e-33 Icl:  1.9980449921324204e-44 skip:  120001 nstart:  0 Beads:  72 sim:  1'
rho_bhw_5 = 'Temp:  6.96 rho:  1.1611151799715829 Area:  1.117525361833692e-32 Icl:  1.9882068244639145e-44 skip:  120001 nstart:  0 Beads:  72 sim:  2'
rho_bhw_5 ='Temp:  6.96 rho:  0.49856160697551244 Area:  4.800033730716591e-33 Icl:  1.9888632947403452e-44 skip:  120001 nstart:  0 Beads:  72 sim:  3'
##DORNHEIM DIPOLE DIPOLE lambda = 3 ( 285 ) and HARMONIC 3meV ############################################################
# with TABLE (40M itr)
rho_bhw_1 = 'Temp:  34.81 rho:  0.07403543405865351 Area:  1.8901805566296284e-34 Icl:  2.6370183012877474e-44 skip:  200003 nstart:  30000 Beads:  72 sim:  1'
rho_bhw_1 = 'Temp:  34.81 rho:  0.07392124494489419 Area:  1.8917559310904566e-34 Icl:  2.643293032114845e-44 skip:  200003 nstart:  30000 Beads:  72 sim:  2 '
rho_bhw_1 = 'Temp:  34.81 rho:  0.07449311837708843 Area:  1.8980485186393375e-34 Icl:  2.6317257776295663e-44 skip:  200003 nstart:  30000 Beads:  72 sim:  3'

rho_bhw_2 = 'Temp:  17.41 rho:  0.25218553711859415 Area:  1.0588954001865875e-33 Icl:  2.1684647627437755e-44 skip:  20003 nstart:  3000 Beads:  72 sim:  1'
rho_bhw_2 = 'Temp:  17.41 rho:  0.150673115089824 Area:  6.385042790175664e-34 Icl:  2.1885049264481874e-44 skip:  200003 nstart:  30000 Beads:  72 sim:  2'
rho_bhw_2 = 'Temp:  17.41 rho:  0.1490986735103017 Area:  6.325125877575738e-34 Icl:  2.1908612645747075e-44 skip:  200003 nstart:  30000 Beads:  72 sim:  3'

rho_bhw_5 = 'Temp:  6.96 rho:  0.3789741004093375 Area:  3.671015627185074e-33 Icl:  2.0010418772597909e-44 skip:  200003 nstart:  30000 Beads:  72 sim:  1'
rho_bhw_5 = 'Temp:  6.96 rho:  0.4771874161592728 Area:  4.6035141386012746e-33 Icl:  1.9928746089246092e-44 skip:  200003 nstart:  30000 Beads:  72 sim:  2'

rho_bhw_1 = 'Temp:  34.81 rho:  0.07559290299401139 Area:  1.9176410869208575e-34 Icl:  3.4936107742937813e-44 skip:  500003 nstart:  0 Beads:  96 sim:  1'

rho_bhw_2 = 'Temp:  17.41 rho:  0.1506152737493438 Area:  6.413100060323872e-34 Icl:  2.9319544627408465e-44 skip:  20003 nstart:  1000 Beads:  96 sim:  1'
rho_bhw_2 = 'Temp:  17.41 rho:  0.15440193214656928 Area:  6.563269797387313e-34 Icl:  2.927020383450638e-44 skip:  500003 nstart:  30000 Beads:  96 sim:  2'
rho_bhw_2 = 'Temp:  17.41 rho:  0.14925315303321943 Area:  6.352341395742965e-34 Icl:  2.9306809152827105e-44 skip:  500003 nstart:  30000 Beads:  96 sim:  3'

rho_bhw5 = 'Temp:  6.96 rho:  0.6558045543510903 Area:  6.33968312715877e-33 Icl:  2.662633149020839e-44 skip:  500003 nstart:  30000 Beads:  96 sim:  1'
rho_bhw5 = 'Temp:  6.96 rho:  0.3692613153038909 Area:  3.5986927009360326e-33 Icl:  2.6842883239875403e-44 skip:  500003 nstart:  30000 Beads:  96 sim:  2'
rho_bhw5 = 'Temp:  6.96 rho:  0.581187270656259 Area:  5.637365462575582e-33 Icl:  2.671642283693071e-44 skip:  500003 nstart:  30000 Beads:  96 sim:  3'


####################################
###########DORNHEIM ^#########################
####################################



################################################### Shorter simulations

# 10 Excitons Moire Potential
rho_5 = ''
rho_51 = ''

rho_100 = 'Temp:  103.4 rho:  0.9430319325999885 Area:  8.100003145158185e-35 Icl:  8.263057431069089e-46 skip:  200003 nstart:  20000 Beads:  32 sim:  1'
rho_100 = 'Temp:  103.4 rho:  0.9625693757405263 Area:  8.278209398935105e-35 Icl:  8.273444349242822e-46 skip:  200003 nstart:  20000 Beads:  32 sim:  2'
rho_100 = 'Temp:  103.4 rho:  0.9354270335807197 Area:  8.025536936540437e-35 Icl:  8.253652191611284e-46 skip:  200003 nstart:  20000 Beads:  32 sim:  3'
rho_100 = 'Temp:  103.4 rho:  0.944315009208291 Area:  8.133031791975861e-35 Icl:  8.285477839432095e-46 skip:  200003 nstart:  20000 Beads:  32 sim:  4'
rho_100 = 'Temp:  103.4 rho:  0.943192666254992 Area:  8.092947749924035e-35 Icl:  8.254453091860166e-46 skip:  200003 nstart:  20000 Beads:  32 sim:  5'
mean, std = (0.9457072034754359, 0.009004808972510754)

rho_155 = 'Temp:  155.09 rho:  0.8324253107872759 Area:  4.932469068671174e-35 Icl:  4.2752581110319574e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  1' # NO Diffusion
rho_155 = 'Temp:  155.09 rho:  0.8487089205754764 Area:  4.952293915209719e-35 Icl:  4.2100852595393856e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  2'
rho_155 = 'Temp:  155.09 rho:  0.8576107883264682 Area:  5.04259004592462e-35 Icl:  4.2423517295501334e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  3'
rho_155 = 'Temp:  155.09 rho:  0.9165863967758067 Area:  5.361136244107076e-35 Icl:  4.220138291273193e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  4'
rho_155 = 'Temp:  155.09 rho:  0.9054218331816718 Area:  5.158858305511538e-35 Icl:  4.11098491208595e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  5'
mean, std = (0.87215064975528, 0.032925771035558105)

rho_200 = 'Temp:  206.79 rho:  0.9710176413496554 Area:  4.94087463121898e-35 Icl:  4.8950676346758465e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  1' # NO Diffusion
rho_200 = 'Temp:  206.79 rho:  0.04156120760867569 Area:  3.703904597422931e-35 Icl:  8.573410461719677e-45 skip:  200003 nstart:  20000 Beads:  16 sim:  2'  # Diffusion
rho_200 = 'Temp:  206.79 rho:  0.6766561119500543 Area:  3.375278397785613e-35 Icl:  4.798699707603495e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  3'    # NO Diffusion
rho_200 = 'Temp:  206.79 rho:  0.6715265232682954 Area:  3.2318422179007783e-35 Icl:  4.6298717594916665e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  4'  # NO Diffusion
rho_200 = 'Temp:  206.79 rho:  0.7479593066823002 Area:  3.674735886726142e-35 Icl:  4.7263964006579074e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  5'   # NO Diffusion
mean, std =(0.6987139806334333, 0.03488461717582711)  # only of last 3 samples

rho_300 = 'Temp:  310.19 rho:  0.6090047477080791 Area:  3.8371701708457947e-34 Icl:  9.092087844945452e-45 skip:  200003 nstart:  50000 Beads:  16' # Diffusion
rho_300 = 'Temp:  310.19 rho:  0.5319764113431913 Area:  6.721484431584802e-34 Icl:  1.8232493328826356e-44 skip:  300000 nstart:  20000 Beads:  16 sim:  1'
rho_300 = 'Temp:  310.19 rho:  0.03921287665383172 Area:  6.77486688825628e-35 Icl:  2.4931321840874755e-44 skip:  7000007 nstart:  100000 Beads:  16 sim:  2'
rho_300 = 'Temp:  310.19 rho:  0.04223883309460876 Area:  4.874630639699175e-35 Icl:  1.6653404999890374e-44 skip:  7000007 nstart:  100000 Beads:  16 sim:  3'
rho_300 = 'Temp:  310.19 rho:  0.10409125833377124 Area:  1.3679155701055817e-34 Icl:  1.896349039111581e-44 skip:  7000007 nstart:  100000 Beads:  16 sim:  4'
rho_300 = 'Temp:  310.19 rho:  0.04521548367827463 Area:  8.424105258774508e-35 Icl:  2.6884987908037147e-44 skip:  7000007 nstart:  100000 Beads:  16 sim:  5'
mean, std =(0.15254689898, 0.1912314076636037)

rho_400 = 'Temp:  413.58 rho:  0.3380034979351278 Area:  4.211934685103656e-34 Icl:  2.397576159439399e-44 skip:  6000000 nstart:  200000 Beads:  16 sim:  1' # Diffusion
rho_400 = 'Temp:  413.58 rho:  0.2786321753881636 Area:  3.3967417223012193e-34 Icl:  2.345542131844346e-44 skip:  6000000 nstart:  200000 Beads:  16 sim:  2'
rho_400 = 'Temp:  413.58 rho:  0.41295254036776186 Area:  4.6567454489504615e-34 Icl:  2.1696732521183402e-44 skip:  6000000 nstart:  200000 Beads:  16 sim:  3'
rho_400 = 'Temp:  413.58 rho:  0.4021714182565222 Area:  5.677284745590507e-34 Icl:  2.7160730891066137e-44 skip:  6000000 nstart:  200000 Beads:  16 sim:  4'
rho_400 = 'Temp:  413.58 rho:  0.3321880674854221 Area:  3.6379043737936577e-34 Icl:  2.1070712877710572e-44 skip:  6000000 nstart:  200000 Beads:  16 sim:  5'
mean, std = (0.352789534472, 0.04939740385669734)

rho_600 ='Temp:  600.0 rho:  0.5775303665379758 Area:  5.311295799943121e-34 Icl:  1.9252421849114525e-44 skip:  6000000 nstart:  30000 Beads:  12 sim:  1'
rho_600 ='Temp:  600.0 rho:  1.2021453775375563 Area:  1.0289421043528536e-33 Icl:  1.7918169993650155e-44 skip:  6000000 nstart:  30000 Beads:  12 sim:  2'
rho_600 ='Temp:  600.0 rho:  1.0219420071836172 Area:  1.147384258055235e-33 Icl:  2.350403040469795e-44 skip:  6000000 nstart:  30000 Beads:  12 sim:  3'
rho_600 ='Temp:  600.0 rho:  1.368796468693988 Area:  1.0166269923062944e-33 Icl:  1.5548283999815934e-44 skip:  6000000 nstart:  30000 Beads:  12 sim:  4'
rho_600 ='Temp:  600.0 rho:  1.4847315593971357 Area:  1.5135617481201788e-33 Icl:  2.1340859560638464e-44 skip:  6000000 nstart:  30000 Beads:  12 sim:  5'
mean, std = (1.1310274, 0.3177902744059988)
rho_1000 ='Temp:  1000.0 rho:  1.900256195430524 Area:  1.0539154631072145e-33 Icl:  1.2900626711010035e-44 skip:  6000000 nstart:  30000 Beads:  8 sim:  4'
rho_1000 ='Temp:  1000.0 rho:  1.5941406140938632 Area:  9.263763925238545e-34 Icl:  1.3516929292617205e-44 skip:  6000000 nstart:  30000 Beads:  8 sim:  1'
rho_1000 ='Temp:  1000.0 rho:  0.9166874079377652 Area:  5.388825556488092e-34 Icl:  1.367382598783121e-44 skip:  6000000 nstart:  30000 Beads:  8 sim:  2'
rho_1000 ='Temp:  1000.0 rho:  1.8182602934170207 Area:  9.518410781842722e-34 Icl:  1.2176587055249238e-44 skip:  6000000 nstart:  30000 Beads:  8 sim:  3'
rho_1000 ='Temp:  1000.0 rho:  1.266840740099092 Area:  6.302426465597554e-34 Icl:  1.1571854257219536e-44 skip:  6000000 nstart:  30000 Beads:  8 sim:  5'
mean, std = (1.499235628, 0.36468225572948726)
rho_1300 ='Temp:  1300.0 rho:  2.015015103540041 Area:  9.595824362974755e-34 Icl:  1.4400067352948698e-44 skip:  6000000 nstart:  30000 Beads:  8 sim:  1'
rho_1300 ='Temp:  1300.0 rho:  2.575020689860659 Area:  1.0856484045214202e-33 Icl:  1.2748791381277776e-44 skip:  6000000 nstart:  30000 Beads:  8 sim:  2'
rho_1300 ='Temp:  1300.0 rho:  2.904438640847669 Area:  1.278460439653001e-33 Icl:  1.331023204214948e-44 skip:  6000000 nstart:  30000 Beads:  8 sim:  4'
rho_1300 ='Temp:  1300.0 rho:  2.2230658608032363 Area:  9.139228948972479e-34 Icl:  1.243133490332106e-44 skip:  6000000 nstart:  30000 Beads:  8 sim:  3'
rho_1300 = 'Temp:  1300.0 rho:  1.4743216130664933 Area:  6.8569113894468224e-34 Icl:  1.40636109544592e-44 skip:  6000000 nstart:  30000 Beads:  8 sim:  5'
mean, std = (2.29501755, 0.28000245)
# 10 Excitons in harmonic Potential
rho_5_1 = 'Temp:  5.17 rho:  1.1512483016728996 Area:  1.0265917285649589e-33 Icl:  1.2867731786180042e-45 skip:  200003 nstart:  20000 Beads:  96 sim:  1'
rho_5_1 = 'Temp:  5.17 rho:  0.9846654559802885 Area:  8.772179153375062e-34 Icl:  1.2855590561069761e-45 skip:  1200000 nstart:  20000 Beads:  96 sim:  1'
rho_5_2 = 'Temp:  5.17 rho:  1.2271175285822455 Area:  1.0905380221274687e-33 Icl:  1.2824129472432624e-45 skip:  1200000 nstart:  20000 Beads:  96 sim:  2'
rho_5_3 = 'Temp:  5.17 rho:  0.9540691133750003 Area:  8.485458533173283e-34 Icl:  1.283419757702548e-45 skip:  1200000 nstart:  20000 Beads:  96 sim:  3'
mean, std =  (1.0552750000000002, 0.1221393725080765)

rho_51_1 = 'Temp:  51.7 rho:  1.0211818424456258 Area:  1.6050080745622974e-34 Icl:  1.5120150717541773e-45 skip:  1200000 nstart:  20000 Beads:  64 sim:  1'
rho_51_2 = 'Temp:  51.7 rho:  1.0083841188436782 Area:  1.582392285887166e-34 Icl:  1.509628695484567e-45 skip:  1200000 nstart:  20000 Beads:  64 sim:  2'
rho_51_3 = 'Temp:  51.7 rho:  0.9924494723338456 Area:  1.5581703419418538e-34 Icl:  1.510387946999105e-45 skip:  1200000 nstart:  20000 Beads:  64 sim:  3'
mean, std =  (1.0073381333333333, 0.011752917012479108)

rho_100 = 'Temp:  103.4 rho:  0.9887073380779569 Area:  3.922535880894139e-35 Icl:  3.81663958307997e-46 skip:  200003 nstart:  20000 Beads:  32 sim:  1'
rho_100 = 'Temp:  103.4 rho:  0.9521869262626861 Area:  3.7735761525149056e-35 Icl:  3.812526641066748e-46 skip:  200003 nstart:  20000 Beads:  32 sim:  2'
rho_100 = 'Temp:  103.4 rho:  0.9566210746604368 Area:  3.796100557492307e-35 Icl:  3.817506160769341e-46 skip:  200003 nstart:  20000 Beads:  32 sim:  3'
rho_100 = 'Temp:  103.4 rho:  0.961473284573178 Area:  3.8207016905644525e-35 Icl:  3.822855581757429e-46 skip:  200003 nstart:  20000 Beads:  32 sim:  4'
rho_100 = 'Temp:  103.4 rho:  0.9508261227444378 Area:  3.762084954384117e-35 Icl:  3.806356628801404e-46 skip:  200003 nstart:  20000 Beads:  32 sim:  5'
mean, std = (0.9619629207999999, 0.013882636186020389)

rho_155 = 'Temp:  155.09 rho:  0.8403111627694442 Area:  2.214723260547953e-35 Icl:  1.9016149375300645e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  1'
rho_155 = 'Temp:  155.09 rho:  0.856420025772128 Area:  2.2754536776658567e-35 Icl:  1.917010232216627e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  2'
rho_155 ='Temp:  155.09 rho:  0.8598758445059708 Area:  2.330476316390528e-35 Icl:  1.9554746522628554e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  3'
rho_155 ='Temp:  155.09 rho:  0.8576897220976247 Area:  2.2450240882729987e-35 Icl:  1.888574169241165e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  4'
rho_155 ='Temp:  155.09 rho:  0.9149688075853871 Area:  2.424966608158266e-35 Icl:  1.9122414161605026e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  5'
mean, std = (0.865853112546111, 0.025518990969505655)

rho_200 = 'Temp:  206.79 rho:  0.7196342901362399 Area:  1.59058173864623e-35 Icl:  2.126307767385436e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  1'
rho_200 = 'Temp:  206.79 rho:  0.7077199947069688 Area:  1.5357019218387745e-35 Icl:  2.0875046176283475e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  2'
rho_200 = 'Temp:  206.79 rho:  0.7226582971155583 Area:  1.6172080039291627e-35 Icl:  2.1528554529762895e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  3'
rho_200 = 'Temp:  206.79 rho:  0.7757469363128356 Area:  1.7260969248801818e-35 Icl:  2.1405584561826824e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  4'
rho_200 = 'Temp:  206.79 rho:  0.7538787679520733 Area:  1.655863161538322e-35 Icl:  2.113026359999935e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  5'
mean, std = (0.7359276572444559, 0.0250800291741976)

rho_300 = 'Temp:  310.19 rho:  0.531766907996796 Area:  9.517389741291782e-36 Icl:  2.5826751768081194e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  1'
rho_300 = 'Temp:  310.19 rho:  0.5075145544437102 Area:  9.379608293558048e-36 Icl:  2.666916647027969e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  2'
rho_300 = 'Temp:  310.19 rho:  0.5405763185108498 Area:  9.745179165482321e-36 Icl:  2.601393516381138e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  3'
rho_300 = 'Temp:  310.19 rho:  0.505399237851452 Area:  9.358612975949353e-36 Icl:  2.6720842458260154e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  4'
rho_300 = 'Temp:  310.19 rho:  0.48923586057545837 Area:  9.149104253114908e-36 Icl:  2.698569050092882e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  5'
mean, std = (0.514898575860326, 0.018693438695243612)

rho_400_1 = 'Temp:  413.58 rho:  0.24940190825894887 Area:  4.8017382626514765e-36 Icl:  2.7782540306276308e-46 skip:  200003 nstart:  20000 Beads:  12 sim:  1'
rho_400_2 = 'Temp:  413.58 rho:  0.2759968716590686 Area:  5.273297462293263e-36 Icl:  2.75709259882832e-46 skip:  200003 nstart:  20000 Beads:  12 sim:  2'
rho_400_3 = 'Temp:  413.58 rho:  0.3197835325313847 Area:  5.793119936457323e-36 Icl:  2.6141449565953373e-46 skip:  200003 nstart:  20000 Beads:  12 sim:  3'
rho_400_4 = 'Temp:  413.58 rho:  0.278364019505326 Area:  5.2482742127370105e-36 Icl:  2.720674967721935e-46 skip:  200003 nstart:  20000 Beads:  12 sim:  4'
rho_400_5 = 'Temp:  413.58 rho:  0.2833751589042247 Area:  5.186408702862717e-36 Icl:  2.6410596029255096e-46 skip:  200003 nstart:  20000 Beads:  12 sim:  5'
mean, std = (0.2813842979878, 0.022536567773055922)
rho_400_1 = 'Temp:  413.58 rho:  0.3036888324037208 Area:  1.1411166533964282e-35 Icl:  7.22958219963822e-46 skip:  2000003 nstart:  500000 Beads:  16 sim:  1'
rho_400_2 = 'Temp:  413.58 rho:  0.29038837646628657 Area:  1.10787953320338e-35 Icl:  7.340494017329713e-46 skip:  2000003 nstart:  500000 Beads:  16 sim:  2'
rho_400_3 = 'Temp:  413.58 rho:  0.29831249919278346 Area:  1.0697147195611356e-35 Icl:  6.899355125380832e-46 skip:  2000003 nstart:  500000 Beads:  16 sim:  3'
rho_400_4 = 'Temp:  413.58 rho:  0.26680912935616163 Area:  1.0289013378523067e-35 Icl:  7.419677426775816e-46 skip:  2000003 nstart:  500000 Beads:  16 sim:  4'
rho_400_5 = 'Temp:  413.58 rho:  0.2853668321123208 Area:  1.0742466908987903e-35 Icl:  7.242900327882242e-46 skip:  2000003 nstart:  500000 Beads:  16 sim:  5'
mean,std = (0.28891296422, 0.012727802174066742)


rho_600_2 = 'Temp:  600.0 rho:  0.11676254032972716 Area:  2.0523496711235814e-36 Icl:  2.4531057269901534e-46 skip:  2400000 nstart:  600000 Beads:  8, s: 2 dt = 0.0001'
rho_600_2 = 'Temp:  600.0 rho:  0.09691932089086044 Area:  2.0459845606054628e-36 Icl:  2.9461878562804517e-46 skip:  2400000 nstart:  600000 Beads:  8 , dt = 0.001'
rho_600_2 = 'Temp:  600.0 rho:  0.08711115628365688 Area:  2.0616359400243447e-36 Icl:  4.954477979838219e-46 skip:  1200001 nstart:  600000 Beads:  12 sim:  2'
rho_600_3 = 'Temp:  600.0 rho:  0.11229383062331293 Area:  2.2524741798488512e-36 Icl:  2.7994477738431625e-46 skip:  13200000 nstart:  700000 Beads:  8 sim:  3'
mean, std = (0.101619601392772, 0.011131099674274158)

rho_600_1 ='Temp:  600.0 rho:  0.09562956526458405 Area:  4.125100603633798e-36 Icl:  9.030297695225391e-46 skip:  2000003 nstart:  500000 Beads:  12 sim:  1'
rho_600_2 = 'Temp:  600.0 rho:  0.08799513869234439 Area:  4.090487791280933e-36 Icl:  9.731417945539901e-46 skip:  2000003 nstart:  500000 Beads:  12 sim:  2'
rho_600_3 = 'Temp:  600.0 rho:  0.08213275152993002 Area:  3.40374130977493e-36 Icl:  8.675607141509968e-46 skip:  2000003 nstart:  500000 Beads:  12 sim:  3'
rho_600_4 = 'Temp:  600.0 rho:  0.09621488840562005 Area:  4.1478765180814244e-36 Icl:  9.024917549603741e-46 skip:  2000003 nstart:  500000 Beads:  12 sim:  4'
rho_600_5 = 'Temp:  600.0 rho:  0.07948610886100897 Area:  3.693637676841742e-36 Icl:  9.727982010542064e-46 skip:  2000003 nstart:  500000 Beads:  12 sim:  5'
mean, std = (0.08829169023291601, 0.0068143998588064605)

rho_1000 = 'Temp:  1000.0 rho:  0.015571402510561605 Area:  4.18207657802589e-37 Icl:  6.247142884431023e-46 skip:  2000003 nstart:  600000 Beads:  8 sim:  1'
rho_1000 = 'Temp:  1000.0 rho:  0.013332915707962172 Area:  3.543351933732888e-37 Icl:  6.1816775207581265e-46 skip:  2000003 nstart:  600000 Beads:  8 sim:  2'
rho_1000 = 'Temp:  1000.0 rho:  0.011545488246317041 Area:  3.2368223748083356e-37 Icl:  6.521143444650708e-46 skip:  2000003 nstart:  600000 Beads:  8 sim:  3'
mean, std = (0.013483266903333334, 0.00164700777303381)

rho_1300 = 'Temp:  1300.0 rho:  0.006275741528338499 Area:  2.0358591335059846e-37 Icl:  9.809417412053749e-46 skip:  2000003 nstart:  600000 Beads:  8 sim:  1'
rho_1300 = 'Temp:  1300.0 rho:  0.008796287100576238 Area:  2.4452371601448082e-37 Icl:  8.405859568094448e-46 skip:  2000003 nstart:  600000 Beads:  8 sim:  2'
rho_1300 = 'Temp:  1300.0 rho:  0.006887965478029861 Area:  1.965573594528573e-37 Icl:  8.628968553529672e-46 skip:  2000003 nstart:  600000 Beads:  8 sim:  3'
mean, std = (0.007319998002099999, 0.0010733984799495499)
# 10 Excitons in anharmonic Potential
rho_5_1 = 'Temp:  5.17 rho:  0.9990297493041157 Area:  9.120701376879302e-34 Icl:  1.3174163754411432e-45 skip:  10000008 nstart:  20000 Beads:  96 sim:  1'
rho_5_2 = 'Temp:  5.17 rho:  1.0178589271793954 Area:  9.308534466990203e-34 Icl:  1.3196749192016981e-45 skip:  10000008 nstart:  20000 Beads:  96 sim:  2'
rho_5_3 = 'Temp:  5.17 rho:  1.0480468822724547 Area:  9.567553532786998e-34 Icl:  1.3173265116105193e-45 skip:  10000008 nstart:  20000 Beads:  96 sim:  3'
mean, std = (1.0216451557, 0.020189431030772757)

rho_51_1 = 'Temp:  51.7 rho:  0.9828409789752152 Area:  1.5274844711367536e-34 Icl:  1.49511820834426e-45 skip:  10000008 nstart:  20000 Beads:  64 sim:  1'
rho_51_2 = 'Temp:  51.7 rho:  0.9665133863251797 Area:  1.5047572894224627e-34 Icl:  1.4977542653244808e-45 skip:  10000008 nstart:  20000 Beads:  64 sim:  2'
rho_51_3 = 'Temp:  51.7 rho:  1.001662959837028 Area:  1.553090946877557e-34 Icl:  1.491616766437912e-45 skip:  10000008 nstart:  20000 Beads:  64 sim:  3'


rho_100 = 'Temp:  103.4 rho:  0.9771582114960419 Area:  3.821217214121608e-35 Icl:  3.762000268683818e-46 skip:  200003 nstart:  20000 Beads:  32 sim:  1'
rho_100 = 'Temp:  103.4 rho:  0.9505570827477549 Area:  3.712679655342369e-35 Icl:  3.757433115809898e-46 skip:  200003 nstart:  20000 Beads:  32 sim:  2'
rho_100 = 'Temp:  103.4 rho:  0.9620611200915848 Area:  3.7547650312201307e-35 Icl:  3.754586233218436e-46 skip:  200003 nstart:  20000 Beads:  32 sim:  3'
rho_100 = 'Temp:  103.4 rho:  0.9544062880534315 Area:  3.719318999594419e-35 Icl:  3.748971330947051e-46 skip:  200003 nstart:  20000 Beads:  32 sim:  4'
rho_100 = 'Temp:  103.4 rho:  0.9711123815766395 Area:  3.7949207859890087e-35 Icl:  3.7593711675934156e-46 skip:  200003 nstart:  20000 Beads:  32 sim:  5'
mean, std  = (0.9630589816105999, 0.009957994517576983)


rho_155 = 'Temp:  155.09 rho:  0.8993539476162927 Area:  2.308227311561547e-35 Icl:  1.851787619343888e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  1'
rho_155 = 'Temp:  155.09 rho:  0.8956948861224796 Area:  2.338217143567726e-35 Icl:  1.883510266545322e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  2'
rho_155 = 'Temp:  155.09 rho:  0.8693779207151456 Area:  2.2325602162198694e-35 Icl:  1.852839587035423e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  3'
rho_155 = 'Temp:  155.09 rho:  0.858657710375375 Area:  2.2034750568032803e-35 Icl:  1.8515323765745674e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  4'
rho_155 = 'Temp:  155.09 rho:  0.8773966083904892 Area:  2.2731123640495767e-35 Icl:  1.869253484846112e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  5'
mean, std = (0.88009603, 0.01546564297488597)


rho_200 = 'Temp:  206.79 rho:  0.8146234523866988 Area:  1.6619090878769084e-35 Icl:  1.9626024489352338e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  1'
rho_200 = 'Temp:  206.79 rho:  0.778912224995877 Area:  1.605402498617494e-35 Icl:  1.9827930916946414e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  2'
rho_200 = 'Temp:  206.79 rho:  0.8531510762907468 Area:  1.7590945188191025e-35 Icl:  1.983559420950028e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  3'
rho_200 = 'Temp:  206.79 rho:  0.7851781007747213 Area:  1.6004871135437643e-35 Icl:  1.9609475882666127e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  4'
rho_200 = 'Temp:  206.79 rho:  0.8262510463458765 Area:  1.6962377510626538e-35 Icl:  1.9749526361388094e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  5'
mean, std = (0.8116231801585643, 0.027264963125322325)

rho_300 = 'Temp:  310.19 rho:  0.5848078467410156 Area:  9.330823990907877e-36 Icl:  2.302396136558929e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  1'
rho_300 = 'Temp:  310.19 rho:  0.5847462952254029 Area:  9.156534044408903e-36 Icl:  2.259627633859373e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  2'
rho_300 = 'Temp:  310.19 rho:  0.6159316546928401 Area:  9.684956019160278e-36 Icl:  2.2690202413511724e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  3'
rho_300 = 'Temp:  310.19 rho:  0.6156589255378682 Area:  9.675549597181295e-36 Icl:  2.2678206480592417e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  4'
rho_300 = 'Temp:  310.19 rho:  0.5735425869852933 Area:  9.259722538326895e-36 Icl:  2.3297297435600488e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  5'
mean, std = (0.5949374616916031, 0.01751767571346139)

rho_400 = 'Temp:  413.58 rho:  0.4604091158421302 Area:  5.8952563109461633e-36 Icl:  1.847702386547005e-46 skip:  200003 nstart:  20000 Beads:  12 sim:  1'
rho_400 = 'Temp:  413.58 rho:  0.36516523567149334 Area:  4.98226277028199e-36 Icl:  1.9688400964814544e-46 skip:  200003 nstart:  20000 Beads:  12 sim:  2'
rho_400 = 'Temp:  413.58 rho:  0.4538720957078236 Area:  6.00498331646871e-36 Icl:  1.909200607774622e-46 skip:  200003 nstart:  20000 Beads:  12 sim:  3'
rho_400 = 'Temp:  413.58 rho:  0.45374782976543976 Area:  6.10627221813479e-36 Icl:  1.9419356837891665e-46 skip:  200003 nstart:  20000 Beads:  12 sim:  4'
rho_400 = 'Temp:  413.58 rho:  0.4566446403244979 Area:  6.181253966686594e-36 Icl:  1.953311309896851e-46 skip:  200003 nstart:  20000 Beads:  12 sim:  5'
mean, std = (0.43796775416800005, 0.036481780642616014)
rho_400 = 'Temp:  413.58 rho:  0.4223561015980694 Area:  1.1429928086605966e-35 Icl:  5.206869626889677e-46 skip:  2000003 nstart:  500000 Beads:  16 sim:  1'
rho_400 = 'Temp:  413.58 rho:  0.4380511570462232 Area:  1.1918585192582852e-35 Icl:  5.234941726538893e-46 skip:  2000003 nstart:  500000 Beads:  16 sim:  2'
rho_400 = 'Temp:  413.58 rho:  0.4461944988081504 Area:  1.176483693980858e-35 Icl:  5.073102989581413e-46 skip:  2000003 nstart:  500000 Beads:  16 sim:  3'
rho_400 = 'Temp:  413.58 rho:  0.4227089324185549 Area:  1.154473545379386e-35 Icl:  5.254780005165307e-46 skip:  2000003 nstart:  500000 Beads:  16 sim:  4'
rho_400 = 'Temp:  413.58 rho:  0.4315659961832725 Area:  1.1706767207182855e-35 Icl:  5.219173594347278e-46 skip:  2000003 nstart:  500000 Beads:  16 sim:  5'
mean, std = (0.43217533720909396, 0.00913743568634919)

rho_600 ='Temp:  600.0 rho:  0.28062668657020934 Area:  3.234852409466455e-36 Icl:  2.4131549365372978e-46 skip:  2000003 nstart:  600000 Beads:  12 sim:  1'
rho_600 = 'Temp:  600.0 rho:  0.19429987011028513 Area:  4.841920517713796e-36 Icl:  5.21680796373216e-46 skip:  2000003 nstart:  500000 Beads:  12 sim:  2'
rho_600 = 'Temp:  600.0 rho:  0.28776888607865203 Area:  6.242315501739162e-36 Icl:  4.541105419067977e-46 skip:  2000003 nstart:  500000 Beads:  12 sim:  3'
rho_600 = 'Temp:  600.0 rho:  0.21401863637078297 Area:  5.1491721473468633e-36 Icl:  5.036693498590564e-46 skip:  2000003 nstart:  500000 Beads:  12 sim:  4'
rho_600 = 'Temp:  600.0 rho:  0.19860604780161514 Area:  4.733259825272294e-36 Icl:  4.989161706465645e-46 skip:  2000003 nstart:  500000 Beads:  12 sim:  5'
mean, std = (0.235064025386287, 0.0407125713226526)

rho_1000 = 'Temp:  1000.0 rho:  0.05203630370160959 Area:  5.483893291843773e-37 Icl:  2.451318513134826e-46 skip:  2000003 nstart:  400000 Beads:  8 sim:  1'
rho_1000 = 'Temp:  1000.0 rho:  0.05542048545780328 Area:  5.9928907268581896e-37 Icl:  2.515261959397909e-46 skip:  2000003 nstart:  400000 Beads:  8 sim:  2'
rho_1000 = 'Temp:  1000.0 rho:  0.04502356084621047 Area:  4.897950356363735e-37 Icl:  2.5304148559254303e-46 skip:  2000003 nstart:  400000 Beads:  8 sim:  3'
rho_1000 = 'Temp:  1000.0 rho:  0.06281642071935248 Area:  6.465181554776407e-37 Icl:  2.394003207463207e-46 skip:  2000003 nstart:  400000 Beads:  8 sim:  4'
rho_1000 = 'Temp:  1000.0 rho:  0.04788861393899072 Area:  5.10474107245847e-37 Icl:  2.4794687579071926e-46 skip:  2000003 nstart:  400000 Beads:  8 sim:  5'
mean, std = (0.052637076606, 0.006200658955744534)

rho_1300 = 'Temp:  1300.0 rho:  0.02857338532010762 Area:  5.2298676693010525e-37 Icl:  5.534642085952961e-46 skip:  2000003 nstart:  500000 Beads:  8 sim:  1'
rho_1300 = 'Temp:  1300.0 rho:  0.02445343479637494 Area:  4.839622342031134e-37 Icl:  5.984558848513195e-46 skip:  2000003 nstart:  500000 Beads:  8 sim:  2'
rho_1300 = 'Temp:  1300.0 rho:  0.025612055349655913 Area:  4.856911727704995e-37 Icl:  5.734245964486917e-46 skip:  2000003 nstart:  500000 Beads:  8 sim:  3'
rho_1300 = 'Temp:  1300.0 rho:  0.029224285600388915 Area:  5.7493050257176615e-37 Icl:  5.9488358579028915e-46 skip:  2000003 nstart:  500000 Beads:  8 sim:  4'
rho_1300 = 'Temp:  1300.0 rho:  0.028959589940459503 Area:  5.381562713547719e-37 Icl:  5.619226479946809e-46 skip:  2000003 nstart:  500000 Beads:  8 sim:  5'
mean, std = (0.027364550201356197, 0.0019498656863535745)

################################ an harmonic 34 meV#############3
rho_51 = 'Temp:  51.7 rho:  0.9938693439693225 Area:  1.156768300401917e-34 Icl:  1.1196932667765647e-45 skip:  1200000 nstart:  10000 Beads:  64 sim:  1'
rho_51 = 'Temp:  51.7 rho:  1.0053052702457819 Area:  1.1688235953327105e-34 Icl:  1.1184922869109821e-45 skip:  1200000 nstart:  10000 Beads:  64 sim:  2'
rho_51 = 'Temp:  51.7 rho:  1.015757520726675 Area:  1.1793830455858955e-34 Icl:  1.1169836505213314e-45 skip:  1200000 nstart:  10000 Beads:  64 sim:  3'
rho_51 = 'Temp:  51.7 rho:  1.010725741262498 Area:  1.1737159780198609e-34 Icl:  1.1171504708951423e-45 skip:  1200000 nstart:  10000 Beads:  64 sim:  4'
rho_51 = 'Temp:  51.7 rho:  1.0231050782932598 Area:  1.1952564908823642e-34 Icl:  1.123887531499483e-45 skip:  1200000 nstart:  10000 Beads:  64 sim:  5'
mean, std = (1.0097525898400002, 0.00987270407839718)

rho_100 = 'Temp:  103.4 rho:  0.9808513777809067 Area:  5.698358026971531e-35 Icl:  5.58892792030039e-46 skip:  200003 nstart:  20000 Beads:  32 sim:  1'
rho_100 = 'Temp:  103.4 rho:  0.9877534763649738 Area:  5.749120465380624e-35 Icl:  5.5993140253035175e-46 skip:  200003 nstart:  20000 Beads:  32 sim:  2'
rho_100 = 'Temp:  103.4 rho:  0.9879644835548748 Area:  5.748561852567027e-35 Icl:  5.597574195963248e-46 skip:  200003 nstart:  20000 Beads:  32 sim:  3'
rho_100 = 'Temp:  103.4 rho:  0.9874058872959947 Area:  5.747790860220239e-35 Icl:  5.599989694590097e-46 skip:  200003 nstart:  20000 Beads:  32 sim:  4'
rho_100 = 'Temp:  103.4 rho:  1.0022402648410438 Area:  5.845278494621892e-35 Icl:  5.610677985220463e-46 skip:  200003 nstart:  20000 Beads:  32 sim:  5'
mean, std = (0.9892430960399998, 0.007022486687808947)

rho_155 = 'Temp:  155.09 rho:  0.9473563271763683 Area:  3.607365961347749e-35 Icl:  2.747388582226867e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  1'
rho_155 = 'Temp:  155.09 rho:  0.906683167321637 Area:  3.398153295034896e-35 Icl:  2.704149259936592e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  2'
rho_155 = 'Temp:  155.09 rho:  0.9522192658835755 Area:  3.5837744162717413e-35 Icl:  2.7154821148919796e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  3'
rho_155 = 'Temp:  155.09 rho:  0.960335949934403 Area:  3.5974115572566584e-35 Icl:  2.7027768146404546e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  4'
rho_155 = 'Temp:  155.09 rho:  0.9973630406818081 Area:  3.721916140274158e-35 Icl:  2.69250525262382e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  5'
mean, std = (0.9527915501988999, 0.02898064133697687)

rho_200 = 'Temp:  206.79 rho:  0.9106613627278319 Area:  2.6967683577606185e-35 Icl:  2.8488443100633506e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  1'
rho_200 = 'Temp:  206.79 rho:  0.9017198126542735 Area:  2.684905080205767e-35 Icl:  2.8644372129197805e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  2'
rho_200 = 'Temp:  206.79 rho:  0.9405413943099865 Area:  2.810399630843857e-35 Icl:  2.8745650984660493e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  3'
rho_200 = 'Temp:  206.79 rho:  0.8876577665721623 Area:  2.6441735239310544e-35 Icl:  2.8656713230511745e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  4'
rho_200 = 'Temp:  206.79 rho:  0.8458842831124529 Area:  2.5327903299493507e-35 Icl:  2.8805158310169347e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  5'
mean, std = (0.897292923738432, 0.030998152500784113)

rho_300 = 'Temp:  310.19 rho:  0.7599731072577808 Area:  1.686854629517533e-35 Icl:  3.2029685525082274e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  1'
rho_300 = 'Temp:  310.19 rho:  0.7908351155796817 Area:  1.7285933963172444e-35 Icl:  3.1541340451880094e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  2'
rho_300 = 'Temp:  310.19 rho:  0.7350231811902566 Area:  1.6188166298921628e-35 Icl:  3.178116864007156e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  3'
rho_300 = 'Temp:  310.19 rho:  0.7070908962074223 Area:  1.5671671505571426e-35 Icl:  3.1982566811010264e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  4'
rho_300 = 'Temp:  310.19 rho:  0.6797515146390751 Area:  1.491899929141139e-35 Icl:  3.1671071087263598e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  5'
mean, std = (0.7345345464, 0.03891400051731513)

rho_400 = 'Temp:  413.58 rho:  0.5924223076639747 Area:  1.0768153986557205e-35 Icl:  3.49721097169549e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  1'
rho_400 = 'Temp:  413.58 rho:  0.5915274404690409 Area:  1.0769926217980521e-35 Icl:  3.503078023477905e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  2'
rho_400 = 'Temp:  413.58 rho:  0.5764915613802833 Area:  1.0457024296507362e-35 Icl:  3.4900137731733147e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  3'
rho_400 = 'Temp:  413.58 rho:  0.5960304826236512 Area:  1.0983065409108001e-35 Icl:  3.545414993874481e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  4'
rho_400 = 'Temp:  413.58 rho:  0.5872289683913438 Area:  1.0988235859635767e-35 Icl:  3.600248513289162e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  5'
mean, std =  (0.5887400214, 0.006734988962651077)

rho_600 = 'Temp:  600.0 rho:  0.3826142387601547 Area:  5.686432985564921e-36 Icl:  3.1112757340992734e-46 skip:  200003 nstart:  20000 Beads:  12 sim:  1'
rho_600 = 'Temp:  600.0 rho:  0.2984123545138162 Area:  4.552985398746873e-36 Icl:  3.1940312628094302e-46 skip:  200003 nstart:  20000 Beads:  12 sim:  2'
rho_600 = 'Temp:  600.0 rho:  0.41309259239530305 Area:  6.185071446246962e-36 Icl:  3.1344187592485382e-46 skip:  200003 nstart:  20000 Beads:  12 sim:  3'
rho_600 = 'Temp:  600.0 rho:  0.4908044179875727 Area:  6.698584948974328e-36 Icl:  2.8571583482540242e-46 skip:  200003 nstart:  20000 Beads:  12 sim:  4'
rho_600 = 'Temp:  600.0 rho:  0.38723302594763215 Area:  6.116151149484129e-36 Icl:  3.3064771379036986e-46 skip:  200003 nstart:  20000 Beads:  12 sim:  5'
mean, std = (0.39443035967600004, 0.06172065418949585)

rho_1000 = 'Temp:  1000.0 rho:  0.09659866513273918 Area:  1.2863995231274487e-36 Icl:  3.0975757431271502e-46 skip:  200003 nstart:  20000 Beads:  8 sim:  1'
rho_1000 = 'Temp:  1000.0 rho:  0.10224440410465167 Area:  1.305939273063395e-36 Icl:  2.970986125822925e-46 skip:  200003 nstart:  20000 Beads:  8 sim:  2'
rho_1000 = 'Temp:  1000.0 rho:  0.10199378595369223 Area:  1.3717234362109148e-36 Icl:  3.128311817336081e-46 skip:  200003 nstart:  20000 Beads:  8 sim:  3'
rho_1000 = 'Temp:  1000.0 rho:  0.11880202507302298 Area:  1.4741527052978813e-36 Icl:  2.8862624519025544e-46 skip:  200003 nstart:  20000 Beads:  8 sim:  4'
rho_1000 = 'Temp:  1000.0 rho:  0.09194732625096771 Area:  1.1905292720311897e-36 Icl:  3.011744754284885e-46 skip:  200003 nstart:  20000 Beads:  8 sim:  5'
mean, std = (0.102317233, 0.009078498920755346)

rho_1300 = 'Temp:  1300.0 rho:  0.05053168446950576 Area:  6.257487963659247e-37 Icl:  3.7445254456447416e-46 skip:  200003 nstart:  20000 Beads:  8 sim:  1'
rho_1300 = 'Temp:  1300.0 rho:  0.07512306848597951 Area:  8.56614057267242e-37 Icl:  3.4480409661271096e-46 skip:  200003 nstart:  20000 Beads:  8 sim:  2'
rho_1300 = 'Temp:  1300.0 rho:  0.04450853062527413 Area:  5.718639064472549e-37 Icl:  3.885169480388485e-46 skip:  200003 nstart:  20000 Beads:  8 sim:  3'
rho_1300 = 'Temp:  1300.0 rho:  0.058873008270845635 Area:  7.0291824521220875e-37 Icl:  3.610348514351344e-46 skip:  200003 nstart:  20000 Beads:  8 sim:  4'
rho_1300 = 'Temp:  1300.0 rho:  0.049810497875237204 Area:  6.176354597304793e-37 Icl:  3.7494872227597566e-46 skip:  200003 nstart:  20000 Beads:  8 sim:  5'
mean, std = (0.055769357789999995, 0.010713974437257143)
# ############################### an harmonic 18 meV #############

rho_51 = 'Temp:  51.7 rho:  0.9963787926743194 Area:  2.2756578713979757e-34 Icl:  2.1971741305271672e-45 skip:  1200000 nstart:  10000 Beads:  64 sim:  1'
rho_51 = 'Temp:  51.7 rho:  0.9862705689362717 Area:  2.2449243688640163e-34 Icl:  2.189715150783574e-45 skip:  1200000 nstart:  10000 Beads:  64 sim:  2'
rho_51 = 'Temp:  51.7 rho:  0.980858646004021 Area:  2.2387961948960544e-34 Icl:  2.1957865378261355e-45 skip:  1200000 nstart:  10000 Beads:  64 sim:  3'
rho_51 = 'Temp:  51.7 rho:  0.9866716372666711 Area:  2.2420037864710005e-34 Icl:  2.185977463157936e-45 skip:  1200000 nstart:  10000 Beads:  64 sim:  4'
rho_51 = 'Temp:  51.7 rho:  0.9970224900398906 Area:  2.286146712226954e-34 Icl:  2.2058761511185826e-45 skip:  1200000 nstart:  10000 Beads:  64 sim:  5'
mean, std = (0.9894404255, 0.006276779410948368)

rho_100 = 'Temp:  103.4 rho:  0.8919602164016442 Area:  1.0459783359654648e-34 Icl:  1.1281301029606167e-45 skip:  200003 nstart:  20000 Beads:  32 sim:  1'
rho_100 = 'Temp:  103.4 rho:  0.9231966109263074 Area:  1.078771236886599e-34 Icl:  1.1241315567613368e-45 skip:  200003 nstart:  20000 Beads:  32 sim:  2'
rho_100 = 'Temp:  103.4 rho:  0.9142911432457927 Area:  1.065933857601164e-34 Icl:  1.1215734677392786e-45 skip:  200003 nstart:  20000 Beads:  32 sim:  3'
rho_100 = 'Temp:  103.4 rho:  0.9282085476760754 Area:  1.083609390405687e-34 Icl:  1.1230760843496818e-45 skip:  200003 nstart:  20000 Beads:  32 sim:  4'
rho_100 = 'Temp:  103.4 rho:  0.888378884009808 Area:  1.0376923263102319e-34 Icl:  1.1237051224491984e-45 skip:  200003 nstart:  20000 Beads:  32 sim:  5'
mean, std = (0.90920706728, 0.016210345568226467)

rho_155 = 'Temp:  155.09 rho:  0.7809900448782988 Area:  6.283967407830432e-35 Icl:  5.805393627614168e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  1'
rho_155 = 'Temp:  155.09 rho:  0.8478876623176842 Area:  6.725547754283243e-35 Icl:  5.723116557878237e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  2'
rho_155 = 'Temp:  155.09 rho:  0.7823307821069247 Area:  6.256554042091652e-35 Icl:  5.7701622828916314e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  3'
rho_155 = 'Temp:  155.09 rho:  0.7900430509082813 Area:  6.29369728684543e-35 Icl:  5.747756206401897e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  4'
rho_155 = 'Temp:  155.09 rho:  0.7417711112473943 Area:  5.972041651042358e-35 Icl:  5.808930207376975e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  5'
mean, std = (0.7886045292434, 0.034080107472987294)

rho_200 = 'Temp:  206.79 rho:  0.6538742090031362 Area:  4.3230760545096015e-35 Icl:  6.3603446713616834e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  1'
rho_200 = 'Temp:  206.79 rho:  0.6573224312662871 Area:  4.2913792811895896e-35 Icl:  6.280589804125508e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  2'
rho_200 = 'Temp:  206.79 rho:  0.641582291287347 Area:  4.2859843778043603e-35 Icl:  6.426584143898993e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  3'
rho_200 = 'Temp:  206.79 rho:  0.6170124619215452 Area:  4.1077034286401046e-35 Icl:  6.404527982647475e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  4'
rho_200 = 'Temp:  206.79 rho:  0.6079083809498418 Area:  4.0690121703239455e-35 Icl:  6.4392137333697275e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  5'
mean, std = (0.63553827866, 0.019770513725322906)

rho_300 = 'Temp:  310.19 rho:  0.4454959952244131 Area:  2.2965582118463195e-35 Icl:  7.438868739342964e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  1'
rho_300 = 'Temp:  310.19 rho:  0.40016279057303233 Area:  2.1368790158989223e-35 Icl:  7.705777645436642e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  2'
rho_300 = 'Temp:  310.19 rho:  0.37078515698556536 Area:  1.9433150167022956e-35 Icl:  7.563000130719432e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  3'
rho_300 = 'Temp:  310.19 rho:  0.3532659767259087 Area:  1.8812908241879437e-35 Icl:  7.684707941413765e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  4'
rho_300 = 'Temp:  310.19 rho:  0.4154915379895956 Area:  2.187602511237752e-35 Icl:  7.59765329969131e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  5'
mean,std = (0.3970402904358, 0.032568929800917704)

rho_400 = 'Temp:  413.58 rho:  0.23516190273830279 Area:  1.1229383839377195e-35 Icl:  9.187575899163056e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  1'
rho_400 = 'Temp:  413.58 rho:  0.2314256706364013 Area:  1.1052367012960867e-35 Icl:  9.188735393113167e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  2'
rho_400 = 'Temp:  413.58 rho:  0.25577182687756234 Area:  1.211013191659527e-35 Icl:  9.10978540894919e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  3'
rho_400 = 'Temp:  413.58 rho:  0.21636046136944456 Area:  1.0511161715614444e-35 Icl:  9.347270206928515e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  4'
rho_400 = 'Temp:  413.58 rho:  0.23697123903545023 Area:  1.1472759964053167e-35 Icl:  9.31502974993945e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  5'
mean, std = (0.23513822012263003, 0.012613433270594014)

rho_600 = 'Temp:  600.0 rho:  0.08574729961891345 Area:  3.8085127716399375e-36 Icl:  9.298109764621706e-46 skip:  200003 nstart:  20000 Beads:  12 sim:  1'
rho_600 = 'Temp:  600.0 rho:  0.07338048613636548 Area:  3.3106195399870685e-36 Icl:  9.444704920421267e-46 skip:  200003 nstart:  20000 Beads:  12 sim:  2'
rho_600 = 'Temp:  600.0 rho:  0.08294916851317734 Area:  3.753734580927398e-36 Icl:  9.473516780105095e-46 skip:  200003 nstart:  20000 Beads:  12 sim:  3'
rho_600 = 'Temp:  600.0 rho:  0.07918745976849008 Area:  3.495552352346954e-36 Icl:  9.241002839970162e-46 skip:  200003 nstart:  20000 Beads:  12 sim:  4'
rho_600 = 'Temp:  600.0 rho:  0.11251266071379508 Area:  4.766615895833737e-36 Icl:  8.86887361996697e-46 skip:  200003 nstart:  20000 Beads:  12 sim:  5'
mean, std =  (0.0867554128446, 0.01352850486085638)

rho_1000 = 'Temp:  1000.0 rho:  0.023670605112113993 Area:  8.80270252302759e-37 Icl:  8.650154305819502e-46 skip:  200003 nstart:  20000 Beads:  8 sim:  1'
rho_1000 = 'Temp:  1000.0 rho:  0.023672284488460642 Area:  9.023823624071802e-37 Icl:  8.866814363340315e-46 skip:  200003 nstart:  20000 Beads:  8 sim:  2'
rho_1000 = 'Temp:  1000.0 rho:  0.02956200018022446 Area:  1.0432319722863218e-36 Icl:  8.208508696529148e-46 skip:  200003 nstart:  20000 Beads:  8 sim:  3'
rho_1000 = 'Temp:  1000.0 rho:  0.027338263425569394 Area:  1.0255543549550792e-36 Icl:  8.725793853519837e-46 skip:  200003 nstart:  20000 Beads:  8 sim:  4'
rho_1000 = 'Temp:  1000.0 rho:  0.025226754168667428 Area:  9.70487268836529e-37 Icl:  8.94840551603678e-46 skip:  200003 nstart:  20000 Beads:  8 sim:  5'
mean, stdd = (0.025893981413599998, 0.0022744160945527554)

rho_1300 = 'Temp:  1300.0 rho:  0.014111437467977713 Area:  4.8679238544658395e-37 Icl:  1.0431170870170968e-45 skip:  200003 nstart:  20000 Beads:  8 sim:  1'
rho_1300 = 'Temp:  1300.0 rho:  0.014131904139176093 Area:  4.804139138570735e-37 Icl:  1.0279581476063786e-45 skip:  200003 nstart:  20000 Beads:  8 sim:  2'
rho_1300 = 'Temp:  1300.0 rho:  0.017250332448504907 Area:  5.731812056257412e-37 Icl:  1.0047430511978958e-45 skip:  200003 nstart:  20000 Beads:  8 sim:  3'
rho_1300 = 'Temp:  1300.0 rho:  0.013830727816968773 Area:  4.5619016780159765e-37 Icl:  9.973817676840294e-46 skip:  200003 nstart:  20000 Beads:  8 sim:  4'
rho_1300 = 'Temp:  1300.0 rho:  0.014715276466711504 Area:  4.8624310918962895e-37 Icl:  9.991842339561221e-46 skip:  200003 nstart:  20000 Beads:  8 sim:  5'
mean, std  = (0.014807937912, 0.0012546653753400022)












