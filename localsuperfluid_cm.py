# This code is the same as localsuperluid.py but here the Area estimator and I moment of inertia is calculated from center of mass of all beads
# Imports
import networkx as nx
from tqdm import tnrange, tqdm, tqdm_notebook
from IPython.display import Image
from numpy import linalg as LA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Iterable
from functions_PIMD import center_of_mass

#Functions:


def remove_nan(index, natoms, skp):
    if index > skp:
        if natoms == 3:  # for N = 3
            if index % 5 == 0 or index % 5 == 1:  # 5 for 3 particles
                return True
        elif natoms == 2:
            if index % 4 == 0 or index % 4 == 1:  # 6 for 4 particles
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
    ######################## Change Axis
    xmid, ymid = 179.524, 0  # [bohr] Center of Moire lattice unit (in y axis it is already centered at zero)
    beads2 = beads2 - [xmid, ymid, 0]
    beads = beads - [xmid, ymid, 0]  ## PERIODIC_NEW  +  or - ??!!
    #######################################
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
        min_beadsx = beads_minimum_image_rhom_x(gc, beads, t)   ## PERIODIC_NEW
        min_beads = beads_minimum_image_rhom_y(gc, min_beadsx, t)   ## PERIODIC_NEW
        A_2, I_cl = superfluid_area(gc, min_beads, t)  # beads is to calculate the area (all files)
        A2 += A_2
        Icl += I_cl
        A_est = np.append(A_est, sum(A_2))  #### ADDED
    print("A2:", A2, "Icl: ", Icl)
    Az = sum(A2) / count_avg
    Icl_m = Icl / count_avg
    return Az, Icl_m, A_est, count_avg


def superfluid_area(gconnected, m_beads, t):
    Atmp = np.zeros(3)
    Itmp = 0
    xcm, ycm = center_of_mass(beads, t, natoms)  # center of mass of all beads in time step
    r_beads_cm = beads[:, t] - [xcm, ycm, 0]  # coord change to COM only of the permutated beads IF CM THEN TAKE the[t] from I_TMP!
    # r_beads_cm = beads[:, t] - [0, 0, 0]  # coord change to COM only of the permutated beads IF CM THEN TAKE the[t] from I_TMP!
    # r_beads_cm = m_beads[:, t]
    # r_beads_cm = beads
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
                        # Atmp += np.cross(beads[j][t][g[i]], beads[j+1][t][g[i]])  # from bead 1  to bead P
                        Atmp += np.cross(r_beads_cm[j][g[i]], r_beads_cm[j + 1][g[i]])  # from bead 1  to bead P
                        Itmp += np.dot(r_beads_cm[j][g[i]], r_beads_cm[j+1][g[i]])
                    else:
                        # Atmp += np.cross(beads[-1][t][g[i]], beads[0][t][g[i+1]])
                        Atmp += np.cross(r_beads_cm[-1][g[i]], r_beads_cm[0][g[i + 1]])
                        Itmp += np.dot(r_beads_cm[-1][g[i]], r_beads_cm[0][g[i+1]])
            else:
                for j in range(M):
                    if j != M-1:
                        # Atmp += np.cross(beads[j][t][g[i]], beads[j+1][t][g[i]])
                        Atmp += np.cross(r_beads_cm[j][g[i]], r_beads_cm[j + 1][g[i]])
                        Itmp += np.dot(r_beads_cm[j][g[i]], r_beads_cm[j+1][g[i]])
                    else:
                        # Atmp += np.cross(beads[-1][t][g[i]], beads[0][t][g[0]])
                        Atmp += np.cross(r_beads_cm[-1][g[i]], r_beads_cm[0][g[0]])
                        Itmp += np.dot(r_beads_cm[-1][g[i]], r_beads_cm[0][g[0]])

    Atmp = (1/2)*Atmp * conv_m_to_bohr**2
    A_squared = Atmp**2
    Itmp = m * Itmp * conv_m_to_bohr**2 # Itm = Itm bad use of words so i update it
    # Itmp = Itmp + m*(np.sqrt(xmid**2+ymid**2)) * conv_m_to_bohr**2  ## Steiner! parallel axis theorem
    return A_squared, Itmp


def permutations_chang(beads2, t):
    ring_dist = []
    pair = []
    for i in range(natoms):  # natoms
        j = periodic_boundary_rhom(beads2, t, i)  ## PERIODIC_NEW
        # delR = beads2[0, t, :, :] - beads2[-1, t, i, :]    # ensures file 1 and file P
        # j = LA.norm(delR, axis=-1).argmin()

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
##################adds the single configurations#####################################
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

def periodic_boundary(beads2, t, i):
    '''
    This is a must to take into account that the lattice is periodic and permutations can be taken into account on adjacent
    lattice units
    :param L: lattice length 3D vector
    :param t: time step
    :param i: the i-th atom
    :return: the closest j-th atom to i-th
    '''
    latx, laty, latz = L[0], L[1], L[2]
    delW = beads2[0, t, :, :] - beads2[-1, t, i, :]
    # need to consider the PBC!
    # x
    dela = delW[:, 0]
    dela = np.where(dela < latx / 2, dela, dela - latx)
    dela = np.where(dela > -latx / 2, dela, dela + latx)
    delW[:, 0] = dela
    # y
    delb = delW[:, 1]
    delb = np.where(delb < laty / 2, delb, delb - laty)
    delb = np.where(delb > -laty / 2, delb, delb + laty)
    delW[:, 1] = delb
    # z
    delc = delW[:, 2]
    delc = np.where(delc < latz / 2, delc, delc - latz)
    delc = np.where(delc > -latz / 2, delc, delc + latz)
    delW[:, 2] = delc
    
    j = LA.norm(delW, axis=-1).argmin()

    return j

def beads_minimum_image(gc, beads, t):
    min_beads = beads
    for inx, g in enumerate(gc):
        g = list(g)
        l = len(g)
        M = len(min_beads)
        for i, n in enumerate(g):  # which permutation
            # is the first bead of the permutation right left top or bottom
            for q in range(2):  # 0 for x axis and 1 for y axis
                if min_beads[0][t][g[0]][q] < 0:  #first beads of the permutation is on the right and bottom to the central axis
                    if i != l - 1: # didnt reach the last quantum particle
                        for j in range(M):  # which bead
                            if j != M - 1:  # unitl bead P
                                #x  then [0]
                                if (abs(min_beads[j][t][g[i]][q] - min_beads[j + 1][t][g[i]][q]) <= L[q]/2):
                                    continue
                                else:
                                    min_beads[j+1][t][g[i]][q] = min_beads[j+1][t][g[i]][q] - L[q]
                            else:
                                if (abs(min_beads[-1][t][g[i]][q] - min_beads[0][t][g[i+1]][q]) <= L[q]/2):
                                    continue
                                else:
                                    min_beads[0][t][g[i+1]][q] = min_beads[0][t][g[i+1]][q] - L[q]
                    else:
                        for j in range(M):
                            if j != M - 1:  # unitl bead P
                                # x  then [0]
                                if (abs(min_beads[j][t][g[i]][q] - min_beads[j + 1][t][g[i]][q]) <= L[q] / 2):
                                    continue
                                else:
                                    min_beads[j + 1][t][g[i]][q] = min_beads[j + 1][t][g[i]][q] - L[q]
                            else:
                                if (abs(min_beads[-1][t][g[i]][q] - min_beads[0][t][g[0]][q]) <= L[q] / 2):
                                    continue
                                else:
                                    min_beads[0][t][g[i]][q] = min_beads[0][t][g[0]][q] - L[q]
                else:  # first bead of the permutation is on the left
                    if i != l - 1: # didnt reach the last quantum particle
                        for j in range(M):  # which bead
                            if j != M - 1:  # unitl bead P
                                #x  then [0]
                                if (abs(min_beads[j][t][g[i]][q] - min_beads[j + 1][t][g[i]][q]) <= L[q]/2):
                                    continue
                                else:
                                    min_beads[j+1][t][g[i]][q] = min_beads[j+1][t][g[i]][q] + L[q]
                            else:
                                if (abs(min_beads[-1][t][g[i]][q] - min_beads[0][t][g[i+1]][q]) <= L[q]/2):
                                    continue
                                else:
                                    min_beads[0][t][g[i+1]][q] = min_beads[0][t][g[i+1]][q] + L[q]
                    else:
                        for j in range(M):
                            if j != M - 1:  # unitl bead P
                                # x  then [0]
                                if (abs(min_beads[j][t][g[i]][q] - min_beads[j + 1][t][g[i]][q]) <= L[q] / 2):
                                    continue
                                else:
                                    min_beads[j+1][t][g[i]][q] = min_beads[j+1][t][g[i]][q] + L[q]
                            else:
                                if (abs(min_beads[-1][t][g[i]][q] - min_beads[0][t][g[0]][q]) <= L[q] / 2):
                                    continue
                                else:
                                    min_beads[0][t][g[i]][q] = min_beads[0][t][g[0]][q] + L[q]
    return min_beads


def periodic_boundary_rhom(beads2, t, i):
    '''
    This is a must to take into account that the lattice is periodic and permutations can be taken into account on adjacent
    lattice units of Rhombus
    :param L: lattice length 3D vector
    :param t: time step
    :param i: the i-th atom
    :return: the closest j-th atom to i-th
    '''
    latx, laty, latz, lattrin = L[0], L[1], L[2], L[3]
    delW = beads2[0, t, :, :] - beads2[-1, t, i, :]
    # need to consider the PBC!
    # x
    dela = delW[:, 0]
    dela = np.where(dela < latx / 2, dela, dela - latx)
    dela = np.where(dela > -latx / 2, dela, dela + latx)
    delW[:, 0] = dela
    # y
    delb = delW[:, 1]
    delb = np.where(delb < laty / 2, delb, delb - laty)
    dela = np.where(delb < laty / 2, dela, dela + lattrin) ## x axis trin
    delb = np.where(delb > -laty / 2, delb, delb + laty)
    dela = np.where(delb > -laty / 2, dela, dela - lattrin)  ## x axis trin
    delW[:, 1] = delb
    delW[:, 0] = dela
    # z
    delc = delW[:, 2]
    delc = np.where(delc < latz / 2, delc, delc - latz)
    delc = np.where(delc > -latz / 2, delc, delc + latz)
    delW[:, 2] = delc

    j = LA.norm(delW, axis=-1).argmin()

    return j


def beads_minimum_image_rhom_x(gc, beads, t):
    min_beads = beads
    for inx, g in enumerate(gc):
        g = list(g)
        l = len(g)
        M = len(min_beads)
        q=0  #x axis  0
        for i, n in enumerate(g):  # which permutation
            # is the first bead of the permutation right left top or bottom
            if min_beads[0][t][g[0]][q] < 0:  #first beads of the permutation is on the right and bottom to the central axis
                if i != l - 1: # didnt reach the last quantum particle
                    for j in range(M):  # which bead
                        if j != M - 1:  # unitl bead P
                            #x  then [0]
                            if (abs(min_beads[j][t][g[i]][q] - min_beads[j + 1][t][g[i]][q]) > L[q]/2):
                                min_beads[j+1][t][g[i]][q] = min_beads[j+1][t][g[i]][q] - L[q]
                        else:
                            if (abs(min_beads[-1][t][g[i]][q] - min_beads[0][t][g[i+1]][q]) > L[q]/2):
                                min_beads[0][t][g[i+1]][q] = min_beads[0][t][g[i+1]][q] - L[q]
                else:
                    for j in range(M):
                        if j != M - 1:  # unitl bead P
                            # x  then [0]
                            if (abs(min_beads[j][t][g[i]][q] - min_beads[j + 1][t][g[i]][q]) > L[q] / 2):
                                min_beads[j + 1][t][g[i]][q] = min_beads[j + 1][t][g[i]][q] - L[q]
                        else:
                            if (abs(min_beads[-1][t][g[i]][q] - min_beads[0][t][g[0]][q]) > L[q] / 2):
                                min_beads[0][t][g[i]][q] = min_beads[0][t][g[0]][q] - L[q]
            else:  # first bead of the permutation is on the left
                if i != l - 1: # didnt reach the last quantum particle
                    for j in range(M):  # which bead
                        if j != M - 1:  # unitl bead P
                            #x  then [0]
                            if (abs(min_beads[j][t][g[i]][q] - min_beads[j + 1][t][g[i]][q]) > L[q]/2):
                                min_beads[j+1][t][g[i]][q] = min_beads[j+1][t][g[i]][q] + L[q]
                        else:
                            if (abs(min_beads[-1][t][g[i]][q] - min_beads[0][t][g[i+1]][q]) > L[q]/2):
                                min_beads[0][t][g[i+1]][q] = min_beads[0][t][g[i+1]][q] + L[q]
                else:
                    for j in range(M):
                        if j != M - 1:  # unitl bead P
                            # x  then [0]
                            if (abs(min_beads[j][t][g[i]][q] - min_beads[j + 1][t][g[i]][q]) > L[q] / 2):
                                min_beads[j+1][t][g[i]][q] = min_beads[j+1][t][g[i]][q] + L[q]
                        else:
                            if (abs(min_beads[-1][t][g[i]][q] - min_beads[0][t][g[0]][q]) > L[q] / 2):
                                min_beads[0][t][g[i]][q] = min_beads[0][t][g[0]][q] + L[q]
    return min_beads

def beads_minimum_image_rhom_y(gc, beads, t):
    min_beads = beads
    for inx, g in enumerate(gc):
        g = list(g)
        l = len(g)
        M = len(min_beads)
        q=1  #x axis  0
        for i, n in enumerate(g):  # which permutation
            # is the first bead of the permutation right left top or bottom
            if min_beads[0][t][g[0]][q] < 0:  #first beads of the permutation is on the bottom to the central axis
                if i != l - 1: # didnt reach the last quantum particle
                    for j in range(M):  # which bead
                        if j != M - 1:  # unitl bead P
                            #x  then [0]
                            if (abs(min_beads[j][t][g[i]][q] - min_beads[j + 1][t][g[i]][q]) > L[q]/2):
                                min_beads[j+1][t][g[i]][q] = min_beads[j+1][t][g[i]][q] - L[q]
                                min_beads[j + 1][t][g[i]][0] = min_beads[j + 1][t][g[i]][0] - L[3]
                        else:
                            if (abs(min_beads[-1][t][g[i]][q] - min_beads[0][t][g[i+1]][q]) > L[q]/2):
                                min_beads[0][t][g[i+1]][q] = min_beads[0][t][g[i+1]][q] - L[q]
                                min_beads[0][t][g[i + 1]][0] = min_beads[0][t][g[i + 1]][0] - L[3]
                else:
                    for j in range(M):
                        if j != M - 1:  # unitl bead P
                            # x  then [0]
                            if (abs(min_beads[j][t][g[i]][q] - min_beads[j + 1][t][g[i]][q]) > L[q] / 2):
                                min_beads[j + 1][t][g[i]][q] = min_beads[j + 1][t][g[i]][q] - L[q]
                                min_beads[j + 1][t][g[i]][0] = min_beads[j + 1][t][g[i]][0] - L[3]  #
                        else:
                            if (abs(min_beads[-1][t][g[i]][q] - min_beads[0][t][g[0]][q]) > L[q] / 2):
                                min_beads[0][t][g[i]][q] = min_beads[0][t][g[0]][q] - L[q]
                                min_beads[0][t][g[i]][0] = min_beads[0][t][g[0]][0] - L[3]
            else:  # first bead of the permutation is on the left
                if i != l - 1: # didnt reach the last quantum particle
                    for j in range(M):  # which bead
                        if j != M - 1:  # unitl bead P
                            #x  then [0]
                            if (abs(min_beads[j][t][g[i]][q] - min_beads[j + 1][t][g[i]][q]) > L[q]/2):
                                min_beads[j+1][t][g[i]][q] = min_beads[j+1][t][g[i]][q] + L[q]
                                min_beads[j + 1][t][g[i]][0] = min_beads[j + 1][t][g[i]][0] + L[3]
                        else:
                            if (abs(min_beads[-1][t][g[i]][q] - min_beads[0][t][g[i+1]][q]) > L[q]/2):
                                min_beads[0][t][g[i+1]][q] = min_beads[0][t][g[i+1]][q] + L[q]
                                min_beads[0][t][g[i + 1]][0] = min_beads[0][t][g[i + 1]][0] + L[3]
                else:
                    for j in range(M):
                        if j != M - 1:  # unitl bead P
                            # x  then [0]
                            if (abs(min_beads[j][t][g[i]][q] - min_beads[j + 1][t][g[i]][q]) > L[q] / 2):
                                min_beads[j+1][t][g[i]][q] = min_beads[j+1][t][g[i]][q] + L[q]
                                min_beads[j + 1][t][g[i]][0] = min_beads[j + 1][t][g[i]][0] + L[3]
                        else:
                            if (abs(min_beads[-1][t][g[i]][q] - min_beads[0][t][g[0]][q]) > L[q] / 2):
                                min_beads[0][t][g[i]][q] = min_beads[0][t][g[0]][q] + L[q]
                                min_beads[0][t][g[i]][0] = min_beads[0][t][g[0]][0] + L[3]
    return min_beads




# Constants
conv_m_to_bohr = 5.29177249*10**-11
hbar = 1.05457182e-34                 # MKS
kb = 1.380649e-23                     # MKS
kB = 0.0083144621                     # Boltzmann const in kJ/mol/K
nbeads=2   #only read first and last bead
count = 0
winding_or_area = True   # True for Winding False for Area
files_range = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
first_bead = '01'


# Constant that change from simulation to simulation ###########################################3
L = [718.096, 310.9446892, 3000, 179.524] # for periodic potentials

# L = [3000, 3000, 3000]  for local potentials
m = 0.84*9.1093837e-31              # mass of exciton
# m = 9.1093837e-31                     # mass of electron
omega = 0.02673                     # [eV]
# omega = 0.003                         # [eV]
omega = omega/27.2114         # [Hartree]                     # only relevant for moire
omega_kJmol = omega/3.8088E-4  # [kJmol]               # only relevant for moire
# ######################################per simulation ###############################################
nstart = 0 # 500000 # 600000#20000 #80000  # (yes53000 51K)  (yes103K125468) #(5K10K31K115000)
last_bead = 64
natoms = 10 #  10 for BOSONS  Number of Bosons in simulation  4 for N4
bhw = 6 # 1/0.83819 (260K) #0.51698126 (600k),  0.31018876(1000K),  0.23860674(1300K)]
skp = 200003  #8000003, 10000008 # 1200000 #200003 #2000003


T = 1 / (kB*(bhw / omega_kJmol))
# T = (bhw * omega * 27.2114) / ((kB / 96.4853))  # for DORNHEIM SIMULATION # [K]
print("Temperature:", T)
p = last_bead

s = 1 # # sim1 sim2 sim3 sim4 sim5
# path_ = "/hirshblab-storage/netanelb2/PIMD/exitons/moire_one_b/boson10_1/bhw031/"
# path = path_ + '/sim{}_{}/'.format(s, 1)
#
path_ = '/hirshblab-storage/netanelb2/PIMD/exitons/moire_one_b/boson10_1/bhw6/'
path = path_ + '/sim{}/{}beads/'.format(s, p)


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


fig = plt.figure(figsize=(5, 5))
q = np.arange(count_t)
plt.plot(q, A_esti, '-', color="blue")
plt.title('Area^2 estimator vs time')
plt.xlabel("time [fs]")
plt.ylabel("Area^2 [bohr^2] ")
plt.legend(loc='lower right')
plt.show()

def area_cutoff(vector, A2_mean, count):
    cutoff = A2_mean * 50
    for i, v in enumerate(vector):
        if v > cutoff:
            for j in range(400):
                vector[i-200+j] = 0
    AA = sum(vector)/count
    return AA, vector

AA , vec = area_cutoff(A_esti, A2_mean, count_t)
rho = 4*m**2*(AA)/((1/(kb*T))*hbar**2*(Icl_mean/last_bead))
# Moire (after boundary conditions and minimum image)
xm = np.array([155.09, 206.79, 260.0 , 310.19, 600, ])
moire_N10 = [0.9430585, 0.814, 0.03958]
moire_N10_std = [0.0195684,  0.04150, 0.0131]



rho_T_100_cm_y = 'Temp:  103.4 rho:  1.0327339855855462 Area:  8.195997915669287e-35 Icl:  7.634759354443244e-46 skip:  200003 nstart:  0 Beads:  32 sim:  1'
rho_T_100_cm_y  ='Temp:  103.4 rho:  1.0442784484612595 Area:  8.310354894367905e-35 Icl:  7.655705844486322e-46 skip:  200003 nstart:  0 Beads:  32 sim:  2'
rho_T_100_cm_y = 'Temp:  103.4 rho:  1.0151540638822008 Area:  8.038952676194527e-35 Icl:  7.618149606910587e-46 skip:  200003 nstart:  0 Beads:  32 sim:  3'
rho_T_100_cm_y = 'Temp:  51.7 rho:  1.0444973477305646 Area:  1.69763269320702e-34 Icl:  1.5635737528084332e-45 skip:  200003 nstart:  0 Beads:  64 sim:  1'
mean, std =  (1.034157435, 0.011964553993011023)

rho_T_155_cm_y = 'Temp:  155.09 rho:  0.9234939082334516 Area:  4.873890082355937e-35 Icl:  3.8078947995855774e-46 skip:  200003 nstart:  0 Beads:  16 sim:  1'
rho_T_155_cm_y = 'Temp:  155.09 rho:  0.9626270409858956 Area:  4.956443549116153e-35 Icl:  3.714970359648801e-46 skip:  200003 nstart:  0 Beads:  16 sim:  2'
mean, std = (0.9430585, 0.01956849999999999)

rho_T_155_mid_y = 'Temp:  155.09 rho:  0.030639071761008923 Area:  4.87389008235594e-35 Icl:  1.147739617584069e-44 skip:  200003 nstart:  0 Beads:  16 sim:  1'
rho_T_155_mid_y = 'Temp:  155.09 rho:  0.031008160224146746 Area:  4.956443549116151e-35 Icl:  1.1532870376083198e-44 skip:  200003 nstart:  0 Beads:  16 sim:  2'
rho_T_155_mid_y = 'Temp:  155.09 rho:  0.03189507856446559 Area:  5.081503025561596e-35 Icl:  1.149507365309593e-44 skip:  200003 nstart:  0 Beads:  16 sim:  3'

rho_T_200_cm_y = 'Temp:  206.79 rho:   0.8563655707197309 Area:  4.815372936645433e-35 Icl:  4.184440276946418e-46 skip:  200003 nstart:  0 Beads:  16 sim:  1'
rho_T_200_cm_y = 'Temp:  206.79 rho:  0.7733575939593467 Area:  3.372052772922984e-35 Icl:  4.194652527377245e-46 skip:  200003 nstart:  0 Beads:  16 sim:  3'
mean, std = (0.814861285, 0.04150428500000003)

rho_T_200_mid_y = 'Temp:  206.79 rho:  0.031213905750741008 Area:  3.758284802564582e-35 Icl:  1.1583066662004827e-44 skip:  200003 nstart:  0 Beads:  16 sim:  1'
rho_T_200_mid_y = 'Temp:  206.79 rho:  0.028430490438346303 Area:  3.4248255373231925e-35 Icl:  1.1588736396924773e-44 skip:  200003 nstart:  0 Beads:  16 sim:  2'
rho_T_200_mid_y = 'Temp:  206.79 rho:  0.027806599459579674 Area:  3.3646526399271925e-35 Icl:  1.1640572409984302e-44 skip:  200003 nstart:  0 Beads:  16 sim:  3'
mean, std = (0.029149966666666666, 0.0014814879014768295)

rho_T_260_cm_y = 'Temp:  260.0 rho:  0.020978297193263315 Area:  4.410282340586055e-35 Icl:  1.0187136025511954e-44 skip:  1000007 nstart:  0 Beads:  16 sim:  1'
rho_T_260_cm_y = 'Temp:  260.0 rho:  0.04861090992586308 Area:  2.5026098422297142e-34 Icl:  5.728866113435182e-45 skip:  1000007 nstart:  0 Beads:  16 sim:  2'
rho_T_260_cm_y = 'Temp:  260.0 rho:  0.04915705695382636 Area:  9.053477952364979e-35 Icl:  5.869034901341105e-45 skip:  1000007 nstart:  0 Beads:  16 sim:  3'
mean, std =  (0.039582052, 0.013156799345395823)

rho_T_260_mid_y = 'Temp:  260.0 rho:  0.017118840639687227 Area:  1.6936300256514913e-35 Icl:  1.1903652401267444e-44 skip:  1000007 nstart:  30000 Beads:  16 sim:  1'

rho_T_300_cm_y = 'Temp:  310.19 rho:  0.01610132508801466 Area:  6.6046015730488235e-34 Icl:  1.1175848463011425e-44 skip:  200003 nstart:  0 Beads:  16 sim:  1'
rho_T_300_cm_y = 'Temp:  310.19 rho:  0.020177990003225347 Area:  6.984482668203182e-35 Icl:  1.1155591365128074e-44 skip:  8000003 nstart:  0 Beads:  16 sim:  2'
rho_T_300_cm_y = 'Temp:  310.19 rho:  0.02458981389138662 Area:  6.094343106335365e-35 Icl:  7.975897115665835e-45 skip:  8000003 nstart:  0 Beads:  16 sim:  3'
mean,std =(0.02028896666666667, 0.003466509493552396)

rho_T_300_mid = 'Temp:  310.19 rho:  0.018943867730971172 Area:  1.597001025552163e-35 Icl:  1.216493475524649e-44 skip:  8000003 nstart:  30000 Beads:  16 sim:  2'
rho_T_300_mid_y = 'Temp:  310.19 rho:  0.016602458027152624 Area:  1.3764602732171925e-35 Icl:  1.1940618913994703e-44 skip:  8000003 nstart:  30000 Beads:  16 sim:  3'
rho_T_300_mid_y = 'Temp:  310.19 rho:  0.014097617668525088 Area:  1.2134992922242976e-35 Icl:  1.2191659015476103e-44 skip:  8000003 nstart:  30000 Beads:  16 sim:  4'
mean, std =  (0.0165479392, 0.0019788204310598636)

rho_T_600_mid_y = 'Temp:  600.0 rho:  0.005163742302118688 Area:  4.350603613644728e-36 Icl:  1.087122399945442e-44 skip:  8000003 nstart:  30000 Beads:  12 sim:  1'
rho_T_600_mid_y = 'Temp:  600.0 rho:  0.004692978260288533 Area:  4.486391455922415e-36 Icl:  1.087243418901901e-44 skip:  8000003 nstart:  30000 Beads:  12 sim:  2'
rho_T_600_mid_y = 'Temp:  600.0 rho:  0.0071950988264814246 Area:  3.456020141706512e-36 Icl:  1.0055396631231686e-44 skip:  8000003 nstart:  30000 Beads:  12 sim:  3'
mean, std =  (0.0056838996, 0.0010857272298818827)

rho_1000_cm_y = 'Temp:  1000.0 rho:  0.002337680459268606 Area:  6.883560167763885e-37 Icl:  6.849283706598162e-45 skip:  8000003 nstart:  30000 Beads:  8 sim:  1'
rho_1000_cm_y = 'Temp:  1000.0 rho:  0.00194839281198589 Area:  5.452481805270231e-37 Icl:  6.50930911254907e-45 skip:  8000003 nstart:  30000 Beads:  8 sim:  2'
rho_1000_cm_y =  'Temp:  1000.0 rho:  0.004282496053801641 Area:  1.2259441081770198e-36 Icl:  6.658722078689569e-45 skip:  8000003 nstart:  30000 Beads:  8 sim:  3'
rho_1000_cm_y = 'Temp:  1000.0 rho:  0.0026830345675167874 Area:  8.073410162771165e-37 Icl:  6.999192503055812e-45 skip:  8000003 nstart:  30000 Beads:  8 sim:  4'
mean, std = (0.0028106, 0.0008894827598104415)

rho_T_1000_mid = 'Temp:  1000.0 rho:  0.011857181525368212 Area:  4.1239313437577516e-36 Icl:  8.089974396467656e-45 skip:  8000003 nstart:  30000 Beads:  8 sim:  1'
rho_T_1000_mid = 'Temp:  1000.0 rho:  0.08425347458049734 Area:  2.7786169422543955e-35 Icl:  7.671107145756654e-45 skip:  8000003 nstart:  30000 Beads:  8 sim:  2'
rho_T_1000_mid = 'Temp:  1000.0 rho:  0.006740760356265146 Area:  2.2170470770233823e-36 Icl:  7.650383688988022e-45 skip:  8000003 nstart:  30000 Beads:  8 sim:  3'
mean, std =(0.03428333333333333, 0.03539558929521523)























rho_T_1000_mid = 'Temp:  1000.0 rho:  0.0880925121181219 Area:  3.061313165909421e-35 Icl:  8.083248936709946e-45 skip:  8000003 nstart:  30000 Beads:  8 sim:  1'
rho_T_1000_mid = 'Temp:  1000.0 rho:  0.14545618770356683 Area:  4.7807040715168144e-35 Icl:  7.64498873766366e-45 skip:  8000003 nstart:  30000 Beads:  8 sim:  2'
mean, std = (0.11677425, 0.02868)
rho_T_1000_cm = 'Temp:  1000.0 rho:  0.10315655505679727 Area:  3.169288580984537e-35 Icl:  7.146314827615238e-45 skip:  8000003 nstart:  30000 Beads:  8 sim:  1'
rho_T_1000_cm = 'Temp:  1000.0 rho:  0.17353554913966784 Area:  5.075098793603013e-35 Icl:  6.802573387795416e-45 skip:  8000003 nstart:  30000 Beads:  8 sim:  2'
rho_T_1000_cm = 'Temp:  1000.0 rho:  0.09595809877803622 Area:  2.8631859394920704e-35 Icl:  6.940409688194151e-45 skip:  8000003 nstart:  30000 Beads:  8 sim:  3'
rho_T_1000_cm = 'Temp:  1000.0 rho:  0.11329356056904728 Area:  3.532212761169517e-35 Icl:  7.252017086374759e-45 skip:  8000003 nstart:  30000 Beads:  8 sim:  4'
rho_T_1000_cm = 'Temp:  1000.0 rho:  0.11891334027709673 Area:  3.313520489380841e-35 Icl:  6.4815111307635e-45 skip:  8000003 nstart:  30000 Beads:  8 sim:  5'
mean, std = (0.120975847, 0.0274520)

rho_T_600_mid = 'Temp:  600.0 rho:  0.04592378913067279 Area:  2.381626465501701e-35 Icl:  1.0856645022137964e-44 skip:  8000003 nstart:  30000 Beads:  12 sim:  1'
rho_T_600_mid = 'Temp:  600.0 rho:  0.07968138909161082 Area:  4.145569119131065e-35 Icl:  1.0891481757257285e-44 skip:  8000003 nstart:  30000 Beads:  12 sim:  2'
mean, std = (0.0628025, 0.01687)
rho_T_600_cm = 'Temp:  600.0 rho:  0.05159911977742709 Area:  2.3825690369438836e-35 Icl:  9.666358653422679e-45 skip:  8000003 nstart:  30000 Beads:  12 sim:  1'
rho_T_600_cm = 'Temp:  600.0 rho:  0.0872563852909689 Area:  4.210812632704556e-35 Icl:  1.0102489408198081e-44 skip:  8000003 nstart:  30000 Beads:  12 sim:  2'
rho_T_600_cm = 'Temp:  600.0 rho:  0.06397869046350302 Area:  2.528592467861612e-35 Icl:  8.273765311274359e-45 skip:  8000003 nstart:  30000 Beads:  12 sim:  3'
rho_T_600_cm = 'Temp:  600.0 rho:  0.04167230032991783 Area:  1.6648310801292952e-35 Icl:  8.363391074955343e-45 skip:  8000003 nstart:  30000 Beads:  12 sim:  4'
rho_T_600_cm = 'Temp:  600.0 rho:  0.11278296265968493 Area:  5.169329866932309e-35 Icl:  9.595120917277741e-45 skip:  8000003 nstart:  30000 Beads:  12 sim:  5'
mean, std = (0.071457138, 0.025669476969197795)

rho_T_400_cm = 'Temp:  413.58 rho:  0.05781942868605201 Area:  3.152796035610123e-35 Icl:  1.0491414797320361e-44 skip:  8000003 nstart:  30000 Beads:  16 sim:  1'
rho_T_400_cm = 'Temp:  413.58 rho:  0.033658417140595774 Area:  1.8802613692958918e-35 Icl:  1.0748219158756229e-44 skip:  8000003 nstart:  30000 Beads:  16 sim:  2'
mean, std = (0.0457345, 0.012084500000000001)

rho_T_300_cm = 'Temp:  310.19 rho:  0.026519819444116764 Area:  2.111410392223736e-35 Icl:  1.148881757581171e-44 skip:  8000003 nstart:  30000 Beads:  16 sim:  2'
rho_T_300_cm = 'Temp:  310.19 rho:  0.0282941064147282 Area:  1.5721874418637785e-35 Icl:  8.018286724862313e-45 skip:  8000003 nstart:  30000 Beads:  16 sim:  3'
rho_T_300_cm = 'Temp:  310.19 rho:  0.02490239169081513 Area:  1.9186808967468536e-35 Icl:  1.1118211553298286e-44 skip:  8000003 nstart:  30000 Beads:  16 sim:  4 '
rho_T_300_cm = 'Temp:  310.19 rho:  0.026304956077311606 Area:  1.7767169877706016e-35 Icl:  9.746617337460387e-45 skip:  8000003 nstart:  30000 Beads:  16 sim:  5'
mean, std = (0.026503750000000003, 0.0012054402463415595)

rho_T_200_cm = ' Temp:  206.79 rho:  0.8865517457938785 Area:  3.9246744442381474e-35 Icl:  4.258743960717767e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  1'
rho_T_200_cm = 'Temp:  206.79 rho:  0.04646213095640164 Area:  3.3880750075040496e-35 Icl:  7.015132439916865e-45 skip:  200003 nstart:  20000 Beads:  16 sim:  2'
rho_T_200_cm = 'Temp:  206.79 rho:  0.7825147247257805 Area:  3.3651723177732767e-35 Icl:  4.1371071758735006e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  3'
rho_T_200_cm = 'Temp:  206.79 rho:  0.782194339128247 Area:  3.2318422179007804e-35 Icl:  3.974819977979599e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  4'
rho_T_200_cm = 'Temp:  206.79 rho:  0.8375843397312889 Area:  3.627366468700625e-35 Icl:  4.166245599579758e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  5'
mean,std = (0.8222110825, 0.0434543075765846)

rho_T_155_cm = 'Temp:  155.09 rho:  0.9329777758380158 Area:  4.932469068671174e-35 Icl:  3.814488569757193e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  1'
rho_T_155_cm = 'Temp:  155.09 rho:  0.9582002072157708 Area:  4.952293915209717e-35 Icl:  3.729008707414924e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  2'
rho_T_155_cm = 'Temp:  155.09 rho:  0.96665869927736 Area:  5.042590045924619e-35 Icl:  3.763775791659972e-46 skip:  200003 nstart:  20000 Beads:  16 sim:  3'
mean,std = (0.9526119233333334, 0.014306395395944038)

rho_T_100_cm = 'Temp:  103.4 rho:  1.0228835076492873 Area:  8.100003145158181e-35 Icl:  7.618000446906713e-46 skip:  200003 nstart:  20000 Beads:  32 sim:  1'
rho_T_100_cm = 'Temp:  103.4 rho:  1.0418080260722753 Area:  8.278209398935105e-35 Icl:  7.644176242814012e-46 skip:  200003 nstart:  20000 Beads:  32 sim:  2'
rho_T_100_cm = 'Temp:  103.4 rho:  1.0139730632316384 Area:  8.025536936540435e-35 Icl:  7.614294369121898e-46 skip:  200003 nstart:  20000 Beads:  32 sim:  3'
mean, std =  (1.0262203333333335, 0.01160646878062213)