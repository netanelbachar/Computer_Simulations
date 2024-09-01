import numpy as np
import matplotlib.pyplot as plt
# This codes solves the finite schrodinger equation for any external potential given. Here the Harmonic and An-harmonic
# potentials are written.

#Units
hbar = 1             # Hartree
m = 0.84             # amu
L = 150              # bohr
n = 51
pot = 'harm'


def laplacian_1D(n):
    D = -2.0 * np.eye(n)
    for i in range(n-1):
        D[i, i+1] = D[i+1, i] = 1
    return D

def laplacian_2D(n):
    D1 = laplacian_1D(n)
    return np.kron(D1, np.eye(n)) + np.kron(np.eye(n), D1)


def oneD_twoN_schrodinger():
    hbar = 1.0
    m = 1.0
    L = 2
    C = 1.0

    n = 50
    eta = 0.001

    x = np.linspace(0, L, n)
    dx = x[1] - x[0]
    X1 = np.kron(x, np.ones(n))
    X2 = np.kron(np.ones(n), x)

    T = -hbar ** 2 / (2 * m) / (dx ** 2) * laplacian_2D(n)
    V = C * np.diag(1.0 * np.abs(1.0 / (X1 - X2 + eta * 1j)**3)) # j  is for complex number
    H = T + V

    E, U = np.linalg.eigh(H)

    psi0 = U[:,0].reshape(n, n)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.imshow(psi0[:,::-1], interpolation='bilinear', extent=[0, L, 0, L])
    plt.show();
    return E

E = oneD_twoN_schrodinger()
print(E)

def two_d_schrodinger(L, n, s, pot):
    '''
    One needs to change k or l according to the strength of the harmonic and an-harmonic trap
    :param L: length of X, Y coord
    :param n: dimension of matrix
    :param s: number of Energy state wanted to be outputed
    :return: eigenvalues , eigenvector
    '''
    # Coordinated
    x = np.linspace(-L, L, n)
    dx = x[1] - x[0]
    X = np.kron(x, np.ones(n))
    Y = np.kron(np.ones(n), x)

    # Energies
    if pot == 'harm':
        #omega = 0.00098215  # 1/s- This number is obtained from sqrt(k/m) where k = 8.1045*10^-7 and m = 0.84 (26.73 meV)
        # it goes 34.73meV, 26.73meV, 18.73meV
        k = 8.10542107*10**-7   #  Hartree / bohr**2 -  1.36831859*10**-6, 8.10542107*10**-7, 3.97972642*10**-7
        V = np.diag(0.5 * k * (X ** 2 + Y ** 2))  # Potential
    elif pot == 'anharm':
        l = 9.5*10**-11       # Hartree / bohr**4  -3.28*10**-11  -  1.7978 *10**-12  - 9.5*10**-11
        V = np.diag(l * (X**4 + Y**4) + 2*l*((X**2)*(Y**2)))


    T = -hbar ** 2 / (2 * m) / (dx ** 2) * laplacian_2D(n)  # Kinetic Energy
    H = T + V                                               # Hamiltonian
    # Eigenvalues and Vectors
    E, U = np.linalg.eigh(H)                                # Eigenvalue/vector
    psi = U[:, s].reshape(n, n)

    return E, psi

E, psi = two_d_schrodinger(L, n, 0, pot)
E1 = E[0:7]
energies_difference = np.zeros(6)
for i in range(0, 6):
    energies_difference[i] = E1[i+1] - E1[i]

print("Energy states (ES) will be printed: 1st ES only 1 print, 2nd ES 2 prints and so on...")
print("Energy states: ", E[0:15])
print("Diff: ", energies_difference[0], energies_difference[2], energies_difference[5])
print(energies_difference)

# plt.xlabel(r'$x_1$')
# plt.ylabel(r'$x_2$')
# shw = plt.imshow(psi[:, ::-1], interpolation='bilinear', extent=[-L, L, -L, L])
# bar = plt.colorbar(shw)
# plt.show()

# Results : The DIFF of the harmonic will be equal to the DIFF of anharmonic
#Harmonic potential 34.73 meV
# Energy states:  [0.00127433 0.00254669 0.00254669 0.00381521 0.00381521 0.00381905 0.00508072 0.00508072 0.00508757 0.00508757 0.00634724 0.00634724 0.00635308 0.00635308 0.00635609]
# Diff: 0.0012723607500896444 0.001268518441069046 0.0012616638318527812
#lambda = 2.08*10**-10

#Harmonic potential 26.73 meV
# Energy states:  [0.00098116 0.00196136 0.00196136 0.00294119 0.00294119 0.00294155 0.00392138 0.00392138 0.00392903 0.00392903 0.00490122 0.00490923 0.00490923 0.00494863 0.00494863]
# Diff:  0.0009801962723151132 0.000979831033444913 0.0009798310334449022
#lambda = 9.53158*10**-11

#Harmonic potential 18.73 meV
# Energy states: [0.00068828 0.00137935 0.00137935 0.00207041 0.00208911 0.00208911 0.00278017 0.00278017 0.00285665 0.00285665 0.00348993 0.00354772 0.00354772 0.00373369 0.00373369]
#Diff:  0.000686574509305213 0.0006853218356778557 0.0006853218356778603
#lambda = 3.28*10**-11
