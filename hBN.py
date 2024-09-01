from functions_PIMD import *

def moire_plot_hbn(method, x_moire_lims):
    import matplotlib.ticker as ticker
    pi = math.pi
    moire_period = 47.28  # Angstrom
    b = (4 * pi) / moire_period
    if method == 'dft':
        V = -0.116  # eV
        V_0 = 0.0183 # eV
        psi1 = -0.0102
        psi2 = -1.352
        psi3 = 1.352
    elif method == 'sgw':
        V = -0.153  # eV
        V_0 = 0.0208  # eV
        psi1 = -0.104
        psi2 = -1.456
        psi3 = 1.456
    X, Y = np.meshgrid(np.linspace(-x_moire_lims, x_moire_lims, 1024),
                       np.linspace(-x_moire_lims,  x_moire_lims, 1024))  # * 10 ** -9  # nm

    Z =  V_0 + 2*V*(np.cos(b*X - psi1) +
                    np.cos((-0.5*b * X) + ((math.sqrt(3)/2)*b * Y) - psi2) +
                    np.cos((-0.5*b * X) - ((math.sqrt(3)/2)*b * Y) - psi3))

    # Z =  V_0 + 2*V*(np.cos(b*Y - psi1) +
    #                 np.cos((-0.5*b * Y) + ((math.sqrt(3)/2)*b * X) - psi2) +
    #                 np.cos((-0.5*b * Y) - ((math.sqrt(3)/2)*b * X) - psi3))

    # # Plot Moire Potential 2D
    levels = np.linspace(Z.min(), Z.max(), 50)
    v = np.linspace(np.min(Z), np.max(Z), 2, endpoint=True)
    fig, ax = plt.subplots()
    plt.set_cmap('coolwarm')
    graph = ax.contourf(X, Y, Z, levels=levels)
    plt.xlabel('x [$\AA$]', fontsize=15)
    plt.xticks(fontsize=15)
    plt.ylabel('y [$\AA$]', fontsize=15)
    plt.yticks(fontsize=15)
    # cbar =  plt.colorbar(graph, format=ticker.FuncFormatter(myfmt), ticks=v, orientation="horizontal") # Scientidic Notation
    cbar = plt.colorbar(graph, ticks=v, orientation="horizontal")
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(label='[meV]', size=15)
    plt.show()

x_moire_lims =60
# method = 'dft'
method = 'sgw'
moire_plot_hbn(method, x_moire_lims)