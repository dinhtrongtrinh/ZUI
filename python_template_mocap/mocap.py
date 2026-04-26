from typing import Tuple

import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
import scipy.io as sio
from matplotlib import animation
from matplotlib import patches as mpatches
from matplotlib import pyplot as plt


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def playmotion(conn, A, B = None):
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.axis('off')

    conns = [x[x!=41] for x in np.split(conn, np.where(conn==41)[0]) if len(x[x!=41])]

    macAs = []
    macBs = []

    m,n = A.shape

    if(B is not None):
        B = B.reshape(3,m//3,n, order='F')
        for conn in conns:
            macBs.append(ax.plot(B[0,conn,0], B[1,conn,0], B[2,conn,0], marker='o', color='r')[0])

    A = A.reshape(3,m//3,n, order='F')
    for conn in conns:
        macAs.append(ax.plot(A[0,conn,0], A[1,conn,0], A[2,conn,0], marker='o', color='b')[0])

    fig.legend(handles =[mpatches.Patch(color='red', label='approximation'),mpatches.Patch(color='blue', label='GT')])
    set_axes_equal(ax)

    def update_points(i, A, B, macAs, macBs, conn):
        for conn, macA in zip(conns, macAs):
            macA.set_data(np.array(A[:2,conn,i]))
            macA.set_3d_properties(A[2,conn,i], 'z')
        for conn, macB in zip(conns, macBs):
            macB.set_data(np.array(B[:2,conn,i]))
            macB.set_3d_properties(B[2,conn,i], 'z')
        return macAs + macBs
    
    ani = animation.FuncAnimation(fig, update_points, n, fargs=(A, B, macAs, macBs, conns), interval=1)
    plt.show()

def fitlin(A: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """ computes the optimal linear fit of A """
    # SVD rozklad. full_matrices=False (neboli 'econ') zabrání tvorbě n x n matice
    U_full, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Vezmeme prvních k směrů pro vytvoření ortonormální báze
    U = U_full[:, :k]
    
    # Projekce do podprostoru (souřadnice C)
    C = U.T @ A
    
    return U, C

def fitaff(A: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ computes the optimal affine fit of A """
    # Těžiště bodů
    b0 = np.mean(A, axis=1)
    
    # Vycentrování - bezpečný broadcasting díky np.newaxis
    A_centered = A - b0[:, np.newaxis]
    
    # Nalezení optimální báze a souřadnic pro vycentrovaná data
    U, C = fitlin(A_centered, k)
    
    return U, C, b0

def erraff(A: np.ndarray) -> np.ndarray:
    """ computes the errors of affine approximations of A """
    m = A.shape[0]
    
    # Rychlé vycentrování
    b0 = np.mean(A, axis=1, keepdims=True)
    A_centered = A - b0
    
    # SVD pouze pro singulární čísla (compute_uv=False), což ušetří hromadu času!
    s = np.linalg.svd(A_centered, compute_uv=False)
    
    # Umocnění singulárních čísel (chyby jsou součty čtverců vynechaných dimenzí)
    s2 = s**2
    s2_padded = np.zeros(m)
    s2_padded[:len(s2)] = s2
    
    # Rychlý výpočet chyby bez cyklů pomocí kumulativní sumy odzadu!
    # d[k] má obsahovat součet s2_padded od indexu k do konce.
    c = np.cumsum(s2_padded[::-1])
    d = np.zeros(m)
    d[:-1] = c[:m-1][::-1] 
    # (d[m-1] automaticky zůstane 0, protože pokud k=m, chyba je 0)
    
    return d

def drawfitline(A: np.ndarray) -> None:
    """ draws the optimal line fitting points from A """
    # Výpočet fitu s dimenzí 1 (přímka)
    U, C, b0 = fitaff(A, 1)
    
    # Projekce bodů zpět do původního prostoru (na přímku)
    B = U @ C + b0[:, np.newaxis]
    
    # Nalezení dvou krajních bodů přímky pro vykreslení úsečky přes celá data
    c_min, c_max = np.min(C), np.max(C)
    p1 = b0 + U[:, 0] * c_min
    p2 = b0 + U[:, 0] * c_max
    
    plt.subplot(221)
    
    # Původní data (červené křížky)
    plt.plot(A[0, :], A[1, :], 'rx')
    
    # Optimální přímka (zelená)
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-')
    
    # Spojnice (červené čáry chyb) nakreslené efektivně najednou
    plt.plot([A[0, :], B[0, :]], [A[1, :], B[1, :]], 'r-', lw=0.8)
    
    plt.axis('equal')
    plt.title('drawfitline')

def plottraj2(C: np.ndarray) -> None:
    """ draws the optimal line fitting points from A """
    plt.subplot(222)
    # Zobrazení postupně po sobě jdoucích bodů (trajektorie)
    plt.plot(C[0, :], C[1, :], 'b.-')
    plt.axis('equal')
    plt.title('plottraj2')


if(__name__ == '__main__'):
    A = sio.loadmat('data/line.mat')['A']
    drawfitline(A)

    conn = np.loadtxt('data/connected_points.txt', comments='%', dtype=int)-1
    filename = 'run4.txt' # see the data folder and try more examples
    A = np.loadtxt('data/' + filename).T
    k = 2 # dimension of affine approximation

    U, C, b0 = fitaff(A,k)
    B = U@C+b0.reshape(-1,1)

    plottraj2(C[:2])

    plt.subplot(212)
    plt.semilogy(erraff(A))
    plt.xlabel('dimension')
    plt.ylabel('error, log scale')
    plt.title('Error of affine approximation \n for motion capture')

    plt.tight_layout()
    playmotion(conn, A, B)
