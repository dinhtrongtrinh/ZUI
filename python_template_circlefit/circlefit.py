import scipy.io as sio
import numpy as np
from math import pi
from matplotlib import pyplot as plt

def quad_to_center(d, e, f):
    """
    Přepočítá reprezentaci kružnice z ax^2 + ay^2 + dx + ey + f = 0 (kde a=1)
    na střed a poloměr: (x - x0)^2 + (y - y0)^2 = r^2.
    
    Doplněním na čtverec dostaneme:
    (x + d/2)^2 + (y + e/2)^2 = (d^2 + e^2)/4 - f
    """
    x0 = -d / 2.0
    y0 = -e / 2.0
    
    # Pojistka pro případ, že numerické chyby způsobí mírně záporné číslo pod odmocninou
    r_squared = (d**2 + e**2) / 4.0 - f
    r = np.sqrt(max(0, r_squared))
    
    return x0, y0, r

def fit_circle_nhom(X):
    """
    Nehomogenní řešení pomocí metody nejmenších čtverců.
    Hledáme d, e, f takové, že d*x_i + e*y_i + f = -(x_i^2 + y_i^2).
    Řešíme přeurčenou soustavu A * v = b.
    """
    x = X[:, 0]
    y = X[:, 1]
    
    # Matice A má sloupce [x, y, 1]
    A = np.column_stack((x, y, np.ones_like(x)))
    # Vektor b obsahuje záporný součet čtverců
    b = -(x**2 + y**2)
    
    # Řešení pomocí metody nejmenších čtverců
    v, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    d, e, f = v
    return d, e, f

def fit_circle_hom(X):
    """
    Homogenní řešení problému.
    Hledáme a, d, e, f, které minimalizují ||A*v|| za podmínky ||v|| = 1.
    A * [a, d, e, f]^T = 0
    """
    x = X[:, 0]
    y = X[:, 1]
    
    # Matice A má sloupce [x^2 + y^2, x, y, 1]
    A = np.column_stack((x**2 + y**2, x, y, np.ones_like(x)))
    
    # Řešení pomocí Singulárního rozkladu matice (SVD)
    U, S, Vh = np.linalg.svd(A)
    
    # Optimální řešení je poslední řádek matice Vh (odpovídá nejmenšímu singulárnímu číslu)
    v = Vh[-1, :]
    a, d, e, f = v
    
    # Podle zadání vydělíme koeficienty číslem 'a', abychom získali rovnici kde a=1
    return d/a, e/a, f/a

def dist(X, x0, y0, r):
    """
    Vypočítá orientovanou vzdálenost bodů od kružnice.
    Záporná hodnota = uvnitř, kladná = vně.
    """
    x = X[:, 0]
    y = X[:, 1]
    
    # Vzdálenost od středu mínus poloměr
    distances = np.sqrt((x - x0)**2 + (y - y0)**2) - r
    return distances

def fit_circle_ransac(X, num_iter, threshold):
    """
    Robustní prokládání kružnice pomocí algoritmu RANSAC.
    """
    best_inliers_count = -1
    best_circle = (0, 0, 0)
    n_points = X.shape[0]
    
    for _ in range(num_iter):
        # 1. Vybereme náhodně 3 body (bez opakování)
        idx = np.random.choice(n_points, 3, replace=False)
        sample = X[idx, :]
        
        # 2. Odhadneme kružnici pomocí 3 bodů (můžeme použít přesné nehomogenní řešení)
        d, e, f = fit_circle_nhom(sample)
        
        # Pojistka proti kolineárním bodům (způsobí nekonečný poloměr)
        if np.isnan(d) or np.isnan(e) or np.isnan(f):
            continue
            
        x0, y0, r = quad_to_center(d, e, f)
        
        # 3. Spočítáme vzdálenosti všech bodů k navržené kružnici
        distances = dist(X, x0, y0, r)
        
        # 4. Spočítáme počet inlierů (bodů s chybou menší než threshold)
        inliers_count = np.sum(np.abs(distances) <= threshold)
        
        # Uložíme si nejlepší výsledek
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_circle = (x0, y0, r)
            
    return best_circle

def plot_circle(x0, y0, r, color, label):
    t = np.arange(0, 2*pi, 0.01)
    X = x0 + r*np.cos(t)
    Y = y0 + r*np.sin(t)
    plt.plot(X, Y, color=color, label=label)

if __name__ == '__main__':
    data = sio.loadmat('data.mat')
    X = data['X'] # inliers
    A = data['A'] # inliers + outliers

    def_nh = fit_circle_nhom(X)
    x0y0r_nh = quad_to_center(*def_nh)
    dnh = dist(X, *x0y0r_nh)

    def_h = fit_circle_hom(X)
    x0y0r_h = quad_to_center(*def_h)
    dh = dist(X, *x0y0r_h)

    results = {'def_nh': def_nh, 'def_h': def_h, 
               'x0y0r_nh': x0y0r_nh, 'x0y0r_h': x0y0r_h,
               'dnh': dnh, 'dh': dh}
    
    try:
        GT = sio.loadmat('GT.mat')
        for key in results:
            print('max difference', np.amax(np.abs(np.array(results[key]) - np.array(GT[key]))), 'in', key)
    except FileNotFoundError:
        print("Soubor GT.mat nenalezen, přeskakuji kontrolu proti Ground Truth.")

    # Spuštění RANSACu na zašuměných datech s outliery
    x = fit_circle_ransac(A, 2000, 0.1)

    plt.figure(1, figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], marker='.', s=3, color='black')
    plot_circle(*x0y0r_h, 'r', 'hom')
    plot_circle(*x0y0r_nh, 'b', 'nhom')
    plt.title('Fit inliers only')
    plt.legend()
    plt.axis('equal')    
    
    plt.subplot(122)
    plt.scatter(A[:, 0], A[:, 1], marker='.', s=2, color='black')
    plot_circle(*x, 'y', 'ransac')
    plt.title('RANSAC with outliers')
    plt.legend()
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()