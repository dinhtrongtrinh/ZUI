import numpy as np

def fit_temps(t: np.ndarray, T: np.ndarray, omega: float):
    # Vytvoření matice plánu (design matrix) přímo v NumPy
    # Sloupce: [1, t, sin(omega*t), cos(omega*t)]
    A = np.column_stack((
        np.ones_like(t), 
        t, 
        np.sin(omega * t), 
        np.cos(omega * t)
    ))
    
    # Elegantní řešení soustavy A*x = T pomocí metody nejmenších čtverců
    # lstsq vrací (řešení, rezidua, hodnost, singulární hodnoty)
    x, _, _, _ = np.linalg.lstsq(A, T, rcond=None)
    return x

def fit_wages(t: np.ndarray, M: np.ndarray):
    # Matice plánu pro lineární trend: [1, t]
    A = np.column_stack((np.ones_like(t), t))
    
    # Opět použijeme lstsq pro stabilitu
    x, _, _, _ = np.linalg.lstsq(A, M, rcond=None)
    return x
    
def quarter2_2009(x: np.ndarray):
    # 2009.25 odpovídá druhému kvartálu (duben-červen)
    return x[0] + x[1] * 2009.25