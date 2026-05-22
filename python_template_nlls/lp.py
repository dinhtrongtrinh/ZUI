import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

def vyhra(c, k):
    c_obj = np.array([0, 0, 0, 0, 0, -1])
    
    A_eq = np.array([[1, 1, 1, 1, 1, 0]])
    b_eq = np.array([k])
    
    A_ub = np.array([
        [-c[0], -c[1], 0, 0, 0, 1],            
        [0, -c[1], -c[2], -c[3], 0, 1],        
        [0, 0, 0, -c[3], -c[4], 1]             
    ])
    b_ub = np.array([0, 0, 0])
    
    bounds = [(0, None)] * 5 + [(None, None)]
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    
    return res.x[:5]


def vyhra2(c, k, m):
    c_obj = np.array([0, 0, 0, -1])
    
    A_eq = np.array([[1, 1, 1, 0]])
    b_eq = np.array([k])
    
    A_ub = np.array([
        [-c[0], 0, 0, 1],     
        [0, -c[1], 0, 1],     
        [0, 0, -c[2], 1]      
    ])
    b_ub = np.array([0, 0, 0])
    
    bounds = [(m, None)] * 3 + [(None, None)]
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    
    return res.x[:3]


def minimaxfit(x, y):
    n, m = x.shape
    c_obj = np.zeros(n + 2)
    c_obj[-1] = 1
    A_top = np.hstack((x.T, np.ones((m, 1)), -np.ones((m, 1))))
    b_top = y.T
    A_bot = np.hstack((-x.T, -np.ones((m, 1)), -np.ones((m, 1))))
    b_bot = -y.T
    
    A_ub = np.vstack((A_top, A_bot))
    b_ub = np.vstack((b_top, b_bot)).flatten()
    bounds = [(None, None)] * (n + 1) + [(0, None)]
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    
    a = res.x[:n]
    b = res.x[n]
    r = res.x[n+1]
    
    return a, b, r


def plotline(x, y, a, b, r):
    """
    Vykreslí body a nalezený minimaxní pás (pouze pro n=1).
    """
    plt.plot(x[0], y[0], 'ko', label='Zadané body')
    
    x_min, x_max = np.min(x), np.max(x)
    xs = np.array([x_min, x_max])
    
    ys_stred = a[0] * xs + b
    ys_horni = ys_stred + r
    ys_dolni = ys_stred - r
    
    plt.plot(xs, ys_stred, 'r-', label='Afinní funkce (a*x + b)')
    plt.plot(xs, ys_horni, 'b-', label='Horní okraj pásu')
    plt.plot(xs, ys_dolni, 'b-', label='Dolní okraj pásu')
    
    plt.axis('tight')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Test 1
    c1 = np.array([1.27, 1.02, 4.70, 3.09, 9.00])
    k1 = 3000
    x1 = vyhra(c1, k1)
    print("Test vyhra(c, k):")
    print(x1)

    # Test 2
    c2 = np.array([1.27, 4.70, 9.00])
    k2 = 3000
    m_val = 400
    x2 = vyhra2(c2, k2, m_val)
    print("\nTest vyhra2(c, k, m):")
    print(x2)

    # Test 3
    x3 = np.array([[1, 2, 3, 3, 2], [4, 1, 2, 5, 6], [7, 8, 9, -5, 7]])
    y3 = np.array([[7, 4, 1, 2, 5]])
    a, b_val, r = minimaxfit(x3, y3)
    print("\nTest minimaxfit(x, y):")
    print(f"a = {a}")
    print(f"b = {b_val}")
    print(f"r = {r}")