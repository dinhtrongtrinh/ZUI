import numpy as np

def fit_temps(t: np.ndarray, T: np.ndarray, omega: float):
    col_ones = np.ones(len(t))
    col_sin = np.sin(omega * t)
    col_cos = np.cos(omega * t)

    A_mat = np.stack((col_ones,t,col_sin,col_cos), axis = 1)
    AT_mat = np.stack((col_ones,t,col_sin,col_cos), axis = 0)

    #(AT*A)^-1
    ATA_inverse = np.linalg.inv(np.dot(AT_mat, A_mat))

    #AT*T
    ATb_mat = np.dot(AT_mat,T)

    x = np.dot(ATA_inverse,ATb_mat)
    return x

def fit_wages(t: np.ndarray, M: np.ndarray):
    one_array = np.ones(len(t))

    #A , AT
    A_matrix = np.stack((one_array,t), axis = 1) 
    AT_matrix = np.stack((one_array,t), axis = 0) 

    #AT*A
    ATA_matrix = np.dot(AT_matrix, A_matrix)

    #(AT*A)^-1
    ATA_inverse = np.linalg.inv(ATA_matrix)

    ATb_mat = np.dot(AT_matrix,M)

    #(AT*A)^-1 * AT *b
    x = np.dot(ATA_inverse,ATb_mat)
    return x
    
def quarter2_2009(x:np.ndarray):
    M = x[0] + x[1] * 2009.25
    return M





