import numpy as np 
from numba import jit

@jit(nopython = True)
def loop(sig, ref):
    Rinv = np.array([100.0,100.0,100.0,100.0])
    H = np.zeros((4))
    e = sig
    end = (sig.shape[0]+1)
    
    for n in range(4,end,1):
        r = ref[n-4:n]
        div = 0.995 + np.sum(r*Rinv*r)
        K = (Rinv*r) / div
        e_n = sig[n-1] - np.sum(r*H)
        H = H + e_n*K
        Rinv = Rinv/0.995 - (K * r * Rinv)/0.995
        e[n-1] = sig[n-1] - np.sum(r*H)
    
    sig = e
    return sig
