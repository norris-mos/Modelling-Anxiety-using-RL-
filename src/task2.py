import numpy as np
import pandas as pd

def observation(V_init,alpha,beta,t,outcome):
    V = np.zeros(t)
    V[0] = V_init
    for i in range(1,t):
        V[t] = V[t-1] + alpha * (outcome[t-1]-V[t-1])

