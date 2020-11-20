#%%
from utils import get_dominance_distribution, V_prima_random
import numpy as np
import pandas as pd
from itertools import product
from scipy.integrate import odeint
import matplotlib.pyplot as plt


#%%
#tiempos de dominancia para distintas intensidades
#coefs

R = 1.0
L = 0.9

M = 0.5
a = 3.4
e = 0.1
g = 3

tau = 15
tauh = 1000

sigma = 0

# initial condition
El0 = 0
Er0 = 0
Hl0 = 0
Hr0 = 0
V0= np.array([El0, Er0, Hl0, Hr0])

# time points
t = np.linspace(0,100000, 25000)


coefs = [M,a,e,g,L,tau,tauh,R, sigma]
y = odeint(V_prima_random,V0,t, args = (coefs, ))

E_l = y[:,0]
E_r = y[:,1]

results = get_dominance_distribution(E_r, E_l, t)
results = {'dominancia_derecho':results['derecho'], 'dominancia_izquierdo':results['izquierdo']}

df_results = pd.DataFrame(results)
# %%
