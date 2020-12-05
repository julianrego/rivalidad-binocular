#%%
from utils import get_dominance_distribution, V_prima_random,integration_ode
import numpy as np
import pandas as pd
from itertools import product
from scipy.integrate import odeint
from scipy.stats import gamma
import matplotlib.pyplot as plt


#%%
#tiempos de dominancia para distintas intensidades
#coefs

R = 1.0
L = 1.0

M = 0.5
a = 3.4
e = 0.1
g = 3

tau = 15
tauh = 1000

# initial condition
El0 = 0.
Er0 = 0.
Hl0 = 0.
Hr0 = 0.
V0= np.array([El0, Er0, Hl0, Hr0])

# time points
steps = 2500000
tmax = 5000000
tstep = tmax/steps
t = np.linspace(0,tmax, steps)

sigma = 0.001*np.sqrt(tstep)

#%%
coefs = [M,a,e,g,L,tau,tauh,R, sigma]
y = integration_ode(V_prima_random,V0,tmax, tstep, coefs = coefs)

E_l = y[:,0]
E_r = y[:,1]

results = get_dominance_distribution(E_r, E_l, t)
results = {'dominancia_derecho':results['derecho'], 'dominancia_izquierdo':results['izquierdo']}

#df_results = pd.DataFrame(results)
# %%
#plt.plot(E_r)
# %%
#plt.plot(E_l)
# %%
df_results = pd.DataFrame(results)
# %%

df_results.dominancia_derecho.hist(bins=30)
plt.xlabel('Tiempo de dominancia (ms)')
plt.ylabel('Count')
plt.legend()
plt.show()

#%%
a, loc, scale = gamma.fit(df_results.dominancia_derecho)
# %%
