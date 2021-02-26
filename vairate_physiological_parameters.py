#%%
from utils import get_dominance_times, V_prima
import numpy as np
import pandas as pd
from itertools import product
from scipy.integrate import odeint
import matplotlib.pyplot as plt
#%%

R = 1.0
L = 1.0

M = 0.5
a = 3.4
e = 0.1
g = 3

tau = 15
tauh = 1000

# initial condition
El0 = 0
Er0 = 0
Hl0 = 0
Hr0 = 0
V0= np.array([El0, Er0, Hl0, Hr0])

# time points
t = np.linspace(0,50000, 5000)
#%%

puntos = 40

#vario M
results_M = []
print("variando \'M\'")
M_values = np.linspace(0, 5, puntos)
for M_i in M_values:
    
    if np.where(M_values == M_i)[0]%5==0:
        print(f'Iteracion N° {np.where(M_values == M_i)[0] + 1}')

    coefs = [M_i,a,e,g,L,tau,tauh,R]
    y = odeint(V_prima,V0,t, args = (coefs, ))

    E_l = y[:,0]
    E_r = y[:,1]

    results = get_dominance_times(E_r, E_l, t)
    results = {
        'M':M_i, 
        'a':a, 
        'e':e, 
        'g':g, 
        'dominancia_derecho':results['derecho'], 
        'dominancia_izquierdo':results['izquierdo']
        }
    results_M.append(results)

#vario a
print("variando \'a\'")
results_a = []
a_values = np.linspace(0, 5, puntos)
for a_i in a_values:

    if np.where(a_values == a_i)[0] %5==0:
        print(f'Iteracion N° {np.where(a_values == a_i)[0] + 1}')

    coefs = [M,a_i,e,g,L,tau,tauh,R]
    y = odeint(V_prima,V0,t, args = (coefs, ))

    E_l = y[:,0]
    E_r = y[:,1]

    results = get_dominance_times(E_r, E_l, t)
    results = {
        'M':M, 
        'a':a_i, 
        'e':e, 
        'g':g, 
        'dominancia_derecho':results['derecho'], 
        'dominancia_izquierdo':results['izquierdo']
        }
    results_a.append(results)

#vario e
print("variando \'e\'")
results_e = []
e_values = np.linspace(0, 5, puntos)
for e_i in e_values:

    if np.where(e_values == e_i)[0]%5==0:
        print(f'Iteracion N° {np.where(e_values == e_i)[0] + 1}')

    coefs = [M,a,e_i,g,L,tau,tauh,R]
    y = odeint(V_prima,V0,t, args = (coefs, ))

    E_l = y[:,0]
    E_r = y[:,1]

    results = get_dominance_times(E_r, E_l, t)
    results = {
        'M':M, 
        'a':a, 
        'e':e_i, 
        'g':g, 
        'dominancia_derecho':results['derecho'], 
        'dominancia_izquierdo':results['izquierdo']
        }
    results_e.append(results)

#vario g
print("variando \'g\'")
results_g = []
g_values = np.linspace(0, 5, puntos)
for g_i in g_values:
   
    if np.where(g_values == g_i)[0]%5==0:
        print(f'Iteracion N° {np.where(g_values == g_i)[0] + 1}')

    coefs = [M,a,e,g_i,L,tau,tauh,R]
    y = odeint(V_prima,V0,t, args = (coefs, ))

    E_l = y[:,0]
    E_r = y[:,1]

    results = get_dominance_times(E_r, E_l, t)
    results = {
        'M':M, 
        'a':a, 
        'e':e, 
        'g':g_i, 
        'dominancia_derecho':results['derecho'], 
        'dominancia_izquierdo':results['izquierdo']
        }
    results_g.append(results)

df_M = pd.DataFrame(results_M)
df_a = pd.DataFrame(results_a)
df_e = pd.DataFrame(results_e)
df_g = pd.DataFrame(results_g)

# %%

plt.plot(df_M.M, df_M.dominancia_derecho/1000, '-o', label = 'M')
plt.plot(df_a.a, df_a.dominancia_derecho/1000, '-o', label = 'a')
plt.plot(df_e.e, df_e.dominancia_derecho/1000, '-o', label = '$\epsilon$')
plt.plot(df_g.g, df_g.dominancia_derecho/1000, '-o', label = 'g')

plt.xlabel('Valor del parametro')
plt.ylabel('Tiempo de dominancia (s)')
plt.legend()
plt.show()

# %%

# %%
#vario la intensidad de ambos impulsos

puntos = 30

#vario M
results_int = []
print("variando \'M\'")
L_values = np.logspace(-1, 0, puntos)
R_values = np.logspace(-1, 0, puntos)
for L_i, R_i in zip(L_values, R_values):
    
    if np.where(L_values == L_i)[0]%5==0:
        print(f'Iteracion N° {np.where(M_values == L_i)[0] + 1}')

    coefs = [M,a,e,g,L_i,tau,tauh,R_i]
    y = odeint(V_prima,V0,t, args = (coefs, ))

    E_l = y[:,0]
    E_r = y[:,1]

    results = get_dominance_times(E_r, E_l, t)
    results = {
        'L':L_i,
        'R':R_i,
        'M':M, 
        'a':a, 
        'e':e, 
        'g':g, 
        'dominancia_derecho':results['derecho'], 
        'dominancia_izquierdo':results['izquierdo']
        }
    results_int.append(results)


# %%
df = pd.DataFrame(results_int)
plt.plot(df.L, df.dominancia_derecho/1000, '-o')#, label = 'M')
plt.xlabel('Intensidad de ambos estimulos')
plt.ylabel('Tiempo de dominancia (s)')
#plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.show()
# %%
