#%%
import numpy as np
import scipy.signal
from scipy.integrate import odeint
from scipy import fft
import matplotlib.pyplot as plt
import pandas as pd

# function that returns dy/dt

def V_prima(V, t,coefs):
    M = coefs[0]
    a = coefs[1]
    e = coefs[2]
    g = coefs[3]
    L = coefs[4]
    tau = coefs[5]
    tauh = coefs[6]
    R = coefs[7]
    El = V[0]
    Er = V[1]
    Hl = V[2]
    Hr = V[3]

    El_temp = (L -a*Er + e*El -g*Hl)
    El_temp = El_temp if El_temp>0 else 0

    Er_temp = (R - a*El + e*Er - g*Hr)
    Er_temp = Er_temp if Er_temp>0 else 0

    El_prima = (-El + M*El_temp)/tau
    Er_prima = (-Er + M*Er_temp)/tau
    Hl_prima = (-Hl + El)/tauh
    Hr_prima = (-Hr + Er)/tauh
    resultado = np.array([El_prima, Er_prima, Hl_prima, Hr_prima])
    return(resultado)
    

#%%

#coefs

L = 0.8
R = 0.6

M = 0.5

a = 3.4
e = 0.1
g = 3

tau = 15
tauh = 1000


coefs = [M,a,e,g,L,tau,tauh,R]

# initial condition
El0 = 0
Er0 = 0
Hl0 = 0
Hr0 = 0
V0= np.array([El0, Er0, Hl0, Hr0])

# time points
t = np.linspace(0,100000, 25000)

# solve ODE
y = odeint(V_prima,V0,t, args = (coefs, ))

#solucion para cada ojo
E_l = y[:,0]
E_r = y[:,1]


# plot results
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(t/1000, E_l)
ax1.set_ylabel('Ojo izquierdo')
ax1.annotate(f'Stimulus = {L}',xy=(0.6, 0.85), xycoords='axes fraction')

ax2.plot(t/1000, E_r)
ax2.set_ylabel('Ojo derecho')
ax2.annotate(f'Stimulus = {R}',xy=(0.6, 0.85), xycoords='axes fraction')


#plt.plot(t,y[:,:2])
plt.xlabel('time (s)')
plt.tight_layout()
#plt.ylabel('y(t)')
plt.show()

# %%

def get_dominance_times(estimulo_derecho, estimulo_izquierdo, t):

    df_r = pd.DataFrame(data={'E':estimulo_derecho}, index = t)
    df_l = pd.DataFrame(data={'E':estimulo_izquierdo}, index = t)


    df_r['threshold'] =(df_r['E']>0.05).astype(int) 
    df_r['dif'] = df_r['threshold'].shift(-1) - df_r['threshold'] 

    df_l['threshold'] =(df_l['E']>0.05).astype(int) 
    df_l['dif'] = df_l['threshold'].shift(-1) - df_l['threshold'] 

    #ojo derecho
    tiempos_arriba_der = []
    tiempos_abajo_der = []
    for idx, e in df_r.iterrows():

        if e.dif==1:
            tiempos_arriba_der.append(idx)
        if e.dif==-1:
            tiempos_abajo_der.append(idx)
        
    tiempos_arriba_der = np.array(tiempos_arriba_der)
    tiempos_abajo_der = np.array(tiempos_abajo_der)

    min_len = min(len(tiempos_arriba_der), len(tiempos_abajo_der))
    derecho = tiempos_abajo_der[:min_len-1] - tiempos_arriba_der[:min_len-1]

    #ojo izquierdo
    tiempos_arriba_izq = []
    tiempos_abajo_izq = []
    for idx, e in df_l.iterrows():

        if e.dif==1:
            tiempos_arriba_izq.append(idx)
        if e.dif==-1:
            tiempos_abajo_izq.append(idx)
        
    tiempos_arriba_izq = np.array(tiempos_arriba_izq)
    tiempos_abajo_izq = np.array(tiempos_abajo_izq)

    min_len = min(len(tiempos_arriba_izq), len(tiempos_abajo_izq))
    izquierdo =  tiempos_abajo_izq[:min_len-1] - tiempos_arriba_izq[:min_len-1]

    #promedios excluyendo los bordes

    derecho_mean = derecho[1:-1].mean()
    izquierdo_mean = izquierdo[1:-1].mean()

    results = {
        'derecho':derecho_mean,
        'izquierdo':izquierdo_mean
    }

    return results

#%%

results = get_dominance_times(E_r, E_l, t)

#%%
#tiempos de dominancia para distintas intensidades
#coefs

R = 1.0

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
t = np.linspace(0,100000, 25000)

final_results = []
for L in np.linspace(0.6, 1, 21):
    coefs = [M,a,e,g,L,tau,tauh,R]
    y = odeint(V_prima,V0,t, args = (coefs, ))

    E_l = y[:,0]
    E_r = y[:,1]

    results = get_dominance_times(E_r, E_l, t)
    results = {'R':R, 'L':L, 'dominancia_derecho':results['derecho'], 'dominancia_izquierdo':results['izquierdo']}
    final_results.append(results)

df_results = pd.DataFrame(final_results)
#%%

#df_results['dominancia_derecho'] = df_results['dominancia_derecho']/100
#df_results['dominancia_izquierdo'] = df_results['dominancia_izquierdo']/100

#%%

plt.plot(df_results.L, df_results.dominancia_derecho/1000, 'o-', label='Ojo derecho')
plt.plot(df_results.L, df_results.dominancia_izquierdo/1000, 'o-', label='Ojo izquierdo')

plt.ylabel('Tiempos de dominancia (s)')
plt.xlabel('Fuerza del estimulo más debil (ojo izquierdo)')
plt.legend()
plt.show()

# %%

df_r = pd.DataFrame(data={'E':E_r}, index = t)
df_l = pd.DataFrame(data={'E':E_l}, index = t)


# %%

df_r['threshold'] =(df_r['E']>0.05).astype(int) 
df_r['dif'] = df_r['threshold'].shift(1) - df_r['threshold'] 

df_l['threshold'] =(df_l['E']>0.05).astype(int) 
df_l['dif'] = df_l['threshold'].shift(1) - df_l['threshold'] 



#%%




tiempos_arriba = []
tiempos_abajo = []
flag = 0
for idx, e in df_r.iterrows():

    if e.dif==1:
        tiempos_arriba.append(idx)
    if e.dif==-1:
        tiempos_abajo.append(idx)
    
tiempos_arriba = np.array(tiempos_arriba)
tiempos_abajo = np.array(tiempos_abajo)


# %%
