#%%
import numpy as np
import matplotlib.pyplot as plt


def V_prima(coefs,V):
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
    
    
def paso_runge_kutta(coefs,V,dt,t):
    k1 = V_prima(coefs,V)
    k2 = V_prima(coefs,V+k1*dt/2)
    k3 = V_prima(coefs,V+k2*dt/2)
    k4 = V_prima(coefs,V+k3*dt)
    V = V + (k1+2*(k2+k3)+k4)/6*dt
    t = t+dt
    return([t,V])
     
          
def integra_runge_kutta(coefs,V0,t0,T,dt):
    t = [t0]
    V = [V0]
    while t[-1]<T:
        Vant = V[-1]
        tant = t[-1]
        paso = paso_runge_kutta(coefs,Vant, dt, tant)
        t_new = paso[0]
        V_new = paso[1]
        t.append(t_new)
        V.append(V_new)
    return([t,V])

def despliega_integracion(integracion):
    t = integracion[0]
    V = integracion[1]
    El = []
    Er = []
    Hl = []
    Hr = []
    for i in range(len(V)):
        El.append(V[i][0])
        Er.append(V[i][1])
        Hl.append(V[i][2])
        Hr.append(V[i][3])
    return([np.array(t),np.array(El),np.array(Er),np.array(Hl), np.array(Hr)])    
 
# %%
    
 
M = 0.5
a = 3.4
e = 0.1
g = 3
L = 0.8
tau = 15
tauh = 1000
R = 0.6

coefs = [M,a,e,g,L,tau,tauh,R]

El0 = 1
Er0 = 0
Hl0 = 1
Hr0 = 0
V0= np.array([El0, Er0, Hl0, Hr0])
    
dt= 5
t0= 0
T= 20000

integracion = integra_runge_kutta(coefs,V0,t0,T,dt)
variables = despliega_integracion(integracion)
t = variables[0]
El = variables[1]
Er = variables[2]
Hl = variables[3]
Hr = variables[4]
plt.plot(t, El, label='El')
plt.plot(t, Er, label= 'Er')
# plt.plot(t, Hl, label='Hl')
# plt.plot(t, Hr, label = 'Hr')
plt.xlabel('tiempo')
plt.legend()
#plt.yscale('log')
plt.show()

# %%
