import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# function that returns dy/dt

def V_prima(V, t, coefs):
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
    El_prima = (-El + M*(L -a*Er + e*El -g*Hl))/tau
    Er_prima = (-Er + M*(R - a*El + e*Er - g*Hr))/tau
    Hl_prima = (-Hl + El)/tauh
    Hr_prima = (-Hr + Er)/tauh
    resultado = np.array([El_prima, Er_prima, Hl_prima, Hr_prima])
    return resultado
    


# initial condition
y0 = np.array([1,0,1,0])

# time points
t = np.linspace(0,20)

# solve ODE
y = odeint(model,y0,t)

# plot results
plt.plot(t,y)
plt.xlabel('time')
plt.ylabel('y(t)')
plt.show()