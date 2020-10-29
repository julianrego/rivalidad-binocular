from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt
# %%
m = GEKKO()    # create GEKKO model
m.time = np.linspace(0,200,70) # time points

# create GEKKO parameter (step 0 to 2 at t=5)
tau = m.Param(value=15)
tau_H = m.Param(value=1000)
M = m.Param(value=1.2)
a = m.Param(value=3.4)
eps = m.Param(value=0.1)
g = m.Param(value=3)
L = m.Param(value=0.2)
R = m.Param(value=0.8)

# create GEKKO variables
E_l = m.Var(1.0)
E_r = m.Var(0.0)
H_l = m.Var(1.0)
H_r = m.Var(0.0)

# create GEEKO equations
m.Equation(tau*E_l.dt()==-E_l+M*(L-a*E_r+eps*E_l-g*H_l))
m.Equation(tau_H*H_l.dt()==-H_l+E_l)
m.Equation(tau*E_r.dt()==-E_r+M*(R-a*E_l+eps*E_r-g*H_r))
m.Equation(tau_H*H_r.dt()==-H_r+E_r)

# solve ODE
m.options.IMODE = 7
m.solve(disp=False)


# plot results
#plt.plot(m.time,u,'g:',label='u(t)')
#plt.plot(m.time,x,'b-',label='x(t)')
#plt.plot(m.time,y,'r--',label='y(t)')
#plt.ylabel('values')
#plt.xlabel('time')
#plt.legend(loc='best')
#plt.show()
# %%%

plt.plot(m.time,E_l,'g:',label='E_l(t)')
plt.plot(m.time,E_r,'b-',label='E_r(t)')
plt.ylabel('values')
plt.xlabel('time')
plt.legend(loc='best')
plt.show()
