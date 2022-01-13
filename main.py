"""
Ntigkaris E. Alexandros

Runge-Kutta 1st and 4th order numerical ODE solution
"""

import numpy as np
import pandas as pd

def RK1(F,v,w,step):
    
    k1 = F(v,w)
    return w + step*k1

def RK4(F,v,w,step):

    k1 = F(v,w)
    k2 = F(v+0.5*step,w+0.5*k1*step)
    k3 = F(v+0.5*step,w+0.5*k2*step)
    k4 = F(v+step,w+k3*step)
    return w + step/6*(k1+2*k2+2*k3+k4)

dy = lambda t,y: y*t*t - 1.1*y

# Initial conditions
xi = 0
xo = 2
yi = 1
h = 0.25
N = int((xo-xi)/h)

# Start RungeKutta
yRK1 = np.zeros(N+1)
yRK4 = np.zeros(N+1)
yRK1[0] = yi
yRK4[0] = yi

for i in range(N):
    yRK1[i+1] = RK1(dy,xi,yRK1[i],h)
    yRK4[i+1] = RK4(dy,xi,yRK4[i],h)
    xi = xi + h

# Compare with actual value
yTrue = lambda t: np.exp(1.0/3.0*(t**3)-1.1*t)
yTrue = [yTrue(i) for i in np.linspace(xi-xo,xo,N+1)]

table = pd.DataFrame({"Analytical":yTrue,"RK1":yRK1,"RK4":yRK4,"Error (RK1)":abs((yTrue-yRK1)/yTrue)*100,"Error (RK4)":abs((yTrue-yRK4)/yTrue)*100})
print(table)
print(f"[stepsize: {h}]")