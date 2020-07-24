from scipy.integrate import solve_ivp
import numpy as np

def f(t,y):
    dy=om*y*1j-a*y
    return dy

a=0.1
om=1
ti=0
tf=100
y0=1
n=1000

tev=np.linspace(ti,tf,n)

sol=solve_ivp(f,(ti,tf),[y0],t_eval=tev)
t=sol.t
y=sol.y


import matplotlib.pyplot as plt

plt.plot(t,y[0])

plt.show()