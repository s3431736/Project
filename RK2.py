import numpy as np
from scipy.integrate import solve_ivp
x0 = 0
xf = 1000
y0 = 5

n = 5
h = (xf - x0) / n
# print(h)

x = np.linspace(x0, xf, n)

y_euler = np.zeros(n)
y_RKII = np.zeros(n)
y_RKIV=np.zeros(n)

# y_plus=np.linspace(x0,xf,n)
f = np.zeros(n)
# f=np.linspace(x0,xf,n)


f = y0 - 0.5 * (x ** 2)
# ^^ for df=-x
i = 0
y_euler[0] = y0
y_RKII[0]=y0

while i < (n - 1):
    y_euler[i + 1] = y_euler[i] - h * x[i]
 
    k_I=-h*x[i]
    k_II=-h*(x[i]+h)
    y_RKII[i+1] = y_RKII[i] + 0.5*(k_I+k_II)


    i += 1
#print(y_RKII)

eul_error=f[n-1]-y_euler[n-1]
RKII_error=f[n-1]-y_RKII[n-1]

print(abs(eul_error),abs(RKII_error))

def d(x,y):
    d=-x
    return d


import matplotlib.pyplot as plt

plt.plot(x, f)

plt.plot(x, y_euler, 'ro')
plt.plot(x, y_RKII, 'bo')

plt.xlabel('x')
plt.ylabel('y')

plt.legend(['analytic solution', 'Eulers method','RKII' ], loc='lower left')
plt.show()