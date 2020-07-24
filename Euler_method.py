import numpy as np

x0 = 0
xf = 10
y0 = 5

n = 100
h = (xf - x0) / n
# print(h)

x = np.linspace(x0, xf, n)

# print(x)
# print(type(x))
# print(len(xsplit))


y_plus = np.zeros(n)
# y_plus=np.linspace(x0,xf,n)
f = np.zeros(n)
# f=np.linspace(x0,xf,n)

# print(len(y_plus[0]))
# print(len(f[0]))
# print(type(xsplit))

f = 5 - 0.5 * (x ** 2)

i = 0
y_plus[0] = y0
while i < (n - 1):
    y_plus[i + 1] = y_plus[i] - h * x[i]
    i += 1

# error=f[0][n-1]-y_plus[0][n-1]
# print(error)

import matplotlib.pyplot as plt

plt.plot(x, f)

plt.plot(x, y_plus, 'ro')

plt.xlabel('x')
plt.ylabel('y')

plt.legend(['analytic solution', 'approximation'], loc='lower left')
plt.show()