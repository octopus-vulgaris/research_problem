from math import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols
from sympy import solveset, solve
import sympy
#from sympy import init_session
#init_session()
from sympy import solve_poly_system

import numpy as np
from scipy.optimize import root

""" some vars
    L_1y, L_1x, L_2x, L_2y = 1.0, 1.0, 1.0, 1.0
    k_1 = 1.0
    k_2 = 1.3

    L_1 = (L_1x**2 + L_1y**2)**0.5
    L_2 = sqrt(L_2x**2 + L_2y**2)
 """

def F_d(d, k = 1., L_x = 1., L_y = 1.):
    # d - spring deformation
    L = (L_x**2 + L_y**2)**0.5 # spring length at initial moment
    L_d = (L_x**2 + (L_y - d)**2)**0.5 # length after deformation
    f_s = k*(L - L_d) # spring force
    f_d = f_s * (L_y - d)/L_d # force to piston
    return f_d

# x_FD test data
def func(x):
    return F_d(x) - F_d(0.2345)

sol = root(func, 0.25)
sol.x
sol.fun

print(F_d(0.2345, 1.3))

# Inverse function. It requares good starting points.
def y_Fd(f_d, k = 1.3):
    kk = k
    x1 = 0
    def func(x):
        return F_d(x, kk) - f_d
    
    sol = root(func, x1)
    return float(sol.x)

# plotting
#region

x = np.linspace(-1, 3, 100)  # Sample data.
nlist = [y_Fd(F_d(i, k = 1), k = 1) for i in x]
nnlist = [F_d(i, k = 1) for i in nlist]

for i in nlist:
    print(i)

# Note that even in the OO-style, we use `.pyplot.figure` to create the Figure.
fig, ax = plt.subplots(figsize=(5, 5), layout='constrained')
ax.plot(x, F_d(x, k = 1), label='k = 1.0')  # Plot some data on the axes.
#ax.plot(x, nlist, label='y(F_d)')
ax.plot(x, F_d(x, k = 1.3), label='k = 1.3')  # Plot more data on the axes...
#ax.plot(x, F_d(x)[2], label='f_s')  # ... and some more.
ax.set_xlabel('x label')  # Add an x-label to the axes.
ax.set_ylabel('y label')  # Add a y-label to the axes.
ax.set_title("Simple Plot")  # Add a title to the axes.
ax.legend()  # Add a legend.

plt.show()
#endregion