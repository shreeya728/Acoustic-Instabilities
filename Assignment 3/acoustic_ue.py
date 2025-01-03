import numpy as np
import math
import matplotlib.pyplot as plt


# Constants
j = 1
omegaj = j * math.pi
kj = j * math.pi
c1 = 0.1
c2 = 0.06
omega1 = math.pi
r = omega1 / omegaj
zetaj = (1 / (2 * math.pi)) * ((c1 * 1 / r) + (c2 * math.sqrt(r)))
#zetaj = 0
#K = 0.025
tau = 0.2
xf = 0.29
gamma = 1.4
ubar = 0.5
c0 = 399.6
M = ubar/c0
# Define initial conditions and range
nj0 = 0.2
nj_dot0 = 0.0
t0 = 0
tf = 80
h = 0.125  # Step size

num_steps = int((tf - t0)/h)

# Define functions for the first derivatives
def f1(nj, nj_dot):
    return nj_dot


def f2(nj, nj_dot, kj, omegaj, zetaj):
    return ((-2 * zetaj * omegaj * nj_dot) - (kj**2 * nj))


def f3(nj, nj_dot, kj, omegaj, zetaj, ufprime):

    #a = ((2 * j * math.pi * K) / (gamma * M)) * math.sin(j * math.pi * xf)
    #b = math.sqrt(abs(1/3 + ufprime)) - math.sqrt(1/3)

    return -2*zetaj*omegaj*nj_dot - (kj**2)*nj - 2*j*math.pi*K*(math.sqrt(abs(1/3 + ufprime)) - math.sqrt(1/3))*math.sin(j*math.pi*xf)



# Define the RK4 solver
def rk4_second_order1(f1, f2, nj0, nj_dot0, h, kj, omegaj, zetaj):
    num_steps1 = int(tau / h)  # Number of steps
    #print(num_steps1)
    #t = t0
    nj = nj0
    nj_dot = nj_dot0


    #ts = [t0]
    njs = [nj0]
    nj_dots = [nj_dot0]
    uprime = []
    pprime = []


    for i in range(num_steps1+1):
        k1nj = h * f1(nj, nj_dot)
        k1nj_dot = h * f2(nj, nj_dot, kj, omegaj, zetaj)

        k2nj = h * f1(nj + k1nj/2, nj_dot + k1nj_dot/2)
        k2nj_dot = h * f2(nj + k1nj/2, nj_dot + k1nj_dot/2, kj, omegaj, zetaj)

        k3nj = h * f1(nj + k2nj/2, nj_dot + k2nj_dot/2)
        k3nj_dot = h * f2(nj + k2nj/2, nj_dot + k2nj_dot/2, kj, omegaj, zetaj)

        k4nj = h * f1(nj + k3nj, nj_dot + k3nj_dot)
        k4nj_dot = h * f2(nj + k3nj, nj_dot + k3nj_dot, kj, omegaj, zetaj)

        nj += (k1nj + 2*k2nj + 2*k3nj + k4nj) / 6
        nj_dot += (k1nj_dot + 2*k2nj_dot + 2*k3nj_dot + k4nj_dot) / 6
        #t += h

        #ts.append(t)
        njs.append(nj)
        nj_dots.append(nj_dot)

        uprime_tval = nj * math.cos(j * math.pi * xf)
        uprime.append(uprime_tval)

        pprime_tval = - ((gamma * M)/(j * math.pi)) * nj_dot * math.sin(j * math.pi * xf)
        pprime.append(pprime_tval)

    #print(len(uprime))
    return njs, nj_dots, uprime, pprime


# Define the RK4 solver
def rk4_second_order2(f1, f2, f3, nj0, nj_dot0, h, kj, omegaj, zetaj):

    nj1, njdot1, u, p = rk4_second_order1(f1, f2, nj0, nj_dot0, h, kj, omegaj, zetaj)



    for i in range(num_steps + 1 - len(u)):
        nj = nj1[-1]
        nj_dot = njdot1[-1]

        k1nj = h * f1(nj, nj_dot)
        k1nj_dot = h * f3(nj, nj_dot, kj, omegaj, zetaj, u[i+1])

        k2nj = h * f1(nj + k1nj/2, nj_dot + k1nj_dot/2)
        k2nj_dot = h * f3(nj + k1nj/2, nj_dot + k1nj_dot/2, kj, omegaj, zetaj, u[i+1])

        k3nj = h * f1(nj + k2nj/2, nj_dot + k2nj_dot/2)
        k3nj_dot = h * f3(nj + k2nj/2, nj_dot + k2nj_dot/2, kj, omegaj, zetaj, u[i+1])

        k4nj = h * f1(nj + k3nj, nj_dot + k3nj_dot)
        k4nj_dot = h * f3(nj + k3nj, nj_dot + k3nj_dot, kj, omegaj, zetaj, u[i+1])

        nj += (k1nj + 2*k2nj + 2*k3nj + k4nj) / 6
        nj_dot += (k1nj_dot + 2*k2nj_dot + 2*k3nj_dot + k4nj_dot) / 6

        #print(nj)
        nj1.append(nj)
        njdot1.append(nj_dot)

        uprime_tval = nj * math.cos(j * math.pi * xf)
        u.append(uprime_tval)

        pprime_tval = - ((gamma * M)/(j * math.pi)) * nj_dot * math.sin(j * math.pi * xf)
        p.append(pprime_tval)

    return nj1, njdot1, u, p


K = 1.4
t_values = np.arange(t0, tf + h, h)

nj2, njdot2, u2, p2 = rk4_second_order2(f1, f2, f3, nj0, nj_dot0, h, kj, omegaj, zetaj)

E = [((0.5 * a**2) + (0.5 * (gamma * M * b)**2))/((gamma * M)**2) for a, b in zip(p2, u2)]

from scipy.signal import find_peaks
E_flat = np.hstack(E)
peaks, _ = find_peaks(E_flat)
#valleys, _ = find_peaks(-E_flat)

plt.plot(t_values, E)
plt.plot(t_values[peaks], E_flat[peaks], "x", color='red')
#plt.plot(t_values[valleys], E_flat[valleys], "x", color='red')
#plt.ylim([-0.2, 0.2])
plt.xlabel('t')
plt.ylabel(r'Energy/($\gamma M)^2$')
#plt.ylabel("u'")
plt.show()