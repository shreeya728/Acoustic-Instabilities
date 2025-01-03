
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
j = 1
omegaj = j * math.pi
kj = j * math.pi
c1 = 0.1
c2 = 0.06
omega1 = math.pi
r = omega1 / omegaj
zetaj = (1 / (2 * math.pi)) * ((c1 * 1 / r) + (c2 * math.sqrt(r)))

#tau = 0.3
xf = 0.29
gamma = 1.4
ubar = 0.5
c0 = 399.6
M = ubar/c0

#nj0 = 0.2
#nj_dot0 = 0.0
t0 = 0
tf = 40
h = 0.1  # Step size

num_steps = int((tf - t0)/h)
U2 = []
U3 = []
K_values = np.linspace(0.1, 2, 20)
tau_values = np.linspace(0.2, 0.8, 20)
Kflip = np.flip(K_values)

  # Define functions for the first derivatives
def f1(nj, nj_dot):
    return nj_dot


def f2(nj, nj_dot, kj, omegaj, zetaj):
    return ((-2 * zetaj * omegaj * nj_dot) - (kj**2 * nj))


def f3(nj, nj_dot, kj, omegaj, zetaj, ufprime, K):
    return -2*zetaj*omegaj*nj_dot - (kj**2)*nj - 2*j*math.pi*K*(math.sqrt(abs(1/3 + ufprime)) - math.sqrt(1/3))*math.sin(j*math.pi*xf)


  # Define the RK4 solver
def rk4_second_order1(f1, f2, nj0, nj_dot0, h, kj, omegaj, zetaj, tau):
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


    for i in range(num_steps1 + 1):
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

    return njs, nj_dots, uprime


  # Define the RK4 solver
def rk4_second_order2(f1, f2, f3, nj0, nj_dot0, h, kj, omegaj, zetaj, K, tau):

    nj1, njdot1, u= rk4_second_order1(f1, f2, nj0, nj_dot0, h, kj, omegaj, zetaj, tau)

    for i in range(num_steps + 1 - len(u)):
        nj = nj1[-1]
        nj_dot = njdot1[-1]

        k1nj = h * f1(nj, nj_dot)
        k1nj_dot = h * f3(nj, nj_dot, kj, omegaj, zetaj, u[i+1], K)

        k2nj = h * f1(nj + k1nj/2, nj_dot + k1nj_dot/2)
        k2nj_dot = h * f3(nj + k1nj/2, nj_dot + k1nj_dot/2, kj, omegaj, zetaj, u[i+1], K)

        k3nj = h * f1(nj + k2nj/2, nj_dot + k2nj_dot/2)
        k3nj_dot = h * f3(nj + k2nj/2, nj_dot + k2nj_dot/2, kj, omegaj, zetaj, u[i+1], K)

        k4nj = h * f1(nj + k3nj, nj_dot + k3nj_dot)
        k4nj_dot = h * f3(nj + k3nj, nj_dot + k3nj_dot, kj, omegaj, zetaj, u[i+1], K)

        nj += (k1nj + 2*k2nj + 2*k3nj + k4nj) / 6
        nj_dot += (k1nj_dot + 2*k2nj_dot + 2*k3nj_dot + k4nj_dot) / 6

          #print(nj)
        nj1.append(nj)
        njdot1.append(nj_dot)

        uprime_tval = nj * math.cos(j * math.pi * xf)
        u.append(uprime_tval)

    return nj1, njdot1, u


Urms_values = np.zeros((len(tau_values), len(K_values)))

Urms2 = np.zeros((len(tau_values), len(Kflip)))
#Urms_values = []
i = 0
j1 = 0
k = 0
# Calculate Urms for each combination of K and tau
for tau in tau_values:
    nj = 0.2
    nj_dot = 0.0
    for K in K_values:
        nj2, njdot2, u2 = rk4_second_order2(f1, f2, f3, nj, nj_dot, h, kj, omegaj, zetaj, K, tau)
        subset = u2[len(u2)//2:]
        Urms = np.sqrt(np.mean(np.square(subset)))
        Urms_values[i, j1] = Urms
        nj = nj2[-1]
        nj_dot = njdot2[-1]
        j1 = j1 + 1

    nj = 0.6
    nj_dot = 0.1
    for K in Kflip:
        nj3, njdot3, u3 = rk4_second_order2(f1, f2, f3, nj, nj_dot, h, kj, omegaj, zetaj, K, tau)
        subset = u3[len(u2)//2:]
        U_rms = np.sqrt(np.mean(np.square(subset)))
        Urms2[i, k] = U_rms
        nj = nj3[-1]
        nj_dot = njdot3[-1]
        k = k + 1
    i = i + 1
    j1 = 0
    k = 0

# Create mesh grid for K and tau
tau_mesh, K_mesh = np.meshgrid(tau_values, K_values)

tau_mesh2, K_mesh2 = np.meshgrid(tau_values, Kflip)

# Plot 3D bifurcation plot with z-axis on the left
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(tau_mesh, K_mesh, Urms_values.T, cmap='cool')
ax.plot_surface(tau_mesh2, K_mesh2, Urms2.T, cmap='plasma')
ax.set_xlabel('$\\tau$')
ax.set_ylabel('K')
ax.set_zlabel('U')
plt.show()