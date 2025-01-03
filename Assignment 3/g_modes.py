
import numpy as np
import math
import matplotlib.pyplot as plt



#K = 0.025
tau = 0.2
xf = 0.29
gamma = 1.4
ubar = 0.5
c0 = 399.6
M = ubar/c0
# Define initial conditions and range
nj0 = 0.18
nj_dot0 = 0.0
t0 = 0
tf = 40
h = 0.125  # Step size

num_steps = int((tf - t0)/h)

# Define functions for the first derivatives
def f1(nj, nj_dot):
    return nj_dot


def f2(nj, nj_dot, kj, omegaj, zetaj):
    return ((-2 * zetaj * omegaj * nj_dot) - (kj**2 * nj))


def f3(nj, nj_dot, kj, omegaj, zetaj, ufprime):

    return -2*zetaj*omegaj*nj_dot - (kj**2)*nj - 2*j*math.pi*K*(math.sqrt(abs(1/3 + ufprime)) - math.sqrt(1/3))*math.sin(j*math.pi*xf)



# Define the RK4 solver
def rk4_second_order1(f1, f2, nj0, nj_dot0, h, kj, omegaj, zetaj):
    num_steps1 = int(tau / h)  # Number of steps
    #print(num_steps1)
    #t = t0
    nj = nj0
    nj_dot = nj_dot0


    #ts = [t0]
    if j==1:
      njs = [nj0]
    else:
      njs = [0]
    nj_dots = [nj_dot0]
    uprime = []
    #pprime = []


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

        if j==1:
          uprime_tval = nj * math.cos(j * math.pi * xf)
        elif j==2:
          uprime_tval = nj * math.cos(j * math.pi * xf) + u2[0][i]
        else:
          uprime_tval = nj * math.cos(j * math.pi * xf) + u2[0][i] + u2[1][i]

        uprime.append(uprime_tval)

    return njs, nj_dots, uprime


# Define the RK4 solver
def rk4_second_order2(f1, f2, f3, nj0, nj_dot0, h, kj, omegaj, zetaj):

    nj1, njdot1, u = rk4_second_order1(f1, f2, nj0, nj_dot0, h, kj, omegaj, zetaj)

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

        if j==1:
          uprime_tval = nj * math.cos(j * math.pi * xf)
        elif j==2:
          uprime_tval = nj * math.cos(j * math.pi * xf) + u2[0][i]
        else:
          uprime_tval = nj * math.cos(j * math.pi * xf) + u2[0][i] + u2[1][i]

        u.append(uprime_tval)


    return nj1, njdot1, u


K = 1.8
c1 = 0.1
c2 = 0.06
t_values = np.arange(t0, tf + h, h)
# Initialize lists for storing results
nj2 = []
njdot2 = []
u2 = []


# Loop through different j values
for j in range(1, 4):
    omegaj = j * np.pi
    kj = j * np.pi
    zetaj = (1 / (2 * np.pi)) * ((c1 * j * np.pi) / (np.pi) + (c2 * np.sqrt(np.pi / (j * np.pi))))

    nj3, njdot3, u3= rk4_second_order2(f1, f2, f3, nj0, nj_dot0, h, kj, omegaj, zetaj)
    nj2.append(nj3)
    njdot2.append(njdot3)
    u2.append(u3)

# Plot nj2 for j=2
nj2[0].pop()
nj2[1].pop()
nj2[2].pop()
# Create subplots
fig, axs = plt.subplots(2, 2)

# Plot data on each subplot
axs[0, 0].plot(t_values, u2[0])
axs[0, 0].set_title("u'")

axs[0, 1].plot(t_values, nj2[0])
axs[0, 1].set_title('n1')

axs[1, 0].plot(t_values, nj2[1])
axs[1, 0].set_title('n2')

axs[1, 1].plot(t_values, nj2[2])
#axs[1, 1].set_ylim([-0.2, 0.2])
axs[1, 1].set_title('n3')

# Adjust layout to prevent overlapping titles
plt.tight_layout()

# Show the plot
plt.show()