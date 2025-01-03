# RK 4
import numpy as np
import matplotlib.pyplot as plt

# Constants
gamma = 1.4
R = 287

# Functions for temperature and speed of sound
def T(x, t0, m):
    return t0 + m * x

def c(x, t0, m):
    return np.sqrt(gamma * R * T(x, t0, m))

# Define functions for the first derivatives
def f1(x, p, z):
    return z

def f2(x, p, z, t0, m, omega):
    return ((-1/T(x, t0, m)) * m * z) - (omega / c(x, t0, m))**2 * p


# Define the RK4 solver
def rk4_second_order(f1, f2, p0, z0, x0, xf, h, t0, m, omega):

    num_steps = int((xf - x0) / h)  # Number of steps
    x = x0
    p = p0
    z = z0

    # Arrays to store results
    xs = [x0]
    ps = [p0]
    zs = [z0]

    for _ in range(num_steps):
        k1p = h * f1(x, p, z)
        k1z = h * f2(x, p, z, t0, m, omega)

        k2p = h * f1(x + h/2, p + k1p/2, z + k1z/2)
        k2z = h * f2(x + h/2, p + k1p/2, z + k1z/2, t0, m, omega)

        k3p = h * f1(x + h/2, p + k2p/2, z + k2z/2)
        k3z = h * f2(x + h/2, p + k2p/2, z + k2z/2, t0, m, omega)

        k4p = h * f1(x + h, p + k3p, z + k3z)
        k4z = h * f2(x + h, p + k3p, z + k3z, t0, m, omega)

        p += (k1p + 2*k2p + 2*k3p + k4p) / 6
        z += (k1z + 2*k2z + 2*k3z + k4z) / 6
        x += h

        xs.append(x)
        ps.append(p)
        zs.append(z)

    return xs, ps, zs

# Define initial conditions and range
p0 = 2000
z0 = 0
x0 = 0
xf = 4
h = 0.1  # Step size
t0_values = [300, 500, 700, 900, 1100]
m_values = [0, -50, -100, -150, -200]


for i in range(5):

  minimum = []
  f_arr = []
  rho_bar = []
  x_values = np.arange(x0, xf + h, h)
  for xval in x_values:

      c_val = np.sqrt(gamma * R * (t0_values[i] + m_values[i] * xval))
      l = 4
      n = 1  # You can adjust n as needed
      f = n * c_val / (4 * l)
      omega = 2 * np.pi * f
      f_arr.append(f)

      rho_bar.append(101325 / (R * T(xval, t0_values[i], m_values[i])))

      xs, ps, zs = rk4_second_order(f1, f2, p0, z0, x0, xf, h, t0_values[i], m_values[i], omega)
      abs_ps = [abs(val) for val in ps]

      minimum.append(abs_ps[-1])

      #print("F= ",f)
      #print("Pmin= ",abs_ps[-1])

  # Zip minimum and omega2 together and sort based on minimum
  sorted_data = sorted(zip(minimum, f_arr))

  # Unzip the sorted data to separate minimum and omega2
  minimum_sorted, f_arr_sorted = zip(*sorted_data)

  #print("Pressure min = ", minimum_sorted[0])
  print("Freq = ", f_arr_sorted[0])

  xs, ps, zs = rk4_second_order(f1, f2, p0, z0, x0, xf, h, t0_values[i], m_values[i], (2 * np.pi * f_arr_sorted[0]))
  abs_ps = [abs(val) for val in ps]
  #print(np.array(zs))
  #print(np.array(rho_bar))
  abs_vs = np.abs(np.array(zs) / np.array(rho_bar))
  abs_vs = abs_vs / (2 * np.pi * f_arr_sorted[0])

  plt.plot(xs,abs_vs)

plt.xlabel("Distance (m)")
plt.ylabel("Acoustic velocity amplitude (m/s)")
plt.title("Closed-open duct")
plt.legend(['T0=300K m=0', 'T0=500K m=-50', 'T0=700K m=-100', 'T0=900K m=-150', 'T0=1100K m=-200'])
plt.grid()
plt.show()