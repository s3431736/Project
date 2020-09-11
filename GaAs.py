import matplotlib.pyplot as plt
import numpy as np
from numpy import trapz
from scipy.integrate import solve_ivp
from scipy.fft import fft
from scipy.fft import fftfreq
from scipy.fft import fftshift, ifftshift
# pop_vv_vec=[1,0]


# dim=2
d_matrix = np.zeros((2, 2))
# print(d_matrix)

pvc_before_pulse = 0
pcv_before_pulse = 0
pvv_before_pulse = 1  # Ground state
pcc_before_pulse = 0  # excited state

matrix_before_pulse = [[pvc_before_pulse, pvv_before_pulse], [pcv_before_pulse, pcc_before_pulse]]
# print(matrix_before_pulse)


pvc_after_pulse = [1, 0]
pcv_after_pulse = np.transpose(pvc_after_pulse)

pvv_after_pulse = 0
pcc_after_pulse = 1  # excited state

# ------------------ time scale -------------------
ti = 0
tf = 20E-12  # s
N = 400
tev = np.linspace(ti, tf, N,  endpoint = False )
# ---- constants and parameters of the system -----
hbar = 6.582119569E-16  # eV*s
ihbar = 1j * hbar
e_mass=9.10938356E-31
eff_mass=e_mass*0.067
nb=3.6
c=3E8
pi= 3.141592653589793
q=1.6e-19
  # eV
wavevectors=np.linspace(0, 0.2, 100)*1E9
hhbar = 1.0545718e-34
E_lvl_spacing = hhbar*wavevectors**2 / (2*eff_mass) * hhbar / q + 1
trans_dipole_mom = np.zeros(wavevectors.shape)+1E-3
dephase_E = 1E-3/5  # eV
pop_cc_ = 0
pop_vv_i = 1

# -------- parameters of optical pulse -------------
Amp = 1
t0 = 5E-12  # seconds
pulse_width = 1000E-16  # seconds
# pulse_width=(pulse_width_fs)*(10**-15)
om = E_lvl_spacing / hbar  # Hz

# -------------- initial conditions ----------------
p0 = np.zeros(wavevectors.shape, dtype=np.complex)


# function computing the EM pulse
def e_pulse(t, t0, omm, Amp, pulse_width):
    return Amp * np.exp((-(t - t0) ** 2) / ((2 * pulse_width) ** 2)) * np.exp(1j * omm * t)


# RHS of the equation (version 1)
def f(t, p, E_lvl_spacing, ihbar, trans_dipole_mom, dephase_E, t0, omm, Amp, pulse_width):
    dp = -np.diag((1 / ihbar) * (-1j * dephase_E - E_lvl_spacing)) @ p - \
         (1 / ihbar) * trans_dipole_mom * e_pulse(t, t0, omm, Amp, pulse_width)
    return dp


# RHS of the equation (version 2)
def f1(t, p, E_lvl_spacing, ihbar, trans_dipole_mom, dephase_E, t0, omm, Amp, pulse_width):
    dp = np.diag((1 / ihbar) * (-1j * dephase_E - E_lvl_spacing + om[0] * hbar)) @ p - \
         (1 / ihbar) * trans_dipole_mom * e_pulse(t, t0, omm*0, Amp, pulse_width)
    return dp

# --------------------- solver ---------------------
sol = solve_ivp(f1, (ti, tf), p0, t_eval=tev, first_step=tf/N, dense_output=True,
                args=(E_lvl_spacing, ihbar, trans_dipole_mom, dephase_E, t0, om[0], Amp, pulse_width))


t = sol.t
p = sol.y

plt.contourf(np.real(p))
plt.show()

pump_signal = e_pulse(t, t0, 0*om[0], Amp, pulse_width)
norm = np.max(np.abs(p))

omegas = np.linspace(-0.01, 0.01, 500) + 1.0 * 0
pf = np.zeros((len(omegas), len(wavevectors)), dtype=np.complex)
eef = np.zeros((len(omegas),), dtype=np.complex)

for j, item in enumerate(omegas):
    omega = item / hbar
    print(omega*t)
    pf[j, :] = np.trapz(p * np.exp(-1j*omega*t), x=t, axis=1)
    ef = np.trapz(pump_signal * np.exp(-1j*omega*t), x=t)
    pf[j, :] = pf[j, :] / ef
    eef[j] = ef

pf = np.sum(pf, axis=1)

plt.plot(omegas, np.imag(pf))
plt.show()

plt.plot(omegas, np.real(pf)+1)
plt.show()

plt.plot(omegas, eef)
plt.show()

# pf = ifftshift(fft(fftshift(p)))
# Ef = ifftshift(fft(fftshift(pump_signal)))
# X = pf / Ef
#
# freq = fftshift(fftfreq(len(p), t[3] - t[2]))
#
# plt.plot(freq*6.05e-34/1.6e-19, np.abs(X))
# plt.show()
#
# mask= freq > 0
#
#
# aw= abs((pi*om/nb*c)*np.real(X))
# aw_norm=np.max(np.abs(aw[0]))
#
# aw_true=2*abs(aw/N)
#
# #numerical integration of EM pulse
# 
# Exp=np.exp(-2*pi*1j*t*om)
# func=Exp*e_pulse(t, t0, om, Amp, pulse_width)
# integration=np.trapz(t,func)
# #plt.plot(t/1e-12, e_pulse(t, t0, om, Amp, pulse_width))
# 
# 
# #plt.plot(t/1e-12, Ef)
# 
# 
# plt.plot(freq, aw[0]/aw_norm)
# #plt.plot(t/1e-12, np.real(pf[0])/norm)
# #plt.plot(t/1e-12, np.imag(pf[0])/norm)
# #plt.legend(['EM pump pulse', 'Polarization, real part', 'Polarization, imaginary part'])
# #plt.xlabel("Time (ps)")
# 
# plt.xlabel("Frequency")
# plt.ylabel("absorption")
# #plt.ylabel("Polarization (a.u.)")
# plt.show()