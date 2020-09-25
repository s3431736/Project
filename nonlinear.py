import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

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

# -------- parameters of Gallium Arsenide -------------

bg_GaAs = 1.424
meff_el = 0.067
meff_holes = 0.48

# ------------------ time scale -------------------

ti = 0
tf = 20E-12  # s
N = 400
tev = np.linspace(ti, tf, N, endpoint=False)

# ---- constants and parameters of the system -----

Tempr = 10
kb = 1.380649e-23

relax_time = 1e-11

hhbar = 1.0545718e-34   # J*s
hbar = 6.582119569E-16  # eV*s
ihbar = 1j * hbar

e_mass = 9.10938356E-31
meff_el *= e_mass
meff_holes *= e_mass
nb = 1.424
c = 3E8
pi = 3.141592653589793
q = 1.6e-19
# eV
wavevectors = np.linspace(0, 0.2, 100) * 1E9

E_cb = bg_GaAs * q + hhbar ** 2 * wavevectors ** 2 / (2 * meff_el)
E_vb = -hhbar ** 2 * wavevectors ** 2 / (2 * meff_holes)
E_lvl_spacing = (E_cb - E_vb) / q
# E_lvl_spacing=hhbar**2*wavevectors**2 / 2*eff_mass

trans_dipole_mom = np.zeros(wavevectors.shape) + 1E-3
dephase_E = 1E-3 / 5  # eV
pop_cc = 0
pop_vv_i = 1

# -------- parameters of optical pulse -------------
Amp = 3
t0 = 5E-12  # seconds
pulse_width = 1000E-16  # seconds
# pulse_width=(pulse_width_fs)*(10**-15)
om = E_lvl_spacing / hbar  # Hz

# -------------- initial conditions ----------------
Ef = 0.5 * bg_GaAs * q

p0 = np.zeros(wavevectors.shape, dtype=np.complex)
pop_cb = 1.0/(1.0+np.exp((E_cb - Ef)/(kb*Tempr)))
pop_vb = 1 - 1.0/(1.0+np.exp(-(E_vb - Ef)/(kb*Tempr)))

p0 = np.hstack([p0, pop_cb, pop_vb])

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
    len_ar = p.shape[0] // 3

    pol = p[:len_ar]
    ne = p[len_ar:2*len_ar]
    nh = p[2 * len_ar: 3 * len_ar]

    dp = np.diag((1 / ihbar) * (-1j * dephase_E - E_lvl_spacing + om[0] * hbar)) @ pol + \
         (1 / ihbar) * (ne-nh) * trans_dipole_mom * e_pulse(t, t0, omm * 0, Amp, pulse_width)

    dh = (2.0 / hbar) * np.imag(trans_dipole_mom * e_pulse(t, t0, omm * 0, Amp, pulse_width) * np.conj(pol)) -\
         (nh-pop_vb) / relax_time
    de = -(2.0 / hbar) * np.imag(trans_dipole_mom * e_pulse(t, t0, omm * 0, Amp, pulse_width) * np.conj(pol)) -\
         (ne-pop_cb) / relax_time

    return np.hstack([dp, de, dh])


# --------------------- solver ---------------------
sol = solve_ivp(f1, (ti, tf), p0, t_eval=tev, first_step=tf / N, dense_output=True,
                args=(E_lvl_spacing, ihbar, trans_dipole_mom, dephase_E, t0, om[0], Amp, pulse_width))

t = sol.t
ans = sol.y
len_ar = ans.shape[0] // 3

p = ans[:len_ar, :]
ne = ans[len_ar:2*len_ar, :]
nh = ans[2*len_ar:3*len_ar, :]

plt.contourf(np.real(p))
plt.show()

plt.contourf(ne)
plt.show()

plt.contourf(nh)
plt.show()

plt.plot(ne[0,:])
plt.plot(nh[0,:])
plt.show()

pump_signal = e_pulse(t, t0, 0 * om[0], Amp, pulse_width)
norm = np.max(np.abs(p))

omegas = np.linspace(-0.01, 0.01, 500) + 1.0 * 0
pf = np.zeros((len(omegas), len(wavevectors)), dtype=np.complex)
eef = np.zeros((len(omegas),), dtype=np.complex)

for j, item in enumerate(omegas):
    omega = item / hbar
    print(omega * t)
    pf[j, :] = np.trapz(p * np.exp(-1j * omega * t), x=t, axis=1)
    ef = np.trapz(pump_signal * np.exp(-1j * omega * t), x=t)
    pf[j, :] = pf[j, :] / ef
    eef[j] = ef

# 1D
pf = 2 * np.sum(pf, axis=1)

plt.plot(omegas, np.imag(pf))

plt.ylabel("absorption")
plt.xlabel("Frequency")
plt.show()

plt.plot(omegas, np.real(pf) + 1)
plt.ylabel("Polarization")
plt.xlabel("Frequency")
plt.show()

plt.plot(omegas, eef)

plt.ylabel("EM Pulse")
plt.xlabel("Frequency")
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
# plt.ylabel("absorption")
# plt.xlabel("Frequency")
# #plt.ylabel("Polarization (a.u.)")
# plt.show()
