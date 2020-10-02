import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def linear(ti=0,
           tf=20E-12,
           N=400,
           bg=1.424,
           meff_el=0.067,
           meff_holes=0.48,
           dip_moment=1e-3,
           dephase_E = 0.5E-3,
           Amp=1,
           t0=5E-12,
           pulse_width=1000E-16,
           dim=3,
           ne=0,
           nh=1):

    # ------------------ time scale -------------------

    tev = np.linspace(ti, tf, N, endpoint=False)

    # ---- constants and parameters of the system -----

    hhbar = 1.0545718e-34   # J*s
    hbar = 6.582119569E-16  # eV*s
    ihbar = 1j * hbar

    e_mass = 9.10938356E-31
    meff_el *= e_mass
    meff_holes *= e_mass
    q = 1.6e-19
    c = 3.0e8
    # eV
    wavevectors = np.linspace(0, 0.2, 100) * 1E9

    E_cb = bg * q + hhbar ** 2 * wavevectors ** 2 / (2 * meff_el)
    E_vb = -hhbar ** 2 * wavevectors ** 2 / (2 * meff_holes)
    E_lvl_spacing = (E_cb - E_vb) / q

    trans_dipole_mom = dip_moment

    # -------- parameters of optical pulse -------------
    om = E_lvl_spacing / hbar  # Hz

    # -------------- initial conditions ----------------
    p0 = np.zeros(wavevectors.shape, dtype=np.complex)

    # function computing the EM pulse
    def e_pulse(t, t0, omm, Amp, pulse_width):
        return Amp * np.exp((-(t - t0) ** 2) / ((2 * pulse_width) ** 2)) * np.exp(1j * omm * t)

    # RHS of the equation (version 1)
    def f(t, p, E_lvl_spacing, ihbar, trans_dipole_mom, dephase_E, t0, omm, Amp, pulse_width):
        dp = -np.diag((1 / ihbar) * (-1j * dephase_E - E_lvl_spacing)) @ p + \
             (1 / ihbar) * (ne-nh) * trans_dipole_mom * e_pulse(t, t0, omm, Amp, pulse_width)
        return dp

    # RHS of the equation (version 2)
    def f1(t, p, E_lvl_spacing, ihbar, trans_dipole_mom, dephase_E, t0, omm, Amp, pulse_width):
        dp = np.diag((1 / ihbar) * (-1j * dephase_E - E_lvl_spacing + om[0] * hbar)) @ p + \
             (1 / ihbar) * (ne-nh) * trans_dipole_mom * e_pulse(t, t0, omm * 0, Amp, pulse_width)
        return dp

    # --------------------- solver ---------------------
    sol = solve_ivp(f1, (ti, tf), p0, t_eval=tev, first_step=tf / N, dense_output=True, atol=1e-9,
                    args=(E_lvl_spacing, ihbar, trans_dipole_mom, dephase_E, t0, om[0], Amp, pulse_width))

    t = sol.t
    p = sol.y

    # plt.contourf(np.real(p))
    # plt.show()

    pump_signal = e_pulse(t, t0, 0 * om[0], Amp, pulse_width)
    norm = np.max(np.abs(p))

    omegas = np.linspace(-0.01, 0.01, 500) + 1.0 * 0
    pf = np.zeros((len(omegas), len(wavevectors)), dtype=np.complex)
    eef = np.zeros((len(omegas),), dtype=np.complex)

    for j, item in enumerate(omegas):
        omega = item / hbar
        # print(omega * t)
        pf[j, :] = np.trapz(p * np.exp(-1j * omega * t), x=t, axis=1)
        ef = np.trapz(pump_signal * np.exp(-1j * omega * t), x=t)
        if dim == 3:  # 3D
            pf[j, :] = (4 * np.pi * wavevectors ** 2) * pf[j, :] * dip_moment / ef
        elif dim == 2:  # 2D
            pf[j, :] = (2 * np.pi * wavevectors) * pf[j, :] * dip_moment/ ef
        else:  # 1D
            pf[j, :] = 2 * pf[j, :] * dip_moment/ ef

        eef[j] = ef

    # 1D
    pf = 2 * np.sum(pf, axis=1)

    pf = pf * (omegas+bg) * q / hhbar / c / 3

    # plt.plot(omegas, np.imag(pf))
    #
    # plt.ylabel("absorption")
    # plt.xlabel("Frequency")
    # plt.show()
    #
    # plt.plot(omegas, np.real(pf) + 1)
    # plt.ylabel("Polarization")
    # plt.xlabel("Frequency")
    # plt.show()
    #
    # plt.plot(omegas, eef)
    #
    # plt.ylabel("EM Pulse")
    # plt.xlabel("Frequency")
    # plt.show()

    return np.imag(pf)


if __name__=='__main__':

    ans = linear(dim=2)