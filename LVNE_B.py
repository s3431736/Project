import matplotlib.pyplot as plt 
import numpy as np
from scipy.integrate import solve_ivp
#pop_vv_vec=[1,0]


#dim=2
d_matrix=np.zeros((2,2))
#print(d_matrix)

pvc_before_pulse=0
pcv_before_pulse=0
pvv_before_pulse=1      #Ground state  
pcc_before_pulse=0      #excited state

matrix_before_pulse=[[pvc_before_pulse,pvv_before_pulse],[pcv_before_pulse,pcc_before_pulse]]
#print(matrix_before_pulse)


pvc_after_pulse=[1,0]
pcv_after_pulse=np.transpose(pvc_after_pulse)

pvv_after_pulse=0
pcc_after_pulse=1       #excited state


ti=0
tf=20E-12           #s
N=500
hbar=6.582119569E-16	                #eV*s
ihbar=1j*hbar
E_lvl_spacing=1     # eV
trans_dipole_mom=1
dephase_E=1E-3         # eV
pop_cc_=0
pop_vv_i=1

Amp=1
t0=10E-12                   #seconds
pulse_width=50E-16          #seconds
#pulse_width=(pulse_width_fs)*(10**-15)
om=E_lvl_spacing/hbar   # Hz

tev=np.linspace(ti,tf,N)

p0=1+0*1j               

def f(t,p,E_lvl_spacing,ihbar,trans_dipole_mom,dephase_E,t0,om,Amp,pulse_width):
    dp =(1/ihbar)*((1j*dephase_E-E_lvl_spacing)*p)+trans_dipole_mom*(np.exp(Amp*(-(t-t0)**2)/((2*pulse_width)**2))*np.exp(1j*om*t))
    return dp

sol=solve_ivp(f,(ti,tf),[p0],t_eval=tev, args=(E_lvl_spacing,ihbar,trans_dipole_mom,dephase_E,t0,om,Amp,pulse_width))

t=sol.t
p=sol.y



plt.plot(t,p[0])
plt.show()



