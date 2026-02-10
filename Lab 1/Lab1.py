# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %%
import uncertainties.unumpy as unp
from uncertainties import ufloat
import numpy as np
import matplotlib.pyplot as plt

#helper functions:
agree = lambda x,y: (x.n-y.n <= np.sqrt(np.square(x.s)+np.square(y.s)))
v_err = lambda v: 0.013*np.abs(v) + 0.008*(10**np.floor(np.log10(v)))
a_err = lambda a: 0.015*np.abs(a) + 0.008**(10**np.floor(np.log10(a)))
r_err = lambda r: 0.01*np.abs(r) + 0.005*(10**np.floor(np.log10(r))) if r!=0 else 0.005
f_err = lambda f: 0.04*np.abs(r) + 0.5*(10**np.floor(np.log10(r))) if f!=0 else 0.5
u_convert = lambda arr,err_fn: [ufloat(el,err_fn(el)) for el in arr]
u_convert_errs = lambda arr,errs: [ufloat(el,errs[i]) for i,el in enumerate(arr)]

# %%
R1 = ufloat(9900,150)
R2 = ufloat(5010,100)

# %%
V = ufloat(12.09,0.17) - ufloat(-12.10,0.17)
print(f"V: {V:.2f}")

# %%
I = V/(R1+R2)
print(f"I: {I:.3e}")

# %%
V1 = I*R1
V2 = I*R2

# %%
print(f"V1: {V1:.3e}")
print(f"V2: {V2:.3e}")

# %%
V_out = ufloat(12.09,0.17) - V1
print(f"V_out: {V_out:.2f}")

# %%
## measured values

I_m = ufloat(1.602e-3,0.032e-3)
V1_m = ufloat(16.05,0.21)
V2_m = ufloat(8.13,0.13)

R1_m = V1_m/I_m
R2_m = V2_m/I_m

print(f"R1 measured: {R1}")
print(f"R2 measured: {R2}")

# %%
v_err(0.708)

# %%
# 1.7
vrmsD=ufloat(0.708,0.017)
vrmsV=ufloat(0.7071,0.00005)
vrmsS=ufloat(0.69858,0.000005)
print(agree(vrmsD,vrmsS))

vrmsC =ufloat(0.99905,5e-6)/unp.sqrt(2)
print(vrmsC)
print(agree(vrmsC,vrmsS))
print(agree(vrmsC,vrmsD))

# %%
DMM_voltages = [0.667,0.698,0.707,0.708,0.709,0.708,0.705,0.682,0.352,0.039]
DMM_u_voltages = [ufloat(v,v_err(v)) for v in DMM_voltages]

for d in DMM_u_voltages: print(f"{d.s:.3f}")


# %%
#1.8
v_out = ufloat(6.02,v_err(6.02))
print(v_out)
i_out = ufloat(0.594,a_err(0.594))
print(i_out)
R_thn = v_out/i_out
print(R_thn)

# %%
R_ls = u_convert([330,1000,3000,10000,33000,100000],r_err)
v_outs = u_convert([0.193,0.538,1.364,3.00,4.60,5.46],v_err)
for v in v_outs: print(v)
I_outs = [v/r for v,r in zip(v_outs,R_ls)]
for I in I_outs: print(f"{I:.3e}")

xline = np.linspace(v_outs[0].n-0.1,v_outs[-1].n+0.1,100)
thv_plotlines = [xline/r for r in R_ls]

i_outs_predicted = [v/(r_thm+r) for v,r_thm,r in zip(v_outs,[R_thn]*7,R_ls)]

plt.errorbar([v.n for v in v_outs],[i.n for i in I_outs],[i.s for i in I_outs],label="Inferred/Measured current/voltage values")
plt.errorbar([v.n for v in v_outs],[i.n for i in i_outs_predicted],[i.s for i in i_outs_predicted],label=r"Thevenin prediction (only plotting the corresponding $V_{out})$",color="red",fmt="o")

for i,line in enumerate(thv_plotlines): plt.plot(xline,[p.n for p in line],label = f"Thevenin prediction ($R_L$={int(R_ls[i].n)}) for full voltage range",alpha=0.5)

plt.vlines([v.n for v in v_outs],I_outs[-1].n-0.0001,I_outs[0].n+0.0001,alpha=0.1,color="gray")

plt.xticks([v.n for v in v_outs])
plt.tick_params(axis='x', labelrotation=45)
plt.xlim(v_outs[0].n-0.1,v_outs[-1].n+0.1)
plt.ylim(I_outs[-1].n-0.0001,I_outs[0].n+0.0001)
plt.ylabel("Current (Amps)")
plt.xlabel("Voltage (V)")
plt.title("Current Vs Voltage for different values of load resistance")
plt.legend(bbox_to_anchor=(0.5, -0.2),loc="upper center")
plt.show()

# %%
r_dmm = ufloat(9.94e3,r_err(9.94e3))
print(r_dmm)

# %%
# 1.9
r_loads = u_convert([0,1e5,1e6,5.6e6],r_err)
print(r_loads)
v_outs = u_convert_errs([0.999,0.91,0.51,0.15],[5e-4,5e-3,5e-3,5e-2])
print(v_outs)
predicted_Zth = [(r*v_outs[i]/(1-v_outs[i])) for i,r in enumerate(r_loads)]
for z in predicted_Zth: print(f"predicted {z} ohms")

# %%
Z_final = sum(predicted_Zth[1:])/len(predicted_Zth[1:])
print(Z_final)

# %%
#1.10
r_loads = u_convert([33,100,330,1000],r_err)
print(r_loads)
v_loads = u_convert_errs([0.39,0.66,0.86,0.95],[5e-3]*4)
print(v_loads)

# %%
z_wavegens = [(1/(1/r+1/Z_final) * (1/v_loads[i] -1)) for i,r in enumerate(r_loads)]
for z in z_wavegens: print(z)

z_final = sum(z_wavegens)/len(z_wavegens)
print(f"final: {z_final}")

# %%
# 1.11
R_min = 12**2/0.25
print(R_min)

# %%
v_12v_m = ufloat(11.93,r_err(11.93))
r_12v = ufloat(680,r_err(680))
print(r_12v)
print(v_12v_m)

# %%
v_12v = ufloat(12.09,0.17)
print(v_12v)

# %%
z_min = r_12v * (v_12v/v_12v_m - 1)
print(z_min)

# %%
#1.12
r_loads = u_convert([1e2,1e3,1e4,1e5,1e6],r_err)
print(r_loads)
v_open = u_convert_errs([1.09,1.09,1.08,1.03,0.709],[5e-3]*4+[5e-4])
v_loads = u_convert_errs([0.990,0.266,0.473,0.504,0.415],[5e-3]*5)
print(v_loads)

# intermediary value for combined resistance of load voltage and voltmeter voltage
R_c = [(1/(1/r+1/Z_final)) for r in r_loads]
for r in R_c: print(f"R_c = {r}")
zv = Z_final
R_d = [rc*zv*(v0-vl)/(zv*vl - rc*v0) for rc,v0,vl in zip(R_c,v_open,v_loads)]

for r in R_d: print(f"R_d = {r}")


# %%
V_d = [v0*(zv+rd)/zv for v0,rd in zip(v_open,R_d)]
for v in V_d: print(v)

# %%
