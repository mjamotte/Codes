import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.axes as axes
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from numpy import linalg
import cmath
import time
from mpl_toolkits import mplot3d

import sys
sys.path.append('/home/maxime/Desktop/Codes/')
import lib_GrossPitaevskii as GP
import lib_Manage_files as Mf

#### COMPARISON OF ENERGIES FOR IT AND SC ######

## Figure

fig_OL1D = pyplot.figure(figsize=(8,8))
ax1 = pyplot.subplot(111)
ax1.set_xlabel('U')

## Initialize some variables

U_IT_s = np.array([])
U_SC_s = np.array([])
mu_IT_s = np.array([])
mu_SC_s = np.array([])
E_kin_s_IT = np.array([])
E_tr_s_IT = np.array([])
E_U_s_IT = np.array([])
E_kin_s_SC = np.array([])
E_tr_s_SC = np.array([])
E_U_s_SC = np.array([])

Efun_SC = np.array([]) # to compute mu = dE_SC/dN
Efun_IT = np.array([]) # to compute mu = dE_IT/dN

## Parameters to pick the datas from Datas_SC nd Datas_IT

list_args = ['N','V','U']
split = ['N_','V_','U_']
params_interv = {'U': [1e-3],'V': [0.5e-4],'N': [1e4-1,1e4+1]}
extension = '.npy'

## Import datas from Datas_IT

path = '/home/maxime/Desktop/Codes/Optical_1D_Lattice_BEC/Imaginary_time/Datas_IT/'
files_names = Mf.select_files(path,extension,list_args,params_interv,split)
split = ['N_']
list_args = ['N']
files_names = Mf.reorder_filesnames(files_names,list_args,split)

for filename in files_names:

	data = np.load(filename+'.npy',allow_pickle=True)
	args_syst = data[0]
	J = args_syst['J']
	Nx = args_syst['Nx']
	V = args_syst['V']
	U_IT = args_syst['U']
	N = args_syst['N']

	args_init = data[1]
	#mu_all = data['mu_all']	
	mu_IT = data[2]
	psi0_IT = data[3]
	E_funct_IT = data[4]
	n0_IT = np.abs(psi0_IT)**2 

	Trap = GP.trap_1D(args_syst)

	Efun_IT = np.append(Efun_IT,E_funct_IT)
	U_IT_s = np.append(U_IT_s,U_IT)
	mu_IT_s = np.append(mu_IT_s,mu_IT)
	E_kin_s_IT = np.append(E_kin_s_IT,GP.E_kin(psi0_IT,args_syst))
	E_tr_s_IT = np.append(E_tr_s_IT,GP.E_trap(psi0_IT,args_syst))
	E_U_s_IT = np.append(E_U_s_IT,GP.E_int(psi0_IT,args_syst))


## Import datas for SC

path = '/home/maxime/Desktop/Codes/Optical_1D_Lattice_BEC/Self_consistent/Datas_SC/'
list_args = ['N','V','U']
split = ['N_','V_','U_']
extension = '.npy'
files_names = Mf.select_files(path,extension,list_args,params_interv,split)
split = ['N_']
list_args = ['N']
files_names = Mf.reorder_filesnames(files_names,list_args,split)

for filename in files_names:

	data = np.load(filename+'.npy',allow_pickle=True)
	args_syst = data[0]
	J = args_syst['J']
	Nx = args_syst['Nx']
	V = args_syst['V']
	U_SC = args_syst['U']
	N = args_syst['N']

	args_init = data[1]
	#mu_all = data['mu_all']	
	mu_SC = data[2]
	psi0_SC = data[3]
	E_funct_SC = data[4]

	## Analytical gaussian 

	Trap = GP.trap_1D(args_syst)
	m = 1/(2*J)
	w0 = np.sqrt(V/m)
	x0 = (Nx-1)/2
	positions = np.arange(Nx)

	Efun_SC = np.append(Efun_SC,E_funct_SC)
	U_SC_s = np.append(U_SC_s,U_SC)
	mu_SC_s = np.append(mu_SC_s,mu_SC)
	E_kin_s_SC = np.append(E_kin_s_SC,GP.E_kin(psi0_SC,args_syst))
	E_tr_s_SC = np.append(E_tr_s_SC,GP.E_trap(psi0_SC,args_syst))
	E_U_s_SC = np.append(E_U_s_SC,GP.E_int(psi0_SC,args_syst))

## Compute mu = dE/dN

dN = 1
dEfun_SCdN = GP.dEdN_O2(Efun_SC,dN)
dEfun_ITdN = GP.dEdN_O2(Efun_IT,dN)	

## Plots

ax1.semilogx(U_IT_s,Efun_IT/N,'ks',fillstyle='none',label="$E^{IT}[\psi]/N$")
ax1.semilogx(U_IT_s,E_kin_s_IT/N+2*J,'bs',fillstyle='none',label="$E^{IT}_{kin}[\psi]/N$")
ax1.semilogx(U_IT_s,E_tr_s_IT,'gs',fillstyle='none',label="$E^{IT}_{trap}[\psi]/N$")
ax1.semilogx(U_IT_s,E_U_s_IT,'rs',fillstyle='none',label="$E^{IT}_U[\psi]/N$")
ax1.semilogx(U_IT_s,mu_IT_s,'cs',fillstyle='none',label="$\mu^{IT}[\psi]/N$")
ax1.semilogx(U_IT,dEfun_ITdN,'*',fillstyle='none',label="$dE^{IT}/dN$")

ax1.semilogx(U_SC_s,Efun_SC/N,'k^',fillstyle='none',label="$E^{SC}[\psi]/N$")
ax1.semilogx(U_SC_s,E_kin_s_SC/N+2*J,'b^',fillstyle='none',label="$E^{SC}_{kin}[\psi]/N$")
ax1.semilogx(U_SC_s,E_tr_s_SC,'g^',fillstyle='none',label="$E^{SC}_{trap}[\psi]/N$")
ax1.semilogx(U_SC_s,E_U_s_SC,'r^',fillstyle='none',label="$E^{SC}_U[\psi]/N$")
ax1.semilogx(U_SC_s,mu_SC_s,'c^',fillstyle='none',label="$\mu^{SC}[\psi]/N$")
ax1.semilogx(U_SC,dEfun_SCdN,'+',fillstyle='none',label="$dE^{SC}/dN$")

pyplot.suptitle('Comparison of energies IT and SC for %s, %s,\
	 Nx = %.1i, J = %.2f, V = %.3e' % \
	 (args_syst['Trap'],args_syst['Symm'],\
	args_syst['Nx'],args_syst['J'],args_syst['V']))

ax1.legend(loc=2);
ax1.grid(axis='both')
pyplot.show()

temp = '1D_comp_energy_ITSC_%s_%s_Nx_%.1i_J_%.2f_V_%.3e' %\
		(args_syst['Trap'],args_syst['Symm'],\
		 args_syst['Nx'],args_syst['J'],args_syst['V'])
