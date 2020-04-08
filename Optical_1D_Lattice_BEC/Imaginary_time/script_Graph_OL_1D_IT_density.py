import numpy as np
import scipy as sc
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as pyplot
import matplotlib.axes as axes
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from numpy import linalg
import time

import sys
sys.path.append('/home/maxime/Desktop/Codes/')
import lib_GrossPitaevskii as GP
import lib_Manage_files as Mf

########################## Graphics for OPTICAL LATTICE ##########################

E_funct_IT_s = np.array([])
diff_all = np.array([])
args_diff_all = np.array([])

## Figure

fig_OL1D = pyplot.figure(figsize=(8,8))
ax2 = pyplot.subplot(111)
ax2.set_xlabel('$x$')
ax2.set_ylabel('$|\psi_0|^2$')#,color='r')

## Select the files you want to plot
path = '/home/maxime/Desktop/Codes/Optical_1D_Lattice_BEC/Imaginary_time/Datas_IT/'
list_args = ['N','V','U']
split = ['N_','V_','U_']
params_interv = {'U': [1e-2],'V': [1e-5],'N': [10,100]}
extension = '.npy'
files_names = Mf.select_files(path,extension,list_args,params_interv,split)

for filename in files_names:

	data = np.load(filename+'.npy',allow_pickle=True)
	args_syst = data[0]
	J = args_syst['J']
	Nx = args_syst['Nx']
	V = args_syst['V']
	U = args_syst['U']
	N = args_syst['N']

	args_init = data[1]
	#mu_all = data['mu_all']	
	mu = data[2]
	psi0 = data[3]
	E_funct_IT = data[4]

	n0 = np.abs(psi0)**2

	## Analytical gaussian 

	Trap = GP.trap_1D(args_syst)
	m = 1/(2*J)
	w0 = np.sqrt(V/m)
	x0 = (Nx-1)/2
	positions = np.arange(Nx)

	## Thomas-Fermi

	#mu = mu_all[-1,int(len(mu_all[0])/2)]
	n_TF = (mu-Trap)/U/N# divide by because here, normalised to 1
	n_TF = n_TF-np.max(n_TF)+np.max(n0)

	## Plot

	#Hermite_0 = (m*w0/np.pi)**0.25*np.exp(-m*w0*(positions-x0)**2/2)
	#ax2.plot(positions,Hermite_0**2,'*g')
	ax2.plot(positions,n0,'^',label="$n_0, U = {}$".format(U),Fillstyle='none')
	ax2.plot(positions,n_TF,'--',label="$n_{TF}$")

## Save the figure
pyplot.suptitle('T-F approx. %s, %s,\
	 Nx = %.1i, N = %.3e, J = %.2f, V = %.3e, U = %.3e' % \
	 (args_syst['Trap'],args_syst['Symm'],\
	args_syst['Nx'],args_syst['N'],args_syst['J'],args_syst['V']\
	,args_syst['U']))

#ax2.set_ylim(0,0.015)
ax2.grid(axis='both')
ax2.legend()
pyplot.show()

temp = '1D_density_Ns_%s_%s_%s_Nx_%.1i_J_%.3f_V_%.3e_U_%.3e' %\
		(args_syst['Method'],args_syst['Trap'],\
		args_syst['Symm'],args_syst['Nx'],args_syst['J'],\
		args_syst['V'],args_syst['U'])
fig_OL1D.savefig("Figures_OL_1D_BEC_IT/fig_"+temp+".pdf")