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

##################Script  OL 1D IT ##########################

path = '/home/maxime/Desktop/Codes/Optical_1D_Lattice_BEC/Self_consistent/Datas_SC'
list_args = ['N','V','U']
split = ['N_','V_','U_']
params_interv = {'U': np.logspace(-6,-3,30),'V': [0.5e-4],\
				'N': [1e4,1e4+1]}
extension = '.npy'
#files_names = Mf.select_files(path,extension,list_args,params_interv,split)
k = 0
for N in params_interv['N']:
	for V in params_interv['V']:
		for U in params_interv['U']:
			start = time.time()

			args_syst = {
			'J' : 1,
			'N' : N,
			'V' : V,
			'Nx' : 201,
			'U' : U,
			'Method' : 'SC',
			'Trap' : 'Harmonic',
			'Symm' : 'Isotropic',
			}

			args_syst.update({'sites_dic' : GP.lattice_1D(args_syst)})

			## Kinetic + Trap part of Hamiltonian

			H_KV = GP.H_1D(args_syst)

			if k==0:
			 	args_init = {
			 	'H_KV' : H_KV,
			 	'err_SC' : 1e-10
			 	}

			else:
				args_init.update({'psi_init' : psi0})

			psi0 = GP.calc_psi0(args_syst,args_init)

			mu = linalg.eigh(H_KV+GP.H_int(psi0,args_syst))[0][0]

			E_funct_SC = GP.energy_functional(psi0,args_syst)

			Trap = GP.trap_1D(args_syst)
			U = args_syst['U']
			N = args_syst['N']
			Nx = args_syst['Nx']
			n_TF = (mu-Trap)/U/N
			diff_TF_n0 = np.sum(np.abs(n_TF-abs(psi0)**2))

			data = [args_syst,args_init,mu,psi0,E_funct_SC,H_KV,diff_TF_n0]
			dataID = '1D_%s_%s_%s_Nx_%.1i_J_%.2f_N_%.4e_V_%.3e_U_%.3e' %\
					(args_syst['Method'],args_syst['Trap'],args_syst['Symm'],\
					args_syst['Nx'],args_syst['J'],N,args_syst['V'],args_syst['U'])
			np.save('Datas_SC/'+dataID,data)

			print("For Nx = ", Nx, ", N = ", N, ", U = ", U, ", V = ", V,\
				"it took",time.time()-start,"secondes")

			k += 1