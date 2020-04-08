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
'''
This code is used to compare to analytical relations between the energies, 
the number of particles and the convergence of the imaginary-time method 
'''

plot_Vs = False
plot_Us = True

######################## PLOT Us ###############################

if plot_Us==True:

	fig_OL1D_Us = pyplot.figure(figsize=(8,8))
	ax2 = pyplot.subplot(111)

	E_funct_SC_s = np.array([])
	E_k_s = []
	E_U_s = []
	E_tr_s = []
	U_s = []
	mu_s = []

	## Select the files you want to plot

	path = '/home/maxime/Desktop/Codes/Optical_1D_Lattice_BEC/Imaginary_time/Datas_IT/'
	list_args = ['N','V','U']
	split = ['N_','V_','U_']
	params_interv = {'U': [1e-6,1e-3],'V': [0.5e-4],'N': [1e4]}
	extension = '.npy'
	files_names = Mf.select_files(path,extension,list_args,params_interv,split)

	split = ['U_']
	list_args = ['U']
	files_names = Mf.reorder_filesnames(files_names,list_args,split)
	
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
		#mu = data[2]
		psi0 = data[3]
		dt = args_init['dt']
		mu = GP.compute_mu(psi0,dt,args_syst,args_init)[0]
		E_funct_IT = data[4]

		## Compute contributions to energy

		E_k = GP.E_kin(psi0,args_syst)
		E_U = GP.E_int(psi0,args_syst)
		E_tr = GP.E_trap(psi0,args_syst)
		E_k_s.append(E_k)
		E_U_s.append(E_U)
		E_tr_s.append(E_tr)
		U_s.append(U)
		mu_s.append(mu)

	E_k_s = np.array(E_k_s)
	E_U_s = np.array(E_U_s)
	E_tr_s = np.array(E_tr_s)
	mu_s = np.array(mu_s)
	U_s = np.array(U_s)
	print(mu_s)

	## To see how E_kin and E_trap vary in fct of U
	
	ax2.semilogx(U_s,E_k_s/N+2*J,'-b',fillstyle='none',label="$E_k/N + 2J$")
	ax2.semilogx(U_s,E_tr_s/N,'-g',fillstyle='none',label="$E_{trap}/N$")
	ax2.semilogx(U_s,E_U_s/N,'-r',fillstyle='none',label="$E_U/N$")
	ax2.semilogx(U_s,E_k_s/N+2*J+E_tr_s/N,'-k',label="$E_{trap}/N+E_k/N$")
	ax2.semilogx(U_s,2/5*(mu_s+2*J),'--r',label="$2\mu/5$")
	ax2.semilogx(U_s,1/5*(mu_s+2*J),'--g',label="$1\mu/5$")
	print("N = ", N, "U = ", U, "V0 = ", V,\
		": U*N*|phi_max|^2 = ",U*N*np.max(np.abs(psi0)**2))

	ax2.set_xlabel('$U$')

	# elif what2plot=='mu_E':			
	# 	ax1.plot(np.log10(N),E_k/N+2*J,'*b',label="$E_k$")
	# 	ax1.plot(np.log10(N),E_tr/N,'*g',label="$E_t$")
	# 	ax1.plot(np.log10(N),E_U/N,'*r',label="$E_U$")
	# 	ax1.plot(np.log10(N),E_funct_IT.real/N,'o',label="$N = {}$".format(N))
	# 	for x in range(len(mu_all[0])):
	# 		ax2.plot(np.arange(len(mu_all)),mu_all[:,x].real) # mu.imag is zero

	## Save the figure

	pyplot.suptitle('%s, approx. %s, %s,\
		 Nx = %.1i, N = %.3e, J = %.2f, V = %.3e' % \
		 (args_syst['Method'],args_syst['Trap'],args_syst['Symm'],\
		args_syst['Nx'],args_syst['N'],args_syst['J'],args_syst['V']))

	ax2.grid(axis='both')
	ax2.legend(loc=2);
	pyplot.show()

	temp = '1D_Energies_Ns_%s_%s_%s_Nx_%.1f_J_%.2f_V_%.1e' %\
			(args_syst['Method'],args_syst['Trap'],args_syst['Symm'],\
			args_syst['Nx'],args_syst['J'],args_syst['V'])
	fig_OL1D_Us.savefig("Figures_OL_1D_BEC_IT/fig_"+temp+".pdf")


################### PLOT Vs #####################

if plot_Vs==True:
	## Initialize the figure

	fig_OL1D_Vs = pyplot.figure(figsize=(8,8))
	ax1 = pyplot.subplot(111)

	E_funct_SC_s = np.array([])
	E_k_s = np.array([])
	E_U_s = np.array([])
	E_tr_s = np.array([])
	w0_s = np.array([])

	## Select the files you want to plot

	path = '/home/maxime/Desktop/Codes/Optical_1D_Lattice_BEC/Imaginary_time/Datas_IT/'
	list_args = ['N','V','U']
	split = ['N_','V_','U_']
	params_interv = {'U': [1e-6,1e-3],'V': [1e-4],\
				'N': [1e4]}
	extension = '.npy'
	files_names = Mf.select_files(path,extension,list_args,params_interv,split)

	split = ['V_']
	list_args = ['V']
	files_names = Mf.reorder_filesnames(files_names,list_args,split)

	for filename in files_names:

		data = np.load(filename+'.npy',allow_pickle=True)
		args_syst = data[0]
		J = args_syst['J']
		Nx = args_syst['Nx']
		V = args_syst['V']
		U = args_syst['U']
		N = args_syst['N']
		m = 1/(2*J)

		args_init = data[1]
		#mu_all = data['mu_all']	
		mu = data[2]
		psi0 = data[3]
		E_funct_SC = data[4]

		## Compute contributions to energy

		E_k = GP.E_kin(psi0,args_syst)
		E_U = GP.E_int(psi0,args_syst)
		E_tr = GP.E_trap(psi0,args_syst)
		E_k_s = np.append(E_k_s,E_k)
		E_U_s = np.append(E_U_s,E_U)
		E_tr_s = np.append(E_tr_s,E_tr)
		w0_s = np.append(w0_s,np.sqrt(V/m))

	## To see how E_kin and E_trap vary in fct of V
	
	#print("w0/2 = ", w0/2)
	ax1.plot(w0_s,E_k_s/N+2*J,'sb',fillstyle='none',label="$E_k$")
	ax1.plot(w0_s,E_tr_s/N,'+g',fillstyle='none',label="$E_t$")
	ax1.plot(w0_s,E_U_s/N,'^r',fillstyle='none',label="$E_U$")
	#ax1.plot(w0,w0/4,'sk',fillstyle='none',label="$w_0/4$")

	ax1.set_xlabel('$\omega$')

	# Save the figure

	pyplot.suptitle('%s, %s, %s,\
		 Nx = %.1i, N = %.3e, J = %.2f, U = %.3e' % \
		 (args_syst['Method'],args_syst['Trap'],args_syst['Symm'],\
		args_syst['Nx'],args_syst['N'],args_syst['J'],\
		args_syst['U']))

	ax1.grid(axis='both')
	#ax1.legend(loc=5);
	pyplot.show()

	temp = '1D_Energies_Ns_%s_%s_%s_Nx_%.1f_J_%.2f_U_%.3e' %\
			(args_syst['Method'],args_syst['Trap'],args_syst['Symm'],\
			args_syst['Nx'],args_syst['J'],args_syst['U'])
	fig_OL1D_Vs.savefig("Figures_OL_1D_BEC_IT/fig_"+temp+".pdf")


