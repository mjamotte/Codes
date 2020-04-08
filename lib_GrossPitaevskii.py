import numpy as np
import scipy as sc
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as pyplot
import matplotlib.axes as axes
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from numpy import linalg
import cmath
import time
from mpl_toolkits import mplot3d
from scipy.integrate import ode
#from numba import jit
#import debugger as bug
############# FUNCTIONS OPTICAL LATTICES #############

'''

^ y	  |	 |	|  |  |
|	  A	 B--A  B--A
|	  |  |  |  |  |
|	  B--A  B--A  B
|	  |  |  |  |  |
|	  A  B--A  B--A
|	(0,0)
|-------------------> x

'''
	

def trap(system,case):

	####################################################################################
	#																				   #	
	#	Trap-potential of amplitude V0 at position (i,j) that is a lattice site.	   #
	#	One needs to calculate x and y from (i,j) depending on the lattice.			   #
	#																				   #
	#	Inputs:	- case: indicates the the approximation used: the kind of 		       #
	#					Trap-potential												   #
	#					used, the boundary	   										   #
	#					conditions, the method to solve GP; 						   #
	#					case = [resolution method, potential, boundary conditions]	   #
	#			- i : integer indicating the lattice site which we calculate the       #
	#			  	  potential at.											   	   	   #
	#			- system: list of parameters = [sites_dic,U,V0,Nx,Ny,				   #
	#											mu,J,kxa_list,kya_list]				   #
	#			   																	   #	
	####################################################################################
		
	sites_dic = system[0]
	V0 = system[2]
	Nx = system[3]
	Ny = system[4]
	L = len(sites_dic)

	Lx = (Nx-1)*3/2
	Ly = Ny*np.sqrt(3)/2

	out = np.zeros(L)

	if case[1]=='Linear': #linear potential

		for i in range(L):

			x = sites_dic[i][0]*3/2
			y = sites_dic[i][1]*np.sqrt(3)/2

	if case[1]=='Harmonic':

		for i in range(L):

			x = sites_dic[i][0]*3/2 # a=1
			#y = sites_dic[i][1]*np.sqrt(3)/2
			out[i] = 1/2*V0*x**2

	return out





############# 1D OPTICAL LATTICE #############

''' 
	The goal of coding this is the understanding of the physics 
	of the Hamiltonian presented in the Lühmann's paper to 
	aquire feelings with correlated hoppings. 
'''

def lattice_1D(args_syst):
	Nx = args_syst['Nx']

	sites_dic = {}
	n = 0
	for i in range(Nx): # fill line by line, from left to right
		sites_dic[n] = np.array([n])
		n += 1
			
	return sites_dic

	return 	


def trap_1D(args_syst):

	V0 = args_syst['V']
	Nx = args_syst['Nx']

	V = np.zeros(Nx)

	for x in range(Nx):
		V[x] = 1/2*V0*(x-(Nx-1)/2)**2

	return V


def H_1D(args_syst):

	J = args_syst['J']
	Nx = args_syst['Nx']

	H = 1j*np.zeros((Nx,Nx))

	if args_syst['Symm']=='Isotropic':

		if args_syst['Trap']=='Harmonic':
			H = np.diag(trap_1D(args_syst))

		for i in range(Nx-1):
			H[i,i+1] = -J
			H[i+1,i] = np.conj(H[i,i+1])

	return H


def calc_psi0(args_syst,args_init):

	if args_syst['Method']=='SC': #SC = self-consistent method
		psi0 = solveGP_SC(args_syst,args_init)
		return psi0

	elif args_syst['Method']=='IT': #Imaginary time
		psi0 = solve_GP_IT(args_syst,args_init)
		#mu_all,psi0,E_funct_IT = solve_GP_IT_RK4(args_syst,args_init)
		return psi0


def solveGP_SC(args_syst,args_init):

	N = args_syst['N']
	U = args_syst['U']
	H_KV = args_init['H_KV']
	err = args_init['err_SC'] # condition to stop convergence

	
	psi0_all = [np.array([])]
	#H_KV = sc.sparse.coo_matrix(H_KV) # KV = Kinetic + Trap	

	#E0_old,eigVecs = sc.sparse.linalg.eigsh(H_KV,which='SA',k=1)
	E0_old,eigVecs = linalg.eigh(H_KV)
	psi_old = np.matrix.transpose(eigVecs[:,0])

	counterSC = 0
	lam = 1/(N*U*100+1) #1/(10*N+1) # to avoid oscillations in energy that slow down the code
	#flag = 0

	while True:

		#H_U = sc.sparse.coo_matrix(H_int(psi_old,args_syst))
		#E0_new,eigVecs = sc.sparse.linalg.eigsh(H_KV+H_U,which='SA',k=1)
		H_U = H_int(psi_old,args_syst)
		eigVecs = linalg.eigh(H_KV+H_U)[1]
		psi_new = np.matrix.transpose(eigVecs[:,0])

		psi_lam = np.sqrt((1-lam)*psi_old**2 + lam*psi_new**2)

		err_psi = np.abs(np.max(psi_lam)\
					-np.max(psi_old))\
					/np.abs(np.max(psi_lam))

		if err_psi<err:
			break

		psi_old = psi_lam
		counterSC += 1			

	print('Number of iterations of self-consistent method =',counterSC)

	return psi_lam


def GP(t,psi_old,args_syst,args_init):

	psi_old = psi_old/np.sqrt(np.sum(np.abs(psi_old)**2))
	H_KV = args_init['H_KV']
	#H_KV = sc.sparse.coo_matrix(H_KV) # KV = Kinetic + Trap
	N = args_syst['N']
	dim = len(psi_old)
	psi_old_co = psi_old[:int(dim/2)] + 1j*psi_old[int(dim/2):]

	# Hopping part + trap part of the GP
	y1 = H_KV.dot(psi_old_co)
	# Interacting part of the GP
	y2 = H_int(psi_old_co,args_syst).dot(psi_old_co)

	y = y1 + y2
	# -d_tau psi(tau) = H psi(tau)
	y = -y
	return np.concatenate((np.real(y),np.imag(y)))



def compute_mu(psi_old,dt,args_syst,args_init):

	H_KV = args_init['H_KV']
	psi_new = sc.linalg.expm(-(H_KV+H_int(psi_old,args_syst))*dt).dot(psi_old)
	## Here, psi_new isn't renormalized

	mu = -np.log(psi_new/psi_old)/dt

	return mu#,psi_new


def RK4_GP(psi_old,args_syst,args_init):

	t = 0 # delete the t dependence once RK4 works
	dt = args_init['dt']

	k1 = GP(t,psi_old,args_syst,args_init)
	psi_1 = psi_old+dt*k1/2
	psi_1 = psi_1/np.sqrt(np.sum(np.abs(psi_1)**2))

	k2 = GP(t+dt/2,psi_1,args_syst,args_init)
	psi_2 = psi_old+dt*k2/2
	psi_2 = psi_2/np.sqrt(np.sum(np.abs(psi_2)**2))

	k3 = GP(t+dt/2,psi_2,args_syst,args_init)
	psi_3 = psi_old+dt*k3
	psi_3 = psi_3/np.sqrt(np.sum(np.abs(psi_3)**2))

	k4 = GP(t+dt,psi_3,args_syst,args_init)
	out = psi_old + 1/6*dt*(k1+2*k2+2*k3+k4)
	out = out/np.sqrt(np.sum(np.abs(out)**2))

	return out
def solve_GP_IT_RK4(args_syst,args_init):

	## Initialisation 
	H_KV = args_init['H_KV']
	#H_KV = sc.sparse.coo_matrix(H_KV) # KV = Kinetic + Trap

	if 'psi_init' in args_init:
		psi_old = args_init['psi_init'] 

	else:
		#gauss = sc.sparse.linalg.eigsh(H_KV)[1][:,0]
		psi_old = linalg.eigh(H_KV)[1][:,0]

	dt = args_init['dt']
	err = args_init['err_IT']
	counterIT = 0
	dim = len(psi_old)
	mu_old = np.zeros(dim)
	flag = 0
	while True:

		## time evolution

		psi_new = RK4_GP(psi_old,args_syst,args_init) # renormalized in RK4_GP

		## Compute mu
	
		mu_new = compute_mu(psi_old,dt,args_syst,args_init)

		if flag==0:
			mu_all = np.array([mu_new])
			flag = 1
		else:
			mu_all = np.append(mu_all,np.array([mu_new]),axis=0)

		half_len = int(len(mu_new)/2) # will give the mu at maximum
		err_mu = abs(mu_old[half_len]-mu_new[half_len])/abs(mu_new[half_len])

		if err_mu<err:
			print('The mu criteria went first')
			break

		psi_old = psi_new
		mu_old = mu_new
		counterIT += 1

	psi0 = psi_new
	E_funct_IT = energy_functional(psi0,args_syst)

	print('The number of iterations for IT =', counterIT)

	for i in range(len(mu_all[0])):
		pyplot.plot(mu_all[:,i])

	pyplot.show()

	return mu_all,psi0,E_funct_IT


def solve_GP_IT(args_syst,args_init):

	## Initialisation 
	H_KV = args_init['H_KV']
	#H_KV = sc.sparse.coo_matrix(H_KV) # KV = Kinetic + Trap

	if 'psi_init' in args_init:
		psi_old = args_init['psi_init'] 
		psi_old = np.concatenate((np.real(psi_old), np.imag(psi_old)))

	else:
		#gauss = sc.sparse.linalg.eigsh(H_KV)[1][:,0]
		gauss = linalg.eigh(H_KV)[1][:,0]
		psi_old = np.concatenate((np.real(gauss), np.imag(gauss)))

	## parameters for set_integrator and GP
	tol = 1e-9 # tolerance
	nsteps = np.iinfo(np.int32).max
	solver = ode(GP)
	solver.set_f_params(args_syst,args_init) # parameters needed in GP_t_real
	solver.set_integrator('dop853', atol=tol, rtol=tol, nsteps=nsteps)

	## Evolution
	t = 0
	dt = args_init['dt']
	err = args_init['err_IT']
	counterIT = 0
	dim = len(psi_old)

	while True:
		## time evolution
		solver.set_initial_value(psi_old, t)
		solver.integrate(t+dt)
		t = t+dt
		psi_new = solver.y

		## Compute mu
		psi_old_co = psi_old[:int(dim/2)] + 1j*psi_old[int(dim/2):]
		psi_new_co = psi_new[:int(dim/2)] + 1j*psi_new[int(dim/2):] # not renorm. yet

		## renormalize
		psi_new = psi_new/np.sqrt(np.sum(abs(psi_new)**2))
		psi_new_co = psi_new[:int(dim/2)] + 1j*psi_new[int(dim/2):]

		err_psi = np.sqrt(np.abs(np.max(np.abs(psi_new_co)**2\
					-np.max(np.abs(psi_old_co)**2))))\
					/np.sqrt(np.max(np.abs(psi_new_co)**2))

		if err_psi<err:
			break

		psi_old = psi_new
		counterIT += 1

	if solver.successful():
		sol = solver.y
		sol = sol/np.sqrt(sum(abs(sol)**2))
		sol_re = sol[:int(dim/2)]
		sol_im = sol[int(dim/2):]
		psi0 = sol_re + 1j*sol_im

	print('The number of iterations for IT =', counterIT)

	return psi0


def H_int(psi,args_syst):

	U = args_syst['U']
	N = args_syst['N']
	return U*(N-1)*np.diag(np.abs(psi)**2)


def E_kin(psi,args_syst):

	J = args_syst['J']
	N = args_syst['N']
	Nx = args_syst['Nx']
	positions = np.arange(Nx-1) # -1 because of the k+1 below
	E_kin = 0
	for k in positions:
		E_kin += -J*N*np.conj(psi[k])*psi[k+1]-J*N*np.conj(psi[k+1])*psi[k]
	return E_kin


def E_int(psi,args_syst):

	N = args_syst['N']
	U = args_syst['U']
	Nx = args_syst['Nx']

	E_U = U*N*(N-1)/2*np.sum(np.abs(psi)**4)  
	return E_U


def E_trap(psi,args_syst):

	N = args_syst['N']
	E_tr = N*np.sum(trap_1D(args_syst)*abs(psi)**2)
	return E_tr


def energy_functional(psi,args_syst):

	E_U = E_int(psi,args_syst)
	E_tr = E_trap(psi,args_syst)
	E_k = E_kin(psi,args_syst)

	return E_U+E_tr+E_k


def impose_sym(vector): 
	## imposes symmetry from the middle of the vector
	Len = len(vector)

	for i in range(int(Len/2)):
		vector[i] = (vector[i]+vector[Len-i-1])*0.5
		vector[Len-i-1] = vector[i]

	norm = np.sqrt(np.sum(np.abs(vector)**2))
	vector = vector/norm
	return vector


def dEdN_O2(E,dN):#(path,N,dN,filesnames):

	# list_args = ['N','V','U']
	# split = ['N_','V_','U_']
	# extension = '.npy'
	# files_names = Mf.select_files(path,extension,list_args,params_interv,split)

	# ## select files with N belonging to [N-dN,N+dN]
	# for filename in filenames:
		
	# 	data = np.load(filename+'.npy',allow_pickle=True)
	# 	E = data[4] 

	dEdN = np.array([])
	for i in range(int(len(E)/3)): # 3 = "order of approx"+1
		dEdN = np.append(dEdN,(E[i*3+2]-E[i*3])/(2*dN))
	return dEdN


def gauss(xs,x0,sigma): # analytical

	Gauss = np.zeros(len(xs))
	i = 0
	for x in xs:
		Gauss[i] = 1/np.sqrt(sigma**2*2*np.pi)*np.exp(-(x-x0)**2/(2*sigma**2))
		i += 1

	return Gauss


def vector2matrix(vector,Nx,Ny):

	# Shape a vector into a Nx x Ny matrix
	L = len(vector)
	if L==Nx*Ny:
		mat = np.zeros((Ny,Nx))

		for i in range(Ny):
			for j in range(Nx):
				mat[i,j] = vector[i*Nx+j]

	else:
		print('Sizes do not match')
		mat = 0

	return mat

