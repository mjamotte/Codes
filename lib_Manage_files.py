import os
import numpy as np


def get_val_from_filename(filename,list_args,split):
	'''
	Returns the values inside a filename that contains some. 
	This function specificaly works for the datafiles in the
	project Optical Lattices. It is easy to adapt the code. 

	Inputs: filename = string.
			list_args = dictionnary containing the name of 
						the parameters.
	'''

	diction = {}
	i = 0
	for k in split:
		diction[list_args[i]] = float((filename.\
			split(k)[1]).split('_')[0])
		i += 1
	return diction


def select_files(path,extension,list_args,params_interv,split):
	'''
	Returns the names of the files contains values belonging
	to the intervals specified in "params_interv". Only the
	parameters you want to select may be specified.

	Inputs: path = string indicating where is the file on the computer.
			list_args = dictionnary containing the name of 
						the parameters.
			params_interv = dictionnary containing the intervals
						which the values of the parameters
						in "list_args" must belong to. 
	'''

	files = []
	f = []
	containing_folder = path.split('/')[-2]

	## r=root, d=directories, f = files
	for r, d, f in os.walk(path):

		for param in params_interv:

			# allows to get only the intersections of files whose the 
			# values in the names belongs to the intersection of the 
			# conditions on the parameters.  

			for file in f:

				filename = (os.path.join(r,file)).\
				split(containing_folder+'/')[1].split(extension)[0]
				value = get_val_from_filename(filename,list_args,split)[param]

				if Is_in_interval(value,params_interv[param])==True:
					## removes the extension of the file (exple: '.npy')
					files.append((os.path.join(r,file)).split(extension)[0])

			f = files.copy()
			files = []

	if len(f)==0:	
		print("No file was in the ranges")
	
	return f # returns the files' names respecting the intervals
	

def Is_in_interval(x,interval):

	if x<=interval[-1] and x>=interval[0]:
		return True
	else: 
		return False


def reorder_filesnames(filesnames,list_args,split):

	'''
	Sorts and returns a list of (strings) names in alphabetic/increasing 
	numbers order for a specific parameter indicated in the input 
	string "list_args"

	Inputs: filesnames = list of strings
			list_args = string that is the name of the parameter
						we want to extract the value of. Must contain
						only one element.
			split = string that is the keyword where to split the name
					(depends on the way of naming the files; see the function
					get_val_from_filename)
	'''
	Len = len(filesnames)
	indices_values = []
	param = list_args[0]
	out = filesnames.copy()
	for i,filename in enumerate(filesnames):

		index_value = [i,get_val_from_filename(filename,list_args,split)[param]]
		indices_values.append(index_value)

	copy = np.array(indices_values.copy())
	temp = np.zeros((Len,2))

	## sort the second column of indices_values from smallest to highest
	for i in range(Len):
		index_min = int(np.argmin(copy[:,1]))
		temp[i] = indices_values[index_min]
		index_max = int(np.argmax(copy[:,1]))
		copy[index_min,1] = copy[index_max][1]+1

	temp = np.array(temp) ## to be able to use int()
	for i in range(Len):
		## out = reordered filesnames
		out[i] = filesnames[int(temp[i,0])]
	return out


## Tests for get_val_from_filename, select_files and Is_in_interval: 

# path = '/home/maxime/Desktop/Codes/Optical_1D_Lattice_BEC/Imaginary_time/Datas_IT/'
# filename = "1D_SC_Harmonic_Isotropic_Nx_201_J_1.00_N_1.000e+05_V_6.928e-05_U_6.842e-04"
# split = ['N_','V_','U_']
# list_args = ['N','V','U']
# params_interv = {'U': [1e-2],'V': [0.5e-3],'N': [9,51]}
# extension = '.npy'
# print(select_files(path,extension,list_args,params_interv,split))

## Tests for reorder_filesnames
# list_args = ['U']
# split = ['U_']
# filesnames = [\
# '1D_SC_Harmonic_Isotropic_Nx_201_J_1.00_N_1.000e+04_V_1.000e-05_U_1.193e-04',\
# '1D_SC_Harmonic_Isotropic_Nx_201_J_1.00_N_1.000e+04_V_1.000e-05_U_3.862e-05',\
# '1D_SC_Harmonic_Isotropic_Nx_201_J_1.00_N_1.000e+04_V_1.000e-04_U_1.933e-04']

#print(reorder_filesnames(filesnames,list_args,split))

##########################################################################
##########################################################################
# def get_close_val_from_filename_list(values,files_names,params_interv):
# 	diff_all = np.array([])
# 	for filename in files_names:

# 		diff = np.array([]) 
# 		for param in params_interv:
# 			diction_val = get_val_from_filename(filename,list_args)

# 			diff_all.append(diff)

# 	for i in range(len(diff_all[0]))
