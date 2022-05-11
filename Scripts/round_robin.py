import numpy as np

def round_robin(RR_interval):

	wind_size = 100
	data_vect = []
	modified_data_vector = []
	ex_index = []
	
	#CREATE DATA VECTOR
	for i in range(len(RR_interval)):

		if i < len(RR_interval):

			if len(RR_interval) >= i+wind_size:
				rr_interval = RR_interval[i:i+wind_size]
				data_vect.append(rr_interval)
			else:
				RR_interval.extend(RR_interval[0:100])
				rr_interval = RR_interval[i:i+wind_size]
				data_vect.append(rr_interval)
	
	#REMOVE EQUIDISTANT VECTORS
	for v in range(len(data_vect)):
		
		if v in ex_index:
			continue
		else:
			next_data = data_vect[v+1: len(data_vect)-1]
			c = 0
			for vi in range(len(next_data)):
				
				if len(data_vect[v])==len(next_data[vi]):

					np_sub_array = np.absolute(np.array(data_vect[v]) - np.array(next_data[vi]))
					result = np.all(np_sub_array == np_sub_array[0])
					if result == True:
						ex_index.append(v+1+vi)
						c+=1
			if c == 0:
				modified_data_vector.append(data_vect[v])


	return
