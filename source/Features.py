import numpy as np

class CreateFeatures:
	"""
	A class for creating the feature vector.
	"""
	
	def ApplyTimeWindow(self,XX, tmin, tmax, sfreq, tmin_original=-0.5):
		"""
		A function to apply the desired time window.
		
		@param XX a matrix of the shape [trial x channel x time]
		@param tmin start point of the time window
		@param tmax end point of the time window
		@param tmin_original original start point of the time window
		"""
		print "Applying the desired time window."
		print
		beginning = np.round((tmin - tmin_original) * sfreq).astype(np.int)
		end = np.round((tmax - tmin_original) * sfreq).astype(np.int)
		XX = XX[:, :, beginning:end].copy()
		
		return XX
	
	def ApplySVD(self,XX,num_components):
		"""
		Apply SVD to each trial and take the most important componetns
		
		@param XX a matrix of the shape [trial x channel x time]
		@param num_components number of componetns to consider in reduction
		"""
		print "appling svd with componetns",num_components
		print
		
		for i in range(np.shape(XX)[0]):
			mat=XX[i,:,:]
			u,s,v=np.linalg.svd(mat,full_matrices=False)
			snew=np.zeros(np.shape(s))
			if num_components > snew.size-1 :
				print "input num_components ",num_components,"is > snew.size-1",snew.size-1
				print "changin to ",snew.size-1
				num_components=snew.size-1				
			snew[0:num_components]=s[0:num_components]
			S=np.diag(snew)
			XX[i,:,:]=np.dot(u,np.dot(S,v))
		
	def ApplyFilter(self,XX,cutoff_min,cutoff_max):
		"""
		Apply a frequency-filter to the time series.
		
		@param XX a matrix of the shape [trial x channel x time]
		@param cutoff_min Minimum value of the cut-off window
		@param cutoff_max Maximum value of the cut-off window
		"""
		print "Applying a filter of ",cutoff_min,cutoff_max
		print 
		
		for i in range(np.shape(XX)[0]):
			for j in range(np.shape(XX)[1]):
				ts=XX[i,j,:]
				fs=scipy.fftpack.fft(ts)
				for k in range(fs.size):
					if abs(fs[k])< cutoff_min or abs(fs[k]) > cutoff_max :
						fs[k] = complex(0,0)
				XX[i,j,:]=np.real(scipy.fftpack.ifft(fs))
	
	
	def ApplyZnorm(self,XX):
		"""
		Apply normalisation.
		@param XX a matrix of the shape [trial x channel x time]
		"""
		
		print "Features Normalization."
		print
		XX -= XX.mean(0)
		XX = np.nan_to_num(XX / XX.std(0))
		return XX
	
	def ReshapeToFeaturesVector(self,XX):
		"""
		Reshape the matrix to a set of features by concatenating 
		the 306 timeseries of each trial in one long vector.
		
		@param XX a matrix of the shape [trial x channel x time]
		"""
		
		print "2D Reshaping: concatenating all 306 timeseries."
		print
		XX = XX.reshape(XX.shape[0], XX.shape[1] * XX.shape[2])
		return XX
