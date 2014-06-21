import numpy as np
from scipy.io import loadmat

class ExtractFeatures:

	def __init__(self,path_to_data=None):
		self._Path2Data=path_to_data
		self._X_train = []
		self._y_train = []
		self._tmin = 0.0
		self._tmax = 0.5
		#print "Restricting MEG data to the interval [%s, %s]sec." % (self._tmin, self._tmax)

	def ApplyTimeWindow(self,XX, tmin, tmax, sfreq, tmin_original=-0.5):
		"""
		A function to apply the desired time window.
		
		@param XX a matrix of the shape [trial x channel x time]
		@param tmin start point of the time window
		@param tmax end point of the time window
		@param tmin_original original start point of the time window
		"""
		#print "Applying the desired time window."
		#print
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
		#print "appling svd with componetns",num_components
		#print
		
		for i in range(np.shape(XX)[0]):
			mat=XX[i,:,:]
			u,s,v=np.linalg.svd(mat,full_matrices=False)
			snew=np.zeros(np.shape(s))
			if int(num_components) > snew.size-1 or num_components < 0:
				print "input num_components ",num_components
				print "changin to ",snew.size-1
				num_components=snew.size-1				
			snew[0:int(num_components)]=s[0:int(num_components)]
			S=np.diag(snew)
			XX[i,:,:]=np.dot(u,np.dot(S,v))


	def ApplyZnorm(self,XX):
		"""
		Apply normalisation.
		@param XX a matrix of the shape [trial x channel x time]
		"""
		
		#print "Features Normalization."
		#print
		XX -= XX.mean(0)
		XX = np.nan_to_num(XX / XX.std(0))
		return XX
	
	def ReshapeToFeaturesVector(self,XX):
		"""
		Reshape the matrix to a set of features by concatenating 
		the 306 timeseries of each trial in one long vector.
		
		@param XX a matrix of the shape [trial x channel x time]
		"""
		
		#print "2D Reshaping: concatenating all 306 timeseries."
		#print
		XX = XX.reshape(XX.shape[0], XX.shape[1] * XX.shape[2])
		return XX

	def ProcessData(self,XX,sfreq,tmin_original,nSVD):
		"""
		Process data 
		@param XX a matrix of the shape [trial x channel x time]
		@param sfreq Frequency of the sampling
		@param tmin_original Original start point of the window.
		"""
		XX=self.ApplyTimeWindow(XX, self._tmin, self._tmax, sfreq,tmin_original)
		XX=self.ApplyZnorm(XX)
		
		self.ApplySVD(XX,nSVD)
		return self.ReshapeToFeaturesVector(XX)
		

	def MakeTrainData(self,subjects_train,nSVD):
		"""
		A function for creating the traing data
		@param subjects_train A vector containing the train data numbers e.g range(1,12)
		"""
	
		if self._Path2Data==None :
			raise RuntimeError("No path to data specified")
		
		if min(subjects_train) < 1 :
			raise RuntimeError("The train data starts from train_subject01.mat")
		
		if max(subjects_train) > 16 :
			raise RuntimeError("The train data ends at train_subject16.mat")
	
		#print "Creating the trainset."
		
		for subject in subjects_train:
			filename = self._Path2Data+'/train_subject%02d.mat' % subject
			#print "Loading", filename
			data = loadmat(filename, squeeze_me=True)
			XX = data['X']
			yy = data['y']
			sfreq = data['sfreq']
			tmin_original = data['tmin']
			#print "Dataset summary:"
			#print "XX:", XX.shape
			#print "yy:", yy.shape
			#print "sfreq:", sfreq
			
			XX=self.ProcessData(XX,sfreq,tmin_original,nSVD)		
		
			self._X_train.append(XX)
			self._y_train.append(yy)
			
		self._X_train = np.vstack(self._X_train)
		self._y_train = np.concatenate(self._y_train)
		#print "Trainset:", self._X_train.shape
		return (self._X_train,self._y_train)

