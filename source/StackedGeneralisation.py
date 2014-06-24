# method described in http://arxiv.org/abs/1404.4175
import numpy as np
import Classify
import sys
from scipy.io import loadmat

class StackedGeneralisation:
	def __init__(self,path_to_data):
		self._Path2Data=path_to_data
		self._subjects_train=range(1,5)
		self._subjects_train_testing=range(5,7)
		self._subjects_test=range(17,24)
		self._tmin = 0.0
		self._tmax = 0.5
		self._first_layer_classifiers=[]
		self._second_layer_classifiers=[]
		self._data_X=[]
		self._data_y=[]
		self._data_layer_1_X=[]
		self._data_layer_1_y=[]
		self._clfr2=[]
		
		self._data_X_testing=[]
		self._data_y_testing=[]
		self._data_layer_1_X_testing=[]
		self._data_layer_1_y_testing=[]

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
		
		return self.ReshapeToFeaturesVector(XX)
		
	def GetTrainData(self,filename):
		print "Loading", filename
		data = loadmat(filename, squeeze_me=True)
		XX = data['X']
		yy = data['y']
		sfreq = data['sfreq']
		tmin_original = data['tmin']
		XX=self.ProcessData(XX,sfreq,tmin_original,0)		
		
		return (XX,yy)

	def CreateFistLayer(self):
		for subject in self._subjects_train:
			filename = self._Path2Data+'/train_subject%02d.mat' % subject
			X,y=self.GetTrainData(filename)
			self._data_layer_1_y.append(y)
			self._data_X.append(X)
			self._data_y.append(y)
			print "shape of X and y are ",np.shape(X),np.shape(y)
			clfr = Classify.LogisticRegression(X, y,None, None)
			self._first_layer_classifiers.append(clfr)
		
		self._data_layer_1_y=np.concatenate(self._data_layer_1_y)
		print "shape of data class labels= ",np.shape(self._data_layer_1_y)
			
		print "toal classifiers =",len(self._first_layer_classifiers),len(self._data_X),len(self._data_y)
		
		#now create first layer of predictions
		for i in range(len(self._subjects_train)):
			print "i= ",i
			ypred_1=[]
			for j in range(len(self._subjects_train)):
				print "j= ",j
				ypred=self._first_layer_classifiers[i].predict(self._data_X[j])
				ypred_1.append(ypred)
			
			ypred_1 = np.concatenate(ypred_1)
			print "length of ypred_1= ",len(ypred_1),np.shape(ypred_1)
			self._data_layer_1_X.append(ypred_1)
			
		self._data_layer_1_X=np.vstack(self._data_layer_1_X).T
		print "shape of layer 1 X=",np.shape(self._data_layer_1_X)
		
		#now the second layer
		print "creating the second layer"
		self._clfr2=Classify.LogisticRegression(self._data_layer_1_X, self._data_layer_1_y,None, None)
		
	def CreateSecondLayer(self):
		for subject in self._subjects_train_testing:
			filename = self._Path2Data+'/train_subject%02d.mat' % subject
			X,y=self.GetTrainData(filename)
			self._data_layer_1_y_testing.append(y)
			self._data_X_testing.append(X)
			self._data_y_testing.append(y)
			print "shape of X and y are ",np.shape(X),np.shape(y)
			#clfr = Classify.LogisticRegression(X, y,None, None)
			#self._first_layer_classifiers.append(clfr)
		
		self._data_layer_1_y_testing=np.concatenate(self._data_layer_1_y_testing)
		print "shape of data class labels= ",np.shape(self._data_layer_1_y_testing)
			
		print "shape of data_X_testing and data_y_testing=",len(self._data_X_testing),len(self._data_y_testing)
		
		#now create first layer of predictions
		for i in range(len(self._subjects_train)):#since we have subjects_train number of features here
			print "i= ",i
			ypred_1=[]
			for j in range(len(self._subjects_train_testing)):
				print "j= ",j
				ypred=self._first_layer_classifiers[i].predict(self._data_X_testing[j])
				ypred_1.append(ypred)
			
			ypred_1 = np.concatenate(ypred_1)
			print "length of ypred_1 testing= ",len(ypred_1),np.shape(ypred_1)
			self._data_layer_1_X_testing.append(ypred_1)
			
		self._data_layer_1_X_testing=np.vstack(self._data_layer_1_X_testing).T
		print "shape of layer 1 X testing=",np.shape(self._data_layer_1_X_testing)
		
		ypred_test=self._clfr2.predict(self._data_layer_1_X_testing)
		
		print "id        prediction      truth"
		for i in range(len(ypred_test)):
			print i, ypred_test[i],self._data_layer_1_y_testing[i]
			
		print "accuracy= " , float(sum(abs(ypred_test-self._data_layer_1_y_testing)))/float(len(self._data_layer_1_y_testing))
		
		
		
		
					
			
