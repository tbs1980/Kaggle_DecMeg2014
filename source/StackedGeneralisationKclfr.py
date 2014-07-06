# method described in http://arxiv.org/abs/1404.4175
import numpy as np
import Classify
import Utils
import sys
from scipy.io import loadmat
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression

class StackedGeneralisationKclfr:
	def __init__(self,path_to_data):
		#self._K=8
		self._Path2Data=path_to_data
		#self._subjects_train=range(self._K,5)
		#self._subjects_train_class=range(1,self._K)
		
		self._subjects_train_class=range(1,8)
		self._subjects_train=range(13,15)
		
		self._subjects_train_testing=range(15,17)
		self._subjects_test=range(17,24)
		self._tmin = 0.0
		self._tmax = 0.5
		self._first_layer_classifiers=[]
		self._data_X=[]
		self._data_y=[]
		self._data_layer_1_X=[]
		self._data_layer_1_y=[]
		self._clfr2=[]
		
		self._data_X_testing=[]
		self._data_y_testing=[]
		self._data_layer_1_X_testing=[]
		self._data_layer_1_y_testing=[]
		
		self._sample_weights=[]
		
		self._ids_test=[]
		self._ypred_test=[]
		
		self._data_layer_1_y_class=[]
		
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

	def ProcessData(self,XX,sfreq,tmin_original):
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
		XX=self.ProcessData(XX,sfreq,tmin_original)		
		
		return (XX,yy)

	def GetTestData(self,filename):
		print "Loading", filename
		data = loadmat(filename, squeeze_me=True)
		XX = data['X']
		ids = data['Id']
		sfreq = data['sfreq']
		tmin_original = data['tmin']
		XX=self.ProcessData(XX,sfreq,tmin_original)		
		
		return (XX,ids)
		
	def CreateFistLayer(self):
		#train on the classifer data
		print "\nmaking the clssifiers"
		for subject in self._subjects_train_class:
			filename = self._Path2Data+'/train_subject%02d.mat' % subject
			X,y=self.GetTrainData(filename)
			clfr = Classify.LogisticRegression(X, y,None, None)
			self._first_layer_classifiers.append(clfr)

		
		#make the predictions for the train data
		print "\ngenerating the train data"
		for subject in self._subjects_train:
			filename = self._Path2Data+'/train_subject%02d.mat' % subject
			X,y=self.GetTrainData(filename)
			self._data_layer_1_y.append(y)
			self._data_X.append(X)
			self._data_y.append(y)
			
		#make all the predictions into one vector
		self._data_layer_1_y=np.concatenate(self._data_layer_1_y)
		
		#return
		
		print "\nmaking the predictions for the train data"
		#now create first layer of predictions
		for i in range(len(self._subjects_train_class)):
			ypred_1=[]
			print ""
			for j in range(len(self._subjects_train)):
				#subject=self._subjects_train[j]
				#filename = self._Path2Data+'/train_subject%02d.mat' % subject
				#X,y=self.GetTrainData(filename)
				#ypred=y
				ypred=self._first_layer_classifiers[i].predict(self._data_X[j])
				print "error of classifer " ,self._subjects_train_class[i],"for data ",self._subjects_train[j],"=", float(sum(abs(ypred-self._data_y[j])))/float(len(self._data_y[j]))*100,"%"
				ypred_1.append(ypred)
			
			#concatenate all the predictions into a feature vector
			ypred_1 = np.concatenate(ypred_1)
			self._data_layer_1_X.append(ypred_1)
		
		self._data_layer_1_X=np.vstack(self._data_layer_1_X).T

		#now the second layer
		print
		print "creating the second layer"
		print 
		
		#print "shape =",np.shape(self._data_layer_1_X),np.shape(self._data_layer_1_y)
		
		#for i in range(np.shape(self._data_layer_1_X)[0]):
			#print i,self._data_layer_1_X[i],self._data_layer_1_y[i]
		
		self._clfr2=Classify.LogisticRegression(self._data_layer_1_X, self._data_layer_1_y,None, None)

	def TestSeconLayer(self):
		for subject in self._subjects_train_testing:
			filename = self._Path2Data+'/train_subject%02d.mat' % subject
			X,y=self.GetTrainData(filename)
			self._data_layer_1_y_testing.append(y)
			self._data_X_testing.append(X)
			self._data_y_testing.append(y)

		self._data_layer_1_y_testing=np.concatenate(self._data_layer_1_y_testing)
		
		#now create first layer of predictions using the classifers made
		for i in range(len(self._subjects_train_class)):
			#print "\n classifer ",i
			ypred_1=[]
			for j in range(len(self._subjects_train_testing)):
				#subject=self._subjects_train_testing[j]
				#filename = self._Path2Data+'/train_subject%02d.mat' % subject
				#X,y=self.GetTrainData(filename)
				#ypred=y
				ypred=self._first_layer_classifiers[i].predict(self._data_X_testing[j])					
				ypred_1.append(ypred)
			
			ypred_1 = np.concatenate(ypred_1)
			self._data_layer_1_X_testing.append(ypred_1)
			
		self._data_layer_1_X_testing=np.vstack(self._data_layer_1_X_testing).T
		
		#print "shape of testing data = ",np.shape(self._data_layer_1_X_testing),np.shape(self._data_layer_1_y_testing)
		
		ypred_test = self._clfr2.predict(self._data_layer_1_X_testing)
		print "error for 2nd classifer =", float(sum(abs(ypred_test-self._data_layer_1_y_testing)))/float(len(self._data_layer_1_y_testing))*100,"%"
		

	def PredictTestData(self):
		print "prediction"
		for subject in self._subjects_test:
			filename = self._Path2Data+'/test_subject%02d.mat' % subject
			X,ids=self.GetTestData(filename)
			self._data_X_testing.append(X)
			self._ids_test.append(ids)
		
		self._ids_test = np.concatenate(self._ids_test)
		
		#now create first layer of predictions
		for i in range(len(self._subjects_train)):#since we have subjects_train number of features here
			ypred_1=[]
			for j in range(len(self._subjects_test)):
				ypred=self._first_layer_classifiers[i].predict(self._data_X_testing[j])
				ypred_1.append(ypred)
			
			ypred_1 = np.concatenate(ypred_1)
			self._data_layer_1_X_testing.append(ypred_1)
			
		self._data_layer_1_X_testing=np.vstack(self._data_layer_1_X_testing).T
		
		
		self._ypred_test = self._clfr2.predict(self._data_layer_1_X_testing)
		
		return (self._ypred_test,self._ids_test)
					
			
