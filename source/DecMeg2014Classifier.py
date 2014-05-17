import numpy as np
from Features import CreateFeatures
import sys
from scipy.io import loadmat
from LogisticRegression import LogReg
from SVM import SuppVectMch

class DecMeg2014Classifier:
	"""
	A class for running a classifier on DecMeg2014 data.
	More info at https://www.kaggle.com/c/decoding-the-human-brain	
	"""

	def __init__(self,path_to_data=None):
		self._Path2Data=path_to_data
		self._X_train = []
		self._y_train = []
		self._X_test = []
		self._ids_test = []
		self._y_test = []
		self._y_pred=[]
		self._tmin = 0.0
		self._tmax = 0.5
		print "Restricting MEG data to the interval [%s, %s]sec." % (self._tmin, self._tmax)
		
	def ProcessData(self,XX,sfreq,tmin_original,nSVD):
		"""
		Process data 
		@param XX a matrix of the shape [trial x channel x time]
		@param sfreq Frequency of the sampling
		@param tmin_original Original start point of the window.
		"""
		cf=CreateFeatures()
	
		XX=cf.ApplyTimeWindow(XX, self._tmin, self._tmax, sfreq,tmin_original)
		XX=cf.ApplyZnorm(XX)
		cmin=5
		cmax=40
		#cf.ApplyFilter(XX,cmin,cmax)
		#cf.ApplySVD(XX,nSVD)
		return cf.ReshapeToFeaturesVector(XX)

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
	
		print "Training on subjects", subjects_train
	
	
		print "Creating the trainset."
		
		for subject in subjects_train:
			filename = self._Path2Data+'/train_subject%02d.mat' % subject
			print "Loading", filename
			data = loadmat(filename, squeeze_me=True)
			XX = data['X']
			yy = data['y']
			sfreq = data['sfreq']
			tmin_original = data['tmin']
			print "Dataset summary:"
			print "XX:", XX.shape
			print "yy:", yy.shape
			print "sfreq:", sfreq
			
			XX=self.ProcessData(XX,sfreq,tmin_original,nSVD)		
		
			self._X_train.append(XX)
			self._y_train.append(yy)
			
		self._X_train = np.vstack(self._X_train)
		self._y_train = np.concatenate(self._y_train)
		print "Trainset:", self._X_train.shape

	
	
	def MakeValidationData(self,subjects_test,nSVD):
		"""
		A function for creating validation data
		@param subjects_test A vector containing the test data numbers e.g range(1,12)
		"""
	
		print "Creating the testset."
		
		for subject in subjects_test:
			filename = self._Path2Data+'/train_subject%02d.mat' % subject
			print "Loading", filename
			data = loadmat(filename, squeeze_me=True)
			if not data.has_key('Id'):
				print "Data has no key called Id. Creating new ones"
				print
				data['Id']=subject*1000+np.arange(np.shape(data['X'])[0])
			XX = data['X']
			yy = data['y']
			ids = data['Id']
			sfreq = data['sfreq']
			tmin_original = data['tmin']
			print "Dataset summary:"
			print "XX:", XX.shape
			print "ids:", ids.shape
			print "sfreq:", sfreq
			
			XX=self.ProcessData(XX,sfreq,tmin_original,nSVD)
			
			self._X_test.append(XX)
			self._ids_test.append(ids)
			self._y_test.append(yy)
		
		self._X_test = np.vstack(self._X_test)
		self._ids_test = np.concatenate(self._ids_test)
		self._y_test = np.concatenate(self._y_test)
		print "Testset:", self._X_test.shape
		
	def RunClassifier(self):
		#clfr=LogReg()
		self._clfr=SuppVectMch()
		
		self._clfr.Train(self._X_train, self._y_train)
		
		self._y_pred=self._clfr.Predict(self._X_test)
		
	
	def ValidationScore(self):
		error=sum(abs(self._y_pred-self._y_test))
		total=self._y_pred.size
		return (total-error)/float(total)*100.
		
	def MakeSubmissionFile(self,filename_submission):
		print "Creating submission file", filename_submission
		f = open(filename_submission, "w")
		print >> f, "Id,Prediction"
		for i in range(len(y_pred)):
			print >> f, str(self._ids_test[i]) + "," + str(self._y_pred[i])
		f.close()
		
	def ValidationProbs(self,valid_file):
		df=self._clfr.DecisionFunction(self._X_test)
		print "shape =",np.shape(df)
		print "Creating validation probability file", valid_file
		f = open(valid_file, "w")
		print >> f, "Id,Prediction"
		for i in range(len(self._y_pred)):
			print >> f, str(self._ids_test[i]) + "," + str(self._y_pred[i]) + "," + str(self._y_test[i]) + "," + str(df[i][0])
		f.close()
		


####################################################################################################	
if __name__ == '__main__':
	if len(sys.argv) == 3 :
		dmc=DecMeg2014Classifier(sys.argv[1])
		nSVD=sys.argv[2]

		subjects_train=[1,3,5,7,9,11,13]#range(1,2)#
		dmc.MakeTrainData(subjects_train,nSVD)
		
		subjects_test=[2,4,6]#range(, 16)
		dmc.MakeValidationData(subjects_test,nSVD)
		
		dmc.RunClassifier()
		
		print "============================="
		print "         score= ",dmc.ValidationScore()
		print "============================="
		
		#dmc.ValidationProbs("valid_prob.txt")
	else:
		print "usage: python ",sys.argv[0],"<path-to-data> <numSVD>"
		print "example: python",sys.argv[0], "./data 10"
	
