import time
import sys
import ExtractFeatures
import Classify
import Utils
import StackedGeneralisation
import StkGenSpectra
import StackedGeneralisationKclfr
import numpy as np

def calibrate(path):
	nsvds=np.arange(10,125,25)
	#nsvds=np.arange(10,25,5)
	bd_tr=np.zeros(np.shape(nsvds))
	bd_cv=np.zeros(np.shape(nsvds))
	acc_tr=np.zeros(np.shape(nsvds))
	acc_cv=np.zeros(np.shape(nsvds))
	for i in range(np.shape(nsvds)[0]):
		print "nsvd= ",nsvds[i]
		subjects_train=range(1,17)
		#subjects_train=range(1,3)
		ef=ExtractFeatures.ExtractFeatures(path)
		X,y=ef.MakeTrainData(subjects_train,nsvds[i])
		X_tr,y_tr,X_cv,y_cv=Classify.PrepareDataSets(X,y)
		#clfr = Classify.RandomForest(X_tr, y_tr, X_cv, y_cv)
		clfr = Classify.LogisticRegression(X_tr, y_tr, X_cv, y_cv)
		#clfr = Classify.BernoulliNaiveBayes(X_tr, y_tr, X_cv, y_cv)
		#clfr = Classify.NaiveBayes(X_tr, y_tr, X_cv, y_cv)#does not work. throws an error saying negative values X
		#clfr = Classify.KNearestNeighbors(X_tr, y_tr, X_cv, y_cv)
		#clfr = Classify.SupportVectorMachine(X_tr, y_tr, X_cv, y_cv)
		#clfr = Classify.NonLinearSupportVectorMachine(X_tr, y_tr, X_cv, y_cv)
		bd_tr[i]=Utils.BinomialDeviance(y_tr,Classify.ComputeProbability(clfr, X_tr))
		bd_cv[i]=Utils.BinomialDeviance(y_cv,Classify.ComputeProbability(clfr, X_cv))
		acc_tr[i]=clfr.score(X_tr, y_tr)
		acc_cv[i]=clfr.score(X_cv, y_cv)
	
	print ""
	print ""
	print "=========================="
	print "Binomia lDeviance"
	print "nsvd","train","cv"
	print "=========================="
	for i in range(np.shape(nsvds)[0]):
		print nsvds[i],bd_tr[i],bd_cv[i]
		
	print ""
	print ""
	print "=========================="
	print "Accuracy"
	print "nsvd","train","cv"
	print "=========================="
	for i in range(np.shape(nsvds)[0]):
		print nsvds[i],acc_tr[i],acc_cv[i]
		
	svdopt=nsvds[bd_cv.argmin()]
	print ""
	print ""
	print "optimal svd componetns= ",svdopt
	print ""
	
	return (clfr,svdopt)
		

def classify_test_data(path,clfr,svdopt,sub_file_name):
	subjects_test = range(17,24)
	ef=ExtractFeatures.ExtractFeatures(path)
	X_test,ids_test=ef.MakeTestData(subjects_test,svdopt)
	y_pred = clfr.predict(X_test)
	Utils.MakeSubmissionFile(y_pred,ids_test,sub_file_name)
	

if __name__ == '__main__':
	if len(sys.argv) == 3 :
		start_time = time.time()
		path=sys.argv[1]
		subfile=sys.argv[2]
		
		#clfr,svdopt=calibrate(path)
		#classify_test_data(path,clfr,svdopt,subfile)
		
		
		
		#sg.FindTheBestChannels()
		
		sg=StackedGeneralisation.StackedGeneralisation(path)		
		sg.CreateFistLayer()
		sg.TestSeconLayer();		
		#y_pred,ids_test=sg.PredictTestData()
		
		#sgh=StkGenSpectra.StackedGeneralisationWithHarmonicCoeffs(path)
		#sgh.CreateFistLayer()
		#sgh.CreateSecondLayer()
		
		#sgKc=StackedGeneralisationKclfr.StackedGeneralisationKclfr(path)
		#sgKc.CreateFistLayer()
		#sgKc.TestSeconLayer()
		#y_pred,ids_test=sgKc.PredictTestData()
		
		
		#Utils.MakeSubmissionFile(y_pred,ids_test,subfile)
		
		print (time.time() - start_time) / 60.0, 'minutes'
	else:
		print "usage: python ",sys.argv[0],"<path-to-data> <output-file>"
		print "example: python",sys.argv[0], "./data ./submission.csv"

