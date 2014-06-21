import time
import sys
import ExtractFeatures
import Classify
import Utils
import numpy as np

def calibrate(path):
	nsvds=np.arange(10,100,5)
	bd_tr=np.zeros(np.shape(nsvds))
	bd_cv=np.zeros(np.shape(nsvds))
	acc_tr=np.zeros(np.shape(nsvds))
	acc_cv=np.zeros(np.shape(nsvds))
	for i in range(np.shape(nsvds)[0]):
		print "nsvd= ",nsvds[i]
		subjects_train=range(1,3)
		ef=ExtractFeatures.ExtractFeatures(path)
		X,y=ef.MakeTrainData(subjects_train,nsvds[i])
		X_tr,y_tr,X_cv,y_cv=Classify.PrepareDataSets(X,y)
		clfr = Classify.RandomForest(X_tr, y_tr, X_cv, y_cv)
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
		
	return nsvds [bd_cv.argmin()]
		

if __name__ == '__main__':
	if len(sys.argv) == 4 :
		start_time = time.time()
		path=sys.argv[1]
		
		svdopt=calibrate(path)
		print ""
		print ""
		print "optimal svd componetns= ",svdopt
		print ""
		
		print (time.time() - start_time) / 60.0, 'minutes'
	else:
		print "usage: python ",sys.argv[0],"<path-to-data> <numSVD> <output-file>"
		print "example: python",sys.argv[0], "./data 10 ./submission.csv"

