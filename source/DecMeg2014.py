import time
import sys
import ExtractFeatures
import Classify
import Utils

if __name__ == '__main__':
	if len(sys.argv) == 4 :
		start_time = time.time()
		path=sys.argv[1]
		subjects_train=range(1,2)
		nSVD=sys.argv[2]
		ef=ExtractFeatures.ExtractFeatures(path)
		X,y=ef.MakeTrainData(subjects_train,nSVD)
		X_tr,y_tr,X_cv,y_cv=Classify.PrepareDataSets(X,y)
		clfr = Classify.RandomForest(X_tr, y_tr, X_cv, y_cv)
		
		print "BinomialDeviance= ",Utils.BinomialDeviance(y_tr,Classify.ComputeProbability(clfr, X_tr))
		print "BinomialDeviance= ",Utils.BinomialDeviance(y_cv,Classify.ComputeProbability(clfr, X_cv))
		
		print (time.time() - start_time) / 60.0, 'minutes'
	else:
		print "usage: python ",sys.argv[0],"<path-to-data> <numSVD> <output-file>"
		print "example: python",sys.argv[0], "./data 10 ./submission.csv"

