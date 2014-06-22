import numpy as np
import sys
from scipy.io import loadmat

def MakeSubmissionFile(y_pred,ids_test,filename_submission):
	print "Creating submission file", filename_submission
	f = open(filename_submission, "w")
	print >> f, "Id,Prediction"
	for i in range(len(y_pred)):
		print >> f, str(ids_test[i]) + "," + str(y_pred[i])
	f.close()
	
def BinomialDeviance(y, prediction):
	""" Calculates the binomial deviance for the prediction. """
	
	binomial_deviance = 0.0
	for i in range(len(prediction)):
		if prediction[i] > .99:
			prediction[i] = .99
		elif prediction[i] < .1:
			prediction[i] = .1
		tmp = y[i] * np.math.log10(prediction[i])
		tmp += (1 - y[i]) * np.math.log10(1 - prediction[i])
		binomial_deviance -= tmp
	binomial_deviance /= float(len(prediction))
	return binomial_deviance

