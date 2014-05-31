from sklearn import svm

class SuppVectMch:
	"""
	A class for running logistic regression.
	"""
	def __init__(self):
		self._clf = svm.SVC(cache_size=1000,probability=False)
		
		print "Classifier:"
		print self._clf

	def Train(self,X_train, y_train):
		"""
		Train on data
		"""
		print "Training."
		self._clf.fit(X_train, y_train)
		
	def Predict(self,X_test):
		"""
		Predict the results.
		"""
		print "Predicting."
		y_pred = self._clf.predict(X_test)
		
		return y_pred
		
	def DecisionFunction(self,X_test):
		return self._clf.decision_function(X_test)
