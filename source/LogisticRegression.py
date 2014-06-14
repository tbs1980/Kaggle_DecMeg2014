from sklearn.linear_model import LogisticRegression

class LogReg:
	"""
	A class for running logistic regression.
	"""
	def __init__(self):
		# Beware! You need 10Gb RAM to train LogisticRegression on all 16 subjects!
		self._clf = LogisticRegression(random_state=0)		
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
		
	def PredictProb(self,X_test):
		"""
		Predict the results.
		"""
		print "Predicting probabilities."
		y_pred_prob = self._clf.predict_proba(X_test)
		
		return y_pred_prob
