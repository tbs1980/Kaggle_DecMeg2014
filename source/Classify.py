import numpy as np
from sklearn import cross_validation
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import NuSVC

def PrepareDataSets(x,y,testing_size=0.5,rand_state=0):
	"""
	Split the data into train and cross-validation sets.
	"""
	
	x_train, x_cv, y_train, y_cv = cross_validation.train_test_split(x, y, test_size=testing_size, random_state=rand_state)
	return (x_train, y_train, x_cv, y_cv)
	
def KNearestNeighbors(x_train, y_train, x_cv, y_cv, k=3):
	"""
	K Nearest Neighbors
	"""
	#print "Classifier: K Nearest Neighbors"
	clfr = KNeighborsClassifier(n_neighbors=k)
	clfr.fit(x_train, y_train)
	#print 'Accuracy in training set: %f' % clfr.score(x_train, y_train)
	#print 'Accuracy in cv set: %f' % clfr.score(x_cv, y_cv)
	return clfr
	
def BernoulliNaiveBayes(x_train, y_train, x_cv, y_cv):
	"""
	Bernoulli Naive Bayes
	"""
	#print "Classifier: Bernoulli Naive Bayes"
	clfr = BernoulliNB()
	clfr.fit(x_train, y_train)
	#print 'Accuracy in training set: %f' % clfr.score(x_train, y_train)
	#print 'Accuracy in cv set: %f' % clfr.score(x_cv, y_cv)
	return clfr
	
def NaiveBayes(x_train, y_train, x_cv, y_cv):
	"""
	Naive Bayes
	"""
	#print "Classifier: Naive Bayes"
	clfr = MultinomialNB()
	clfr.fit(x_train, y_train)
	#print 'Accuracy in training set: %f' % clfr.score(x_train, y_train)
	#print 'Accuracy in cv set: %f' % clfr.score(x_cv, y_cv)
	return clfr

def RandomForest(x_train, y_train, x_cv, y_cv,sw=None):
	"""
	Random Forest
	"""
	#print "Classifier: Random Forest"
	clfr =  RandomForestClassifier(n_estimators = 50, max_features="log2")
	if sw != None:
		clfr.fit(x_train, y_train,sample_weight=sw)
	else:
		clfr.fit(x_train, y_train)
	#print 'Accuracy in training set: %f' % clfr.score(x_train, y_train)
	#if y_cv != None:
	#	print 'Accuracy in cv set: %f' % clfr.score(x_cv, y_cv)
	
	return clfr
	
def LogisticRegression(x_train, y_train, x_cv, y_cv):
	"""
	Logistic Regression
	"""
	print "Classifier: Logistic Regression"
	clfr = linear_model.LogisticRegression(penalty='l2', C=.03)
	#clfr = linear_model.LogisticRegression()
	clfr.fit(x_train, y_train)
	#print 'Accuracy in training set: %f' % clfr.score(x_train, y_train)
	#if y_cv != None:
		#print 'Accuracy in cv set: %f' % clfr.score(x_cv, y_cv)
	
	return clfr
	
def SupportVectorMachine(x_train, y_train, x_cv, y_cv,sw):
	"""
	Support Vector Machine
	"""
	#print "Classifier: Support Vector Machine"
	clfr = SVC(probability=True)
	if sw != None:
		clfr.fit(x_train, y_train,sample_weight=sw)
	else:
		clfr.fit(x_train, y_train)
	#print 'Accuracy in training set: %f' % clfr.score(x_train, y_train)
	#if y_cv != None:
		#print 'Accuracy in cv set: %f' % clfr.score(x_cv, y_cv)
	
	return clfr

def NonLinearSupportVectorMachine(x_train, y_train, x_cv, y_cv):
	"""
	Non Linear Support Vector Machine
	"""
	#print "Classifier: Support Vector Machine"
	clfr = NuSVC(probability=True)
	clfr.fit(x_train, y_train)
	#print 'Accuracy in training set: %f' % clfr.score(x_train, y_train)
	#if y_cv != None:
		#print 'Accuracy in cv set: %f' % clfr.score(x_cv, y_cv)
	
	return clfr
	
	
def SGDClassifier(x_train, y_train, x_cv, y_cv):
	"""
	SGD-Classifier
	"""
	#print "Classifier: Support Vector Machine"
	clfr = linear_model.SGDClassifier(loss="log",penalty="l1",alpha=0.03)
	clfr.fit(x_train, y_train)
	#print 'Accuracy in training set: %f' % clfr.score(x_train, y_train)
	#if y_cv != None:
		#print 'Accuracy in cv set: %f' % clfr.score(x_cv, y_cv)
	
	return clfr
	
def ComputeProbability(clfr, x):
	""" Gets the probability of being good. """
	
	prob = np.array(clfr.predict_proba(x))
	return prob[:, 1]

	

