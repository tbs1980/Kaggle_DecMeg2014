import numpy as np
from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

def LogRegWithRBMFeatures(x_train, y_train, x_cv, y_cv):
	"""
	Logistic regression using RBM features
	http://scikit-learn.org/stable/auto_examples/plot_rbm_logistic_classification.html
	"""
	logistic = linear_model.LogisticRegression()
	#rbm = BernoulliRBM(random_state=0, verbose=True)
	rbm = BernoulliRBM()
	classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
	rbm.learning_rate = 0.06
	rbm.n_iter = 20
	rbm.n_components = 100
	#logistic.C = 6000.0
	
	classifier.fit(x_train, y_train)
	
	#classifier = BernoulliRBM(n_components = 10)
	#classifier.fit(x_train, y_train)
	
	return classifier
