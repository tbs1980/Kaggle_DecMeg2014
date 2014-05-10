"""A modified version of DecMeg2014 example code.

Simple prediction of the class labels of the test set by:
- pooling all the triaining trials of all subjects in one dataset.
- Extracting the MEG data in the first 500ms from when the
  stimulus starts.
- Smooth the data with some kernel
- Using a linear classifier (logistic regression).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.io import loadmat
import scipy.signal
from scipy import fftpack

def create_features(XX, tmin, tmax, sfreq, tmin_original=-0.5):
    """Creation of the feature space:
    - restricting the time window of MEG data to [tmin, tmax]sec.
    - Concatenating the 306 timeseries of each trial in one long
      vector.
    - Normalizing each feature independently (z-scoring).
    """
    print "Applying the desired time window."
    beginning = np.round((tmin - tmin_original) * sfreq).astype(np.int)
    end = np.round((tmax - tmin_original) * sfreq).astype(np.int)
    XX = XX[:, :, beginning:end].copy()
    
    # make some kind of kernel, there are many ways to do this...
    ksize=10
    t = 1 - np.abs(np.linspace(-1, 1, ksize))
    kernel = t.reshape(ksize, 1) * t.reshape(1, ksize)
    kernel /= kernel.sum()   # kernel should sum to 1!  :) 
    
    #print "smoothing the data"
    for i in range(np.shape(XX)[0]):
    	XX[i,:,:]=scipy.signal.convolve2d(XX[i,:,:],kernel, mode='same')
    	img=XX[i,:,:]
    	F1=fftpack.fft2(img)
    	F2 = fftpack.fftshift( F1 )
    	psd2D = np.abs( F2 )**2
    	XX[i,:,:]=psd2D

    print "2D Reshaping: concatenating all 306 timeseries."
    XX = XX.reshape(XX.shape[0], XX.shape[1] * XX.shape[2])

    print "Features Normalization."
    XX -= XX.mean(0)
    XX = np.nan_to_num(XX / XX.std(0))

    return XX
    
if __name__ == '__main__':

    print "DecMeg2014: https://www.kaggle.com/c/decoding-the-human-brain"
    print
    subjects_train = range(1, 7) # use range(1, 17) for all subjects
    print "Training on subjects", subjects_train 

    # We throw away all the MEG data outside the first 0.5sec from when
    # the visual stimulus start:
    tmin = 0.0
    tmax = 0.500
    print "Restricting MEG data to the interval [%s, %s]sec." % (tmin, tmax)

    X_train = []
    y_train = []
    X_test = []
    ids_test = []
    y_test = []

    print
    print "Creating the trainset."
    for subject in subjects_train:
        filename = '/arxiv/projects/Kaggle/Data/DecMeg2014/data/train_subject%02d.mat' % subject
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

        XX = create_features(XX, tmin, tmax, sfreq)

        X_train.append(XX)
        y_train.append(yy)

    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    print "Trainset:", X_train.shape

    print
    print "Creating the testset."
    #subjects_test = range(17, 24)
    subjects_test = range(8, 9)
    for subject in subjects_test:
        #filename = 'data/test_subject%02d.mat' % subject
        filename = '/arxiv/projects/Kaggle/Data/DecMeg2014/data/train_subject%02d.mat' % subject
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

        XX = create_features(XX, tmin, tmax, sfreq)

        X_test.append(XX)
        ids_test.append(ids)
        y_test.append(yy)

    X_test = np.vstack(X_test)
    ids_test = np.concatenate(ids_test)
    y_test = np.concatenate(y_test)
    print "Testset:", X_test.shape

    print
    clf = LogisticRegression(random_state=0) # Beware! You need 10Gb RAM to train LogisticRegression on all 16 subjects!
    print "Classifier:"
    print clf
    print "Training."
    clf.fit(X_train, y_train)
    print "Predicting."
    y_pred = clf.predict(X_test)

    print
    filename_submission = "smoothed_submission.csv"
    #filename_submission = "example_submission.csv"
    print "Creating submission file", filename_submission
    f = open(filename_submission, "w")
    print >> f, "Id,Prediction"
    for i in range(len(y_pred)):
        #print >> f, str(ids_test[i]) + "," + str(y_pred[i]) 
        print >> f, str(ids_test[i]) + "," + str(y_pred[i]) + "," + str(y_test[i])

    f.close()
    print "Done."


