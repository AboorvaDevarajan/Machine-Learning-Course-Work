#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 1 (Naive Bayes) mini-project 

    use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1

"""
    
import sys
from sklearn.naive_bayes import GaussianNB
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

# Defining Classifier - Gaussian Naive Bayes
clf = GaussianNB()

# Fitting the training data set
# training the test data set labels
t0 = time()
clf.fit(features_train,lables_train)
print("training naive bayes:", round(time()-t0, 3), "s")

#predicting the test dataset labels using the trained statistics
t0 = time()
pred = clf.pred(features_test)
print("predicting naive bayes:", round(time()-t0, 3), "s")


accuracy = clf.score(features_test,labels_test)
print(accuracy)

#########################################################


