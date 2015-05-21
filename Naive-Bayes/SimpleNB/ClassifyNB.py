def classify(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    from sklearn.naive_bayes import GaussianNB
    ### create classifier
    cl = GaussianNB()
    ### fit the classifier on the training features and labels
    cl.fit(features_train,labels_train)
    ### return the fit classifier
    return cl
        
    ### your code goes here!
    

