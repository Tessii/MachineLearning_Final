#!/usr/bin/python
from time import time
from sklearn.metrics import accuracy_score, precision_score, recall_score

def test_clf(features_train, labels_train, features_test, labels_test, clf):
    t0 = time()
    clf.fit(features_train, labels_train) 
    t1 = time()
    pred = clf.predict(features_test)

    acc = accuracy_score(labels_test, pred)
    print ("Accuracy:", acc)

    pre = precision_score(labels_test, pred)
    print ("Precision:", pre)
    
    rec = recall_score(labels_test, pred)
    print ("Recall:", rec)
    
    print ("training time:", round(time()-t0, 3), "s")
    print ("predicting time:", round(time()-t1, 3), "s")
    print ("\n")
    return clf
