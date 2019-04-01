#!/usr/bin/python

import sys
import pickle
import pandas as pd
import csv
import numpy as np
from collections import Counter

# Run: python poi_id.py

sys.path.append("../tools/")

# sklearn
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Given .py
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# Own .py
from poi_cleaner import clean_data
from poi_features import create_and_select_features
from poi_classifier import test_clf

# other
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import classification_report

### Task 1: Select what features you'll use.
raw_features_list = ['poi','name', 'salary', 'to_messages', 'deferral_payments',
                   'total_payments','loan_advances', 'bonus',
                   'email_address', 'restricted_stock_deferred', 'deferred_income',
                   'total_stock_value', 'expenses', 'from_poi_to_this_person',
                   'exercised_stock_options', 'from_messages', 'other',
                   'from_this_person_to_poi', 'long_term_incentive',
                   'shared_receipt_with_poi', 'restricted_stock', 'director_fees']

old_features_list =  ['poi', 'salary', 'to_messages', 'deferral_payments',
                      'total_payments','loan_advances', 'bonus',
                      'restricted_stock_deferred', 'deferred_income',
                      'total_stock_value', 'expenses', 'from_poi_to_this_person',
                      'exercised_stock_options', 'from_messages', 'other',
                      'from_this_person_to_poi', 'long_term_incentive',
                      'shared_receipt_with_poi', 'restricted_stock', 'director_fees'] 

features_list =  ['poi', 'salary', 'to_messages', 'deferral_payments',
                'total_payments','loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income',
                 'total_stock_value', 'expenses', 'from_poi_to_this_person',
                 'exercised_stock_options', 'from_messages', 'other',
                 'from_this_person_to_poi', 'long_term_incentive',
                 'shared_receipt_with_poi', 'restricted_stock', 'director_fees'] #without name and email adress

financial_features =["salary", "deferral_payments", "total_payments", "loan_advances",
                     "bonus","restricted_stock_deferred","deferred_income",
                     "total_stock_value","expenses","exercised_stock_options",
                     "other","long_term_incentive","restricted_stock","director_fees"]

incomplete_features = ["loan_advances", "director_fees",
                       "restricted_stock_deferred", "deferral_payments"]

incomplete_persons = ['WHALEY DAVID A','WROBEL BRUCE','LOCKHART EUGENE E',
                      'THE TRAVEL AGENCY IN THE PARK','GRAMM WENDY L']



## Load the dictionary containing the dataset
with open('final_project_dataset.pkl','rb') as f:
    enron_data = pickle.load(f)

## Clean Dataset
clean_data(enron_data, raw_features_list, financial_features, incomplete_persons)

## Import cleaned CSV
cleaned_enron = pd.read_csv("enron.csv")
cleaned_enron = cleaned_enron.set_index("name")

### Task 3: Create new feature(s)

my_dataset, new_features = create_and_select_features(cleaned_enron)

print(type(features_list))

'''# Remove feature list
for feature in removed_features:
    #print(feature)
    features_list.remove(feature)
    #print(type(features_list))'''

# Run: python poi_id.py

# Add new features
for feature in new_features:
    #print(type(features_list))
    #print(feature)
    features_list.append(feature)


### Task 4: Try a varity of classifiers
''' Please name your classifier clf for easy export below.
Note that if you want to do PCA or other multi-stage operations,
you'll need to use Pipelines. For more info:
http://scikit-learn.org/stable/modules/pipeline.html'''


def prepare_dataset_alternative(my_dataset, features_list):
    my_dataset = my_dataset.fillna(0)
    my_datadict = my_dataset.to_dict('index')
    #print(my_datadict['GLISAN JR BEN F'])
    data = featureFormat(my_datadict, features_list, sort_keys = True)
    #print(type(data))
    labels, features = targetFeatureSplit(data)

    print("Split successful")
    
    features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
    #print(len(features_train))
    #print(len(labels_train))
    
    return features_train, features_test, labels_train, labels_test

# Run: python poi_id.py

## Test classifiers with old dataset
features_train, features_test, labels_train, labels_test = prepare_dataset_alternative(cleaned_enron, old_features_list)

'''print(features_train)
print(len(features_train))
print(len(labels_train))
print(labels_train)'''

print ("Naives Bayes with old dataset:")
clf = GaussianNB()
clf = test_clf(features_train, labels_train, features_test, labels_test, clf)

print ("SVC with old dataset:")
clf = SVC(kernel="rbf")
clf = test_clf(features_train, labels_train, features_test, labels_test, clf)

print ("AdaBoost with old dataset:")
clf = AdaBoostClassifier()
clf = test_clf(features_train, labels_train, features_test, labels_test, clf)

print ("RandomForest with old dataset:")
clf = RandomForestClassifier()
clf = test_clf(features_train, labels_train, features_test, labels_test, clf)

print ("GradientBoosting with old dataset:")
clf = GradientBoostingClassifier()
clf = test_clf(features_train, labels_train, features_test, labels_test, clf)

print ("DecisionTree with old dataset:")
clf = DecisionTreeClassifier()
clf = test_clf(features_train, labels_train, features_test, labels_test, clf)

print ("KNeighbors with old dataset:")
clf = KNeighborsClassifier(n_neighbors = 3)
clf = test_clf(features_train, labels_train, features_test, labels_test, clf)       

#print(features_list)

## Test classifiers with new dataset
features_train, features_test, labels_train, labels_test = prepare_dataset_alternative(my_dataset, features_list)

print ("Naives Bayes with new dataset:")
clf = GaussianNB()
clf = test_clf(features_train, labels_train, features_test, labels_test, clf)

print ("SVC with new dataset:")
clf = SVC(kernel="rbf")
clf = test_clf(features_train, labels_train, features_test, labels_test, clf)

print ("AdaBoost with new dataset:")
clf = AdaBoostClassifier()
clf = test_clf(features_train, labels_train, features_test, labels_test, clf)

print ("RandomForest with new dataset:")
clf = RandomForestClassifier()
clf = test_clf(features_train, labels_train, features_test, labels_test, clf)

print ("GradientBoosting with new dataset:")
clf = GradientBoostingClassifier()
clf = test_clf(features_train, labels_train, features_test, labels_test, clf)

print ("DecisionTree with new dataset:")
clf = DecisionTreeClassifier()
clf = test_clf(features_train, labels_train, features_test, labels_test, clf)

print ("KNeighbors with new dataset:")
clf = KNeighborsClassifier(n_neighbors = 3)
clf = test_clf(features_train, labels_train, features_test, labels_test, clf)

# Run: python poi_id.py

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
'''using our testing script. Check the tester.py script in the final project
folder for details on the evaluation method, especially the test_classifier
function. Because of the small size of the dataset, the script uses
stratified shuffle split cross validation. For more info: 
http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html'''

print("Tune your classifier")

print("Best n_neighbors = 3")
print ("KNeighbors with new dataset and tuned parameters:")
clf = KNeighborsClassifier(n_neighbors = 2)
clf = test_clf(features_train, labels_train, features_test, labels_test, clf)

print ("KNeighbors with new dataset and tuned parameters:")
clf = KNeighborsClassifier(n_neighbors = 4)
clf = test_clf(features_train, labels_train, features_test, labels_test, clf)


''' weights : str or callable, optional (default = ‘uniform’)
weight function used in prediction. Possible values:

‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
‘distance’ : weight points by the inverse of their distance. in this case,
closer neighbors of a query point will have a greater influence than neighbors
which are further away.
[callable] : a user-defined function which accepts an array of distances,
and returns an array of the same shape containing the weights.'''

print("Best Weights = ‘uniform’")
print ("KNeighbors with new dataset and tuned parameters:")
clf = KNeighborsClassifier(n_neighbors = 3, weights = 'uniform')
clf = test_clf(features_train, labels_train, features_test, labels_test, clf)

print ("KNeighbors with new dataset and tuned parameters:")
clf = KNeighborsClassifier(n_neighbors = 3, weights = 'distance')
clf = test_clf(features_train, labels_train, features_test, labels_test, clf)

print("Best Algotithm: 'brute' (Shortest training and predicting time)")
print ("KNeighbors with new dataset and tuned parameters:")
clf = KNeighborsClassifier(n_neighbors=3, algorithm = 'ball_tree')
clf = test_clf(features_train, labels_train, features_test, labels_test, clf)

print ("KNeighbors with new dataset and tuned parameters:")
clf = KNeighborsClassifier(n_neighbors=3, algorithm = 'kd_tree')
clf = test_clf(features_train, labels_train, features_test, labels_test, clf)

print ("KNeighbors with new dataset and tuned parameters:")
clf = KNeighborsClassifier(n_neighbors=3, algorithm = 'brute')
clf = test_clf(features_train, labels_train, features_test, labels_test, clf)

print ("KNeighbors with new dataset and tuned parameters:")
clf = KNeighborsClassifier(n_neighbors=3, algorithm = 'auto')
clf = test_clf(features_train, labels_train, features_test, labels_test, clf)

print("Best p: 2 (Shortest training and predicting time)")
''' Power parameter for the Minkowski metric.
When p = 1, this is equivalent to using manhattan_distance (l1),
and euclidean_distance (l2) for p = 2.
For arbitrary p, minkowski_distance (l_p) is used.'''


print ("KNeighbors with new dataset and tuned parameters:")
clf = KNeighborsClassifier(n_neighbors=3, algorithm = 'brute', p = 1)
clf = test_clf(features_train, labels_train, features_test, labels_test, clf)

print ("KNeighbors with new dataset and tuned parameters:")
clf = KNeighborsClassifier(n_neighbors=3, algorithm = 'brute', p = 3)
clf = test_clf(features_train, labels_train, features_test, labels_test, clf)

print ("KNeighbors with new dataset and tuned parameters:")
clf = KNeighborsClassifier(n_neighbors=3, algorithm = 'brute', p = 2)
clf = test_clf(features_train, labels_train, features_test, labels_test, clf)

## Importance:
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


clf = KNeighborsClassifier()


# Run: python poi_id.py

### Task 6: Dump your classifier, dataset, and features_list so anyone can
''' check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results. '''

dump_classifier_and_data(clf, my_dataset, features_list)

'''
#my_dataset = my_dataset.to_dict('index')
#print(my_dataset)

'''
