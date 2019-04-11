#!/usr/bin/python

import sys
import pickle
import pandas as pd
import csv
import numpy as np
from collections import Counter
from collections import defaultdict

# Run: python poi_id_2.py

sys.path.append("../tools/")

# sklearn
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import classification_report

# Given .py
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# Own .py
from poi_cleaner import clean_data
from poi_features import create_features, split_dataset, split_dataset_with_optimization
from poi_classifier import test_clf

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

## Clean Dataset using poi_cleaner.py
clean_data(enron_data, raw_features_list, financial_features, incomplete_persons)

## Import cleaned CSV
cleaned_enron = pd.read_csv("enron.csv")
cleaned_enron = cleaned_enron.set_index("name")

### Task 3: Create new feature(s) using poi_features.py
my_dataset, new_features = create_features(cleaned_enron)

# Run: python poi_id_2.py

# Add new features to the feature list
for feature in new_features:
    #print(type(features_list))
    #print(feature)
    features_list.append(feature)

### Task 4: Try a varity of classifiers
''' Please name your classifier clf for easy export below.
Note that if you want to do PCA or other multi-stage operations,
you'll need to use Pipelines. For more info:
http://scikit-learn.org/stable/modules/pipeline.html'''

## Test classifiers with old dataset
features_train, features_test, labels_train, labels_test = split_dataset(cleaned_enron, old_features_list)

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

## Test classifiers with new dataset
features_train, features_test, labels_train, labels_test = split_dataset(my_dataset, features_list)

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

# Run: python poi_id_2.py

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
'''using our testing script. Check the tester.py script in the final project
folder for details on the evaluation method, especially the test_classifier
function. Because of the small size of the dataset, the script uses
stratified shuffle split cross validation. For more info: 
http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html'''

features_train, features_test, labels_train, labels_test, feature_selection = split_dataset_with_optimization(my_dataset, old_features_list)

# Using Pipeline to test parameters
pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', feature_selection),
        ('clf', KNeighborsClassifier(weights='distance', algorithm='ball_tree')) ])

grid_search = GridSearchCV(pipeline, {
        'feature_selection__kbest__k': [2, 3, 4, 5, 7, 10],
        'feature_selection__pca__n_components': [2, 5, 10, ],

        'clf__n_neighbors': [2, 3, 4, 6, 10],
        'clf__weights': ['distance', 'uniform'],
        'clf__algorithm': ['kd_tree', 'ball_tree', 'auto', 'brute'],}, scoring='recall')



grid_search.fit(features_train, labels_train)

clf = pipeline.set_params(**grid_search.best_params_)
pipeline.fit(features_train, labels_train)

print("Best Parameters: ")
best_para = grid_search.best_params_
print(best_para)


print("features")

#selected_features = feature_names[feature_selector_cv.support_].tolist()
#selected_features
#print(kbest())

pred = clf.predict(features_test)

report = classification_report(labels_test, pred)
print (report)

acc = accuracy_score(labels_test, pred)
print ("Accuracy:", acc)

pre = precision_score(labels_test, pred)
print ("Precision:", pre)
    
rec = recall_score(labels_test, pred)
print ("Recall:", rec)

# Run:  python poi_id_2.py

'''### Task 6: Dump your classifier, dataset, and features_list so anyone can
 check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results. 

print(my_dataset.head())
my_dataset_np = my_dataset.to_dict('index')
my_dataset_np = {k: {k2: 0 if v2 == 'nan' else v2 for k2, v2 in v.items()} \
                    for k, v in my_dataset_np.items()}

print(type(my_dataset_np))
print(my_dataset_np)


#print(type(enron_data))
#print((enron_data))

dump_classifier_and_data(clf, my_dataset_np, features_list)'''
