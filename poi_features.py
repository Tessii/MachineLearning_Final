#!/usr/bin/python

import sys
import pickle
import pandas as pd
import csv
import numpy as np
from collections import Counter

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

# Create new features
def create_features(dataset):
    new_features =["Ratio_bonus_salary", "Fraction_bonus_total_payments", "Fraction_salary_total_payments",
                   "Fraction_stock_value_payments", "from_poi", "to_poi"]

    dataset = dataset.drop('email_address', axis =1)

    # New financial features
    dataset["Ratio_bonus_salary"] = dataset["bonus"] / dataset["salary"]
    dataset["Fraction_bonus_total_payments"] = dataset["bonus"] / dataset['total_payments']
    dataset["Fraction_salary_total_payments"] = dataset["salary"] / dataset['total_payments']
    dataset["Fraction_stock_value_payments"] = dataset["total_stock_value"] / dataset['total_payments']

    # New email features
    dataset["from_poi"] = dataset["from_this_person_to_poi"] / dataset["from_messages"]
    dataset["to_poi"] = dataset["from_poi_to_this_person"] / dataset["to_messages"]

    dataset.replace(np.inf, 0, inplace=True)
    return dataset, new_features

#Split data in labels and features and create test and train dataset
def split_dataset(my_dataset, features_list):
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


# Split data in labels and features
def split_dataset_with_optimization(my_dataset, features_list):
    my_dataset = my_dataset.fillna(0)
    my_datadict = my_dataset.to_dict('index')
    # print(my_datadict['GLISAN JR BEN F'])
    data = featureFormat(my_datadict, features_list, sort_keys = True)
    # print(type(data))
    labels, features = targetFeatureSplit(data)

    # Splitting dataset in train and test using StratifiedShuffleSplit
    cv = StratifiedShuffleSplit(labels, n_iter=100, test_size=0.75, random_state=42)
    for train_index, test_index in cv:
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        for ii in train_index:
            features_train.append(features[ii])
            labels_train.append(labels[ii])
        for jj in test_index:
            features_test.append(features[jj])
            labels_test.append(labels[jj])
        
    #Using SelectKBest for feature selection
    feature_selection = FeatureUnion([
        ('kbest', SelectKBest(f_classif)),
        ('pca', PCA())])

    return features_train, features_test, labels_train, labels_test, feature_selection







