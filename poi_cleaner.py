 #!/usr/bin/python

import sys
import pickle
import pandas as pd
import csv

# Run: python poi_id.py

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".



def clean_data(enron_data, features_list, financial_features, incomplete_persons):

    ### Load CSV
    with open('raw_enron.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=features_list)
        writer.writeheader()
        for name in enron_data.keys():
            n = {'name':name}
            n.update(enron_data[name])
            writer.writerow(n)

    ### Task 2: Remove outliers     
    ### Cleaning
    def count_nans(dataset):
        d = {}
        for person in dataset:
            for key, value in dataset[person].items():
                if value == "NaN":
                    if key in d:
                        d[key] += 1
                    else:
                        d[key] = 1
        return d

    print("NaNs:", count_nans(enron_data))

    ## Replace NaNs
    for feature in financial_features:
        for person in enron_data:
            if enron_data[person][feature] == "NaN":
                enron_data[person][feature] = 0

    ## Remove incomplete persons
    for person in incomplete_persons:
        enron_data.pop(person, None)

    '''## Remove based on incomplete features
    for person in enron_data:
        for feature in incomplete_features:
            enron_data[person].pop(feature)'''

    #print(enron_data)

    ## Remove Total
    enron_data.pop('TOTAL', None)

    for person in enron_data:
        enron_data[person].pop('email_address')
    
    print("NaNs:", count_nans(enron_data))
    print("Cleaning Successful")

    ### Load CSV
    with open('enron.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=features_list)
        writer.writeheader()
        for name in enron_data.keys():
            n = {'name':name}
            n.update(enron_data[name])
            writer.writerow(n)
