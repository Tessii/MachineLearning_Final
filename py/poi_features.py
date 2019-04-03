#!/usr/bin/python

import numpy as np

def create_and_select_features(dataset):
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
    
    print(dataset["Ratio_bonus_salary"])
    print(dataset["Fraction_bonus_total_payments"])
    print(dataset["Fraction_salary_total_payments"])
    print(dataset["Fraction_stock_value_payments"])
    print(dataset["from_poi"])
    print(dataset["to_poi"])
    return dataset, new_features
