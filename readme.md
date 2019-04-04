# Final Project

## Agenda:
1. Summary
2. Data Exploration
3. Feature
4. Algorithm
5. Parameter
6. Validation
7. Evaluation metrics

# Summary
[relevant rubric items: “data exploration”, “outlier investigation”]

## Goal
> Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it.

In this project, I will play detective, and put my new skills to use by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal.

## Background Dataset
> As part of your answer, give some background on the dataset and how it can be used to answer the project question.

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. [...] this data [contains] a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity. [Source]( https://classroom.udacity.com/nanodegrees/nd002-airbus/parts/96af8ace-b538-440f-83e6-4cc6eb42caa5/modules/317428862475461/lessons/3174288624239847/concepts/31803986370923)

## Used 
- Python 3
- R Studio

## Github structure
- R = All files regarding Data Exploration in R Studio
- Py = All supporting py files
- Other = Additional files (e.g. PDFs)

# Data Exploration
To explore the dataset I used R Studio. Please look at the HTML.

## Outliers / Incomplete Data
> Were there any outliers in the data when you got it, and how did you handle those?
**Outlier**
Based on the analysis, the line "Total" was identified and excluded.

**Incomplete Persons** Based on counting NaNs the following persons were excluded as 80 % of their data is missing:
- Whaley David A
- Wrobel Bruce
- Lockhart Eugene E
- The Travel Agency in the park
- Gramm Wendy

**Incomplete features** Based on counting NaNs the following features had over 80% missing data: were excluded as 80 % of their data is missing:
- loan_advances
- director_fees 
- restricted_stock_deferred
- deferral_payments
Those features belong to the group **Financial features**. Based on the insider-pay.pdf I believe that NaNs of financial features are equivalent to zero. Therefore, I replaced all NaNs of financial features. As a result I mistakenly assumed that these features are incomplete. 

Furthermore, the following persons were identified as potential outliers based on the high values:

**High Salary**:
- Frevert Mark
- Skilling Jeffrey
- Lay Kenneth

**High Bonus**:
- Lay Kenneth
- Lavorato John

A detailed investigation showed that two out of four persons were POIs. In conclsuion, these points are not outliers.

# Features
[relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]

## Given Features
The data provides three types of features.
**Structure** (Dict in Dict):
{Name: {Feature1: Value1, Feature2: Value2...}
{'METTS MARK': {'salary': 365788, 'to_messages': 807....}

**Financial features (in US dollars)**:
- salary
- deferral_payments
- total_payments
- loan_advances
- bonus
- restricted_stock_deferred
- deferred_income
- total_stock_value
- expenses
- exercised_stock_options
- long_term_incentive
- restricted_stock
- director_fees

**Email features (ea; except email_address(string))**
- to_messages
- email_address
- from_poi_to_this_person
- from_messages
- from_this_person_to_poi
- shared_receipt_with_poi

**POI labels (integer)**:
- poi

## New Feature
> As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the datase
> -- explain what feature you tried to make, and the rationale behind it.
> (You do not necessarily have to use it in the final analysis, only engineer and test it.)

By calculating the ratio or fraction between the given features. It provides us a new way to identify frauds by seeing the different features in relation to each other.
(E.g. a high salary doesn't mean someone is a POI, but the high ratio between bonus / salary could be an indicator.)

### New financial features
- Ratio_bonus_salary = bonus / salary
- Fraction_bonus_total_payments = bonus / total_payments
- Fraction_salary_total_payments = salary / total_payments
- Fraction_stock_value_payments = total_stock_value / total_payments

### New email features
- from_poi = from_this_person_to_poi / from_messages
- to_poi = from_poi_to_this_person / to_messages

## Scaling
> Did you have to do any scaling? Why or why not?

Scaling is important, when you are comparing features with different scales (e.g. height (ft) and weight (lbs)) to avoid that one feature had a greater impact on the result. As financial features are all in US Dollar and the email features are all in unit. Scaling was not necassary. Especially by using the new features, which show relations between features. 

## Used Feature
> What features did you end up using in your POI identifier, and what selection process did you use to pick them?

Following features wer not used:
Email address at is provides no value for prediction.

Following given features were used:
features_list =  ['salary', 'to_messages', 'deferral_payments',
                'total_payments','loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income',
                 'total_stock_value', 'expenses', 'from_poi_to_this_person',
                 'exercised_stock_options', 'from_messages', 'other',
                 'from_this_person_to_poi', 'long_term_incentive',
                 'shared_receipt_with_poi', 'restricted_stock', 'director_fees']. 
                 
All other created features were used as well.

# Algorithm
[relevant rubric item: “pick an algorithm”]

## Algoithm choice
> What algorithm did you end up using?
**sklearn.neighbors.KNeighborsClassifier**

The principle behind nearest neighbor methods is to find a predefined number of training samples closest in distance to the new point, and predict the label from these. The number of samples can be a user-defined constant (k-nearest neighbor learning), or vary based on the local density of points (radius-based neighbor learning). The distance can, in general, be any metric measure: standard Euclidean distance is the most common choice. Neighbors-based methods are known as non-generalizing machine learning methods, since they simply “remember” all of its training data (possibly transformed into a fast indexing structure such as a Ball Tree or KD Tree). (Source:
[Source](https://scikit-learn.org/stable/modules/neighbors.html)


## Which other I have tried
> What other one(s) did you try? How did model performance differ between algorithms? 

### Old Dataset
Naives Bayes with old dataset:
- Accuracy: 0.2619047619047619
- Precision: 0.12121212121212122
- Recall: 0.6666666666666666
- training time: 0.004 s
- predicting time: 0.003 s

SVC with old dataset:
- Accuracy: 0.8571428571428571
- Precision: 0.0
- Recall: 0.0
- training time: 0.012 s
- predicting time: 0.005 s

AdaBoost with old dataset:
- Accuracy: 0.8809523809523809
- Precision: 0.6666666666666666
- Recall: 0.3333333333333333
- training time: 0.072 s
- predicting time: 0.005 s

RandomForest with old dataset:
- Accuracy: 0.8809523809523809
- Precision: 0.6666666666666666
- Recall: 0.3333333333333333
- training time: 0.014 s
- predicting time: 0.003 s

GradientBoosting with old dataset:
- Accuracy: 0.8571428571428571
- Precision: 0.5
- Recall: 0.3333333333333333
- training time: 0.06 s
- predicting time: 0.004 s

DecisionTree with old dataset:
- Accuracy: 0.8095238095238095
- Precision: 0.0
- Recall: 0.0
- training time: 0.002 s
- predicting time: 0.001 s

KNeighbors with old dataset:
- Accuracy: 0.8571428571428571
- Precision: 0.0
- Recall: 0.0
- training time: 0.01 s
- predicting time: 0.003 s

### New Dataset:
Naives Bayes with new dataset:
Accuracy: 0.2619047619047619
Precision: 0.12121212121212122
Recall: 0.6666666666666666
training time: 0.002 s
predicting time: 0.002 s

SVC with new dataset:
Accuracy: 0.8571428571428571
Precision: 0.0
Recall: 0.0
training time: 0.003 s
predicting time: 0.002 s

AdaBoost with new dataset:
Accuracy: 0.8571428571428571
Precision: 0.5
Recall: 0.3333333333333333
training time: 0.082 s
predicting time: 0.006 s

RandomForest with new dataset:
Accuracy: 0.8809523809523809
Precision: 1.0
Recall: 0.16666666666666666
training time: 0.014 s
predicting time: 0.002 s

GradientBoosting with new dataset:
Accuracy: 0.8571428571428571
Precision: 0.5
Recall: 0.3333333333333333
training time: 0.061 s
predicting time: 0.002 s

DecisionTree with new dataset:
Accuracy: 0.7857142857142857
Precision: 0.2857142857142857
Recall: 0.3333333333333333
training time: 0.002 s
predicting time: 0.002 s

KNeighbors with new dataset:
Accuracy: 0.9047619047619048
Precision: 1.0
Recall: 0.3333333333333333
training time: 0.003 s
predicting time: 0.002 s 

# Parameter
[relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]

## Puprose and Effect of parameters
> What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?

Tuning the parameters of algorithm is the process of changing, testing and updating the parameters with the goal to improve the outcome of the classifier. By tuning the parameters its important to look on different metrics to evualate your choosen parameters. This is because, on the hand you could increase the accuracy, but on the other hand you could decrease the recall.

## Application on my algorithm
> How did you tune the parameters of your particular algorithm? What parameters did you tune?
> (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, 
> identify and briefly explain how you would have done it for the model that was not your final choice 
> or a different model that does utilize parameter tuning, e.g. a decision tree classifier). 

Using KNeighborsClassifier it was necessary to set the parameter **n_neighbors** to three. I also tried to tune other parameters, but just tuning the **n_neighbors** paramater to 3, remained the best tuning.

## Possible Parameter for KNeighbors
- **n_neighbors** : int, optional (default = 5)
Number of neighbors to use by default for kneighbors queries.
- **weights** : str or callable, optional (default = ‘uniform’)
weight function used in prediction. Possible values:
- **‘uniform’** : uniform weights. All points in each neighborhood are weighted equally.
- **‘distance’** : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
- **[callable]** : a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights.
- **algorithm** : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional
Algorithm used to compute the nearest neighbors:
‘ball_tree’ will use BallTree
‘kd_tree’ will use KDTree
‘brute’ will use a brute-force search.
‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.
Note: fitting on sparse input will override the setting of this parameter, using brute force.
- **leaf_size** : int, optional (default = 30)
Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.
- **p** : integer, optional (default = 2)
Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
- **metric** : string or callable, default ‘minkowski’
the distance metric to use for the tree. The default metric is minkowski, and with p=2 is equivalent to the standard Euclidean metric. See the documentation of the DistanceMetric class for a list of available metrics.
- **metric_params** : dict, optional (default = None)
Additional keyword arguments for the metric function.
- **n_jobs** : int or None, optional (default=None)
The number of parallel jobs to run for neighbors search. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details. Doesn’t affect fit method.
[Source](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

# Validation
[relevant rubric items: “discuss validation”, “validation strategy”]
> What is validation, and what’s a classic mistake you can make if you do it wrong?

The purpose of validation is to achieve a stable performance of the algorithm on all data. Without validation, we would use  all available data, which would lead to a great performance on the used dataset, an overfit, but would alsoprobaly lead to a bad performance on unseen data.

Therefore, we split the avaiable data in a training and testset to use them for cross validation.

# Evaluation metrics
[relevant rubric item: “usage of evaluation metrics”]

> How did you validate your analysis?  
By using the following metrics and mainly focussing on the *Recall*

1. Accuracy
2. Recall
3. Precision
4. training time & predicting time*

## Used Evaluation metrics
> Give at least 2 evaluation metrics and your average performance for each of them.
> Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance.

**1. Accuracy**
lc = number of items labeled correctly
ai = all items

| Category | Description |
| --- | --- |
| *In general*:  | Accuracy =  lc / ai | The best value is 1 and the worst value is 0.|
| *Risk*: | By a skewed distrubution this metric could be misleading. | 
| *In the Enron case*: | Accuracy = Correctly labeled whether a person is or is not a POI / all persons. In this case, there are more non-POIs than POIs and therefore its easier to achieve a high accuracy.|

**2. Recall**
tp = number of true positives
fn = number of false negatives

| Category | Description |
| --- | --- |
| *In general*: | tp / (tp + fn) The best value is 1 and the worst value is 0. |
| *Risk*:| High risk of false negatives solutions |
| *In the Enron case*:| This metric is really important in this case as the target of the algo is to identify as many positive samples even by risiking to identify innocent people along the way.|

**3.Precision**
tp = number of true positives
fp = number of false positives

| Category          | Description |
| --- | --- |
|*In general*: |tp / (tp + fp)| The best value is 1 and the worst value is 0.|
|*In the Enron case*:| This metric is a great addition to the previous metric. It shows the confidence level how likely it is that a flaged person is a real POI.| 

**4. training time & predicting time**
Those times shows how long it took to train a algotihm and how long its prediction takes.
For all used algorithm both times were incredible small, always below one second. The short times are result of the small dataset.
