## Clearing the work environment
# import sys
#
# this = sys.modules[__name__]
# for n in dir():
#     if n[0] != '_' and n != 'this':
#         delattr(this, n)

## Importing packages
#import PIL
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

## Set the working directory
try:
    os.chdir("/Users/sdini/PycharmProjects/CropClass/")
except:
    os.chdir("/mnt/CropClass/")

data = pd.read_csv("./Images/pixels.csv")
## Exploratory data analysis

# Remove any pixels that have a cloud probability of over 2.
data = data[data.cloud_prob <= 2]
X_all = data.iloc[:, :-3]
y_all = data["label"]

## Exploratory data analysis

# number of classes
print(y_all.value_counts())

plt.boxplot(X_all)
plt.xticks(np.arange(np.shape(X_all)[1])+1, list(X_all.columns))

le = preprocessing.LabelEncoder()
y_all_nu = le.fit_transform(y_all)

plt.boxplot(X_all)
plt.xticks(np.arange(np.shape(X_all)[1])+1, list(X_all.columns))

# Splitting the data into train and test for later validation and confusion matrix
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all_nu, test_size=0.3, random_state=42)

# making sure the proportion of classes are the same across train and test
print(pd.Series(y_train).value_counts()/len(y_train))
print(pd.Series(y_test).value_counts()/len(y_test))

## Preprocessing
# Impute missing values

## Random Forest -----------------------------------------
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

## X should be standardised as log regression is sensitive to scale of X
X_all_st = StandardScaler().fit_transform(X_all)

## To overcome the issue of overfitting to the majority class, I randomly d
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.pipeline import Pipeline

## Simple random forest ------------------------------------------------------
from sklearn.svm import SVC
param_grid = {'C':[1,10,100,1000],
              'gamma':[1,0.1,0.001,0.0001],
              'kernel':['linear','rbf']
              }

SVM = SVC(class_weight = 'balanced') #  class_weight = 'balanced'

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 1001)

SVM_cv = GridSearchCV(
    estimator=SVM,
    param_grid = param_grid,
    scoring = 'balanced_accuracy',
    return_train_score = True,
    n_jobs = -1,
    cv=skf.split(X_all,y_all_nu)
)

X_all_st = StandardScaler().fit_transform(X_all)

## Saving the model
# pkl_filename = "SVM_cv_results_" + "CW_None" + ".pkl"
pkl_filename = "SVM_str_cv_results.pkl"
if True:
    SVM_cv_results = SVM_cv.fit(X_all_st, y_all_nu)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(SVM_cv_results.best_params_, file)
else:
    with open(pkl_filename, 'rb') as file:
        SVM_cv_results = pickle.load(file)

print(SVM_cv_results.best_params_)

# cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)
# scores_SVM = cross_val_score(SVM_cv_results.best_estimator_, X_all, y_all_nu, scoring='roc_auc', cv=cv, n_jobs=-1)
# print(scores_SVM)

## Fit the model to training dataset and Check the results on the
fit_traindata = SVM_cv_results.best_estimator_.fit(X_train, y_train)

plot_confusion_matrix(fit_traindata, X_test, y_test)
plt.title("SVM")
plt.savefig("./Results/" + pkl_filename[:-4] + ".png")
