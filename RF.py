######################################
## Saber Dini 5/8/2020
## Crops classification in satelite images
## using Random Forests
######################################
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

## Checking which number correspond to what class
pd.crosstab(y_all, y_all_nu)

plt.boxplot(X_all)
plt.xticks(np.arange(np.shape(X_all)[1])+1, list(X_all.columns))

# Splitting the data into train and test for later validation and confusion matrix
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all_nu, test_size=0.3, random_state=123)

# making sure the proportion of classes are the same across train and test
print(pd.Series(y_train).value_counts()/len(y_train))
print(pd.Series(y_test).value_counts()/len(y_test))

## Preprocessing
# Impute missing values

## Random Forest -----------------------------------------
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, plot_confusion_matrix


## To overcome the issue of overfitting to the majority class, I randomly d
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.pipeline import Pipeline

## Simple random forest ------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier
param_grid = {
    'bootstrap': [False],
    'max_depth': [12, 18, 24, 30, 36, 42, None], # 10, 50, 100,
    'max_features': ['auto'],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [2, 5],
    'n_estimators': [500, 1000, 1500, 2000] # , 500, 1000
}

RF = RandomForestClassifier(oob_score = False, class_weight = 'balanced')

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 123)

RF_cv = GridSearchCV(
    estimator=RF,
    param_grid = param_grid,
    scoring = 'balanced_accuracy',
    return_train_score = True,
    n_jobs = -1,
    cv=skf.split(X_all,y_all_nu)
)

## Saving the model
# pkl_filename = "RF_cv_results_" + "CW_None" + ".pkl"
pkl_filename = "RF_strcv_fine_results.pkl"
if True: # Change this to False to run the CV which can take a while
    RF_cv_results = RF_cv.fit(X_all, y_all_nu)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(RF_cv_results.best_estimator_, file)
else:
    with open(pkl_filename, 'rb') as file:
        # RF_cv_results = pickle.load(file)
        RF_cv_results = RandomForestClassifier(bootstrap=False, class_weight='balanced', max_depth=24,
                                               min_samples_leaf=2, min_samples_split=5,
                                               n_estimators=1000)

# print(RF_cv_results.best_params_)

# cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)
# scores_RF = cross_val_score(RF_cv_results.best_estimator_, X_all, y_all_nu, scoring='roc_auc', cv=cv, n_jobs=-1)
# print(scores_RF)

## Fit the model to training dataset and Check the results on the
# fit_traindata = RF_cv_results.best_estimator_.fit(X_train, y_train)

fit_traindata = RF_cv_results.fit(X_train, y_train)

confusion_matrix(y_true=y_test, y_pred=fit_traindata.predict(X_test), normalize="true")

plot_confusion_matrix(fit_traindata, X_test, y_test, normalize="true")
plt.xticks([0, 1, 2], ["Cotton", "Other", "Sorghum"])
plt.yticks([0, 1, 2], ["Cotton", "Other", "Sorghum"])
plt.title("RF")
plt.savefig("./Results/" + pkl_filename[:-4] + ".png")
