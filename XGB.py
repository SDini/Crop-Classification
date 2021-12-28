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
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
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

try:
    plt.boxplot(X_all)
    plt.xticks(np.arange(np.shape(X_all)[1])+1, list(X_all.columns))
except:
    print("no graphic device available")

le = preprocessing.LabelEncoder()
y_all_nu = le.fit_transform(y_all)

plt.boxplot(X_all)
plt.xticks(np.arange(np.shape(X_all)[1])+1, list(X_all.columns))

# Splitting the data into train and test for later validation and confusion matrix
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all_nu, test_size=0.3, random_state=42)

##### XGBoost ------------------------------------------
import xgboost as xgb

param_grid = {
    #'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.6, 1.0],
    'max_depth': [3, 5],
    'n_estimators': [300, 600]
}

XGB = xgb.XGBClassifier(objective='multi:softmax',
                  seed=123,
                  #learning_rate=0.3,
                  # n_estimators=300,
                  )

xgb_cv = GridSearchCV(
    estimator=XGB,
    param_grid=param_grid,
    scoring='balanced_accuracy',
    #return_train_score=True,
    n_jobs=-1,
    cv=3
)

pkl_filename = "XGB_cv_results.pkl"
if True:
    xgb_cv_results = xgb_cv.fit(X_all, y_all_nu)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(xgb_cv_results, file)
else:
    with open(pkl_filename, 'rb') as file:
        xgb_cv_results = pickle.load(file)

fit_traindata = xgb_cv_results.best_estimator_.fit(X_train, y_train)

# fit_traindata = xgb_cv_results.fit(X_train, y_train)

confusion_matrix(y_true=y_test, y_pred=fit_traindata.predict(X_test), normalize="true")

plot_confusion_matrix(fit_traindata, X_test, y_test, normalize="true")
plt.title("XGBoost")
plt.savefig("./Results/" + pkl_filename[:-4] + ".png")
