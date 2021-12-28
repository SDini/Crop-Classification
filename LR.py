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

## Preprocessing
# Impute missing values


## Elastic-net regularised Logistic regression -----------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from  sklearn import preprocessing
from sklearn.model_selection import cross_val_score, RepeatedKFold
# from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, plot_confusion_matrix


param_grid = {
    'l1_ratio': np.linspace(0, 1, 5).tolist(),
    'C': np.logspace(-3,3,10).tolist()
}

LR = LogisticRegression(random_state=1, class_weight = 'balanced', penalty= 'elasticnet', n_jobs = -1, solver='saga', max_iter=5000) # 'balanced'

LR_cv = GridSearchCV(
    estimator = LR,
    param_grid = param_grid,
    scoring = 'balanced_accuracy', # Try auroc as well
    return_train_score = True,
    n_jobs = -1,
    cv = 5
)

## X should be standardised as log regression is sensitive to scale of X
X_all_st = StandardScaler().fit_transform(X_all)

## To overcome the issue of overfitting to the majority class, I randomly d
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.pipeline import Pipeline

## Saving the model
# pkl_filename = "LR_cv_results_" + "CW_None" + ".pkl"
pkl_filename = "LR_model.pkl"
if True:
    LR_cv_results = LR_cv.fit(X=X_all_st, y=y_all_nu)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(LR_cv_results, file)
else:
    with open(pkl_filename, 'rb') as file:
        LR_cv_results = pickle.load(file)

print(LR_cv_results.best_params_)

## Fit the model to training dataset and Check the results on the
stder = StandardScaler()
X_train_st = stder.fit_transform(X_train)
X_test_st = stder.transform(X_test)

fit_traindata = LR_cv_results.best_estimator_.fit(X_train_st, y_train)

plt.figure()
plot_confusion_matrix(fit_traindata, X_test_st, y_test)
plt.title("LR")
plt.savefig("./Results/" + pkl_filename[:-4] + ".png")

# labels = ([y_all.unique().tolist(), y_all.unique().tolist()])
