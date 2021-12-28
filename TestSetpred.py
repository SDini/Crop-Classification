import PIL
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import MinMaxScaler

## Set the working directory
try:
    os.chdir("/Users/sdini/PycharmProjects/CropClass/")
except:
    os.chdir("/mnt/CropClass/")

def load_image(path):
    return np.array(PIL.Image.open(path))[:, :]

bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

data_dims = (np.size(load_image("Images/B01.tif")), len(bands))
data_test = pd.DataFrame(np.zeros(data_dims), columns=bands)
for i in bands:
    data_test.loc[:, i] = pd.Series(load_image("Images/" + i + ".tif").flatten())
    # WARNING: bear in mind, for unflattening, flatten() concats rows!!!

## Scaling the pixel values to be on the same range as the train dataset
data_train = pd.read_csv("./Images/pixels.csv")
data_train = data_train[data_train.cloud_prob <= 2]

data_test_sc = data_test.copy()

minmax_sc = MinMaxScaler()
data_test_sc = pd.DataFrame(minmax_sc.fit_transform(data_test_sc))


## Here I am assuming that the ranges of train and test datasets match
# for i in data_test.columns:
#     minmax_sc = MinMaxScaler(feature_range = (min(data_train.loc[:, i]), max(data_train.loc[:, i])))
#     data_test_sc.loc[:, i] = minmax_sc.fit_transform(np.array(data_test.loc[:, i]).reshape(-1, 1))
#

plt.boxplot(data_test_sc)
plt.xticks(np.arange(np.shape(data_train.iloc[:, :-3])[1])+1, data_test_sc.columns.tolist())

with open("RF_strcv_fine_results.pkl", 'rb') as file:
    RF_cv_results = pickle.load(file)

# Importing masked data points
img_mask = load_image("Images/mask.png")
print(img_mask.shape)
# Creating a matrix whose elements are 0 or >0 if one of R, G or B are >0
index_mask = img_mask[:,:,0] + img_mask[:,:,1] + img_mask[:,:,2]
index_mask = index_mask.flatten()
# index_mask = pd.Series(load_image("Images/mask.png").flatten())
data_test_sc_masked = data_test_sc.loc[index_mask>0, :]
data_test_sc_masked.describe()

plt.boxplot(data_test_sc_masked)

pred_testdata = RF_cv_results.predict(data_test_sc_masked)

# create a blank image
pred_img_flat = np.full((np.shape(data_test_sc)[0], ), 3)
pred_img_flat[index_mask>0] = pred_testdata

im_size = np.shape(load_image("Images/B01.tif"))
pred_testdata_img = np.reshape(pred_img_flat, im_size)


from matplotlib import colors
# 0: Cotton; 1: Other; 2: Sorghum; 3: masked
cmap = colors.ListedColormap(['white', 'purple', 'red', 'black'], 'indexed')
plt.figure(figsize=(8,8))
plt.imshow(pred_testdata_img, cmap=cmap, interpolation='none')
plt.savefig("./Results/Predtest.png")