from PCANet import *
import sys
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score, make_scorer,recall_score,accuracy_score,classification_report
from sklearn import svm

import h5py
import matplotlib.pyplot as plt
import numpy as np

import time

import pandas as pd

hf = h5py.File('../../../data.h5', 'r')
images = hf.get('images')
labels = hf.get('labels')

print(images.shape)
print(labels.shape)
print(labels[3257])
#plt.imshow(images[3257], cmap ='gray' )
#plt.show()
#print(labels)

from sklearn.model_selection import cross_val_score,cross_validate,StratifiedKFold,RepeatedStratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer,MinMaxScaler,LabelEncoder

mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)
#print(labels)
#print(labels.shape)
labels = LabelEncoder().fit_transform([''.join(str(l)) for l in labels])
print(labels)
print(labels.shape)


accuracy_list_full = []
recall_list_full = []
precision_list_full = []

def myMethod(L1value, dim_reduction):
  accuracy_list = []
  recall_list = []
  precision_list = []
  skf = StratifiedKFold(n_splits = 10, shuffle=True, random_state=None)
  for train_index, test_index in skf.split(images,labels):
    X_train, X_test = images[train_index], images[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    net = PCANet(k1=5, k2=5, L1=L1value, L2=8, block_size=50, overlapping_radio=0, dim_reduction=dim_reduction)
    net.fit(X_train, y_train)
    prediction = net.predict(X_test)
    print(accuracy_score(y_test, prediction))
    
    y_pred = prediction
    y_true = y_test

      # concatena
      ##y_pred_full = np.concatenate((y_pred_full, y_pred))
      ##y_true_full = np.concatenate((y_true_full, y_true))
      ##print(y_pred)
      ##print(y_true)
    a = accuracy_score(y_true, y_pred)
    r = recall_score(y_true, y_pred, average='macro')
    p = precision_score(y_true, y_pred, average='macro')
    accuracy_list.append(a)
    recall_list.append(r)
    precision_list.append(p)

  accuracy_list_full.append(np.mean(accuracy_list))
  recall_list_full.append(np.mean(recall_list))
  precision_list_full.append(np.mean(precision_list))
  return (np.mean(accuracy_list),np.mean(recall_list),np.mean(precision_list))


dim_reduction = 75
# start timer
start = time.perf_counter()
for count in range(30):
  print(count)
  result = myMethod(8, dim_reduction)
  print("Accuracy:",result[0], "Recall:",result[1], "Precision:",result[2])


print("ARRAY FINAL:")
print("DIMENSION REDUCTION")
print("accuracy: ", accuracy_list_full)
print("recall: ", recall_list_full)
print("precision: ", precision_list_full)

print("FINAL:")
print("accuracy: ", np.mean(accuracy_list_full))
print("recall: ", np.mean(recall_list_full))
print("precision: ", np.mean(precision_list_full))

print("Desvio padrão:")
print("accuracy: ", np.std(accuracy_list_full))
print("recall: ", np.std(recall_list_full))
print("precision: ", np.std(precision_list_full))

end = time.perf_counter()
print("Ellapsed time (30 repetitions): "+ str(end - start))

## SALVANDO EM CSV

res_final = [dim_reduction, end - start, np.mean(accuracy_list_full), np.mean(recall_list_full), np.mean(precision_list_full), np.std(accuracy_list_full), np.std(recall_list_full), np.std(precision_list_full)]

df = pd.DataFrame([res_final])

df.to_csv('data_3.csv',header=['reduction_to', 'time', 'mn_acc', 'mn_rec', 'mn_pre', 'std_acc', 'std_rec', 'std_pre'])
