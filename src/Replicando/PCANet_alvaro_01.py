from PCANet import *
from data_loader import *
import sys
from sklearn.metrics import accuracy_score
from sklearn import svm

import numpy as np
import pandas as pd

# Load images
train_images, train_labels, test_images, test_labels = load_mnist('data/MNIST')
print("Loading dataset...")
print(train_images.shape)
print(test_images.shape)

# Select some images
print("Selecting dataset...")
train_images = train_images[:20000]
train_labels = train_labels[:20000]
test_images = test_images[:10000]
test_labels = test_labels[:10000]
print(train_images.shape)
print(test_images.shape)

L1_all = [2, 4, 6, 8, 10, 12]
#L1_all = [2]
n_repetitions = 1

#df = pd.DataFrame()  

data = []
for L1_value in L1_all:
	print("VALOR ATUAL DE L1 ", L1_value)
	line_acc = []
	for rep in range(n_repetitions):
		print("Repetição Atual: ", rep)
		net = PCANet(k1=7, k2=7, L1=L1_value, L2=1, block_size=7, overlapping_radio=0.5)
		net.fit(train_images, train_labels)
		prediction = net.predict(test_images)
		acc = accuracy_score(test_labels, prediction)
		print(acc)
		line_acc.append(acc)

	mean = np.mean(line_acc)
	std = np.std(line_acc)
	line = [L1_value, mean, std]
	data.append(line)


df = pd.DataFrame(data, columns=['L1','mean','std'])
df.to_csv("MNIST_L1_{}reps_20k_10k.csv".format(n_repetitions),index=False)