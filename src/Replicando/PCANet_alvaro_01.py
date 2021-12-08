from PCANet import *
from data_loader import *
import sys
from sklearn.metrics import accuracy_score
from sklearn import svm

# Load images
train_images, train_labels, test_images, test_labels = load_mnist('data/MNIST')
print("Loading dataset...")
print(train_images.shape)
print(test_images.shape)

# Select some images
print("Selecting dataset...")
train_images = train_images[:5000]
train_labels = train_labels[:5000]
test_images = test_images[:5000]
test_labels = test_labels[:5000]
print(train_images.shape)
print(test_images.shape)

net = PCANet(k1=7, k2=7, L1=24, L2=8, block_size=7, overlapping_radio=0.5)
net.fit(train_images, train_labels)
prediction = net.predict(test_images)
print(accuracy_score(test_labels, prediction))