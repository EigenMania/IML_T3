#################################
#            IML_T3             #         
#################################
#
# File Name: main.py
#
# Course: 252-0220-00L Introduction to Machine Learning
#
# Authors: Adrian Esser (aesser@student.ethz.ch)
#          Abdelrahman-Shalaby (shalabya@student.ethz)

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
import sys

import matplotlib.pyplot as plt

#################################
# 1) Import and preprocess data #       
#################################
train_data = pd.read_hdf("train.h5", "train")
test_data = pd.read_hdf("test.h5", "test")

# Load
y_train = np.array(train_data)[:,0]
X_train = np.array(train_data)[:,1:]
X_test = np.array(test_data)
N = np.shape(X_train)[1] # number of features in input vector

# Example
results = np.genfromtxt('sample.csv', delimiter=',', skip_header=1)

# Convert classes to one hot encodings
#print(y_train) # before
lb = preprocessing.LabelBinarizer()
lb.fit(y_train)
y_train = lb.transform(y_train)
L = np.shape(y_train)[1] # the number of labels
#print(y_train) # after

# Mean remove the training and test data
X_train = (X_train - np.mean(X_train, axis=0))/(np.std(X_train, axis=0))
X_test = (X_test - np.mean(X_test, axis=0))/(np.std(X_test, axis=0))

# NOTE: I found that we were not overfitting, so I figured it's 
#       better to include all of the data. 
# Split training data further into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=42)

# NOTE: uncomment if you want to see plot of histogram
#plt.hist(X_test[:,50])
#plt.show()

#################################
# 2) Set up Network Structure   #       
#################################
RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

alpha = 0.1  # learning rate
inputs_ = tf.placeholder(tf.float32, (None, N))
y_true = tf.placeholder(tf.float32, (None, L))
do_dropout = True

def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    bias = tf.random_normal((1,shape[1]), stddev=0.1)
    return tf.Variable(weights), tf.Variable(bias)

# First hidden layer
nhu1 = 200
w1, b1 = init_weights((N, nhu1))
o1 = tf.nn.relu(tf.matmul(inputs_, w1) + b1)

l1 = tf.layers.dense(inputs=inputs_,units=500,activation=tf.nn.relu)
l1_o = tf.layers.dropout(l1,rate=0.4,training=do_dropout) 

nhu2 = 200
w2, b2 = init_weights((nhu1, nhu2))
o2 = tf.nn.relu(tf.matmul(o1, w2) + b2)

l2 = tf.layers.dense(inputs=l1_o,units=500,activation=tf.nn.relu)
l2_o = tf.layers.dropout(l2,rate=0.4,training=do_dropout) 

nhu3 = 200
w3, b3 = init_weights((nhu2, nhu3))
o3 = tf.nn.relu(tf.matmul(o2, w3) + b3)

l3 = tf.layers.dense(inputs=l2_o,units=500,activation=tf.nn.relu)
l3_o = tf.layers.dropout(l3,rate=0.4,training=do_dropout) 


#nhu4 = 200
#w4, b4 = init_weights((nhu3, nhu4))
#o4 = tf.nn.relu(tf.matmul(o3, w4) + b4)


# Second hidden layer
nhuO = L
wO, bO = init_weights((nhu3, nhuO))
#out = tf.nn.relu(tf.matmul(o3, wO) + bO)

out = tf.layers.dense(inputs=l3_o,units=L,activation=tf.nn.relu)

# Softmax function
y_hat = tf.nn.softmax(out)
pred_y = tf.argmax(y_hat, 1)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_hat))
opt = tf.train.GradientDescentOptimizer(alpha).minimize(cost)
acc = tf.metrics.accuracy(labels=tf.argmax(y_true,1), predictions=pred_y)

#################################
# 3) Train Network!             #       
#################################
sess = tf.Session()
init_g = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
sess.run(init_g)
sess.run(init_l)

k = 200 # number of batches
kf = KFold(n_splits=k, shuffle=True)

epochs = 20
acc_val_vect = []
acc_train_vect = []

for e in range(epochs):
    print("Epoch {}/{}".format(e+1, epochs))
    # First evaluate performance on validation set
    acc_val = sess.run(acc, feed_dict={inputs_: X_val, y_true: y_val})[0]
    acc_train = sess.run(acc, feed_dict={inputs_: X_train, y_true: y_train})[0]

    acc_val_vect.append(acc_val)
    acc_train_vect.append(acc_train)

    # Split data into batches and train 
    for _, t_idx in kf.split(X_train):
        X_b = X_train[t_idx,:]
        y_b = y_train[t_idx,:]

        _ = sess.run(opt, feed_dict={inputs_:X_b, y_true:y_b})

    print("Training Accuracy: {}".format(acc_train)) 
    print("Validation Accuracy: {}".format(acc_val))


# Final performance on both sets
acc_val = sess.run(acc, feed_dict={inputs_: X_val, y_true: y_val})[0]
acc_train = sess.run(acc, feed_dict={inputs_: X_train, y_true: y_train})[0]

#acc_val_vect.append(acc_val)
acc_train_vect.append(acc_train)

#y_train_pred = sess.run(pred_y, feed_dict={inputs:X_train})
#print(y_train_pred)
#print(y_train)

# NOTE: Option to plot results
e_vect = range(epochs+1)
plt.plot(e_vect, acc_train_vect, 'b')
plt.plot(e_vect, acc_val_vect, 'm')
plt.show()

#################################
# 4) Obtain Predictions!        #       
#################################
y_test = sess.run(pred_y, feed_dict={inputs_:X_test})

results[:,1] = y_test
np.savetxt('results.csv', results, fmt='%d,%d', newline='\n', comments='')






