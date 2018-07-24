
# coding: utf-8

# In[75]:


""" Random Forest.
Implement Random Forest algorithm with TensorFlow, and apply it to predict 
SSD life Recycle. This example is using the blackblaze data of 
HDDas training samples（https://www.backblaze.com/b2/hard-drive-test-data.html）.
Author: Xiaobaicai Yan
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources

# Ignore all GPUs, tf random forest does not benefit from it.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Ignore all GPUs, tf random forest does not benefit from it.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pandas as pd


# In[76]:


# Import HDD data
features = pd.read_csv("data",sep = ',',encoding = 'utf-8')
features.head(5)

import numpy as np
labels = np.array(features['failure'])
features = features.drop('failure',axis = 1)
feature_list = list(features.columns)
features = np.array(features)
#train, test = train_test_split(mydata,test_size = 0.4)
#print(train)


# In[77]:


from sklearn.cross_validation import train_test_split
train_features,test_features,train_labels,test_labels= train_test_split(features,labels,test_size = 0.4)
print('Training Features Shape:',train_features.shape)
print('Training Labels Shape:',train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:',test_labels.shape)


# In[78]:


def predict_error(predictions):
    
    for index in range(len(predictions)):
        if(predictions[index] >= 0.5):
            predictions[index] = 1
        else:
            predictions[index] = 0


# In[79]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
rf = RandomForestRegressor(n_estimators = 100)
rf.fit(train_features,train_labels)
predictions = rf.predict(test_features)
predict_error(predictions)
print(predictions)


# In[74]:


model = GaussianNB()
model.fit(train_features,train_labels)
predict = model.predict(test_features)
print('Result:',predict)


# In[96]:


def count(predict,label):
    false_pos = 0
    false_neg = 0
    true_pos = 0
    true_neg = 0
    for item in label:
        if(item == 0):
            true_pos += 1
        else:
            true_neg += 1
    #print(true_pos)
    #print(true_neg)
    for index in range(len(predict)):
        if(predict[index] != label[index]):
            if(label[index] == 0):
                false_pos += 1
            else:
                false_neg += 1
    
    FPR = false_pos / (false_pos + true_neg)
    FNR = false_neg / (false_neg + true_pos)
    
    #print("false_pos:",false_pos,"false_neg:",false_neg,"FPR:",FPR,"FNR:",FNR)
    
    return FPR,FNR


# In[108]:


batch_size = 50


# In[109]:


x = np.zeros(shape = (batch_size,1))
y = np.zeros(shape = (batch_size,1))


# In[110]:


for i in range(batch_size):
    train_features,test_features,train_labels,test_labels= train_test_split(features,labels,test_size = 0.4)
    rf = RandomForestRegressor(n_estimators = 100)
    rf.fit(train_features,train_labels)
    predictions = rf.predict(test_features)
    predict_error(predictions)
    FPR,FNR = count(predictions,test_labels)
    x[i] = [FPR]
    y[i] = [FNR]


# In[111]:


import matplotlib.pyplot as plt
p1 = plt.scatter(x,y)


# In[ ]:


# Parameters
num_steps = 500 # Total steps to train
batch_size = 1024 # The number of samples per batch
num_classes = 2 # The 10 digits
num_features = 11 # 11 smart features
num_trees = 10
max_nodes = 1000

# Input and Target data
X = tf.placeholder(tf.float32, shape=[None, num_features])
# For random forest, labels must be integers (the class id)
Y = tf.placeholder(tf.int32, shape=[None])

# Random Forest Parameters
hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                      num_features=num_features,
                                      num_trees=num_trees,
                                      max_nodes=max_nodes).fill()

# Build the Random Forest
forest_graph = tensor_forest.RandomForestGraphs(hparams)
# Get training graph and loss
train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)

# Measure the accuracy
infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the variables (i.e. assign their default value) and forest resources
init_vars = tf.group(tf.global_variables_initializer(),
    resources.initialize_resources(resources.shared_resources()))

# Start TensorFlow session
sess = tf.Session()

# Run the initializer
sess.run(init_vars)

# Training
for i in range(1, num_steps + 1):
    # Prepare Data
    # Get the next batch of MNIST data (only images are needed, not labels)
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    _, l = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
    if i % 50 == 0 or i == 1:
        acc = sess.run(accuracy_op, feed_dict={X: batch_x, Y: batch_y})
        print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))

# Test Model
test_x, test_y = mnist.test.images, mnist.test.labels
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))

