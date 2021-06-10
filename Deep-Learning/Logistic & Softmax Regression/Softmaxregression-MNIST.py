# -*- coding: utf-8 -*-
"""
Created on Sun Oct 07 10:35:29 2020

@author: Hossein
"""
from keras.datasets import mnist
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

(X_train,y_train),(X_test, y_test) = mnist.load_data()
classes = ['0', '1', '2','3', '4','5', '6', '7', '8', '9']
nclasses = len(classes)

#normalizing and flattening the data
image_vector_size = 28*28
X_train_final = X_train.reshape(X_train.shape[0], image_vector_size)/255.0
X_test_final = X_test.reshape(X_test.shape[0], image_vector_size)/255.0

#**************Training the model**************************
clf = LogisticRegression(penalty='none', 
                         tol=0.05, solver= 'saga',
                         multi_class='multinomial').fit(X_train_final, y_train)

#***********************************Important weights plot***********************
#clf.coef_.shape
#clf.coef_
#np.min(clf.coef_)
scale = np.max(np.abs(clf.coef_))
p = plt.figure(figsize=(25, 2.5));

for i in range(nclasses):
    p = plt.subplot(1, nclasses, i + 1)
    p = plt.imshow(clf.coef_[i].reshape(28, 28),
                  cmap=plt.cm.RdBu, vmin=-scale, vmax=scale);
    p = plt.axis('off')
    p = plt.title('Class %i' % i);
plt.show()    
    
#sample number
print("Choosing a sample")
sample_idx = 1235
#plotting image
plt.imshow(X_test_final[sample_idx].reshape(28,28), cmap='gray');
plt.title('Label: %s\n' % y_test[sample_idx]);
plt.axis('off');
plt.show()    
    
#computing the output of each neuron for our choosen sample i.e sample_idx = 1235   
z = [ clf.intercept_[k] + np.dot(clf.coef_[k], X_test_final[sample_idx]) for k in range(10) ]

#sending the output vector to softmax to get the probabilities
def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=0, keepdims=True)

probs=softmax(z)


#plotting the result related to chosen sample
sns.barplot(np.arange(0,10), probs);
plt.ylabel("Probability");
plt.xlabel("Class");
plt.show()

#computing the accuracy of the model
y_pred = clf.predict(X_test_final)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy= {0:.2f}%".format(accuracy*100))


