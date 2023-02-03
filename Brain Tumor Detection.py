#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[4]:


import os
#os.getcwd()
path = os.listdir('../brain-tumor-detection-master/brain_tumor/Training/')
classes = {'no_tumor':0,'pituitary_tumor': 1}


# In[5]:


import cv2
X=[]
Y=[]
for cls in classes:
    pth='../brain-tumor-detection-master/brain_tumor/Training/'+cls
    for j in os.listdir(pth):
        img = cv2.imread(pth+'/'+j,0)
        img = cv2.resize(img,(200,200))
        X.append(img)
        Y.append(classes[cls])


# In[6]:


get_ipython().system('pip install opencv-python')


# In[7]:


X = np.array(X)
Y = np.array(Y)

X_updated = X.reshape(len(X), -1)


# In[8]:


X = np.array(X)
Y = np.array(Y)

X_updated = X.reshape(len(X), -1)


# In[9]:


np.unique(Y)


# In[10]:


pd.Series(Y).value_counts()


# In[11]:


import pandas as pd


# In[12]:


X.shape, X_updated.shape


# In[13]:


plt.imshow(X[0], cmap = 'gray')


# In[14]:


xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10, test_size=.20)


# In[15]:


xtrain.shape, xtest.shape


# In[16]:


print(xtrain.max(),xtrain.min())
print(xtest.max(), xtest.min())
xtrain = xtrain/255
xtest = xtest/255
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())


# In[17]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[18]:


import warnings
warnings.filterwarnings('ignore')

lg = LogisticRegression(C=0.1)
lg.fit(xtrain, ytrain)


# In[19]:


sv = SVC()
sv.fit(xtrain, ytrain)


# In[20]:


print("Training Score:", lg.score(xtrain, ytrain))
print("Testing Score:", lg.score(xtest, ytest))


# In[21]:


print ("Training Score:", sv.score(xtrain, ytrain))
print("Testing Score:", sv.score(xtest, ytest))


# In[22]:


pred = sv.predict(xtest)


# In[23]:


misclassified = np.where(ytest!=pred)
misclassified


# In[24]:


print("Total misclassified Samples: ", len(misclassified[0]))
print(pred[36], ytest[36])


# In[25]:


dec = {0: 'No Tumor', 1: 'Positive Tumor'}


# In[26]:


plt.figure(figsize = (12,8))
p = os.listdir('../brain-tumor-detection-master/brain_tumor/Testing/')
C=1
for i in os.listdir('../brain-tumor-detection-master/brain_tumor/Testing/no_tumor/')[:9]:
    plt.subplot(3,3,C)

    img = cv2.imread('../brain-tumor-detection-master/brain_tumor/Testing/no_tumor/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    C+=1


# In[76]:


plt.figure(figsize = (12,8))
p = os.listdir('../brain-tumor-detection-master/brain_tumor/Testing/')
C=1
for i in os.listdir('../brain-tumor-detection-master/brain_tumor/Testing/pituitary_tumor/')[:16]:
    plt.subplot(4,4,C)

    img = cv2.imread('../brain-tumor-detection-master/brain_tumor/Testing/pituitary_tumor/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    C+=1


# In[ ]:




