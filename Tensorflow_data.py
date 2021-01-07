#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
import os


# In[2]:


import cv2


# In[3]:


DATADIR = "/home/hrishikesh/tensorflow/PetImages"


# In[4]:


DATADIR


# In[20]:


CATAGORIES = ["Dog","Cat"]


# In[22]:


for category in CATAGORIES:
    path = os.path.join(DATADIR, category) 
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array,cmap="gray")
        plt.show()
        break 
    break


# In[10]:


print(img_array.shape)


# In[15]:


IMG_SIZE = 50

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array,cmap="gray")


# In[16]:


training_data = []


# In[28]:


CATAGORIES = ["Dog","Cat"]

def create_training_data():
    for category in CATAGORIES:
        path = os.path.join(DATADIR, category) 
        class_num = CATAGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array,class_num])
            except Exception as e:
                pass
            
create_training_data()


# In[29]:


print(len(training_data))


# In[30]:


import random


# In[31]:


random.shuffle(training_data)


# In[34]:


for sample in training_data:
    print(sample[1])


# In[35]:


X = []
y = []


# In[36]:


for features,label in training_data:
    X.append(features)
    y.append(label)


# In[37]:


X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


# In[ ]:




