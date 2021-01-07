#!/usr/bin/env python
# coding: utf-8

# # TENSORFLOW BEGINNING

# In[1]:


import tensorflow as tf


# In[2]:


tf


# In[3]:


tf.__version__


# In[4]:


mnist = tf.keras.datasets.mnist


# In[5]:


mnist


# In[6]:


tf.keras.datasets


# In[7]:


(x_train, y_train),(x_test,y_test) = mnist.load_data()


# In[8]:


x_train


# In[9]:


y_train


# In[10]:


x_test


# In[11]:


import matplotlib.pyplot as plt


# In[12]:


plt.imshow(x_train[0])


# In[13]:


x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)


# In[14]:


model = tf.keras.models.Sequential()


# In[15]:


model.add(tf.keras.layers.Flatten())


# In[16]:


model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))


# In[17]:


model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))


# In[18]:


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[19]:


model.fit(x_train,y_train,epochs=3)


# In[20]:


loss, accuracy = model.evaluate(x_test,y_test)
print(loss,accuracy)


# In[21]:


model.save('first_model')


# In[23]:


new_model = tf.keras.models.load_model('first_model')


# In[24]:


predictions = new_model.predict([x_test])


# In[25]:


print(predictions)


# In[27]:


import numpy as np


# In[28]:


np.argmax(predictions[0])


# In[30]:


plt.imshow(x_test[0])


# In[32]:


np.argmax(predictions[1])


# In[33]:


plt.imshow(x_test[1])


# In[34]:


np.argmax(predictions[10])


# In[35]:


plt.imshow(x_test[10])


# In[36]:


np.argmax(predictions[-1])


# In[37]:


plt.imshow(x_test[-1])


# In[38]:


np.argmax(predictions[-100])


# In[39]:


plt.imshow(x_test[-100])


# In[ ]:




