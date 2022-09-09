#!/usr/bin/env python
# coding: utf-8

# 
# # Face Identification Model

# In[1]:


TrainingImagePath="F:/datascientist/Billionaires Classification Model/using _CNN/Crop/Train"
validationImagePath = "F:/datascientist/Billionaires Classification Model/using _CNN/Crop/Test"


# # By using top Billionaires Images dataset we have created Face Identification Model

# In[2]:


# Running the data type of the array


# In[3]:


import cv2
data = cv2.imread('F:/datascientist/Billionaires Classification Model/Dataset/Mukesh Ambani/Corporate Governance.jpg')
data


# In[4]:


from keras.preprocessing.image import ImageDataGenerator


# In[5]:


train_datagen = ImageDataGenerator(rescale=1./225)
 
test_datagen = ImageDataGenerator(rescale=1./225)
 


# In[ ]:





# In[6]:


training_set = train_datagen.flow_from_directory(TrainingImagePath,
                                                 target_size=(64,64),
                                                 batch_size=10,
                                                 class_mode='categorical')


# In[7]:


validation_set = test_datagen.flow_from_directory(validationImagePath,
                                                  target_size=(64,64),
                                                  batch_size=10,
                                                  class_mode='categorical')


# In[8]:


from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense


# In[9]:


classifier= Sequential()


# In[10]:



classifier.add(Convolution2D
               
               (32, kernel_size=(2, 2),
                             input_shape=(64,64,3), activation='relu')) 


# In[11]:


classifier.add(MaxPool2D(pool_size=(2,2)))


# In[12]:


classifier.add(Convolution2D(64, kernel_size=(2, 2), activation='relu'))


# In[13]:


classifier.add(MaxPool2D(pool_size=(2,2)))


# In[14]:


classifier.add(Convolution2D(64, kernel_size=(2, 2), activation='relu')) 


# In[15]:


classifier.add(Flatten()) 


# In[16]:



classifier.add(Dense(64, activation='relu'))
classifier.add(Dense(10, activation='softmax'))
classifier.summary()


# In[17]:


classifier.compile(loss='categorical_crossentropy', 
                   optimizer = 'rmsprop', metrics=["accuracy"])


# In[18]:


history = classifier.fit_generator(training_set, epochs=16, 
                         validation_data=validation_set)


# In[19]:


import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


# In[20]:


epochs = range(1, len(acc) + 1)


# In[21]:


plt.plot(epochs, acc, 'bo',label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()


# In[22]:


plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# # Model 1 : Maximum accuracy of 90% is been attained on 13th Epoch

# # Creating a another Model by decreasing the learning rate and momentum

# In[51]:


TrainingImagePath="F:/datascientist/Billionaires Classification Model/using _CNN/Crop/Train"
validationImagePath = "F:/datascientist/Billionaires Classification Model/using _CNN/Crop/Test"


# In[52]:


import cv2
data = cv2.imread('F:/datascientist/Billionaires Classification Model/Dataset/Mukesh Ambani/Corporate Governance.jpg')
data


# In[53]:


from keras.preprocessing.image import ImageDataGenerator


# In[54]:


train_datagen = ImageDataGenerator(rescale=1./225)
 
test_datagen = ImageDataGenerator(rescale=1./225)


# In[55]:


training_set = train_datagen.flow_from_directory(TrainingImagePath,
                                                 target_size=(64,64),
                                                 batch_size=10,
                                                 class_mode='categorical')


# In[56]:


validation_set = test_datagen.flow_from_directory(validationImagePath,
                                                  target_size=(64,64),
                                                  batch_size=10,
                                                  class_mode='categorical')


# # Building Model

# In[57]:


from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense


# In[58]:


classifier= Sequential()
classifier.add(Convolution2D
               
               (32, kernel_size=(2, 2),
                             input_shape=(64,64,3), activation='relu')) 


# In[59]:


classifier.add(MaxPool2D(pool_size=(2,2)))


# In[61]:


classifier.add(Convolution2D(64, kernel_size=(2, 2), activation='relu'))


# In[62]:


classifier.add(MaxPool2D(pool_size=(2,2)))


# In[63]:


classifier.add(Convolution2D(64, kernel_size=(2, 2), activation='relu')) 


# In[64]:


classifier.add(Flatten()) 
classifier.add(Dense(64, activation='relu'))
classifier.add(Dense(10, activation='softmax'))
classifier.summary()


# In[65]:



classifier.compile(loss='categorical_crossentropy', 
                   optimizer="adam", metrics=["accuracy"])


# In[66]:


history = classifier.fit_generator(training_set, epochs=40, 
                         validation_data=validation_set)


# ##### Model 2 : Maximum accuracy of 80.39% is been attained

# # Model 2 has attained less accuracy then model1

# In[ ]:




