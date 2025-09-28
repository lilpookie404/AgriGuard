#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[2]:


import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ## Data Preprocessing

# ### Training Image Preprocessing

# In[3]:


print(tf.config.list_physical_devices('GPU'))


# In[7]:


import os
print(os.getcwd())  # shows your current working directory


# In[8]:


training_set = tf.keras.utils.image_dataset_from_directory(
    '/Users/vaishnaviawadhiya/Downloads/Plant_Disease_Dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)


# ### Validation Image Preprocessing

# In[12]:


validation_set = tf.keras.utils.image_dataset_from_directory(
    '/Users/vaishnaviawadhiya/Downloads/Plant_Disease_Dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)


# In[13]:


for x,y in training_set:
    print(x,x.shape)
    print(y,y.shape)
    break


# ### To avoid Overshooting
# 1. Choose small learning rate default 0.001 we are taking 0.0001
# 2. There may be chance of Underfitting, so increase number of neuron
# 3. Add more Convolution layer to extract more feature from images there may be possibilty that model unable to capture relevant feature or model is confusing due to lack of feature so feed with more feature

# ## Building Model

# In[14]:


from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout
from tensorflow.keras.models import Sequential


# In[15]:


model = Sequential()


# ## Building Convolution Layer

# In[16]:


model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=[128,128,3]))
model.add(Conv2D(filters=32,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))


# In[17]:


model.add(Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=64,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))


# In[18]:


model.add(Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=128,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))


# In[19]:


model.add(Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=256,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))


# In[20]:


model.add(Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=512,kernel_size=3,activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))


# In[21]:


model.add(Dropout(0.25)) # To avoid Overfitting


# In[22]:


model.add(Flatten())


# In[23]:


model.add(Dense(units=1500,activation='relu'))


# In[24]:


model.add(Dropout(0.4))


# In[25]:


#Output Layer
model.add(Dense(units=38,activation='softmax'))


# ### Compiling Model

# In[27]:


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# In[28]:


model.summary()


# ### Model Training

# In[29]:


training_history = model.fit(x=training_set,validation_data=validation_set,epochs=10)


# In[ ]:





# ## Model Evaluation

# In[30]:


#Model Evaluation on Training set
train_loss,train_acc = model.evaluate(training_set)


# In[31]:


print(train_loss,train_acc)


# In[32]:


#Model on Validation set
val_loss,val_acc = model.evaluate(validation_set)


# In[33]:


print(val_loss,val_acc)


# In[ ]:





# ### Saving Model

# In[36]:


model.save("trained_model.keras")


# In[37]:


training_history.history


# In[55]:


#Recording History in json
import json
with open("training_hist.json","w") as f:
    json.dump(training_history.history,f)


# In[ ]:





# In[39]:


training_history.history['val_accuracy']


# ### Accuracy Visualization

# In[40]:


epochs = [i for i in range(1,11)]
plt.plot(epochs,training_history.history['accuracy'],color='red',label='Training Accuracy')
plt.plot(epochs,training_history.history['val_accuracy'],color='blue',label='Validation Accuracy')
plt.xlabel("No. of Epochs")
plt.ylabel("Accuracy Result")
plt.title("Visualization of Accuracy Result")
plt.legend()
plt.show()


# In[ ]:





# ### Some other metrics for model evaluation

# In[41]:


class_name = validation_set.class_names
class_name


# In[ ]:





# In[43]:


test_set = tf.keras.utils.image_dataset_from_directory(
    '/Users/vaishnaviawadhiya/Downloads/Plant_Disease_Dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=False,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)


# In[44]:


y_pred = model.predict(test_set)
y_pred,y_pred.shape


# In[45]:


predicted_categories = tf.argmax(y_pred,axis=1)


# In[46]:


predicted_categories


# In[47]:


true_categories = tf.concat([y for x,y in test_set],axis=0)
true_categories


# In[48]:


Y_true = tf.argmax(true_categories,axis=1)
Y_true


# In[ ]:





# ![image.png](attachment:f464cbcc-5d6b-4f32-835c-9aabe0f9c5d4.png)

# In[ ]:





# In[49]:


from sklearn.metrics import classification_report,confusion_matrix


# In[50]:


print(classification_report(Y_true,predicted_categories,target_names=class_name))


# In[51]:


cm = confusion_matrix(Y_true,predicted_categories)
cm


# In[ ]:





# ### Confusion Matrix Visualization

# In[54]:


plt.figure(figsize=(40,40))
sns.heatmap(cm,annot=True,annot_kws={'size':10})
plt.xlabel("Predicted Class",fontsize=20)
plt.ylabel("Actual Class",fontsize=20)
plt.title("Plant Disease Prediction Confusion Matrix",fontsize=25)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




