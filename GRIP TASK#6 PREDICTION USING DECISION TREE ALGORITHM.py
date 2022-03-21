#!/usr/bin/env python
# coding: utf-8

# # THE SPARKS FOUNDATION

# # DATA SCIENCE AND BUSINESS ANALYTICS INTERNSHIP

# # #TASK6 PREDICTION USING DECISION TREE ALGORITHM

# # AUTHOR: SHREYA GHOSH

# OBJECTIVE:CREATING A DECISION TREE CLASSIFIER AND PREDICTING THE RIGHT CLASS FOR NEWLY FEEDED DATA

# THE STEPS WE WILL BE FOLLOWING FOR THIS TASK ARE AS FOLLOWS:
STEP 1-IMPORTING THE DATASET FROM EXCEL
STEP 2-PREPARING THE DATASET
STEP 3-TRAINING THE MODEL
STEP 4-TESTING THE MODEL/MAKING PREDICTIONS
STEP 5-VISUALIZATION OF THE MODEL
STEP 6-EVALUATING THE MODEL/DIAGONISTIS
# # STEP 1 IMPORTING THE DATASET FROM EXCEL

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib as plt


# In[3]:


#Importing the dataset from csv file 
df=pd.read_csv("C:/Users/Shreya/OneDrive/Desktop/GRIP FILE/Iris.csv")
df


# In[20]:


#To get the name of the unique data points in the Species column
Species=df.Species.unique()
Species


# In[5]:


#To get the number of rows and columns of the dataset
df.shape


# In[6]:


#Let us now drop the unnecessary column namely Id
df=df.drop(["Id"],axis="columns")
df.head()


# In[8]:


#To check for any missing or null values in the dataset
df.isna().sum()


# # STEP 2 PREPARING THE DATASET

# We will first divide the data set into independent (X variable) and dependent (Y variable) variable.

# In[9]:


X=df.iloc[:,0:4].values
X


# In[10]:


Y=df.iloc[:,4].values
Y


# Now that we have defined our X and Y variable we will split the data set into testing and training data

# In[11]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)


# In[12]:


X_train.shape,X_test.shape,Y_train.shape,Y_test.shape


# # STEP 3 TRAINING THE DATASET

# In this step we will be training our X_train and Y_train dataset

# In[13]:


model=DecisionTreeClassifier(criterion="entropy")


# In[14]:


train=model.fit(X_train,Y_train)
train


# # STEP 4 TESTING AND PREDICTINGTHE MODEL

# In[15]:


pred=model.predict(X_test)
pred


# Let us now predict the Species with Sepal Length 4.6cm,Sepal Width 3.1cm,Petal Length 1.5cm, Petal Widtht 0.2 cm

# In[16]:


a=model.predict([[4.6,3.1,1.5,0.2]])
print("The species of the flower is",a)


# # STEP 5 VISUALIZATION OF THE MODEL

# In[17]:


#Textual Representation of the Desicion Tree
Text_rep=tree.export_text(model)
Text_rep


# In[23]:


plt.rcParams["figure.figsize"]=[20,20]
tree.plot_tree(model,feature_names=df.columns,class_names=Species,filled=True)


# # STEP 6 EVALUATING THE MODEL/DIAGONISTIS

# In this step we will evaluate the accuracy of the model

# In[24]:


model.score(X_test,Y_test)

