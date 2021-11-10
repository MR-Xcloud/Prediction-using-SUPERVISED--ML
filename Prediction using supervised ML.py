#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("http://bit.ly/w-data")
df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:



sns.boxplot(data=df[["Hours","Scores"]])


# In[6]:


df.plot.scatter(x="Hours",y="Scores")
plt.title("Hours vs. Scores")
plt.grid()
plt.show()


# In[7]:



X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values


# In[8]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                    test_size = 0.20, random_state = 0)


# In[9]:


#Here we are using 80% of our dataset for training and 20% of the data for testing.


# In[10]:


#TRAINING THE ALGORITHM
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[11]:


line = regressor.coef_*X+regressor.intercept_

df.plot.scatter(x="Hours",y="Scores")
plt.plot(X, line);
plt.grid()
plt.show()


# In[12]:


#LETS MAKE SOME PREDICTIONS
y_pred = regressor.predict(X_test)
print(y_pred)


# In[13]:


#COMPARING ACTUAL SCORE VS PREDICTED SCORE
df_compare = pd.DataFrame({"Actual Score":y_test,"Predicted Score":y_pred})
df_compare


# In[14]:


#HERE WE NEEDED TO PREDICT THE SCORE IF A STUDENT STUDIES FOR 9.25 HOURS/DAY
my_hours = np.array([[9.25]])
my_pred = regressor.predict(my_hours)
print("No of Hours = {}".format(my_hours[0][0]))
print("Predicted Score = {}".format(my_pred[0]))


# In[15]:


#EVALUATING THE MODEL
import sklearn.metrics as metrics

explained_variance=metrics.explained_variance_score(y_test, y_pred)
mean_absolute_error=metrics.mean_absolute_error(y_test, y_pred) 
mse=metrics.mean_squared_error(y_test, y_pred) 
mean_squared_log_error=metrics.mean_squared_log_error(y_test, y_pred)
median_absolute_error=metrics.median_absolute_error(y_test, y_pred)
r2=metrics.r2_score(y_test, y_pred)

print('Explained Variance: ', round(explained_variance,4))    
print('mean_squared_log_error: ', round(mean_squared_log_error,4))
print('r2: ', round(r2,4))
print('MAE: ', round(mean_absolute_error,4))
print('MSE: ', round(mse,4))
print('RMSE: ', round(np.sqrt(mse),4))


# In[ ]:




