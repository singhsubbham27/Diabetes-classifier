#!/usr/bin/env python
# coding: utf-8

# # Diabetes classifier

# # Install Libraries

# In[34]:


import pandas as pd
import matplotlib.pyplot as plt
#from pandas.plotlib import scatter_matrix 
from sklearn import linear_model,metrics,model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import classification_report


# # Data preparation and initial analysis
# 
# Here we focus on preparing and analyzing health indicator data for diabetes from the year 2015 and 2021.The goal is to combine these datasets to from a comprehensive views of the data, which can be better use for the machine learning.The following are taken:
# 
# 1.Import libraries
# 
# 2.Load the Data
# 
# 3.Combine the DataFrames

# In[17]:


# file names and ulrs
#filepath_2015='http://localhost:8888/edit/Downloads/2015.csv'
#filepath_2021='http://localhost:8888/edit/Downloads/2021.csv'
df1=pd.read_csv("2015.csv")
df2=pd.read_csv("2021.csv")

#Combine the two DataFrame
combined_df=pd.concat([df1,df2],axis=0).reset_index(drop=True)


# In[19]:


# Display the first few data of combined dataframe
combined_df.head()


# In[21]:


combined_df.shape


# # Removing Features
# 
# Here we remove features deemed irrelevant for modeling purposes.we specify a list of columns to be removed and then drop there columns from the combinedd DataFrame

# In[22]:


# Removes irrelevent features from dataset
columns_to_remove=['CholCheck','AnyHealthcare','NoDocbcCost','Education','Income']
reduced_df=combined_df.drop(columns=columns_to_remove)
reduced_df.head()


# # Check for the Missing Values
# 
# check for the missing values in the dataset.Note there are no missing values will found.

# In[24]:


# check for the missing values in the dataset
missing_values=reduced_df.isnull().sum()
missing_values


# In[25]:


reduced_df.describe()


# In[26]:


# check range of values of specified features to determine suitable data types
features_to_optimize=['BMI','GenHlth','MentHlth','PhysHlth','Age']
data_types_optimization=(reduced_df[features_to_optimize].describe().loc[['min','max']])


#memory used before reducing data types
memory_before=reduced_df.memory_usage(index=True).sum()

data_types_optimization


# # Data Type Conversion for Efficiency
# We enhance the dataset's memory efficiency by converting specified binary columns to boolean data types. We print the memory usage before and after the operation to demonstrate the effectiveness of this optimization in reducing the dataset's memory consumption.

# In[27]:


binary_columns = ['Diabetes_binary', 'HighBP', 'HighChol', 'Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'DiffWalk', 'Sex']
for column in binary_columns:
    reduced_df[column] = reduced_df[column].astype('bool')

# After data types reduces memory size
memory_after = reduced_df.memory_usage(index=True).sum()

print("Dataframe memory used before:", memory_before)
print("Dataframe memory used after:  ", memory_after)


# # Logistic Regression Model for Diabetes Prediction
# Here we prepare data for machine learning, specifically using a logistic regression model to predict diabetes. First, numerical columns are identified and scaled using MinMaxScaler to ensure all features contribute equally to the model without bias from varying scales. A logistic regression model is then initialized with specific parameters. The dataset is split into features (X_log) and the target variable (y_log). The data is further divided into training and test sets to evaluate the model's performance on unseen data.
# 
# After training the logistic regression model, predictions are made on the test set. The model's effectiveness is assessed using accuracy, confusion matrix, and classification report, providing a comprehensive overview of its predictive capabilities in distinguishing between diabetic and non-diabetic individuals

# In[35]:


###### Logistic Regression algorithm ######

# Selecting numerical columns (excluding binary/boolean columns)
numerical_columns = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age']

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the numerical features
reduced_df[numerical_columns] = scaler.fit_transform(reduced_df[numerical_columns])

mylog_model = linear_model.LogisticRegression(solver='saga', max_iter=1000)

# 'X' is the feature set and 'y' is the target variable
X_log = reduced_df.drop('Diabetes_binary', axis=1)
y_log = reduced_df['Diabetes_binary'].astype('bool')  # Ensuring the target is boolean

# Splitting the dataset into the Training set and Test set
X_log_train, X_log_test, y_log_train, y_log_test = model_selection.train_test_split(X_log, y_log, test_size=0.25, random_state=42)

mylog_model.fit(X_log_train, y_log_train)

y_pred_log = mylog_model.predict(X_log_test)

# Evaluate the model
accuracy = accuracy_score(y_log_test, y_pred_log)
conf_matrix = confusion_matrix(y_log_test, y_pred_log)
class_report = classification_report(y_log_test, y_pred_log)

print("\nLogistic Regression prediction results:")
print(f"Accuracy: {round(accuracy*100,2)} %")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)


# # Cross-Validation of Logistic Regression Model
# Next we perform a cross-validation process on the logistic regression model to verify its reliability across different subsets of the dataset. By utilizing the KFold method with 5 splits and shuffling enabled, the dataset is divided into distinct subsets to conduct multiple training and testing cycles. The average of these scores is calculated and displayed, offering a robust measure of the model's overall performance. This approach helps to ensure that the model's predictive accuracy is not overly dependent on any particular partition of the data, thereby increasing confidence in its generalizability.

# In[38]:


# Verify model by averaging different test/train splits
k_folds = KFold(n_splits = 5, shuffle=True)
# The number of folds determines the test/train split for each iteration. 
# So 5 folds has 5 different mutually exclusive training sets. 
# That's a 1 to 4 (or .20 to .80) testing/training split for each of the 5 iterations.

log_scores = cross_val_score(mylog_model, X_log, y_log)
# This shows the average score. Print 'scores' to see an array of individual iteration scores.
print("Logistic Regression Average Prediction Score: ", round(log_scores.mean()*100, 2), "%")


# # Confusion Matrix Visual
# Confusion matrixes are used to evaluate the performance in classifying diabetic and non-diabetic individuals.

# In[39]:


# Plot confusion matrix
graph_confusion_matrix = ConfusionMatrixDisplay.from_predictions(y_log_test, y_pred_log)


# # Histogram Visual
# Here we generate histograms for all numerical data, providing a visual distribution analysis of each (non-binary) feature.

# In[40]:


# Plot histogram
graph_histogram = reduced_df.hist()


# In[ ]:




