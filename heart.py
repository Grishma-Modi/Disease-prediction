import pandas as pd
import numpy as np
#from sklearn.externals import joblib
import joblib
from joblib import dump, load
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.linear_model import LogisticRegression
#warnings.filterwarnings("ignore", category=DeprecationWarning) 
from sklearn.preprocessing import StandardScaler
import random
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate


"""data = pd.read_csv("heart.csv")
data["trestbps"]=np.log(data["trestbps"])

data=data.drop(["fbs"],axis=1)
data=data.drop(["ca"],axis=1)
data["chol"]=np.log(data["chol"])
target=data["target"]
print(data.shape[1])

np.random.shuffle(data.values)
data=data.drop(["target"],axis=1)
print(data.columns)
sc= StandardScaler()
data=sc.fit_transform(data)

lr=LogisticRegression()
lr.fit(data,target)
cv_results = cross_validate(lr, data,target, cv=10)
print(cv_results)

joblib.dump(lr,"model2")"""



# Loading the dataset
df = pd.read_csv('heart.csv')


# # **Exploring the dataset**

# In[3]:


# Returns number of rows and columns of the dataset
df.shape


# In[4]:


# Returns an object with all of the column headers
df.columns


# In[5]:


# Returns different datatypes for each columns (float, int, string, bool, etc.)
df.dtypes


# In[6]:


# Returns the first x number of rows when head(x). Without a number it returns 5
df.head()


# In[7]:


# Returns the last x number of rows when tail(x). Without a number it returns 5
df.tail()


# In[8]:


# Returns true for a column having null values, else false
df.isnull().any()


# In[9]:


# Returns basic information on all columns
df.info()


# In[10]:


# Returns basic statistics on numeric columns
df.describe().T


# # **Data Visualization**

# In[11]:


# Importing essential libraries
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[12]:


# Plotting histogram for the entire dataset
fig = plt.figure(figsize = (15,15))
ax = fig.gca()
g = df.hist(ax=ax)


# In[13]:


# Visualization to check if the dataset is balanced or not
g = sns.countplot(x='target', data=df)
plt.xlabel('Target')
plt.ylabel('Count')


# # **Feature Engineering**

# ### Feature Selection

# In[14]:


# Selecting correlated features using Heatmap

# Get correlation of all the features of the dataset
corr_matrix = df.corr()
top_corr_features = corr_matrix.index

# Plotting the heatmap
plt.figure(figsize=(20,20))
sns.heatmap(data=df[top_corr_features].corr(), annot=True, cmap='RdYlGn')


# # **Data Preprocessing**

# ## Handling categorical features
# 
# After exploring the dataset, I observed that converting the categorical variables into dummy variables using 'get_dummies()'. Though we don't have any strings in our dataset it is necessary to convert ('sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal') these features.
# 
# *Example: Consider the 'sex' column, it is a binary feature which has 0's and 1's as its values. Keeping it as it is would lead the algorithm to think 0 is lower value and 1 is a higher value, which should not be the case since the gender cannot be ordinal feature.*

# In[15]:


dataset = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])


# ## Feature Scaling

# In[16]:


dataset.columns


# In[17]:


from sklearn.preprocessing import StandardScaler
standScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standScaler.fit_transform(dataset[columns_to_scale])


# In[18]:


dataset.head()


# In[19]:


# Splitting the dataset into dependent and independent features
X = dataset.drop('target', axis=1)
y = dataset['target']


# # **Model Building**
# 
# I will be experimenting with 3 algorithms:
# 1. KNeighbors Classifier
# 2. Decision Tree Classifier
# 3. Random Forest Classifier

# ## KNeighbors Classifier Model

# In[20]:


# Importing essential libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


# In[21]:


# Finding the best accuracy for knn algorithm using cross_val_score 
knn_scores = []
for i in range(1, 21):
  knn_classifier = KNeighborsClassifier(n_neighbors=i)
  cvs_scores = cross_val_score(knn_classifier, X, y, cv=10)
  knn_scores.append(round(cvs_scores.mean(),3))


# In[22]:


# Plotting the results of knn_scores
plt.figure(figsize=(20,15))
plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')
for i in range(1,21):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1, 21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')


# In[40]:


# Training the knn classifier model with k value as 12
knn_classifier = KNeighborsClassifier(n_neighbors=12)
cvs_scores = cross_val_score(knn_classifier, X, y, cv=10)
print("KNeighbours Classifier Accuracy with K=12 is: {}%".format(round(cvs_scores.mean(), 4)*100))


# ## Decision Tree Classifier

# In[24]:


# Importing essential libraries
from sklearn.tree import DecisionTreeClassifier


# In[25]:


# Finding the best accuracy for decision tree algorithm using cross_val_score 
decision_scores = []
for i in range(1, 11):
  decision_classifier = DecisionTreeClassifier(max_depth=i)
  cvs_scores = cross_val_score(decision_classifier, X, y, cv=10)
  decision_scores.append(round(cvs_scores.mean(),3))


# In[26]:


# Plotting the results of decision_scores
plt.figure(figsize=(20,15))
plt.plot([i for i in range(1, 11)], decision_scores, color = 'red')
for i in range(1,11):
    plt.text(i, decision_scores[i-1], (i, decision_scores[i-1]))
plt.xticks([i for i in range(1, 11)])
plt.xlabel('Depth of Decision Tree (N)')
plt.ylabel('Scores')
plt.title('Decision Tree Classifier scores for different depth values')


# In[27]:


# Training the decision tree classifier model with max_depth value as 3
decision_classifier = DecisionTreeClassifier(max_depth=3)
cvs_scores = cross_val_score(decision_classifier, X, y, cv=10)
print("Decision Tree Classifier Accuracy with max_depth=3 is: {}%".format(round(cvs_scores.mean(), 4)*100))


# ## Random Forest Classifier

# In[28]:


# Importing essential libraries
from sklearn.ensemble import RandomForestClassifier


# In[29]:


# Finding the best accuracy for random forest algorithm using cross_val_score 
forest_scores = []
for i in range(10, 101, 10):
  forest_classifier = RandomForestClassifier(n_estimators=i)
  cvs_scores = cross_val_score(forest_classifier, X, y, cv=5)
  forest_scores.append(round(cvs_scores.mean(),3))


# In[30]:


# Plotting the results of forest_scores
plt.figure(figsize=(20,15))
plt.plot([n for n in range(10, 101, 10)], forest_scores, color = 'red')
for i in range(1,11):
    plt.text(i*10, forest_scores[i-1], (i*10, forest_scores[i-1]))
plt.xticks([i for i in range(10, 101, 10)])
plt.xlabel('Number of Estimators (N)')
plt.ylabel('Scores')
plt.title('Random Forest Classifier scores for different N values')


# In[39]:


# Training the random forest classifier model with n value as 90
forest_classifier = RandomForestClassifier(n_estimators=90)
cvs_scores = cross_val_score(forest_classifier, X, y, cv=5)
print("Random Forest Classifier Accuracy with n_estimators=90 is: {}%".format(round(cvs_scores.mean(), 4)*100))


"""joblib.dump(lr,"model2")"""

# Best accuracy using KNN: 84.48%