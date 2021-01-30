import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
#from sklearn.externals import joblib
import joblib
from joblib import dump, load


import pickle

# DATA FOR PRED
data=pd.read_csv("diabetes.csv")
print(data.head())

# Renaming DiabetesPedigreeFunction as DPF
data = data.rename(columns={'DiabetesPedigreeFunction':'DPF'})

# Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
data_copy = data.copy(deep=True)
data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

# Replacing NaN value by mean, median depending upon distribution
data_copy['Glucose'].fillna(data_copy['Glucose'].mean(), inplace=True)
data_copy['BloodPressure'].fillna(data_copy['BloodPressure'].mean(), inplace=True)
data_copy['SkinThickness'].fillna(data_copy['SkinThickness'].median(), inplace=True)
data_copy['Insulin'].fillna(data_copy['Insulin'].median(), inplace=True)
data_copy['BMI'].fillna(data_copy['BMI'].median(), inplace=True)


# Model Building
from sklearn.model_selection import train_test_split
X = data.drop(columns='Outcome')
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# Creating Random Forest Model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

# Creating a pickle file for the classifier
#filename = 'diabetes-prediction-rfc-model.pkl'
#pickle.dump(classifier, open(filename, 'wb'))


predictions = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
accuracy=confusion_matrix(y_test,predictions)

accuracy


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

predictions






"""import xgboost as xgb
from xgboost.sklearn import XGBClassifier"""
#from sklearn import cross_validation, metrics   #Additional scklearn functions
#from sklearn.grid_search import GridSearchCV   #Perforing grid search


# fit model no training data
"""model = XGBClassifier()
model.fit(X_train, y_train)
print(model)


# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))"""






"""from sklearn.model_selection import GridSearchCV"""
# Create the parameter grid based on the results of random search 
"""parameters = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}"""

"""parameters = {
    'bootstrap': [True],
    'max_depth': [90],
    'max_features': [2],
    'min_samples_leaf': [4],
    'min_samples_split': [12],
    'n_estimators': [300]
}"""

# Create a based model
#rf = RandomForestRegressor()
# Instantiate the grid search model
#grid_search = GridSearchCV(estimator = classifier, param_grid = param_grid, 
                         # cv = 3, n_jobs = -1, verbose = 2)


"""grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
accuracy_best = grid_search.best_score_
accuracy_best


grid_search.best_params_"""


#logreg.fit(X,y.reshape(-1,))
joblib.dump(classifier,"model1")


"""logreg=LogisticRegression()

X=data.iloc[:,:8]
print(X.shape[1])

y=data[["Outcome"]]

X=np.array(X)
y=np.array(y)

logreg.fit(X,y.reshape(-1,))
joblib.dump(logreg,"model1")
"""

#Best accuracy using random forest: 81%