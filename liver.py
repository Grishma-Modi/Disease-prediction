import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#from sklearn.externals import joblib
import joblib
from joblib import dump, load



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score



#File does not contain headers so we need to load the headers manually
#features = ["Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase", "Alamine_Aminotransferase", "Aspartate_Aminotransferase", "Total_Protiens", "Albumin", "Albumin_and_Globulin_Ratio", "Dataset"]
#data = pd.read_csv('indian_liver_patient.csv', names = features)
data = pd.read_csv('indian_liver_patient.csv')
data.head()

data.info()


"""#Transfrom Gender string into float values
le = preprocessing.LabelEncoder()
le.fit(['Male','Female'])
data.loc[:,'Gender'] = le.transform(data['Gender'])

#Remove rows with missing values
data = data.dropna(how = 'any', axis = 0)

#Also transform Selector variable into usual conventions followed
data['Selector'] = data['Selector'].map({2:0, 1:1})
"""


data=data.fillna(method="ffill")
data.Gender=data.Gender.map({"Female":1,"Male":0})
data["Dataset"]=data["Dataset"].map({1:0,2:1})
data.head()

np.random.shuffle(data.values)
print(data.shape[1])
print(data.columns)



"""Y=data["Dataset"]
X=data.drop(["Dataset"],axis=1)
print(X)
print(Y)

data.head()

data.describe()

X_train, X_test, y_train, y_test = train_test_split(data, data['Dataset'], random_state = 0)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Determining the healthy-affected split
print("Positive records:", data['Dataset'].value_counts().iloc[0])
print("Negative records:", data['Dataset'].value_counts().iloc[1])


#Determine statistics based on age
plt.figure(figsize=(12, 10))
plt.hist(data[data['Dataset'] == 1]['Age'], bins = 16, align = 'mid', rwidth = 0.5, color = 'black', alpha = 0.8)
plt.xlabel('Age')
plt.ylabel('Number of Patients')
plt.title('Frequency-Age Distribution')
plt.grid(True)
plt.savefig('fig1')
plt.show()


#correlation-matrix
plt.subplots(figsize=(12, 10))
plt.title('Pearson Correlation of Features')
# Draw the heatmap using seaborn
sns.heatmap(data.corr(),linewidths=0.25, vmax=1.0, square=True,annot=True)
plt.savefig('fig2')
plt.show()


#Using normal data
logreg = LogisticRegression(C = 0.1).fit(X_train, y_train)
print("Logistic Regression Classifier on unscaled test data:")
print("Accuracy:", logreg.score(X_test, y_test))
print("Precision:", precision_score(y_test, logreg.predict(X_test)))
print("Recall:", recall_score(y_test, logreg.predict(X_test)))
print("F-1 score:", f1_score(y_test, logreg.predict(X_test)))


#Using feature-scaled data
logreg_scaled = LogisticRegression(C = 0.1).fit(X_train_scaled, y_train)
print("Logistic Regression Classifier on scaled test data:")
print("Accuracy:", logreg_scaled.score(X_test_scaled, y_test))
print("Precision:", precision_score(y_test, logreg_scaled.predict(X_test_scaled)))
print("Recall:", recall_score(y_test, logreg_scaled.predict(X_test_scaled)))
print("F-1 score:", f1_score(y_test, logreg_scaled.predict(X_test_scaled)))


#Using normal data
svc_clf = SVC(C = 0.1, kernel = 'rbf').fit(X_train, y_train)
print("SVM Classifier on unscaled test data:")
print("Accuracy:", svc_clf.score(X_test, y_test))
print("Precision:", precision_score(y_test, svc_clf.predict(X_test)))
print("Recall:", recall_score(y_test, svc_clf.predict(X_test)))
print("F-1 score:", f1_score(y_test, svc_clf.predict(X_test)))

#Using scaled data
svc_clf_scaled = SVC(C = 0.1, kernel = 'rbf').fit(X_train_scaled, y_train)
print("SVM Classifier on scaled test data:")
print("Accuracy:", svc_clf_scaled.score(X_test_scaled, y_test))
print("Precision:", precision_score(y_test, svc_clf_scaled.predict(X_test_scaled)))
print("Recall:", recall_score(y_test, svc_clf_scaled.predict(X_test_scaled)))
print("F-1 score:", f1_score(y_test, svc_clf_scaled.predict(X_test_scaled)))


#Using normal data
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
print("k-NN Classifier on unscaled test data:")
print("Accuracy:", knn.score(X_test, y_test))
print("Precision:", precision_score(y_test, knn.predict(X_test)))
print("Recall:", recall_score(y_test, knn.predict(X_test)))
print("F-1 score:", f1_score(y_test, knn.predict(X_test)))


#Using scaled data
knn_scaled = KNeighborsClassifier(n_neighbors = 5)
knn_scaled.fit(X_train_scaled, y_train)
print("SVM Classifier on scaled test data:")
print("Accuracy:", knn_scaled.score(X_test_scaled, y_test))
print("Precision:", precision_score(y_test, knn_scaled.predict(X_test_scaled)))
print("Recall:", recall_score(y_test, knn_scaled.predict(X_test_scaled)))
print("F-1 score:", f1_score(y_test, knn_scaled.predict(X_test_scaled)))



#using normal data
rfc = RandomForestClassifier(n_estimators = 20)
rfc.fit(X_train, y_train)
print("SVM Classifier on unscaled test data:")
print("Accuracy:", rfc.score(X_test, y_test))
print("Precision:", precision_score(y_test, rfc.predict(X_test)))
print("Recall:", recall_score(y_test, rfc.predict(X_test)))
print("F-1 score:", f1_score(y_test, rfc.predict(X_test)))


#using scaled data
rfc_scaled = RandomForestClassifier(n_estimators = 20)
rfc_scaled.fit(X_train_scaled, y_train)
print("Random Forest Classifier on scaled test data:")
print("Accuracy:", rfc_scaled.score(X_test_scaled, y_test))
print("Precision:", precision_score(y_test, rfc_scaled.predict(X_test_scaled)))
print("Recall:", recall_score(y_test, rfc_scaled.predict(X_test_scaled)))
print("F-1 score:", f1_score(y_test, rfc_scaled.predict(X_test_scaled)))

joblib.dump(rfc,"model4")"""



data=pd.read_csv("indian_liver_patient.csv")
data=data.fillna(method="ffill")
data.Gender=data.Gender.map({"Female":1,"Male":0})
data["Dataset"]=data["Dataset"].map({1:0,2:1})
np.random.shuffle(data.values)
print(data.shape[1])
print(data.columns)


target=data["Dataset"]
source=data.drop(["Dataset"],axis=1)

sm=SMOTE()
sc=StandardScaler()
lr=LogisticRegression()
source=sc.fit_transform(source)
X_train,X_test,y_train,y_test= train_test_split(source,target,test_size=0.25)
X_train, y_train=sm.fit_sample(X_train,y_train)
cv=cross_validate(lr,X_train,y_train,cv=10)
lr.fit(X_train,y_train)
print(cv)
joblib.dump(lr,"model4")


"""Y_pred = lr.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, Y_pred)
cm

predictions = lr.predict(X_test)
predictions

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
"""

# accuracy: 60.96%"""
