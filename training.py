import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

data_df= pd.read_csv("C:\\Users\\aksha\\PycharmProjects\\breastcancer\\DATA_BREAST_CANCER.csv")
print(data_df.head())

print(data_df.describe())

print(pd.DataFrame(data_df.isna().sum()))

data_df =data_df.set_index('Id')
data_df.drop(columns=['Unnamed: 0'],axis = 1,inplace = True)

_ =  sns.countplot(data_df.diagnosis,label='count')
B, M = data_df.diagnosis.value_counts()
print('B = ',B)
print('M = ',M)
encoder = LabelEncoder()
data_df.diagnosis = encoder.fit_transform(data_df.diagnosis)
# print(data_df.head())accuracy_score(y_test,y_pred)

# sns.pairplot(data_df,dropna=True)
# plt

X = data_df.drop(columns=['diagnosis'],axis=1).values
y = data_df['diagnosis']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
classifier = RandomForestClassifier(n_estimators = 100)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))

print(accuracy_score(y_test,y_pred))


