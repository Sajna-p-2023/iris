import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
     

iris_data = pd.read_excel("/workspaces/iris/iris .xls")
     

iris_data

iris_data.describe()
     

iris_data.isna().sum()



x = iris_data.drop(['Classification'],axis = 1)
y = iris_data['Classification']
     

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =  train_test_split(x,y,test_size=0.2, random_state=42)
     

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model  = lr.fit(x_train,y_train)
lr_predictions = model.predict(x_test)

     

from sklearn.metrics import accuracy_score


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
     

model.fit(x_train, y_train)
     
KNeighborsClassifier()


model.score(x_test, y_test)
     


from sklearn.svm import SVC
svm_class = SVC(kernel = 'linear')
model = svm_class.fit(x_train,y_train)
svm_pred = model.predict(x_test)
     


from sklearn.tree import DecisionTreeClassifier
dt_class = DecisionTreeClassifier()
model = dt_class.fit(x_train,y_train)
dt_pred = model.predict(x_test)
     

print('Logistic regression Accuracy : ',accuracy_score(y_test,lr_predictions))
print('SVM linear Accuracy : ',accuracy_score(y_test,svm_pred))
print('KNN Accuracy : ',model.score(x_test, y_test))
print('DT Accuracy : ', accuracy_score(y_test,dt_pred))

# save the model
import pickle
filename = 'savedmodel.pkl'
pickle.dump(model, open(filename, 'wb'))
     

load_model = pickle.load(open(filename,'rb'))
     

load_model.predict([[6.0, 2.2, 4.0, 1.0]])