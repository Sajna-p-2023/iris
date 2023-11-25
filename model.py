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
print('Logistic regression Accuracy : ',accuracy_score(y_test,lr_predictions))

# save the model
import pickle
pickle_file="savedmodel.pickle"
with open(pickle_file,'wb') as file:
    pickle.dump(model,file)
    

