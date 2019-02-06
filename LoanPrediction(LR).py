import pandas as pd
import numpy as np
import matplotlib as pyplot
import math

# Machine Learning Imports
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

# For evaluating our ML results
from sklearn import metrics


df = pd.read_csv(r"C:\Users\UIX\Desktop\Ashish\projects\LoanPrediction\train.csv")

df_train = df.drop('Loan_ID',1)

df_train.dtypes

mode_col_list = ['Gender','Married','Education','Self_Employed','Property_Area','Dependents']

for col in mode_col_list :
    #mode = df_train[col].mode().to_string()
    mode = df_train[col].mode()[0]#.to_string()
    df_train[col].fillna(mode, inplace=True)
    
#math.ceil(df_train['ApplicantIncome'].mean())   
#math.floor(df_train['ApplicantIncome'].mean())   
 
mean_col_list = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']
 

for col1 in mean_col_list :
    mean = df_train[col1].mean()
    df_train[col1].fillna(mean,inplace=True)
     
     
df_train.head()
     

def Loan_Status_Check(x):
    if x!="Y":
        return 1
    else:
        return 0
    

df_train['Had_Loan'] = df_train['Loan_Status'].apply(Loan_Status_Check)
df_train
 

Groupby = df_train.groupby('Had_Loan').mean()


df_train = df_train.drop('Loan_Status',1)


Y = df_train.Had_Loan
Y.tail()


Y = np.ravel(Y)
Y
#################################################3
# replace the value of String  in the binary form

df_train['Property_Area'] = df_train['Property_Area'].map({'Rural':0, 'Semiurban':1, 'Urban':2})


df_train['Dependents'] = df_train['Dependents'].map({'0':0, '1':1, '2':2,'3+':3})


df_train['Gender'] = df_train['Gender'].map({'Male':1, 'Female':0})


df_train['Married'] = df_train['Married'].map({'Yes':0, 'No':1})


df_train['Education'] = df_train['Education'].map({'Graduate':0, 'Not Graduate':1})


df_train['Self_Employed'] = df_train['Self_Employed'].map({'Yes':0, 'No':1})



####################################################3
df_train_new = df_train.iloc[:,0:10].values   

df_test_new = df_train.iloc[:,11].values   

model_LR = LogisticRegression()

predicted = model_LR.fit(df_train_new,df_test_new)

model_LR.score(df_train_new,df_test_new)

###################################################

from sklearn.model_selection import train_test_split

train_X,test_X,train_Y,test_Y = train_test_split(df_train_new,df_test_new,test_size=0.3,shuffle=True)

predicted_LR = model_LR.fit(train_X,train_Y)

result = predicted_LR.predict(test_X)
result

model_LR.score(train_Y,test_X)


# Train and Test Accuracy
print("Train Accuracy :: ", accuracy_score(train_Y, predicted_LR.predict(train_X)))
print("Test Accuracy  :: ", accuracy_score(test_Y, result))