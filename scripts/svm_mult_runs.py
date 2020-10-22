import pandas as pd
import numpy as np
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
from sklearn import svm
from sklearn.utils.validation import column_or_1d
from sklearn.metrics import accuracy_score  
from sklearn.metrics import roc_auc_score

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

def file_ids():
  print('File IDs')
  id_train_features = '1LnKdehYxXYXIUhxO1NFeu5GebmKpbYjY'
  id_test_features = '1CIuprbEUeJDEdg0WO5Sq2-xV8OBElXLX'
  id_train_labels = '1dzgH430IP8VV4Jw6InbfTiMEFUxZk1pO'
  id_test_labels = '10dwdJ5EI_w-HfeM3F8CyctlqAnuW3RY9'
  return id_train_features,id_test_features,id_train_labels,id_test_labels


def define_datasets(id_train_features,id_test_features,id_train_labels,id_test_labels):
  print('Loading the datasets')
  downloaded = drive.CreateFile({'id':id_train_features})   # replace the id with id of file you want to access
  downloaded.GetContentFile('train_features.csv')  
  # Dataset is now stored in a Pandas Dataframe
  train_X = pd.read_csv('train_features.csv',header=None)
  train_X.head()


  downloaded = drive.CreateFile({'id':id_test_features})   # replace the id with id of file you want to access
  downloaded.GetContentFile('test_features.csv')  
  # Dataset is now stored in a Pandas Dataframe
  test_X = pd.read_csv('test_features.csv',header=None)
  test_X.head()

  downloaded = drive.CreateFile({'id':id_train_labels})   # replace the id with id of file you want to access
  downloaded.GetContentFile('train_labels.csv')  
  # Dataset is now stored in a Pandas Dataframe
  train_Y = pd.read_csv('train_labels.csv',header=None)
  train_Y.head()

  downloaded = drive.CreateFile({'id':id_test_labels})   # replace the id with id of file you want to access
  downloaded.GetContentFile('test_labels.csv')  
  # Dataset is now stored in a Pandas Dataframe
  test_Y = pd.read_csv('test_labels.csv',header=None)
  test_Y.head()
  test_Y = test_Y[5]

  return train_X,test_X,train_Y,test_Y

def final_ds(train_X,test_X,train_Y,test_Y):
  print('Converting datasets to correct shape')
  x = np.array(train_X)
  y = np.array(train_Y)
  y = column_or_1d(y, warn=True)
  x_t = np.array(test_X)
  y_t = np.array(test_Y)
  y_t = column_or_1d(y_t, warn=True)
  print('Shape of training dataset',x.shape)
  print('Shape of label tensor:', y.shape)
  print('Shape of test dataset',x_t.shape)
  print('Shape of test label tensor:', y_t.shape)

  return x,y,x_t,y_t


def model_fit(x,y,x_t,y_t,i):
  print('Model fitting begins')
  clf = svm.SVC(probability=True,random_state=i)
  clf.fit(x, y)
  print('Model fitting stops')

  print('Calculating accuracy score')
  results = clf.predict(x_t)
  results_train = clf.predict(x)
  preds_train = clf.predict_proba(x)
  preds_test = clf.predict_proba(x_t) 
  acc_score = accuracy_score(y_t, results)
  acc_train = accuracy_score(y,results_train)
  
  print('Calculating  ROC AUC')
  a = np.zeros(preds_test.shape[0])
  j = 0
  for i in preds_test:
    a[j] = i[1]
    j = j+1

  roc = roc_auc_score(y_t, a)

  print('Calculating  ROC AUC train')
  b = np.zeros(preds_train.shape[0])
  j = 0
  for i in preds_train:
    b[j] = i[1]
    j = j+1

  roc_train = roc_auc_score(y, b)

  print(acc_score,'Accuracy score for test set')
  print(roc,'ROC AUC score for test set')
  print(acc_train,'Accuracy score for train set')
  print(roc_train,'ROC AUC Score for trains set')

def mult_runs():

  #import_libs()
  #authenticate()

  id_train_features,id_test_features,id_train_labels,id_test_labels = file_ids()
  train_X,test_X,train_Y,test_Y = define_datasets(id_train_features,id_test_features,id_train_labels,id_test_labels)

  x,y,x_t,y_t = final_ds(train_X,test_X,train_Y,test_Y)
  
  for i in range(0,2):
    print('For the ith run where i =',i)
    model_fit(x,y,x_t,y_t,i)

mult_runs()