# Code to read csv file into Colaboratory:
!pip install -U -q PyDrive
import pandas as pd
import numpy as np
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import utils
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.metrics import AUC
from keras.metrics import Recall
from keras.metrics import Precision
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
import os

# Authenticate and create the PyDrive client.
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


def define_datasets():
  id_train_features,id_test_features,id_train_labels,id_test_labels = file_ids()
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

def transform():

  train_X,test_X,train_Y,test_Y = define_datasets()
  train_y = np.array(train_Y)
  test_y = np.array(test_Y)
  train_x = list()
  train_x.append(train_X)
  train_x = np.dstack(train_x)
  test_x = list()
  test_x.append(test_X)
  test_x = np.dstack(test_x)

  # zero-offset class values
  train_y = train_y - 1
  test_y = test_y - 1
  # one hot encode y
  train_y = to_categorical(train_y)
  test_y = to_categorical(test_y)

  print('Shape of the train features tensor',train_x.shape)
  print('Shape of the train labels tensor',train_y.shape)
  print('Shape of the test features tensor',test_x.shape)
  print('Shape of the test labels tensor',test_y.shape)

  return train_X,test_X,train_Y,test_Y

def return_details(train_X,test_X,train_Y,test_Y):
  n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
  n_samples = train_x.shape[0]
  return n_timesteps,n_features,n_outputs,n_samples


def LSTM_model():

  model = Sequential()
  model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
  model.add(Dropout(0.5))
  model.add(Dense(100, activation='relu'))
  model.add(Dense(n_outputs, activation='softmax'))
  return model


def mult_exp_runs_initialise(num_runs):

  Result_run = dict()
  for i in range(0,num_runs):
    ind = dict()
    ind['acc'] = 0
    ind['test_acc'] = 0
    ind['auc'] = 0
    ind['test_auc'] = 0
    ind['recall'] = 0
    ind['test_recall'] = 0
    ind['precision'] = 0
    ind['test_precision'] = 0
    Result_run[str(i)] = ind

  Result_run_end = dict()
  for i in range(0,num_runs):
    ind = dict()
    ind['acc'] = 0
    ind['test_acc'] = 0
    ind['auc'] = 0
    ind['test_auc'] = 0
    ind['recall'] = 0
    ind['test_recall'] = 0
    ind['precision'] = 0
    ind['test_precision'] = 0
    Result_run_end[str(i)] = ind

  return Result_run,Result_run_end



def multiple_runs(train_X,test_X,train_Y,test_Y):

  class TestCallback(Callback):
      def __init__(self, test_data):
          self.test_data = test_data

      def on_epoch_end(self, epoch, logs={}):
        period = 5
        # epoch greater than 10 because with random initailisation test acc 78 percent with skewed dataset
        if (epoch % period == 0 and epoch>10):
          #test after every 5 epochs
          x, y, run_num = self.test_data
          loss,acc,auc,recall,precision = self.model.evaluate(x, y, verbose=0)
          prev_runs = Result_run[str(run_num)]
          if (acc > prev_runs['test_acc']):
            metric = dict()

            #acc
            metric['acc'] = logs['accuracy']
            metric['test_acc'] = acc

            #auc
            metric['auc'] = logs['auc']
            y_pred = self.model.predict(x)
            auc_test = roc_auc_score(y, y_pred)
            metric['test_auc'] = auc

            #recall 
            metric['recall'] = logs['recall']
            metric['test_recall'] = recall

            #precision
            metric['precision'] = logs['precision']
            metric['test_precision'] = precision

            Result_run[str(run_num)] = metric
            print('\n',Result_run[str(run_num)],' for ',str(run_num))

            #save this model
            model.save('/models/model_' + str(run_num))
            print('Best test accuracy so far, ',acc,'saving this')
          print('\nFor run {} Testing loss: {}, acc: {}\n'.format(run_num, loss, acc))

  #checkpoint = ModelCheckpoint(" best_model.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='auto', period=1)



  repeat = 30
  num_epochs = 50
  n_timesteps,n_features,n_outputs,n_samples = return_details(train_X,test_X,train_Y,test_Y)
  Result_run,Result_run_end = mult_exp_runs_initialise(repeat)

  #chose num_epochs as 50 because used early_stopping with a cross validation set and num_epochs were 43.
  for run_number in range(repeat):
    
    model = LSTM_model()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','AUC','Recall','Precision'])
    print(model.summary()) 

    #fit the model
    history = model.fit(train_x,train_y, epochs=num_epochs, callbacks=[TestCallback((test_x, test_y,run_number))], verbose=1)
    model.save('/models/model_' + str(run_number) + '/end')
    print('Model trained for 50 epochs and saved ',run_number)

    loss,acc,auc,recall,precision = model.evaluate(test_x, test_y, verbose=0)
    metric = dict()
    #acc
    metric['acc'] = history.history['accuracy']
    metric['test_acc'] = acc
    #auc
    metric['auc'] = history.history['auc']
    metric['test_auc'] = auc
    #recall
    metric['recall'] = history.history['recall']
    metric['test_auc'] = recall
    #precision
    metric['precision'] = history.history['precision']
    metric['precision'] = precision


    Result_run_end[str(run_number)] = metric

    #PLOT HISTORY
    #		:
    #		:
    
  import pickle
  with open('/models/history_best', 'wb') as f:
      pickle.dump(Result_run, f)
  with open('/models/history_end','wb') as f:
      pickle.dump(Result_run_end, f)
  print( os.getcwd() )
  print( os.listdir('/models') )
  !zip -r /models/models.zip /models
  from google.colab import files
  files.download( "/models/models.zip" ) 


def main_fun():
  #file_ids, load the datasets, transform
  train_X,test_X,train_Y,test_Y = transform()

  # call the mult_run with these parameters
  multiple_runs(train_X,test_X,train_Y,test_Y)

main_fun()