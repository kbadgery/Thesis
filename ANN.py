# -*- coding: utf-8 -*-
"""

@author: Kip Badgery
"""

import numpy as np
import pandas as pd
import os
# Data Pre-processing 
data_directory = "Data Collection 2"

subjects_cv = os.listdir(data_directory) #subjects to be used for cross validation
n_features = 5 # number of features used in model

window_length_s=2 #sliding window size in seconds
sample_rate=64 #Hz

window_length=window_length_s*sample_rate # window length in samples
window_overlap=0.5 #percentage of sliding window that overlaps
window_step=window_length*(1-window_overlap) # number of samples that the window steps forward for each iteration

#intiialize variables
z_mean_t=[]
z_std_t=[]
mag_std_t=[]
mag_meancrossing_t=[]
mag_z_correlation_t=[]
exercise=[]
subject_index=[0]

# importing data 
for i in range (0,len(subjects_cv)): # looping through each subject to extract data 
    
    files=os.listdir(os.path.join(data_directory,subjects_cv[i])) # files in current subjects directory
    
    for j in range (0,len(files)): # looping through the files within each subect folder
         
        file_path=os.path.join(data_directory,subjects_cv[i],files[j])
        
        dataset = pd.read_csv(file_path)
        
         # extracting acceleration data
        aX = dataset.iloc[:,4].values
        aY = dataset.iloc[:,5].values
        aZ = dataset.iloc[:,6].values
        aMag = np.sqrt(np.square(aX)+np.square(aY)) #magnitude of the XY acceleration 
        
        # extract features from each sliding window
        for k in np.arange(0,len(aX),window_length*window_overlap):
            
            filename=files[j]
            exercise.append(filename[:-23]) 
            
            # variables are named as follows : axis_metric_domain  (for domain t=time domain, f = frequency domain)
            k=int(k)
            
            # features used:
            # Mean Acceleartion Z-axis time-domain
            # Standard Dev Acceleartion Z-axis time-domain
            # Standard Dev Acceleartion XY-axis time-domain
            # Mean Crossings Acceleration XY-axis time-domain
            # Correlation Coefficient between XY and Z Axes time-domain
            
            z_mean_t.append(np.mean(aX[k:k+window_length]))  
            z_std_t.append(np.std(aX[k:k+window_length])) 
            mag_std_t.append(np.std(aMag[k:k+window_length])) 
            

            data_shifted=aMag[k:k+window_length]-np.mean(aMag[k:k+window_length]) #shifting data down by the mean so that mean crossing rate can be calculated
            mag_meancrossing_t.append(((data_shifted[:-1] * data_shifted[1:]) < 0).sum())  
            
            mag_z_correlation_matrix=np.corrcoef(aMag[k:k+window_length],aZ[k:k+window_length])
            mag_z_correlation_t.append(mag_z_correlation_matrix[0,1]) 
    
            
    
# creating feature (X) and class(y) data sets
y=np.asarray(exercise) 

X=np.zeros((len(y),n_features))
X[:,0]=np.asarray(z_mean_t)
X[:,1]=np.asarray(z_std_t)
X[:,2]=np.asarray(mag_std_t)
X[:,3]=np.asarray(mag_meancrossing_t)
X[:,4]=np.asarray(mag_z_correlation_t) 
   
  
from sklearn.preprocessing import LabelBinarizer
encoder_y=LabelBinarizer()
y=encoder_y.fit_transform(y)

# Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X=sc.fit_transform(X)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# building the Classification ANN

""" The Below Model's hyperparameters were chosen from a downsampled hyperparmeter grid search (See bottom of script),
the cross validation accuracy of this model was 89%, with a training accuracy of 92%, this model was chosed by attempting to maximize
validation accuracy while mininimizing the difference between training accuracy and validation accuracy to minimize overfitting """

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from talos.metrics.keras_metrics import recall

classifier = Sequential()
classifier.add(Dense(units = 10, kernel_initializer = 'normal', activation = 'relu', input_dim = 5))
classifier.add(Dense(units = 10, kernel_initializer = 'normal', activation = 'relu'))
classifier.add(Dense(units = 4, kernel_initializer = 'normal', activation = 'softmax'))              
classifier.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy', recall])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 400)


y_pred_prob = classifier.predict(X_test)

# convert the predicted probabilities of each exercise to an actual prediction
y_pred = (y_pred_prob == y_pred_prob.max(axis=1)[:,None]).astype(int)

# perform inverse transform on one-hot encoded data to get final predictions array
y_pred_1D = encoder_y.inverse_transform(y_pred,threshold=1)
y_test_1D = encoder_y.inverse_transform(y_test,threshold=1)


# Metrics
from sklearn.metrics import confusion_matrix, recall_score
cm = confusion_matrix(y_test_1D, y_pred_1D)
recall=recall_score(y_test_1D, y_pred_1D,average='micro')

print(cm)

print("Recall:", recall)



""" Hyper Parameter Grid Search Using Talos 

import talos as ta
from talos.model import hidden_layers, lr_normalizer
from talos.metrics.keras_metrics import fmeasure, recall
from keras.optimizers import Adam, Nadam
from keras.activations import relu, elu, softmax
from keras.losses import categorical_crossentropy, logcosh
from keras.layers import Dropout


# optimizing the model
def classification_model (x_train, y_train, x_val, y_val, params):
    model = Sequential()
    model.add(Dense(params['first_neuron'], input_dim=x_train.shape[1],
                    activation=params['activation'],
                    kernel_initializer='normal'))
    
    model.add(Dropout(params['dropout']))
    

    hidden_layers(model, params, 4)

    model.add(Dense(4, activation=params['last_activation'],
                    kernel_initializer='normal'))
    
    model.compile(loss=params['losses'],
                  # here we add a regulizer normalization function from Talos
                  optimizer=params['optimizer'](lr=lr_normalizer(params['lr'],params['optimizer'])),
                  metrics=['accuracy', recall])
    
    history = model.fit(x_train, y_train, 
                        validation_data=[x_val, y_val],
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        verbose=0)
    return history, model
         

# setting the hyperparemeters to be used in hyperparemeter grid search
p = {'lr': [],
     'first_neuron': [10, 20],
     'hidden_layers':[2,6],
     'batch_size': [1, 50],
     'epochs': [100],
     'dropout': [0],
     'weight_regulizer':[None],
     'emb_output_dims': [None],
     'shape':['brick','diamond'],
     'optimizer': [Adam],
     'losses': [categorical_crossentropy],
     'activation':[relu],
     'last_activation': [softmax]}

# and run the experiment
t = ta.Scan(x=X_train,
            y=y_train,
            model=classification_model,
            grid_downsample=1,
            params=p,
            dataset_name='Exercise',
            experiment_no='4')
   
        
"""        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
