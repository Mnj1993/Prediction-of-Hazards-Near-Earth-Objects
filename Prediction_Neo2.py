# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#data analysis libraries 
import numpy as np
import pandas as pd

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
#ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Read using Pandas

data = pd.read_csv('/kaggle/input/nasa-nearest-earth-objects/neo.csv')
data.head(5)

data.info()

data.shape

data.isnull().sum()

data.describe()

# Check for Duplicates 

print(f'Duplicates in Dataset: {data.duplicated().sum()}, ({np.round(100*data.duplicated().sum()/len(data),1)}%)')
print('')

# Split into X and y, Irrelevant features dropped, Hazardous transformed into int

X = data.drop(['id','name','est_diameter_max','orbiting_body','sentry_object','hazardous'],axis=1) 

y = data.hazardous.astype('int')
print(X.shape,y.shape)

# Train/Test/Split Method

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Nueral Network

classifier = Sequential()
classifier.add(Dense(12, input_dim=4, activation='relu'))
classifier.add(Dense(8, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))
classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier.summary()

# Fit Model

history = classifier.fit(X_train, y_train, batch_size = 18, epochs = 10,
    validation_split=0.1,verbose = 1,shuffle=True)

# Accuracy 90.2% (5 epochs) 

_, accuracy = classifier.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))
