import sys
import numpy as np
import pandas as pd
from collections import Counter

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Calculating accuracy manually just for fun
def accuracy(y_actual, y_predicted):
    correction_predictions_count = 0
    for i in range(0, len(y_actual)):
        if y_predicted[i] == y_actual[i]:
            correction_predictions_count = correction_predictions_count + 1

    acc_score = (correction_predictions_count / len(y_actual)) * 100    
    return acc_score

if len(sys.argv) != 2:
    print("Usage: python app.py dataset.csv")
    sys.exit(1)

datasets = ['mushrooms.csv', 'sample.csv']
if sys.argv[1] not in datasets:
    print("Invalid dataset file")
    sys.exit(1)

print("Loading Dataset...", sys.argv[1])
data = pd.read_csv(sys.argv[1])

x = data[['cap_diameter','cap_shape','gill_attachment','gill_color','stem_height','stem_width','stem_color','season']].values
y = data[['class']].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
sample = np.array([[1372,2,2,10,3.8074667544799388,1545,11,1.8042727086281731]])

votes = []

#######################

# SVM Model

#######################


svm_model = svm.SVC(kernel="linear", C=1.0)
svm_model.fit(x_train, y_train)
y_predicted = svm_model.predict(x_test)

# just for better understanding as accuracy is calculated by comaring actual and predicted values
y_actual = y_test

# make prediction for single sample
prediction = svm_model.predict(sample)
votes.insert(0, prediction[0])
gender = '🟡 Edible' if prediction[0] == 1 else '⚫ Poisonous'

print(gender, "is the predicted class by SVM model")


#######################

# ANN Model

#######################


ann_model = Sequential()
ann_model.add(Input(shape=(8,)))
ann_model.add(Dense(16, input_dim=8, activation='relu'))
ann_model.add(Dense(8, activation='relu'))
ann_model.add(Dense(1, activation='sigmoid'))

ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann_model.fit(x_train, y_train, epochs=50, batch_size=8, verbose=0)


ann_model.evaluate(x_test, y_test, verbose=0)
prediction = ann_model.predict(sample)
predicted_class = (prediction > 0.5).astype(int)
votes.insert(1, prediction[0])
gender = '🟡 Edible' if prediction[0] == 1 else '⚫ Poisonous'
print(gender, "is the predicted class by ANN model")


#######################

# RandomForest Model

#######################


randomforest_model = RandomForestClassifier(n_estimators=100, random_state=42)
randomforest_model.fit(x_train, y_train)
y_predicted = randomforest_model.predict(x_test)

prediction = randomforest_model.predict(sample)
votes.insert(2, prediction[0])
gender = '🟡 Edible' if prediction[0] == 1 else '⚫ Poisonous'

print(gender, "is the predicted class by RF model")# # Calculating accuracy manually just for fun