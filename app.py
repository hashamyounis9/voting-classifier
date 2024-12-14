import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, auc
from sklearn.model_selection import train_test_split
from sklearn import svm


# generating mock data
np.random.seed(0)
sample_count  = 1000

heights_male = np.random.normal(175, 10, sample_count)
weights_male = np.random.normal(75, 15, sample_count)


heights_female = np.random.normal(160, 10, sample_count)
weights_female = np.random.normal(60, 10, sample_count)

heights = np.concatenate([heights_male, heights_female])
weights = np.concatenate([weights_male, weights_female])
labels = np.concatenate([np.ones(sample_count), np.zeros(sample_count)])

x = np.column_stack((heights, weights))
y = labels

# split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


svm_model = svm.SVC(kernel="linear", C=1.0)
svm_model.fit(x_train, y_train)

y_predicted = svm_model.predict(x_test)

# just for better understanding as accuracy is calculated by comaring actual and predicted values
y_actual = y_test

# make prediction for single sample
new_sample = np.array([[170, 65]])
prediction = svm_model.predict(new_sample)
gender = 'M' if prediction[0] == 1 else 'F'

print("Gender of new prediction is:", gender)


# Calculating accuracy manually just for fun
correction_predictions_count = 0
for i in range(0, len(y_actual)):
    if y_predicted[i] == y_actual[i]:
        correction_predictions_count = correction_predictions_count + 1

print("Total predictions are:", len(y_actual))
print("Correct predictions are:", correction_predictions_count)
acc_score = (correction_predictions_count / len(y_actual)) * 100
print("Manual Accuracy Score:", acc_score)