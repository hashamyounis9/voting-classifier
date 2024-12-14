import numpy as np
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn import svm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier



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
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


svm_model = svm.SVC(kernel="linear", C=1.0)
svm_model.fit(x_train, y_train)

y_predicted = svm_model.predict(x_test)

# just for better understanding as accuracy is calculated by comaring actual and predicted values
y_actual = y_test

# make prediction for single sample
new_sample = np.array([[170, 65]])
prediction = svm_model.predict(new_sample)
gender = 'M' if prediction[0] == 1 else 'F'

print("SVM Prediction:", gender)


# # Calculating accuracy manually just for fun
# correction_predictions_count = 0
# for i in range(0, len(y_actual)):
#     if y_predicted[i] == y_actual[i]:
#         correction_predictions_count = correction_predictions_count + 1

# print("Total predictions are:", len(y_actual))
# print("Correct predictions are:", correction_predictions_count)
# acc_score = (correction_predictions_count / len(y_actual)) * 100
# print("Manual Accuracy Score:", acc_score)
# auc_score = auc(y_actual, y_predicted)
# print("Manual Accuracy Score:", auc_score)






################################






scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

ann_model = Sequential([
    Dense(16, input_dim=2, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# print("Training the model...")
ann_model.fit(x_train, y_train, epochs=50, batch_size=8, verbose=0)

# print("Evaluating model")
ann_model.evaluate(x_test, y_test, verbose=0)

# new_sample = np.array([[170, 65]])
# new_sample = scaler.transform(new_sample)
prediction = ann_model.predict(new_sample)
predicted_class = (prediction > 0.5).astype(int)

gender = "M" if predicted_class[0][0] == 1 else "F"
print("ANN prediction:", gender)