import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
from mpi4py import MPI

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


def voting_simulation(cumulative_predictions):  
    final_output = []

    for i in range(0, len(cumulative_predictions[0])):
        count_one = 0
        count_zero = 0

        d = []
        for x in range(0, len(cumulative_predictions)):
            d.append(cumulative_predictions[x][i])

        for j in range(0,len(d)):
            if d[j] == 1:
                count_one = count_one+1
            else:
                count_zero = count_zero+1
        if count_one > count_zero:
            final_output.append(1)
        else:
            final_output.append(0)
    return final_output

def make_pretty(predictions):
    pretty_predictions = []
    for i in range(len(predictions)):
        if predictions[i] == 1:
            pretty_predictions.append("⚫")
        else:
            pretty_predictions.append("🟡")
    return pretty_predictions


def simple_formatting(predictions, final_output):

    print("+{:-^6}+{:-^6}+{:-^6}+{:-^6}+".format("", "", "", "")) 
    print("|{:^6}|{:^6}|{:^6}|{:^6}|".format("SVM", "ANN", "RF", "Final"))
    print("+{:-^6}+{:-^6}+{:-^6}+{:-^6}+".format("", "", "", "")) 
    final_output = make_pretty(final_output[:150])
    predictions[0] = make_pretty(predictions[0][:150])
    predictions[1] = make_pretty(predictions[1][:150]) 
    predictions[2] = make_pretty(predictions[2][:150])
    # predictions[3] = make_pretty(predictions[3][:150])
    # predictions[4] = make_pretty(predictions[4][:150])

    for i in range(0, len(predictions[0])):
        print("|{:^5}|{:^5}|{:^5}|{:^5}|"
              .format(predictions[0][i], predictions[1][i], predictions[2][i], final_output[i]))






comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    data = pd.read_csv('mushrooms.csv')

    x = data[
        [
            'cap_diameter', 
            'cap_shape', 
            'gill_attachment', 
            'gill_color', 
            'stem_height', 
            'stem_width', 
            'stem_color', 
            'season'
            ]
        ].values
    y = data[
        [
            'class'
            ]
        ].values.ravel()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    for node_number in range(1, size):
        comm.send((x_train, x_test, y_train, y_test), dest=node_number)

    predictions = []
    for node_number in range(1, size):
        model_predictions = comm.recv(source=node_number)
        predictions.append(model_predictions)
    
    svm_predictinos = predictions[0]["SVM"]
    ann_predictions = predictions[1]["ANN"]
    rf_predictions = predictions[2]["RandomForest"]

    predictions = [svm_predictinos, ann_predictions, rf_predictions]

    final_output = voting_simulation(predictions)
    print()
    print("🟡 = Edible")
    print("⚫ = Poisonous")
    print()
    print()
    print(simple_formatting(predictions=predictions, final_output=final_output))

    print("\n\nSVM Accuracy: ", accuracy_score(y_test, svm_predictinos))
    print("SVM Confusion Matrix: \n", confusion_matrix(y_test, svm_predictinos))
    print("\nSVM Classification Report: \n", classification_report(y_test, svm_predictinos))

    print("\n\nANN Accuracy: ", accuracy_score(y_test, ann_predictions))
    print("ANN Confusion Matrix: \n", confusion_matrix(y_test, ann_predictions))
    print("\nANN Classification Report: \n", classification_report(y_test, ann_predictions))

    print("\n\nRF Accuracy: ", accuracy_score(y_test, rf_predictions))
    print("RF Confusion Matrix: \n", confusion_matrix(y_test, rf_predictions))
    print("\nRF Classification Report: \n", classification_report(y_test, rf_predictions))

    print("\n\nFinal Accuracy: ", accuracy_score(y_test, final_output))
    print("Final Confusion Matrix: \n", confusion_matrix(y_test, final_output))
    print("\nSVM Classification Report: \n", classification_report(y_test, final_output))
    # first combo: 0.94
    # second combo: 0.96
    # third combo: 0.95


else:

    if rank == 1:
        x_train, x_test, y_train, y_test = comm.recv(source=0)
        model = svm.SVC(kernel="linear", C=1.0)

        # C=0.1 accuracy: 0.64
        # C=1.0 accuracy: 0.65
        # C=10.0 accuracy: 0.64

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        prediction_report = {"SVM": predictions.tolist()}
        comm.send(prediction_report, dest=0)

    elif rank == 2:
        x_train, x_test, y_train, y_test = comm.recv(source=0)
        model = Sequential([
            Input(shape=(8,)),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=100, batch_size=16, verbose=0)

        # epochs=50, batch_size=8, accuracy: 0.90
        # epochs=100, batch_size=16, accuracy: 0.93
        # epochs=150, batch_size=32, accuracy: 0.93

        predictions = (model.predict(x_test) > 0.5).astype(int).flatten()
        prediction_report = {"ANN": predictions.tolist()}
        comm.send(prediction_report, dest=0)

    elif rank == 3:
        x_train, x_test, y_train, y_test = comm.recv(source=0)
        model = RandomForestClassifier(n_estimators=200)

        # n_estimators=50, accuracy: 0.98
        # n_estimators=100, accuracy: 0.98
        # n_estimators=200, accuracy: 0.99

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        prediction_report = {"RandomForest": predictions.tolist()}
        comm.send(prediction_report, dest=0)