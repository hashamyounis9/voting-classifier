from mpi4py import MPI
import sys
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define the number of worker ranks needed
required_ranks = 5  # 1 for rank 0 (master), and 5 for the models

if rank == 0:
    if len(sys.argv) != 2:
        print("Usage: python app.py dataset.csv")
        sys.exit(1)

    datasets = ['mushrooms.csv', 'sample.csv']
    if sys.argv[1] not in datasets:
        print("Invalid dataset file")
        sys.exit(1)

    print("Loading Dataset...", sys.argv[1])
    data = pd.read_csv(sys.argv[1])

    x = data[['cap_diameter', 'cap_shape', 'gill_attachment', 'gill_color',
              'stem_height', 'stem_width', 'stem_color', 'season']].values
    y = data[['class']].values

    # Split and scale data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Send data to all worker ranks
    for r in range(1, size):
        comm.send((x_train, x_test, y_train, y_test), dest=r)

    # Receive predictions from workers
    predictions = {}
    for r in range(1, size):
        model_predictions = comm.recv(source=r)
        predictions[f"Model_{r}"] = model_predictions

    print("\nModel predictions on x_test:")
    for model, pred in predictions.items():
        print(f"{model}: {pred}")

else:
    # Receive data
    x_train, x_test, y_train, y_test = comm.recv(source=0)

    # Select model based on rank
    if rank == 1:  # SVM
        model = svm.SVC(kernel="linear", C=1.0)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

    elif rank == 2:  # ANN
        model = Sequential([
            Input(shape=(8,)),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=50, batch_size=8, verbose=0)
        predictions = (model.predict(x_test) > 0.5).astype(int).flatten()

    elif rank == 3:  # Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

    elif rank == 4:  # Logistic Regression
        model = LogisticRegression()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

    elif rank == 5:  # k-NN
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

    else:
        predictions = None

    # Send predictions back to rank 0
    comm.send(predictions.tolist(), dest=0)