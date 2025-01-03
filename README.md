
# Voting Classifier

The **Voting Classifier** is an [Ensemble Learning](https://en.wikipedia.org/wiki/Ensemble_learning) technique where multiple [Classification Algorithms](https://datascientest.com/en/classification-algorithms-definition-and-main-models) are combined to improve the final prediction. This approach can lead to better performance by aggregating the outputs of several individual models. Below is a diagram illustrating how a voting classifier works:

<p align="center">
  <img src="voting_classifier.png" alt="Voting Classifier" />
</p>

## Voting using MPI

<p align="center">
  <img src="architecture.jpeg" />
</p>

## Features
- Combines multiple classification models for better prediction results.
- Demonstrates the use of ensemble learning techniques.

## Installation

To get started with the Voting Classifier, follow the steps below:

### 1. Clone the Repository
```bash
git clone https://github.com/hashamyounis9/voting-classifier.git
```

### 2. Set Up Locally

Navigate to the project directory:
```bash
cd voting-classifier
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Classifier

### 1. Running the Application
To run the Voting Classifier, use the following command:

```bash
python app.py sample.csv
```

### 2. Sample Data

Note: `sample.csv` is used during development and contains a subset of the actual dataset `mushrooms.csv`. You can replace `sample.csv` with your own dataset for further testing.
