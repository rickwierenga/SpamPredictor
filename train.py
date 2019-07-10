#! /usr/local/bin/ python3.7
import os
import sys

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

data_path = 'data/features.npz'

if __name__ == '__main__':
    # Check if training data is present.
    if not os.path.isfile(data_path):
        print('Please download the data and extract the features first')
        sys.exit(1)

    # Load the training data and take 20% for testing.
    print('Loading training data')
    data = np.load(data_path)['data']
    X = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train the svm.
    print('Training support vector machine. This can take a while.')
    spam_predictor = SVC(C=0.1, kernel='linear')
    spam_predictor.fit(X_train, y_train)

    # Evaluate the model.
    predictions = spam_predictor.predict(X_train)
    print('Training accuracy: {:2f}%'.format(np.mean(predictions == y_train) * 100))

    predictions = spam_predictor.predict(X_test)
    print('Training accuracy: {:2f}%'.format(np.mean(predictions == y_test) * 100))
