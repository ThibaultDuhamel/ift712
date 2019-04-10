#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 12:45:40 2019

@author: shih3801
"""

import numpy as np
import pandas as pd
from Classifier import *
from DataManager import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Super class for each classifier method
class RandomForest(Classifier):
    def __init__(self):
        self.estimators = 10
    """
    Train the model with the x_train features array and
    the y_train one-hot labels
    Return : nothing
    """
    def train(self, x_train, y_train):
        # Construct the model with all the hyperparameters
        self.model = RandomForestClassifier(n_estimators=self.estimators)
        # Train the model weights
        self.model.fit(x_train, y_train)

    """
    Test the model with the x_test features array
    Return : an array of predicted labels, as one-hot vectors for each class
    """
    def test(self, x_test):
        return self.model.predict(x_test)

    """
    Train/test the model with different hyper parameters values,
    and select the best ones
    Return : nothing
    """
    def cross_validation(self, x_train, y_train):
        best_estimator = 0
        best_score = 0
        max_k = 10
        splits_length = int(x_train.shape[0]/10)
        for K in range(max_k):
            for myEstimator in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                self.estimators = myEstimator
                x_validation_fold = x_train[
                     int(K*splits_length): int((K+1)*splits_length)]
                x_train_fold = np.concatenate(
                    (x_train[:int(K*splits_length)],
                     x_train[int((K+1)*splits_length):]), axis=0)
                y_validation_fold = y_train[
                    int(K*splits_length): int((K+1)*splits_length)]
                y_train_fold = np.concatenate(
                    (y_train[:int(K*splits_length)],
                     y_train[int((K+1)*splits_length):]), axis=0)
                self.train(x_train_fold, y_train_fold)
                y_predict = self.test(x_validation_fold)
                print ("\nAdaBoost - Train Confusion Matrix\n\n", pd.crosstab(y_validation_fold,y_predict,rownames = ["Actuall"],colnames = ["Predicted"]))      
                print ("\nAdaBoost  - Train accuracy", round(accuracy_score(y_validation_fold,y_predict),3))
                print ("\nAdaBoost  - Train Classification Report\n", classification_report(y_validation_fold,y_predict))

                score = accuracy_score(y_validation_fold, y_predict)
                print("Accuracy score is ", score)
                if best_score < score:
                    best_score = score
                    best_estimator = myEstimator
                print("accuracy score:", best_score)
                print("best_estimator: %d" % (best_estimator))

if __name__ == "__main__":
    rf = RandomForest()
    dm = DataManager()
    dm.load_CSV("leaf-classification/train.csv","leaf-classification/test.csv") 
    x = dm.x_train
    y = dm.y_train_integer
    rf.cross_validation(x, y)