#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 12:45:40 2019

@author: shih3801
"""

import numpy as np
from Classifier import *
from DataManager import *
from sklearn.ensemble import RandomForestClassifier

# Super class for each classifier method
class RandomForest(Classifier):
    """
    Using RandomForestClassifier to train Random Forest model
    """
    def __init__(self):
        self.estimators = 10
        self.cri_par = 'gini'

    def train(self, x_train, y_train):
        """
        Train the model with the x_train features array and
        the y_train one-hot labels
        Return : nothing
        """
        # Construct the model with all the hyperparameters
        self.model = RandomForestClassifier(
                        n_estimators=self.estimators, criterion=self.cri_par)
        # Train the model weights
        self.model.fit(x_train, y_train)

    def test(self, x_test):
        """
        Test the model with the x_test features array
        Return : an array of predicted labels,
        as one-hot vectors for each class
        """
        return self.model.predict(x_test)

    def cross_validation(self, x_train, y_train):
        """
        Train/test the model with different hyper parameters values,
        and select the best ones
        Return : nothing
        """
        best_estimator = 0
        best_cri_par = 'gini'
        best_accuracy = 0
        max_k = 10
        splits_length = int(x_train.shape[0]/10)
        for criterion in ['gini', 'entropy']:
            for estimator in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                self.estimators = estimator
                self.cri_par = criterion
                accuracy_mean = 0.0
                logloss_mean = 0.0
                for K in range(max_k):
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
                    accuracy_mean += self.accuracy(y_predict,
                                                   y_validation_fold)
                    logloss_mean += self.log_loss(y_predict, y_validation_fold)
                accuracy_mean /= 10
                logloss_mean /= 10
                print("estimator =", estimator, ", criterion =", criterion,
                      ", Validation Accuracy =", accuracy_mean,
                      ",logloss =", logloss_mean)
                if accuracy_mean > best_accuracy:
                    best_accuracy = accuracy_mean
                    best_estimator = estimator
                    best_cri_par = criterion

        # Select the best value found
        self.estimators = best_estimator
        self.cri_par = best_cri_par
        print("Best values : estimators =", self.estimators,
              ", criterion =", self.cri_par,
              ", Validation Accuracy =", best_accuracy)
        self.train(x_train, y_train)

if __name__ == "__main__":
    rf = RandomForest()
    dm = DataManager()
    dm.load_CSV("leaf-classification/train.csv","leaf-classification/test.csv") 
    x = dm.x_train
    y = dm.y_train_integer
    rf.cross_validation(x, y)