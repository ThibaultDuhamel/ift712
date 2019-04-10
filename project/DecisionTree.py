#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 12:45:40 2019

@author: shih3801
"""

import numpy as np
from Classifier import *
from DataManager import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Super class for each classifier method
class DecisionTree(Classifier):
    # Initialisation of the required parameters
    def __init__(self):
        self.criterion = 'gini'
        self.depth = 1
        self.leaf = 1
        self.model = None

    def train(self, x_train, y_train):
        """Train the model with the x_train features array and the y_train_
        integer encoding
        Return : nothing
        """
        # Construct the model with hyperparameters
        self.model = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.depth,
            min_samples_split=2,
            min_samples_leaf=self.leaf,
            random_state=42)
        self.model.fit(x_train, y_train)    # Train the model weights

    def test(self, x_test):
        """Test the model with the x_test features array
        Return : an array of predicted labels, as integer vector for each class
        """
        return self.model.predict(x_test)

    def cross_validation(self, x_train, y_train):
        """
        Train/test the model with different hyper parameters values,
        and select the best ones
        """
        best_cri_par = 0
        best_dep_par = 0
        best_leaf_par = 0
        best_score = 0
        max_k = 10
        splits_length = int(x_train.shape[0] / 10)

        for cri_par in ['gini', 'entropy']:
            for leaf_par in range(1, 3):
                for dep_par in range(7, 12):
                    self.criterion = cri_par
                    self.depth = dep_par
                    self.leaf = leaf_par
                    for K in range(max_k):
                        x_validation_fold = x_train[int(
                            K * splits_length): int((K + 1) * splits_length)]
                        x_train_fold = np.concatenate(
                            (x_train[:int(K * splits_length)],
                             x_train[int((K + 1) * splits_length):]), axis=0)
                        y_validation_fold = y_train[int(
                            K * splits_length): int((K + 1) * splits_length)]
                        y_train_fold = np.concatenate(
                            (y_train[:int(K * splits_length)],
                             y_train[int((K + 1) * splits_length):]), axis=0)
                        # Train the model with new parameters
                        self.train(x_train_fold, y_train_fold)
                        # Test the model with the value of k and output a
                        y_predict = self.test(x_validation_fold)

                        pd.crosstab(
                            y_validation_fold,
                            y_predict,
                            rownames=["Actuall"],
                            colnames=["Predicted"])
                        # Use real y-test data and y-prodict data to
                        # compare and get the accuracy
                        score = accuracy_score(y_validation_fold, y_predict)

                    if best_score < score:
                        best_score = score
                        best_cri_par = cri_par
                        best_dep_par = dep_par
                        best_leaf_par = leaf_par
                print("ccuracy score: %f, best_cri_par: %s, best_dep_par:%d,\
                    best_leaf_par:%d " % (best_score, best_cri_par,
                                          best_dep_par, best_leaf_par))
        return (best_cri_par, best_dep_par, best_leaf_par)
