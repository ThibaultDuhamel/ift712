#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 12:45:40 2019

@author: shih3801
"""

import numpy as np
from Classifier import *
from DataManager import *
from DecisionTree import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score


# Super class for each classifier method
class AdaBoost(Classifier):
    '''Use AdaBoostClassifier to adjust the model'''

    def __init__(self):
        """Initialisation of the required parameters"""

        self.estimator = 5000
        self.l_rate = 1
        self.model = None

    def train(self, x_train, y_train, dtree):
        """Train the model with the x_train features and
        the y_train integer encoding
            Return : nothing"""

        self.model = AdaBoostClassifier(base_estimator=dtree,
                                        n_estimators=self.estimator,
                                        learning_rate=self.l_rate,
                                        random_state=42)
        self.model.fit(x_train, y_train)

    def test(self, x_test):
            """
            Test the model with the x_test features array
            Return : an array of predicted labels,_
            as integer encoding for each class
            """
            return self.model.predict(x_test)

    def cross_validation(self, x_train, y_train):
        """
        Train/test the model with different hyper parameters values,_
        and select the best ones
        Return : nothing
        """
        best_estimator = 0
        best_l_rate = 0
        best_score = 0
        max_k = 10
        splits_length = int(x_train.shape[0] / 10)

        dTree = DecisionTree()    # DecisionTree
        # Read DecisionTree class and get the best hyperparameters for
        hyperP = dTree.cross_validation(x_train, y_train)
        for estimator in [50, 500, 1000, 2000, 5000]:
            for l_rate in [1, 2]:
                self.estimator = estimator
                self.l_rate = l_rate
                for K in range(max_k):
                    # Use trained model of DecisionTree as
                    # the parameter of Adaboost
                    clf = DecisionTreeClassifier(
                        criterion=hyperP[0], min_samples_split=2,
                        max_depth=hyperP[1], min_samples_leaf=hyperP[2])
                    x_validation_fold = x_train[
                        int(K * splits_length):int((K+1) * splits_length)]
                    x_train_fold = np.concatenate(
                        (x_train[:int(K * splits_length)],
                            x_train[int((K + 1) * splits_length):]),
                        axis=0)
                    y_validation_fold = y_train[
                        int(K * splits_length):int((K + 1) * splits_length)]
                    y_train_fold = np.concatenate(
                        (y_train[:int(K * splits_length)],
                            y_train[int((K + 1) * splits_length):]),
                        axis=0)
                    # Train Adaboost
                    self.train(x_train_fold, y_train_fold, clf)
                    # Get the predict leaf type
                    y_predict = self.test(x_validation_fold)

                    score = accuracy_score(y_validation_fold, y_predict)
                    if best_score < score:
                        best_score = score
                        best_estimator = estimator
                        best_l_rate = l_rate
            print("accuracy score:%f, best_estimator: %s, best_l_rate:%d" %
                 (best_score, best_estimator, best_l_rate))


if __name__ == "__main__":
    dm = DataManager()
    dm.load_CSV("leaf-classification/train.csv",
                "leaf-classification/test.csv")
    ada = AdaBoost()
    x = dm.x_train
    # Leaf labels - Integer Encoding
    y = dm.y_train_integer
    ada.cross_validation(x, y)