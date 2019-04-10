import numpy as np
from Classifier import *
from DataManager import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


# Super class for each classifier method
class AdaBoost(Classifier):
	'''Use AdaBoostClassifier to adjust the model'''

	def __init__(self):
		"""Initialisation of the required parameters"""
		self.estimator = 5000
		self.l_rate = 1
		self.clf = DecisionTreeClassifier(criterion='gini', min_samples_split=2,max_depth=11,min_samples_leaf=1)
		self.model = None

	def train(self, x_train, y_train):
		"""Train the model with the x_train features and
		the y_train integer encoding
			Return : nothing"""
		self.model = AdaBoostClassifier(base_estimator=self.clf,
										n_estimators=self.estimator,
										learning_rate=self.l_rate)
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
		best_accuracy = 0
		max_k = 10
		splits_length = int(x_train.shape[0] / 10)

		for estimator in [25, 50, 75, 100, 200]:
			for l_rate in [0.01, 0.1, 1]:
				self.estimator = estimator
				self.l_rate = l_rate
				accuracy_mean = 0.0
				for K in range(max_k):
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
					self.train(x_train_fold, y_train_fold)
					# Get the predict leaf type
					y_predict = self.test(x_validation_fold)
					accuracy_mean += self.accuracy(y_predict, y_validation_fold)
				accuracy_mean /= 10
				print("n_estimator =",estimator,", learning_rate =",l_rate,", Validation Accuracy = ",accuracy_mean)
				if accuracy_mean > best_accuracy:
					best_accuracy = accuracy_mean
					best_estimator = estimator
					best_l_rate = l_rate

		#Select the best value found
		self.estimator = best_estimator
		self.l_rate = best_l_rate
		print("Best values : n_estimator =",estimator,", learning_rate =",l_rate,", Validation Accuracy = ",best_accuracy)
		self.train(x_train,y_train)



if __name__ == "__main__":
	dm = DataManager()
	dm.load_CSV("leaf-classification/train.csv",
				"leaf-classification/test.csv")
	ada = AdaBoost()
	x = dm.x_train
	# Leaf labels - Integer Encoding
	y = dm.y_train_integer
	ada.cross_validation(x, y)
