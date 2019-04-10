import numpy as np
from Classifier import *
from DataManager import *
from sklearn.ensemble import RandomForestClassifier

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
		best_accuracy = 0
		max_k = 10
		splits_length = int(x_train.shape[0]/10)
		for estimators in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
			self.estimators = estimators
			accuracy_mean = 0.0
			logloss_mean = 0.0
			for K in range(max_k):
				x_validation_fold = x_train[int(K*splits_length): int((K+1)*splits_length)]
				x_train_fold = np.concatenate((x_train[:int(K*splits_length)],x_train[int((K+1)*splits_length):]), axis=0)
				y_validation_fold = y_train[int(K*splits_length): int((K+1)*splits_length)]
				y_train_fold = np.concatenate((y_train[:int(K*splits_length)],y_train[int((K+1)*splits_length):]), axis=0)
				self.train(x_train_fold, y_train_fold)
				y_predict = self.test(x_validation_fold)
				accuracy_mean += self.accuracy(y_predict, y_validation_fold)
				logloss_mean += self.log_loss(y_predict, y_validation_fold)
			accuracy_mean /= 10
			logloss_mean /= 10
			print("estimators =",estimators,", Validation Accuracy =",accuracy_mean,", logloss =",logloss_mean)

			if accuracy_mean > best_accuracy:
				best_accuracy = accuracy_mean
				best_estimator = estimators

		#Select the best value found
		self.estimators = best_estimator
		print("Best values : estimators =",self.estimators)
		self.train(x_train, y_train)

if __name__ == "__main__":
	rf = RandomForest()
	dm = DataManager()
	dm.load_CSV("leaf-classification/train.csv","leaf-classification/test.csv")
	x = dm.x_train
	y = dm.y_train_integer
	rf.cross_validation(x, y)
