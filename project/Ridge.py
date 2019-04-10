import numpy as np
from Classifier import *
from DataManager import *
import sklearn.linear_model as sklm

#Super class for each classifier method
class Ridge(Classifier):

	#Initialisation of the required parameters
	def __init__(self):
		#Regularization factor, for L2 penalty
		self.alpha = 0.1

		#Ridge model
		self.model = None

	"""
	Train the model with the x_train features array and the y_train one-hot labels
	Return : nothing
	"""
	def train(self, x_train, y_train):
		self.model = sklm.Ridge(alpha=self.alpha)
		self.model.fit(x_train,y_train)

	"""
	Test the model with the x_test features array
	Return : an array of predicted labels
	"""
	def test(self, x_test):
		return self.model.predict(x_test)

	"""
	Train/test the model with different hyper parameters values, and select the best ones
	Return : nothing
	"""
	def cross_validation(self, x_train, y_train):
		best_accuracy = 0.0
		best_alpha = 0.01

		splits_length = int(x_train.shape[0]/10)

		#Testing combinations of hyperparameters
		for alpha in [0.000000001, 0.00000001, 0.0000001, 0.000001,0.000005,0.00001,0.00005,0.0001,0.0005,0.001,0.01,0.1,1]:
			#Test the model with hyperparameters values
			self.alpha = alpha

			accuracy_mean = 0.0
			for K in range(10):
				#Split the data into 10 folds and use the #K for validation
				x_validation_fold = x_train[int(K*splits_length) : int((K+1)*splits_length)]
				x_train_fold = np.concatenate((x_train[:int(K*splits_length)],x_train[int((K+1)*splits_length):]), axis=0)
				y_validation_fold = y_train[int(K*splits_length) : int((K+1)*splits_length)]
				y_train_fold = np.concatenate((y_train[:int(K*splits_length)],y_train[int((K+1)*splits_length):]), axis=0)
				#Test the model with the value of k and output a distribution
				self.train(x_train_fold,y_train_fold)
				results = self.test(x_validation_fold)
				#Compute accuracy and compare with the best value found
				accuracy_mean += self.accuracy(results, y_validation_fold)
			accuracy_mean /= 10

			print("alpha =",alpha,", Accuracy = ",accuracy_mean)
			if accuracy_mean > best_accuracy:
				best_accuracy = accuracy_mean
				best_alpha = alpha
		#Select the best value found
		self.alpha = best_alpha
		print("Best values : alpha =",self.alpha)
		self.train(x_train,y_train)


dm = DataManager()
dm.load_CSV("leaf-classification/train.csv", "leaf-classification/test.csv")
r = Ridge()
x = dm.x_train
y = dm.y_train
r.cross_validation(x, y)
print("Accuracy training :", r.accuracy(r.test(x), y))
