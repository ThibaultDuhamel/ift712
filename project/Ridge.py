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

		#Testing combinations of hyperparameters
		for alpha in [0.000000001, 0.00000001, 0.0000001, 0.000001,0.000005,0.00001,0.00005,0.0001,0.0005,0.001,0.01,0.1,1]:
			#Random permutation
			perm = np.random.permutation(x_train.shape[0])
			x = x_train[perm]
			y = y_train[perm]
			#80% for training and 20% for validation
			x_train_split = x[:int(0.8*x.shape[0])]
			y_train_split = y[:int(0.8*x.shape[0])]
			x_val_split = x[int(0.8*x.shape[0]):]
			y_val_split = y[int(0.8*y.shape[0]):]

			#Test the model with hyperparameters values
			self.alpha = alpha
			self.train(x_train_split,y_train_split)
			results = self.test(x_val_split)
			#Compute accuracy and compare with the best value found
			accuracy = self.accuracy(results, y_val_split)
			print("alpha =",alpha,", Accuracy = ",accuracy)
			if accuracy > best_accuracy:
				best_accuracy = accuracy
				best_alpha = alpha
		#Select the best value found
		self.alpha = best_alpha
		print("Best values : alpha =",self.alpha)
		self.train(x_train,y_train)


dm = DataManager()
dm.load_CSV("leaf-classification/train.csv", "leaf-classification/test.csv")
r = Ridge()
perm = np.random.permutation(dm.x_train.shape[0])
x = dm.x_train[perm]
y = dm.y_train[perm]
x_train_split = x[:int(0.8*x.shape[0])]
y_train_split = y[:int(0.8*x.shape[0])]
x_val_split = x[int(0.8*x.shape[0]):]
y_val_split = y[int(0.8*y.shape[0]):]
r.cross_validation(x_train_split, y_train_split)
print("Accuracy training :", r.accuracy(r.test(x_val_split), y_val_split))
