import numpy as np
from sklearn import svm
from Classifier import *
from DataManager import *

#Super class for each classifier method
class SVM(Classifier):

	#Initialisation of the required parameters
	def __init__(self):
		#SVM penalty parameter, to control the softness of the separation
		self.C = 1.0
		#SVM kernel, which can be linear, rbf, sigmoidal, poly
		self.kernel = "linear"

		#If kernel == rbf or poly or sigmoid, control the kernel coefficient
		self.gamma = 0.1

		#If kernel == poly, control the polynom degree
		self.degree = 1

		#Build the SVM model with sklearn
		self.model = None

	"""
	Train the model with the x_train features array and the y_train one-hot labels
	Return : nothing
	"""
	def train(self, x_train, y_train):
		#Construct the model with all the hyperparameters, even though some may not be used
		self.model = svm.SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, degree=self.degree)
		#Train the model
		self.model.fit(x_train,y_train)

	"""
	Test the model with the x_test features array
	Return : an array of predicted labels, as label for each class
	"""
	def test(self, x_test):
		return self.model.predict(x_test)

	"""
	Train/test the model with different hyper parameters values, and select the best ones
	Return : nothing
	"""
	def cross_validation(self, x_train, y_train):
		pass

dm = DataManager()
dm.load_CSV("leaf-classification/train.csv", "leaf-classification/test.csv")
s = SVM()
s.train(dm.x_train, dm.y_train_strings)
print("Accuracy training :", s.accuracy(s.test(dm.x_train), dm.y_train_strings))
print("Best C =",s.C)
