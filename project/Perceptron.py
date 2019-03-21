import numpy as np
from Classifier import *
from DataManager import *
import sklearn.neural_network as sknn

#Super class for each classifier method
class Perceptron(Classifier):

	#Initialisation of the required parameters
	def __init__(self):
		#Regularization factor
		self.alpha = 0.01
		#Learning rate
		self.learning_rate = 0.01
		#Constant to multiply the hidden layer size, 100*self.layers_factor units
		self.layer_factor = 100
		#Perceptron model
		self.model = None

	"""
	Train the model with the x_train features array and the y_train one-hot labels
	Return : nothing
	"""
	def train(self, x_train, y_train):
		#Construct the model with all the hyperparameters
		self.model = sknn.MLPClassifier(hidden_layer_sizes=(100*self.layer_factor), alpha=self.alpha, learning_rate_init=self.learning_rate, max_iter=40, verbose=False, tol=0.01)
		#Train the model weights
		self.model.fit(x_train,y_train)

	"""
	Test the model with the x_test features array
	Return : an array of predicted labels, as one-hot vectors for each class
	"""
	def test(self, x_test):
		return self.model.predict_proba(x_test)

	"""
	Train/test the model with different hyper parameters values, and select the best ones
	Return : nothing
	"""
	def cross_validation(self, x_train, y_train):
		best_accuracy = 0.0
		best_alpha = 0.01
		best_learning_rate = 0.01
		best_layer_factor = 1

		#Testing combinations of hyperparameters
		for layer_factor in [2,4,8,16,32]:
			for alpha in [0.00001,0.0001,0.001,0.01,0.1,1]:
				for learning_rate in [0.0001,0.001,0.01,0.1,1]:
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
					self.layer_factor = layer_factor
					self.alpha = alpha
					self.learning_rate = learning_rate
					self.train(x_train_split,y_train_split)
					results = self.test(x_val_split)
					#Compute accuracy and compare with the best value found
					accuracy = self.accuracy(results, y_val_split)
					print("layer_factor =",layer_factor,", alpha =",alpha,", learning_rate =",learning_rate,", Accuracy = ",accuracy)
					if accuracy > best_accuracy:
						best_accuracy = accuracy
						best_layer_factor = layer_factor
						best_alpha = alpha
						best_learning_rate = learning_rate
		#Select the best value found
		self.layer_factor = best_layer_factor
		self.alpha = best_alpha
		self.learning_rate = best_learning_rate
		print("Best values : layer_factor =",self.layer_factor,", alpha =",self.alpha,", learning_rate=",self.learning_rate)
		self.train(x_train,y_train)

dm = DataManager()
dm.load_CSV("leaf-classification/train.csv", "leaf-classification/test.csv")
s = Perceptron()
perm = np.random.permutation(dm.x_train.shape[0])
x = dm.x_train[perm]
y = dm.y_train[perm]
x_train_split = x[:int(0.8*x.shape[0])]
y_train_split = y[:int(0.8*x.shape[0])]
x_val_split = x[int(0.8*x.shape[0]):]
y_val_split = y[int(0.8*y.shape[0]):]
s.cross_validation(x_train_split, y_train_split)
print("Accuracy training :", s.accuracy(s.test(x_val_split), y_val_split))
