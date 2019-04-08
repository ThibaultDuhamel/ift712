import numpy as np
import sklearn as sk

#Super class for each classifier method
class Classifier:

	#Initialisation of the required parameters
	def __init__(self):
		pass

	"""
	Train the model with the x_train features array and the y_train one-hot labels
	Return : nothing
	"""
	def train(self, x_train, y_train):
		pass

	"""
	Test the model with the x_test features array
	Return : an array of predicted labels, as probabilities for each class
	"""
	def test(self, x_test):
		pass

	"""
	Train/test the model with different hyper parameters values, and select the best ones
	Return : nothing
	"""
	def cross_validation(self, x_train, y_train):
		pass

	"""
	Compute the percentage of correct predictions over the total number of predictions (100*correct/total).
	The two inputs must be arrays of probabilities
	Return : the percentage of correct predictions, a float from 0 to 100
	"""
	def accuracy(self, predicted_labels, true_labels):
		accuracy = 0.0

		#First version is for string labels
		if isinstance(predicted_labels[0], str):
			for predicted,true in zip(predicted_labels,true_labels):
				if predicted==true:
					accuracy+=1

		#Second version if for one-hot labels
		else:
			for predicted,true in zip(predicted_labels,true_labels):
				if np.argmax(predicted) == np.argmax(true):
					accuracy+=1

		return 100*accuracy/predicted_labels.shape[0]

	def log_loss(self, predicted_labels, true_labels):
		log_loss = 0.0

		#First version is for string labels
		if isinstance(predicted_labels[0], str):
			for predicted,true in zip(predicted_labels,true_labels):
				if predicted!=true:
					log_loss-=np.log(10e-15)
			return log_loss/predicted_labels.shape[0]
		#Second version if for one-hot labels
		else:
			return sk.metrics.log_loss(true_labels, predicted_labels)
