import numpy as np
from Classifier import *
from DataManager import *

class kNN(Classifier):

	def __init__(self, k):
		#Number of exemples to take into account for minimal distance voting
		self.k = k
		#Training features to compare with
		self.x_train = np.array([])
		#Training labels to compare with
		self.y_train = np.array([])
	
	"""
	Train the model with the x_train features array and the y_train one-hot labels
	Return : nothing
	"""
	def train(self, x_train, y_train):
		print("")
		print("Training kNN method...")
		#There is no training part for this method, except saving a set of examples that will vote during testing
		self.x_train = x_train
		self.y_train = y_train
		print("Done")

	"""
	Test the model with the x_test features array
	Return : an array of predicted labels
	"""
	def test(self, x_test):
		print("")
		print("Testing kNN method with k=" + str(self.k) + "...")
		
		nb_train_examples = self.x_train.shape[0]
		result_labels = []
		
		for x in x_test:
			#First, calculate the array of distances between x and each example in x_train
			x_matrix = np.tile(x, (nb_train_examples,1))
			distances_array = np.sum(np.power(self.x_train-x_matrix,2), axis=1)
			#Then, get the k indices for which the distances are the smallest
			indices_minimal = np.argpartition(distances_array, self.k)[:self.k]
			#Get labels associated to those indices
			votes_labels = self.y_train[indices_minimal]
			#Get the label with the highest amount of votes
			unique_labels, counts = np.unique(votes_labels, axis=0, return_counts=True)
			vote = unique_labels[np.argmax(counts)]
			result_labels.append(vote)
		
		print("Done")
		return np.array(result_labels)

dm = DataManager()
dm.load_CSV("leaf-classification/train.csv", "leaf-classification/test.csv")

knn = kNN(5)
knn.train(dm.x_train,dm.y_train)
labels = knn.test(dm.x_train)
