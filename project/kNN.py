import numpy as np
from Classifier import *
from DataManager import *

class kNN(Classifier):

	def __init__(self):
		#Number of exemples to take into account for minimal distance voting
		self.k = 1
		#Training features to compare with
		self.x_train = np.array([])
		#Training labels to compare with
		self.y_train = np.array([])

	"""
	Train the model with the x_train features array and the y_train one-hot labels
	Return : nothing
	"""
	def train(self, x_train, y_train):
		#There is no training part for this method, except saving a set of examples that will vote during testing
		self.x_train = x_train
		self.y_train = y_train

	"""
	Test the model with the x_test features array
	Return : an array of predicted labels
	"""
	def test(self, x_test):
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
			#If there is a tie, select a random label amongst the ones with min distance
			indices_of_max_votes = np.argwhere(counts==np.max(counts))
			vote = unique_labels[indices_of_max_votes[np.random.randint(indices_of_max_votes.shape[0])]]
			result_labels.append(vote[0])

		return np.array(result_labels)

	"""
	Find the best k by cross validating on 10-folds
	Return : nothing
	"""
	def cross_validation(self, x_train, y_train):
		max_k = 10
		#Size of a fold.
		splits_length = int(x_train.shape[0]/10)

		best_accuracy = 0.0
		best_k = 1

		#k is the hyperparameters, K is the number of folds
		for k in range(max_k):
			self.k = k+1
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
			print("k =",k+1,", Validation Accuracy =", accuracy_mean)
			if accuracy_mean > best_accuracy:
				best_accuracy = accuracy_mean
				best_k = k+1
		#Select the best value found
		self.k = best_k

dm = DataManager()
dm.load_CSV("leaf-classification/train.csv", "leaf-classification/test.csv")
knn = kNN()
perm = np.random.permutation(dm.x_train.shape[0])
knn.cross_validation(dm.x_train[perm], dm.y_train_strings[perm])
print("")
print("Best k =",knn.k)
