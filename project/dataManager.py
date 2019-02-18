import csv
import numpy as np

class DataManager():

	def __init__(self):
		#Lists of leaf features examples and labels for training
		self.x_train = np.array([])
		self.y_train = np.array([])
		#Lists of leaf features examples without labels for testing
		self.x_test = np.array([])

	"""
	Reads the csv files located at path_train and path_test, and stores them in the class variables defined above
	"""
	def load_CSV(self, path_train, path_test):
		print("------------------------")
		print("Loading training data...")
		#First part : training data
		with open(path_train) as csv_file_train:
			csv_file_train_reader = csv.reader(csv_file_train)
			#Skip header
			next(csv_file_train_reader)
			
			#list of the 192 features
			features = []
			#list of the leaf labels (string labels) corresponding to the features
			labels_string = []
			for row in csv_file_train_reader:
				labels_string.append(row[1])
				features.append(row[2:])
			self.x_train = np.array(features)
			
			#Establish a link between leaf names and unique assigned ids
			unique_labels = np.unique(labels_string)
			#Leaf labels (but this time converted to one-hot vectors) corresponding to the features
			self.y_train = np.zeros((len(features),
			unique_labels.shape[0]))
			for l in range(len(labels_string)):
				self.y_train[l][np.where(unique_labels==labels_string[l])[0]] = 1
		print("-> " + str(self.x_train.shape[0]) + " training examples loaded")
	
		print("Loading testing data...")
		#Second part : testing data
		with open(path_test) as csv_file_test:
			csv_file_test_reader = csv.reader(csv_file_test)
			#Skip header
			next(csv_file_test_reader)

			#list of the 192 features
			features = []
			for row in csv_file_test_reader:
				features.append(row[1:])
			self.x_test = np.array(features)
		print("-> " + str(self.x_test.shape[0]) + " testing examples loaded")
		print("------------------------")

dm = DataManager()
dm.load_CSV("leaf-classification/train.csv", "leaf-classification/test.csv")
