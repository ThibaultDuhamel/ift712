import numpy as np
from sklearn import svm
from Classifier import *
from DataManager import *

#Super class for each classifier method
class SVM(Classifier):

	#Initialisation of the required parameters
	def __init__(self):
		#SVM penalty parameter, to control the softness of the separation
		self.C = 1
		#SVM kernel, which can be linear, rbf, sigmoidal, poly
		self.kernel = "linear"

		#If kernel == rbf or poly or sigmoid, controls the kernel coefficient
		self.gamma = 0.1

		#If kernel == rbf or poly or sigmoid, controls the independent term in the kernel function
		self.coef0 = 0

		#If kernel == poly, controls the polynom degree
		self.degree = 1

		#Build the SVM model with sklearn
		self.model = None

	"""
	Train the model with the x_train features array and the y_train one-hot labels
	Return : nothing
	"""
	def train(self, x_train, y_train):
		#Construct the model with all the hyperparameters, even though some may not be used
		self.model = svm.SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, degree=self.degree, coef0=self.coef0)
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
		best_accuracy = 10.0
		best_C = 0.00001
		best_kernel = "rbf"
		#In the case of poly, rbf, sigmoid
		best_gamma = 0.001
		best_coef0 = 0
		#In the case of poly
		best_degree = 1
		splits_length = int(x_train.shape[0]/5)

		#Testing combinations of hyperparameters
		for kernel in ["linear", "rbf", "poly", "sigmoid"]:
			#We only want to search gamma and degree for specific kernels
			gamma_range = [0.1] #Default value, not used
			degree_range = [1] #Default value, not used
			coef0_range = [0] #Default value, not used
			if kernel == "rbf" or kernel=="sigmoid":
				gamma_range = [0.0001, 0.001, 0.01, 0.1, 1]
				degree_range = [1] #Default value, not used
				coef0_range = [-4,-2,0,2,4]
			elif kernel=="poly":
				gamma_range = [0.0001, 0.001, 0.01, 0.1, 1]
				degree_range = [1,2,4,8]
				coef0_range = [-4,-2,0,2,4]
			#Find hyperparameters only if necessary
			for gamma in gamma_range:
				for degree in degree_range:
					for coef0 in coef0_range:
						for C in [0.1, 1, 10, 100, 1000, 10000]:
							#Test the model with hyperparameters values
							self.C = C
							self.kernel = kernel
							self.gamma = gamma
							self.degree = degree
							self.coef0 = coef0

							accuracy_mean = 0.0
							log_loss_mean = 0.0
							for K in range(5):
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
								log_loss_mean += self.log_loss(results, y_validation_fold)
							accuracy_mean /= 5
							log_loss_mean /= 5
							print("kernel =",kernel,", gamma =",gamma,", degree =",degree,", coef0 =",coef0,", C =",C,", Validation Accuracy = ",accuracy_mean, ", Validation log_loss = ",log_loss_mean)
							if accuracy_mean > best_accuracy:
								best_accuracy = accuracy_mean
								best_C = C
								best_kernel = kernel
								best_gamma = gamma
								best_degree = degree
								best_coef0 = coef0
		#Select the best value found
		self.C = best_C
		self.kernel = best_kernel
		self.gamma = best_gamma
		self.degree = best_degree
		self.coef0 = best_coef0
		print("Best values : kernel =",self.kernel,", gamma =",self.gamma,", degree =",self.degree,", coef0 =",self.coef0,", C =",self.C)
		self.train(x_train,y_train)

dm = DataManager()
dm.load_CSV("leaf-classification/train.csv", "leaf-classification/test.csv")
dm.center_normalize_data()
s = SVM()
#s.cross_validation(dm.x_train, dm.y_train_strings)
s.train(dm.x_train,dm.y_train_strings)
print("Accuracy training :", s.accuracy(s.test(dm.x_train), dm.y_train_strings))
print("Log loss training :", s.log_loss(s.test(dm.x_train), dm.y_train_strings))

print("Testing model...")
dm.write_CSV("test_results.csv", s.test(dm.x_test))
