import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Classifier import *
from DataManager import *
from DecisionTree import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,classification_report
from sklearn.externals.six.moves import zip
from sklearn.svm import SVC


#Super class for each classifier method
class AdaBoost(Classifier):
	
	#Initialisation of the required parameters
	def __init__(self):
		#Regularization factor
		self.estimator = 5000
		self.l_rate = 1
		self.model = None

	"""
	Train the model with the x_train features array and the y_train one-hot labels
	Return : nothing
	"""
	def train(self, x_train, y_train, dtree):
		self.model = AdaBoostClassifier(base_estimator= dtree, n_estimators=self.estimator,learning_rate=self.l_rate,random_state=42)
		self.model.fit(x_train, y_train)
		#Train the model weights
		
		
	"""
	Test the model with the x_test features array
	Return : an array of predicted labels, as one-hot vectors for each class
	"""
	def test(self, x_test):
		return self.model.predict(x_test)
		

	"""
	Train/test the model with different hyper parameters values, and select the best ones
	Return : nothing
	"""	
	def cross_validation(self, x_train, y_train):
		best_estimator= 0
		best_l_rate= 0
		best_score = 0
		max_k = 10
		splits_length = int(x_train.shape[0]/10)
		'''Todo
		dTree = DecisionTree()
		hyperP = dTree.cross_validation(x_train, y_train)
		
		hyperP = {}
		hyperP.criterion = dTree.criterion
		hyperP.depth = dTree.depth
		hyperP.leaf = dTree.leaf
		'''
		
		for estimator in [50, 500, 1000, 2000, 5000]:
			for l_rate in [1,2]:
				self.estimator = estimator
				self.l_rate = l_rate
				for K in range(max_k):
					#clf = DecisionTreeClassifier(criterion=hyperP[0],min_samples_split=2,max_depth=hyperP[1],min_samples_leaf=hyperP[2])	
					clf = DecisionTreeClassifier(criterion='gini',min_samples_split=2,max_depth=11,min_samples_leaf=1)
					#svc=SVC(probability=True, kernel='rbf')
					x_validation_fold = x_train[int(K*splits_length) : int((K+1)*splits_length)]
					x_train_fold = np.concatenate((x_train[:int(K*splits_length)],x_train[int((K+1)*splits_length):]), axis=0)
					y_validation_fold = y_train[int(K*splits_length) : int((K+1)*splits_length)]
					y_train_fold = np.concatenate((y_train[:int(K*splits_length)],y_train[int((K+1)*splits_length):]), axis=0)					
					self.train(x_train_fold, y_train_fold, clf)
					y_prodict = self.test(x_validation_fold)
					
					print ("\nAdaBoost - Train Confusion Matrix\n\n",pd.crosstab(y_validation_fold,y_prodict,rownames = ["Actuall"],colnames = ["Predicted"]))      
					print ("\nAdaBoost  - Train accuracy",round(accuracy_score(y_validation_fold,y_prodict),3))
					print ("\nAdaBoost  - Train Classification Report\n",classification_report(y_validation_fold,y_prodict))
					score = accuracy_score(y_validation_fold,y_prodict)
					if best_score < score:
						best_score = score
						best_estimator = estimator
						best_l_rate = l_rate
					print("accuracy score:",best_score)
					print("best_estimator: %s,best_l_rate:%d"%(best_estimator,best_l_rate))
	'''
		best_accuracy = 0.0
		print("The accuracy is: ", best_accuracy)
	'''


if __name__ == "__main__":
	dm = DataManager()
	dm.load_CSV("leaf-classification/train.csv","leaf-classification/test.csv")	
	ada = AdaBoost()
	x = dm.x_train
	y = dm.y_train
	ada.cross_validation(x, y)
	
 		