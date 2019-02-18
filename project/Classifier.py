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
