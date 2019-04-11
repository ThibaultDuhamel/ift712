from Classifier import Classifier
from DataManager import DataManager
from AdaBoost import AdaBoost
from RandomForest import RandomForest
from SVM import SVM
from kNN import kNN
from Perceptron import Perceptron
from Ridge import Ridge
import sys
import warnings


warnings.filterwarnings('ignore')
"""
Usage : main.py [method] [cv] [cn] [f] [w]
method can be AdaBoost, RandomForest, SVM, kNN, Perceptron, Ridge
cv can be 1 for cross-validation or 0 otherwise
cn can be 1 to use centered/normalize data, or 0 otherwise
f can be 1 to use images features, or 0 otherwise
w can be 1 to write a submission file or 0 otherwise
"""

# Check number of parameters
if len(sys.argv) != 6:
    print("Wrong arguments")
    print("Usage : main.py [method] [cv] [cn] [f] [w]")
    print("cv can be 1 for cross-validation or 0 otherwise")
    print("cn can be 1 to use centered/normalize data, or 0 otherwise")
    print("f can be 1 to use images features, or 0 otherwise")
    print("w can be 1 to write a submission file or 0 otherwise")
    exit()

method = sys.argv[1]
cv = sys.argv[2]
cn = sys.argv[3]
f = sys.argv[4]
w = sys.argv[5]

# If method is incorrect, exit
if method not in ["AdaBoost", "RandomForest", "SVM",
                  "kNN", "Perceptron", "Ridge"]:
    print("Wrong methods. Possible values:")
    print("AdaBoost, RandomForest, SVM, kNN, Perceptron, Ridge")
    exit()

# If cv is incorrect, exit
if cv not in ['0', '1']:
    print("Wrong cross-validation parameter. Possible values:")
    print("1 = cross-validation, 0 = no cross-validation")
    exit()

# If cn is incorrect, exit
if cn not in ['0', '1']:
    print("Wrong cross-validation parameter. Possible values:")
    print("1 = center/normalize data, 0 = raw data")
    exit()

# If f is incorrect, exit
if f not in ['0', '1']:
    print("Wrong cross-validation parameter. Possible values:")
    print("1 = add image features, 0 = no image features")
    exit()

# If w is incorrect, exit
if w not in ['0', '1']:
    print("Wrong w parameters. Possible values:")
    print("1 = write submission file, 0 = no submission file")
    exit()

# Load leaf data
dm = DataManager()
dm.load_CSV("leaf-classification/train.csv",
            "leaf-classification/test.csv")
# Extract image features if needed
if f == '1':
    dm.extract_features_images("leaf-classification/images/")
# Center/normalize if needed
if cn == '1':
    dm.center_normalize_data()

x = dm.x_train
y = None
classifier = None
# Select classifier, and corresponding labels
if method == "AdaBoost":
    y = dm.y_train_integer
    classifier = AdaBoost()
elif method == "RandomForest":
    y = dm.y_train_integer
    classifier = RandomForest()
elif method == "SVM":
    y = dm.y_train_strings
    classifier = SVM()
elif method == "kNN":
    y = dm.y_train_strings
    classifier = kNN()
elif method == "Ridge":
    y = dm.y_train
    classifier = Ridge()
elif method == "Perceptron":
    y = dm.y_train
    classifier = Perceptron()

# If cv=1, cross-validate
# If cv=0, train with default parameters
if cv == '1':
    print("Cross-validating model " + method)
    classifier.cross_validation(x, y)
else:
    print("Training model " + method + " with default hyperparameters")
    classifier.train(x, y)

# Finally, assessing train accuracy
print("Evaluating model")
train_predict = classifier.test(x)
train_accuracy = classifier.accuracy(train_predict, y)
train_logloss = classifier.log_loss(train_predict, y)
print("Training accuracy =", train_accuracy, "%")
print("Training logloss =", train_logloss)

# If needed, write submission file
if w == '1':
    dm.write_CSV("test_results.csv", classifier.test(dm.x_test))
