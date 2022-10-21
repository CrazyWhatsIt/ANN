import os, sys
import numpy as np
from sklearn import datasets

#import libraries as needed

def readDataLabels(): 
	#read in the data and the labels to feed into the ANN
	data = datasets.load_digits()
	X = data.data
	y = data.target

	return X,y

def to_categorical(y):
	#Convert the nominal y values to categorical
	result = []
	for label in y:
		assert label < 10;
		one_hot = np.zeros(10)
		one_hot[label] = 1
		result.append(one_hot)
	return np.array(result)
	
def train_test_split(X,y,n=0.8):
    # Instructions: split data in training and testing sets.
    assert len(X) == len(y)
    assert n < 1
    assert n > 0
    training_data = []
    testing_data = []
    training_labels = []
    testing_labels = []
    for index in range(0, len(X)):
        # note that the seed can be set with random.seed().
        # useful for debugging.
        roll = random.uniform(0, 1)
        if roll < n:
            training_data.append(X[index])
            training_labels.append(y[index])
        else:
            testing_data.append(X[index])
            testing_labels.append(y[index])
    training_data = np.array(training_data)
    testing_data = np.array(testing_data)
    training_labels = np.array(training_labels)
    testing_labels = np.array(testing_labels)
    return training_data, training_labels, testing_data, testing_labels

def normalize_data(data): #TODO
	# normalize/standardize the data
    data_norm = (data - np.amin(data))/(np.amax(data)-np.amin(data))
    return data_norm
