import os, sys
import numpy as np
import math

from data import readDataLabels, normalize_data, train_test_split, to_categorical
from utils import accuracy_score, CrossEntropyLoss, SigmoidActivation, SoftmaxActivation

# Create an MLP with 8 neurons
# Input -> Hidden Layer -> Output Layer -> Output
# Neuron = f(w.x + b)
# Do forward and backward propagation

mode = 'train'      # train/test... Optional mode to avoid training incase you want to load saved model and test only.

class ANN:
    def __init__(self, num_input_features, num_hidden_units, num_outputs, hidden_unit_activation, output_activation, loss_function):
        self.num_input_features = num_input_features
        self.num_hidden_units = num_hidden_units
        self.num_outputs = num_outputs

        self.hidden_unit_activation = hidden_unit_activation
        self.output_activation = output_activation
        self.loss_function = loss_function


    def initialize_weights(self):   # TODO
        # Create and Initialize the weight matrices
        # Never initialize to all zeros. Not Cool!!!
        # Try something like uniform distribution. Do minimal research and use a cool initialization scheme.
        self.w1 = np.random.uniform(0, 1, size=(self.num_hidden_units, self.num_input_features)) # input_w is of shape 16 by 64.
        self.b1 = np.random.uniform(0, 1, size=self.num_hidden_units) # input_b is of shape 1 by 16.
        self.w2 = np.random.uniform(0, 1, size=(self.num_outputs, self.num_hidden_units)) # hidden_w is of shape 10 by 16.
        self.b2 = np.random.uniform(0, 1, size=self.num_outputs) # hidden_b is of shape 1 by 10.

    def forward(self, x):      # TODO
        # x = input matrix
        # hidden activation y = f(z), where z = w.x + b
        # output = g(z'), where z' =  w'.y + b'
        # Trick here is not to think in terms of one neuron at a time
        # Rather think in terms of matrices where each 'element' represents a neuron
        # and a layer operation is carried out as a matrix operation corresponding to all neurons of the layer
        self.x = x # x is of shape 1 by 64, or N by 64 for a sample of size N.
        self.z = np.dot(self.w1, self.x.T) + self.b1.T
        self.z1 = self.hidden_unit_activation(self.z)
        self.z2 = np.dot(self.w2, self.z1.T) + self.b2.T
        self.y_pred = self.output_activation(self.z2)

#    def backward(self, y_pred, y_gt):     # TODO
#        CEL = CrossEntropyLoss()
#        CEL(y_pred, y_gt)
#        dL_dy_pred = CEL.grad()
#        dy_pred_dz2 = 
#        dz2_dz1 = 
#        dz2_dw2 = 
#        dz1_dz = 
#        dz_dw1 = 
#        self.dL_dw2 = dL_dy_pred * dy_pred_dz2 * dz2_dw2
#        self.dL_dw1 = dL_dy_pred * dy_pred_dz2 * dz2_dz1 * dz1_dz * dz_dw1
#
#    def update_params(self, learning_rate=0.01):    # TODO
#        # Take the optimization step.
#        return
#
#    def train(self, x, y_gt, learning_rate=0.01, num_epochs=10):
#        for epoch in range(num_epochs):
#            self.forward(x)
#            self.backward(self.y_pred, y_gt)
#
#    def test(self, test_dataset):
#        accuracy = 0    # Test accuracy
#        # Get predictions from test dataset
#        # Calculate the prediction accuracy, see utils.py
#        return accuracy


def main(argv):
    ann = ANN(num_input_features=64,num_hidden_units=16, num_outputs=10, hidden_unit_activation=SigmoidActivation, output_activation=SoftmaxActivation, loss_function=CrossEntropyLoss)
    # Load dataset
    X,y = readDataLabels()      # dataset[0] = X, dataset[1] = y
    # Split data into train and test split. call function in data.py
    train_x,train_labels, test_y,test_labels = train_test_split(X,y)
    # call ann->train()... Once trained, try to store the model to avoid re-training everytime
    if mode == 'train':
        ann.train(train_x, train_labels)        # Call ann training code here
    else:
        # Call loading of trained model here, if using this mode (Not required, provided for convenience)
        raise NotImplementedError
    # Call ann->test().. to get accuracy in test set and print it.
    ann.test(test_y,test_labels)

if __name__ == "__main__":
    main(sys.argv)
