import os, sys
import numpy as np
import math
from data import readDataLabels, normalize_data, train_test_split, to_categorical
from utils import CrossEntropyLoss, SigmoidActivation, SoftmaxActivation, accuracy_score


# Create an MLP with 8 neurons
# Input -> Hidden Layer -> Output Layer -> Output
# Neuron = f(w.x + b)
# Do forward and backward propagation

mode = 'train'      # train/test... Optional mode to avoid training incase you want to load saved model and test only.

class ANN:
    def __init__(self, num_input_features, num_hidden_units, num_outputs, hidden_unit_activation,
                 output_activation, loss_function):
        self.y_gt = None
        self.y_pred = None
        self.z2 = None
        self.z1 = None
        self.z = None
        self.data = None
        self.bias2 = None
        self.weight2 = None
        self.bias1 = None
        self.weight1 = None
        self.num_input_features = num_input_features
        self.num_hidden_units = num_hidden_units
        self.num_outputs = num_outputs
        self.hidden_unit_activation = hidden_unit_activation
        self.output_activation = output_activation
        self.loss_function = loss_function
        
    def initialize_weights(self):
        # Create and Initialize the weight matrices
        # Never initialize to all zeros. Not Cool!!!
        # Try something like uniform distribution. Do minimal research and use a cool initialization scheme.
        self.weight1 = np.random.uniform(0, 1, size=(self.num_input_features, self.num_hidden_units)) # w1 is of shape 64 by 16.
        self.bias1 = np.random.uniform(0, 1, size=self.num_hidden_units) # b1 is of shape 1 by 16.
        self.weight2 = np.random.uniform(0, 1, size=(self.num_hidden_units, self.num_outputs)) # hidden_w is of shape 16 by 10.
        self.bias2 = np.random.uniform(0, 1, size=self.num_outputs) # b2 is of shape 1 by 10.

    def forward(self, input_x, input_y):
        # x = input matrix
        # hidden activation y = f(z), where z = w.x + b
        # output = g(z'), where z' =  w'.y + b'
        # Trick here is not to think in terms of one neuron at a time
        # Rather think in terms of matrices where each 'element' represents a neuron
        # and a layer operation is carried out as a matrix operation corresponding to all neurons of the layer

        self.data = input_x  # x is of shape 1 by 64, or N by 64 for a sample of size N.
        self.y_gt = to_categorical(input_y)
        self.z = self.data.dot(self.weight1) + self.bias1
        self.z1 = self.hidden_unit_activation(self.z)
        self.z2 = self.z1.dot(self.weight2) + self.bias2
        self.y_pred = self.output_activation(self.z2)

#    def backward(self, y_gt):     # TODO
#        dL_dy_pred = self.CEL.grad()
#
#        SMA = SoftmaxActivation()
#        SMA(self.z2)
#        dy_pred_dz2 = SMA.__grad__()
#
#        dz2_dz1 = self.w2
#        dz2_dw2 = self.z1
#
#        SigA = SigmoidActivation()
#        SigA(self.z)
#        dz1_dz = SigA.__grad__()
#        dz_dw1 = self.x
#        self.dL_dw2 = dL_dy_pred * dy_pred_dz2 * dz2_dw2
#        self.dL_dw1 = dL_dy_pred * dy_pred_dz2 * dz2_dz1 * dz1_dz * dz_dw1
#
#        dz2_db2 = 1
#        self.dL_db2 = dL_dy_pred * dy_pred_dz2 * dz2_db2
#        dz_db1 = 1
#        self.dL_db1 = dL_dy_pred * dy_pred_dz2 * dz2_dz1 * dz1_dz * dz_db1
#
#
#    def update_params(self, learning_rate=0.01):    # TODO
#        # Take the optimization step.
#        self.w1 = self.w1 - learning_rate*self.dL_dw1
#        self.w2 = self.w2 - learning_rate*self.dL_dw2
#        self.b1 = self.b1 - learning_rate*self.dL_db1
#        self.b2 = self.b2 - learning_rate*self.dL_db2

    def train(self, x, y, learning_rate=0.01, num_epochs=1, noisily=True):
        self.initialize_weights()
        for epoch in range(num_epochs):
            if noisily: print('epoch ', epoch)
            self.forward(x, y) # self.forward records all intermediate variables in forward propagation.
            if noisily: print('Training loss is ', self.loss_function(self.y_pred, self.y_gt))
            if noisily: print('Training accuracy is ', accuracy_score(y, np.argmax(self.y_pred, axis=1)))
#            self.backward(y_gt)
#            self.update_params()

    def test(self, x, y):
        # Get predictions from test dataset
        self.forward(x, y)
        # Calculate the prediction accuracy, see utils.py
        accuracy = accuracy_score(y, np.argmax(self.y_pred, axis=1))
        return accuracy

def main(argv):
    ann = ANN(num_input_features=64, num_hidden_units=16, num_outputs=10, hidden_unit_activation=SigmoidActivation(),
              output_activation=SoftmaxActivation(), loss_function=CrossEntropyLoss())
    # Load dataset
    X, y = readDataLabels()      # dataset[0] = X, dataset[1] = y
    # Split data into train and test split. call function in data.py
    train_x, train_y, test_x, test_y = train_test_split(X, y)
    train_x = normalize_data(train_x)
    test_x = normalize_data(test_x)
    # call ann->train()... Once trained, try to store the model to avoid re-training everytime
    if mode == 'train':
        ann.train(x=train_x, y=train_y)        # Call ann training code here
    else:
        # Call loading of trained model here, if using this mode (Not required, provided for convenience)
        raise NotImplementedError
    # Call ann->test().. to get accuracy in test set and print it.
    print("Accuracy is ", ann.test(test_x, test_y))

if __name__ == "__main__":
    main(sys.argv)