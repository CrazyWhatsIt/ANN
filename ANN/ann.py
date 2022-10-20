import os, sys
import numpy as np
import math

from data import readDataLabels, normalize_data, train_test_split, to_categorical
from utils import CrossEntropyLoss, SigmoidActivation, SoftmaxActivation, accuracy_score

# Create an MLP with 8 neurons
# Input -> Hidden Layer -> Output Layer -> Output
# Neuron = f(w.x + b)
# Do forward and backward propagation

mode = 'train'  # train/test... Optional mode to avoid training incase you want to load saved model and test only.


class ANN:
    def __init__(self, num_input_features, num_hidden_units, num_outputs, hidden_unit_activation, output_activation,
                 loss_function):
        # model attributes
        self.num_input_features = num_input_features
        self.num_hidden_units = num_hidden_units
        self.num_outputs = num_outputs
        self.hidden_unit_activation = hidden_unit_activation
        self.output_activation = output_activation
        self.loss_function = loss_function
        # weights
        self.weight1 = None
        self.bias1 = None
        self.weight2 = None
        self.bias2 = None
        self.initialize_weights()
        # data
        self.training_data = None # do we need to remember the training data? should just need to remember the model.
        self.predicted_values = None # do we need to remember the predicted values on a single run?


    def initialize_weights(self):
        # Create and Initialize the weight matrices
        # Never initialize to all zeros. Not Cool!!!
        # Try something like uniform distribution. Do minimal research and use a cool initialization scheme.
        self.weight1 = np.random.uniform(0, 1, size=(self.num_input_features, self.num_hidden_units))
        self.bias1 = np.random.uniform(0, 1, size=self.num_hidden_units)
        self.weight2 = np.random.uniform(0, 1, size=(self.num_hidden_units, self.num_outputs))
        self.bias2 = np.random.uniform(0, 1, size=self.num_outputs)

    def forward(self, input_array):
        # x = input matrix
        # the inner product is z = w.x + b where w are the weights and b is the bias.

        # hidden activation y = f(z), where z = w.x + b
        # output = g(z'), where z' =  w'.y + b'
        # Trick here is not to think in terms of one neuron at a time
        # Rather think in terms of matrices where each 'element' represents a neuron
        # and a layer operation is carried out as a matrix operation corresponding to all neurons of the layer
        # here we calculate the inner product for the input layer.
        input_layer_inner_product = np.dot(input_array, self.weight1) + self.bias1
        # next we should apply the sigmoid function to the input inner product.
        sigmoid_output = self.hidden_unit_activation(input_layer_inner_product)
        # next, take the output of the sigmoid function and apply the inner product for the hidden layer.
        hidden_layer_inner_product = np.dot(sigmoid_output, self.weight2) + self.bias2
        # apply the softmax activation function to the output of the hidden layer activation function.
        self.predicted_values = self.output_activation(hidden_layer_inner_product)
        return self.predicted_values

    def backward(self, labels):
        # start by calculating the error with the loss function.
        error = self.loss_function(self.predicted_values, labels)
        print(np.shape(error))
        # dL_dy_pred = CEL.grad()
        #
        # SMA = SoftmaxActivation()
        # SMA(self.z2)
        # dy_pred_dz2 = SMA.__grad__()
        #
        # dz2_dz1 = self.w2
        # dz2_dw2 = self.z1
        #
        # SigA = SigmoidActivation()
        # SigA(self.z)
        # dz1_dz = SigA.__grad__()
        # dz_dw1 = self.x
        # self.dL_dw2 = dL_dy_pred * dy_pred_dz2 * dz2_dw2
        # self.dL_dw1 = dL_dy_pred * dy_pred_dz2 * dz2_dz1 * dz1_dz * dz_dw1
        #
        # dz2_db2 = 1
        # self.dL_db2 = dL_dy_pred * dy_pred_dz2 * dz2_db2
        # dz_db1 = 1
        # self.dL_db1 = dL_dy_pred * dy_pred_dz2 * dz2_dz1 * dz1_dz * dz_db1

    def update_params(self, learning_rate=0.01):  # TODO
        raise NotImplementedError
        # Take the optimization step.
        self.w1 = self.w1 - learning_rate * self.dL_dw1
        self.w2 = self.w2 - learning_rate * self.dL_dw2
        self.b1 = self.b1 - learning_rate * self.dL_db1
        self.b2 = self.b2 - learning_rate * self.dL_db2

    def train(self, data, labels, learning_rate=0.01, num_epochs=100):
        self.initialize_weights()
        for epoch in range(num_epochs):
            print('epoch ', epoch)
            self.forward(input_array=data)  # self.forward records all intermediate variables in forward propagation.
            #self.backward(labels)
            #self.update_params()

    def test(self, test_x, test_labels):
        raise NotImplementedError
        accuracy = 0  # Test accuracy
        y_pred = self.forward(test_x)
        accuracy = accuracy_score(test_labels, y_pred)
        # Get predictions from test dataset
        # Calculate the prediction accuracy, see utils.py
        accuracy = accuracy_score(test_labels, self.y_pred)
        return accuracy


def main(argv):
    debug = True
    if debug: print("!! Start the artificial neural network !!")
    if debug: print("Construct the ANN Object.")
    ann = ANN(num_input_features=64, num_hidden_units=16, num_outputs=10, hidden_unit_activation=SigmoidActivation(),
              output_activation=SoftmaxActivation(), loss_function=CrossEntropyLoss())
    if debug: print("Load the dataset.")
    X,y = readDataLabels()
    # Split data into train and test split. call function in data.py
    if debug: print("Split the dataset.")
    train_x, train_y, test_x, test_y = train_test_split(X,y)
    if debug: print("Normalize training and test data.")
    train_x = normalize_data(train_x)
    test_x = normalize_data(test_x)
    if debug: print("Catagorize the training and testing output.")
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    # call ann->train()... Once trained, try to store the model to avoid re-training everytime
    if mode == 'train':
        ann.train(data=train_x, labels=train_y)        # Call ann training code here
    else:
        # Call loading of trained model here, if using this mode (Not required, provided for convenience)
        raise NotImplementedError
    # Call ann->test().. to get accuracy in test set and print it.
    # ann.test(test_x, test_y)
#
# if __name__ == "__main__":
#    main(sys.argv)
    if debug: print("!! END !!")

main("")