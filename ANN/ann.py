import os, sys
import numpy as np
import math

from data import readDataLabels, normalize_data, train_test_split, to_categorical
from utils import accuracy_score,MSELoss,CrossEntropyLoss,SigmoidActivation,SoftmaxActivation


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
        self.initialize_weights()


    def initialize_weights(self):   # TODO
        self.input_hidden_w=np.random.normal(loc=0.0,scale=1.0,szie=(self.num_hidden_units,self.num_input_features))
        self.hidden_output_w=np.random.normal(loc=0.0,scale=1.0,size=(self.num_outputs,self.num_hidden_units))
        # Create and Initialize the weight matrices
        # Never initialize to all zeros. Not Cool!!!
        # Try something like uniform distribution. Do minimal research and use a cool initialization scheme.
        return

    def forward(self,input):      # TODO
        self.input=input
        input_hidden=np.dot(self.input, self.input_hidden_w)
        self.final_input_hidden=self.hidden_unit_activation(input_hidden)
        hidden_input=np.dot(self.final_input_hidden,self.hidden_output_w)
        self.final_hidden_output=self.output_activation(hidden_input)
        
        # x = input matrix
        # hidden activation y = f(z), where z = w.x + b
        # output = g(z'), where z' =  w'.y + b'
        # Trick here is not to think in terms of one neuron at a time
        # Rather think in terms of matrices where each 'element' represents a neuron
        # and a layer operation is carried out as a matrix operation corresponding to all neurons of the layer
        return  self.final_hidden_output

    def backward(self,loss):     # TODO
        self.ouput_loss=loss
        self.hidden_loss=np.dot(self.hidden_output_w,self.ouput_loss)
        self.d_hidden_output=np.dot((self.ouput_loss*CrossEntropyLoss.grad()),self.final_hidden_output.T)
        self.d_input_output=np.dot((self.hidden_loss*CrossEntropyLoss.grad()),self.input.T)              
        

    def update_params(self):    # TODO
        self.hidden_output_w+=self.d_hidden_output
        self.input_hidden_w+=self.d_input_output
        # Take the optimization step.


    def train(self, dataset_x, dataset_y, learning_rate=0.01, num_epochs=100):
        for epoch in range(num_epochs):
            y=self.forward(dataset_x)
            loss=CrossEntropyLoss(dataset_y,y)
            self.backward(loss)
            self.update_params()


    def test(self, test_dataset):
        accuracy = 0    # Test accuracy
        y_pred=self.forward(test_dataset)
        y_target=dataset_y
        accuracy = accuracy_score(y_target, y_pred)
        # Get predictions from test dataset
        # Calculate the prediction accuracy, see utils.py
        return accuracy


def main(argv):
    ann = ANN(num_input_features=input_list,num_hidden_units=16, num_outputs=10, hidden_unit_activation=SigmoidActivation(), output_activation=SoftmaxActivation(), loss_function=CrossEntropyLoss())
    # Load dataset
    dataset_x, dataset_y = readDataLabels()      # dataset[0] = X, dataset[1] = y
    train_x, train_y = train_test_split(dataset_x,dataset_y)
    # Split data into train and test split. call function in data.py
    # call ann->train()... Once trained, try to store the model to avoid re-training everytime
    if mode == 'train':
        ann.train(train_x, train_y)       # Call ann training code here
    else:
        # Call loading of trained model here, if using this mode (Not required, provided for convenience)
        raise NotImplementedError

    # Call ann->test().. to get accuracy in test set and print it.


if __name__ == "__main__":
    main(sys.argv)
