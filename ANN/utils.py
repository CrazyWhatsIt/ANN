import numpy as np
import math


class MSELoss:      # For Reference
    def __init__(self):
        # Buffers to store intermediate results.
        self.current_prediction = None
        self.current_gt = None
        pass

    def __call__(self, y_pred, y_gt):
        self.current_prediction = y_pred
        self.current_gt = y_gt

        # MSE = 0.5 x (GT - Prediction)^2
        loss = 0.5 * np.power((y_gt - y_pred), 2)
        return loss

    def grad(self):
        # Derived by calculating dL/dy_pred
        gradient = -1 * (self.current_gt - self.current_prediction)

        # We are creating and emptying buffers to emulate computation graphs in
        # Modern ML frameworks such as Tensorflow and Pytorch. It is not required.
        self.current_prediction = None
        self.current_gt = None

        return gradient


class CrossEntropyLoss:     # TODO: Make this work!!!
    def __init__(self):
        # Buffers to store intermediate results.
        self.current_prediction = None
        self.current_gt = None
        pass

#y_gt = np.random.randint(2, size=10)
#y_pred = np.random.uniform(0, 1, size=10)
    def __call__(self, y_pred, y_gt): # y_pred should be an array of probabilities
        # TODO: Calculate Loss Function
        self.current_prediction = y_pred
        self.current_gt = y_gt
        loss = y_gt*np.log(y_pred) + (1-y_gt)*np.log(1-y_pred)
        return loss

    def grad(self):
        # TODO: Calculate Gradients for back propagation
        gradient = self.current_gt/self.current_prediction - (1-self.current_gt)/(1-self.current_prediction)

        self.current_prediction = None
        self.current_gt = None
        return gradient


class SoftmaxActivation:    # TODO: Make this work!!!
    def __init__(self):
        self.z = None
        pass
    def __call__(self, z):
        # TODO: Calculate Activation Function
        self.z = z
        exps = np.exp(z)
        y = exps / np.sum(exps)
        return y
    def __grad__(self):
        # TODO: Calculate Gradients.. Remember this is calculated w.r.t. input to the function -> dy/dz
        s = y.reshape(-1,1)
        gradient = np.diagflat(s) - np.dot(s, s.T)
        return gradient


class SigmoidActivation:    # TODO: Make this work!!!
    def __init__(self):
        self.z = None
        pass

    def __call__(self, z):
        # TODO: Calculate Activation Function
        sigmoid = 1/(1+np.exp(-z))
        return y

    def __grad__(self):
        # TODO: Calculate Gradients.. Remember this is calculated w.r.t. input to the function -> dy/dz
        gradient = y*(1-y)
        return gradient


class ReLUActivation:
    def __init__(self):
        self.z = None
        pass

    def __call__(self, z):
        # y = f(z) = max(z, 0) -> Refer to the computational model of an Artificial Neuron
        self.z = z
        y = np.maximum(z, 0)
        return y

    def __grad__(self):
        # dy/dz = 1 if z was > 0 or dy/dz = 0 if z was <= 0
        gradient = np.where(self.z > 0, 1, 0)
        return gradient


def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy
