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


# This block that is commented out was meant to be a multi-class cross-entropy loss function implementation.
# We did not manage to get it to work in backpropagation.
#class CrossEntropyLoss:     # TODO: Make this work!!!
#    def __init__(self):
#        # Buffers to store intermediate results.
#        self.current_prediction = None
#        self.current_gt = None
#        pass
#    def __call__(self, y_pred, y_gt): # y_pred should be an array of probabilities
#        # TODO: Calculate Loss Function
#        y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
#        self.current_prediction = y_pred
#        self.current_gt = y_gt
#        log_prob = np.log(y_pred)
#        loss_by_sample = -np.sum(y_gt*log_prob, axis=1)
#        loss = np.sum(loss_by_sample, axis=0)
#        return loss
#    def grad(self):
#        # TODO: Calculate Gradients for back propagation
#        # Derived by calculating dL/dy_pred
#        gradient = -self.current_gt/self.current_prediction
#        self.current_prediction = None
#        self.current_gt = None
#        return gradient


class CrossEntropyLoss:     # TODO: Make this work!!!
    def __init__(self):
        # Buffers to store intermediate results.
        self.current_prediction = None
        self.current_gt = None
        pass
    def __call__(self, y_pred, y_gt):
        # TODO: Calculate Loss Function
        self.y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        self.y_gt = y_gt
        loss = - y_gt * np.log(y_pred) - (1 - y_gt) * np.log(1 - y_pred)
        return loss
    def grad(self):
        # TODO: Calculate Gradients for back propagation
        self.y_pred = np.clip(self.y_pred, 1e-15, 1 - 1e-15)
        gradient = - (self.y_gt / self.y_pred) + (1 - self.y_gt) / (1 - self.y_pred)
        return gradient


class SoftmaxActivation:    # TODO: Make this work!!!
    def __init__(self):
        self.z = None
        pass
    def __call__(self, z):
        # TODO: Calculate Activation Function
        self.z = z
        y = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True) # sum across axis=1
        self.y = y
        return y
    def __grad__(self):
        # TODO: Calculate Gradients.. Remember this is calculated w.r.t. input to the function -> dy/dz
        gradient = self.y*(1-self.y)
        return gradient


class SigmoidActivation:    # TODO: Make this work!!!
    def __init__(self):
        self.z = None
        pass
    def __call__(self, z):
        # TODO: Calculate Activation Function
        self.z = z
        y = 1/(1+np.exp(-z))
        self.y = y
        return y
    def __grad__(self):
        # TODO: Calculate Gradients.. Remember this is calculated w.r.t. input to the function -> dy/dz
        gradient = self.y*(1-self.y)
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


def accuracy_score(y_pred, y_gt):
    y_pred = np.argmax(y_pred, axis=1)
    y_gt = np.argmax(y_gt, axis=1)
    accuracy = np.sum(y_pred == y_gt)/y_gt.shape[0]
    return accuracy
