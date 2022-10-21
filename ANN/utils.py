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

    def __call__(self, y_pred, y_gt): # y_pred should be an array of probabilities
        # TODO: Calculate Loss Function
        self.current_prediction = y_pred
        self.current_gt = y_gt
        log_prob = np.log(y_pred)
        loss = -np.sum(y_gt*log_prob)
        return loss # sum across both classes and samples

    def grad(self):
        # TODO: Calculate Gradients for back propagation
        # Derived by calculating dL/dy_pred
        gradient = -np.sum(self.current_gt/self.current_prediction)

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
        y = np.exp(z) / np.sum(np.exp(z)) # sum across axis=1
        self.y = y
        return y
    def __grad__(self):
        # TODO: Calculate Gradients.. Remember this is calculated w.r.t. input to the function -> dy/dz
        gradient = self.y*(1-self.y)
        return gradient
#    def __grad__(self):
#        jacobian = np.diag(y)
#        for i in range(len(jacobian)):
#            for j in range(len(jacobian)):
#                if i==j:
#                    jacobian[i][j] = y[i]*(1-y[i])
#                else:
#                    jacobian[i][j] = -y[i]*y[j]
#        return jacobian
#    def __grad__(self):
#        s = self.y.reshape(-1,1)
#        gradient = np.diagflat(s) - np.dot(s, s.T)
#        return gradient


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


def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy
