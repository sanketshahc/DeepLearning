import numpy as np
from math import *

velocity = 0
# hyper parameters
EPOCHS = 1000
BATCH = 10
RATE = .03
MOMENTUM = .1
velocity = lambda g: MOMENTUM * velocity - RATE * g
DECAY = .03

class Function(object):
    def __init__(self):
        pass

    def apply_forward(self, *args):
        pass
        ## apply forward....save outputs?

class Tikhonov(Function):
    @staticmethod
    def forward(logit, dim):
        pass
    @staticmethod
    def backward(logit):
        """"
        """

class Ridge_Regression(Function):
    def forward(self, *args):
        pass

    def backward(self, *args):
        pass


class Poisson_Regression(Function):
    """
    Log-Linear (generalized linear function)...not totally sure what that means.
    """

    def forward(self, *args):
        pass

    def backward(self, *args):
        pass


class Poisson_Loss(Function):
    def forward(self, *args):
        pass

    def backward(self, *args):
        pass


class Mean_Squared_Error(Function):
    def forward(self, *args):
        """

        """
        pass

    def backward(self, *args):
        pass


class Softmax_Regression(Function):
    """
    input logits of shape:
    BATCHES x BATCH SIZE x FEATURE HEIGHT x LAYER OUTPUT WIDTH (CLASSES)
        N/10 x 10 x 1 x 3
    """

    @staticmethod
    def forward(logit, dim=-1):
        """
        dim is dim along which...typically [-1]
        :param logit:
        :param dim:
        :return:
        """
        # assert shape of logit
        e = logit.exp()
        soft = e / e.sum(dim)
        return soft

    @staticmethod
    def backward(l_fn):
        """"
        not needed....also really hard
        """
        pass


class MatMul(Function):
    @staticmethod
    def forward(a, b):
        return a @ b

    @staticmethod
    def backward(a, b):
        """
        not generalized! assuming ab or XW ordering, dy/dw

        multiply by outer in the actual outer function.
        """
        return a.transpose()


class Cross_Entropy(Function):
    @staticmethod
    def forward(y, y_hat, dim=-1, r_fn=None):
        """
        curry in lam, p, w
        only used for softmax classification....
        input final actvation, in this case softmax
        dim is dim for loss summation, typically -1,

        this is the element wise function, it must be averaged over the batch elsewhere
        """
        # assert shapes of y , y_hat are equal, y_hat can be array class
        # assert r_fn has right class (function class)
        loss = (y * y_hat.log()).sum(dim) * -1

        def _reg(lam, p, w):
            return loss + r_fn.forward(lam, p, w)

        return _reg if r_fn else loss
        # currently out put is shape [n,]
        # asser shape of return is 2d?

    @staticmethod
    def backward(y, x, w, y_hat_fn, logit_fn, r_fn=None):
        """
        y is y, x is input, yhatfn is softmax, logitfn is matmul, r_fn is regul...
        curry in lam, p, w for regularization
        only used with softmax at moment, can just use softmax forward for y_hat (will have
        access in network...will also have access to X, W.
        """
        # assert y, y_hat area equal
        # assert y_hat, logitfn issubclass(Softmax_Regression, Function)
        d_loss = (y_hat_fn(logit_fn(x,w)) - y) * logit_fn.backward(x)

        def _reg(lam, p, w):
            return d_loss + r_fn.backward(lam, p, w)

        return _reg if r_fn else d_loss



class Regularize(Function):
    @staticmethod
    def forward(lam, p, w):
        """
        p is the degree of norm...ie 1 or 2 or
        can only be 1 or 2 for now.
        lam is regularization constant
        w is weights
        """
        assert p == 1 or 2
        norm = ((abs(w) ** p).sum()) ** (1 / p)
        return lam / p * norm ** p

    @staticmethod
    def backward(lam, p, w):
        assert p == 1 or 2

        def sgn(w):
            return 1 if w > 0 else -1 if w < 0 else 0

        if p == 1:
            return sgn(w)
        if p == 2:
            return lam * w


class Single_Layer_Network(object):
    """"

    this wil have to store the model weight and bias values
    this will have a forward call. something like the Linear class in the minitorch.
    bias embedded in inputs

    data should be 'imported' before being input, but the batching and standardizing happens
    during training. weights should be imported as random array.
    Be sure of target shape depending on training type...
    """

    def __init__(self, inputs, weights, targets, forward_fn, loss_fn, n_classes =1):
        """:type
        inputs, weights should be arrays,
        classes are number of classes, 1 for regression
        forward is the forward function being called....can be lambda if combining
        """
        self.W = weights
        self.X = inputs
        self.Y = targets
        self.C = n_classes
        self.forward_fn= forward_fn
        self.loss_fn = loss_fn
        # self.bias is embedded in inputs as column of 1s

        # pre batching shape assertions:
        assert self.Y.shape[-1] == self.C
        assert self.X.shape[-1] == self.W.shape[-2]
        assert self.W.shape[-1] == self.C

        #type assertions
        assert isinstance(self.X, np.ndarray)
        assert isinstance(self.W, np.ndarray)
        assert isinstance(self.C, np.ndarray)
        assert isinstance(self.Y, np.ndarray)
        assert issubclass(self.forward_fn, Function)
        assert issubclass(self.loss_fn, Function)

        # if dtype == 'input':
        #     shape(batches, batch_size, feat_h, feat_w)
        # if dtype == 'weights':
        #     shape(batches, feat_d, feat_w, output)
        # if dtype == 'output':
        #     shape(batches, feat_d, feat_w, output)

    def forward(self):
        return self.forward_fn.forward(self.inputs,self.weights)

    def update(*args):
        """
        updating weights....
        new velocity = (mu)(old velocity) - (a)(new gradient)
        new weights = old weights + new velocity
        """
        pass

    def evaluate(*args):
        '''
        Load up eval data?

        on eval(test) data:`
        evaluate for correctness. same thing as above, just additionally to a loss equation,
        record accuracy (max index vs hot v index for each class, then average the rates together)
        '''
        # instead just build a 3d array that is epoch x Y x Yhat:
        # per training:
        #   track an 'evaluation' array, from which produce total accuracies and confusion matrix
        #   track total loss, in list form, indices = epoch
        #   per epoch:
        #      track total epoch slice of eval array
        #      track loss figure total of epoch
        #      per batch:
        #          for each combination of y, yhat (argmax),
        #               add a count to that index of a y x yhat sized array
        #          calculate total loss or average loss, and add to epoch total
        #          if criteria meets testing checkpoint,
        #               then also do above for testing data as well,in a separate array/list

    def train(self):
        # gradient descent
        #   remember to average gradient across batch before subtracting from loss (loss is summed over
        #   batch....

        """
        call batcher on inputs
        call forward: (can

        loss equations (against training data) minus the regularization norm
            additionally to a loss equation,
            mean per class accuracy (max index vs hot v index)
        derivitave of loss
        """
        # prep inputs
        X = Pipeline.batch(self.X)
        assert len(self.X.shape) == 4
        assert len(self.W.shape) == 4
        assert self.X.shape[1] == BATCH
        assert self.X.shape[0] == self.W.shape[0] == self.X.shape[-1]/BATCH



class Pipeline(object):
    '''
    generator to pull mini batches of the txt data, totalling ~100 examples.
    Batch size = 10
    Rate =
    Features = 2
    epochs = 1000 (loops over data, loops of descent algo)

    Load up eval data???? output eval data???
    '''

    def __init__(self):
        pass

    def iris(self, data, *range):
        """
        :param string data: path for data, string
        :

        :return: no worries about shape here
        first col label (1,2,3) next 2 features (int,flouat, flout)

        must make a 1-hot vector from y's, and input with corresponding index...
        .perhaps this function just
        ouputs a list, and the bather is what batches....

        """

    def cifar(self):
        """

        use import code from before, out put a list of whole data / whole hot vector labels
        """
        pass

    def msd(self):
        """
        will need to output data as 1 hot vector, but for regression 'label' needs to be single
        scalar...if classification, take range of different labels and make a vector of length?
        in any case, outputting lists of total data
        """
        pass

    def batch(self, shape):
        """
        the batcher is what worries about shapes...it takes output from above functions and
        outputs the right shape....consider function currying for this....

        I wonder if there's any way to make an iterator out of this....
        essentially what is required is iterating along the data pairs, forming a 'tensor' for
        each batch of shape (batch x output x input)

        you can make a generator function to do precisely this....
        """
        pass

    def standardize(self):
        """
        use a modified normalization
        (only for input data)
        x = 2(x - min)/(max -min) -1
        """


class Plot(object):
    """
    Visualize loss curve for test, and training data....
    1. loss curve vs epoch
    2. accuracy vs epoch
    Visualize the partitioning of space....
    - achieve by calling forward on the grid of points in the plot...(arbitrary density)

    this instance gets inputed the 'evaluation matricies' for testing and training data after
    training...maybe the training function outputs it?

    it also gets the loss log
    """

    def __init__(self):
        pass

    def curves(self):
        pass

    def spaces(self):

        pass

    def histrogram(self):
        pass

    def confusion(self):
        """
        takes input of list of output-index comparisons, as well as the label list of same indexes..
        this function counts each pair, and plots the grid of 3d data

        sum up array along epoch dimension and simply do a plt.matshow, style as below.

        :return:
        """



        pass




# in minitorch, the network is a module class which stores the bias and weights...essentially
# each layer stores weights / bias....and the weights store the grad..


# Is it accurate to think of each class def as an encapsulation or frame, and If python doesn't
# find a binding in the present frame, it moves up to the parent, and keeps going up to find it?
# And while the subclass has access to all these (class level, not instance level) things,
# it's place in the structure is essentially just set by whatever is passed into the initial
# "class ClassName(ParentClass):" definition? So both the Superclasses and Subclasses exist in
# the same space, the super/sub relationship is just defined by that one link....if that makes
# sense? Sorry if that's confusing.  Thank


# currying is returning a function..
