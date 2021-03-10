import numpy as np
from math import *

# hyper parameters
EPOCHS = 1000
RC = .05
BATCH = 10
RATE = .03
MOMENTUM = .1
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
        def round_helper(Y):
            """
            round array input and return.
            """
            pass
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
        assert isinstance(y_hat, np.ndarray)
        assert y.shape == y_hat.shape
        assert issubclass(r_fn, Function) if r_fn else None

        loss = (y * y_hat.log()).sum(dim) * -1

        def _reg(lam, p, w):
            return loss + r_fn.forward(lam, p, w)

        return _reg if r_fn else loss
        # currently out put is shape [n,]
        # asser shape of return is 2d?

    @staticmethod
    def backward(y, x, w, y_hat_fn, logit_fn, r_fn=None):
        """
        y is y, x is input, yhatfn is output fn or softmax, logitfn is matmul, r_fn is regul...
        curry in lam, p, w for regularization
        only used with softmax at moment, can just use softmax forward for y_hat (will have
        access in network...will also have access to X, W.

        arguments for other loss backwards should be same for other functions, but w/o fn inputs
        """
        assert issubclass(y_hat_fn, Function)
        assert issubclass(logit_fn, Function)
        d_loss =  logit_fn.backward(x,w) @ (y_hat_fn.forward(logit_fn.forward((x,w))) - y)
        #chain ruling for dl/dw
        assert d_loss.shape == (BATCH,w.shape[-2],w.shape[-1]), \
            f"ooops check loss deriv, it's {d_loss.shape}"
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
    during training. weights should be imported as random aπrray.
    Be sure of target shape depending on training type...

    ATTN: will have to hardcode loss backwards unfort...too complicated here to gneralize.
    """

    def __init__(
            self,
            inputs,
            targets,
            test_inputs,
            test_targets,
            loss_fn,
            output_fn,
            logit_fn,
            feat_d=1,
            n_classes=1
    ):
        """
        inputs, weights should be arrays,
        classes are number of classes, 1 for regression
        forward is the forward function being called....can be lambda if combining, or just the
        loss fn is the loss function CLASS
        logit fn is the logit_fn for
        """
        # FUNCTIONAL ARGS:
        self.r_fn = Regularize
        self.loss_fn = loss_fn
        self.logit_fn = logit_fn # sometimes same as output
        self.output_fn = output_fn # only different in case of softmax regression

        # DATA AND DIMENSIONAL ARGS:
        self.X = inputs
        self.Y = targets
        self.C = n_classes
        feat_w = self.X.shape[-1]  # feat width (includes bias)
        self.W = np.random.rand(feat_d, feat_w, self.C) # inititalizing random weights.
        # self.bias is embedded in inputs (X) as column of 1s
        self.X_t = test_inputs
        self.Y_t = test_targets

        # PRE-BATCHING SHAPE ASSERTIONS:
        assert self.Y.shape[-1] == self.C
        assert self.Y_t.shape[-1] == self.C
        assert self.X.shape[-1] == self.W.shape[-2]
        assert self.X_t.shape[-1] == self.W.shape[-2]
        assert self.W.shape[-1] == self.C

        # UDPATING, LOGGING, EVALUATION
        # self.eval_array is an array data structure to track accuracy and generate accuracy
        # visualization like confusino matrix and mean accuracy curves. if only one class (
        # regression), then array is (epochs x 2 x 1), otherwise (epochs x classes x classes)
        if self.C == 1:
            self.training_eval_array = np.zeros((EPOCHS, 2, 1)) # not needed
            self.testing_eval_array = np.zeros((EPOCHS, 2, 1))  # not needed
        else:
            self.training_eval_array = np.zeros((EPOCHS, self.C, self.C))
            self.testing_eval_array = np.zeros((EPOCHS, self.C, self.C))
        self.velocity = 0 # inititalized at 0
        self.training_losses = []
        self.testing_losses = []

        # NOTES ON SHAPE:
        # if dtype == 'input':
        #     shape(batches, batch_size, feat_h, feat_w)
        # if dtype == 'weights':
        #     shape(batches, feat_d, feat_w, output)
        # not actually different weights obj per batch, but just helpful to think in that way in
        # terms of shape
        # if dtype == 'output':
        #     shape(batches, feat_d, feat_h, output)

        # TYPE ASSERTIONS
        assert isinstance(self.X, np.ndarray)
        assert isinstance(self.W, np.ndarray)
        assert isinstance(self.C, np.ndarray)
        assert isinstance(self.Y, np.ndarray)
        assert issubclass(self.logit_fn, Function)
        assert issubclass(self.output_fn, Function)
        assert issubclass(self.loss_fn, Function)

    def forward(self):  # argument must be curried
        if isinstance(self.output_fn,Softmax_Regression):
            ret = lambda x,w: self.output_fn.forward(self.logit_fn.forward(x,w))
        else:
            ret = lambda x,w: self.output_fn.forward(x,w)

        return ret

    # def loss(self): # argument must be curried
    #     return self.loss_fn.forward
    # Called directly in trainin

    def update(self,d_loss):
        """
        input: d_loss
        output: none
        (changes weights only)

        updating weights....
        new velocity = (mu)(old velocity) - (a/BATCH)(new gradient.sum(0)
        new weights = old weights + new velocity

        the d_loss is a 3d matrix of Batch x Weights basically. So the idea is to take the
        average along the first dimension. That's really the key insight, is the shape of the
        gradient is extra dimensional...makse sense tho since each ouput has a gradienet!

        so here, the idea is to take average of weights
        """
        d_loss_sum = d_loss.sum(0)
        assert d_loss_sum.shape[-2] == self.W.shape[-2]
        assert d_loss_sum.shape[-1] == self.W.shape[-1]
        self.velocity = MOMENTUM * self.velocity - RATE * d_loss_sum/BATCH
        self.W = self.W + self.velocity

    def evaluate(self,mode,epoch,Y,Y_hat):
        """
        needs to be called per batch per epoch
        """
        assert mode == "training" or "testing"
        eval_array = self.training_eval_array if mode == "training" else self.testing_eval_array
        Y_hat_argsmax = np.argmax(Y_hat, axis=-1)
        Y_argsmax = np.argmax(Y, axis=-1)
        for j,k in zip(Y_hat_argsmax,Y_argsmax):
            i = epoch
            eval_array[i,j,k] += 1 #adding count to eval array

    def testing(self,epoch,p):
        '''
        called per epoch
        '''
        for X, Y in Pipeline.batch_gen(self.X_t, self.Y_t, output=self.C):
            W = self.W  # should already be shaped for batch, initiated at random

            # per batch shape assertions
            assert len(X.shape) == 3
            assert len(Y.shape) == 3
            assert len(W.shape) == 3
            assert X.shape[0] == W.shape[0] == X.shape[-1] / BATCH
            assert X.shape[1] == BATCH
            assert X.shape[0] == W.shape[0] == X.shape[-1] / BATCH
            Y_hat = self.forward()(X, W)  # currying to forward_fn
            if not p:
                loss = self.loss_fn.forward(Y, Y_hat)
            elif p:
                loss = self.loss_fn.forward(Y, Y_hat, self.r_fn)(RC, p, W)
            self.testing_losses.append(loss.sum())
            self.evaluate("testing",epoch,Y,Y_hat)

        #     todo print a report on log

        # instead just build a 3d array that is epoch x Y x Yhat:
        # per training:
        #   track an 'evaluation' array, from which produce total accuracies and confusion matrix
        #   track total loss, in list form, indices = epoch
        #   per epoch:
        #      track loss figure total of epoch
        #      track total epoch slice of eval array
        #      per batch:
        #          calculate total loss or average loss, and add to epoch total
        #          for each combination of y, yhat (argmax),
        #               add a count to that index of a y x yhat sized array
        #          if criteria meets testing checkpoint,
        #               then also do above for testing data as well,in a separate array/list



    def train(self, p):

        """
        p = 0 for no regularization....must specify! p is the degree of regularization

        call batcher on inputs
        call forward: (can

        loss equations (against training data) minus the regularization norm
            additionally to a loss equation,
            mean per class accuracy (max index vs hot v index)
        derivitave of loss
        """

        for epoch in EPOCHS:
            # stuff per epoch
            for X, Y in Pipeline.batch_gen(self.X, self.Y, output=self.C):
                W = self.W  # should already be shaped for batch, initiated at random
                # per batch shape assertions
                assert len(X.shape) == 3
                assert len(Y.shape) == 3
                assert len(W.shape) == 3
                assert X.shape[0] == W.shape[0] == X.shape[-1] / BATCH
                assert X.shape[1] == BATCH
                assert X.shape[0] == W.shape[0] == X.shape[-1] / BATCH
                Y_hat = self.forward()(X, W)  # currying to forward_fn
                if not p:
                    loss = self.loss_fn.forward(Y, Y_hat)
                elif p:
                    loss = self.loss_fn.forward(Y, Y_hat, self.r_fn)(RC, p, W)
                self.training_losses.append(loss.sum())

                #########
                # ATTN: will have to hardcode for each network type :(
                ########
                if not p:
                    d_loss = self.loss_fn.backward(
                        Y,
                        X,
                        W,
                        self.output_fn,
                        self.logit_fn,
                        self.r_fn)
                elif p:
                    d_loss = self.loss_fn.backward(
                        Y,
                        X,
                        W,
                        self.output_fn,
                        self.logit_fn,
                        self.r_fn)(RC, p, W)
                #####
                # << outputs dl/dw
                self.update(d_loss)
                self.evaluate("training",epoch,Y,Y_hat)
            self.testing(epoch,p)


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

    def batch_gen(self, shape):
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



# rememeber each output example in the batch has a gradient with rt to weights input
# etc....event thought they're being evaluated at the same weight values.....and so we are
# summing across the whole ouput example batch, to get an overall loss, so that we can get in
# a sense an overall gradient. if we average the loss across the batch then we get the
# average gradient wrt to weights. ...so if we've been given that value, then we can ...
#  if wwe valuate it at a specific random value, we' can see then the contribution of shift
# in each weight to the average loss....of the batch, and so that value, which necessarily
# must be the size of teh weight matrix, can be multiplied by rate and subtracted