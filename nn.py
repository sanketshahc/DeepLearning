import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import random
from math import *

# hyper parameters
EPOCHS = 1000
RC = .05
BATCH = 10
RATE = .03
MOMENTUM = .1
DECAY = .03

RESOURCES = {
    "iris_train": "resources/iris-train.txt",
    "iris_test": "resources/iris-test.txt",
    "cifar_train": "resources/cifar-10-batches-py/data_batch_1",
    "cifar_test": "resources/cifar-10-batches-py/test_batch",
    "msd_train": ("resources/YearPredictionMSD.txt", [0, 463713]),
    "msd_test": ("resources/YearPredictionMSD.txt", [-51630, -1])
}


def randomize_helper(*sets):
    """
    be sure and call function with a * if needing different sets
    an arbitrary set of indexed sets must be same length
    returns same sets randomized, but with same order of sets
    """
    count = len(sets[0])
    for s in sets:
        assert count == len(s), "sets not same length"
    rand = [[] for s in sets]  # new sets
    indices = [i for i in range(count)]
    while indices:
        i = indices.pop(random.randint(0, len(indices) - 1))
        for k, s in enumerate(sets):
            rand[k].append(s[i])

    return rand


def hot_helper(Y):
    """
    input normal Y array of labels or targets

    """
    labels = set(Y)
    Y_hot = []
    _labels = [l for l in labels]  # index 0 = 1, 1 = 2, etc
    for y in Y:
        # y_hot = [0 for i in len(hot)]
        y_hot = [0 for i in _labels]
        for l, m in enumerate(_labels):
            y_hot[l] = 1 if m == y else 0
        Y_hot.append(y_hot)

    return np.array(Y_hot)


### cifar:
# changing from binary to python dictionary of (batch, label, data, filename)
def unpickle_helper(file):
    """input file path, output dictionary object, of labels, data, filenames """
    with open(file, 'br') as f:
        d = pickle.load(f, encoding='bytes')  # need to specify encoding

    return d


# helper function to check when to stop iterating through batch (counting function for total
# images in arr2d)
def countel_helper(collection, s=0):
    """
    counts total first dimension elements in a multidimensional array
    """
    for nested in collection:
        if type(nested) == list:
            s += len(nested)
            countel_helper(nested)

    return s

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
    def forward(logit, dim = -1):
        """
        dim is dim along which...typically [-1]
        :param logit:
        :param dim:
        :return:
        """
        assert isinstance(logit,np.ndarray)
        assert isinstance(dim, int), print(type(dim))
        # print(logit.shape)
        e = np.exp(logit)
        e_sum = e.sum(dim)
        e_sum = e_sum.reshape(e_sum.shape[-2],e_sum.shape[-1],1)
        soft = e / e_sum
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
        return np.matmul(a,b)

    @staticmethod
    def backward(a, b):
        """
        not generalized! assuming ab or XW ordering, dy/dw

        multiply by outer in the actual outer function.
        """
        assert len(a.shape) == 3, f"ashape: {a.shape}"
        return a.transpose(0,2,1)


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
        assert y.shape == y_hat.shape, f"y.shape: {y.shape}, yhat.shape{y_hat.shape}"
        if r_fn:
            assert issubclass(r_fn, Function)

        loss = (y * np.log(y_hat)).sum(dim) * -1

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
        A = logit_fn.backward(x,w)
        B = (y_hat_fn.forward(logit_fn.forward(x,w)) - y)
        d_loss = logit_fn.forward(A,B)
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
            n_classes=1,
            bias = 'embedded'
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
        feat_w = self.X.shape[-1]
        feat_w = feat_w + 1 if bias == 'embedded' else feat_w # feat width (includes bias)
        self.W = np.random.rand(feat_d, feat_w, self.C) # inititalizing random weights.
        # self.bias is embedded in inputs (X) as column of 1s
        self.X_t = test_inputs
        self.Y_t = test_targets

        # PRE-BATCHING SHAPE ASSERTIONS:
        assert self.Y.shape[-1] == self.C
        assert self.Y_t.shape[-1] == self.C
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
        # assert isinstance(self.C, np.ndarray)
        assert isinstance(self.Y, np.ndarray)
        assert issubclass(self.logit_fn, Function)
        assert issubclass(self.output_fn, Function)
        assert issubclass(self.loss_fn, Function)

    def forward(self):  # argument must be curried
        if self.output_fn == Softmax_Regression:
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
            assert X.shape[-1] == self.W.shape[-2]
            assert X.shape[1] == BATCH
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

        for epoch in range(EPOCHS):
            # stuff per epoch
            for X, Y in Pipeline.batch_gen(self.X, self.Y, output=self.C):
                W = self.W  # should already be shaped for batch, initiated at random
                # per batch shape assertions
                assert len(X.shape) == 3
                assert len(Y.shape) == 3
                assert len(W.shape) == 3
                assert X.shape[0] == BATCH
                assert X.shape[-1] == W.shape[-2]
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
                        )
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
            print(loss)


class Pipeline(object):
    '''
    generator to pull mini batches of the txt data, totalling ~100 examples.
    Batch size = 10
    Rate =
    Features = 2
    epochs = 1000 (loops over data, loops of descent algo)

    Load up eval data???? output eval data???
    '''

    @staticmethod
    def delimited(path, delim, randomize=False, mode="classification"):
        """
        takes text file, reads line by line, and outputs array of inputs, outputs, and hot-vector
        outputs
        delim must be string...iris is space, msd is comma....
        range is the closed set of rows from which to take examples...ie [0,10]. use 0-start indexes, not numbers
        None will be return
        full range.
        if randomize, have randomize function...

        can have an argument for type of data needed

        unsure
        """
        # assert correct file types
        X, Y = [], []
        interval = None
        (path, interval) = (path[0], path[1]) if type(path) == tuple else (path, interval)
        lines = open(path).readlines()
        line_generator = enumerate((line for line in lines))
        if interval:  # for specific line intervals.
            interval[0] = interval[0] + len(lines) if interval[0] < 0 else interval[0]
            interval[1] = interval[1] + len(lines) if interval[1] < 0 else interval[1]
        for l, line in line_generator:
            if interval:
                if l < interval[0]:
                    continue
                elif l > interval[1]:
                    break
            row = [float(i) for i in line.split(delim)]
            y = row.pop(0)  # hot vectorize below this is good y for regression
            Y.append(y)  # may need to make single figure array
            X.append(row)  # y popped out already

        X, Y = (X, Y) if not randomize else randomize_helper(X, Y)
        Y = Y if mode == "regression" else hot_helper(Y)
        return np.array(X), np.array(Y)

    @staticmethod
    def cifar(path, mode="classification"):
        """
        mode can be classification or regression
        cifar has 10 classes. This function simply outputs the batch as inputs and labels.
        possibility of reshaping /processing before output if required. outputs hotvector if required.

        input is already randomized
        """
        batch = unpickle_helper(path)  # convert from binary
        # if needing to reshape or process before outputing, use this:
        # Y = []
        # X = []
        # for y, x in zip(batch[b'labels'], batch[b'data']):
        #     # if needing to reshape to channeled 2d image:
        #     #     j.reshape(3,32,32) or in some also transpose(1,2,0
        #     Y.append(y) # if only regression ok
        #     X.append(x)

        cifar_labels = {
            0: 'airplane',
            1: 'automobile',
            2: 'bird',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck'
        }
        Y = batch[b'labels']
        Y = Y if mode == "regression" else hot_helper(Y)
        X = batch[b'data']
        return X, Y, cifar_labels

    @staticmethod
    def batch_gen(X, Y, output=1, feat_h=1, feat_d=1, add_bias=True, to_standardize=True):
        """
        note that this takes in inputs X and targets Y and ouputs a batched tuple.

        assuming input shape has examples as first dim, followed by sample shape.
        only inputs and targets are batched.

        Input: BATCHES x BATCH SIZE x FEATURE HEIGHT x INPUT FEATURE WIDTH
                N/10 x 10 x 1 x 2
        Weights: BATCHES x FEATURE DEPTH x FEATURE WIDTH (WEIGHTS) x LAYER OUTPUT WIDTH (CLASSES)
                N/10 x 1 x 2 x 3
        Output: BATCHES x BATCH SIZE x FEATURE HEIGHT x LAYER OUTPUT WIDTH (CLASSES)
                N/10 x 10 x 1 x 3
        input entire input matrix, indicating whether normalized or not.

        """
        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        assert Y.shape[0] == X.shape[0]

        # slice / reshape X,Y to fit batch size...
        classes = Y.shape[-1]  # how many different classes?
        feat_w = X.shape[-1]  # number feat_w
        batch_size = BATCH
        batches = X.shape[0] // batch_size  # number batches
        ex = batches * batch_size  # new number of examples
        sliced_X = X[0:ex, ...]  # using ellipses!
        sliced_Y = Y[0:ex, ...]
        batched_X = sliced_X.reshape(batches, batch_size, feat_h, feat_w)
        batched_Y = sliced_Y.reshape(batches, batch_size, feat_h, classes)
        batch_bias = np.ones((batch_size, feat_h, 1))
        for b in range(batches):
            # i = (slice(None),) * fixed_dims + (b,)
            batch_x = batched_X[b, ...]
            batch_x = Pipeline.standardize(batch_x, mode='feature') if to_standardize else batch_x
            # print(batch_x.shape, batch_bias.shape)
            batch_x = np.concatenate((  # adding bias column.
                batch_x, batch_bias
            ), axis=-1)
            assert batch_x[:, -1].all() == 1
            batch_y = batched_Y[b, ...]
            yield (batch_x, batch_y)
            # should be iterated on in trainer, as a generator.

    @staticmethod
    def standardize(X, mode='feature'):
        """
        use a modified normalizatiown
        (only for input data)
        x = 2(x - min)/(max -min) -1

        input X

        mode
         all: all features for all examples,
         feature: each feature vector across all examples in input
         batch: each feature vector across all examples in batch, per batch

         for assertions: will need hypothesis testing module... for assertclose.
        """
        if mode == 'all':
            X = 2 * (X - X.min()) / (X.max() - X.min()) - 1
            assert X[:, -1].all() == 1
            assert X.max() == 1 and X.min() == -1
            return X
        elif mode == 'feature':
            # fixed_dims = len(X.shape) - 1
            for f in range(X.shape[-1]):
                # i = (slice(None),) * fixed_dims + (f,)  # sliceNone == ':'
                i = (..., f)
                mX = X[i].max()
                nX = X[i].min()
                X[i] = (2
                        * (X[i] - nX)
                        / (mX - nX)
                        - 1)
                mX = float(X[i].max())
                nX = float(X[i].min())
                # assert np.testing.assert_almost_equal(X[i].mean(), 0, decimal=6)
                # any assertions about mean here?
                # print(nX, mX)
                np.testing.assert_almost_equal(1.0, mX, decimal=1), \
                f"Max not 1.0, but {mX}"
                np.testing.assert_almost_equal(-1.0, nX, decimal=1), \
                f"Min not -1.0, but {nX}"
            assert X[:, -1].all() == 1
            return X
        elif mode == 'z_feature':
            for f in range(X.shape[-1]):
                i = (..., f)
                X[i] = (X[i] - X[i].mean()) / X[i].std()
                assert np.testing.assert_almost_equal(X[i].mean(), 0, decimal=7)
            assert X[:, -1].all() == 1
            return X
        else:
            raise ValueError("Choose mode from all, feature, or z_feature")


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

    @staticmethod
    def curves(independent, *dependent, ind_label, dep_label, title):
        """
        everything after unpacker must be kewword arguements!
        must call with unpacker if tupling....
        single independent variable, multiple dependent allowed
        labels/title are strings

        assuming first dependent is training, next is test

        Currently only supporting 2 depenedents....
        """
        assert iter(independent)
        assert iter(dependent)
        fig, ax1 = plt.subplots(1, 1, sharex=True, sharey=True)
        ax1.plot(independent, dependent[0], color="red", label="Training")
        ax1.plot(independent, dependent[1], color="blue", label="Test")
        ax1.set(xlabel=ind_label, ylabel=dep_label, title=title)
        plt.legend(loc="lower right", frameon=False)
        plt.show()

    @staticmethod
    def spaces_scatter(X, Y, fn, size=50):
        """
        this can only plot 3d data (2 features, 1 output as color). spaces plot (plot pts,
        model output)
        input 2d inputs and 1d y
        fn is the model function.
        :return:
        """
        x1_min, x1_max = X[:, 0].min() - .15, X[:, 0].max() + .15
        x2_min, x2_max = X[:, 1].min() - .15, X[:, 1].max() + .15
        dim = (x1_max - x1_min) / size  # mod
        # this essentially gridifies the range, but separates the arrays into layers for easy
        # array based computation
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, dim),
                               np.arange(x2_min, x2_max, dim))
        # 1's for bias?
        # X_space = np.concatenate((np.ones((xx1.shape[0] * xx1.shape[1], 1))
        #                           , np.c_[xx1.ravel(), xx2.ravel()]), axis=1)
        X_space = np.c_[xx1.ravel(), xx2.ravel()]  # listed coordinates for feeding.
        h = fn(X_space)  # will be the model forward function
        h = h.reshape(xx1.shape)  # reshaping to align with xx, yy length...kind of dumb tbh.
        plt.contourf(xx1, xx2, h)
        plt.scatter(X[:, 0], X[:, 1],
                    c=Y, edgecolor='k')  # unclear if s param needed...
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

    @staticmethod
    def histogram(*data, ind_label, dep_label, title):
        """
        each data input is tuple of len3:
        1d data and count dict and data label, although dict isn't really needed.
        everything after unpacker is kwarg
        this is a 1d histogram. hist method auto-counts the data, so no need to count separately.
        output is simple histogram
        """
        fig, ax = plt.subplots(1, 1, sharey=True, tight_layout=True)
        ax.set(xlabel=ind_label, ylabel=dep_label,
               title=title)
        c = lambda: random.random()
        for each in data:
            assert len(each) == 3, "Please ensure data input is correct tuple of tuples"
            ax.hist(each[0], bins=len(each[1].keys()), label=each[2], color=(c(), c(), c()))

        plt.legend(loc="best", frameon=False)
        plt.show()
        plt.legend()

    @staticmethod
    def confusion(confusion_array, label_dict, y_label="Predicted", x_label="Empirical"):
        """
        takes as input the confusion array (predicted x empirical) and label dict
        remeber this is assuming the keys in dict are same as indices in the y / y_hat vectors
        assert that empircal data is on x axis, and predicted on y
        """
        # assert label dict is dict
        plt.matshow(confusion_array, cmap='cool', norm=matplotlib.colors.LogNorm())
        tick_marks = np.arange(len(label_dict))
        plt.xticks(tick_marks, label_dict.values(), rotation=45)
        plt.yticks(tick_marks, label_dict.values())
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.show()


class Datasim(object):
    def __init__(self, epoch_count, label_count, label_dict=None):
        size = epoch_count * label_count ** 2
        randarr = np.random.rand((size))
        for i, j in enumerate(randarr):
            randarr[i] = 1 if j > .5 else 0
        # epoch x yhat x y
        self.epoch_count = epoch_count
        self.label_count = label_count
        self.label_dict = label_dict
        self.eval_arr = (randarr.reshape(epoch_count, label_count, label_count)
                         + np.diag([random.randint(0, 10)] * label_count))

    def accuracy_list(self):
        """
        takes the eval array,  outputs list in a list. For concatenatino purposes and the curves
        graph..
        """
        accuracies = []
        for i in range(self.eval_arr.shape[0]):
            x = self.eval_arr[i, :, :]
            assert x.shape == (
                self.eval_arr.shape[-2], self.eval_arr.shape[-1]), f'shape is {x.shape}'
            _x = x.sum(1)
            assert _x.shape == (self.eval_arr.shape[-1],), f'shape is {_x.shape}'
            _sum_acc = 0
            for j in range(x.shape[0]):
                accuracy = x[j][j] / _x[j]
                _sum_acc += accuracy
            mean_acc = _sum_acc / x.shape[0]
            accuracies.append(mean_acc)
        return accuracies

    def loss_list(self):
        pass

    @staticmethod
    def hist_list(count):
        # count is number of different items
        data = np.random.randint(1000, count + 1000, count ** 2)
        dict = {}
        for i in data:
            if i not in dict.keys():
                dict[i] = 1
            else:
                dict[i] += 1
        return data, dict, f"test{random.randint(0,9)}"

    @staticmethod
    def data_2d(count):
        """
        This outputs X, Y, and Y_hat
        """
        X, Y = [], []
        for i in range(count):
            # random.seed(134314)
            x_1 = random.random()
            # random.seed(2434567)
            x_2 = random.random()
            X.append([x_1, x_2])
        for x_1, x_2 in X:
            y = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
            y = 2 if x_2 > .5 else y
            Y.append(y)
        return np.array(X), np.array(Y)

    @staticmethod
    def grid_data(X_model, jitter=3415e-5):
        """
        idea here is to bring in X_model, which is generated from plotting function. X_model is
        must output np array
        """
        # j = lambda: jitter * -1 if random.random() > .5 else jitter
        j = lambda: 1

        Y_hat = []
        for x_1, x_2 in X_model:
            y = 1 if x_1 < (0.2 * j()) or x_1 > (0.8 * j()) else 0
            y = 2 if x_2 > (.5 * j()) else y
            Y_hat.append(y)
        return np.array(Y_hat)

    def label_dict(self):
        assert self.label_dict, "No Dict"
        return self.label_dict

    def epoch_list(self):
        return [i for i in range(self.epoch_count)]

    def confusion(self):
        return self.eval_arr.sum(0)


# def __init__(
#         self,
#         inputs,
#         targets,
#         test_inputs,
#         test_targets,
#         loss_fn, cross
#         output_fn, soft
#         logit_fn, matmul
#         feat_d=1,
#         n_classes=1
# ):

inputs, targets = Pipeline.delimited(RESOURCES["iris_train"],' ', True)
test_inputs, test_targets = Pipeline.delimited(RESOURCES["iris_test"],' ', True)


first = Single_Layer_Network(inputs,targets,test_inputs,test_targets,Cross_Entropy,
                             Softmax_Regression,MatMul,n_classes=3)

first.train(0)





# in minitorch, the network is a module class which stores the bias and weights...essentially
# each layer stores weights / bias....and the weights store the grad..
#
#
# Is it accurate to think of each class def as an encapsulation or frame, and If python doesn't
# find a binding in the present frame, it moves up to the parent, and keeps going up to find it?
# And while the subclass has access to all these (class level, not instance level) things,
# it's place in the structure is essentially just set by whatever is passed into the initial
# "class ClassName(ParentClass):" definition? So both the Superclasses and Subclasses exist in
# the same space, the super/sub relationship is just defined by that one link....if that makes
# sense? Sorry if that's confusing.  Thank
#
#
# currying is returning a function..
#
#
#
# rememeber each output example in the batch has a gradient with rt to weights input
# etc....event thought they're being evaluated at the same weight values.....and so we are
# summing across the whole ouput example batch, to get an overall loss, so that we can get in
# a sense an overall gradient. if we average the loss across the batch then we get the
# average gradient wrt to weights. ...so if we've been given that value, then we can ...
#  if wwe valuate it at a specific random value, we' can see then the contribution of shift
# in each weight to the average loss....of the batch, and so that value, which necessarily
# must be the size of teh weight matrix, can be multiplied by rate and subtracted