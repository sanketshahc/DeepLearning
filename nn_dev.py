import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import random

BATCH = 10

# paths to data
# if tuples, second term is indices of file to take"
resources = {
    "iris_train": "resources/iris-train.txt",
    "iris_test": "resources/iris-test.txt",
    "cifar_train": "resources/cifar-10-batches-py/data_batch_1",
    "cifar_test": "resources/cifar-10-batches-py/test_batch",
    "msd_train": ("resources/YearPredictionMSD.txt", [0, 463713]),
    "msd_test": ("resources/YearPredictionMSD.txt", [-51630, -1])
}

def randomize(*sets):
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

    X, Y = X, Y if not randomize else randomize(X, Y)
    Y = Y if mode == "regression" else hot_helper(Y)
    return X, Y


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
    classes = Y.shape[-1] # how many different classes?
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
        batch_x = standardize(batch_x, mode='feature') if to_standardize else batch_x
        print(batch_x.shape, batch_bias.shape)
        batch_x = np.concatenate((     # adding bias column.
            batch_x, batch_bias
        ), axis = -1)
        assert batch_x[:, -1].all() == 1
        batch_y = batched_Y[b, ...]
        yield (batch_x, batch_y)
        # should be iterated on in trainer, as a generator.


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
            print(nX,mX)
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
            ax.hist(each[0], bins=len(each[1].keys()), label = each[2], color = (c(),c(),c()))

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


class Datagen(object):
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


######

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
data = Datagen(1000, 10, label_dict=cifar_labels)
X, Y = data.data_2d(100)
