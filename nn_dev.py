import pickle
import numpy as np
from numpy import ma
import matplotlib
from matplotlib import ticker, cm
import matplotlib.pyplot as plt
import random

# paths to data
# if tuples, second term is indices of file to take"
resources = {
    "iris_train": "resources/iris-train.txt",
    "iris_test": "resources/iris-test.txt",
    "cifar_train": "resources/cifar-10-batches-py/data_batch_1",
    "cifar_test": "",
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
    line_generator = enumerate((line for line in open(path).readlines()))
    if interval:
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

    return Y_hot


### cifar:
# changing from binary to python dictionary of (batch, label, data, filename)
def unpickle(file):
    """input file path, output dictionary object, of labels, data, filenames """
    with open(file, 'br') as f:
        d = pickle.load(f, encoding='bytes')  # need to specify encoding
    return d


# helper function to check when to stop iterating through batch (counting function for total
# images in arr2d)
def count_elements(collection, s=0):
    """
    counts total first dimension elements in a multidimensional array
    """
    for nested in collection:
        if type(nested) == list:
            s += len(nested)
            count_elements(nested)

    return s


def cifar(path, mode="classification"):
    """
    mode can be classification or regression
    cifar has 10 classes. This function simply outputs the batch as inputs and labels.
    possibility of reshaping /processing before output if required. outputs hotvector if required.

    input is already randomized
    """
    batch = unpickle(path)  # convert from binary
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
        # Taken from towardsdatascience blogpost, but modified >>>
        x1_min, x1_max = X[:, 0].min() -.15, X[:, 0].max()+ .15
        x2_min, x2_max = X[:, 1].min() -.15, X[:, 1].max()+.15
        dim = (x1_max - x1_min) / size  # mod
        # this essentially gridifies the range, but separates the arrays into layers for easy
        # array based computation
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, dim),
                               np.arange(x2_min, x2_max, dim))
        # 1's for bias?
        # X_space = np.concatenate((np.ones((xx1.shape[0] * xx1.shape[1], 1))
        #                           , np.c_[xx1.ravel(), xx2.ravel()]), axis=1)
        # # <<< Taken from towardsdatascience blogpost, but modified

        X_space = np.c_[xx1.ravel(), xx2.ravel()] #listed coordinates for feeding.
        h = fn(X_space)  # will be the model forward function
        h = h.reshape(xx1.shape)  # reshaping to align with xx, yy length...kind of dumb tbh.
        plt.contourf(xx1, xx2, h)
        plt.scatter(X[:, 0], X[:, 1],
                    c=Y, edgecolor='k')  # unclear if s param needed...
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

    @staticmethod
    def histogram(data, ind_label, dep_label, title):
        """
        this is a 1d histogram. input is a tuple of 1d on the numberlinespace, and then a dict of counts
        assert that data
        output is simple histogram
        """
        fig, ax = plt.subplots(1, 1, sharey=True, tight_layout=True)
        ax.hist(data[0], bins=len(data[1].keys()))
        ax.set(xlabel=ind_label, ylabel=dep_label,
               title=title)
        plt.show()

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
            assert x.shape == (self.eval_arr.shape[-2], self.eval_arr.shape[-1]), f'shape is {x.shape}'
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
        return data, dict

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
data = Datagen(1000,10,label_dict=cifar_labels)
X, Y = data.data_2d(100)
#
# N = 100
# x = np.linspace(-3.0, 3.0, N)
# y = np.linspace(-2.0, 2.0, N)
#
# X, Y = np.meshgrid(x, y)
#
# # A low hump with a spike coming out.
# # Needs to have z/colour axis on a log scale so we see both hump and spike.
# # linear scale only shows the spike.
# Z1 = np.exp(-X**2 - Y**2)
# Z2 = np.exp(-(X * 10)**2 - (Y * 10)**2)
# z = Z1 + 50 * Z2
#
# # Put in some negative values (lower left corner) to cause trouble with logs:
# z[:5, :5] = -1
#
# # The following is not strictly essential, but it will eliminate
# # a warning.  Comment it out to see the warning.
# z = ma.masked_where(z <= 0, z)
#
#
# # Automatic selection of levels works; setting the
# # log locator tells contourf to use a log scale:
# fig, ax = plt.subplots()
# cs = ax.contourf(X, Y, z, locator=ticker.LogLocator(), cmap=cm.PuBu_r)
#
# # Alternatively, you can manually set the levels
# # and the norm:
# # lev_exp = np.arange(np.floor(np.log10(z.min())-1),
# #                    np.ceil(np.log10(z.max())+1))
# # levs = np.power(10, lev_exp)
# # cs = ax.contourf(X, Y, z, levs, norm=colors.LogNorm())
#
# cbar = fig.colorbar(cs)
#
# plt.show()
