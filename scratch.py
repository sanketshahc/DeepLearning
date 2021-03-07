import pickle
import numpy
import random

# paths to data
resources = {
    "iris": "resources/iris-train.txt",
    "cifar": "resources/cifar-10-batches-py/data_batch_1",
    "msd": "resources/YearPredictionMSD.txt"
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
    rand = [[]for s in sets] # new sets
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
    delim must be string

    if randomize, have randomize function...

    can have an argument for type of data needed

    unsure
    """
    # assert correct file types
    file = open(path)
    lines = file.readlines()
    Y = []
    X = []
    for line in lines:
        row = [float(i) for i in line.split(delim)]
        y = row.pop(0)  # hot vectorize below this is good y for regression
        Y.append(y)  # may need to make single figure array
        X.append(row)  # y popped out already

    X, Y = X, Y if not randomize else randomize(X,Y)
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

    Y = batch[b'labels']
    Y = Y if mode == "regression" else hot_helper(Y)
    X = batch[b'data']
    return X, Y

