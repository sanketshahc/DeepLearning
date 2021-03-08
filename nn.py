import numpy as np

# hyper parameters
EPOCHS =
BATCH =
RATE =
MOMENTUM =
velocity =
DECAY =

class Function(object):
    def __init__(self):
        pass

    def apply_forward(self,*args):
        pass
        ## apply forward....save inputs?


class Softmax_Regression(Function):
    def __init__(self):
        # any overriding or additional attrs
        pass

    def forward(self,*args):
        pass

    def backward(self,*args):
        pass


class Poisson_Regression(Function):
    """":type
    Log-Linear (generalized linear function)...not totally sure what that means.
    """
    def __init__(self):
        pass

    def forward(self,*args):
        pass

    def backward(self,*args):
        pass


class Ridge_Regression(Function):
    def __init__(self):
        pass

    def forward(self,*args):
        pass

    def backward(self,*args):
        pass


class Poisson_Loss(Function):
    def __init__(self):
        pass

    def forward(self,*args):
        pass

    def backward(self,*args):
        pass


class Cross_Entropy(Function):
    def __init__(self):
        pass

    def forward(self,*args):
        pass

    def backward(self,*args):
        pass


class Mean_Squared_Error(Function):
    def __init__(self):
        pass

    def forward(self,*args):
        pass

    def backward(self,*args):
        pass


class Regularize(Function):
    def __init__(self):
        pass

    def forward(self,*args):
        pass

    def backward(self,*args):
        pass


class Loss_Function(Function):
    def __init__(self):
        pass

    def regularize(self,*args):
        pass

    def cross_entropy(self,*args):
        pass

    def mean_squared_error(self,*args):
        pass

    def poisson(self,*args):
        pass


# May not need wrapper
class Tensor(object):
    def __init__(self):
        pass

    def history(self,*args):
        pass
#   Any expansion to Numpy capability wrapped here

class Network(object):
    """"
    this wil have to store the model weight and bias values
    this will have a forward call. something like the Linear class in the minitorch.
    """
    def __init__(self):

        pass
    # consider putting below functions in class also


# consider nesting training functions in 1 function
def train(*args):
    """
    standardize data (z = (x-u)/s
    shape the data arrays
        weights = n/10 x 10 x 3 x 2
        input = n/10 x 10 x 1 x 2
        output = n/10 x 10 x 3 x 1
    call forward:
    a = weights * input + bias
    softmax (a)

    loss equations (against training data) minus the regularization norm
        additionally to a loss equation,
        mean per class accuracy (max index vs hot v index)
    derivitave of loss
    """
    pass

# gradient descent
#   remember to average gradient across batch before subtracting from loss (loss is summed over
#   batch....
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
    #
    #
    pass



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
        pass




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
        # Plotting decision regions
        # x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        # y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        # xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
        #                      np.arange(y_min, y_max, 0.1))
        # X = np.concatenate((np.ones((xx.shape[0] * xx.shape[1], 1))
        #                     , np.c_[xx.ravel(), yy.ravel()]), axis=1)
        # h = hypothesis(theta, X, 2)
        # h = h.reshape(xx.shape)
        # plt.contourf(xx, yy, h)
        # plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
        #             s=30, edgecolor='k')
        # plt.xlabel("Marks obtained in 1st Exam")
        # plt.ylabel("Marks obtained in 2nd Exam")
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

        # labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        #
        # def plot_confusion_matrix(df_confusion, title='Confusion Matrix', xlabels=labels, ylabels=labels, cmap='Blues'):
        #     plt.matshow(df_confusion, cmap=cmap) # imshow
        #     plt.colorbar()
        #     tick_marks = np.arange(len(df_confusion.columns))
        #     plt.xticks(tick_marks, labels, rotation=45)
        #     plt.yticks(tick_marks, labels)
        #     plt.ylabel(df_confusion.index.name)
        #     plt.xlabel(df_confusion.columns.name)
        #
        # # Source: https://stackoverflow.com/questions/2148543/how-to-write-a-confusion-matrix-in-python


        pass



# Testing module
#   testing for each function

#todo test the data functions



# in minitorch, the network is a module class which stores the bias and weights...essentially
# each layer stores weights / bias....and the weights store the grad..


#Is it accurate to think of each class def as an encapsulation or frame, and If python doesn't
# find a binding in the present frame, it moves up to the parent, and keeps going up to find it?
# And while the subclass has access to all these (class level, not instance level) things,
# it's place in the structure is essentially just set by whatever is passed into the initial
# "class ClassName(ParentClass):" definition? So both the Superclasses and Subclasses exist in
# the same space, the super/sub relationship is just defined by that one link....if that makes
# sense? Sorry if that's confusing.  Thank


#currying is returning a function..