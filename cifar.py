import pickle
import matplotlib.pyplot as plt
import numpy

path = 'cifar-10-batches-py/data_batch_1'
# path to the data batch, in binary format

# changing from binary to python dictionary of (batch, label, data, filename)
def unpickle(file):
    """input file path, output dictionary object, of labels, data, filenames """
    with open(file, 'br') as f:
        d = pickle.load(f, encoding='bytes') # need to specify encoding
    return d

# helper function to check when to stop iterating through batch (counting function)
def count_elements(coll, s=0):
    for nested in coll:
        if type(nested) == list:
            s += len(nested)
            count_elements(nested)
    return s


# Main Function...wrap
batch = unpickle(path) # convert from binary
arr2d = [[], [], [], [], [], [], [], [], [], []] # note that [[]]*10 makes 10 copies lol, no good
while count_elements(arr2d) < 30:
    for i,j in zip(batch[b'labels'], batch[b'data']):
        # iterate through zip. output array is
        # indexed by label number
        if len(arr2d[i]) < 3:
            # print(i)
            arr2d[i].append(
                numpy.transpose(j.reshape(3, 32, 32), (1,2,0)))
            #reshape then transpose,
            # move 3 to end
        else:
            None


# turns subplots into images from data from array
def image_helper(data_arr,fig, subs):
    for i, row in enumerate(data_arr):
        for j, data in enumerate(row):
            subs[j][i].set_xticks([]) # reformatting subplot (remove ticks)
            subs[j][i].set_yticks([])
            subs[j][i].imshow(data)

    fig.show()
    fig.savefig('grid.png')

figure = plt.figure(figsize=(10,3), dpi = 600) # make figure object, sized right
fig_subs = figure.subplots(3, 10) # add subplots
figure.subplots_adjust(wspace=0, hspace=0) # some fomratting
image_helper(arr2d, figure, fig_subs)



#     ok, so iterate through pairs, if i in array is less than 4 items,
#     then reshape, transpose rebind, then add to index i in arr2d
#     if less than 4....once all are over 3, then move to the image concatenation function

# image concatenation
# next function is concatentation, cat the inner arrays, and vertcat the outer ones....
# then plot the image...!


# instantiate numpy array for final array?
# numpy
# get dictionary,
# must check label array, create a zip,
# iterate through pairs, add image to appropriate index in 2d array,
