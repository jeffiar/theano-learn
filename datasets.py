import os, struct
from array import array as pyarray 
from numpy import append, array, int8, uint8, zeros, asarray
import theano
import cPickle as pickle

DATA_PATH = os.environ['HOME'] + "/.data"

## Taken from http://g.sweyla.com/blog/2012/mnist-numpy/ (and changed a few lines)
def mnist(dataset="training", path=os.path.join(DATA_PATH, "mnist"), asbytes=False, selection=None, flatten=True):
    """
    Loads MNIST files into a 3D numpy array.

    You have to download the data separately from [MNIST]_. It is recommended
    to set the environment variable ``MNIST`` to point to the folder where you
    put the data, so that you don't have to select path. On a Linux+bash setup,
    this is done by adding the following to your ``.bashrc``::

        export MNIST=/path/to/mnist

    Parameters
    ----------
    dataset : str 
        Either "training" or "testing", depending on which dataset you want to
        load. 
    path : str 
        Path to your MNIST datafiles. The default is ``None``, which will try
        to take the path from your environment variable ``MNIST``. The data can
        be downloaded from http://yann.lecun.com/exdb/mnist/.
    asbytes : bool
        If True, returns data as ``numpy.uint8`` in [0, 255] as opposed to
        ``numpy.float64`` in [0.0, 1.0].
    selection : slice
        Using a `slice` object, specify what subset of the dataset to load. An
        example is ``slice(0, 20, 2)``, which would load every other digit
        until--but not including--the twentieth.
    flatten : bool
        Specify whether to return each data sample as a 1-by-784 vector rather than
        in the standard 28 by 28 matrix grid

    Returns
    -------
    images : ndarray
        Image data of shape ``(N, rows, cols)``, where ``N`` is the number of images.
    labels : ndarray
        Array of size ``N`` describing the labels.

    Examples
    --------
    Assuming that you have downloaded the MNIST database and set the
    environment variable ``$MNIST`` point to the folder, this will load all
    images and labels from the training set:

    >>> images, labels = datasets.load_mnist('training') # doctest: +SKIP

    """
    
    if dataset not in ["training", "testing"]:
        raise ValueError("Data set must be 'testing' or 'training'")

    # if we've created and pickled the datasets already, load them from cache
    filename = "data/mnist_" + dataset + ".pkl"
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            pic = pickle.load(f)
        return pic['images'], pic['labels']

    # The files are assumed to have these names and should be found in 'path'
    files = {
        'training': ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte'),
        'testing': ('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'),
    }

    if path is None:
        try:
            path = os.environ['MNIST']
        except KeyError:
            raise ValueError("Unspecified path requires environment variable $MNIST to be set")

    images_fname = os.path.join(path, files[dataset][0])
    labels_fname = os.path.join(path, files[dataset][1])

    with open(labels_fname, 'rb') as flbl:
        magic_nr, size = struct.unpack(">II", flbl.read(8))
        labels_raw = pyarray("b", flbl.read())

    with open(images_fname, 'rb') as fimg:
        magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
        images_raw = pyarray("B", fimg.read())

    indices = range(size)
    if selection:
        indices = indices[selection] 
    N = len(indices)

    images = zeros((N, rows, cols), dtype=int)
    labels = zeros((N), dtype=int)
    for i, index in enumerate(indices):
        images[i] = array(images_raw[ indices[i]*rows*cols : (indices[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = labels_raw[indices[i]]

    if not asbytes:
        images = images.astype(theano.config.floatX)/255.0

    if flatten:
        images = asarray([image.flatten() for image in images])

    # write pickled objects so we don't have to create them again
    with open(filename, 'wb') as f:
        pickle.dump({'images':images, 'labels':labels}, f, 2)
    return (images, labels)

if __name__=="__main__":
    mnist()
    print 1 
    mnist('testing')
    print 2 
