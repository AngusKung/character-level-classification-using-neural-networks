
import numpy as np
import pdb

from sklearn.metrics import precision_recall_fscore_support


def predictNtransform(model, x, y):
    print "Predicting yhat..."
    y_hat = model.predict(x)
    y = np.argmax(y, axis=1)
    y_hat = np.argmax(y_hat, axis=1)
    return y, y_hat
   

def calPrecisionRecall(y, y_hat, dataset_name):
    print "\n",dataset_name,":"
    pr = precision_recall_fscore_support(y, y_hat)
    print "Precision\t",'\t'.join(["{0:.3f}".format(num) for num in pr[0]])
    print "Recall\t",'\t'.join(["{0:.3f}".format(num) for num in pr[1]])
    return pr


def readMemmap():
    xtra_shape = (26010, 452)
    ytra_shape = (26010, 12)
    xval_shape = (6503, 452)
    yval_shape = (6503, 12)
    
    xtra_fn = "./memmap/xtra_"+str(xtra_shape)
    ytra_fn = "./memmap/ytra_"+str(ytra_shape)
    xval_fn = "./memmap/xval_"+str(xval_shape)
    yval_fn = "./memmap/yval_"+str(yval_shape)

    # Read data from memmap
    xtra = np.memmap(xtra_fn, dtype='float32', mode='r', shape=xtra_shape)
    ytra = np.memmap(ytra_fn, dtype='float32', mode='r', shape=ytra_shape)
    xval = np.memmap(xval_fn, dtype='float32', mode='r', shape=xval_shape)
    yval = np.memmap(yval_fn, dtype='float32', mode='r', shape=yval_shape)

    return xtra, ytra, xval, yval
