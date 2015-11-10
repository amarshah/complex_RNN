import cPickle
import gzip
import theano
import pdb
from fftconv import cufft, cuifft
import numpy as np
import theano.tensor as T
from theano.ifelse import ifelse
from models import *
from optimizations import *
import os
import math
from math import log

def onehot(x,numclasses=None):                                                                               
    x = np.array(x)
    if x.shape==():                                                                                          
        x = x[np.newaxis]
    if numclasses is None:
        numclasses = x.max() + 1
    result = np.zeros(list(x.shape) + [numclasses],dtype=theano.config.floatX)
    z = np.zeros(x.shape)
    for c in range(numclasses):
        z *= 0
        z[np.where(x==c)] = 1
        result[...,c] += z
    return np.float32(result)

def get_data(seq_length=200, framelen=1):
    alphabetsize = 50
    
    datadir = os.environ['FUEL_DATA_PATH']
    data = np.load(os.path.join(datadir, 'PennTreebankCorpus/char_level_penntree.npz'))
    trainset = data['train']
    validset = data['valid']

    # end of sentence: \n
    allletters = " etanoisrhludcmfpkgybw<>\nvN.'xj$-qz&0193#285\\764/*"
    dictionary = dict(zip(list(set(allletters)), range(alphabetsize)))
    invdict = {v: k for k, v in dictionary.items()}
    # add all possible 2-grams to dataset
    #all2grams = ''.join([a+b for b in allletters for a in allletters])
    #trainset = np.hstack((np.array([dictionary[c] for c in all2grams]), trainset))

    numtrain = len(trainset) - (len(trainset) % seq_length)
    numvalid = len(validset) - (len(validset) % seq_length)
    
    numvis = framelen * alphabetsize
    train_features_numpy = onehot(trainset[:numtrain]).reshape(-1, alphabetsize * seq_length * framelen)
    valid_features_numpy = onehot(validset[:numvalid]).reshape(-1, alphabetsize * seq_length * framelen)
    
    del trainset, validset
    
    np.random.shuffle(train_features_numpy)
    np.random.shuffle(valid_features_numpy)

    train_features = np.swapaxes(np.asarray([ex.reshape(seq_length * framelen, alphabetsize) for ex in train_features_numpy]), 0, 1)
    valid_features = np.swapaxes(np.asarray([ex.reshape(seq_length * framelen, alphabetsize) for ex in valid_features_numpy]), 0, 1)

    
    return [train_features, valid_features]






# Warning: assumes n_batch is a divisor of number of data points
# Suggestion: preprocess outputs to have norm 1 at each time step
def main(n_iter, n_batch, n_hidden, n_hidden_LSTM, time_steps, learning_rate, savefile, scale_penalty, use_scale):

 
    # --- Set optimization params --------
    gradient_clipping = np.float32(50000)

    # --- Set data params ----------------
    n_input = 50
    n_output = 50
  
    [train_x, valid_x] = get_data(seq_length=time_steps)
    
    num_batches = train_x.shape[1] / n_batch - 1
    
    train_y = train_x[1:,:,:]
    train_x = train_x[:-1,:,:]

    valid_y = valid_x[1:,:,:]
    valid_x = valid_x[:-1,:,:]

   #######################################################################

    # --- Compile theano graph and gradients
 
    inputs, parameters, costs = complex_RNN_LSTM(n_input, n_hidden, n_hidden_LSTM, n_output, scale_penalty, out_every_t=True)
    if not use_scale:
        parameters.pop() 
   
#    import pdb; pdb.set_trace()
    gradient_clipping = np.float32('1.0')
    gradients = T.grad(costs[0], parameters)
    gradients = gradients[:7] + [T.clip(g, -gradient_clipping, gradient_clipping)
                                for g in gradients[7:]]
 


    s_train_x = theano.shared(train_x, borrow=True)
    s_train_y = theano.shared(train_y, borrow=True)

    s_valid_x = theano.shared(valid_x, borrow=True)
    s_valid_y = theano.shared(valid_y, borrow=True)

#    import pdb; pdb.set_trace()

    # --- Compile theano functions --------------------------------------------------

    index = T.iscalar('i')

    updates, rmsprop = rms_prop(learning_rate, parameters, gradients)

    givens = {inputs[0] : s_train_x[:, n_batch * index : n_batch * (index + 1), :],
              inputs[1] : s_train_y[:, n_batch * index : n_batch * (index + 1), :]}

    givens_valid = {inputs[0] : s_valid_x,
                    inputs[1] : s_valid_y}
    
       
    train = theano.function([index], [costs[1], costs[2]], givens=givens, updates=updates)
    valid = theano.function([], [costs[1], costs[2]], givens=givens_valid)

    # --- Training Loop ---------------------------------------------------------------
    train_loss = []
    valid_loss = []
    best_params = [p.get_value() for p in parameters]
    best_valid_loss = 1e6
    for i in xrange(n_iter):
     #   pdb.set_trace()
        if (i % num_batches == 0):
            # SHUFFLE AFTER EVERY EPOCH
            P = np.random.permutation(train_x.shape[1])
            s_train_x = s_train_x[:,P,:]
            s_train_y = s_train_y[:,P,:]

        [cent, acc] = train(i % num_batches)
        train_loss.append(cent)
        print "Iteration:", i
        print "Bits per character:", log(math.e, 2.0) * cent
        print "Test Perplexity:", 2.0**(5.6 * log(math.e, 2.0) * cent)
        print "accuracy:", 100 * acc
        print

        if (i % 50==0):
            [cent, acc] = valid()
            print
            print "VALIDATION"
            print "Bits per character:", log(math.e, 2.0) * cent
            print "Test Perplexity:", 2.0**(5.6 * log(math.e, 2.0) * cent)
            print "accuracy:", 100 * acc
            print
            valid_loss.append(cent)

            if cent < best_valid_loss:
                best_params = [p.get_value() for p in parameters]
                best_valid_loss = cent
                print "BEST VALUE!"

            save_vals = {'parameters': [p.get_value() for p in parameters],
                         'rmsprop': [r.get_value() for r in rmsprop],
                         'train_loss': train_loss,
                         'valid_loss': valid_loss,
                         'best_params': best_params,
                         'best_valid_loss': best_valid_loss}

            cPickle.dump(save_vals,
                         file(savefile, 'wb'),
                         cPickle.HIGHEST_PROTOCOL)


    
if __name__=="__main__":
    kwargs = {'n_iter': 50000,
              'n_batch': 20,
              'n_hidden': 10000,
              'n_hidden_LSTM': 256,
              'time_steps': 500,
              'learning_rate': np.float32(0.001),
              'savefile': '/data/lisatmp3/arjovskm/complex_RNN/penn_tree_500.pkl',
              'scale_penalty': 100,
              'use_scale': True}

    main(**kwargs)
