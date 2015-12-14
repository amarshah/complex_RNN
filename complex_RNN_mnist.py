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

# Warning: assumes n_batch is a divisor of number of data points
# Suggestion: preprocess outputs to have norm 1 at each time step
def main(n_iter, n_batch, n_hidden, time_steps, learning_rate, savefile, scale_penalty, use_scale, reload_progress, model, n_hidden_lstm):



    np.random.seed(1234)
    #import pdb; pdb.set_trace()
    # --- Set optimization params --------

    # --- Set data params ----------------
    n_input = 1
    n_output = 10
    ##### MNIST processing ################################################      

        
    # load and preprocess the data
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = cPickle.load(gzip.open("mnist.pkl.gz", 'rb'))
    n_data = train_x.shape[0]
    num_batches = n_data / n_batch
    
    # shuffle data order
    inds = range(n_data)
    np.random.shuffle(inds)
    train_x = np.ascontiguousarray(train_x[inds, :time_steps])
    train_y = np.ascontiguousarray(train_y[inds])
    n_data_valid = valid_x.shape[0]
    inds_valid = range(n_data_valid)
    np.random.shuffle(inds_valid)
    valid_x = np.ascontiguousarray(valid_x[inds_valid, :time_steps])
    valid_y = np.ascontiguousarray(valid_y[inds_valid])

    # reshape x
    train_x = np.reshape(train_x.T, (time_steps, n_data, 1))
    valid_x = np.reshape(valid_x.T, (time_steps, valid_x.shape[0], 1))

    # change y to one-hot encoding
    temp = np.zeros((n_data, n_output))
    # import pdb; pdb.set_trace()
    temp[np.arange(n_data), train_y] = 1
    train_y = temp.astype('float32')

    temp = np.zeros((n_data_valid, n_output)) 
    temp[np.arange(n_data_valid), valid_y] = 1
    valid_y = temp.astype('float32')
    
    # Random permutation of pixels
    P = np.random.permutation(time_steps)
    train_x = train_x[P, :, :]
    valid_x = valid_x[P, :, :]

   #######################################################################

    # --- Compile theano graph and gradients
 
    gradient_clipping = np.float32(1)
    if (model == 'LSTM'):   
        inputs, parameters, costs = LSTM(n_input, n_hidden_LSTM, n_output)
    elif (model == 'complex_RNN'):
        gradient_clipping = np.float32(100000)
        inputs, parameters, costs = complex_RNN(n_input, n_hidden, n_output, scale_penalty)
    elif (model == 'complex_RNN_LSTM'):
        inputs, parameters, costs = complex_RNN_LSTM(n_input, n_hidden, n_hidden_lstm, n_output, scale_penalty)
    elif (model == 'IRNN'):
        inputs, parameters, costs = IRNN(n_input, n_hidden, n_output)
    elif (model == 'RNN'):
        inputs, parameters, costs = RNN(n_input, n_hidden, n_output)
    else:
        print "Unsuported model:", model
        return
   
    gradients = T.grad(costs[0], parameters)

#   GRADIENT CLIPPING
    gradients = gradients[:7] + [T.clip(g, -gradient_clipping, gradient_clipping)
            for g in gradients[7:]]
 
    s_train_x = theano.shared(train_x)
    s_train_y = theano.shared(train_y)

    s_valid_x = theano.shared(valid_x)
    s_valid_y = theano.shared(valid_y)


    # --- Compile theano functions --------------------------------------------------

    index = T.iscalar('i')

    updates, rmsprop = rms_prop(learning_rate, parameters, gradients)

    givens = {inputs[0] : s_train_x[:, n_batch * index : n_batch * (index + 1), :],
              inputs[1] : s_train_y[n_batch * index : n_batch * (index + 1), :]}

    givens_valid = {inputs[0] : s_valid_x,
                   inputs[1] : s_valid_y}
   
    
    train = theano.function([index], [costs[0], costs[2]], givens=givens, updates=updates)
    valid = theano.function([], [costs[1], costs[2]], givens=givens_valid)

    #import pdb; pdb.set_trace()

    # --- Training Loop ---------------------------------------------------------------
    train_loss = []
    test_loss = []
    test_acc = []
    best_params = [p.get_value() for p in parameters]
    best_test_loss = 1e6
    for i in xrange(n_iter):
     #   pdb.set_trace()

        [cross_entropy, acc] = train(i % num_batches)
        train_loss.append(cross_entropy)
        print "Iteration:", i
        print "cross_entropy:", cross_entropy
        print "accurracy", acc * 100
        print

        if (i % 100==0):
            [valid_cross_entropy, valid_acc] = valid()
            print
            print "VALIDATION"
            print "cross_entropy:", valid_cross_entropy
            print "accurracy", valid_acc * 100
            print 
            test_loss.append(valid_cross_entropy)
            test_acc.append(valid_acc)

            if valid_cross_entropy < best_test_loss:
                print "NEW BEST!"
                best_params = [p.get_value() for p in parameters]
                best_test_loss = valid_cross_entropy

            save_vals = {'parameters': [p.get_value() for p in parameters],
                         'rmsprop': [r.get_value() for r in rmsprop],
                         'train_loss': train_loss,
                         'test_loss': test_loss,
                         'best_params': best_params,
                         'test_acc': test_acc,
                         'best_test_loss': best_test_loss}

            cPickle.dump(save_vals,
                         file(savefile, 'wb'),
                         cPickle.HIGHEST_PROTOCOL)





    
if __name__=="__main__":
    kwargs = {'n_iter': 1000000,
              'n_batch': 20,
              'n_hidden': 512,
              'time_steps': 28*28,
              'learning_rate': np.float32(0.0005),
              'savefile': '/data/lisatmp3/arjovskm/complex_RNN/2015-11-08-IRNN-permuted_mnist.pkl',
              'scale_penalty': 5,
              'use_scale': True,
              'reload_progress': True,
              'model': 'complex_RNN',
              'n_hidden_lstm': 100}
    main(**kwargs)
