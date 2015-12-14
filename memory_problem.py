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
import argparse, timeit


def generate_data(time_steps, n_data, n_sequence):
    seq = np.random.randint(1, high=9, size=(n_data, n_sequence))
    zeros1 = np.zeros((n_data, time_steps-1))
    zeros2 = np.zeros((n_data, time_steps))
    marker = 9 * np.ones((n_data, 1))
    zeros3 = np.zeros((n_data, n_sequence))

    x = np.concatenate((seq, zeros1, marker, zeros3), axis=1).astype('int32')
    y = np.concatenate((zeros3, zeros2, seq), axis=1).astype('int32')
    
    return x.T, y.T

    
def main(n_iter, n_batch, n_hidden, time_steps, learning_rate, savefile, model, input_type, out_every_t, loss_function):

    # --- Set data params ----------------
    n_input = 10
    n_output = 9
    n_sequence = 10
    n_train = int(1e5)
    n_test = int(1e4)
    num_batches = int(n_train / n_batch)
  

    # --- Create data --------------------
    train_x, train_y = generate_data(time_steps, n_train, n_sequence)
    test_x, test_y = generate_data(time_steps, n_test, n_sequence)

    s_train_x = theano.shared(train_x)
    s_train_y = theano.shared(train_y)

    s_test_x = theano.shared(test_x)
    s_test_y = theano.shared(test_y)

    
    # --- Create theano graph and compute gradients ----------------------

    gradient_clipping = np.float32(1)

    if (model == 'LSTM'):           
        inputs, parameters, costs = LSTM(n_input, n_hidden, n_output, input_type=input_type,
                                         out_every_t=out_every_t, loss_function=loss_function)
        gradients = T.grad(costs[0], parameters)
        gradients = [T.clip(g, -gradient_clipping, gradient_clipping) for g in gradients]

    elif (model == 'complex_RNN'):
        inputs, parameters, costs = complex_RNN(n_input, n_hidden, n_output, input_type=input_type,
                                                out_every_t=out_every_t, loss_function=loss_function)
        gradients = T.grad(costs[0], parameters)

    elif (model == 'IRNN'):
        inputs, parameters, costs = IRNN(n_input, n_hidden, n_output, input_type=input_type,
                                         out_every_t=out_every_t, loss_function=loss_function)
        gradients = T.grad(costs[0], parameters)
        gradients = [T.clip(g, -gradient_clipping, gradient_clipping) for g in gradients]

    elif (model == 'RNN'):
        inputs, parameters, costs = tanhRNN(n_input, n_hidden, n_output, input_type=input_type,
                                            out_every_t=out_every_t, loss_function=loss_function)
        gradients = T.grad(costs[0], parameters)
        gradients = [T.clip(g, -gradient_clipping, gradient_clipping) for g in gradients]
    
    else:
        print "Unsuported model:", model
        return

 
    # --- Compile theano functions --------------------------------------------------

    index = T.iscalar('i')

    updates, rmsprop = rms_prop(learning_rate, parameters, gradients)

    givens = {inputs[0] : s_train_x[:, n_batch * index : n_batch * (index + 1)],
              inputs[1] : s_train_y[:, n_batch * index : n_batch * (index + 1)]}

    givens_test = {inputs[0] : s_test_x,
                   inputs[1] : s_test_y}
    
   
    
    train = theano.function([index], costs[0], givens=givens, updates=updates)
    test = theano.function([], [costs[0], costs[1]], givens=givens_test)

    # --- Training Loop ---------------------------------------------------------------

    train_loss = []
    test_loss = []
    test_acc = []
    best_params = [p.get_value() for p in parameters]
    best_rms = [r.get_value() for r in rmsprop]
    best_test_loss = 1e6
    for i in xrange(n_iter):
        if (n_iter % num_batches == 0):
            inds = np.random.permutation(n_train)
            data_x = s_train_x.get_value()
            s_train_x.set_value(data_x[:,inds])
            data_y = s_train_y.get_value()
            s_train_y.set_value(data_y[:,inds])

        ce = train(i % num_batches)
        train_loss.append(ce)
        print "Iteration:", i
        print "cross entropy:", ce
        print

        if (i % 50==0):
            ce, acc = test()
            print
            print "TEST"
            print "cross entropy:", ce
            print 
            test_loss.append(ce)
            test_acc.append(acc)

            if ce < best_test_loss:
                best_params = [p.get_value() for p in parameters]
                best_rms = [r.get_value() for r in rmsprop]
                best_test_loss = ce

            save_vals = {'parameters': [p.get_value() for p in parameters],
                         'rmsprop': [r.get_value() for r in rmsprop],
                         'train_loss': train_loss,
                         'test_loss': test_loss,
                         'test_acc': test_acc,
                         'best_params': best_params,
                         'best_rms': best_rms,
                         'best_test_loss': best_test_loss,
                         'model': model,
                         'time_steps': time_steps}

            cPickle.dump(save_vals,
                         file(savefile, 'wb'),
                         cPickle.HIGHEST_PROTOCOL)

            
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="training a model")
    parser.add_argument("n_iter", type=int, default=20000)
    parser.add_argument("n_batch", type=int, default=20)
    parser.add_argument("n_hidden", type=int, default=512)
    parser.add_argument("time_steps", type=int, default=200)
    parser.add_argument("learning_rate", type=float, default=0.001)
    parser.add_argument("savefile")
    parser.add_argument("model", default='complex_RNN')
    parser.add_argument("input_type", default='categorical')
    parser.add_argument("out_every_t", default='False')
    parser.add_argument("loss_function", default='MSE')

    args = parser.parse_args()
    dict = vars(args)

    kwargs = {'n_iter': dict['n_iter'],
              'n_batch': dict['n_batch'],
              'n_hidden': dict['n_hidden'],
              'time_steps': dict['time_steps'],
              'learning_rate': np.float32(dict['learning_rate']),
              'savefile': dict['savefile'],
              'model': dict['model'],
              'input_type': dict['input_type'],
              'out_every_t': 'True'==dict['out_every_t'],
              'loss_function': dict['loss_function']}

    main(**kwargs)
