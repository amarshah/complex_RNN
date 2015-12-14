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

# Warning: assumes n_batch is a divisor of number of data points
# Suggestion: preprocess outputs to have norm 1 at each time step
def main(n_iter, n_batch, n_hidden, time_steps, learning_rate, savefile, scale_penalty, use_scale,
         model, n_hidden_lstm, loss_function, cost_every_t):


    # --- Manage data --------------------
    f = file('/u/shahamar/complex_RNN/trainingRNNs-master/memory_data_300.pkl', 'rb')
    dict = cPickle.load(f)
    f.close()


    train_x = dict['train_x'] 
    train_y = dict['train_y']
    test_x = dict['test_x'] 
    test_y = dict['test_y'] 


    #import pdb; pdb.set_trace()
    #cPickle.dump(dict, file('/u/shahamar/complex_RNN/trainingRNNs-master/memory_data.pkl', 'wb'))

    #import pdb; pdb.set_trace()

    n_train = train_x.shape[1]
    n_test = test_x.shape[1]
    n_input = train_x.shape[2]
    n_output = train_y.shape[2]

    num_batches = int(n_train / n_batch)
    time_steps = train_x.shape[0]
    
   #######################################################################

    gradient_clipping = np.float32(1)

    if (model == 'LSTM'):   
        inputs, parameters, costs = LSTM(n_input, n_hidden_lstm, n_output,
                                         out_every_t=cost_every_t, loss_function=loss_function)
        gradients = T.grad(costs[0], parameters)
        gradients = [T.clip(g, -gradient_clipping, gradient_clipping) for g in gradients]

    elif (model == 'complex_RNN'):
        inputs, parameters, costs = complex_RNN(n_input, n_hidden, n_output, scale_penalty,
                                                out_every_t=cost_every_t, loss_function=loss_function)
        if use_scale is False:
            parameters.pop()
        gradients = T.grad(costs[0], parameters)

    elif (model == 'complex_RNN_LSTM'):
        inputs, parameters, costs = complex_RNN_LSTM(n_input, n_hidden, n_hidden_lstm, n_output, scale_penalty,
                                                     out_every_t=cost_every_t, loss_function=loss_function)

    elif (model == 'IRNN'):
        inputs, parameters, costs = IRNN(n_input, n_hidden, n_output,
                                         out_every_t=cost_every_t, loss_function=loss_function)
        gradients = T.grad(costs[0], parameters)
        gradients = [T.clip(g, -gradient_clipping, gradient_clipping) for g in gradients]

    elif (model == 'RNN'):
        inputs, parameters, costs = tanhRNN(n_input, n_hidden, n_output,
                                            out_every_t=cost_every_t, loss_function=loss_function)
        gradients = T.grad(costs[0], parameters)
        gradients = [T.clip(g, -gradient_clipping, gradient_clipping) for g in gradients]

    else:
        print "Unsuported model:", model
        return
 

   




    s_train_x = theano.shared(train_x)
    s_train_y = theano.shared(train_y)

    s_test_x = theano.shared(test_x)
    s_test_y = theano.shared(test_y)


    # --- Compile theano functions --------------------------------------------------

    index = T.iscalar('i')

    updates, rmsprop = rms_prop(learning_rate, parameters, gradients)

    givens = {inputs[0] : s_train_x[:, n_batch * index : n_batch * (index + 1), :],
              inputs[1] : s_train_y[:, n_batch * index : n_batch * (index + 1), :]}

    givens_test = {inputs[0] : s_test_x,
                   inputs[1] : s_test_y}
    
   
    
    train = theano.function([index], costs[0], givens=givens, updates=updates)
    test = theano.function([], costs[1], givens=givens_test)

    # --- Training Loop ---------------------------------------------------------------

    # f1 = file('/data/lisatmp3/shahamar/memory/RNN_100.pkl', 'rb')
    # data1 = cPickle.load(f1)
    # f1.close()
    # train_loss = data1['train_loss']
    # test_loss = data1['test_loss']
    # best_params = data1['best_params']
    # best_test_loss = data1['best_test_loss']

    # for i in xrange(len(parameters)):
    #     parameters[i].set_value(data1['parameters'][i])

    # for i in xrange(len(parameters)):
    #     rmsprop[i].set_value(data1['rmsprop'][i])


    train_loss = []
    test_loss = []
    best_params = [p.get_value() for p in parameters]
    best_test_loss = 1e6
    for i in xrange(n_iter):
#        start_time = timeit.default_timer()
     #   pdb.set_trace()

        if (n_iter % int(num_batches) == 0):
            #import pdb; pdb.set_trace()
            inds = np.random.permutation(int(n_train))
            data_x = s_train_x.get_value()
            s_train_x.set_value(data_x[:,inds,:])
            data_y = s_train_y.get_value()
            s_train_y.set_value(data_y[:,inds,:])


        mse = train(i % int(num_batches))
        train_loss.append(mse)
        print "Iteration:", i
        print "mse:", mse
        print

        if (i % 50==0):
            mse = test()
            print
            print "TEST"
            print "mse:", mse
            print 
            test_loss.append(mse)

            if mse < best_test_loss:
                best_params = [p.get_value() for p in parameters]
                best_test_loss = mse

            save_vals = {'parameters': [p.get_value() for p in parameters],
                         'rmsprop': [r.get_value() for r in rmsprop],
                         'train_loss': train_loss,
                         'test_loss': test_loss,
                         'best_params': best_params,
                         'best_test_loss': best_test_loss,
                         'model': model,
                         'time_steps': time_steps}

            cPickle.dump(save_vals,
                         file(savefile, 'wb'),
                         cPickle.HIGHEST_PROTOCOL)

        
#        print 'time'
#        print timeit.default_timer() - start_time 


    
if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="training a model")
    parser.add_argument("n_iter", type=int, default=20000)
    parser.add_argument("n_batch", type=int, default=20)
    parser.add_argument("n_hidden", type=int, default=512)
    parser.add_argument("time_steps", type=int, default=200)
    parser.add_argument("learning_rate", type=float, default=0.001)
    parser.add_argument("savefile")
    parser.add_argument("scale_penalty", type=float, default=5)
    parser.add_argument("use_scale", default=True)
    parser.add_argument("model", default='complex_RNN')
    parser.add_argument("n_hidden_lstm", type=int, default=100)
    parser.add_argument("loss_function", default='MSE')
    parser.add_argument("cost_every_t", default=False)


    args = parser.parse_args()
    dict = vars(args)

    #import pdb; pdb.set_trace()
    
    

    kwargs = {'n_iter': dict['n_iter'],
              'n_batch': dict['n_batch'],
              'n_hidden': dict['n_hidden'],
              'time_steps': dict['time_steps'],
              'learning_rate': np.float32(dict['learning_rate']),
              'savefile': dict['savefile'],
              'scale_penalty': dict['scale_penalty'],
              'use_scale': dict['use_scale'],
              'model': dict['model'],
              'n_hidden_lstm': dict['n_hidden_lstm'],
              'loss_function': dict['loss_function'],
              'cost_every_t': dict['cost_every_t']}

    main(**kwargs)
