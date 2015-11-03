import cPickle
import gzip
import theano
import pdb
from fftconv import cufft, cuifft
import numpy as np
import theano.tensor as T
from theano.ifelse import ifelse



def initialize_matrix(n_in, n_out, name, rng):
    bin = np.sqrt(6. / (n_in + n_out))
    values = np.asarray(rng.uniform(low=-bin,
                                    high=bin,
                                    size=(n_in, n_out)),
                        dtype=theano.config.floatX)
    return theano.shared(value=values, name=name, borrow=True)


# computes Theano graph
# returns symbolic parameters, costs, inputs 
# there are n_hidden real units and a further n_hidden imaginary units 
def complex_RNN(n_input, n_hidden, n_output, scale_penalty):
   

    np.random.seed(1234)
    rng = np.random.RandomState(1234)

    # Initialize parameters: theta, V_re, V_im, hidden_bias, U, out_bias, h_0
    V_re = initialize_matrix(n_input, n_hidden, 'V_re', rng)
    V_im = initialize_matrix(n_input, n_hidden, 'V_im', rng)


    project_mat = initialize_matrix(2*n_hidden, n_hidden, 'project_mat', rng)
    relu1_mat = initialize_matrix(n_hidden, n_hidden, 'relu1_mat', rng)
    relu1_bias = theano.shared(np.zeros((n_hidden,), dtype=theano.config.floatX),
                               name='relu1_bias', borrow=True)
    relu2_mat = initialize_matrix(n_hidden, n_hidden, 'relu2_mat', rng)
    relu2_bias = theano.shared(np.zeros((n_hidden,), dtype=theano.config.floatX),
                               name='relu2_bias', borrow=True)

    U = initialize_matrix(n_hidden, n_output, 'U', rng)
    hidden_bias = theano.shared(np.asarray(rng.uniform(low=-0.01,
                                                       high=0.01,
                                                       size=(n_hidden,)),
                                           dtype=theano.config.floatX), 
                                name='hidden_bias', borrow=True)
    
    reflection = initialize_matrix(2, 2*n_hidden, 'reflection', rng)
    out_bias = theano.shared(np.zeros((n_output,), dtype=theano.config.floatX),
                             name='out_bias', borrow=True)
    theta = initialize_matrix(3, n_hidden, 'theta', rng)
    bucket = np.sqrt(2.) * np.sqrt(3. / 2 / n_hidden)
    h_0 = theano.shared(np.asarray(rng.uniform(low=-bucket,
                                               high=bucket,
                                               size=(1, 2 * n_hidden)), 
                                   dtype=theano.config.floatX),
                        name='h_0', borrow=True)
    
    scale = theano.shared(np.ones((n_hidden,), dtype=theano.config.floatX),
                          name='scale', borrow=True)


    parameters = [V_re, V_im, hidden_bias, theta, 
                  relu1_mat, relu1_bias, relu2_mat, relu2_bias,
                  U, out_bias, h_0, reflection, scale] 

    x = T.tensor3()
    y = T.tensor3()
    ########
#    theano.config.compute_test_value = 'warn'
#    x.tag.test_value = np.random.rand(200,7,50).astype('float32') 
#    y.tag.test_value = np.random.rand(200,7,50).astype('float32')
    ########

    index_permute = np.random.permutation(n_hidden)
 
    # define the recurrence used by theano.scan
    def recurrence(x_t, y_t, h_prev, theta, V_re, V_im, hidden_bias, scale, U, out_bias):  
        def do_fft(input, n_hidden):
            fft_input = T.reshape(input, (input.shape[0], 2, n_hidden))
            fft_input = fft_input.dimshuffle(0,2,1)
            fft_output = cufft(fft_input) / T.sqrt(n_hidden)
            fft_output = fft_output.dimshuffle(0,2,1)
            output = T.reshape(fft_output, (input.shape[0], 2*n_hidden))
            return output

        def do_ifft(input, n_hidden):
            ifft_input = T.reshape(input, (input.shape[0], 2, n_hidden))
            ifft_input = ifft_input.dimshuffle(0,2,1)
            ifft_output = cuifft(ifft_input) / T.sqrt(n_hidden)
            ifft_output = ifft_output.dimshuffle(0,2,1)
            output = T.reshape(ifft_output, (input.shape[0], 2*n_hidden))
            return output


        def scale_diag(input, n_hidden, diag):
            input_re = input[:, :n_hidden]
            input_im = input[:, n_hidden:]
            Diag = T.nlinalg.AllocDiag()(diag)
            input_re_times_Diag = T.dot(input_re, Diag)
            input_im_times_Diag = T.dot(input_im, Diag)

            return T.concatenate([input_re_times_Diag, input_im_times_Diag], axis=1)

        def times_diag(input, n_hidden, diag):
            input_re = input[:, :n_hidden]
            input_im = input[:, n_hidden:]
            Re = T.nlinalg.AllocDiag()(T.cos(diag))
            Im = T.nlinalg.AllocDiag()(T.sin(diag))
            input_re_times_Re = T.dot(input_re, Re)
            input_re_times_Im = T.dot(input_re, Im)
            input_im_times_Re = T.dot(input_im, Re)
            input_im_times_Im = T.dot(input_im, Im)

            return T.concatenate([input_re_times_Re - input_im_times_Im,
                                  input_re_times_Im + input_im_times_Re], axis=1)

        def vec_permutation(input, n_hidden, index_permute):
            re = input[:, :n_hidden]
            im = input[:, n_hidden:]
            re_permute = re[:, index_permute]
            im_permute = im[:, index_permute]

            return T.concatenate([re_permute, im_permute], axis=1)      
        
        def times_reflection(input, n_hidden, reflection):
            input_re = input[:, :n_hidden]
            input_im = input[:, n_hidden:]
            reflect_re = reflection[n_hidden:]
            reflect_im = reflection[:n_hidden]

            vstarv = (reflect_re**2 + reflect_im**2).sum()
            input_re_reflect = input_re - 2 / vstarv * (T.outer(T.dot(input_re, reflect_re), reflect_re) 
                                                        + T.outer(T.dot(input_re, reflect_im), reflect_im) 
                                                        - T.outer(T.dot(input_im, reflect_im), reflect_re) 
                                                        + T.outer(T.dot(input_im, reflect_re), reflect_im))
            input_im_reflect = input_im - 2 / vstarv * (T.outer(T.dot(input_im, reflect_re), reflect_re) 
                                                        + T.outer(T.dot(input_im, reflect_im), reflect_im) 
                                                        + T.outer(T.dot(input_re, reflect_im), reflect_re) 
                                                        - T.outer(T.dot(input_re, reflect_re), reflect_im))

            return T.concatenate([input_re_reflect, input_im_reflect], axis=1)      


        # Compute hidden linear transform
        step1 = times_diag(h_prev, n_hidden, theta[0,:])
#        step2 = do_fft(step1, n_hidden)
        step2 = step1
        step3 = times_reflection(step2, n_hidden, reflection[0,:])
        step4 = vec_permutation(step3, n_hidden, index_permute)
        step5 = times_diag(step4, n_hidden, theta[1,:])
#        step6 = do_ifft(step5, n_hidden)
        step6 = step5
        step7 = times_reflection(step6, n_hidden, reflection[1,:])
        step8 = times_diag(step7, n_hidden, theta[2,:])     
        step9 = scale_diag(step8, n_hidden, scale)
        
        hidden_lin_output = step9
        
        # Compute data linear transform
        data_lin_output_re = T.dot(x_t, V_re)
        data_lin_output_im = T.dot(x_t, V_im)
        data_lin_output = T.concatenate([data_lin_output_re, data_lin_output_im], axis=1)
        
        # Total linear output        
        lin_output = hidden_lin_output + data_lin_output
        lin_output_re = lin_output[:, :n_hidden]
        lin_output_im = lin_output[:, n_hidden:] 


        # Apply non-linearity ----------------------------

        # scale RELU nonlinearity
        modulus = T.sqrt(lin_output_re ** 2 + lin_output_im ** 2)
        rescale = T.maximum(modulus + hidden_bias.dimshuffle('x',0), 0.) / (modulus + 1e-5)
        nonlin_output_re = lin_output_re * rescale
        nonlin_output_im = lin_output_im * rescale    

        h_t = T.concatenate([nonlin_output_re, 
                             nonlin_output_im], axis=1) 


        warp1 = h_t[:, ::2] + h_t[:, 1::2] 
#        import pdb; pdb.set_trace()
        warp2 = T.maximum(T.dot(warp1, relu1_mat) + relu1_bias.dimshuffle('x', 0), 0.)
        warp3 = T.maximum(T.dot(warp2, relu2_mat) + relu2_bias.dimshuffle('x', 0), 0.)          
        unnormalized_predict_t = T.dot(warp3, U) + out_bias.dimshuffle('x', 0)

        predict_t = T.nnet.softmax(unnormalized_predict_t)
        cost_t = T.nnet.categorical_crossentropy(predict_t, y_t)
        
        return h_t, cost_t

    # compute hidden states
    h_0_batch = T.tile(h_0, [x.shape[1], 1])
    non_sequences = [theta, V_re, V_im, hidden_bias, scale, U, out_bias]
    [hidden_states, costs], updates = theano.scan(fn=recurrence,
                                                  sequences=[x, y],
                                                  non_sequences=non_sequences,
                                                  outputs_info=[h_0_batch, None])
   
    cost = costs.mean() + scale_penalty * ((scale-1)**2).sum()
    cost.name = 'cross_entropy_sum_penalized'

    log_prob = -y.shape[0]*cost
    log_prob.name = 'log_prob'

    costs = [cost, log_prob]

    return [x, y], parameters, costs

 
def clipped_gradients(grad_clip, gradients):
    clipped_grads = [T.clip(g, -gradient_clipping, gradient_clipping)
                     for g in gradients]
    return clipped_grads

def gradient_descent(learning_rate, parameters, gradients):        
    updates = [(p, p - learning_rate * g) for p, g in zip(parameters, gradients)]
    return updates

def gradient_descent_momentum(learning_rate, momentum, parameters, gradients):
    velocities = [theano.shared(np.zeros_like(p.get_value(), 
                                              dtype=theano.config.floatX)) for p in parameters]

    updates1 = [(vel, momentum * vel - learning_rate * g) 
                for vel, g in zip(velocities, gradients)]
    updates2 = [(p, p + vel) for p, vel in zip(parameters, velocities)]
    updates = updates1 + updates2
    return updates 


def rms_prop(learning_rate, parameters, gradients):        
    rmsprop = [theano.shared(1e-3*np.ones_like(p.get_value())) for p in parameters]
    new_rmsprop = [0.9 * vel + 0.1 * (g**2) for vel, g in zip(rmsprop, gradients)]

    updates1 = zip(rmsprop, new_rmsprop)
    updates2 = [(p, p - learning_rate * g / T.sqrt(rms)) for 
                p, g, rms in zip(parameters, gradients, new_rmsprop)]
    updates = updates1 + updates2
    return updates, rmsprop
    


def penntreebank(dataset=None):
    data = np.load('/data/lisa/data/PennTreebankCorpus/pentree_char_and_word.npz')
    if dataset == 'train':
        return data['train_chars']
    elif dataset == 'valid':
        return data['valid_chars']
    elif dataset == 'test':
        return data['test_chars']
    else:
        return data

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
    data = penntreebank()
    trainset = data['train_chars']
    validset = data['valid_chars']
    # end of sentence: \n
    allletters = " etanoisrhludcmfpkgybw<>\nvN.'xj$-qz&0193#285\\764/*"
    dictionary = dict(zip(list(set(allletters)), range(alphabetsize)))
    invdict = {v: k for k, v in dictionary.items()}
    # add all possible 2-grams to dataset
    #all2grams = ''.join([a+b for b in allletters for a in allletters])
    #trainset = np.hstack((np.array([dictionary[c] for c in all2grams]), trainset))

    numtrain, numvalid = len(trainset) / seq_length * seq_length,\
        len(validset) / seq_length * seq_length
    train_features_numpy = onehot(trainset[:numtrain]).reshape(seq_length, numtrain/seq_length, alphabetsize)
    valid_features_numpy = onehot(validset[:numvalid]).reshape(seq_length, numvalid/seq_length, alphabetsize)
    del trainset, validset

    ntrain = numtrain/seq_length
    inds = np.random.permutation(ntrain)
    train_features = train_features_numpy[:,inds,:]

    nvalid = numvalid/seq_length
    inds = np.random.permutation(nvalid)
    valid_features = valid_features_numpy[:,inds,:]
    
    return [train_features, valid_features]






# Warning: assumes n_batch is a divisor of number of data points
# Suggestion: preprocess outputs to have norm 1 at each time step
def main(n_iter, n_batch, n_hidden, time_steps, learning_rate, savefile, scale_penalty, use_scale):

 
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
 
    inputs, parameters, costs = complex_RNN(n_input, n_hidden, n_output, scale_penalty)
    if not use_scale:
        parameters.pop() 
   
#    import pdb; pdb.set_trace()
    gradients = T.grad(costs[0], parameters)

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
    
       
    train = theano.function([index], costs[0], givens=givens, updates=updates)
    valid = theano.function([], costs[0], givens=givens_valid)

    # --- Training Loop ---------------------------------------------------------------
    train_loss = []
    valid_loss = []
    best_params = [p.get_value() for p in parameters]
    best_valid_loss = 1e6
    for i in xrange(n_iter):
     #   pdb.set_trace()

        cent = train(i % num_batches)
        train_loss.append(cent)
        print "Iteration:", i
        print "cross entropy:", cent
        print

        if (i % 25==0):
            cent = valid()
            print
            print "VALIDATION"
            print "cross entropy:", cent
            print 
            valid_loss.append(cent)

            if cent < best_valid_loss:
                best_params = [p.get_value() for p in parameters]
                best_valid_loss = cent

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
              'n_batch': 100,
              'n_hidden': 1024,
              'time_steps': 100,
              'learning_rate': np.float32(0.001),
              'savefile': '/data/lisatmp3/shahamar/2015-11-02-penntree-nofft-1scalepen-deepout.pkl',
              'scale_penalty': 1,
              'use_scale': True}

    main(**kwargs)
