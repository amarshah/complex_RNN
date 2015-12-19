"""
TO DO: x and y different type
"""



import theano, cPickle
import theano.tensor as T
import numpy as np
from fftconv import cufft, cuifft

def initialize_matrix(n_in, n_out, name, rng):
    bin = np.sqrt(6. / (n_in + n_out))
    values = np.asarray(rng.uniform(low=-bin,
                                    high=bin,
                                    size=(n_in, n_out)),
                                    dtype=theano.config.floatX)
    return theano.shared(value=values, name=name)

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


def times_diag(input, n_hidden, diag, swap_re_im):
    d = T.concatenate([diag, -diag])
    
    Re = T.cos(d).dimshuffle('x',0)
    Im = T.sin(d).dimshuffle('x',0)

    input_times_Re = input * Re
    input_times_Im = input * Im

    output = input_times_Re + input_times_Im[:, swap_re_im]
   
    return output
    
    
def vec_permutation(input, index_permute):
    return input[:, index_permute]      

    
def times_reflection(input, n_hidden, reflection):
    input_re = input[:, :n_hidden]
    input_im = input[:, n_hidden:]
    reflect_re = reflection[:n_hidden]
    reflect_im = reflection[n_hidden:]
   
    vstarv = (reflection**2).sum()
    
    input_re_reflect_re = T.dot(input_re, reflect_re)
    input_re_reflect_im = T.dot(input_re, reflect_im)
    input_im_reflect_re = T.dot(input_im, reflect_re)
    input_im_reflect_im = T.dot(input_im, reflect_im)

    a = T.outer(input_re_reflect_re - input_im_reflect_im, reflect_re)
    b = T.outer(input_re_reflect_im + input_im_reflect_re, reflect_im)
    c = T.outer(input_re_reflect_re - input_im_reflect_im, reflect_im)
    d = T.outer(input_re_reflect_im + input_im_reflect_re, reflect_re)
         
    output = input
    output = T.inc_subtensor(output[:, :n_hidden], - 2. / vstarv * (a + b))
    output = T.inc_subtensor(output[:, n_hidden:], - 2. / vstarv * (d - c))

    return output    


def compute_cost_t(lin_output, loss_function, y_t):
    if loss_function == 'CE':
        RNN_output = T.nnet.softmax(lin_output)
        cost_t = T.nnet.categorical_crossentropy(RNN_output, y_t).mean()
        acc_t =(T.eq(T.argmax(RNN_output, axis=-1), y_t)).mean(dtype=theano.config.floatX)
    elif loss_function == 'MSE':
        cost_t = ((lin_output - y_t)**2).mean()
        acc_t = theano.shared(np.float32(0.0))

    return cost_t, acc_t

def initialize_data_nodes(loss_function, input_type, out_every_t):
    x = T.tensor3() if input_type == 'real' else T.matrix(dtype='int32')
    if loss_function == 'CE':
        y = T.matrix(dtype='int32') if out_every_t else T.vector(dtype='int32')
    else:  
        y = T.tensor3() if out_every_t else T.matrix()
    return x, y
        


def IRNN(n_input, n_hidden, n_output, input_type='real', out_every_t=False, loss_function='CE'):
    np.random.seed(1234)
    rng = np.random.RandomState(1234)

    x, y = initialize_data_nodes(loss_function, input_type, out_every_t)
    inputs = [x, y]

    h_0 = theano.shared(np.zeros((1, n_hidden), dtype=theano.config.floatX))
    V = initialize_matrix(n_input, n_hidden, 'V', rng)
    W = theano.shared(np.identity(n_hidden, dtype=theano.config.floatX))
    out_mat = initialize_matrix(n_hidden, n_output, 'out_mat', rng)
    hidden_bias = theano.shared(np.zeros((n_hidden,), dtype=theano.config.floatX))
    out_bias = theano.shared(np.zeros((n_output,), dtype=theano.config.floatX))

    parameters = [h_0, V, W, out_mat, hidden_bias, out_bias]

    def recurrence(x_t, y_t, h_prev, cost_prev, acc_prev, V, W, hidden_bias, out_mat, out_bias):
        if loss_function == 'CE':
            data_lin_output = V[x_t]
        else:
            data_lin_output = T.dot(x_t, V)
        
        h_t = T.nnet.relu(T.dot(h_prev, W) + data_lin_output + hidden_bias.dimshuffle('x', 0))
        if out_every_t:
            lin_output = T.dot(h_t, out_mat) + out_bias.dimshuffle('x', 0)
            cost_t, acc_t = compute_cost_t(lin_output, loss_function, y_t)
        else:
            cost_t = theano.shared(np.float32(0.0))
            acc_t = theano.shared(np.float32(0.0))
 
        return h_t, cost_t, acc_t
    
    non_sequences = [V, W, hidden_bias, out_mat, out_bias]

    h_0_batch = T.tile(h_0, [x.shape[1], 1])

    if out_every_t:
        sequences = [x, y]
    else:
        sequences = [x, T.tile(theano.shared(np.zeros((1,1), dtype=theano.config.floatX)), [x.shape[0], 1, 1])]

    outputs_info = [h_0_batch, theano.shared(np.float32(0.0)), theano.shared(np.float32(0.0))]
    
    [hidden_states, cost_steps, acc_steps], updates = theano.scan(fn=recurrence,
                                                                  sequences=sequences,
                                                                  non_sequences=non_sequences,
                                                                  outputs_info = outputs_info)
   
    if not out_every_t:
        lin_output = T.dot(hidden_states[-1,:,:], out_mat) + out_bias.dimshuffle('x', 0)
        costs = compute_cost_t(lin_output, loss_function, y)
    else:
        cost = cost_steps.mean()
        accuracy = acc_steps.mean()
        costs = [cost, accuracy]

    return inputs, parameters, costs



def tanhRNN(n_input, n_hidden, n_output, input_type='real', out_every_t=False, loss_function='CE'):
    np.random.seed(1234)
    rng = np.random.RandomState(1234)

    x, y = initialize_data_nodes(loss_function, input_type, out_every_t)
    inputs = [x, y]

    h_0 = theano.shared(np.zeros((1, n_hidden), dtype=theano.config.floatX))
    V = initialize_matrix(n_input, n_hidden, 'V', rng)
    W = initialize_matrix(n_hidden, n_hidden, 'W', rng)
    out_mat = initialize_matrix(n_hidden, n_output, 'out_mat', rng)
    hidden_bias = theano.shared(np.zeros((n_hidden,), dtype=theano.config.floatX))
    out_bias = theano.shared(np.zeros((n_output,), dtype=theano.config.floatX))
    parameters = [h_0, V, W, out_mat, hidden_bias, out_bias]

    def recurrence(x_t, y_t, h_prev, cost_prev, acc_prev, V, W, hidden_bias, out_mat, out_bias):
        if loss_function == 'CE':
            data_lin_output = V[x_t]
        else:
            data_lin_output = T.dot(x_t, V)
        
        h_t = T.tanh(T.dot(h_prev, W) + data_lin_output + hidden_bias.dimshuffle('x', 0))
        if out_every_t:
            lin_output = T.dot(h_t, out_mat) + out_bias.dimshuffle('x', 0)
            cost_t, acc_t = compute_cost_t(lin_output, loss_function, y_t)
        else:
            cost_t = theano.shared(np.float32(0.0))
            acc_t = theano.shared(np.float32(0.0))
 
        return h_t, cost_t, acc_t 
    
    non_sequences = [V, W, hidden_bias, out_mat, out_bias]

    h_0_batch = T.tile(h_0, [x.shape[1], 1])

    if out_every_t:
        sequences = [x, y]
    else:
        sequences = [x, T.tile(theano.shared(np.zeros((1,1), dtype=theano.config.floatX)), [x.shape[0], 1, 1])]

    outputs_info = [h_0_batch, theano.shared(np.float32(0.0)), theano.shared(np.float32(0.0))]
        
    [hidden_states, cost_steps, acc_steps], updates = theano.scan(fn=recurrence,
                                                                  sequences=sequences,
                                                                  non_sequences=non_sequences,
                                                                  outputs_info=outputs_info)
   
    if not out_every_t:
        lin_output = T.dot(hidden_states[-1,:,:], out_mat) + out_bias.dimshuffle('x', 0)
        costs = compute_cost_t(lin_output, loss_function, y)
    else:
        cost = cost_steps.mean()
        accuracy = acc_steps.mean()
        costs = [cost, accuracy]

    return inputs, parameters, costs



def LSTM(n_input, n_hidden, n_output, input_type='real', out_every_t=False, loss_function='CE'):
    np.random.seed(1234)
    rng = np.random.RandomState(1234)

    W_i = initialize_matrix(n_input, n_hidden, 'W_i', rng)
    W_f = initialize_matrix(n_input, n_hidden, 'W_f', rng)
    W_c = initialize_matrix(n_input, n_hidden, 'W_c', rng)
    W_o = initialize_matrix(n_input, n_hidden, 'W_o', rng)
    U_i = initialize_matrix(n_hidden, n_hidden, 'U_i', rng)
    U_f = initialize_matrix(n_hidden, n_hidden, 'U_f', rng)
    U_c = initialize_matrix(n_hidden, n_hidden, 'U_c', rng)
    U_o = initialize_matrix(n_hidden, n_hidden, 'U_o', rng)
    V_o = initialize_matrix(n_hidden, n_hidden, 'V_o', rng)
    b_i = theano.shared(np.zeros((n_hidden,), dtype=theano.config.floatX))
    b_f = theano.shared(np.ones((n_hidden,), dtype=theano.config.floatX))
    b_c = theano.shared(np.zeros((n_hidden,), dtype=theano.config.floatX))
    b_o = theano.shared(np.zeros((n_hidden,), dtype=theano.config.floatX))
    h_0 = theano.shared(np.zeros((1, n_hidden), dtype=theano.config.floatX))
    state_0 = theano.shared(np.zeros((1, n_hidden), dtype=theano.config.floatX))
    out_mat = initialize_matrix(n_hidden, n_output, 'out_mat', rng)
    out_bias = theano.shared(np.zeros((n_output,), dtype=theano.config.floatX))
    parameters = [W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o, V_o, b_i, b_f, b_c, b_o, h_0, state_0, out_mat, out_bias]

    x, y = initialize_data_nodes(loss_function, input_type, out_every_t)
    
    def recurrence(x_t, y_t, h_prev, state_prev, cost_prev, acc_prev,
                   W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o, V_o, b_i, b_f, b_c, b_o, out_mat, out_bias):
        
        if loss_function == 'CE':
            x_t_W_i = W_i[x_t]
            x_t_W_c = W_c[x_t]
            x_t_W_f = W_f[x_t]
            x_t_W_o = W_o[x_t]
        else:
            x_t_W_i = T.dot(x_t, W_i)
            x_t_W_c = T.dot(x_t, W_c)
            x_t_W_f = T.dot(x_t, W_f)
            x_t_W_o = T.dot(x_t, W_o)
            
        input_t = T.nnet.sigmoid(x_t_W_i + T.dot(h_prev, U_i) + b_i.dimshuffle('x', 0))
        candidate_t = T.tanh(x_t_W_c + T.dot(h_prev, U_c) + b_c.dimshuffle('x', 0))
        forget_t = T.nnet.sigmoid(x_t_W_f + T.dot(h_prev, U_f) + b_f.dimshuffle('x', 0))

        state_t = input_t * candidate_t + forget_t * state_prev

        output_t = T.nnet.sigmoid(x_t_W_o + T.dot(h_prev, U_o) + T.dot(state_t, V_o) + b_o.dimshuffle('x', 0))

        h_t = output_t * T.tanh(state_t)

        if out_every_t:
            lin_output = T.dot(h_t, out_mat) + out_bias.dimshuffle('x', 0)
            cost_t, acc_t = compute_cost_t(lin_output, loss_function, y_t)
        else:
            cost_t = theano.shared(np.float32(0.0))
            acc_t = theano.shared(np.float32(0.0))
 
        return h_t, state_t, cost_t, acc_t

    non_sequences = [W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o, V_o, b_i, b_f, b_c, b_o, out_mat, out_bias]

    h_0_batch = T.tile(h_0, [x.shape[1], 1])
    state_0_batch = T.tile(state_0, [x.shape[1], 1])
    
    if out_every_t:
        sequences = [x, y]
    else:
        sequences = [x, T.tile(theano.shared(np.zeros((1,1), dtype=theano.config.floatX)), [x.shape[0], 1, 1])]

    outputs_info = [h_0_batch, state_0_batch, theano.shared(np.float32(0.0)), theano.shared(np.float32(0.0))]
        
    [hidden_states, states, cost_steps, acc_steps], updates = theano.scan(fn=recurrence,
                                                                          sequences=sequences,
                                                                          non_sequences=non_sequences,
                                                                          outputs_info=outputs_info)
    
    if not out_every_t:
        lin_output = T.dot(hidden_states[-1,:,:], out_mat) + out_bias.dimshuffle('x', 0)
        costs = compute_cost_t(lin_output, loss_function, y)            
    else:
        cost = cost_steps.mean()
        accuracy = acc_steps.mean()
        costs = [cost, accuracy]

    return [x, y], parameters, costs


def complex_RNN(n_input, n_hidden, n_output, input_type='real', out_every_t=False, loss_function='CE'):

    np.random.seed(1234)
    rng = np.random.RandomState(1234)

    # Initialize parameters: theta, V_re, V_im, hidden_bias, U, out_bias, h_0
    V = initialize_matrix(n_input, 2*n_hidden, 'V', rng)
    U = initialize_matrix(2 * n_hidden, n_output, 'U', rng)
    hidden_bias = theano.shared(np.asarray(rng.uniform(low=-0.01,
                                                       high=0.01,
                                                       size=(n_hidden,)),
                                           dtype=theano.config.floatX), 
                                name='hidden_bias')
    
    reflection = initialize_matrix(2, 2*n_hidden, 'reflection', rng)
    out_bias = theano.shared(np.zeros((n_output,), dtype=theano.config.floatX), name='out_bias')
    theta = theano.shared(np.asarray(rng.uniform(low=-np.pi,
                                                 high=np.pi,
                                                 size=(3, n_hidden)),
                                     dtype=theano.config.floatX), 
                                name='theta')

    bucket = np.sqrt(3. / 2 / n_hidden) 
    h_0 = theano.shared(np.asarray(rng.uniform(low=-bucket,
                                               high=bucket,
                                               size=(1, 2 * n_hidden)), 
                                   dtype=theano.config.floatX),
                        name='h_0')
    
    parameters = [V, U, hidden_bias, reflection, out_bias, theta, h_0]

    x, y = initialize_data_nodes(loss_function, input_type, out_every_t)
    
    index_permute = np.random.permutation(n_hidden)

    index_permute_long = np.concatenate((index_permute, index_permute + n_hidden))
    swap_re_im = np.concatenate((np.arange(n_hidden, 2*n_hidden), np.arange(n_hidden)))
    
    # define the recurrence used by theano.scan
    def recurrence(x_t, y_t, h_prev, cost_prev, acc_prev, theta, V, hidden_bias, out_bias, U):  

        # Compute hidden linear transform
        step1 = times_diag(h_prev, n_hidden, theta[0,:], swap_re_im)
        step2 = do_fft(step1, n_hidden)
        step3 = times_reflection(step2, n_hidden, reflection[0,:])
        step4 = vec_permutation(step3, index_permute_long)
        step5 = times_diag(step4, n_hidden, theta[1,:], swap_re_im)
        step6 = do_ifft(step5, n_hidden)
        step7 = times_reflection(step6, n_hidden, reflection[1,:])
        step8 = times_diag(step7, n_hidden, theta[2,:], swap_re_im)     
        
        hidden_lin_output = step8
        
        # Compute data linear transform
        if loss_function == 'CE':
            data_lin_output = V[T.cast(x_t, 'int32')]
        else:
            data_lin_output = T.dot(x_t, V)
            
        # Total linear output        
        lin_output = hidden_lin_output + data_lin_output


        # Apply non-linearity ----------------------------

        # scale RELU nonlinearity
        modulus = T.sqrt(lin_output**2 + lin_output[:, swap_re_im]**2)
        rescale = T.maximum(modulus + T.tile(hidden_bias, [2]).dimshuffle('x', 0), 0.) / (modulus + 1e-5)
        h_t = lin_output * rescale
        
        if out_every_t:
            lin_output = T.dot(h_t, U) + out_bias.dimshuffle('x', 0)
            cost_t, acc_t = compute_cost_t(lin_output, loss_function, y_t)
        else:
            cost_t = theano.shared(np.float32(0.0))
            acc_t = theano.shared(np.float32(0.0))
        
        return h_t, cost_t, acc_t

    # compute hidden states
    h_0_batch = T.tile(h_0, [x.shape[1], 1])
    non_sequences = [theta, V, hidden_bias, out_bias, U]
    if out_every_t:
        sequences = [x, y]
    else:
        sequences = [x, T.tile(theano.shared(np.zeros((1,1), dtype=theano.config.floatX)), [x.shape[0], 1, 1])]

    outputs_info=[h_0_batch, theano.shared(np.float32(0.0)), theano.shared(np.float32(0.0))]
    
    [hidden_states, cost_steps, acc_steps], updates = theano.scan(fn=recurrence,
                                                                  sequences=sequences,
                                                                  non_sequences=non_sequences,
                                                                  outputs_info=outputs_info)

    if not out_every_t:
        lin_output = T.dot(hidden_states[-1,:,:], U) + out_bias.dimshuffle('x', 0)
        costs = compute_cost_t(lin_output, loss_function, y)
    else:
        cost = cost_steps.mean()
        accuracy = acc_steps.mean()
        costs = [cost, accuracy]

    return [x, y], parameters, costs




 

