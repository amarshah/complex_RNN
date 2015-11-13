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
    reflect_re = reflection[:n_hidden]
    reflect_im = reflection[n_hidden:]

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

def IRNN(n_input, n_hidden, n_output, out_every_t=False, loss_function='CE'):
    np.random.seed(1234)
    rng = np.random.RandomState(1234)

    x = T.tensor3()
    if out_every_t:
        y = T.tensor3()
    else:
        y = T.matrix()
    inputs = [x, y]

    h_0 = theano.shared(np.zeros((1, n_hidden), dtype=theano.config.floatX))
    V = initialize_matrix(n_input, n_hidden, 'V', rng)
    W = theano.shared(np.identity(n_hidden, dtype=theano.config.floatX))
    out_mat = initialize_matrix(n_hidden, n_output, 'out_mat', rng)
    hidden_bias = theano.shared(np.zeros((n_hidden,), dtype=theano.config.floatX))
    out_bias = theano.shared(np.zeros((n_output,), dtype=theano.config.floatX))

    parameters = [h_0, V, W, out_mat, hidden_bias, out_bias]

    def recurrence(x_t, y_t, h_prev, cost_prev, acc_prev, V, W, hidden_bias, out_mat, out_bias):
        h_t = T.nnet.relu(T.dot(h_prev, W) + T.dot(x_t, V) + hidden_bias.dimshuffle('x', 0))
        if out_every_t:
            lin_output = T.dot(h_t, out_mat) + out_bias.dimshuffle('x', 0)
            if loss_function == 'CE':
                RNN_output = T.nnet.softmax(lin_output)
                cost_t = T.nnet.categorical_crossentropy(RNN_output, y_t).mean()
                acc_t =(T.eq(T.argmax(RNN_output, axis=-1), T.argmax(y_t, axis=-1))).mean(dtype=theano.config.floatX)
            elif loss_function == 'MSE':
                cost_t = ((lin_output - y_t)**2).mean()
                acc_t = theano.shared(np.float32(0.0))
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
    
    [hidden_states, cost_steps, acc_steps], updates = theano.scan(fn=recurrence,
                                                                  sequences=sequences,
                                                                  non_sequences=non_sequences,
                                                                  outputs_info = [h_0_batch, theano.shared(np.float32(0.0)), theano.shared(np.float32(0.0))])
   
    if not out_every_t:
        lin_output = T.dot(hidden_states[-1,:,:], out_mat) + out_bias.dimshuffle('x', 0)

        # define the cost
        if loss_function == 'CE':
            RNN_output = T.nnet.softmax(lin_output)
            cost = T.nnet.categorical_crossentropy(RNN_output, y).mean()
            cost_penalty = cost

            # compute accuracy
            accuracy = T.mean(T.eq(T.argmax(RNN_output, axis=-1), T.argmax(y, axis=-1)))

            costs = [cost_penalty, cost, accuracy]
        elif loss_function == 'MSE':
            cost = ((lin_output - y)**2).mean()
            cost_penalty = cost

            costs = [cost_penalty, cost]


    else:
        cost = cost_steps.mean()
        accuracy = acc_steps.mean()
        cost_penalty = cost
        costs = [cost_penalty, cost, accuracy]



    return inputs, parameters, costs


def tanhRNN(n_input, n_hidden, n_output, out_every_t=False, loss_function='CE'):
    np.random.seed(1234)
    rng = np.random.RandomState(1234)

    x = T.tensor3()
    if out_every_t:
        y = T.tensor3()
    else:
        y = T.matrix()
    inputs = [x, y]

    h_0 = theano.shared(np.zeros((1, n_hidden), dtype=theano.config.floatX))
    V = initialize_matrix(n_input, n_hidden, 'V', rng)
    W = initialize_matrix(n_hidden, n_hidden, 'W', rng)
    out_mat = initialize_matrix(n_hidden, n_output, 'out_mat', rng)
    hidden_bias = theano.shared(np.zeros((n_hidden,), dtype=theano.config.floatX))
    out_bias = theano.shared(np.zeros((n_output,), dtype=theano.config.floatX))
    parameters = [h_0, V, W, out_mat, hidden_bias, out_bias]

#    hidden_bias_batch = T.tile(hidden_bias, [x.shape[1], 1])

    def recurrence(x_t, y_t, h_prev, cost_prev, acc_prev, V, W, hidden_bias, out_mat, out_bias):
        h_t = T.tanh(T.dot(h_prev, W) + T.dot(x_t, V) + hidden_bias.dimshuffle('x', 0))
        if out_every_t:
            lin_output = T.dot(h_t, out_mat) + out_bias.dimshuffle('x', 0)
            if loss_function == 'CE':
                RNN_output = T.nnet.softmax(lin_output)
                cost_t = T.nnet.categorical_crossentropy(RNN_output, y_t).mean()
                acc_t =(T.eq(T.argmax(RNN_output, axis=-1), T.argmax(y_t, axis=-1))).mean(dtype=theano.config.floatX)
            elif loss_function == 'MSE':
                cost_t = ((lin_output - y_t)**2).mean()
                acc_t = theano.shared(np.float32(0.0))
        else:
            cost_t = theano.shared(np.float32(0.0))
            acc_t = theano.shared(np.float32(0.0))
 
        #import pdb; pdb.set_trace()
        return h_t, cost_t, acc_t 
    
    non_sequences = [V, W, hidden_bias, out_mat, out_bias]

    h_0_batch = T.tile(h_0, [x.shape[1], 1])

    if out_every_t:
        sequences = [x, y]
    else:
        sequences = [x, T.tile(theano.shared(np.zeros((1,1), dtype=theano.config.floatX)), [x.shape[0], 1, 1])]
    
    [hidden_states, cost_steps, acc_steps], updates = theano.scan(fn=recurrence,
                                                                  sequences=sequences,
                                                                  non_sequences=non_sequences,
                                                                  outputs_info = [h_0_batch, theano.shared(np.float32(0.0)), theano.shared(np.float32(0.0))])
   
    if not out_every_t:
        lin_output = T.dot(hidden_states[-1,:,:], out_mat) + out_bias.dimshuffle('x', 0)

        # define the cost
        if loss_function == 'CE':
            RNN_output = T.nnet.softmax(lin_output)
            cost = T.nnet.categorical_crossentropy(RNN_output, y).mean()
            cost_penalty = cost

            # compute accuracy
            accuracy = T.mean(T.eq(T.argmax(RNN_output, axis=-1), T.argmax(y, axis=-1)))

            costs = [cost_penalty, cost, accuracy]
        elif loss_function == 'MSE':
            cost = ((lin_output - y)**2).mean()
            cost_penalty = cost

            costs = [cost_penalty, cost]


    else:
        cost = cost_steps.mean()
        accuracy = acc_steps.mean()
        cost_penalty = cost 
        costs = [cost_penalty, cost, accuracy]



    return inputs, parameters, costs


def LSTM(n_input, n_hidden, n_output, out_every_t=False, loss_function='CE'):
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

    x = T.tensor3()
    if out_every_t:
        y = T.tensor3()
    else:
        y = T.matrix()
    
    def recurrence(x_t, y_t, h_prev, state_prev, cost_prev, acc_prev, W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o, V_o, b_i, b_f, b_c, b_o, out_mat, out_bias):
        input_t = T.nnet.sigmoid(T.dot(x_t, W_i) + T.dot(h_prev, U_i) + b_i.dimshuffle('x', 0))
        candidate_t = T.tanh(T.dot(x_t, W_c) + T.dot(h_prev, U_c) + b_c.dimshuffle('x', 0))
        forget_t = T.nnet.sigmoid(T.dot(x_t, W_f) + T.dot(h_prev, U_f) + b_f.dimshuffle('x', 0))

        state_t = input_t * candidate_t + forget_t * state_prev

        output_t = T.nnet.sigmoid(T.dot(x_t, W_o) + T.dot(h_prev, U_o) + T.dot(state_t, V_o) + b_o.dimshuffle('x', 0))

        h_t = output_t * T.tanh(state_t)

        if out_every_t:
            lin_output = T.dot(h_t, out_mat) + out_bias.dimshuffle('x', 0)
            if loss_function == 'CE':
                RNN_output = T.nnet.softmax(lin_output)
                cost_t = T.nnet.categorical_crossentropy(RNN_output, y_t).mean()
                acc_t =(T.eq(T.argmax(RNN_output, axis=-1), T.argmax(y_t, axis=-1))).mean(dtype=theano.config.floatX)
            elif loss_function == 'MSE':
                cost_t = ((lin_output - y_t)**2).mean()
                acc_t = theano.shared(np.float32(0.0))
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
    
    [hidden_states, states, cost_steps, acc_steps], updates = theano.scan(fn = recurrence, sequences = sequences, non_sequences = non_sequences, outputs_info = [h_0_batch, state_0_batch, theano.shared(np.float32(0.0)), theano.shared(np.float32(0.0))])
    
    if not out_every_t:
        lin_output = T.dot(hidden_states[-1,:,:], out_mat) + out_bias.dimshuffle('x', 0)

        # define the cost
        if loss_function == 'CE':
            RNN_output = T.nnet.softmax(lin_output)
            cost = T.nnet.categorical_crossentropy(RNN_output, y).mean()
            cost_penalty = cost

            # compute accuracy
            accuracy = T.mean(T.eq(T.argmax(RNN_output, axis=-1), T.argmax(y, axis=-1)))

            costs = [cost_penalty, cost, accuracy]
        elif loss_function == 'MSE':
            cost = ((lin_output - y)**2).mean()
            cost_penalty = cost 

            costs = [cost_penalty, cost]


    else:
        cost = cost_steps.mean()
        accuracy = acc_steps.mean()
        cost_penalty = cost
        costs = [cost_penalty, cost, accuracy]

    return [x, y], parameters, costs

# there are n_hidden real units and a further n_hidden imaginary units 
def complex_RNN_LSTM(n_input, n_hidden, n_hidden_lstm, n_output, scale_penalty, out_every_t=False, loss_function='CE'):
    
    np.random.seed(1234)
    rng = np.random.RandomState(1234)

    # Initialize parameters: theta, V_re, V_im, hidden_bias, U, out_bias, h_0
    V_re = initialize_matrix(n_input, n_hidden, 'V_re', rng)
    V_im = initialize_matrix(n_input, n_hidden, 'V_im', rng)
    hidden_bias = theano.shared(np.asarray(rng.uniform(low=-0.01,
                                                       high=0.01,
                                                       size=(n_hidden,)),
                                           dtype=theano.config.floatX), 
                                name='hidden_bias')
    
    reflection = initialize_matrix(2, 2*n_hidden, 'reflection', rng)
    theta = initialize_matrix(3, n_hidden, 'theta', rng)
    bucket = np.sqrt(2.) * np.sqrt(3. / 2 / n_hidden)
    h_0 = theano.shared(np.asarray(rng.uniform(low=-bucket,
                                               high=bucket,
                                               size=(1, 2 * n_hidden)), 
                                   dtype=theano.config.floatX),
                        name='h_0')
    
    scale = theano.shared(np.ones((n_hidden,), dtype=theano.config.floatX),
                          name='scale')

    parameters = [V_re, V_im, hidden_bias, theta, h_0, reflection, scale] 

    W_i = initialize_matrix(n_hidden_lstm, n_hidden_lstm, 'W_i', rng)
    W_f = initialize_matrix(n_hidden_lstm, n_hidden_lstm,  'W_f', rng)
    W_c = initialize_matrix(n_hidden_lstm, n_hidden_lstm, 'W_c', rng)
    W_o = initialize_matrix(n_hidden_lstm, n_hidden_lstm, 'W_o', rng)
    U_i = initialize_matrix(n_hidden_lstm, n_hidden_lstm, 'U_i', rng)
    U_f = initialize_matrix(n_hidden_lstm, n_hidden_lstm, 'U_f', rng)
    U_c = initialize_matrix(n_hidden_lstm, n_hidden_lstm, 'U_c', rng)
    U_o = initialize_matrix(n_hidden_lstm, n_hidden_lstm, 'U_o', rng)
    V_o = initialize_matrix(n_hidden_lstm, n_hidden_lstm, 'V_o', rng)
    b_i = theano.shared(np.zeros((n_hidden_lstm,), dtype=theano.config.floatX))
    b_f = theano.shared(np.ones((n_hidden_lstm,), dtype=theano.config.floatX))
    b_c = theano.shared(np.zeros((n_hidden_lstm,), dtype=theano.config.floatX))
    b_o = theano.shared(np.zeros((n_hidden_lstm,), dtype=theano.config.floatX))
    h_0_lstm = theano.shared(np.zeros((1, n_hidden_lstm), dtype=theano.config.floatX))
    lstm_state_0 = theano.shared(np.zeros((1, n_hidden_lstm), dtype=theano.config.floatX))
    out_mat = initialize_matrix(n_hidden_lstm, n_output, 'out_mat', rng)
    out_bias = theano.shared(np.zeros((n_output,), dtype=theano.config.floatX))
    parameters = parameters + [W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o, V_o, b_i, b_f, b_c, b_o, h_0_lstm, lstm_state_0, out_mat, out_bias]

    bin = np.sqrt(6. / (2 * n_hidden + n_hidden_lstm))
    random_projection_hidden_to_lstm = values = np.asarray(rng.uniform(low=-bin, high=bin, size=(2 * n_hidden, n_hidden_lstm)), dtype=theano.config.floatX)


    x = T.tensor3()
    if out_every_t:
        y = T.tensor3()
    else:
        y = T.matrix()
    index_permute = np.random.permutation(n_hidden)
 
    # define the recurrence used by theano.scan
    def recurrence(x_t, y_t, h_prev, lstm_h_prev, lstm_state_prev, cost_prev, acc_prev, theta, V_re, V_im, hidden_bias, scale, W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o, V_o, b_i, b_f, b_c, b_o, out_bias, out_mat):  
        
        
        # Compute hidden linear transform
        step1 = times_diag(h_prev, n_hidden, theta[0,:])
        step2 = do_fft(step1, n_hidden)
        step3 = times_reflection(step2, n_hidden, reflection[0,:])
        step4 = vec_permutation(step3, n_hidden, index_permute)
        step5 = times_diag(step4, n_hidden, theta[1,:])
        step6 = do_ifft(step5, n_hidden)
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

        # nonlinear mod and phase operations
#        lin_output_mod = T.sqrt(lin_output_re ** 2 + lin_output_im ** 2)
#        lin_output_phase = T.arctan(lin_output_im / (lin_output_re + 1e-5))
        
#        nonlin_output_mod = T.maximum(lin_output_mod + hidden_bias.dimshuffle('x',0), 0.) \
#            / (lin_output_mod + 1e-5)
#        m1 = T.exp(log_phase_bias[:n_hidden])
#        m2 = T.exp(log_phase_bias[n_hidden:])
#        left = (lin_output_phase + 0.5 * np.pi) * m1.dimshuffle('x',0) - 0.5 * np.pi
#        right = (lin_output_phase - 0.5 * np.pi) * m2.dimshuffle('x',0) + 0.5 * np.pi
#        condition = - 0.5 * np.pi * (m1 + m2 - 2) / (m1 - m2)
#        nonlin_output_phase = T.switch(T.lt(lin_output_phase, condition), left, right)

#        nonlin_output_re = nonlin_output_mod * T.cos(nonlin_output_phase)
#        nonlin_output_im = nonlin_output_mod * T.sin(nonlin_output_phase)


        # scale RELU nonlinearity
        modulus = T.sqrt(lin_output_re ** 2 + lin_output_im ** 2)
        rescale = T.maximum(modulus + hidden_bias.dimshuffle('x',0), 0.) / (modulus + 1e-7)
        nonlin_output_re = lin_output_re * rescale
        nonlin_output_im = lin_output_im * rescale

        # relu on each part
#        nonlin_output_re = T.maximum(lin_output_re + hidden_bias.dimshuffle('x', 0), 0.)
#        nonlin_output_im = T.maximum(lin_output_im + hidden_bias.dimshuffle('x', 0), 0.)        

        h_t = T.concatenate([nonlin_output_re, 
                             nonlin_output_im], axis=1) 
        
        lstm_input_t = T.dot(h_t, random_projection_hidden_to_lstm)

        input_t = T.nnet.sigmoid(T.dot(lstm_input_t , W_i) + T.dot(lstm_h_prev, U_i) + b_i.dimshuffle('x', 0))
        candidate_t = T.tanh(T.dot(lstm_input_t , W_c) + T.dot(lstm_h_prev, U_c) + b_c.dimshuffle('x', 0))
        forget_t = T.nnet.sigmoid(T.dot(lstm_input_t , W_f) + T.dot(lstm_h_prev, U_f) + b_f.dimshuffle('x', 0))

        lstm_state_t = input_t * candidate_t + forget_t * lstm_state_prev

        output_t = T.nnet.sigmoid(T.dot(lstm_input_t , W_o) + T.dot(lstm_h_prev, U_o) + T.dot(lstm_state_t, V_o) + b_o.dimshuffle('x', 0))

        lstm_h_t = output_t * T.tanh(lstm_state_t)


        if out_every_t:
            lin_output = T.dot(lstm_h_t, out_mat) + out_bias.dimshuffle('x', 0)
            if loss_function == 'CE':
                RNN_output = T.nnet.softmax(lin_output)
                cost_t = T.nnet.categorical_crossentropy(RNN_output, y_t).mean()
                acc_t =(T.eq(T.argmax(RNN_output, axis=-1), T.argmax(y_t, axis=-1))).mean(dtype=theano.config.floatX)
            elif loss_function == 'MSE':
                cost_t = ((lin_output - y_t)**2).mean()
                acc_t = theano.shared(np.float32(0.0))
        else:
            cost_t = theano.shared(np.float32(0.0))
            acc_t = theano.shared(np.float32(0.0))
       
        return h_t, lstm_h_t, lstm_state_t, cost_t, acc_t

    # compute hidden states
    h_0_batch = T.tile(h_0, [x.shape[1], 1])
    h_0_lstm_batch = T.tile(h_0_lstm, [x.shape[1], 1])
    lstm_state_0_batch = T.tile(lstm_state_0, [x.shape[1], 1])
    
    if out_every_t:
        sequences = [x, y]
    else:
        sequences = [x, T.tile(theano.shared(np.zeros((1,1), dtype=theano.config.floatX)), [x.shape[0], 1, 1])]
    
    non_sequences = [theta, V_re, V_im, hidden_bias, scale, W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o, V_o, b_i, b_f, b_c, b_o, out_bias, out_mat]
    outputs_info=[h_0_batch, h_0_lstm_batch, lstm_state_0_batch, theano.shared(np.float32(0.0)), theano.shared(np.float32(0.0))]
    [hidden_states, hidden_lstm_states, lstm_states, cost_steps, acc_steps], updates = theano.scan(fn=recurrence,
                                                                                        sequences=sequences,
                                                                                        non_sequences=non_sequences,
                                                                                        outputs_info=outputs_info)
    if not out_every_t:
        lin_output = T.dot(hidden_lstm_states[-1,:,:], out_mat) + out_bias.dimshuffle('x', 0)

        # define the cost
        if loss_function == 'CE':
            RNN_output = T.nnet.softmax(lin_output)
            cost = T.nnet.categorical_crossentropy(RNN_output, y).mean()
            cost_penalty = cost + scale_penalty * ((scale - 1) ** 2).sum()

            # compute accuracy
            accuracy = T.mean(T.eq(T.argmax(RNN_output, axis=-1), T.argmax(y, axis=-1)))

            costs = [cost_penalty, cost, accuracy]
        elif loss_function == 'MSE':
            cost = ((lin_output - y)**2).mean()
            cost_penalty = cost + scale_penalty * ((scale - 1) ** 2).sum()

            costs = [cost_penalty, cost]

    else:
        cost = cost_steps.mean()
        accuracy = acc_steps.mean()
        cost_penalty = cost + scale_penalty * ((scale - 1) ** 2).sum()
        costs = [cost_penalty, cost, accuracy]


    return [x, y], parameters, costs


def complex_RNN(n_input, n_hidden, n_output, scale_penalty, out_every_t=False, loss_function='CE'):
    
    np.random.seed(1234)
    rng = np.random.RandomState(1234)

    # Initialize parameters: theta, V_re, V_im, hidden_bias, U, out_bias, h_0
    V_re = initialize_matrix(n_input, n_hidden, 'V_re', rng)
    V_im = initialize_matrix(n_input, n_hidden, 'V_im', rng)
    U = initialize_matrix(2 * n_hidden, n_output, 'U', rng)
    hidden_bias = theano.shared(np.asarray(rng.uniform(low=-0.01,
                                                       high=0.01,
                                                       size=(n_hidden,)),
                                           dtype=theano.config.floatX), 
                                name='hidden_bias')
    
    reflection = initialize_matrix(2, 2*n_hidden, 'reflection', rng)
    out_bias = theano.shared(np.zeros((n_output,), dtype=theano.config.floatX), name='out_bias')
    theta = initialize_matrix(3, n_hidden, 'theta', rng)
    bucket = np.sqrt(2.) * np.sqrt(3. / 2 / n_hidden)
    h_0 = theano.shared(np.asarray(rng.uniform(low=-bucket,
                                               high=bucket,
                                               size=(1, 2 * n_hidden)), 
                                   dtype=theano.config.floatX),
                        name='h_0')
    
    scale = theano.shared(np.ones((n_hidden,), dtype=theano.config.floatX),
                          name='scale')
    parameters = [V_re, V_im, U, hidden_bias, reflection, out_bias, theta, h_0, scale]


    x = T.tensor3()
    if out_every_t:
        y = T.tensor3()
    else:
        y = T.matrix()
    index_permute = np.random.permutation(n_hidden)
    
    # define the recurrence used by theano.scan
    def recurrence(x_t, y_t, h_prev, cost_prev, acc_prev, theta, V_re, V_im, hidden_bias, scale, out_bias, U):  

        # Compute hidden linear transform
        step1 = times_diag(h_prev, n_hidden, theta[0,:])
        step2 = step1
#        step2 = do_fft(step1, n_hidden)
        step3 = times_reflection(step2, n_hidden, reflection[0,:])
        step4 = vec_permutation(step3, n_hidden, index_permute)
        step5 = times_diag(step4, n_hidden, theta[1,:])
        step6 = step5
#        step6 = do_ifft(step5, n_hidden)
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
        if out_every_t:
            lin_output = T.dot(h_t, U) + out_bias.dimshuffle('x', 0)
            if loss_function == 'CE':
                RNN_output = T.nnet.softmax(lin_output)
                cost_t = T.nnet.categorical_crossentropy(RNN_output, y_t).mean()
                acc_t =(T.eq(T.argmax(RNN_output, axis=-1), T.argmax(y_t, axis=-1))).mean(dtype=theano.config.floatX)
            elif loss_function == 'MSE':
                cost_t = ((lin_output - y_t)**2).mean()
                acc_t = theano.shared(np.float32(0.0))
        else:
            cost_t = theano.shared(np.float32(0.0))
            acc_t = theano.shared(np.float32(0.0))
        
        return h_t, cost_t, acc_t

    # compute hidden states
    h_0_batch = T.tile(h_0, [x.shape[1], 1])
    non_sequences = [theta, V_re, V_im, hidden_bias, scale, out_bias, U]
    if out_every_t:
        sequences = [x, y]
    else:
        sequences = [x, T.tile(theano.shared(np.zeros((1,1), dtype=theano.config.floatX)), [x.shape[0], 1, 1])]
    [hidden_states, cost_steps, acc_steps], updates = theano.scan(fn=recurrence,
                                                                  sequences=sequences,
                                                                  non_sequences=non_sequences,
                                                                  outputs_info=[h_0_batch, theano.shared(np.float32(0.0)), theano.shared(np.float32(0.0))])

    if not out_every_t:
        lin_output = T.dot(hidden_states[-1,:,:], U) + out_bias.dimshuffle('x', 0)

        # define the cost
        if loss_function == 'CE':
            RNN_output = T.nnet.softmax(lin_output)
            cost = T.nnet.categorical_crossentropy(RNN_output, y).mean()
            cost_penalty = cost + scale_penalty * ((scale - 1) ** 2).sum()

            # compute accuracy
            accuracy = T.mean(T.eq(T.argmax(RNN_output, axis=-1), T.argmax(y, axis=-1)))

            costs = [cost_penalty, cost, accuracy]
        elif loss_function == 'MSE':
            cost = ((lin_output - y)**2).mean()
            cost_penalty = cost + scale_penalty * ((scale - 1) ** 2).sum()

            costs = [cost_penalty, cost]


    else:
        cost = cost_steps.mean()
        cost_penalty = cost + scale_penalty * ((scale - 1) ** 2).sum()
        accuracy = acc_steps.mean()
        costs = [cost_penalty, cost, accuracy]


    return [x, y], parameters, costs





 

def complex_RNN_derivs(n_input, n_hidden, n_output, scale_penalty):
   
    np.random.seed(1234)
    rng = np.random.RandomState(1234)

    n_theta = 3
    n_reflect = 2

    # Initialize parameters: theta, V_re, V_im, hidden_bias, U, out_bias, h_0
    V_re = initialize_matrix(n_input, n_hidden, 'V_re', rng)
    V_im = initialize_matrix(n_input, n_hidden, 'V_im', rng)


    U = initialize_matrix(2*n_hidden, n_output, 'U', rng)
    out_bias = theano.shared(np.zeros((n_output,), dtype=theano.config.floatX),
                             name='out_bias', borrow=True)

    scale = theano.shared(np.ones((n_hidden,), dtype=theano.config.floatX),
                          name='scale', borrow=True)
    reflection = initialize_matrix(n_reflect, 2*n_hidden, 'reflection', rng)

    theta = initialize_matrix(n_theta, n_hidden, 'theta', rng)
    hidden_bias = theano.shared(np.asarray(rng.uniform(low=-0.01,
                                                       high=0.01,
                                                       size=(n_hidden,)),
                                           dtype=theano.config.floatX), 
                                name='hidden_bias', borrow=True)    
    bucket = np.sqrt(2.) * np.sqrt(3. / 2 / n_hidden)
    h_0 = theano.shared(np.asarray(rng.uniform(low=-bucket,
                                               high=bucket,
                                               size=(1, 2*n_hidden)), 
                                   dtype=theano.config.floatX),
                        name='h_0', borrow=True)


    parameters = [h_0, V_re, V_im, hidden_bias, theta, 
                  reflection, scale, U, out_bias] 

    x = T.tensor3()
    y = T.tensor3()

    index_permute = np.random.permutation(n_hidden)
    reverse_index_permute = np.zeros_like(index_permute)
    reverse_index_permute[index_permute] = range(n_hidden)

    ## DEFINE FUNCTIONS FOR NONLINEARITY------------------------------------------
    def complex_nonlinearity(mod, bias, nu=100):
        inp = mod + bias.dimshuffle('x',0)
        out1 = inp + 1./nu
        out2 = 1. / (nu - inp)
        return T.switch(T.ge(inp, 0), out1, out2)

    def complex_nonlinearity_inverse(mod, bias, nu=100):
        out1 = mod - 1./nu
        out2 = nu - 1./mod
        return T.switch(T.ge(mod, 1./nu), out1, out2) - bias.dimshuffle('x', 0)         

    def apply_nonlinearity(lin, bias, nu=100):        
        n_h = bias.shape[0]
        lin_re = lin[:, :n_h]
        lin_im = lin[:, n_h:]        
        mod = T.sqrt(lin_re**2 + lin_im**2)
        rescale = complex_nonlinearity(mod, bias, nu) / mod         
        return T.tile(rescale, [1, 2]) * lin
        
    def apply_nonlinearity_inverse(h, bias, nu=100):
        n_h = bias.shape[0]
        modh = T.sqrt(h[:,:n_h]**2 + h[:,n_h:]**2)
        rescale = complex_nonlinearity_inverse(modh, bias, nu) / modh
        return T.tile(rescale, [1, 2]) * h

    def compute_nonlinearity_deriv(lin, bias, nu=100):
        n_h = bias.shape[0]
        lin_re = lin[:, :n_h]
        lin_im = lin[:, n_h:]        
        modlin = T.sqrt(lin_re**2 + lin_im**2)
        
        rescale = complex_nonlinearity(modlin, bias, nu) / modlin
        
        inp = modlin + bias.dimshuffle('x', 0)
        opt1 = 1.
        opt2 = 1. / ((nu - inp)**2)
        deriv = T.switch(T.ge(inp, 0), opt1, opt2) 
                           
        return deriv, rescale, modlin         


    def compute_nonlinearity_bias_derivative(mod, bias, nu=100):
        n_h = bias.shape[0]
        inp = mod + bias.dimshuffle('x', 0)
        
        opt1 = 1.
        opt2 = 1. / (nu - inp)**2
        dmoddb = T.switch(T.ge(inp, 0), opt1, opt2)
        return dmoddb

    ###DEFINE FUNCTIONS FOR HIDDEN TO COST-------------------------------------------
    def hidden_output(h, U, out_bias, y):
        unnormalized_predict = T.dot(h, U) + out_bias.dimshuffle('x', 0)
        predict = T.nnet.softmax(unnormalized_predict)
        cost = T.nnet.categorical_crossentropy(predict, y)        
        return cost, predict

    def hidden_output_derivs(h, U, out_bias, y):
        cost, predict = hidden_output(h, U, out_bias, y)

        n_batch = h.shape[0]
        dcostdunnormalized_predict = (predict - y)  

        dcostdU = T.batched_dot(h.dimshuffle(0,1,'x'),
                                dcostdunnormalized_predict.dimshuffle(0,'x',1))

        dcostdout_bias = dcostdunnormalized_predict
       
        return dcostdU, dcostdout_bias

    def compute_dctdht(h, U, out_bias, y):
        cost, predict = hidden_output(h, U, out_bias, y)

        n_batch = h.shape[0]
        dcostdunnormalized_predict = (predict - y)  

        dcostdh = T.dot(dcostdunnormalized_predict, U.T)

        return dcostdh
        

    # define the recurrence used by theano.scan
    def recurrence(x_t, y_t, h_prev,
                   theta, reflection, V_re, V_im, hidden_bias, scale, U, out_bias):  


        # ----------------------------------------------------------------------
        # COMPUTES FORWARD PASS

        # Compute hidden linear transform
        step1 = times_diag(h_prev, n_hidden, theta[0,:])
        step2 = do_fft(step1, n_hidden)
#        step2 = step1
        step3 = times_reflection(step2, n_hidden, reflection[0,:])
        step4 = vec_permutation(step3, n_hidden, index_permute)
        step5 = times_diag(step4, n_hidden, theta[1,:])
        step6 = do_ifft(step5, n_hidden)
#        step6 = step5
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

        # Apply non-linearity 
        h_t = apply_nonlinearity(lin_output, hidden_bias)
        unnormalized_predict_t = T.dot(h_t, U) + out_bias.dimshuffle('x', 0)

        predict_t = T.nnet.softmax(unnormalized_predict_t)
        cost_t = T.nnet.categorical_crossentropy(predict_t, y_t)
        
        return h_t, cost_t

    # compute hidden states
    n_batch = x.shape[1]
    h_0_batch = T.tile(h_0, [n_batch, 1])
    non_sequences = [theta, reflection, V_re, V_im, hidden_bias, scale, U, out_bias]
    [hidden_states, costs], updates = theano.scan(fn=recurrence,
                                                  sequences=[x, y],
                                                  non_sequences=non_sequences,
                                                  outputs_info=[h_0_batch, None])

    costs_per_data = costs.sum(axis=0)

    cost = costs_per_data.mean()
    cost.name = 'cross_entropy'

    log_prob = -cost #check this
    log_prob.name = 'log_prob'

    costs = [cost, log_prob]

    # -----------------------------------------------------
    # START GRADIENT COMPUTATION

    dV_re = T.alloc(0., n_batch, n_input, n_hidden)
    dV_im = T.alloc(0., n_batch, n_input, n_hidden)
    dtheta = T.alloc(0., n_batch, n_theta, n_hidden)
    dreflection = T.alloc(0., n_batch, n_reflect, 2*n_hidden)
    dhidden_bias = T.alloc(0., n_batch, n_hidden)
    dU = T.alloc(0., n_batch, 2*n_hidden, n_output)
    dout_bias = T.alloc(0., n_batch, n_output)
    dscale = T.alloc(0., n_batch, n_hidden)


    def gradient_recurrence(x_t_plus_1, y_t_plus_1, y_t, isend_t, dh_t_plus_1, h_t_plus_1,
                            dV_re_t_plus_1, dV_im_t_plus_1, dhidden_bias_t_plus_1, dtheta_t_plus_1, 
                            dreflection_t_plus_1, dscale_t_plus_1, dU_t_plus_1, dout_bias_t_plus_1,
                            V_re, V_im, hidden_bias, theta, reflection, scale, U, out_bias):  
        
        
        dV_re_t = dV_re_t_plus_1
        dV_im_t = dV_im_t_plus_1
        dhidden_bias_t = dhidden_bias_t_plus_1 
        dtheta_t = dtheta_t_plus_1 
        dreflection_t = dreflection_t_plus_1
        dscale_t = dscale_t_plus_1
        dU_t = dU_t_plus_1
        dout_bias_t = dout_bias_t_plus_1

        # Compute h_t --------------------------------------------------------------------------        
        data_linoutput_re = T.dot(x_t_plus_1, V_re)
        data_linoutput_im = T.dot(x_t_plus_1, V_im) 
        data_linoutput = T.concatenate([data_linoutput_re, data_linoutput_im], axis=1)

        total_linoutput = apply_nonlinearity_inverse(h_t_plus_1, hidden_bias)
        hidden_linoutput = total_linoutput - data_linoutput
    
        step8 = scale_diag(hidden_linoutput, n_hidden, 1. / scale)
        step7 = times_diag(step8, n_hidden, -theta[2,:])
        step6 = times_reflection(step7, n_hidden, reflection[1,:])
#        step5 = step6
        step5 = do_fft(step6, n_hidden)
        step4 = times_diag(step5, n_hidden, -theta[1,:])
        step3 = vec_permutation(step4, n_hidden, reverse_index_permute)
        step2 = times_reflection(step3, n_hidden, reflection[0,:])
#        step1 = step2
        step1 = do_ifft(step2, n_hidden)
        step0 = times_diag(step1, n_hidden, -theta[0,:])
        
        h_t = step0
                
        # Compute deriv contributions to hidden to output params------------------------------------------------
        dU_contribution, dout_bias_contribution = \
            hidden_output_derivs(h_t_plus_1, U, out_bias, y_t_plus_1)
        
        dU_t = dU_t + dU_contribution
        dout_bias_t = dout_bias_t + dout_bias_contribution

        # Compute derivative of linoutputs -------------------------------------------------------------------        
        deriv, rescale, modTL = compute_nonlinearity_deriv(total_linoutput, hidden_bias)
 
        dh_t_plus_1_TL = dh_t_plus_1 * total_linoutput
        matrix = dh_t_plus_1_TL[:, :n_hidden] + dh_t_plus_1_TL[:, n_hidden:]
        matrix = matrix * (deriv - rescale) / (modTL**2)

        dtotal_linoutput_re = dh_t_plus_1[:, :n_hidden] * rescale \
            + total_linoutput[:, :n_hidden] * matrix 

        dtotal_linoutput_im = dh_t_plus_1[:, n_hidden:] * rescale \
            + total_linoutput[:, n_hidden:] * matrix 

        dtotal_linoutput = T.concatenate([dtotal_linoutput_re, dtotal_linoutput_im], axis=1)


        dhidden_linoutput = dtotal_linoutput
        ddata_linoutput = dtotal_linoutput

        # Compute deriv contributions to hidden bias-------------------------------------------------------        
        dhidden_bias_contribution_re = dh_t_plus_1_TL[:, :n_hidden] * deriv / modTL
        dhidden_bias_contribution_im = dh_t_plus_1_TL[:, n_hidden:] * deriv / modTL

        dhidden_bias_t = dhidden_bias_t + dhidden_bias_contribution_re \
            + dhidden_bias_contribution_im

        # Compute derivative of h_t -------------------------------------------------------------------

        # use transpose conjugate operations 
        dstep8 = scale_diag(dhidden_linoutput, n_hidden, scale)
        dstep7 = times_diag(dstep8, n_hidden, -theta[2,:])
        dstep6 = times_reflection(dstep7, n_hidden, reflection[1,:])
#        dstep5 = dstep6
        dstep5 = do_fft(dstep6, n_hidden)
        dstep4 = times_diag(dstep5, n_hidden, -theta[1,:])
        dstep3 = vec_permutation(dstep4, n_hidden, reverse_index_permute)
        dstep2 = times_reflection(dstep3, n_hidden, reflection[0,:])
#        dstep1 = dstep2
        dstep1 = do_ifft(dstep2, n_hidden)
        dstep0 = times_diag(dstep1, n_hidden, -theta[0,:])

        dh_t = dstep0         
        dh_t_contribution = compute_dctdht(h_t, U, out_bias, y_t)
        dh_t = theano.ifelse.ifelse(T.eq(isend_t, 0), dh_t + dh_t_contribution, dh_t)
      

        # Compute deriv contributions to Unitary parameters ----------------------------------------------------

        # scale------------------------------------------------
        dscale_contribution = dhidden_linoutput * step8 
        dscale_t = dscale_t + dscale_contribution[:, :n_hidden] \
            + dscale_contribution[:, n_hidden:]

        # theta2-----------------------------------------------
        dtheta2_contribution = dstep8 * times_diag(step7, n_hidden, theta[2,:] + 0.5 * np.pi)
        dtheta_t = T.inc_subtensor(dtheta_t[:, 2, :], dtheta2_contribution[:, :n_hidden] +
                                   dtheta2_contribution[:, n_hidden:])

        # reflection1-----------------------------------------
        v_re = reflection[1, :n_hidden]
        v_im = reflection[1, n_hidden:]
        vstarv = (v_re ** 2 + v_im ** 2).sum()

        dstep7_re = dstep7[:, :n_hidden]
        dstep7_im = dstep7[:, n_hidden:]
        step6_re = step6[:, :n_hidden]
        step6_im = step6[:, n_hidden:]

        v_re_dot_v_re = T.dot(v_re, v_re.T)
        v_im_dot_v_im = T.dot(v_im, v_im.T)
        v_im_dot_v_re = T.dot(v_im, v_re.T)

        dstep7_re_dot_v_re = T.dot(dstep7_re, v_re.T).dimshuffle(0, 'x') #n_b x 1
        dstep7_re_dot_v_im = T.dot(dstep7_re, v_im.T).dimshuffle(0, 'x')
        step6_re_dot_v_re = T.dot(step6_re, v_re.T).dimshuffle(0, 'x')
        step6_re_dot_v_im = T.dot(step6_re, v_im.T).dimshuffle(0, 'x')
        dstep7_im_dot_v_re = T.dot(dstep7_im, v_re.T).dimshuffle(0, 'x')
        dstep7_im_dot_v_im = T.dot(dstep7_im, v_im.T).dimshuffle(0, 'x')
        step6_im_dot_v_re = T.dot(step6_im, v_re.T).dimshuffle(0, 'x')
        step6_im_dot_v_im = T.dot(step6_im, v_im.T).dimshuffle(0, 'x')

        dstep7_re_timesum_step6_re = (dstep7_re * step6_re).sum(axis=1)
        dstep7_re_timesum_step6_im = (dstep7_re * step6_im).sum(axis=1)
        dstep7_im_timesum_step6_re = (dstep7_im * step6_re).sum(axis=1)
        dstep7_im_timesum_step6_im = (dstep7_im * step6_im).sum(axis=1)

        #--------

        dstep7_re_RedOpdv_re_term1 = - 2. / vstarv * (dstep7_re * step6_re_dot_v_re
                                                      + dstep7_re_dot_v_re * step6_re
                                                      - dstep7_re * step6_im_dot_v_im
                                                      + dstep7_re_dot_v_im * step6_im)

        outer_sum = (T.outer(step6_re_dot_v_re, v_re) 
                     + T.outer(step6_re_dot_v_im, v_im)
                     - T.outer(step6_im_dot_v_im, v_re)
                     + T.outer(step6_im_dot_v_re, v_im))
        dstep7_re_RedOpdv_re_term2 = 4. / (vstarv**2) * T.outer((dstep7_re * outer_sum).sum(axis=1), v_re)

        dstep7_im_ImdOpdv_re_term1 = - 2. / vstarv * (dstep7_im * step6_im_dot_v_re
                                                      + dstep7_im_dot_v_re * step6_im
                                                      + dstep7_im * step6_re_dot_v_im
                                                      - dstep7_im_dot_v_im * step6_re)

        outer_sum = (T.outer(step6_im_dot_v_re, v_re) 
                     + T.outer(step6_im_dot_v_im, v_im)
                     + T.outer(step6_re_dot_v_im, v_re)
                     - T.outer(step6_re_dot_v_re, v_im))
        dstep7_im_ImdOpdv_re_term2 = 4. / (vstarv**2) * T.outer((dstep7_im * outer_sum).sum(axis=1), v_re)

        dv_re_contribution = (dstep7_re_RedOpdv_re_term1 + dstep7_re_RedOpdv_re_term2 
                              + dstep7_im_ImdOpdv_re_term1 + dstep7_im_ImdOpdv_re_term2)

        #---------

        dstep7_re_RedOpdv_im_term1 = - 2. / vstarv * (dstep7_re * step6_re_dot_v_im
                                                      + dstep7_re_dot_v_im * step6_re
                                                      - dstep7_re_dot_v_re * step6_im
                                                      + dstep7_re * step6_im_dot_v_re)

        outer_sum = (T.outer(step6_re_dot_v_re, v_re) 
                     + T.outer(step6_re_dot_v_im, v_im)
                     - T.outer(step6_im_dot_v_im, v_re)
                     + T.outer(step6_im_dot_v_re, v_im))
        dstep7_re_RedOpdv_im_term2 = 4. / (vstarv**2) * T.outer((dstep7_re * outer_sum).sum(axis=1), v_im)


        dstep7_im_ImdOpdv_im_term1 = - 2. / vstarv * (dstep7_im * step6_im_dot_v_im
                                                      + dstep7_im_dot_v_im * step6_im
                                                      + dstep7_im_dot_v_re * step6_re
                                                      - dstep7_im * step6_re_dot_v_re)

        outer_sum = (T.outer(step6_im_dot_v_re, v_re) 
                     + T.outer(step6_im_dot_v_im, v_im)
                     + T.outer(step6_re_dot_v_im, v_re)
                     - T.outer(step6_re_dot_v_re, v_im))
        dstep7_im_ImdOpdv_im_term2 = 4. / (vstarv**2) * T.outer((dstep7_im * outer_sum).sum(axis=1), v_im)

        dv_im_contribution = (dstep7_re_RedOpdv_im_term1 + dstep7_re_RedOpdv_im_term2 
                              + dstep7_im_ImdOpdv_im_term1 + dstep7_im_ImdOpdv_im_term2)

        dreflection_t = T.inc_subtensor(dreflection_t[:, 1, :n_hidden], dv_re_contribution)
        dreflection_t = T.inc_subtensor(dreflection_t[:, 1, n_hidden:], dv_im_contribution)


        # theta1-----------------------------------------------------
        dtheta1_contribution = dstep5 * times_diag(step4, n_hidden, theta[1,:] + 0.5 * np.pi)
        dtheta_t = T.inc_subtensor(dtheta_t[:, 1, :], dtheta1_contribution[:, :n_hidden] + dtheta1_contribution[:, n_hidden:])
       
        # reflection0------------------------------------------------
        v_re = reflection[0, :n_hidden]
        v_im = reflection[0, n_hidden:]
        vstarv = (v_re ** 2 + v_im ** 2).sum()
        
        dstep3_re = dstep3[:, :n_hidden]
        dstep3_im = dstep3[:, n_hidden:]
        step2_re = step2[:, :n_hidden]
        step2_im = step2[:, n_hidden:]

        v_re_dot_v_re = T.dot(v_re, v_re.T)
        v_im_dot_v_im = T.dot(v_im, v_im.T)
        v_im_dot_v_re = T.dot(v_im, v_re.T)

        dstep3_re_dot_v_re = T.dot(dstep3_re, v_re.T).dimshuffle(0, 'x') #n_b x 1
        dstep3_re_dot_v_im = T.dot(dstep3_re, v_im.T).dimshuffle(0, 'x')
        step2_re_dot_v_re = T.dot(step2_re, v_re.T).dimshuffle(0, 'x')
        step2_re_dot_v_im = T.dot(step2_re, v_im.T).dimshuffle(0, 'x')
        dstep3_im_dot_v_re = T.dot(dstep3_im, v_re.T).dimshuffle(0, 'x')
        dstep3_im_dot_v_im = T.dot(dstep3_im, v_im.T).dimshuffle(0, 'x')
        step2_im_dot_v_re = T.dot(step2_im, v_re.T).dimshuffle(0, 'x')
        step2_im_dot_v_im = T.dot(step2_im, v_im.T).dimshuffle(0, 'x')

        dstep3_re_timesum_step2_re = (dstep3_re * step2_re).sum(axis=1)
        dstep3_re_timesum_step2_im = (dstep3_re * step2_im).sum(axis=1)
        dstep3_im_timesum_step2_re = (dstep3_im * step2_re).sum(axis=1)
        dstep3_im_timesum_step2_im = (dstep3_im * step2_im).sum(axis=1)

        #--------

        dstep3_re_RedOpdv_re_term1 = - 2. / vstarv * (dstep3_re * step2_re_dot_v_re
                                                      + dstep3_re_dot_v_re * step2_re
                                                      - dstep3_re * step2_im_dot_v_im
                                                      + dstep3_re_dot_v_im * step2_im)

        outer_sum = (T.outer(step2_re_dot_v_re, v_re) 
                     + T.outer(step2_re_dot_v_im, v_im)
                     - T.outer(step2_im_dot_v_im, v_re)
                     + T.outer(step2_im_dot_v_re, v_im))
        dstep3_re_RedOpdv_re_term2 = 4. / (vstarv**2) * T.outer((dstep3_re * outer_sum).sum(axis=1), v_re)

        dstep3_im_ImdOpdv_re_term1 = - 2. / vstarv * (dstep3_im * step2_im_dot_v_re
                                                      + dstep3_im_dot_v_re * step2_im
                                                      + dstep3_im * step2_re_dot_v_im
                                                      - dstep3_im_dot_v_im * step2_re)

        outer_sum = (T.outer(step2_im_dot_v_re, v_re) 
                     + T.outer(step2_im_dot_v_im, v_im)
                     + T.outer(step2_re_dot_v_im, v_re)
                     - T.outer(step2_re_dot_v_re, v_im))
        dstep3_im_ImdOpdv_re_term2 = 4. / (vstarv**2) * T.outer((dstep3_im * outer_sum).sum(axis=1), v_re)

        dv_re_contribution = (dstep3_re_RedOpdv_re_term1 + dstep3_re_RedOpdv_re_term2 
                              + dstep3_im_ImdOpdv_re_term1 + dstep3_im_ImdOpdv_re_term2)

        #---------

        dstep3_re_RedOpdv_im_term1 = - 2. / vstarv * (dstep3_re * step2_re_dot_v_im
                                                      + dstep3_re_dot_v_im * step2_re
                                                      - dstep3_re_dot_v_re * step2_im
                                                      + dstep3_re * step2_im_dot_v_re)

        outer_sum = (T.outer(step2_re_dot_v_re, v_re) 
                     + T.outer(step2_re_dot_v_im, v_im)
                     - T.outer(step2_im_dot_v_im, v_re)
                     + T.outer(step2_im_dot_v_re, v_im))
        dstep3_re_RedOpdv_im_term2 = 4. / (vstarv**2) * T.outer((dstep3_re * outer_sum).sum(axis=1), v_im)


        dstep3_im_ImdOpdv_im_term1 = - 2. / vstarv * (dstep3_im * step2_im_dot_v_im
                                                      + dstep3_im_dot_v_im * step2_im
                                                      + dstep3_im_dot_v_re * step2_re
                                                      - dstep3_im * step2_re_dot_v_re)

        outer_sum = (T.outer(step2_im_dot_v_re, v_re) 
                     + T.outer(step2_im_dot_v_im, v_im)
                     + T.outer(step2_re_dot_v_im, v_re)
                     - T.outer(step2_re_dot_v_re, v_im))
        dstep3_im_ImdOpdv_im_term2 = 4. / (vstarv**2) * T.outer((dstep3_im * outer_sum).sum(axis=1), v_im)

        dv_im_contribution = (dstep3_re_RedOpdv_im_term1 + dstep3_re_RedOpdv_im_term2 
                              + dstep3_im_ImdOpdv_im_term1 + dstep3_im_ImdOpdv_im_term2)

        dreflection_t = T.inc_subtensor(dreflection_t[:, 0, :n_hidden], dv_re_contribution)
        dreflection_t = T.inc_subtensor(dreflection_t[:, 0, n_hidden:], dv_im_contribution)

        # theta0------------------------------------------------------------------------------
        dtheta0_contribution = dstep1 * times_diag(step0, n_hidden, theta[0,:] + 0.5 * np.pi)
        dtheta_t = T.inc_subtensor(dtheta_t[:, 0,:], dtheta0_contribution[:, :n_hidden] +
                                   dtheta0_contribution[:, n_hidden:])          

        # Compute deriv contributions to V --------------------------------------------------
        ddata_linoutput_re = ddata_linoutput[:, :n_hidden]
        ddata_linoutput_im = ddata_linoutput[:, n_hidden:]
        dV_re_contribution = T.batched_dot(x_t_plus_1.dimshuffle(0,1,'x'),
                                           ddata_linoutput_re.dimshuffle(0,'x',1))
        dV_im_contribution = T.batched_dot(x_t_plus_1.dimshuffle(0,1,'x'),
                                           ddata_linoutput_im.dimshuffle(0,'x',1))

        dV_re_t = dV_re_t + dV_re_contribution
        dV_im_t = dV_im_t + dV_im_contribution


        return [dh_t, h_t,
                dV_re_t, dV_im_t, dhidden_bias_t, dtheta_t,
                dreflection_t, dscale_t, dU_t, dout_bias_t] 


    yprev = y
    yprev = T.set_subtensor(yprev[1:], y[0:-1])
    isend = T.alloc(0, x.shape[0])
    isend = T.set_subtensor(isend[0], 1)

    h_T = hidden_states[-1,:,:]
    dh_T = compute_dctdht(h_T, U, out_bias, y[-1, :, :]) 

    non_sequences = [V_re, V_im, hidden_bias, theta, reflection, scale, U, out_bias]
    outputs_info = [dh_T, h_T,
                    dV_re, dV_im, dhidden_bias, dtheta, 
                    dreflection, dscale, dU, dout_bias]

    
    [dhs, hs,
     dV_res, dV_ims, dhidden_biass, dthetas, 
     dreflections, dscales, dUs, dout_biass], updates = theano.scan(fn=gradient_recurrence,
                                                                    sequences=[x[::-1], y[::-1], yprev[::-1], isend[::-1]],
                                                                    non_sequences=non_sequences,
                                                                    outputs_info=outputs_info)

     
    dh_0 = dhs[-1].dimshuffle(0,'x',1) 
    grads_per_datapoint = [dh_0,
                           dV_res[-1], dV_ims[-1], 
                           dhidden_biass[-1], dthetas[-1], 
                           dreflections[-1], dscales[-1],
                           dUs[-1], dout_biass[-1]]

    gradients = [g.mean(axis=0) for g in grads_per_datapoint]

    costs.append(costs[1]) # CHANGGGGGGGEEEEEEEEEEEE!!!!!!!!!!!!!!!!!!!!!!!-----------------------
    
    return [x, y], parameters, costs, gradients
