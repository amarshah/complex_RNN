import theano
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

    h_0 = theano.shared(np.zeros((1, n_hidden)))
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

    h_0 = theano.shared(np.zeros((1, n_hidden)))
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
