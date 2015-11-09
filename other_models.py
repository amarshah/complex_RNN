import theano
import theano.tensor as T
import numpy as np

def initialize_matrix(n_in, n_out, name, rng):
    bin = np.sqrt(6. / (n_in + n_out))
    values = np.asarray(rng.uniform(low=-bin,
                                    high=bin,
                                    size=(n_in, n_out)),
                                    dtype=theano.config.floatX)
    return theano.shared(value=values, name=name)


def IRNN(n_input, n_hidden, n_output):
    np.random.seed(1234)
    rng = np.random.RandomState(1234)

    x = T.tensor3()
    y = T.matrix()
    inputs = [x, y]

    h_0 = theano.shared(np.zeros((1, n_hidden)))
    V = initialize_matrix(n_input, n_hidden, 'V', rng)
    W = theano.shared(np.identity(n_hidden, dtype=theano.config.floatX))
    U = initialize_matrix(n_hidden, n_output, 'U', rng)
    hidden_bias = theano.shared(np.zeros((1, n_hidden), dtype=theano.config.floatX))
    out_bias = theano.shared(np.zeros((1, n_output), dtype=theano.config.floatX))
    parameters = [h_0, V, W, U, hidden_bias, out_bias]

    hidden_bias_batch = T.tile(hidden_bias, [x.shape[1], 1])

    def recurrence(x_t, h_prev, V, W, hidden_bias_batch):
        return T.nnet.relu(T.dot(h_prev, W) + T.dot(x_t, V) + hidden_bias_batch)
    
    non_sequences = [V, W, hidden_bias_batch]

    h_0_batch = T.tile(h_0, [x.shape[1], 1])

    hidden_states, updates = theano.scan(fn = recurrence, sequences = x, non_sequences = non_sequences, outputs_info = h_0_batch)

    # ONE OUTPUT
    out_bias_batch = T.tile(out_bias, [x.shape[1], 1])
    linear_output = T.dot(hidden_states[-1, :, :], U) + out_bias_batch
    
    # CALCULATE LOSS
    output = T.nnet.softmax(linear_output)
    loss = T.nnet.categorical_crossentropy(output, y).mean()
    accuracy = T.mean(T.eq(T.argmax(output, axis=-1), T.argmax(y, axis=-1)))


    costs = [loss, loss, accuracy]


    return inputs, parameters, costs

def tanhRNN(n_input, n_hidden, n_output):
    np.random.seed(1234)
    rng = np.random.RandomState(1234)

    x = T.tensor3()
    y = T.matrix()
    inputs = [x, y]

    h_0 = theano.shared(np.zeros((1, n_hidden)))
    V = initialize_matrix(n_input, n_hidden, 'V', rng)
    W = initialize_matrix(n_hidden, n_hidden, 'W', rng)
    U = initialize_matrix(n_hidden, n_output, 'U', rng)
    hidden_bias = theano.shared(np.zeros((1, n_hidden), dtype=theano.config.floatX))
    out_bias = theano.shared(np.zeros((1, n_output), dtype=theano.config.floatX))
    parameters = [h_0, V, W, U, hidden_bias, out_bias]

    hidden_bias_batch = T.tile(hidden_bias, [x.shape[1], 1])

    def recurrence(x_t, h_prev, V, W, hidden_bias_batch):
        return T.tanh(T.dot(h_prev, W) + T.dot(x_t, V) + hidden_bias_batch)
    
    non_sequences = [V, W, hidden_bias_batch]

    h_0_batch = T.tile(h_0, [x.shape[1], 1])

    hidden_states, updates = theano.scan(fn = recurrence, sequences = x, non_sequences = non_sequences, outputs_info = h_0_batch)

    # ONE OUTPUT
    out_bias_batch = T.tile(out_bias, [x.shape[1], 1])
    linear_output = T.dot(hidden_states[-1, :, :], U) + out_bias_batch
    
    # CALCULATE LOSS
    output = T.nnet.softmax(linear_output)
    loss = T.nnet.categorical_crossentropy(output, y).mean()
    accuracy = T.mean(T.eq(T.argmax(output, axis=-1), T.argmax(y, axis=-1)))

    costs = [loss, loss, accuracy]


    return inputs, parameters, costs


def LSTM(n_input, n_hidden, n_output):
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
    b_i = theano.shared(np.zeros((1, n_hidden), dtype=theano.config.floatX))
    b_f = theano.shared(np.zeros((1, n_hidden), dtype=theano.config.floatX))
    b_c = theano.shared(np.zeros((1, n_hidden), dtype=theano.config.floatX))
    b_o = theano.shared(np.zeros((1, n_hidden), dtype=theano.config.floatX))
    h_0 = theano.shared(np.zeros((1, n_hidden), dtype=theano.config.floatX))
    state_0 = theano.shared(np.zeros((1, n_hidden), dtype=theano.config.floatX))
    out_mat = initialize_matrix(n_hidden, n_output, 'out_mat', rng)
    out_bias = theano.shared(np.zeros((1, n_output), dtype=theano.config.floatX))
    parameters = [W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o, V_o, b_i, b_f, b_c, b_o, h_0, state_0, out_mat, out_bias]

    x = T.tensor3()
    y = T.matrix()
    
    b_i_batch = T.tile(b_i, [x.shape[1], 1])
    b_f_batch = T.tile(b_f, [x.shape[1], 1])
    b_c_batch = T.tile(b_c, [x.shape[1], 1])
    b_o_batch = T.tile(b_o, [x.shape[1], 1])

    def recurrence(x_t, h_prev, state_prev, W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o, V_o, b_i_batch, b_f_batch, b_c_batch, b_o_batch):
        input_t = T.nnet.sigmoid(T.dot(x_t, W_i) + T.dot(h_prev, U_i) + b_i_batch)
        candidate_t = T.tanh(T.dot(x_t, W_c) + T.dot(h_prev, U_c) + b_c_batch)
        forget_t = T.nnet.sigmoid(T.dot(x_t, W_f) + T.dot(h_prev, U_f) + b_f_batch)

        state_t = input_t * candidate_t + forget_t * state_prev

        output_t = T.nnet.sigmoid(T.dot(x_t, W_o) + T.dot(h_prev, U_o) + T.dot(state_t, V_o) + b_o_batch)

        h_t = output_t * T.tanh(state_t)

        return h_t, state_t

    non_sequences = [W_i, W_f, W_c, W_o, U_i, U_f, U_c, U_o, V_o, b_i_batch, b_f_batch, b_c_batch, b_o_batch]

    h_0_batch = T.tile(h_0, [x.shape[1], 1])
    state_0_batch = T.tile(state_0, [x.shape[1], 1])

    [hidden_states, states], updates = theano.scan(fn = recurrence, sequences = x, non_sequences = non_sequences, outputs_info = [h_0_batch, state_0_batch])

    # ONE OUTPUT
    out_bias_batch = T.tile(out_bias, [x.shape[1], 1])
    linear_output = T.dot(hidden_states[-1, :, :], out_mat) + out_bias_batch
    
    # CALCULATE LOSS
    output = T.nnet.softmax(linear_output)
    loss = T.nnet.categorical_crossentropy(output, y).mean()
    accuracy = T.mean(T.eq(T.argmax(output, axis=-1), T.argmax(y, axis=-1)))

    costs = [loss, loss, accuracy]


    return [x, y], parameters, costs


