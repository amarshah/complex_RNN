import cPickle
import gzip
import theano
import pdb
from fftconv import cufft, cuifft
import numpy as np
import theano.tensor as T
from theano.ifelse import ifelse


class HadamardOp(theano.Op):
    __props__ = ()

#    import pdb; pdb.set_trace()
    
    def hadamard(self, v):
        n_in = v.shape[1]
        n = n_in
        while n > 1:
            for n_start in xrange(0, n_in, n):
                first_half = np.sqrt(0.5) * v[:,n_start : (n_start + n/2)]
                second_half = np.sqrt(0.5) * v[:,(n_start + n/2) : (n_start + n)]
                v[:,n_start : (n_start + n/2)] = first_half + second_half
                v[:,(n_start + n/2) : (n_start + n)] = first_half - second_half
            n = n/2
        return v

    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        z[0] = self.hadamard(x)

    def grad(self, inputs, output_grads):
        return [HadamardOp()(output_grads[0])]




def initialize_matrix(n_in, n_out, name, rng):
    bin = np.sqrt(6. / (n_in + n_out))
    values = np.asarray(rng.uniform(low=-bin,
                                    high=bin,
                                    size=(n_in, n_out)),
                        dtype=theano.config.floatX)
    return theano.shared(value=values, name=name)


# computes Theano graph
# returns symbolic parameters, costs, inputs 
# there are n_hidden real units and a further n_hidden imaginary units 
def complex_RNN(n_input, n_hidden, n_output, scale_penalty):
    
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


    parameters = [V_re, V_im, hidden_bias, theta, U, out_bias, h_0, reflection, scale] 

    x = T.tensor3()
    y = T.matrix()#T.tensor3()
#    x.tag.test_value = np.random.rand(100,10,1).astype('float32') 
#    y.tag.test_value = np.random.rand(100,10,1).astype('float32')
     
    # TEMPORARY Hadamard computation
    def sethadamard(n):
        if n==1:
            return np.array([[1]], dtype=theano.config.floatX)
        else:
            H = sethadamard(n/2)
            col1 = np.concatenate((H, H), axis=0)
            col2 = np.concatenate((H, -H), axis=0)
            return np.sqrt(1./2) * np.concatenate((col1, col2), axis=1)

    index_permute = np.random.permutation(n_hidden)
 
    # define the recurrence used by theano.scan
    def recurrence(x_t, h_prev, theta, V_re, V_im, hidden_bias, scale):  
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
            input_re_reflect = input_re - 2 / vstarv * (T.outer(T.dot(input_re, reflect_re), reflect_re) +
                                                        T.outer(T.dot(input_im, reflect_im), reflect_im))
            input_im_reflect = input_im - 2 / vstarv * (-T.outer(T.dot(input_re, reflect_im), reflect_im) +
                                                        T.outer(T.dot(input_im, reflect_re), reflect_re))

            return T.concatenate([input_re_reflect, input_im_reflect], axis=1)      


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
        
        hidden_lin_output = step7
        
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
        rescale = T.maximum(modulus + hidden_bias.dimshuffle('x',0), 0.) / (modulus + 1e-5)
        nonlin_output_re = lin_output_re * rescale
        nonlin_output_im = lin_output_im * rescale

        # relu on each part
#        nonlin_output_re = T.maximum(lin_output_re + hidden_bias.dimshuffle('x', 0), 0.)
#        nonlin_output_im = T.maximum(lin_output_im + hidden_bias.dimshuffle('x', 0), 0.)        

        h_t = T.concatenate([nonlin_output_re, 
                             nonlin_output_im], axis=1) 

        return h_t

    # compute hidden states
    h_0_batch = T.tile(h_0, [x.shape[1], 1])
    non_sequences = [theta, V_re, V_im, hidden_bias, scale]
    hidden_states, updates = theano.scan(fn=recurrence,
                                         sequences=x,
                                         non_sequences=non_sequences,
                                         outputs_info=h_0_batch)

    # define hidden to output graph
    RNN_output = T.dot(hidden_states[-1,:,:], U) + out_bias.dimshuffle('x', 0)
    
    # define the cost
    cost = ((RNN_output - y)**2).mean()
    cost.name = 'mse'
    cost_penalty = cost + scale_penalty * ((scale-1)**2).sum()
    cost_penalty.name = 'mse_penalty'

    costs = [cost_penalty, cost]

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
    


# Warning: assumes n_batch is a divisor of number of data points
# Suggestion: preprocess outputs to have norm 1 at each time step
def main(n_iter, n_batch, n_hidden, time_steps, learning_rate, savefile, scale_penalty, use_scale):

    #import pdb; pdb.set_trace()
 
    # --- Set optimization params --------
    gradient_clipping = np.float32(50000)

    # --- Set data params ----------------
    n_input = 2
    n_output = 1
  

    # --- Manage data --------------------
    n_train = 1e5
    n_test = 1e4
    num_batches = n_train / n_batch
    
    train_x = np.asarray(np.zeros((time_steps, n_train, 2)),
                         dtype=theano.config.floatX)
    

    train_x[:,:,0] = np.asarray(np.random.uniform(low=0.,
                                                  high=1.,
                                                  size=(time_steps, n_train)),
                                dtype=theano.config.floatX)
    
#    inds = np.asarray([np.random.choice(time_steps, 2, replace=False) for i in xrange(train_x.shape[1])])    
    inds = np.asarray(np.random.randint(time_steps/2, size=(train_x.shape[1],2)))
    inds[:, 1] += time_steps/2  
    
    for i in range(train_x.shape[1]):
        train_x[inds[i, 0], i, 1] = 1.0
        train_x[inds[i, 1], i, 1] = 1.0
 
    train_y = (train_x[:,:,0] * train_x[:,:,1]).sum(axis=0)
    train_y = np.reshape(train_y, (n_train, 1))

    test_x = np.asarray(np.zeros((time_steps, n_test, 2)),
                        dtype=theano.config.floatX)
    

    test_x[:,:,0] = np.asarray(np.random.uniform(low=0.,
                                                 high=1.,
                                                 size=(time_steps, n_test)),
                                dtype=theano.config.floatX)
    
    inds = np.asarray([np.random.choice(time_steps, 2, replace=False) for i in xrange(test_x.shape[1])])    
    for i in range(test_x.shape[1]):
        test_x[inds[i, 0], i, 1] = 1.0
        test_x[inds[i, 1], i, 1] = 1.0
 
   
    test_y = (test_x[:,:,0] * test_x[:,:,1]).sum(axis=0)
    test_y = np.reshape(test_y, (n_test, 1)) 


   #######################################################################

    # --- Compile theano graph and gradients
 
    inputs, parameters, costs = complex_RNN(n_input, n_hidden, n_output, scale_penalty)
    if not use_scale:
        parameters.pop() 
   
    gradients = T.grad(costs[0], parameters)

    s_train_x = theano.shared(train_x)
    s_train_y = theano.shared(train_y)

    s_test_x = theano.shared(test_x)
    s_test_y = theano.shared(test_y)


    # --- Compile theano functions --------------------------------------------------

    index = T.iscalar('i')

    updates, rmsprop = rms_prop(learning_rate, parameters, gradients)

    givens = {inputs[0] : s_train_x[:, n_batch * index : n_batch * (index + 1), :],
              inputs[1] : s_train_y[n_batch * index : n_batch * (index + 1), :]}

    givens_test = {inputs[0] : s_test_x,
                   inputs[1] : s_test_y}
    
   
    
    train = theano.function([index], costs[0], givens=givens, updates=updates)
    test = theano.function([], costs[1], givens=givens_test)

    # --- Training Loop ---------------------------------------------------------------
    train_loss = []
    test_loss = []
    best_params = [p.get_value() for p in parameters]
    best_test_loss = 1e6
    for i in xrange(n_iter):
     #   pdb.set_trace()

        mse = train(i % num_batches)
        train_loss.append(mse)
        print "Iteration:", i
        print "mse:", mse
        print

        if (i % 25==0):
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
                         'best_test_loss': best_test_loss}

            cPickle.dump(save_vals,
                         file(savefile, 'wb'),
                         cPickle.HIGHEST_PROTOCOL)





    
if __name__=="__main__":
    kwargs = {'n_iter': 20000,
              'n_batch': 20,
              'n_hidden': 512,
              'time_steps': 200,
              'learning_rate': np.float32(0.001),
              'savefile': '/data/lisatmp3/shahamar/2015-10-29-adding-200-fft-noscale.pkl',
              'scale_penalty': 10,
              'use_scale': True}

    main(**kwargs)
