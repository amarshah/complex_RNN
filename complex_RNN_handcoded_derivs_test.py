import cPickle
import gzip
import theano
import pdb
from fftconv import cufft, cuifft
import numpy as np
import theano.tensor as T
from theano.ifelse import ifelse


### we need to hand code derivatives to save memory. 
### scan saves all the hidden units which kills memory - it isn't needed

############################################################3

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

    n_theta = 3
    n_reflect = 2

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
    ########
    theano.config.compute_test_value = 'warn'
    time_steps_test = 5
    batch_size_test = 10
    x.tag.test_value = np.random.rand(time_steps_test, batch_size_test, n_input).astype('float32') 
    temp = np.zeros((time_steps_test, batch_size_test, n_output)).astype('float32')
    ind = np.random.randint(n_output, size=(time_steps_test, batch_size_test))
    for time in xrange(time_steps_test):
        for batch in xrange(batch_size_test):
            temp[time, batch, ind[time, batch]] = np.float32(1.) 
    y.tag.test_value = temp
    ########

    index_permute = np.random.permutation(n_hidden)
    reverse_index_permute = np.zeros_like(index_permute)
    reverse_index_permute[index_permute] = range(n_hidden)

    # DEFINE FUNCTIONS FOR COMPLEX UNITARY TRANSFORMS----------------------------
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
        
        input_re_reflect = input_re - 2. / vstarv * (T.outer(T.dot(input_re, reflect_re), reflect_re) 
                                                     + T.outer(T.dot(input_re, reflect_im), reflect_im) 
                                                     - T.outer(T.dot(input_im, reflect_im), reflect_re) 
                                                     + T.outer(T.dot(input_im, reflect_re), reflect_im))
        input_im_reflect = input_im - 2. / vstarv * (T.outer(T.dot(input_im, reflect_re), reflect_re) 
                                                     + T.outer(T.dot(input_im, reflect_im), reflect_im) 
                                                     + T.outer(T.dot(input_re, reflect_im), reflect_re) 
                                                     - T.outer(T.dot(input_re, reflect_re), reflect_im))

        return T.concatenate([input_re_reflect, input_im_reflect], axis=1)      

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

        dtotal_linoutput = dh_t_plus_1 * T.tile(rescale, [1, 2]) \
            + T.tile(matrix, [1, 2]) * total_linoutput 

        dhidden_linoutput = dtotal_linoutput
        ddata_linoutput = dtotal_linoutput

        # Compute deriv contributions to hidden bias-------------------------------------------------------        
        dhidden_bias_contribution = dh_t_plus_1_TL * T.tile(deriv / modTL, [1, 2])

        dhidden_bias_t = dhidden_bias_t + dhidden_bias_contribution[:, :n_hidden] \
            + dhidden_bias_contribution[:, n_hidden:]

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

    actual_gradients = T.grad(costs[0], parameters, disconnected_inputs='ignore')
    
    for i in range(len(parameters)):
        print ((np.abs(gradients[i]-actual_gradients[i])/(actual_gradients[i] + 1e-6)).tag.test_value).max()

    print

    for i in range(len(parameters)):
        print ((np.abs(gradients[i]-actual_gradients[i])).tag.test_value).max()
    

    import pdb; pdb.set_trace()

    return [x, y], parameters, costs, gradients#, actual_gradients

 
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
 
    inputs, parameters, costs, gradients = complex_RNN(n_input, n_hidden, n_output, scale_penalty)

    def test_verify_grad():
        def fun(h_0, V_re, V_im, hidden_bias, theta, reflection, scale, U, out_bias):
            return costs[0]

        T.verify_grad(fun, [p.get_value() for p in parameters], rng=rng)
       

    if not use_scale:
        del parameters[-3] 
   
    s_train_x = theano.shared(train_x, borrow=True)
    s_train_y = theano.shared(train_y, borrow=True)

    s_valid_x = theano.shared(valid_x, borrow=True)
    s_valid_y = theano.shared(valid_y, borrow=True)


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
    kwargs = {'n_iter': 10,
              'n_batch': 20,
              'n_hidden': 1000,
              'time_steps': 10,
              'learning_rate': np.float32(0.001),
              'savefile': '/data/lisatmp3/shahamar/2015-11-03-gradient-tests.pkl',
              'scale_penalty': 1,
              'use_scale': True}

    main(**kwargs)
