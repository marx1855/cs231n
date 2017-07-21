from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    i = 0
    d1 = x.shape[0]
    
    temp = 1 
    for size in x.shape:
        temp *= size
        #print (temp)
    temp /= d1
    
    
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################

    x_trans = x.reshape(d1,temp)

    out = x_trans.dot(w)
    out = out + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    db = np.sum(dout, axis=0)
    dw_ = np.dot(x.T, dout)
    #print (x.T.shape)
    #print (dout.shape)
    #print (dw_.shape)
    dw = dw_.reshape(w.shape, order = 'F')
    dx = np.dot(dout, w.T)
    #print (x.shape)
    dx = dx.reshape(x.shape)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.array(x)
    out[x < 0] = 0
    #print (out)

    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dRelu = np.zeros_like(dout)
    dRelu[x > 0] = 1
    dx = np.multiply(dRelu, dout)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        sample_mean = np.sum(x, axis=0) / N
        #print sample_mean
        diff_mean = x - sample_mean
        #print x
        sample_var = np.sum(np.multiply(diff_mean, diff_mean), axis=0) / N
        var_root = np.sqrt(sample_var + eps)
        xhat = np.divide(diff_mean, var_root)
        #print "gamma:" 
        #print gamma.shape
        '''
        print "sample_mean_shape = "
        print sample_mean.shape
        print "x_shape = "
        print x.shape
        print "x_hat_shape = "
        print xhat.shape
        print "gamma_shape = "
        print gamma.shape
        print "beta_shape = "
        print beta.shape
        '''
        y = np.multiply(xhat, gamma) + beta
        out = y
        cache = (x, xhat, gamma, beta, sample_mean, sample_var, diff_mean, var_root)
        
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        diff_mean = x - running_mean
        #sample_var = np.sum(np.mutiply(diff_mean, diff_mean)) / N
        
        var_root = np.sqrt(running_var)
        xhat = np.divide(diff_mean, var_root)
        
        #y = gamma * xhat + beta
        y = np.multiply(xhat, gamma) + beta
        out = y
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    x, xhat, gamma, beta, sample_mean, sample_var, diff_mean, var_root = cache
    N, D = x.shape
    dxhat = np.multiply(dout, gamma)
    '''
    print "dxhat.shape"
    print dxhat.shape
    print "diff_mean.shape"
    print diff_mean.shape
    print "sample_var.shape"
    print sample_var.shape
    '''
    
    dvar = (-0.5) * np.sum(np.multiply(np.multiply(dxhat, diff_mean), np.power(sample_var + 1e-5, (-1.5))),axis = 0)
    '''
    print "dxhat"
    print dxhat.shape
    print "var_root"
    print var_root.shape
    print "dvar"
    print dvar.shape
    print "diff_mean"
    print diff_mean.shape
    '''
    dmean = np.sum(np.multiply(dxhat, -1 / var_root), axis=0) + np.multiply(dvar, np.sum((-2) * diff_mean / N, axis = 0))
    #print "dmean"
    #print dmean.shape
    dx = np.multiply(dxhat, 1/var_root) + np.multiply(2 * diff_mean / N, dvar) + dmean / N
    
    dgamma = np.sum(np.multiply(dout, xhat), axis=0)
    dbeta = np.sum(dout, axis = 0)

       
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    N, D = dout.shape
    
    x, xhat, gamma, beta, sample_mean, sample_var, diff_mean, var_root = cache
    
    x_mu = sample_mean
    inv_var = 1./sample_var
    x_hat = xhat

    # intermediate partial derivatives
    dxhat = dout * gamma

	# final partial derivatives
    dx = (1 / N) * inv_var * (N*dxhat - np.sum(dxhat, axis=0) 
        - x_hat*np.sum(dxhat*x_hat, axis=0))
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(x_hat*dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = np.random.rand(x.shape[0], x.shape[1]) < p
        out = x * mask
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x * p
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    pad = conv_param['pad']
    stride = conv_param['stride']
    (N, C, H, W) = x.shape
    (F, C, HH, WW) = w.shape
    
    
    H_y = int(1 + (H + 2 * pad - HH) / stride)
    W_y = int(1 + (W + 2 * pad - WW) / stride)
    
    y = np.zeros((N, F, H_y, W_y))
    
    for n in range(N):
        for c in range (C):
            x_pad = np.pad(x[n,c,:,:], pad, 'constant')
            #x_pad.shape = ((H + 2 * pad), (W + 2 * pad))
            #print "x_pad.shape = "
            #print x_pad.shape
            
            (H_pad, W_pad) = x_pad.shape
            
            for f in range(F):
                kernels = w[f,:,:,:]
                #print "kernels.shape="
                #print kernels.shape
                
                #############################
                #  Convolution Calculation  #
                #############################
                
                for h_y in range(H_y):
                    for w_y in range(W_y):
                        #print x_pad[stride*h_y:stride*h_y + HH, stride*w_y:stride*w_y + WW]
                        
                        y[n, f, h_y, w_y] += np.sum(
                                                np.multiply(x_pad[stride*h_y:stride*h_y + HH, 
                                                                  stride*w_y:stride*w_y + WW],
                                                             kernels[c,:,:]))


    for n in range(N):
        for f in range(F):
            y[n, f, :, :] += b[f]
    out = y  
            
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    (x, w, b, conv_param) = cache
    pad = conv_param['pad']
    stride = conv_param['stride']
    (N, C, H, W) = x.shape
    (F, C, HH, WW) = w.shape
    (N, F, H_y, W_y) = dout.shape
    dx = np.zeros(x.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)
    
    for n in range(N):                     # for each output of one input x
        for f in range(F):                 # for each output after convolution with each kernel with servel channels
            dout_2D = dout[n,f,:,:]        # extract one 2D gradient matrix
#            print "dout_2D.shape"
#            print dout_2D
            
            for c in range(C):             # for each channel
                kernel = w[f,c,:,:]
                for h_y in range(H_y):      # for each row in 2D gradient matrix
                    for w_y in range(W_y):  # for each element in 2D gradient matrix
                         
                        for hh in range(HH):
                            for ww in range(WW):
                                
                                if h_y*stride+hh < pad or h_y*stride + hh >= H + pad:
                                    next
                                elif w_y*stride+ww < pad or w_y*stride + ww >= W + pad:
                                    next
                                else:
                                    dx[n,c,h_y*stride + hh - 1,w_y*stride + ww - 1] += dout_2D[h_y,w_y] * kernel[hh, ww]
                                    dw[f,c,hh,ww] += dout_2D[h_y,w_y] * x[n,c,h_y*stride + hh - 1,w_y*stride + ww - 1]
    for n in range(N):
        for f in range(F):
            db[f] += np.sum(dout[n,f,:,:])

    '''
chain rule:
        
[X_pad] -------
          dker \  X*k
              (x)-------------
               /  dout_2D[h,w]\
[kernel]-------                \
          dX                    \     out
                               (+)------- 
                                /     dout_2D[h,w]
b   ----------------------------                      
                  dout_2D[h,w]
                  
X_pad[0,0]----------
           d/k[0,0] \
                   (x)---------------
                    /  dout_2D[h,w]  \
kernel[0,0]---------                  \  
           d/x[0,0]                    (+)--------------
                                      /    dout_2D[h,w] \
X_pad[0,1]----------                 /                   \
           d/k[0,1] \               /                     \
                   (x)--------------                       \
                    /  dout_2D[h,w]                         \
kernel[0,1]---------                                         \   X*k
           d/x[0,1]                                         (+)---------
                                                             /   dout_2D[h,w]
X_pad[0,2]----------                                        /   
           d/k[0,2] \                                      /
                   (x)---------------                     /
                    /  dout_2D[h,w]  \                   /
kernel[0,2]---------                  \                 /
           d/x[0,2]                  (+)----------------
                                      /
X_pad[1,0]----------                 /  dout_2D[h,w]       
           d/k[1,0] \               /         
                   (x)--------------         
                    /  dout_2D[h,w]            
kernel[1,0]---------              
           d/x[1,0]
    .
    .
    .
    .


    '''    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    (N, C, H, W) = x.shape
    p_h = pool_param['pool_height']
    p_w = pool_param['pool_width']
    stride = pool_param['stride']
    
    F_h = p_h
    F_w = p_w
    S = stride
    
    H_y = (H - F_h)/S + 1
    W_y = (W - F_w)/S + 1
    
    y = np.zeros((N, C, H_y, W_y))

    for h_y in range(H_y):
        for w_y in range(W_y):
            y[:,:,h_y, w_y] = np.max(np.max(x[:,:,h_y*stride:h_y*stride + F_h, w_y*stride:w_y*stride + F_w], axis=3), axis=2)
            #print np.max(np.max(x[:,:,h_y*stride:h_y*stride + F_h, w_y*stride:w_y*stride + F_w], axis=3), axis=2)

    out = y
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    
    (x, pool_param) = cache
    
    (N, C, H, W) = x.shape
    dx = np.zeros(x.shape)
    
    p_h = pool_param['pool_height']
    p_w = pool_param['pool_width']
    stride = pool_param['stride']
    
    F_h = p_h
    F_w = p_w
    S = stride
    
    H_y = (H - F_h)/S + 1
    W_y = (W - F_w)/S + 1
    
    for n in range(N):
        for c in range(C):
            for h_y in range(H_y):
                for w_y in range(W_y):
                    pos = np.unravel_index(np.argmax(x[n,c,h_y*stride:h_y*stride + F_h, w_y*stride:w_y*stride + F_w]), (F_h, F_w))
                    
                    #print pos
                    dx[n,c,h_y*stride + pos[0],w_y*stride + pos[1]] = dout[n,c,h_y, w_y]


    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
