import numpy as np
from random import shuffle
#from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_W = W.shape[1]
  df = np.zeros_like(X.dot(W))
  for i in xrange(num_train):
    loss_sum = 0.0
    loss_tmp = []
    for j in xrange(num_W):
      s = X[i].dot(W[:,j])
      loss_sum += np.exp(s)
      loss_tmp.append(np.exp(s))
    tmp_pi = -np.log(loss_tmp[y[i]]/loss_sum)
    loss += tmp_pi
    df[i,:] = loss_tmp/loss_sum
    df[i,y[i]] -= 1

  #print (df)
  dW = np.dot(X.T, df)
  dW /= num_train
  #db = np.sum(dscores, axis=0, keepdims=True)
  dW += reg*W
    
  loss /= num_train 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss += 0.5*reg*np.sum(W*W)
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]

  exp_term = np.exp(X.dot(W))
  sum_term = np.sum(exp_term, axis=1)
  #print (exp_term.shape)
  #print (sum_term.shape)

  temp = exp_term / np.sum(exp_term, axis=1, keepdims=True)
  #print (temp.shape)
  loss = np.sum(-np.log(temp[range(num_train),y]))
  loss /= num_train
  loss += 0.5*reg*np.sum(W*W)
    
    
  df = temp
  df[range(num_train),y] -= 1
  dW = np.dot(X.T, df)
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

