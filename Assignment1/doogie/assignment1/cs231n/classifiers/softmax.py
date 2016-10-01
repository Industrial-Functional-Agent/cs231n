import numpy as np
from random import shuffle

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
  num_classes = W.shape[1]
  num_train = X.shape[0]
    
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
    scores = X[i].dot(W)
    exp_scores = np.exp(scores)
    correct_class_score = exp_scores[y[i]]
    normalized_probability = correct_class_score / np.sum(exp_scores)
    normalized_log_probability = -np.log(normalized_probability)
    loss += normalized_log_probability
    for j in xrange(num_classes):
      if j == y[i]:
        dW[:, y[i]] += (1 / np.sum(exp_scores) - 1 / exp_scores[y[i]]) * exp_scores[y[i]] * X[i]
      else:
        dW[:, j] +=  1 / np.sum(exp_scores) * exp_scores[j] * X[i]
  loss /= num_train  
  dW /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  exp_scores = np.exp(scores)
  correct_class_score = exp_scores[np.arange(num_train), y]
  class_score_sum = np.sum(exp_scores, axis=1)
  normalized_probability = correct_class_score / class_score_sum
  normalized_log_probability = -np.log(normalized_probability)
  loss = np.sum(normalized_log_probability)
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  #dlog = -1 / normalized_probability
  #dmul = dlog * correct_class_score
  #dinv = -1 / np.square(class_score_sum) * dmul
  dinv = 1 / class_score_sum
  dexp = exp_scores * dinv.reshape(num_train, 1)

  dcorrect = np.zeros((num_train, num_classes))
  dcorrect[np.arange(num_train), y] = -1
  dexp += dcorrect
  dW = X.T.dot(dexp)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

