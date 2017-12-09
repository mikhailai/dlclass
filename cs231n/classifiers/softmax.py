import numpy as np
from random import shuffle
from past.builtins import xrange

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
  num_samples = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for s in range(num_samples):
      scores = W.T.dot(X[s])
      max_score = np.max(scores)
      scores -= max_score
      escores = np.exp(scores)
      esum = np.sum(escores)
      prob = escores[y[s]] / esum
      loss -= np.log(prob)
      for j in range(W.shape[1]):
          if y[s] == j:
              dW[:, j] += -X[s]
          dW[:, j] += escores[j] / esum * X[s]
  loss /= num_samples
  dW /= num_samples
  loss += reg * np.sum(W ** 2)
  dW += 2 * reg * W;
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  # Numeric stability:
  scores -= np.amax(scores, axis=1, keepdims=True)
  escores = np.exp(scores)
  esums = np.sum(escores, axis=1, keepdims=True)
  correct_scores = scores[np.arange(len(scores)), y]
  loss = (np.sum(np.log(esums)) - np.sum(correct_scores)) / len(X) + reg * np.sum(W ** 2)

  multiplier = escores / esums
  multiplier[np.arange(len(multiplier)), y] -= 1;
  np.dot(X.T, multiplier, out=dW)
  dW /= len(X)
  dW += 2 * reg * W;
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

