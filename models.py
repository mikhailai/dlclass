import numpy as np

class TwoLayerNet(object):
  """
  Two-layer networks:  dense -> ReLU -> dense -> logistic
  """
  def __init__(self, D, H, reg=0.0):
    self.params = dict()
    # Just using Xavier weight initialization:
    self.reg = reg
    # Note, input data size: [N, D].
    self.params['W1'] = np.random.randn(D, H).astype("float32") / np.sqrt(D)
    self.params['b1'] = np.zeros([H], dtype=np.float32)
    self.params['W2'] = np.random.randn(H).astype("float32") / np.sqrt(H)
    self.params['b2'] = np.zeros([1], dtype=np.float32)
    
  def loss(self, X, y=None):
    N = len(X)
    H1 = X.dot(self.params['W1']) + self.params['b1']
    H1_relu = np.maximum(H1, 0)
    scores = H1_relu.dot(self.params['W2']) + self.params['b2']

    if y is None:
      return (scores > 0).astype("int")
    
    scores2 = np.zeros([len(scores), 2])
    scores2[:, 1] = scores
    scores2 -= np.amax(scores2, axis=1, keepdims=1)
    escores2 = np.exp(scores2)
    escores2sum = np.sum(escores2, axis=1)

    loss = np.mean(-scores2[np.arange(len(y)), y] + np.log(escores2sum))

    grads = {}
    
    grad_score = (-y + escores2[:, 1] / escores2sum) / N;
    
    grads['b2'] = np.sum(grad_score)
    grads['W2'] = H1_relu.T.dot(grad_score)
    grad_H1 = np.reshape(grad_score, [len(grad_score), 1]).dot(
        np.reshape(self.params['W2'], [1, len(self.params['W2'])]))  # Relu grad.
    grad_H1[H1_relu == 0] = 0
    grads['b1'] = np.sum(grad_H1, axis=0)
    grads['W1'] = X.T.dot(grad_H1)
    
    for key in ['W1', 'W2']:
      loss += self.reg * np.sum(self.params[key] ** 2)
      grads[key] += 2 * self.reg * self.params[key]
        
    return loss, grads
