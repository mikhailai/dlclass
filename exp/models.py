import numpy as np
import layers
import layers2


def _pred_logistic(scores):
  return (scores[:, 0] > 0).astype("int")

def _pred_multiclass(scores):
  return np.argmax(scores, axis=1)


def fc_multilayer_net(dims, reg=0, dtype=np.float32):
  """ 
  Creates a fully-connected multi-layer network for classification
  [affine -> relu] -> [affine -> relu] ... affine -> logistic_loss
  """
  shared_params = layers2.SharedParams()
  shared_params.reg = reg
  lst = list()
  for i in range(1, len(dims)):
    lst.append(layers2.AffineLayer(str(i), shared_params, dims[i-1], dims[i], dtype))
    if i != len(dims) - 1:
      lst.append(layers2.ReluLayer())
  if dims[-1] == 1:
    return Model(lst, shared_params, _pred_logistic, layers2.logistic_loss)
  else:
    return Model(lst, shared_params, _pred_multiclass, layers.softmax_loss)


class Model(object):
  def __init__(self, layers, shared_params, pred_fn, loss_fn):
    self.layers = layers
    self.shared_params = shared_params
    self.pred_fn = pred_fn
    self.loss_fn = loss_fn
    self.params = {}
    
  def loss(self, X, y=None):
    self.params.clear()
    scores = X
    for layer in self.layers:
      scores = layer.forward(scores, self.params)
    if y is None:
      return self.pred_fn(scores)
    # Computing the loss and gradients:
    loss, dout = self.loss_fn(scores, y)
    for layer in self.layers:
      loss += layer.regularization_loss()
    grads = {}
    for layer in reversed(self.layers):
      dout = layer.backward(dout, grads)
    return loss, grads


class TwoLayerNet(object):
  """
  Two-layer networks:  dense -> ReLU -> dense -> logistic
  """
  def __init__(self, D, H, reg=0.0, dtype=np.float32):
    self.params = dict()
    # Just using Xavier weight initialization:
    self.reg = reg
    # Note, input data size: [N, D].
    self.params['W1'] = np.random.randn(D, H).astype(dtype) / np.sqrt(D)
    self.params['b1'] = np.zeros([H], dtype=dtype)
    self.params['W2'] = np.random.randn(H, 1).astype(dtype) / np.sqrt(H)
    self.params['b2'] = np.zeros([1], dtype=dtype)
    
  def loss(self, X, y=None):
    N = len(X)
    H1 = X.dot(self.params['W1']) + self.params['b1']
    H1_relu = np.maximum(H1, 0)
    scores = H1_relu.dot(self.params['W2']) + self.params['b2']

    if y is None:
      return _pred_logistic(scores)
    
    assert len(scores.shape) == 2 and scores.shape[1] == 1
    scores2 = np.zeros([len(scores), 2])
    scores2[:, 1] = scores[:, 0]
    scores2 -= np.amax(scores2, axis=1, keepdims=1)
    escores2 = np.exp(scores2)
    escores2sum = np.sum(escores2, axis=1)

    loss = np.mean(-scores2[np.arange(len(y)), y] + np.log(escores2sum))

    grads = {}
    
    grad_score = (-y + escores2[:, 1] / escores2sum) / N;
    grad_score = np.reshape(grad_score, [len(grad_score), 1])
    
    grads['b2'] = np.sum(grad_score, axis=0)
    grads['W2'] = H1_relu.T.dot(grad_score)
    grad_H1 = grad_score.dot(self.params['W2'].T)  # Relu grad.
    grad_H1[H1_relu == 0] = 0
    grads['b1'] = np.sum(grad_H1, axis=0)
    grads['W1'] = X.T.dot(grad_H1)
        
    for key in ['W1', 'W2']:
      loss += self.reg * np.sum(self.params[key] ** 2)
      grads[key] += 2 * self.reg * self.params[key]
        
    return loss, grads
