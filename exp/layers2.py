import abc
from layers import *

def logistic_loss(scores, y):
    """
    Computes the loss and gradient for the logistic regression.

    Inputs:
    - scores: Input vector, of shape (N,) - one value for each input.
    - y: Integer vector of 0/1 values, of shape (N,)

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    assert len(scores.shape) == 2 and scores.shape[1] == 1
    scores2 = np.zeros([len(scores), 2])
    scores2[:, 1] = scores[:, 0]
    scores2 -= np.amax(scores2, axis=1, keepdims=1)
    escores2 = np.exp(scores2)
    escores2sum = np.sum(escores2, axis=1)

    loss = np.mean(-scores2[np.arange(len(y)), y] + np.log(escores2sum))
    grad_score = (-y + escores2[:, 1] / escores2sum) / len(scores)
    grad_score = np.reshape(grad_score, [len(grad_score), 1])
    return loss, grad_score


class SharedParams(object):
  def __init__(self, reg=0):
    self.reg = reg

    
class Layer(object):
  """ 
  Neural network layer: stores parameteres and the cache in the object
  """
  __metaclass__ = abc.ABCMeta
  @abc.abstractmethod
  def forward(self, X, params):
    """ Forward newtork pass: returns a modified X value."""
    pass
  @abc.abstractmethod
  def regularization_loss(self):
    return 0
  @abc.abstractmethod
  def backward(self, dout, grads):
    """ Network backward pass: accepts upstream gradient,
        returns downstream data gradient, and fills in downstream
        parameter gradients. """
    pass


class AffineLayer(Layer):
  def __init__(self, param_suffix, shared_params, fan_in, fan_out, dtype=np.float32):
    """ Using Xavier initialization by default """
    self.param_suffix = param_suffix
    self.shared_params = shared_params
    self.W = np.random.randn(fan_in, fan_out).astype(dtype) / np.sqrt(fan_in)
    self.b = np.zeros(fan_out, dtype=dtype)
    self.cache = None

  def forward(self, X, params):
    out, self.cache = affine_forward(X, self.W, self.b)
    params['W' + self.param_suffix] = self.W
    params['b' + self.param_suffix] = self.b
    return out
    
  def regularization_loss(self):
    return 0.5 * self.shared_params.reg * np.sum(self.W ** 2)
   
  def backward(self, dout, grads):
    dx, dw, db = affine_backward(dout, self.cache)
    dw += self.shared_params.reg * self.W
    grads['W' + self.param_suffix] = dw
    grads['b' + self.param_suffix] = db
    return dx
 
  def __repr__(self):
    return "Affine(%s)[%d, %d]:%s" % (
        (self.param_suffix,) + self.W.shape + (self.W.dtype,))


class ReluLayer(Layer):
  def __init__(self):
    self.cache = None
    
  def forward(self, X, params):
    out, self.cache = relu_forward(X)
    return out

  def backward(self, dout, grads):
    return relu_backward(dout, self.cache)

  def __repr__(self):
    return "Relu"
