import numpy as np

def rel_error(exp, actual):
  return (np.mean(np.abs(exp - actual) / 
          np.maximum(1e-8, np.abs(exp))))

def eval_numeric_gradient(f, w, h=1e-6):
  dw = np.zeros_like(w)
  y0 = f()
  it = np.nditer(w, flags=['multi_index'], op_flags=['readwrite'])
  for _ in it:
    idx = it.multi_index
    prev = w[idx]
    delta = max(abs(prev), 1e-3) * h
    w[idx] = prev + delta
    dw[idx] = (f() - y0) / delta
    w[idx] = prev
  return dw