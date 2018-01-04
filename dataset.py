import numpy as np
import collections

NORM_EPSILON = 1e-9
NormValues = collections.namedtuple("NormValues", ['mean', 'div'])

class ProstatDS(object):
  def __init__(self, filename=None):
    self.X = None
    self.y = None
    self.norm_values = None
    self.header = None
    self.row_ids = None
    if filename:
      with open(filename) as fh:
        header_line = fh.readline()
        self.header = tuple(header_line.split())
        self.row_ids = []
        X = []
        y = []
        for line in fh:
          items = line.split()
          if len(items) != len(self.header):
            raise RuntimeError("Invalid line: expected %d elements, but got %d\n%s" % 
                               (len(self.header), len(items), line))
          self.row_ids.append(items[0])
          y.append(int(items[-1]))
          X.append([float(v) for v in items[1:-1]])
        # Converting into NumPy arrays:
        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)
            
  @property
  def header_X(self):
    return self.header[1:-1]

  def normalize(self, norm_values = None):
    if self.norm_values:
      raise RuntimeError("Already normalized")
    if not norm_values:
      mean = np.mean(self.X, axis=0)
      div = np.sqrt(np.var(self.X, axis=0) + NORM_EPSILON)
      norm_values = NormValues(mean, div)
    self.norm_values = norm_values
    self.X -= self.norm_values.mean
    self.X /= self.norm_values.div
    return self.norm_values

  def serialize(self, filename):
    y = self.y
    if self.norm_values:
      X = (self.X * self.norm_values.div) + self.norm_values.mean
    else:
      X = self.X
    with open(filename, "w") as fh:
      fh.write("\t".join(self.header))
      fh.write("\n")
      for i in range(len(X)):
        vals1 = [("%.3f" % v).rstrip("0").rstrip(".") for v in X[i, :]]
        vals2 = [v.lstrip("0") if v.find(".") > 0 else v for v in vals1]
        xvals = ([self.row_ids[i]] + vals2 + [str(int(y[i]))])
        fh.write("\t".join(xvals))
        fh.write("\n")


def _random_split_indices(N, fractions):
  """ Given an array size, and the fractions, returns a list
  of list of random indices that split an array N in the random
  fractions with the given sizes
  """
  if type(fractions) is not list:
    fractions = [fractions]
  lengths = [round(v * N) for v in fractions]
  lengths_sum = np.sum(lengths)
  if lengths_sum >= N:
    raise ValueError("The sum of sizes it too large!")
  lengths.append(N - lengths_sum)

  # Split the list of indices into portions of the given lengths:
  all_indices = np.random.permutation(N)
  part_indices = []
  start = 0
  for len in lengths:
    cur_idx = all_indices[start:start + len]
    cur_idx.sort()
    part_indices.append(cur_idx)
    start += len
  return part_indices


def dataset_split(in_set, fractions):
  N = in_set.X.shape[0]
  part_indices = _random_split_indices(N, fractions)
  if in_set.norm_values:
    raise ValueError("Must split before normalizing!")

  out = []
  for idx in part_indices:
    ds = ProstatDS()
    ds.X = in_set.X[idx, :]
    ds.y = in_set.y[idx]
    ds.header = in_set.header
    ds.row_ids = [in_set.row_ids[i] for i in idx]
    out.append(ds)
  return out
