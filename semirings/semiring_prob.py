import numpy as np

# from pudb import set_trace; set_trace() 

from semirings.semiring import SemiRing
import warnings
warnings.simplefilter("error", RuntimeWarning)

# TODO: Account for reversed division, and other reversed operations.

class ProbSemiRing(SemiRing):
  def __init__(self, prob = None, log = None):
    assert prob is not None or log is not None, 'Specify the probability or its log'
    if prob == 0.0:
      self.v = - np.inf
    else:
      self.v = log if prob is None else np.log(prob)

  def __float__(self):
    return np.exp(self.v)

  def __repr__(self):
    return str(np.exp(self.v))

  def __str__(self):
    return repr(self)

  def __add__(self, other):
    if self.v == - np.inf or other.v == - np.inf:
      return ProbSemiRing(log=max(self.v, other.v))
    return ProbSemiRing(log=np.logaddexp(self.v, other.v))

  def __mul__(self, other):
    if not isinstance(other, self.__class__):
      other_obj = ProbSemiRing(prob=other)
    else:
      other_obj = other
    return ProbSemiRing(log=(self.v + other_obj.v))

  def __rmul__(self, other):
    if not isinstance(other, self.__class__):
      other_obj = ProbSemiRing(prob=other)
    else:
      other_obj = other
    return ProbSemiRing(log=(self.v + other_obj.v))

  def __div__(self, other):
    if not isinstance(other, self.__class__):
      other_obj = ProbSemiRing(prob=other)
    else:
      other_obj = other
    return ProbSemiRing(log=(self.v - other_obj.v))

  def __gt__(self, other):
    if not isinstance(other, self.__class__):
      other_obj = ProbSemiRing(prob=other)
    else:
      other_obj = other
    return self.v > other_obj.v

  def __ge__(self, other):
    if not isinstance(other, self.__class__):
      other_obj = ProbSemiRing(prob=other)
    else:
      other_obj = other
    return self.v >= other_obj.v

  def __sub__(self, other):
    if not isinstance(other, self.__class__):
      other_obj = ProbSemiRing(prob=other)
    else:
      other_obj = other
    if self.v == other_obj.v:
      return ProbSemiRing(log=(-np.inf)) # Probability = 0.0
    return ProbSemiRing(log=np.log(np.exp(self.v) - np.exp(other_obj.v)))

  def __abs__(self):
    return abs(self.v)

  def __eq__(self, other):
    if not isinstance(other, self.__class__):
      other_value = np.log(other) if other != 0.0 else - np.inf
    else:
      other_value = other.v
    if other_value == self.v:
      return True
    return round(abs(np.exp(other_value) - np.exp(self.v)), 7) == 0.0

  def __neq__(self, other):
    return not self.__eq__(other)

  def is_zero(self):
    return self.v == - np.inf
