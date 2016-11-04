class SemiRing(object):
    """
    A semiring operation.

    Implements + and *

    """
    def __add__(self, other):
        raise NotImplementedError()

    def __mul__(self, other):
        raise NotImplementedError()

    @classmethod
    def one(cls):
        raise NotImplementedError()

    @classmethod
    def zero(cls):
        raise NotImplementedError()

    @classmethod
    def make(cls, v):
        raise NotImplementedError()
