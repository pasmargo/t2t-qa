import unittest
import numpy as np
from operator import mul
# from pudb import set_trace; set_trace()

from semirings.semiring_prob import ProbSemiRing

class ProbSemiRingTestCase(unittest.TestCase):
  """
  Test whether the conversion back and forth from the probability domain to the
  logarithm domain is consistent and works as expected when used with other
  functions from mathematical packages.
  """
  def setUp(self):
    self.prob_big = ProbSemiRing(0.999999)
    self.prob_small = ProbSemiRing(0.00001)
    self.a = ProbSemiRing(0.2)
    self.b = ProbSemiRing(0.3)
    self.c = ProbSemiRing(0.4)
    self.zero = ProbSemiRing(0.0)
    self.one = ProbSemiRing(1.0)

  def test_Multiplication(self):
    result = self.a * self.b
    expected_result = ProbSemiRing(0.06)
    self.assertAlmostEqual(expected_result, result)

  def test_MultiplicationWithFloatRight(self):
    result = self.a * 1.0
    expected_result = ProbSemiRing(0.2)
    self.assertAlmostEqual(expected_result, result)

  def test_MultiplicationWithFloatLeft(self):
    result = 1.0 * self.a
    expected_result = ProbSemiRing(0.2)
    self.assertAlmostEqual(expected_result, result)

  def test_Division(self):
    result = self.a / self.b
    expected_result = ProbSemiRing(0.666666666)
    self.assertAlmostEqual(expected_result, result)

  def test_DivisionWithFloat(self):
    result = self.b / 2.0
    expected_result = ProbSemiRing(0.15)
    self.assertAlmostEqual(expected_result, result)

  def test_ComparisonGT(self):
    self.assertTrue(self.b > self.a)

  def test_ComparisonGE(self):
    self.assertTrue(self.b >= self.a)

  def test_ComparisonGE2(self):
    self.assertTrue(self.b >= self.b)

  def test_ComparisonGTFloat(self):
    self.assertTrue(self.b > 0.2)

  def test_ComparisonGEFloat(self):
    self.assertTrue(self.b >= 0.2)

  def test_ComparisonGE2Float(self):
    self.assertTrue(self.b >= 0.3)

  def test_Addition(self):
    result = self.a + self.b
    expected_result = ProbSemiRing(0.5)
    self.assertAlmostEqual(expected_result, result)

  def test_AdditionZeroZero(self):
    result = self.zero + self.zero
    expected_result = self.zero
    self.assertAlmostEqual(expected_result, result)

  def test_Sorted(self):
    array = [self.a, self.c, self.b]
    sorted_array = sorted(array)
    expected_array = [self.a, self.b, self.c]
    self.assertAlmostEqual(expected_array, sorted_array)

  def test_Max(self):
    array = [self.a, self.c, self.b]
    maximum = max(array)
    expected_maximum = self.c
    self.assertAlmostEqual(expected_maximum, maximum)

  def test_NumpyMax(self):
    array = np.array([self.a, self.c, self.b])
    maximum = array.max()
    expected_maximum = self.c
    self.assertAlmostEqual(expected_maximum, maximum)

  def test_NumpyMin(self):
    array = np.array([self.a, self.c, self.b])
    maximum = array.min()
    expected_maximum = self.a
    self.assertAlmostEqual(expected_maximum, maximum)

  def test_NumpyArgMax(self):
    array = np.array([self.a, self.c, self.b])
    maximum = array.argmax()
    expected_maximum = 1
    self.assertAlmostEqual(expected_maximum, maximum)

  def test_ReduceMult(self):
    array = [self.a, self.c, self.b]
    maximum = reduce(mul, array)
    expected_maximum = self.a * self.b * self.c
    self.assertAlmostEqual(expected_maximum, maximum)

  def test_CommutativeMonoid1(self):
    result = (self.a + self.b) + self.c
    expected_result = self.a + (self.b + self.c)
    self.assertAlmostEqual(expected_result, result)

  def test_CommutativeMonoid2Zero(self):
    result = self.a + self.zero
    expected_result = self.zero + self.a
    self.assertAlmostEqual(expected_result, result)

  def test_CommutativeMonoid3(self):
    result = self.a + self.b
    expected_result = self.b + self.a
    self.assertAlmostEqual(expected_result, result)

  def test_MonoidIdentity1(self):
    result = (self.a * self.b) * self.c
    expected_result = self.a * (self.b * self.c)
    self.assertAlmostEqual(expected_result, result)

  def test_MonoidIdentity2(self):
    result = self.one * self.a 
    expected_result = self.a * self.one
    self.assertAlmostEqual(expected_result, result)

  def test_MultiplicationDistributes1(self):
    result = self.a * (self.b + self.c)
    expected_result = (self.a * self.b) + (self.a * self.c)
    self.assertAlmostEqual(expected_result, result)

  def test_MultiplicationDistributes2(self):
    result = (self.a + self.b) * self.c
    expected_result = (self.a * self.c) + (self.b * self.c)
    self.assertAlmostEqual(expected_result, result)

  def test_MultiplicationZeroAnihilatesA(self):
    result = self.zero * self.a
    expected_result = self.zero
    self.assertAlmostEqual(expected_result, result)

    result = self.a * self.zero
    expected_result = self.zero
    self.assertAlmostEqual(expected_result, result)

  def test_MultiplicationZeroAnihilatesBig(self):
    result = self.zero * self.prob_big
    expected_result = self.zero
    self.assertAlmostEqual(expected_result, result)

    result = self.prob_big * self.zero
    expected_result = self.zero
    self.assertAlmostEqual(expected_result, result)

  def test_MultiplicationZeroAnihilatesSmall(self):
    result = self.zero * self.prob_small
    expected_result = self.zero
    self.assertAlmostEqual(expected_result, result)

    result = self.prob_small * self.zero
    expected_result = self.zero
    self.assertAlmostEqual(expected_result, result)

  def test_MultiplicationZeroAnihilatesZero(self):
    result = self.zero * self.zero
    expected_result = self.zero
    self.assertAlmostEqual(expected_result, result)

    result = self.zero * self.zero
    expected_result = self.zero
    self.assertAlmostEqual(expected_result, result)

  def test_MultiplicationZeroAnihilatesOne(self):
    result = self.zero * self.one
    expected_result = self.zero
    self.assertAlmostEqual(expected_result, result)

    result = self.one * self.zero
    expected_result = self.zero
    self.assertAlmostEqual(expected_result, result)

  def test_FloatCast(self):
    result = float(self.one)
    self.assertTrue(isinstance(result, float))
    self.assertEqual(1.0, result)

if __name__ == '__main__':
  suite1  = unittest.TestLoader().loadTestsFromTestCase(ProbSemiRingTestCase)
  suites  = unittest.TestSuite([suite1])
  unittest.TextTestRunner(verbosity=2).run(suites)

