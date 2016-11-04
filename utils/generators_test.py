from operator import mul

import unittest

from utils.generators import OrderedProduct, PeekIterable, GeneratorsList

class Item:
  """
  This is a dummy class that contains a score and some arbitrary data,
  and that only serves tests in this file.
  """
  def __init__(self, score, data):
    self.score = score
    self.data = data

  def __eq__(self, other):
    return isinstance(other, self.__class__) and \
           self.score == other.score and \
           self.data == other.data

  def __ne__(self, other):
    return not self.__eq__(other)

  def __repr__(self):
    return "({0}, {1})".format(self.score, self.data)

class GeneratorsListTestCase(unittest.TestCase):
  """
  Test whether we are capable of generating items from a list of generators
  in descending order value (where value can be extracted using a user
  defined function).
  """
  def setUp(self):
    iter1 = iter([Item(.95, '1a'),
                  Item(.85, '1b'),
                  Item(.84, '1c'),
                  Item(.82, '1d'),
                  Item(.6,  '1e')])
    iter2 = iter([Item(.8,  '2a'),
                  Item(.75, '2b'),
                  Item(.72, '2c'),
                  Item(.65, '2d')])
    iter3 = iter([Item(.9,  '3a'),
                  Item(.7,  '3b'),
                  Item(.3,  '3c')])
    self.iters = [iter1, iter2, iter3]
    self.order_function = lambda i: i.score

  def test_SequenceCorrect(self):
    expected_outputs = \
      [Item(.95, '1a'),
       Item(.9,  '3a'),
       Item(.85, '1b'),
       Item(.84, '1c'),
       Item(.82, '1d'),
       Item(.8,  '2a'),
       Item(.75, '2b'),
       Item(.72, '2c'),
       Item(.7,  '3b'),
       Item(.65, '2d'),
       Item(.6,  '1e'),
       Item(.3,  '3c')]
    gen_list = GeneratorsList(self.iters, self.order_function)
    sorted_items = gen_list.items()
    for i, expected_output in enumerate(expected_outputs):
      generated_output = sorted_items.next()
      self.assertEqual(expected_output, generated_output)

class PeekIterableTestCase(unittest.TestCase):
  def setUp(self):
    self.iterable = iter(xrange(10))

  def test_PeekFirstXrange(self):
    head, iterable = PeekIterable(self.iterable)
    self.assertEqual(head, 0)
    self.assertEqual(list(iterable), list(xrange(10)))

  def test_PeekSecondXrange(self):
    self.iterable.next()
    head, iterable = PeekIterable(self.iterable)
    self.assertEqual(head, 1)
    self.assertEqual(list(iterable), list(xrange(1, 10)))

class OrderedProductTestCase(unittest.TestCase):
  def setUp(self):
    list1 = [Item(.9,  '1a'),
             Item(.85, '1b'),
             Item(.84, '1c'),
             Item(.8,  '1d'),
             Item(.6,  '1e')]
    list2 = [Item(.8,  '2a'),
             Item(.75, '2b'),
             Item(.7,  '2c'),
             Item(.65, '2d')]
    list3 = [Item(.9,  '3a'),
             Item(.7,  '3b'),
             Item(.3,  '3c'),
             Item(.2,  '3d')]
    iter1 = iter(list1)
    iter2 = iter(list2)
    iter3 = iter(list3)
    self.iters = [iter1, iter2, iter3]
    self.lists = [list1, list2, list3]
    self.order_function = lambda ii: reduce(mul, [i.score for i in ii])

  def test_FirstItemsCorrect(self):
    expected_outputs = [(Item(.9,  '1a'), Item(.8,  '2a'), Item(.9,  '3a')),
                        (Item(.85, '1b'), Item(.8,  '2a'), Item(.9,  '3a')),
                        (Item(.9,  '1a'), Item(.75, '2b'), Item(.9,  '3a')),
                        (Item(.84, '1c'), Item(.8,  '2a'), Item(.9,  '3a')),
                        (Item(.8,  '1d'), Item(.8,  '2a'), Item(.9,  '3a'))]
    ordered_product = OrderedProduct(*self.iters, key=self.order_function)
    for i, expected_output in enumerate(expected_outputs):
      generated_output = ordered_product.next()
      self.assertEqual(expected_output, generated_output)

  def test_OrderedOutput(self):
    ordered_product = OrderedProduct(*self.iters, key=self.order_function)
    previous_score = 1.0
    for i, gen_output in enumerate(ordered_product):
      gen_score = self.order_function(gen_output)
      self.assertLessEqual(gen_score, previous_score)
      previous_score = gen_score
    self.assertEqual(79, i)

  def test_FromRealData(self):
    item_lists = \
      [[Item(0.01530562004040494, '1a'),
        Item(0.0036904715261238613, '1b')],
       [Item(4.3130060393107921e-06, '2a'),
        Item(4.1338497316655474e-07, '2b')],
       [Item(0.00085005736486004797, '3a'),
        Item(0.00014387175687900362, '3b'),
        Item(2.3273372435343642e-05, '3c')],
       [Item(2.9707487246695427e-07, '4a'),
        Item(2.3453279405328996e-07, '4b')],
       [Item(0.01969830710754087, '5a'),
        Item(0.0034698766961831083, '5b')],
       [Item(7.6075602223108301e-05, '6a'),
        Item(3.8037801111495495e-05, '6b')],
       [Item(0.0025729986602252594, '7a'),
        Item(0.00085276574551284982, '7b')],
       [Item(0.0025554617060364491, '8a'),
        Item(0.00057056701667213884, '8b')]]
    iters = [iter(l) for l in item_lists]
    ordered_product = OrderedProduct(*iters, key=self.order_function)
    previous_score = 1.0
    head_item = Item(1.5260267133099999e-05, '9a')
    order_func = lambda p, pp: p.score * reduce(mul, [t.score for t in pp])
    for i, gen_output in enumerate(ordered_product):
      gen_score = order_func(head_item, gen_output)
      self.assertLessEqual(gen_score, previous_score)
      previous_score = gen_score
    self.assertEqual(383, i)

  def test_InOrderLM(self):
    lm_order_func = lambda score, state: score * 1.0
    expected_outputs = [(Item(.9,  '1a'), Item(.8,  '2a'), Item(.9,  '3a')),
                        (Item(.85, '1b'), Item(.8,  '2a'), Item(.9,  '3a')),
                        (Item(.9,  '1a'), Item(.75, '2b'), Item(.9,  '3a')),
                        (Item(.84, '1c'), Item(.8,  '2a'), Item(.9,  '3a')),
                        (Item(.8,  '1d'), Item(.8,  '2a'), Item(.9,  '3a'))]
    ordered_product = OrderedProduct(*self.iters, key=self.order_function, lm=lm_order_func)
    for i, expected_output in enumerate(expected_outputs):
      generated_output = ordered_product.next()
      self.assertEqual(expected_output, generated_output)

  def test_OutOfOrderLM(self):
    # We use a non-linear function to make items appear out-of-order.
    lm_order_func = lambda score, state: score * state.items[0].score ** 2
    ordered_product = OrderedProduct(*self.iters, key=self.order_function, lm=lm_order_func)
    ordered_product = list(ordered_product)
    import itertools
    ordered_product_expected = list(itertools.product(*self.lists))
    ordered_product_expected = sorted(ordered_product_expected,
      key=lambda ii: reduce(mul, [i.score for i in ii]) * ii[0].score ** 2,
      reverse=True)
    expected_len = len(ordered_product_expected)
    obtained_len = len(ordered_product)
    self.assertEqual(expected_len, obtained_len)
    for i in range(expected_len):
      expected_item = ordered_product_expected[i]
      obtained_item = ordered_product[i]
      self.assertEqual(expected_item, obtained_item,
        msg='At position {0}: expected {1}, got {2}'.format(
        i, expected_item, obtained_item))

  def test_OutOfOrderLMWithBound(self):
    # We use a non-linear function to make items appear out-of-order.
    lm_order_func = lambda score, state: score * state.items[-1].score ** 2
    expected_outputs = [(Item(.9,  '1a'), Item(.8,  '2a'), Item(.9,  '3a')),
                        (Item(.85, '1b'), Item(.8,  '2a'), Item(.9,  '3a')),
                        (Item(.9,  '1a'), Item(.75, '2b'), Item(.9,  '3a')),
                        (Item(.84, '1c'), Item(.8,  '2a'), Item(.9,  '3a')),
                        (Item(.8,  '1d'), Item(.8,  '2a'), Item(.9,  '3a'))]
    ordered_product = \
      OrderedProduct(*self.iters, key=self.order_function, lm=lm_order_func, bound=0.8)
    for i, expected_output in enumerate(expected_outputs):
      generated_output = ordered_product.next()
      self.assertEqual(expected_output, generated_output)

  def test_OutOfOrderComplexLM(self):
    # We use a non-linear function to demote all third items with data == '3a'.
    def lm_order_func(score, state):
      if state.items[-1].data == '3a':
        return 0.0
      else:
        return score
    expected_outputs = [(Item(.9,  '1a'), Item(.8,  '2a'), Item(.7,  '3b')),
                        (Item(.85, '1b'), Item(.8,  '2a'), Item(.7,  '3b')),
                        (Item(.9,  '1a'), Item(.75, '2b'), Item(.7,  '3b')),
                        (Item(.84, '1c'), Item(.8,  '2a'), Item(.7,  '3b')),
                        (Item(.8,  '1d'), Item(.8,  '2a'), Item(.7,  '3b'))]
    ordered_product = OrderedProduct(*self.iters, key=self.order_function, lm=lm_order_func)
    for i, expected_output in enumerate(expected_outputs):
      generated_output = ordered_product.next()
      self.assertEqual(expected_output, generated_output)

if __name__ == '__main__':
  suite1  = unittest.TestLoader().loadTestsFromTestCase(OrderedProductTestCase)
  suite2  = unittest.TestLoader().loadTestsFromTestCase(PeekIterableTestCase)
  suite3  = unittest.TestLoader().loadTestsFromTestCase(GeneratorsListTestCase)
  suites  = unittest.TestSuite([suite1,
                                suite2, suite3])
  unittest.TextTestRunner(verbosity=2).run(suites)

