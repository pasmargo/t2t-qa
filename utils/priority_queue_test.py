import unittest

# from pudb import set_trace; set_trace()

from extraction.extractor_beam import Transformation
from linguistics.similarity import Similarity
from utils.priority_queue import PriorityQueue

class PriorityQueueTestCase(unittest.TestCase):
  def setUp(self):
    self.similarity = Similarity(1.0, 'dummy', None, None)

    t0_src_path = (0,)
    t0_trg_path = (10,)
    t0_src_subpaths = ((0, 0), (0, 1))
    t0_trg_subpaths = ((10, 0), (10, 1))
    self.t0 = Transformation(t0_src_path, t0_trg_path,
                             t0_src_subpaths, t0_trg_subpaths, self.similarity)

    t1_src_path = (1,)
    t1_trg_path = (11,)
    t1_src_subpaths = ((1, 0), (1, 1))
    t1_trg_subpaths = ((11, 0), (11, 1))
    self.t1 = Transformation(t1_src_path, t1_trg_path,
                             t1_src_subpaths, t1_trg_subpaths, self.similarity)

    t2_src_path = (2,)
    t2_trg_path = (12,)
    t2_src_subpaths = ((2, 0), (2, 1))
    t2_trg_subpaths = ((12, 0), (12, 1))
    self.t2 = Transformation(t2_src_path, t2_trg_path,
                             t2_src_subpaths, t2_trg_subpaths, self.similarity)

    t0bis_src_path = (0,)
    t0bis_trg_path = (10,)
    t0bis_src_subpaths = ((3, 0), (3, 1))
    t0bis_trg_subpaths = ((13, 0), (13, 1))
    self.t0bis = Transformation(t0bis_src_path, t0bis_trg_path,
                                t0bis_src_subpaths, t0bis_trg_subpaths, self.similarity)

    self.q_costs = PriorityQueue(2)
    self.q_probs = PriorityQueue(2, reverse=True)

  def test_RegularInsertions(self):
    self.q_costs.Push(0.3, self.t0)
    self.q_costs.Push(0.2, self.t1)
    self.assertEqual(2, len(self.q_costs.queue))
    self.assertIn(self.t0, self.q_costs.GetItems())
    self.assertIn(self.t1, self.q_costs.GetItems())

    self.q_probs.Push(0.3, self.t0)
    self.q_probs.Push(0.2, self.t1)
    self.assertEqual(2, len(self.q_probs.queue))
    self.assertIn(self.t0, self.q_probs.GetItems())
    self.assertIn(self.t1, self.q_probs.GetItems())

  def test_InsertionsOverflow(self):
    self.q_costs.Push(0.2, self.t0)
    self.q_costs.Push(0.3, self.t1)
    self.q_costs.Push(0.1, self.t2)
    self.assertEqual(2, len(self.q_costs.queue))
    self.assertIn(self.t0, self.q_costs.GetItems())
    self.assertIn(self.t2, self.q_costs.GetItems())
    self.assertNotIn(self.t1, self.q_costs.GetItems())

    self.q_probs.Push(0.2, self.t0)
    self.q_probs.Push(0.3, self.t1)
    self.q_probs.Push(0.1, self.t2)
    self.assertEqual(2, len(self.q_probs.queue))
    self.assertIn(self.t0, self.q_probs.GetItems())
    self.assertIn(self.t1, self.q_probs.GetItems())
    self.assertNotIn(self.t2, self.q_probs.GetItems())

  def test_InsertionsSameElementOverflow(self):
    self.q_costs.Push(0.1, self.t0)
    self.q_costs.Push(0.2, self.t1)
    self.q_costs.Push(0.1, self.t0)
    self.assertEqual(2, len(self.q_costs.queue))
    self.assertIn(self.t0, self.q_costs.GetItems())
    self.assertIn(self.t1, self.q_costs.GetItems())

    self.q_probs.Push(0.1, self.t0)
    self.q_probs.Push(0.2, self.t1)
    self.q_probs.Push(0.1, self.t0)
    self.assertEqual(2, len(self.q_probs.queue))
    self.assertIn(self.t0, self.q_probs.GetItems())
    self.assertIn(self.t1, self.q_probs.GetItems())

    self.q_probs.Push(0.2, self.t0)
    self.q_probs.Push(0.1, self.t1)
    self.q_probs.Push(0.2, self.t0)
    self.assertEqual(2, len(self.q_probs.queue))
    self.assertIn(self.t0, self.q_probs.GetItems())
    self.assertIn(self.t1, self.q_probs.GetItems())

  def test_InsertionsCostReplacement(self):
    self.q_costs.Push(0.3, self.t0)
    self.q_costs.Push(0.2, self.t1)
    self.q_costs.Push(0.1, self.t0)
    self.assertEqual(2, len(self.q_costs.queue))
    self.assertIn(self.t1, self.q_costs.GetItems())
    self.assertIn(self.t0, self.q_costs.GetItems())
    self.assertEqual(0.1, self.q_costs.GetBestScore())
    self.assertEqual(self.t0, self.q_costs.GetBestScoreItem())

    self.q_probs.Push(0.1, self.t0)
    self.q_probs.Push(0.2, self.t1)
    self.q_probs.Push(0.3, self.t0)
    self.assertEqual(2, len(self.q_probs.queue))
    self.assertIn(self.t0, self.q_probs.GetItems())
    self.assertIn(self.t1, self.q_probs.GetItems())
    self.assertEqual(0.3, self.q_probs.GetBestScore())
    self.assertEqual(self.t0, self.q_probs.GetBestScoreItem())

  """
  def test_InsertionsCostReplacementSimilarItem(self):
    self.q_costs.Push(0.3, self.t0)
    self.q_costs.Push(0.2, self.t1)
    self.q_costs.Push(0.1, self.t0bis)
    self.assertEqual(2, len(self.q_costs.queue))
    self.assertIn(self.t1, self.q_costs.GetItems())
    self.assertNotEqual(self.t0.src_subpaths,
                        self.q_costs.GetBestScoreItem().src_subpaths)
    self.assertEqual(self.t0bis.src_subpaths,
                     self.q_costs.GetBestScoreItem().src_subpaths)

    self.q_probs.Push(0.3, self.t0)
    self.q_probs.Push(0.2, self.t1)
    self.q_probs.Push(0.4, self.t0bis)
    self.assertEqual(2, len(self.q_probs.queue))
    self.assertIn(self.t1, self.q_probs.GetItems())
    self.assertNotEqual(self.t0.src_subpaths,
                        self.q_probs.GetBestScoreItem().src_subpaths)
    self.assertEqual(self.t0bis.src_subpaths,
                     self.q_probs.GetBestScoreItem().src_subpaths)
  """

  def test_InsertionsCostNotReplacementSimilarItem(self):
    self.q_costs.Push(0.1, self.t0)
    self.q_costs.Push(0.2, self.t1)
    self.q_costs.Push(0.4, self.t0bis)
    self.assertEqual(2, len(self.q_costs.queue))
    self.assertIn(self.t1, self.q_costs.GetItems())
    self.assertEqual(self.t0.src_subpaths,
                     self.q_costs.GetBestScoreItem().src_subpaths)
    self.assertNotEqual(self.t0bis.src_subpaths,
                        self.q_costs.GetBestScoreItem().src_subpaths)

    self.q_probs.Push(0.2, self.t0)
    self.q_probs.Push(0.1, self.t1)
    self.q_probs.Push(0.0, self.t0bis)
    self.assertEqual(2, len(self.q_probs.queue))
    self.assertIn(self.t1, self.q_probs.GetItems())
    self.assertEqual(self.t0.src_subpaths,
                     self.q_probs.GetBestScoreItem().src_subpaths)
    self.assertNotEqual(self.t0bis.src_subpaths,
                        self.q_probs.GetBestScoreItem().src_subpaths)

  def test_FilterPreservesHeapStructure(self):
    big_q = PriorityQueue(3)
    big_q.min_threshold = 5

    big_q.Push(5.0, self.t1)
    big_q.Push(0.1, self.t0)
    big_q.Push(0.2, self.t2)

    self.assertEqual(3, len(big_q.queue))
    
    big_q.FilterCache()

    self.assertEqual(2, len(big_q.queue))

    self.assertIn(self.t0, big_q.GetItems())
    self.assertIn(self.t2, big_q.GetItems())
    self.assertEqual(max(big_q.queue, key=lambda x: abs(x[0])), big_q.queue[0])

if __name__ == '__main__':
  suite1  = unittest.TestLoader().loadTestsFromTestCase(PriorityQueueTestCase)
  suites  = unittest.TestSuite([suite1])
  unittest.TextTestRunner(verbosity=2).run(suites)

