import unittest
# from pudb import set_trace; set_trace()

from linguistics.similarity import Similarity
from linguistics.similarity_costs_deps import *
from utils.tree_tools import Tree, TreePattern, tree_or_string

class LexicalSimilarityDepsTestCase(unittest.TestCase):
  def setUp(self):
    self.similarity_scorer = LexicalSimilarityDeps()
    self.kScore = self.similarity_scorer.kScore

  def test_TerminalToTerminalDifferent(self):
    tree1 = tree_or_string('the')
    tree2 = tree_or_string('la')
    path1 = ()
    path2 = ()
    subpaths1 = []
    subpaths2 = []
    tree_pattern1 = TreePattern(tree1, path1, subpaths1)
    tree_pattern2 = TreePattern(tree2, path2, subpaths2)
    similarities = \
      self.similarity_scorer.GetSimilarity(tree_pattern1, tree_pattern2)
    self.assertEqual(0, len(similarities))

  def test_TerminalToTerminalEqual(self):
    tree1 = tree_or_string('the')
    tree2 = tree_or_string('the')
    path1 = ()
    path2 = ()
    subpaths1 = []
    subpaths2 = []
    tree_pattern1 = TreePattern(tree1, path1, subpaths1)
    tree_pattern2 = TreePattern(tree2, path2, subpaths2)
    similarities = \
      self.similarity_scorer.GetSimilarity(tree_pattern1, tree_pattern2)
    self.assertEqual(1, len(similarities))
    expected_similarities = \
      [Similarity(self.kScore, 'copy', tree_pattern1, tree_pattern2)]
    self.assertListEqual(expected_similarities, similarities)

  def test_TerminalToTerminalSimilar(self):
    tree1 = tree_or_string('italian')
    tree2 = tree_or_string('european')
    path1 = ()
    path2 = ()
    subpaths1 = []
    subpaths2 = []
    tree_pattern1 = TreePattern(tree1, path1, subpaths1)
    tree_pattern2 = TreePattern(tree2, path2, subpaths2)
    similarities = \
      self.similarity_scorer.GetSimilarity(tree_pattern1, tree_pattern2)
    self.assertEqual(1, len(similarities))
    expected_similarities = \
      [Similarity(self.kScore, 'hypernym', tree_pattern1, tree_pattern2)]
    self.assertListEqual(expected_similarities, similarities)

  def test_NodeEqual(self):
    tree1 = tree_or_string('(is (italian the) smart)')
    tree2 = tree_or_string('(italian smart)')
    path1 = (0,)
    path2 = ()
    subpaths1 = [(0,0)]
    subpaths2 = [(0,)]
    tree_pattern1 = TreePattern(tree1, path1, subpaths1)
    tree_pattern2 = TreePattern(tree2, path2, subpaths2)
    similarities = \
      self.similarity_scorer.GetSimilarity(tree_pattern1, tree_pattern2)
    self.assertEqual(1, len(similarities))
    expected_similarities = \
      [Similarity(self.kScore, 'copy', tree_pattern1, tree_pattern2)]
    self.assertListEqual(expected_similarities, similarities)

  def test_NodeSimilar(self):
    tree1 = tree_or_string('(is (italian the) smart)')
    tree2 = tree_or_string('(european smart)')
    path1 = (0,)
    path2 = ()
    subpaths1 = [(0,0)]
    subpaths2 = [(0,)]
    tree_pattern1 = TreePattern(tree1, path1, subpaths1)
    tree_pattern2 = TreePattern(tree2, path2, subpaths2)
    similarities = \
      self.similarity_scorer.GetSimilarity(tree_pattern1, tree_pattern2)
    self.assertEqual(1, len(similarities))
    expected_similarities = \
      [Similarity(self.kScore, 'hypernym', tree_pattern1, tree_pattern2)]
    self.assertListEqual(expected_similarities, similarities)

  def test_NodeSimilar(self):
    tree1 = tree_or_string('(is (italian the) smart)')
    tree2 = tree_or_string('(european smart)')
    path1 = (0,)
    path2 = ()
    subpaths1 = [(0,0)]
    subpaths2 = [(0,)]
    tree_pattern1 = TreePattern(tree1, path1, subpaths1)
    tree_pattern2 = TreePattern(tree2, path2, subpaths2)
    similarities = \
      self.similarity_scorer.GetSimilarity(tree_pattern1, tree_pattern2)
    self.assertEqual(1, len(similarities))
    expected_similarities = \
      [Similarity(self.kScore, 'hypernym', tree_pattern1, tree_pattern2)]
    self.assertListEqual(expected_similarities, similarities)

  def test_NodeDifferent(self):
    tree1 = tree_or_string('(is (italian the) smart)')
    tree2 = tree_or_string('(french smart)')
    path1 = (0,)
    path2 = ()
    subpaths1 = [(0,0)]
    subpaths2 = [(0,)]
    tree_pattern1 = TreePattern(tree1, path1, subpaths1)
    tree_pattern2 = TreePattern(tree2, path2, subpaths2)
    similarities = \
      self.similarity_scorer.GetSimilarity(tree_pattern1, tree_pattern2)
    self.assertEqual(0, len(similarities))

  def test_NodeToLeafEqual(self):
    tree1 = tree_or_string('(is (italian the) smart)')
    tree2 = tree_or_string('(french smart)')
    path1 = (1,)
    path2 = (0,)
    subpaths1 = []
    subpaths2 = []
    tree_pattern1 = TreePattern(tree1, path1, subpaths1)
    tree_pattern2 = TreePattern(tree2, path2, subpaths2)
    similarities = \
      self.similarity_scorer.GetSimilarity(tree_pattern1, tree_pattern2)
    self.assertEqual(1, len(similarities))
    expected_similarities = \
      [Similarity(self.kScore, 'copy', tree_pattern1, tree_pattern2)]
    self.assertListEqual(expected_similarities, similarities)

  def test_NodeToLeafDifferent(self):
    tree1 = tree_or_string('(is (italian the) smart)')
    tree2 = tree_or_string('(french proud)')
    path1 = (1,)
    path2 = (0,)
    subpaths1 = []
    subpaths2 = []
    tree_pattern1 = TreePattern(tree1, path1, subpaths1)
    tree_pattern2 = TreePattern(tree2, path2, subpaths2)
    similarities = \
      self.similarity_scorer.GetSimilarity(tree_pattern1, tree_pattern2)
    self.assertEqual(0, len(similarities))

  def test_NodeToLeafSimilar(self):
    tree1 = tree_or_string('(is (italian the) smart)')
    tree2 = tree_or_string('(french bright)')
    path1 = (1,)
    path2 = (0,)
    subpaths1 = []
    subpaths2 = []
    tree_pattern1 = TreePattern(tree1, path1, subpaths1)
    tree_pattern2 = TreePattern(tree2, path2, subpaths2)
    similarities = \
      self.similarity_scorer.GetSimilarity(tree_pattern1, tree_pattern2)
    self.assertEqual(1, len(similarities))
    expected_similarities = \
      [Similarity(self.kScore, 'synonym', tree_pattern1, tree_pattern2)]
    self.assertListEqual(expected_similarities, similarities)

if __name__ == '__main__':
  suite1  = unittest.TestLoader().loadTestsFromTestCase(LexicalSimilarityDepsTestCase)
  suites  = unittest.TestSuite([suite1])
  unittest.TextTestRunner(verbosity=2).run(suites)

