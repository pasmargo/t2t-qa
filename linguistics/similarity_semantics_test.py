#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

# from pudb import set_trace; set_trace()

from linguistics.similarity import Similarity
from linguistics.similarity_semantics import (InnerNodesDifference,
  VariableDifference, TreeDifferenceComplexity, EntityDifference,
  TreeSize, VariableDifferenceIndividual, EntityDifferenceIndividual)
from utils.tree_tools import TreePattern, tree_or_string

class TreeSizeTestCase(unittest.TestCase):
  def setUp(self):
    self.tree_size = TreeSize()

  def test_TerminalToTerminal(self):
    src_tree = tree_or_string(u'@Yamamoto')
    trg_tree = tree_or_string(u'@山本')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.tree_size.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 2
    self.assertEqual(expected_cost, cost)

  def test_TerminalToPreterminal(self):
    src_tree = tree_or_string(u'@Yamamoto')
    trg_tree = tree_or_string(u'(N @山本)')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.tree_size.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 5
    self.assertEqual(expected_cost, cost)

  def test_NonterminalToPreterminal(self):
    src_tree = tree_or_string(u'(NP (N Mr.) (N @Yamamoto))')
    trg_tree = tree_or_string(u'(N @山本)')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.tree_size.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 29
    self.assertEqual(expected_cost, cost)

  def test_NonterminalToNonterminal(self):
    src_tree = tree_or_string(u'(NP (N Mr.) (N @Yamamoto))')
    trg_tree = tree_or_string(u'(NP (N @山本) (N san))')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.tree_size.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 50
    self.assertEqual(expected_cost, cost)

class EntityDifferenceTestCase(unittest.TestCase):
  def setUp(self):
    self.entity_difference = EntityDifference()

  def test_2to2(self):
    src_tree = tree_or_string(u'(NP (N @Sugita) (N @Yamamoto))')
    trg_tree = tree_or_string(u'(NP (N @杉田) (N @山本))')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.entity_difference.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 0.0
    self.assertEqual(expected_cost, cost)

  def test_2to1(self):
    src_tree = tree_or_string(u'(NP (N @Sugita) (N @Yamamoto))')
    trg_tree = tree_or_string(u'(N @杉田)')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.entity_difference.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 1.0
    self.assertEqual(expected_cost, cost)

  def test_2to0(self):
    src_tree = tree_or_string(u'(NP (N @Sugita) (N @Yamamoto))')
    trg_tree = tree_or_string(u'です')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.entity_difference.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 2.0
    self.assertEqual(expected_cost, cost)

  def test_1to0(self):
    src_tree = tree_or_string(u'@Sugita')
    trg_tree = tree_or_string(u'です')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.entity_difference.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 1.0
    self.assertEqual(expected_cost, cost)

  def test_0to1(self):
    src_tree = tree_or_string(u'the')
    trg_tree = tree_or_string(u'@杉田')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.entity_difference.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 1.0
    self.assertEqual(expected_cost, cost)

  def test_1to1(self):
    src_tree = tree_or_string(u'@Sugita')
    trg_tree = tree_or_string(u'@杉田')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.entity_difference.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 0.0 # All entities are equal.
    self.assertEqual(expected_cost, cost)

class TreeDifferenceComplexityTestCase(unittest.TestCase):
  def setUp(self):
    self.tree_complexity = TreeDifferenceComplexity()

  def test_0to0(self):
    src_tree = tree_or_string('e-1')
    trg_tree = tree_or_string('e-2')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.tree_complexity.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 0.0
    self.assertEqual(expected_cost, cost)

  def test_0to1(self):
    src_tree = tree_or_string('e-1')
    trg_tree = tree_or_string('(:tense e-2)')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.tree_complexity.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 1.0
    self.assertEqual(expected_cost, cost)

  def test_1to0(self):
    src_tree = tree_or_string('(:index e-1)')
    trg_tree = tree_or_string('e-2')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.tree_complexity.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 1.0
    self.assertEqual(expected_cost, cost)

  def test_1to1(self):
    src_tree = tree_or_string('(:index e-1)')
    trg_tree = tree_or_string('(:tense e-2)')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.tree_complexity.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 2.0
    self.assertEqual(expected_cost, cost)

  def test_2to1(self):
    src_tree = tree_or_string('(:index (:mode e-1))')
    trg_tree = tree_or_string('(:tense e-2)')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.tree_complexity.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 3.0
    self.assertEqual(expected_cost, cost)

  def test_2to2(self):
    src_tree = tree_or_string('(:index (:mode e-1))')
    trg_tree = tree_or_string('(:tense (:mode e-2))')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.tree_complexity.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 4.0
    self.assertEqual(expected_cost, cost)

class VariableDifferenceTestCase(unittest.TestCase):
  def setUp(self):
    self.variable_difference = VariableDifference()

  def test_Equal1to2(self):
    src_tree = tree_or_string('(:index e-1)')
    trg_tree = tree_or_string('(:tense e-2)')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.variable_difference.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 0.0 # All variables are equal.
    self.assertEqual(expected_cost, cost)

  def test_Equal1to20(self):
    src_tree = tree_or_string('(:index e-1)')
    trg_tree = tree_or_string('(:tense e-20)')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.variable_difference.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 0.0 # All variables are equal.
    self.assertEqual(expected_cost, cost)

  def test_Different1to2(self):
    src_tree = tree_or_string('(:tense e-1)')
    trg_tree = tree_or_string('(:index x-2)')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.variable_difference.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 2.0
    self.assertEqual(expected_cost, cost)

  def test_EqualAndDifferent1to2(self):
    src_tree = tree_or_string('(:tense (:index e-1) (:mode x-2))')
    trg_tree = tree_or_string('(:index e-2)')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.variable_difference.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 1.0
    self.assertEqual(expected_cost, cost)

  def test_Equal2to2(self):
    src_tree = tree_or_string('(:tense (:index e-1) (:mode x-2))')
    trg_tree = tree_or_string('(:index (:tense e-2) (:index x-3))')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.variable_difference.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 0.0
    self.assertEqual(expected_cost, cost)

  def test_Different2to0(self):
    src_tree = tree_or_string('(:tense (:index e-1) (:mode x-2))')
    trg_tree = tree_or_string('y-2')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.variable_difference.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 3.0
    self.assertEqual(expected_cost, cost)

class EntityDifferenceIndividualTestCase(unittest.TestCase):
  def setUp(self):
    self.ent_ind = EntityDifferenceIndividual()
    self.ent_ind.kProb = 0.0

  def test_PreterminalEqualEntity(self):
    src_tree = tree_or_string(u'(N @杉田)')
    trg_tree = tree_or_string(u'(N @Sugita)')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.ent_ind.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertEqual(0, len(similarities))

    similarities = self.ent_ind.GetSimilar(src_tree_pat)
    self.assertEqual(0, len(similarities))

  def test_TerminalEqualEntity(self):
    src_tree = tree_or_string(u'(N @杉田)')
    trg_tree = tree_or_string(u'(N @Sugita)')
    src_tree_pat = TreePattern(src_tree, (0,), [])
    trg_tree_pat = TreePattern(trg_tree, (0,), [])
    similarities = self.ent_ind.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertEqual(1, len(similarities))
    expected_similarities = [Similarity(0.0, 'entity_copy', src_tree_pat, trg_tree_pat)]
    self.assertEqual(expected_similarities, similarities)

    similarities = self.ent_ind.GetSimilar(src_tree_pat)
    result_pattern = TreePattern(u'@Sugita', (), [])
    expected_similarities = \
      [Similarity(0.0, 'entity_copy', src_tree_pat, result_pattern)]
    self.assertEqual(expected_similarities, similarities)

  def test_TerminalNoEntity(self):
    src_tree = tree_or_string(u'(N noentity)')
    trg_tree = tree_or_string(u'(N @Sugita)')
    src_tree_pat = TreePattern(src_tree, (0,), [])
    trg_tree_pat = TreePattern(trg_tree, (0,), [])
    similarities = self.ent_ind.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertEqual(0, len(similarities))

    similarities = self.ent_ind.GetSimilar(src_tree_pat)
    self.assertEqual(0, len(similarities))

  def test_TerminalEqualEntity(self):
    src_tree = tree_or_string(u'(N @杉田山本)')
    trg_tree = tree_or_string(u'(N @Sugita_Yamamoto)')
    src_tree_pat = TreePattern(src_tree, (0,), [])
    trg_tree_pat = TreePattern(trg_tree, (0,), [])
    similarities = self.ent_ind.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertEqual(1, len(similarities))
    expected_similarities = [Similarity(0.0, 'entity_copy', src_tree_pat, trg_tree_pat)]
    self.assertEqual(expected_similarities, similarities)

    similarities = self.ent_ind.GetSimilar(src_tree_pat)
    result_pattern = TreePattern(u'@Sugita_Yamamoto', (), [])
    expected_similarities = \
      [Similarity(0.0, 'entity_copy', src_tree_pat, result_pattern)]
    self.assertEqual(expected_similarities, similarities)

class VariableDifferenceIndividualTestCase(unittest.TestCase):
  def setUp(self):
    self.var_ind = VariableDifferenceIndividual()
    self.var_ind.kProb = 0.0

  def test_PreterminalEqualVar(self):
    src_tree = tree_or_string('(:index e-1)')
    trg_tree = tree_or_string('(:tense e-2)')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.var_ind.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertEqual(0, len(similarities))

    similarities = self.var_ind.GetSimilar(src_tree_pat)
    self.assertEqual(0, len(similarities))

  def test_TerminalEqualVar(self):
    src_tree = tree_or_string('(:index e-1)')
    trg_tree = tree_or_string('(:tense e-2)')
    src_tree_pat = TreePattern(src_tree, (0,), [])
    trg_tree_pat = TreePattern(trg_tree, (0,), [])
    similarities = self.var_ind.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertEqual(1, len(similarities))
    expected_similarities = [Similarity(0.0, 'var_copy', src_tree_pat, trg_tree_pat)]
    self.assertEqual(expected_similarities, similarities)

    similarities = self.var_ind.GetSimilar(src_tree_pat)
    result_pattern = TreePattern('e-1', (), [])
    expected_similarities = \
      [Similarity(0.0, 'var_copy', src_tree_pat, result_pattern)]
    self.assertEqual(expected_similarities, similarities)

  def test_TerminalEqualVarUpper(self):
    src_tree = tree_or_string('(:index A-1)')
    trg_tree = tree_or_string('(:tense e-2)')
    src_tree_pat = TreePattern(src_tree, (0,), [])
    trg_tree_pat = TreePattern(trg_tree, (0,), [])
    similarities = self.var_ind.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertEqual(1, len(similarities))
    expected_similarities = [Similarity(0.0, 'var_copy', src_tree_pat, trg_tree_pat)]
    self.assertEqual(expected_similarities, similarities)

    similarities = self.var_ind.GetSimilar(src_tree_pat)
    result_pattern = TreePattern('A-1', (), [])
    expected_similarities = \
      [Similarity(0.0, 'var_copy', src_tree_pat, result_pattern)]
    self.assertEqual(expected_similarities, similarities)

  def test_TerminalNoVar(self):
    src_tree = tree_or_string('(:index novar)')
    trg_tree = tree_or_string('(:tense e-2)')
    src_tree_pat = TreePattern(src_tree, (0,), [])
    trg_tree_pat = TreePattern(trg_tree, (0,), [])
    similarities = self.var_ind.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertEqual(0, len(similarities))

    similarities = self.var_ind.GetSimilar(src_tree_pat)
    self.assertEqual(0, len(similarities))

class InnerNodesDifferenceTestCase(unittest.TestCase):
  def setUp(self):
    self.inner_nodes = InnerNodesDifference()

  def test_Equal1to2(self):
    src_tree = tree_or_string('(:index e-1)')
    trg_tree = tree_or_string('(:index e-2)')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.inner_nodes.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 0.0 # All inner nodes are equal.
    self.assertEqual(expected_cost, cost)

  def test_Different1to2(self):
    src_tree = tree_or_string('(:tense e-1)')
    trg_tree = tree_or_string('(:index e-2)')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.inner_nodes.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 2.0
    self.assertEqual(expected_cost, cost)

  def test_EqualAndDifferent1to2(self):
    src_tree = tree_or_string('(:tense (:index e-1))')
    trg_tree = tree_or_string('(:index e-2)')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.inner_nodes.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 1.0
    self.assertEqual(expected_cost, cost)

  def test_Equal2to2(self):
    src_tree = tree_or_string('(:tense (:index e-1))')
    trg_tree = tree_or_string('(:index (:tense e-2))')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.inner_nodes.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 0.0
    self.assertEqual(expected_cost, cost)

  def test_Different1to0(self):
    src_tree = tree_or_string('(:index e-1)')
    trg_tree = tree_or_string('e-2')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.inner_nodes.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 1.0
    self.assertEqual(expected_cost, cost)

  def test_Different2to0(self):
    src_tree = tree_or_string('(:tense (:index e-1))')
    trg_tree = tree_or_string('e-2')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.inner_nodes.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 2.0
    self.assertEqual(expected_cost, cost)

if __name__ == '__main__':
  suite1  = unittest.TestLoader().loadTestsFromTestCase(InnerNodesDifferenceTestCase)
  suite2  = unittest.TestLoader().loadTestsFromTestCase(VariableDifferenceTestCase)
  # suite3  = unittest.TestLoader().loadTestsFromTestCase(TreeDifferenceComplexityTestCase)
  suite4  = unittest.TestLoader().loadTestsFromTestCase(EntityDifferenceTestCase)
  suite5  = unittest.TestLoader().loadTestsFromTestCase(TreeSizeTestCase)
  suite6  = unittest.TestLoader().loadTestsFromTestCase(VariableDifferenceIndividualTestCase)
  suite7  = unittest.TestLoader().loadTestsFromTestCase(EntityDifferenceIndividualTestCase)
  suites  = unittest.TestSuite([suite1, suite2,
                                # suite3,
                                suite4, suite5, suite6, suite7])
  unittest.TextTestRunner(verbosity=2).run(suites)

