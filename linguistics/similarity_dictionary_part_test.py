#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from linguistics.similarity import Similarity as SimilarityOrig
from linguistics.similarity_dictionary_part import DictionaryCostPart
from utils.tree_tools import TreePattern, tree_or_string

class Similarity(SimilarityOrig):
  def __eq__(self, other):
    return isinstance(self, other.__class__) and \
           self.source == other.source and \
           self.target == other.target and \
           self.relation == other.relation

class DictionaryCostPartGetSimilarityTestCase(unittest.TestCase):
  def setUp(self):
    feature_weight = 1.0
    dict_filename = './linguistics/similarity_dictionary_test.txt'
    self.dict_cost = DictionaryCostPart(dict_filename, feature_weight)

  def test_TerminalLongFullMatch(self):
    src_tree = tree_or_string(u'私_は_食べる')
    trg_tree = tree_or_string(u'i eat')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.dict_cost.GetSimilarity(src_tree_pat, trg_tree_pat)
    expected_similarities = [Similarity(0.2, 'dict_part', src_tree_pat, trg_tree_pat)]
    self.assertEqual(expected_similarities, similarities)

  def test_TerminalShortFullMatch(self):
    src_tree = tree_or_string(u'私_は_食べる')
    trg_tree = tree_or_string(u'eat')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.dict_cost.GetSimilarity(src_tree_pat, trg_tree_pat)
    expected_similarities = [Similarity(0.2, 'dict_part', src_tree_pat, trg_tree_pat)]
    self.assertEqual(expected_similarities, similarities)

  def test_TerminalSingleFullMatch(self):
    src_tree = tree_or_string(u'食べる')
    trg_tree = tree_or_string(u'eat')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.dict_cost.GetSimilarity(src_tree_pat, trg_tree_pat)
    expected_similarities = [Similarity(0.2, 'dict_part', src_tree_pat, trg_tree_pat)]
    self.assertEqual(expected_similarities, similarities)

  def test_TerminalSingleFullMatch2(self):
    src_tree = tree_or_string(u'食べた')
    trg_tree = tree_or_string(u'ate')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.dict_cost.GetSimilarity(src_tree_pat, trg_tree_pat)
    expected_similarities = [Similarity(0.1, 'dict_part', src_tree_pat, trg_tree_pat)]
    self.assertEqual(expected_similarities, similarities)

class DictionaryCostPartGetSimilarTestCase(unittest.TestCase):
  def setUp(self):
    feature_weight = 1.0
    dict_filename = './linguistics/similarity_dictionary_test.txt'
    self.dict_cost = DictionaryCostPart(dict_filename, feature_weight)
    self.maxDiff = None

  def test_TerminalLongFullMatch(self):
    src_tree = tree_or_string(u'私_は_食べる')
    src_tree_pat = TreePattern(src_tree, (), [])
    similarities = self.dict_cost.GetSimilar(src_tree_pat)

    expected_similarities = []
    trg_tree = tree_or_string(u'I_eat')
    trg_tree_pat = TreePattern(trg_tree, (), [])
    expected_similarities.append(Similarity(0.9, 'dict_part', src_tree_pat, trg_tree_pat))
    trg_tree = tree_or_string(u'eat')
    trg_tree_pat = TreePattern(trg_tree, (), [])
    expected_similarities.append(Similarity(0.9, 'dict_part', src_tree_pat, trg_tree_pat))
    trg_tree = tree_or_string(u'ate')
    trg_tree_pat = TreePattern(trg_tree, (), [])
    expected_similarities.append(Similarity(0.8, 'dict_part', src_tree_pat, trg_tree_pat))
    self.assertItemsEqual(expected_similarities, similarities)

  def test_TerminalSingleFullMatch(self):
    src_tree = tree_or_string(u'食べる')
    src_tree_pat = TreePattern(src_tree, (), [])
    similarities = self.dict_cost.GetSimilar(src_tree_pat)

    expected_similarities = []
    trg_tree = tree_or_string(u'eat')
    trg_tree_pat = TreePattern(trg_tree, (), [])
    expected_similarities.append(Similarity(0.9, 'dict_part', src_tree_pat, trg_tree_pat))
    trg_tree = tree_or_string(u'ate')
    trg_tree_pat = TreePattern(trg_tree, (), [])
    expected_similarities.append(Similarity(0.8, 'dict_part', src_tree_pat, trg_tree_pat))
    trg_tree = tree_or_string(u'I_eat')
    trg_tree_pat = TreePattern(trg_tree, (), [])
    expected_similarities.append(Similarity(0.9, 'dict_part', src_tree_pat, trg_tree_pat))
    self.assertItemsEqual(expected_similarities, similarities)

  def test_TerminalSingleFullMatch2(self):
    src_tree = tree_or_string(u'食べた')
    src_tree_pat = TreePattern(src_tree, (), [])
    similarities = self.dict_cost.GetSimilar(src_tree_pat)

    expected_similarities = []
    trg_tree = tree_or_string(u'eat')
    trg_tree_pat = TreePattern(trg_tree, (), [])
    expected_similarities.append(Similarity(0.8, 'dict_part', src_tree_pat, trg_tree_pat))
    trg_tree = tree_or_string(u'ate')
    trg_tree_pat = TreePattern(trg_tree, (), [])
    expected_similarities.append(Similarity(0.9, 'dict_part', src_tree_pat, trg_tree_pat))
    self.assertItemsEqual(expected_similarities, similarities)

if __name__ == '__main__':
  suite1  = unittest.TestLoader().loadTestsFromTestCase(DictionaryCostPartGetSimilarityTestCase)
  suite2  = unittest.TestLoader().loadTestsFromTestCase(DictionaryCostPartGetSimilarTestCase)
  suites  = unittest.TestSuite([suite1, suite2])
  unittest.TextTestRunner(verbosity=2).run(suites)

