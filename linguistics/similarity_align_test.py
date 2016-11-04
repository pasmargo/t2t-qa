#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import unittest

from linguistics.similarity import Similarity
from linguistics.similarity_align import AlignmentCost
from utils.tree_tools import TreePattern, tree_or_string

# TODO: test alignments in the presence of epsilon transitions (variables).
# TODO: test alignments when spurious in target, and in source&target.
class AlignmentCostTestCase(unittest.TestCase):
  def setUp(self):
    alignment_fname = './linguistics/alignments_test.txt'
    self.alignment_cost = AlignmentCost(alignment_fname, 1.0)

  # No spurious (unaligned) words neither in source nor target.

  def test_ConsistentAlignPreterminals(self):
    src_tree = tree_or_string(u'(A (B b) (C c))')
    trg_tree = tree_or_string(u'(AA (BB bb) (CC cc))')
    src_treep = TreePattern(src_tree, (0,), [])
    trg_treep = TreePattern(trg_tree, (0,), [])
    similarities = self.alignment_cost.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.alignment_cost.kCost, similarities[0].score)

  def test_ConsistentAlignLeaves(self):
    src_tree = tree_or_string(u'(A (B b) (C c))')
    trg_tree = tree_or_string(u'(AA (BB bb) (CC cc))')
    src_treep = TreePattern(src_tree, (0,0), [])
    trg_treep = TreePattern(trg_tree, (0,0), [])
    similarities = self.alignment_cost.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.alignment_cost.kCost, similarities[0].score)

  def test_ConsistentAlignNodes(self):
    src_tree = tree_or_string(u'(A (B b) (C c))')
    trg_tree = tree_or_string(u'(AA (BB bb) (CC cc))')
    src_treep = TreePattern(src_tree, (), [])
    trg_treep = TreePattern(trg_tree, (), [])
    similarities = self.alignment_cost.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.alignment_cost.kCost, similarities[0].score)

  def test_ViolateAlignLeaves(self):
    src_tree = tree_or_string(u'(A (B b) (C c))')
    trg_tree = tree_or_string(u'(AA (BB bb) (CC cc))')
    src_treep = TreePattern(src_tree, (0,0), [])
    trg_treep = TreePattern(trg_tree, (1,0), [])
    similarities = self.alignment_cost.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.alignment_cost.kCostViolation, similarities[0].score)

  def test_ViolateAlignNodeToPreterminal(self):
    src_tree = tree_or_string(u'(A (B b) (C c))')
    trg_tree = tree_or_string(u'(AA (BB bb) (CC cc))')
    src_treep = TreePattern(src_tree, (), [])
    trg_treep = TreePattern(trg_tree, (0,), [])
    similarities = self.alignment_cost.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.alignment_cost.kCostViolation, similarities[0].score)

  def test_ViolateAlignPreterminalToNode(self):
    src_tree = tree_or_string(u'(A (B b) (C c))')
    trg_tree = tree_or_string(u'(AA (BB bb) (CC cc))')
    src_treep = TreePattern(src_tree, (0,), [])
    trg_treep = TreePattern(trg_tree, (), [])
    similarities = self.alignment_cost.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.alignment_cost.kCostViolation, similarities[0].score)

  def test_ViolateAlignPreterminal(self):
    src_tree = tree_or_string(u'(A (B b) (C c))')
    trg_tree = tree_or_string(u'(AA (BB bb) (CC cc))')
    src_treep = TreePattern(src_tree, (0,), [])
    trg_treep = TreePattern(trg_tree, (1,), [])
    # from pudb import set_trace; set_trace()
    similarities = self.alignment_cost.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.alignment_cost.kCostViolation, similarities[0].score)

  # Spurious (unaligned) words in source.

  def test_ConsistentAlignPreterminalsReordering(self):
    src_tree = tree_or_string(u'(A (B b) (X x) (C c))')
    trg_tree = tree_or_string(u'(A (CC cc) (BB bb))')
    src_treep = TreePattern(src_tree, (0,), [])
    trg_treep = TreePattern(trg_tree, (1,), [])
    similarities = self.alignment_cost.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.alignment_cost.kCost, similarities[0].score)

  def test_ConsistentAlignNodes(self):
    src_tree = tree_or_string(u'(A (B b) (X x) (C c))')
    trg_tree = tree_or_string(u'(A (CC cc) (BB bb))')
    src_treep = TreePattern(src_tree, (), [])
    trg_treep = TreePattern(trg_tree, (), [])
    similarities = self.alignment_cost.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.alignment_cost.kCost, similarities[0].score)

  def test_ViolateAlignPreterminalsSrcUnaligned(self):
    src_tree = tree_or_string(u'(A (B b) (X x) (C c))')
    trg_tree = tree_or_string(u'(A (CC cc) (BB bb))')
    src_treep = TreePattern(src_tree, (1,), [])
    trg_treep = TreePattern(trg_tree, (0,), [])
    similarities = self.alignment_cost.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.alignment_cost.kCostViolation, similarities[0].score)

  def test_ViolateAlignPreterminalsReordering(self):
    src_tree = tree_or_string(u'(A (B b) (X x) (C c))')
    trg_tree = tree_or_string(u'(A (CC cc) (BB bb))')
    src_treep = TreePattern(src_tree, (0,), [])
    trg_treep = TreePattern(trg_tree, (0,), [])
    similarities = self.alignment_cost.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.alignment_cost.kCostViolation, similarities[0].score)

  # Spurious (unaligned) words in target.

  def test_ConsistentAlignPreterminalsTrgSpurious(self):
    src_tree = tree_or_string(u'(A (B b) (C c))')
    trg_tree = tree_or_string(u'(A (CC cc) (X x) (BB bb))')
    src_treep = TreePattern(src_tree, (0,), [])
    trg_treep = TreePattern(trg_tree, (2,), [])
    similarities = self.alignment_cost.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.alignment_cost.kCost, similarities[0].score)

  def test_ConsistentAlignNodesTrgSpurious(self):
    src_tree = tree_or_string(u'(A (B b) (C c))')
    trg_tree = tree_or_string(u'(A (CC cc) (X x) (BB bb))')
    src_treep = TreePattern(src_tree, (), [])
    trg_treep = TreePattern(trg_tree, (), [])
    similarities = self.alignment_cost.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.alignment_cost.kCost, similarities[0].score)

  def test_ViolateAlignTrgSpurious(self):
    src_tree = tree_or_string(u'(A (B b) (C c))')
    trg_tree = tree_or_string(u'(A (CC cc) (X x) (BB bb))')
    src_treep = TreePattern(src_tree, (0,), [])
    trg_treep = TreePattern(trg_tree, (1,), [])
    similarities = self.alignment_cost.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.alignment_cost.kCostViolation, similarities[0].score)

  # Spurious (unaligned) words in source AND target.

  def test_ConsistentAlignNodesSrcTrgSpurious(self):
    src_tree = tree_or_string(u'(A (B b) (X x) (C c))')
    trg_tree = tree_or_string(u'(A (CC cc) (XX xx) (BB bb))')
    src_treep = TreePattern(src_tree, (), [])
    trg_treep = TreePattern(trg_tree, (), [])
    similarities = self.alignment_cost.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.alignment_cost.kCost, similarities[0].score)

  def test_ConsistentAlignPreterminalsSrcTrgSpurious(self):
    src_tree = tree_or_string(u'(A (B b) (X x) (C c))')
    trg_tree = tree_or_string(u'(A (CC cc) (XX xx) (BB bb))')
    src_treep = TreePattern(src_tree, (0,), [])
    trg_treep = TreePattern(trg_tree, (2,), [])
    similarities = self.alignment_cost.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.alignment_cost.kCost, similarities[0].score)

  def test_ViolateAlignPreterminalToSpuriousSrcTrgSpurious(self):
    src_tree = tree_or_string(u'(A (B b) (X x) (C c))')
    trg_tree = tree_or_string(u'(A (CC cc) (XX xx) (BB bb))')
    src_treep = TreePattern(src_tree, (0,), [])
    trg_treep = TreePattern(trg_tree, (1,), [])
    similarities = self.alignment_cost.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.alignment_cost.kCostViolation, similarities[0].score)

  def test_ViolateAlignWrongPreterminalsSrcTrgSpurious(self):
    src_tree = tree_or_string(u'(A (B b) (X x) (C c))')
    trg_tree = tree_or_string(u'(A (CC cc) (XX xx) (BB bb))')
    src_treep = TreePattern(src_tree, (0,), [])
    trg_treep = TreePattern(trg_tree, (0,), [])
    similarities = self.alignment_cost.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.alignment_cost.kCostViolation, similarities[0].score)

  # Cross-constituents.

  def test_ConsistentAlignSub(self):
    src_tree = tree_or_string(u'(A (B (C c) (D d)) (E e))')
    trg_tree = tree_or_string(u'(AA (BB bb) (FF ff))')
    src_treep = TreePattern(src_tree, (0,), [])
    trg_treep = TreePattern(trg_tree, (0,), [])
    similarities = self.alignment_cost.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.alignment_cost.kCost, similarities[0].score)

  def test_ConsistentAlignPreterminal(self):
    src_tree = tree_or_string(u'(A (B (C c) (D d)) (E e))')
    trg_tree = tree_or_string(u'(AA (BB bb) (FF ff))')
    src_treep = TreePattern(src_tree, (1,), [])
    trg_treep = TreePattern(trg_tree, (1,), [])
    similarities = self.alignment_cost.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.alignment_cost.kCost, similarities[0].score)

  def test_ConsistentAlignNodes(self):
    src_tree = tree_or_string(u'(A (B (C c) (D d)) (E e))')
    trg_tree = tree_or_string(u'(AA (BB bb) (FF ff))')
    src_treep = TreePattern(src_tree, (), [])
    trg_treep = TreePattern(trg_tree, (), [])
    similarities = self.alignment_cost.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.alignment_cost.kCost, similarities[0].score)

  def test_ViolateAlignSub1(self):
    src_tree = tree_or_string(u'(A (B (C c) (D d)) (E e))')
    trg_tree = tree_or_string(u'(AA (BB bb) (FF ff))')
    src_treep = TreePattern(src_tree, (0,0), [])
    trg_treep = TreePattern(trg_tree, (0,), [])
    similarities = self.alignment_cost.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.alignment_cost.kCostViolation, similarities[0].score)

  def test_ViolateAlignSub2(self):
    src_tree = tree_or_string(u'(A (B (C c) (D d)) (E e))')
    trg_tree = tree_or_string(u'(AA (BB bb) (FF ff))')
    src_treep = TreePattern(src_tree, (0,1), [])
    trg_treep = TreePattern(trg_tree, (0,), [])
    similarities = self.alignment_cost.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.alignment_cost.kCostViolation, similarities[0].score)

  def test_ViolateAlignSub3(self):
    src_tree = tree_or_string(u'(A (B (C c) (D d)) (E e))')
    trg_tree = tree_or_string(u'(AA (BB bb) (FF ff))')
    src_treep = TreePattern(src_tree, (0,), [])
    trg_treep = TreePattern(trg_tree, (), [])
    similarities = self.alignment_cost.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.alignment_cost.kCostViolation, similarities[0].score)

  # Cross-constituent violations.

  def test_ViolateAlignCross1(self):
    src_tree = tree_or_string(u'(A (B (C c) (D d)) (X x))')
    trg_tree = tree_or_string(u'(AA (BB bb) (FF ff))')
    src_treep = TreePattern(src_tree, (0,), [])
    trg_treep = TreePattern(trg_tree, (0,), [])
    similarities = self.alignment_cost.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.alignment_cost.kCostViolation, similarities[0].score)

  def test_ViolateAlignCross2(self):
    src_tree = tree_or_string(u'(A (B (C c) (D d)) (X x))')
    trg_tree = tree_or_string(u'(AA (BB bb) (FF ff))')
    src_treep = TreePattern(src_tree, (0,1), [])
    trg_treep = TreePattern(trg_tree, (1,), [])
    similarities = self.alignment_cost.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.alignment_cost.kCostViolation, similarities[0].score)

  def test_ViolateAlignCross3(self):
    src_tree = tree_or_string(u'(A (B (C c) (D d)) (X x))')
    trg_tree = tree_or_string(u'(AA (BB bb) (FF ff))')
    src_treep = TreePattern(src_tree, (0,), [])
    trg_treep = TreePattern(trg_tree, (1,), [])
    similarities = self.alignment_cost.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.alignment_cost.kCostViolation, similarities[0].score)

  def test_ConsistentAlignCross1(self):
    src_tree = tree_or_string(u'(A (B (C c) (D d)) (X x))')
    trg_tree = tree_or_string(u'(AA (BB bb) (FF ff))')
    src_treep = TreePattern(src_tree, (0,), [])
    trg_treep = TreePattern(trg_tree, (), [])
    similarities = self.alignment_cost.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.alignment_cost.kCost, similarities[0].score)

  def test_ConsistentAlignCross1(self):
    src_tree = tree_or_string(u'(A (B (C c) (D d)) (X x))')
    trg_tree = tree_or_string(u'(AA (BB bb) (FF ff))')
    src_treep = TreePattern(src_tree, (), [])
    trg_treep = TreePattern(trg_tree, (), [])
    similarities = self.alignment_cost.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.alignment_cost.kCost, similarities[0].score)


if __name__ == '__main__':
  suite1 = unittest.TestLoader().loadTestsFromTestCase(AlignmentCostTestCase)
  suites = unittest.TestSuite([suite1])
  unittest.TextTestRunner(verbosity=2).run(suites)

