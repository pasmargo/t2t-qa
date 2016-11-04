#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

# from pudb import set_trace; set_trace()

from linguistics.similarity_pre import Identity
from training.transductionrule import *
from training.ruleindex import *
from training.transducer import xT
from training.wrtg import (SourceProjectionFromDerivationStrict,
  TargetProjectionFromDerivation)
from utils.tree_tools import tree_or_string

class IdentitySimilarityTestCase(unittest.TestCase):
  def setUp(self):
    self.kCost = 1e-300
    self.S = 'q0'

  def test_IdentityTokenGetRelevant(self):
    rules = []
    rule_backoffs = [Identity()]
    rule_index = RuleIndexT2T(rules, rule_backoffs)
    tree1 = tree_or_string(u'学生')
    tree2 = None
    relevant_rules = rule_index.GetRelevantRules(tree1, ((), self.S), tree2, ())
    expected_rule = XTRule(self.S, tree_or_string(u'学生'),
                                   tree_or_string(u'学生'),
                                   {}, self.kCost)
    self.assertListEqual([expected_rule], relevant_rules)

  def test_IdentityPreterminalGetRelevant(self):
    rules = []
    rule_backoffs = [Identity()]
    rule_index = RuleIndexT2T(rules, rule_backoffs)
    tree1 = tree_or_string(u'(NN 学生)')
    tree2 = None
    relevant_rules = rule_index.GetRelevantRules(tree1, ((), self.S), tree2, ())
    expected_rule = XTRule(self.S, tree_or_string(u'(NN ?x0|)'),
                                   tree_or_string(u'(NN ?x0|)'),
                                   {(0,) : self.S}, self.kCost)
    self.assertListEqual([expected_rule], relevant_rules)

  def test_IdentityNoNeedTerminalTransduce(self):
    rule0 = XTRule(self.S, tree_or_string('(NP ?x0|JJ ?x1|NN)'),
                           tree_or_string('(NP ?x1|NN ?x0|JJ)'),
                           {(0,) : self.S, (1,) : self.S}, 1.0)
    rule1 = XTRule(self.S, tree_or_string('(JJ ?x0|)'),
                           tree_or_string('(JJ ?x0|)'),
                           {(0,) : 't'}, 1.0)
    rule2 = XTRule(self.S, tree_or_string('(NN ?x0|)'),
                           tree_or_string('(NN ?x0|)'),
                           {(0,) : 't'}, 0.8)
    rule3 = XTRule('t', tree_or_string('beautiful'),
                        tree_or_string('bonita'),
                        {}, 1.0)
    rule4 = XTRule('t', tree_or_string('house'),
                        tree_or_string('casa'),
                        {}, 1.0)
    rules = [rule0, rule1, rule2, rule3, rule4]
    rule_backoffs = [Identity()]
    rule_index = RuleIndexT2T(rules, rule_backoffs)
    tree1 = tree_or_string(u'(NP (JJ beautiful) (NN house))')
    tree2 = None
    transducer = xT(self.S, rules, rule_backoffs)
    wrtg = transducer.Transduce(tree1)
    derivation = wrtg.ObtainBestDerivation()
    src_projection, _ = SourceProjectionFromDerivationStrict(derivation)
    expected_src_projection = tree_or_string(u'(NP (NN house) (JJ beautiful))')
    self.assertEqual(expected_src_projection, src_projection)

    trg_projection, _ = TargetProjectionFromDerivation(derivation)
    expected_trg_projection = tree_or_string(u'(NP (NN casa) (JJ bonita))')
    self.assertEqual(expected_trg_projection, trg_projection)

  def test_IdentityNeedTerminalTransduce(self):
    rule0 = XTRule(self.S, tree_or_string('(NP ?x0|JJ ?x1|NN)'),
                           tree_or_string('(NP ?x1|NN ?x0|JJ)'),
                           {(0,) : self.S, (1,) : self.S}, 1.0)
    rule1 = XTRule(self.S, tree_or_string('(JJ ?x0|)'),
                           tree_or_string('(JJ ?x0|)'),
                           {(0,) : 't'}, 1.0)
    rule2 = XTRule(self.S, tree_or_string('(NN ?x0|)'),
                           tree_or_string('(NN ?x0|)'),
                           {(0,) : 't'}, 0.8)
    rule3 = XTRule('t', tree_or_string('beautiful'),
                        tree_or_string('bonita'),
                        {}, 1.0)
    rule4 = XTRule('t', tree_or_string('house'),
                        tree_or_string('casa'),
                        {}, 1.0)
    rules = [rule0, rule1, rule2, rule3, rule4]
    rule_backoffs = [Identity()]
    rule_index = RuleIndexT2T(rules, rule_backoffs)
    tree1 = tree_or_string(u'(NP (JJ beautiful) (NN home))')
    tree2 = None
    transducer = xT(self.S, rules, rule_backoffs)
    wrtg = transducer.Transduce(tree1)
    derivation = wrtg.ObtainBestDerivation()
    src_projection, _ = SourceProjectionFromDerivationStrict(derivation)
    expected_src_projection = tree_or_string(u'(NP (NN home) (JJ beautiful))')
    self.assertEqual(expected_src_projection, src_projection)

    trg_projection, _ = TargetProjectionFromDerivation(derivation)
    expected_trg_projection = tree_or_string(u'(NP (NN home) (JJ bonita))')
    self.assertEqual(expected_trg_projection, trg_projection)

  def test_IdentityNeedPreterminalTransduce(self):
    rule0 = XTRule(self.S, tree_or_string('(NP ?x0|JJ ?x1|XX)'),
                           tree_or_string('(NP ?x1|XX ?x0|JJ)'),
                           {(0,) : self.S, (1,) : self.S}, 1.0)
    rule1 = XTRule(self.S, tree_or_string('(JJ ?x0|)'),
                           tree_or_string('(JJ ?x0|)'),
                           {(0,) : self.S}, 1.0)
    rule2 = XTRule(self.S, tree_or_string('(NN ?x0|)'),
                           tree_or_string('(NN ?x0|)'),
                           {(0,) : self.S}, 0.8)
    rule3 = XTRule(self.S, tree_or_string('beautiful'),
                        tree_or_string('bonita'),
                        {}, 1.0)
    rule4 = XTRule(self.S, tree_or_string('house'),
                        tree_or_string('casa'),
                        {}, 1.0)
    rules = [rule0, rule1, rule2, rule3, rule4]
    rule_backoffs = [Identity()]
    rule_index = RuleIndexT2T(rules, rule_backoffs)
    tree1 = tree_or_string(u'(NP (JJ beautiful) (XX home))')
    tree2 = None
    transducer = xT(self.S, rules, rule_backoffs)
    wrtg = transducer.Transduce(tree1)
    derivation = wrtg.ObtainBestDerivation()
    src_projection, _ = SourceProjectionFromDerivationStrict(derivation)
    expected_src_projection = tree_or_string(u'(NP (XX home) (JJ beautiful))')
    self.assertEqual(expected_src_projection, src_projection)

    trg_projection, _ = TargetProjectionFromDerivation(derivation)
    expected_trg_projection = tree_or_string(u'(NP (XX home) (JJ bonita))')
    self.assertEqual(expected_trg_projection, trg_projection)

if __name__ == '__main__':
  suite1  = unittest.TestLoader().loadTestsFromTestCase(IdentitySimilarityTestCase)
  suites  = unittest.TestSuite([suite1])
  unittest.TextTestRunner(verbosity=2).run(suites)

