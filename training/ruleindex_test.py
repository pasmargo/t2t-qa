#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from training.transductionrule import *
from training.ruleindex import *
from utils.tree_tools import Tree, tree_or_string

class T2TGetRelevantRulesTestCase(unittest.TestCase):
  def setUp(self):
    rule1 = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        {(0,) : 'q', (1,) : 'q'}, 1.0)
    rule2 = XTRule('q', tree_or_string('(DT ?x0|)'),
                        tree_or_string('(DT ?x0|)'),
                        {(0,) : 't'}, 1.0)
    rule3 = XTRule('q', tree_or_string('(NN ?x0|)'),
                        tree_or_string('(NN ?x0|)'),
                        {(0,) : 't'}, 0.8)
    rule4 = XTRule('t', tree_or_string('the'),
                        tree_or_string('la'),
                        {}, 1.0)
    rule5 = XTRule('t', tree_or_string('house'),
                        tree_or_string('casa'),
                        {}, 1.0)
    rule6 = XTRule('t', tree_or_string('house'),
                        tree_or_string('morada'),
                        {}, 1.0)
    rule7 = XTRule('q', tree_or_string('(NN ?x0|)'),
                        tree_or_string('(JJ ?x0|)'),
                        {(0,) : 't'}, 0.2)
    rule8 = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        tree_or_string('(NP (DT el) ?x1|NN)'),
                        {(1,) : 'q'}, 1.0)
    rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8]
    self.rule_index = RuleIndexT2T(rules)

  def test_RhsVarMatchesJpEn(self):
    rule0 = XTRule('q', tree_or_string(u'(NP ?x0|NN)'),
                        tree_or_string(u'?x0|NN'),
                        {() : 'q'}, 1.0)
    rule1 = XTRule('q', tree_or_string(u'(NN ?x0|)'),
                        tree_or_string(u'(NN ?x0|)'),
                        {(0,) : 't'}, 1.0)
    rule2 = XTRule('t', tree_or_string(u'学生'),
                        tree_or_string(u'student'),
                        {}, 1.0)
    rules = [rule0, rule1, rule2]
    rule_index = RuleIndexT2T(rules)
    tree1 = Tree.fromstring(u'(NP (NN 学生))')
    tree2 = Tree.fromstring(u'(NN student)')
    relevant_rules = rule_index.GetRelevantRules(tree1, ((), 'q'), tree2, ())
    expected_rule = XTRule('q', tree_or_string(u'(NP ?x0|NN)'),
                                tree_or_string(u'?x0|NN'),
                                {() : 'q'}, 1.0)
    self.assertEqual(1, len(relevant_rules))
    self.assertListEqual([expected_rule], relevant_rules)

  def test_RhsVarWrongTypeNoMatchesJpEn(self):
    rule0 = XTRule('q', tree_or_string(u'(NP ?x0|NN)'),
                        tree_or_string(u'?x0|JJ'),
                        {() : 'q'}, 1.0)
    rule1 = XTRule('q', tree_or_string(u'(NN ?x0|)'),
                        tree_or_string(u'(NN ?x0|)'),
                        {(0,) : 't'}, 1.0)
    rule2 = XTRule('t', tree_or_string(u'学生'),
                        tree_or_string(u'student'),
                        {}, 1.0)
    rules = [rule0, rule1, rule2]
    rule_index = RuleIndexT2T(rules)
    tree1 = Tree.fromstring(u'(NP (NN 学生))')
    tree2 = Tree.fromstring(u'(NN student)')
    relevant_rules = rule_index.GetRelevantRules(tree1, ((), 'q'), tree2, ())
    self.assertEqual(0, len(relevant_rules))

  def test_RhsVarNoTypeMatchesJpEn(self):
    rule0 = XTRule('q', tree_or_string(u'(NP ?x0|NN)'),
                        tree_or_string(u'?x0|'),
                        {() : 'q'}, 1.0)
    rule1 = XTRule('q', tree_or_string(u'(NN ?x0|)'),
                        tree_or_string(u'(NN ?x0|)'),
                        {(0,) : 't'}, 1.0)
    rule2 = XTRule('t', tree_or_string(u'学生'),
                        tree_or_string(u'student'),
                        {}, 1.0)
    rules = [rule0, rule1, rule2]
    rule_index = RuleIndexT2T(rules)
    tree1 = Tree.fromstring(u'(NP (NN 学生))')
    tree2 = Tree.fromstring(u'(NN student)')
    relevant_rules = rule_index.GetRelevantRules(tree1, ((), 'q'), tree2, ())
    self.assertEqual(1, len(relevant_rules))

  def test_SingleRuleMatchAtLeafJpEn(self):
    rule0 = XTRule('q', tree_or_string(u'(NP ?x0|NN)'),
                        tree_or_string(u'(NP (DT a) ?x0|NN)'),
                        {(1,) : 'q'}, 1.0)
    rule1 = XTRule('q', tree_or_string(u'(NN ?x0|)'),
                        tree_or_string(u'(NN ?x0|)'),
                        {(0,) : 't'}, 1.0)
    rule2 = XTRule('t', tree_or_string(u'学生'),
                        tree_or_string(u'student'),
                        {}, 1.0)
    rules = [rule0, rule1, rule2]
    rule_index = RuleIndexT2T(rules)
    tree1 = Tree.fromstring(u'(NP (NN 学生))')
    tree2 = Tree.fromstring(u'(NP (DT a) (NN student))')
    relevant_rules = rule_index.GetRelevantRules(tree1, ((0,0), 't'), tree2, (1,0))
    self.assertEqual(1, len(relevant_rules))
    expected_rule = XTRule('t', tree_or_string(u'学生'),
                                tree_or_string(u'student'),
                                {}, 1.0)
    self.assertListEqual([expected_rule], relevant_rules)

  def test_SingleRuleMatchAtRoot(self):
    tree1 = Tree.fromstring('(NP (DT the) (NN house))')
    tree2 = Tree.fromstring('(NP (DT la) (NN casa))')
    relevant_rules = self.rule_index.GetRelevantRules(tree1, ((), 'q'), tree2, ())
    self.assertEqual(1, len(relevant_rules))
    expected_rule = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                                tree_or_string('(NP ?x0|DT ?x1|NN)'),
                                {(0,) : 'q', (1,) : 'q'}, 1.0)
    self.assertListEqual([expected_rule], relevant_rules)

  def test_SingleWrongRuleNotMatchAtRoot(self):
    tree1 = Tree.fromstring('(NP (DT the) (JJ house))')
    tree2 = Tree.fromstring('(NP (DT la) (NN casa))')
    relevant_rules = self.rule_index.GetRelevantRules(tree1, ((), 'q'), tree2, ())
    self.assertEqual(0, len(relevant_rules))

  def test_TwoRulesMatchAtRoot(self):
    tree1 = Tree.fromstring('(NP (DT the) (NN house))')
    tree2 = Tree.fromstring('(NP (DT el) (NN casa))')
    relevant_rules = self.rule_index.GetRelevantRules(tree1, ((), 'q'), tree2, ())
    self.assertEqual(2, len(relevant_rules))
    expected_rule1 = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                                 tree_or_string('(NP ?x0|DT ?x1|NN)'),
                                 {(0,) : 'q', (1,) : 'q'}, 1.0)
    expected_rule2 = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                                 tree_or_string('(NP (DT el) ?x1|NN)'),
                                 {(1,) : 'q'}, 1.0)
    self.assertListEqual([expected_rule1, expected_rule2], relevant_rules)

  def test_SingleRuleMatchAtLevel2NN(self):
    tree1 = Tree.fromstring('(NP (DT the) (NN house))')
    tree2 = Tree.fromstring('(NP (DT la) (NN casa))')
    relevant_rules = self.rule_index.GetRelevantRules(tree1, ((1,), 'q'), tree2, (1,))
    self.assertEqual(1, len(relevant_rules))
    expected_rule = XTRule('q', tree_or_string('(NN ?x0|)'),
                                tree_or_string('(NN ?x0|)'),
                                {(0,) : 't'}, 0.8)
    self.assertListEqual([expected_rule], relevant_rules)

  def test_SingleRuleMatchAtLevel2JJ(self):
    tree1 = Tree.fromstring('(NP (DT the) (NN house))')
    tree2 = Tree.fromstring('(NP (DT la) (JJ casa))')
    relevant_rules = self.rule_index.GetRelevantRules(tree1, ((1,), 'q'), tree2, (1,))
    self.assertEqual(1, len(relevant_rules))
    expected_rule = XTRule('q', tree_or_string('(NN ?x0|)'),
                                tree_or_string('(JJ ?x0|)'),
                                {(0,) : 't'}, 0.2)
    self.assertListEqual([expected_rule], relevant_rules)

  def test_SingleRuleNoMatchAtLevel2JJ(self):
    tree1 = Tree.fromstring('(NP (DT the) (NN house))')
    tree2 = Tree.fromstring('(NP (DT la) (PP casa))')
    relevant_rules = self.rule_index.GetRelevantRules(tree1, ((1,), 'q'), tree2, (1,))
    self.assertEqual(0, len(relevant_rules))

  def test_UntypedBranchesMatch(self):
    tree1 = Tree.fromstring('(NP (DT the) (NN house))')
    tree2 = Tree.fromstring('(NP (DT la) (PP casa))')
    rule = XTRule('q', tree_or_string('(NP ?x0| ?x1|)'),
                       tree_or_string('(NP ?x0|DT ?x1|PP)'),
                       {(0,) : 'q', (1,) : 'q'}, 1.0)
    rule_index = RuleIndexT2T([rule])
    relevant_rules = rule_index.GetRelevantRules(tree1, ((), 'q'), tree2, ())
    self.assertEqual(1, len(relevant_rules))
 
  def test_SingleTypedVarLHS(self):
    tree1 = Tree.fromstring('(NP (DT the) (NN house))')
    tree2 = Tree.fromstring('(NP (DT la) (PP casa))')
    rule = XTRule('q', tree_or_string('?x0|NP'),
                       tree_or_string('(NP ?x0|DT ?x1|PP)'),
                       {(0,) : 'q', (1,) : 'q'}, 1.0)
    rule_index = RuleIndexT2T([rule])
    relevant_rules = rule_index.GetRelevantRules(tree1, ((), 'q'), tree2, ())
    self.assertEqual(1, len(relevant_rules))
 
  def test_SingleTypedVarLHSDifferentType(self):
    tree1 = Tree.fromstring('(NP (DT the) (NN house))')
    tree2 = Tree.fromstring('(NP (DT la) (PP casa))')
    rule = XTRule('q', tree_or_string('?x0|PP'),
                       tree_or_string('(NP ?x0|DT ?x1|PP)'),
                       {(0,) : 'q', (1,) : 'q'}, 1.0)
    rule_index = RuleIndexT2T([rule])
    relevant_rules = rule_index.GetRelevantRules(tree1, ((), 'q'), tree2, ())
    self.assertEqual(0, len(relevant_rules))
 
  def test_SingleTypedVarLHSNoType(self):
    tree1 = Tree.fromstring('(NP (DT the) (NN house))')
    tree2 = Tree.fromstring('(NP (DT la) (PP casa))')
    rule = XTRule('q', tree_or_string('?x0|'),
                       tree_or_string('(NP ?x0|DT ?x1|PP)'),
                       {(0,) : 'q', (1,) : 'q'}, 1.0)
    rule_index = RuleIndexT2T([rule])
    relevant_rules = rule_index.GetRelevantRules(tree1, ((), 'q'), tree2, ())
    self.assertEqual(1, len(relevant_rules))
 
  def test_SingleTypedVarRHS(self):
    tree1 = Tree.fromstring('(NP (DT the) (NN house))')
    tree2 = Tree.fromstring('(NP (DT la) (PP casa))')
    rule = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                       tree_or_string('?x0|NP'),
                       {() : 'q'}, 1.0)
    rule_index = RuleIndexT2T([rule])
    relevant_rules = rule_index.GetRelevantRules(tree1, ((), 'q'), tree2, ())
    self.assertEqual(1, len(relevant_rules))
 
  def test_SingleVarLHSLevel2(self):
    tree1 = Tree.fromstring('(NP (DT the) (NN house))')
    tree2 = Tree.fromstring('(NP (DT la) (PP casa))')
    rule = XTRule('q', tree_or_string('(DT ?x0|)'),
                       tree_or_string('?x0|'),
                       {() : 'q'}, 1.0)
    rule_index = RuleIndexT2T([rule])
    relevant_rules = rule_index.GetRelevantRules(tree1, ((0,), 'q'), tree2, (0, 0))
    self.assertEqual(1, len(relevant_rules))
 
if __name__ == '__main__':
  suite1  = unittest.TestLoader().loadTestsFromTestCase(T2TGetRelevantRulesTestCase)
  suites  = unittest.TestSuite([suite1])
  unittest.TextTestRunner(verbosity=2).run(suites)

