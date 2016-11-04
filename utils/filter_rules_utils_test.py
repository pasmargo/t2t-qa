#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from nltk import ImmutableTree

from training.transductionrule import XTRule
from utils.filter_rules_utils import RuleFilter
from utils.tree_tools import tree_or_string

class RuleFilterTestCase(unittest.TestCase):
  def test_PositiveMatchState(self):
    conds = ['q']
    rule_filter = RuleFilter(conds)
    rule = XTRule('q',
                  tree_or_string('(DT ?x0|)'),
                  tree_or_string('(DT ?x0|)'),
                  {(0,) : 'q1'}, 1.0)
    self.assertTrue(rule_filter.signal_rule(rule))

  def test_NegativeMatchState(self):
    conds = ['p']
    rule_filter = RuleFilter(conds)
    rule = XTRule('q',
                  tree_or_string('(DT ?x0|)'),
                  tree_or_string('(DT ?x0|)'),
                  {(0,) : 'q1'}, 1.0)
    self.assertFalse(rule_filter.signal_rule(rule))

  def test_PositiveMatchLHSString(self):
    conds = ['q,lhs:is_str']
    rule_filter = RuleFilter(conds)
    rule = XTRule('q',
                  tree_or_string('the'),
                  tree_or_string('(DT ?x0|)'),
                  {(0,) : 'q1'}, 1.0)
    self.assertTrue(rule_filter.signal_rule(rule))

  def test_NegativeMatchLHSString(self):
    conds = ['q,lhs:is_str']
    rule_filter = RuleFilter(conds)
    rule = XTRule('q',
                  tree_or_string('(DT ?x0|)'),
                  tree_or_string('(DT ?x0|)'),
                  {(0,) : 'q1'}, 1.0)
    self.assertFalse(rule_filter.signal_rule(rule))

  def test_PositiveMatchLHSStringRHSString(self):
    conds = ['q,lhs:is_str,rhs:is_str']
    rule_filter = RuleFilter(conds)
    rule = XTRule('q',
                  tree_or_string('the'),
                  tree_or_string('the'),
                  {}, 1.0)
    self.assertTrue(rule_filter.signal_rule(rule))

  def test_PositiveMatchLHSStringRHSStringMultipleConds(self):
    conds = ['p', 'q,lhs:is_str,rhs:is_str']
    rule_filter = RuleFilter(conds)
    rule = XTRule('q',
                  tree_or_string('the'),
                  tree_or_string('the'),
                  {}, 1.0)
    self.assertTrue(rule_filter.signal_rule(rule))

if __name__ == '__main__':
  suite1  = unittest.TestLoader().loadTestsFromTestCase(RuleFilterTestCase)
  suites  = unittest.TestSuite([suite1])
  unittest.TextTestRunner(verbosity=2).run(suites)

