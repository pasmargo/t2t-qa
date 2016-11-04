#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from decoder.decode_qa_utils import CVTInserter
from utils.tree_tools import tree_or_string

class CVTInserterTestCase(unittest.TestCase):
  def setUp(self):
    self.cvt_inserter = CVTInserter(cache_filename='.cvt_cache_test')

  def tearDown(self):
    self.cvt_inserter.close()

  def test_TickerSymbol(self):
    tree = tree_or_string(
      u'(ID !fb:finance.stock_exchange.companies_traded '
            'fb:en.new_york_stock_exchange_inc)')
    tree_new = self.cvt_inserter.insert_cvt_if_needed(tree)
    expected_tree = tree_or_string(
      u'(ID !fb:business.stock_ticker_symbol.ticker_symbol '
          '(ID !fb:finance.stock_exchange.companies_traded '
               'fb:en.new_york_stock_exchange_inc))')
    self.assertEqual(expected_tree, tree_new)

  def test_TickerSymbolWithCount(self):
    tree = tree_or_string(
      u'(COUNT '
          '(ID !fb:finance.stock_exchange.companies_traded '
               'fb:en.new_york_stock_exchange_inc))')
    tree_new = self.cvt_inserter.insert_cvt_if_needed(tree)
    expected_tree = tree_or_string(
      u'(COUNT '
           '(ID !fb:business.stock_ticker_symbol.ticker_symbol '
               '(ID !fb:finance.stock_exchange.companies_traded '
                    'fb:en.new_york_stock_exchange_inc)))')
    self.assertEqual(expected_tree, tree_new)

  def test_NumberObjSubj(self):
    tree = tree_or_string(
      u'(ID !fb:amusement_parks.park.annual_visits '
            'fb:en.magic_kingdom)')
    tree_new = self.cvt_inserter.insert_cvt_if_needed(tree)
    expected_tree = tree_or_string(
      u'(ID !fb:measurement_unit.dated_integer.number '
           '(ID !fb:amusement_parks.park.annual_visits '
                'fb:en.magic_kingdom))')
    self.assertEqual(expected_tree, tree_new)

  def test_PersonSubjSubj(self):
    tree = tree_or_string(
      u'(ID fb:business.employment_tenure.company '
            'fb:en.gap_inc)')
    tree_new = self.cvt_inserter.insert_cvt_if_needed(tree)
    expected_tree = tree_or_string(
      u'(ID !fb:business.employment_tenure.person '
           '(ID fb:business.employment_tenure.company '
                'fb:en.gap_inc))')
    self.assertEqual(expected_tree, tree_new)

  def test_PersonSubjSubjDoubleChildren(self):
    tree = tree_or_string(
      u'(ID '
          '(ID fb:business.employment_tenure.title '
              'fb:en.president)'
         u'(ID fb:business.employment_tenure.company '
              'fb:en.gap_inc))')
    tree_new = self.cvt_inserter.insert_cvt_if_needed(tree)
    expected_tree = tree_or_string(
      u'(ID !fb:business.employment_tenure.person '
          '(ID fb:business.employment_tenure.title '
              'fb:en.president)'
         u'(ID fb:business.employment_tenure.company '
              'fb:en.gap_inc))')
    self.assertEqual(expected_tree, tree_new)

  def test_EntityNoCVT(self):
    tree = tree_or_string(u'fb:en.president')
    tree_new = self.cvt_inserter.insert_cvt_if_needed(tree)
    expected_tree = tree_or_string(u'fb:en.president')
    self.assertEqual(expected_tree, tree_new)

  def test_NumberNoCVT(self):
    tree = tree_or_string(u'1998_-1_-1')
    tree_new = self.cvt_inserter.insert_cvt_if_needed(tree)
    expected_tree = tree_or_string(u'1998_-1_-1')
    self.assertEqual(expected_tree, tree_new)

if __name__ == '__main__':
  suite1 = unittest.TestLoader().loadTestsFromTestCase(CVTInserterTestCase)
  suites = unittest.TestSuite([suite1])
  unittest.TextTestRunner(verbosity=2).run(suites)

