#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import unittest

from linguistics.similarity import Similarity
from linguistics.similarity_qa import (NoSimilarityQA, CountOp, EntityLinkingCost,
  PredicateLinkingCost, BridgeLinkingCost, NounPhraseCost, UriSurfCost)
from qald.grounding import Linker
from training.transductionrule import XTRule
from utils.tree_tools import TreePattern, tree_or_string

class UriSurfTestCase(unittest.TestCase):
  def setUp(self):
    self.uri_surf = UriSurfCost()
    self.entity_relation = self.uri_surf.entity_relation
    self.predicate_relation = self.uri_surf.predicate_relation
    self.bridge_relation = self.uri_surf.bridge_relation

  def test_Entity(self):
    src_tree = tree_or_string(u'(NN linebackers)')
    trg_tree = tree_or_string(u'fb:en.linebacker')
    src_treep = TreePattern(src_tree, (), [])
    trg_treep = TreePattern(trg_tree, (), [])
    similarities = self.uri_surf.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.entity_relation, similarities[0].relation)
    self.assertTrue(2.0 < similarities[0].score < 2.5)

  def test_NoOverlap(self):
    src_tree = tree_or_string(u'(NN linebackers)')
    trg_tree = tree_or_string(u'fb:en.yyyyzzzz')
    src_treep = TreePattern(src_tree, (), [])
    trg_treep = TreePattern(trg_tree, (), [])
    similarities = self.uri_surf.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(0, len(similarities))

  @unittest.expectedFailure
  def test_ShortAndLong(self):
    src_tree = tree_or_string(u'(NN li)')
    trg_tree = tree_or_string(u'fb:en.linebackers')
    src_treep = TreePattern(src_tree, (), [])
    trg_treep = TreePattern(trg_tree, (), [])
    similarities = self.uri_surf.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(0, len(similarities))

  def test_Predicate(self):
    src_tree = tree_or_string(u'(NN pets)')
    trg_tree = tree_or_string(u'fb:base.famouspets.pet_ownership.pet')
    src_treep = TreePattern(src_tree, (), [])
    trg_treep = TreePattern(trg_tree, (), [])
    similarities = self.uri_surf.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.predicate_relation, similarities[0].relation)
    self.assertTrue(2.0 < similarities[0].score < 2.5)

  def test_Bridge(self):
    src_tree = tree_or_string(u'(NN pets)')
    trg_tree = tree_or_string(u'(ID fb:blabla.bridge fb:base.famouspets.pet_ownership.pet)')
    src_treep = TreePattern(src_tree, (), [])
    trg_treep = TreePattern(trg_tree, (), [])
    similarities = self.uri_surf.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertEqual(self.bridge_relation, similarities[0].relation)
    extra_cost = self.uri_surf.extra_cost
    self.assertTrue(2.0 + extra_cost < similarities[0].score < 2.5 + extra_cost)

  def test_BridgeNo(self):
    src_tree = tree_or_string(u'(NN zzz)')
    trg_tree = tree_or_string(u'(ID fb:blabla.bridge fb:base.famouspets.pet_ownership.pet)')
    src_treep = TreePattern(src_tree, (), [])
    trg_treep = TreePattern(trg_tree, (), [])
    similarities = self.uri_surf.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(0, len(similarities))

class NounPhraseTestCase(unittest.TestCase):
  def setUp(self):
    self.np_similarity = NounPhraseCost()

  def test_NounPhrase(self):
    src_tree = tree_or_string(u'(NP (NNP Masaru) (NNP Yamamoto))')
    trg_tree = tree_or_string(u'yyy')
    src_treep = TreePattern(src_tree, (), [])
    trg_treep = TreePattern(trg_tree, (), [])
    similarities = self.np_similarity.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    cost = similarities[0].score
    expected_cost = self.np_similarity.cost_np
    self.assertAlmostEqual(expected_cost, cost)

  def test_NounPhraseVar(self):
    src_tree = tree_or_string(u'(NP (NNP Masaru) (NNP Yamamoto) ?x0|)')
    trg_tree = tree_or_string(u'yyy')
    src_treep = TreePattern(src_tree, (), [(2,)])
    trg_treep = TreePattern(trg_tree, (), [])
    similarities = self.np_similarity.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(0, len(similarities))

  def test_NNP(self):
    src_tree = tree_or_string(u'(NNP Masaru)')
    trg_tree = tree_or_string(u'yyy')
    src_treep = TreePattern(src_tree, (), [])
    trg_treep = TreePattern(trg_tree, (), [])
    similarities = self.np_similarity.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    cost = similarities[0].score
    expected_cost = self.np_similarity.cost_no_np
    self.assertAlmostEqual(expected_cost, cost)

class NoSimilarityQATestCase(unittest.TestCase):
  def setUp(self):
    self.no_similarity = NoSimilarityQA()

  def test_TerminalToTerminal(self):
    src_tree = tree_or_string(u'Yamamoto')
    trg_tree = tree_or_string(u'山本')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.no_similarity.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = self.no_similarity.kSubstitutionCost
    self.assertEqual(expected_cost, cost)

  def test_TerminalToPreterminal(self):
    src_tree = tree_or_string(u'Yamamoto')
    trg_tree = tree_or_string(u'(NN 山本)')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.no_similarity.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = self.no_similarity.kSubstitutionCost
    self.assertEqual(expected_cost, cost)

  def test_PreterminalToTerminal(self):
    src_tree = tree_or_string(u'(NN Yamamoto)')
    trg_tree = tree_or_string(u'山本')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.no_similarity.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = self.no_similarity.kSubstitutionCost
    self.assertEqual(expected_cost, cost)

  def test_NonterminalToNonterminal2to2(self):
    src_tree = tree_or_string(u'(NP (NN Mr.) (NN Yamamoto))')
    trg_tree = tree_or_string(u'(NP (NN 山本) (NN -san))')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.no_similarity.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 2.0 * self.no_similarity.kSubstitutionCost
    self.assertEqual(expected_cost, cost)

  def test_NonterminalToNonterminal3to2(self):
    src_tree = tree_or_string(u'(NP (NN Mr.) (NN T.) (NN Yamamoto))')
    trg_tree = tree_or_string(u'(NP (NN 山本) (NN -san))')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.no_similarity.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 2.0 * self.no_similarity.kSubstitutionCost \
                  + 1.0 * self.no_similarity.kDeletionCost
    self.assertEqual(expected_cost, cost)

  def test_NonterminalToNonterminal2to3(self):
    src_tree = tree_or_string(u'(NP (NN Mr.) (NN Yamamoto))')
    trg_tree = tree_or_string(u'(NP (NN 山本) (NN T.) (NN -san))')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [])
    similarities = self.no_similarity.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 2.0 * self.no_similarity.kSubstitutionCost \
                  + 1.0 * self.no_similarity.kInsertionCost
    self.assertEqual(expected_cost, cost)

  def test_NonterminalToNonterminal2to2Variable(self):
    src_tree = tree_or_string(u'(NP (NN Mr.) (NN Yamamoto))')
    trg_tree = tree_or_string(u'(NP (NN 山本) (NN T.) (NN -san))')
    src_tree_pat = TreePattern(src_tree, (), [])
    trg_tree_pat = TreePattern(trg_tree, (), [(1,)])
    similarities = self.no_similarity.GetSimilarity(src_tree_pat, trg_tree_pat)
    self.assertTrue(1, len(similarities))
    cost = similarities[0].score
    expected_cost = 2.0 * self.no_similarity.kSubstitutionCost
    self.assertEqual(expected_cost, cost)

class CountOpTestCase(unittest.TestCase):
  def setUp(self):
    linker = Linker()
    self.count_op = CountOp(1.0, linker)

  def tearDown(self):
    self.count_op.Close()

  def test_Count(self):
    rule = XTRule('q1',
                  tree_or_string('(ROOT (SBARQ (WHNP (WHADJP (WRB how) (JJ many)) ?x0|NNS) ?x1|SQ (. ?)))'),
                  tree_or_string('(ID COUNT (ID ?x0| ?x1|))'),
                  {(0, 0) : 'q2', (0, 1) : 'q2'}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    similarities = self.count_op.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertAlmostEqual(self.count_op.kCostPositive, similarities[0].score)

  def test_CountNo(self):
    rule = XTRule('q1',
                  tree_or_string('(ROOT (SBARQ (WHNP (WHADJP (WRB how) (JJ many)) ?x0|NNS) ?x1|SQ (. ?)))'),
                  tree_or_string('(ID MAX (ID ?x0| ?x1|))'),
                  {(0, 0) : 'q2', (0, 1) : 'q2'}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    similarities = self.count_op.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertAlmostEqual(self.count_op.kCostNegative, similarities[0].score)

  def test_CountNo2(self):
    rule = XTRule('q1',
                  tree_or_string('(ROOT (SBARQ (WHNP (WHADJP (WRB how) (JJ much)) ?x0|NNS) ?x1|SQ (. ?)))'),
                  tree_or_string('(ID COUNT (ID ?x0| ?x1|))'),
                  {(0, 0) : 'q2', (0, 1) : 'q2'}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    similarities = self.count_op.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertAlmostEqual(self.count_op.kCostNegative, similarities[0].score)

  def test_CountEmpty(self):
    rule = XTRule('q1',
                  tree_or_string('(ROOT (SBARQ (WHNP (WHADJP (WRB how) (JJ much)) ?x0|NNS) ?x1|SQ (. ?)))'),
                  tree_or_string('(ID MAX (ID ?x0| ?x1|))'),
                  {(0, 0) : 'q2', (0, 1) : 'q2'}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    similarities = self.count_op.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertAlmostEqual(self.count_op.kCostNeutral, similarities[0].score)

  def test_CountPredicateTrue(self):
    rule = XTRule('q1',
                  tree_or_string('(ROOT (SBARQ (WHNP (WHADJP (WRB how) (JJ many)) ?xx0|) ?x1|SQ (. ?)))'),
                  tree_or_string('(ID !fb:architecture.building.floors ?x1|)'),
                  {(1,) : 'q2'}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    similarities = self.count_op.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertAlmostEqual(self.count_op.kCostPositive, similarities[0].score)

  def test_CountPredicateNumberTrue(self):
    rule = XTRule('q1',
                  tree_or_string('(ROOT (SBARQ (WHNP (WHADJP (WRB how) (JJ many)) ?x0|) ?x1|SQ (. ?)))'),
                  tree_or_string('(ID !fb:measurement_unit.dated_integer.number (ID ?x0| ?x1|))'),
                  {(0, 0): 'q2', (0, 1) : 'q2'}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    similarities = self.count_op.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertAlmostEqual(self.count_op.kCostPositive, similarities[0].score)

  def test_CountPredicateFalse(self):
    rule = XTRule('q1',
                  tree_or_string('(ROOT (SBARQ (WHNP (WHADJP (WRB how) (JJ many)) ?xx0|) ?x1|SQ (. ?)))'),
                  tree_or_string('(ID !fb:education.academic_post.institution ?x1|)'),
                  {(1,) : 'q2'}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    similarities = self.count_op.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertAlmostEqual(self.count_op.kCostNegative, similarities[0].score)

  def test_CountPredicateHowMuchFalse(self):
    rule = XTRule('q1',
                  tree_or_string('(ROOT (SBARQ (WHNP (WHADJP (WRB how) (JJ much)) ?xx0|) ?x1|SQ (. ?)))'),
                  tree_or_string('(ID !fb:architecture.building.floors ?x1|)'),
                  {(1,) : 'q2'}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    similarities = self.count_op.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertAlmostEqual(self.count_op.kCostNegative, similarities[0].score)

  def test_CountPredicateReverseFalse(self):
    rule = XTRule('q1',
                  tree_or_string('(ROOT (SBARQ (WHNP (WHADJP (WRB how) (JJ many)) ?xx0|) ?x1|SQ (. ?)))'),
                  tree_or_string('(ID fb:architecture.building.floors ?x1|)'),
                  {(1,) : 'q2'}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    similarities = self.count_op.GetSimilarity(src_treep, trg_treep)
    self.assertEqual(1, len(similarities))
    self.assertAlmostEqual(self.count_op.kCostNegative, similarities[0].score)

class EntityLinkingTestCase(unittest.TestCase):
  """
  Here we document the entity linking capabilities of the Solr index.
  """
  def setUp(self):
    self.cache_filename = '.entity_linking_cache_test'
    self.entity_linking_cost = EntityLinkingCost(cache_filename=self.cache_filename)
    self.cost = 1.0
    self.relation = 'entity'
    self.maxDiff = None

  def tearDown(self):
    self.entity_linking_cost.Close()
    # os.remove(self.cache_filename)

  def test_EntitySimilarName(self):
    path, subpaths = (), []
    tree_pattern1 = \
      TreePattern(tree_or_string('(NP (NN Albert) (NN Einstein))'), path, subpaths)
    similar = self.entity_linking_cost.GetSimilar(tree_pattern1)

    tree_pattern2 = \
      TreePattern(tree_or_string('fb:en.albert_einstein'), path, subpaths)

    similarities = self.entity_linking_cost.GetSimilarity(tree_pattern1, tree_pattern2)

    cost = 3 * self.cost
    expected_similarities = [Similarity(cost, self.relation, tree_pattern1, tree_pattern2)]
    for expected_similarity in expected_similarities:
      self.assertIn(expected_similarity, similar)
      self.assertIn(expected_similarity, similarities)

  def test_EntitySimilarNameShort(self):
    path, subpaths = (), []
    tree_pattern1 = \
      TreePattern(tree_or_string('(NP (NN Einstein))'), path, subpaths)
    similar = self.entity_linking_cost.GetSimilar(tree_pattern1)

    tree_pattern2 = \
      TreePattern(tree_or_string('fb:en.albert_einstein'), path, subpaths)
    similarities = self.entity_linking_cost.GetSimilarity(tree_pattern1, tree_pattern2)

    cost = 2 * self.cost
    expected_similarities = [Similarity(cost, self.relation, tree_pattern1, tree_pattern2)]
    for expected_similarity in expected_similarities:
      self.assertIn(expected_similarity, similar)
      self.assertIn(expected_similarity, similarities)

  def test_EntitySimilarNameReversed(self):
    path, subpaths = (), []
    tree_pattern1 = \
      TreePattern(tree_or_string('(NP (NN Einstein) (NN Albert))'), path, subpaths)
    similar = self.entity_linking_cost.GetSimilar(tree_pattern1)

    tree_pattern2 = \
      TreePattern(tree_or_string('fb:en.albert_einstein'), path, subpaths)
    similarities = self.entity_linking_cost.GetSimilarity(tree_pattern1, tree_pattern2)

    cost = 3 * self.cost
    expected_similarities = [Similarity(cost, self.relation, tree_pattern1, tree_pattern2)]
    for expected_similarity in expected_similarities:
      self.assertIn(expected_similarity, similar)
      self.assertIn(expected_similarity, similarities)

  def test_EntitySimilarNameReversedPunct(self):
    path, subpaths = (), []
    tree_pattern1 = \
      TreePattern(tree_or_string('(NP (NN Einstein) (PU ,) (NN Albert))'), path, subpaths)
    similar = self.entity_linking_cost.GetSimilar(tree_pattern1)

    tree_pattern2 = \
      TreePattern(tree_or_string('fb:en.albert_einstein'), path, subpaths)
    similarities = self.entity_linking_cost.GetSimilarity(tree_pattern1, tree_pattern2)

    cost = 4 * self.cost
    expected_similarities = [Similarity(cost, self.relation, tree_pattern1, tree_pattern2)]
    for expected_similarity in expected_similarities:
      self.assertIn(expected_similarity, similar)
      self.assertIn(expected_similarity, similarities)

  def test_EntitySimilarNameAbbreviated(self):
    path, subpaths = (), []
    tree_pattern1 = \
      TreePattern(tree_or_string('(NP (NN A) (PU .) (NN Einstein))'), path, subpaths)
    similar = self.entity_linking_cost.GetSimilar(tree_pattern1)

    tree_pattern2 = \
      TreePattern(tree_or_string('fb:en.albert_einstein'), path, subpaths)
    similarities = self.entity_linking_cost.GetSimilarity(tree_pattern1, tree_pattern2)

    cost = 4 * self.cost
    expected_similarities = [Similarity(cost, self.relation, tree_pattern1, tree_pattern2)]
    for expected_similarity in expected_similarities:
      self.assertIn(expected_similarity, similar)
      self.assertIn(expected_similarity, similarities)

  def test_UnaryContinent(self):
    path, subpaths = (), []
    tree_pattern1 = \
      TreePattern(tree_or_string('(NN continent)'), path, subpaths)
    similar = self.entity_linking_cost.GetSimilar(tree_pattern1)

    tree_pattern2 = \
      TreePattern('fb:location.continent', path, subpaths)
    similarities = self.entity_linking_cost.GetSimilarity(tree_pattern1, tree_pattern2)

    cost = 2 * self.cost
    expected_similarities = [Similarity(cost, self.relation, tree_pattern1, tree_pattern2)]
    for expected_similarity in expected_similarities:
      self.assertIn(expected_similarity, similar)
      self.assertIn(expected_similarity, similarities)

  def test_UnaryLawyer(self):
    path, subpaths = (), []
    tree_pattern1 = \
      TreePattern(tree_or_string('(NN lawyer)'), path, subpaths)
    similar = self.entity_linking_cost.GetSimilar(tree_pattern1)

    tree_pattern2 = \
      TreePattern('fb:en.criminal_defense_lawyer', path, subpaths)
    tree_pattern3 = \
      TreePattern('fb:en.attorney', path, subpaths)
    similarities = \
      self.entity_linking_cost.GetSimilarity(tree_pattern1, tree_pattern2) + \
      self.entity_linking_cost.GetSimilarity(tree_pattern1, tree_pattern3)

    cost = 2 * self.cost
    expected_similarities = [Similarity(cost, self.relation, tree_pattern1, tree_pattern2),
                             Similarity(cost, self.relation, tree_pattern1, tree_pattern3)]
    for expected_similarity in expected_similarities:
      self.assertIn(expected_similarity, similar)
      self.assertIn(expected_similarity, similarities)

class PredicateLinkingTestCase(unittest.TestCase):
  """
  Here we document the predicate linking capabilities of the Solr index.
  """
  def setUp(self):
    self.cache_filename = '.predicate_linking_cache_test'
    self.predicate_linking_cost = PredicateLinkingCost(cache_filename=self.cache_filename)
    self.cost = 1.0
    self.relation = 'predicate'
    # self.maxDiff = 10

  def tearDown(self):
    self.predicate_linking_cost.Close()
    # os.remove(self.cache_filename)

  @unittest.expectedFailure
  def test_BearIn(self):
    path, subpaths = (), []
    tree_pattern1 = \
      TreePattern(tree_or_string('(VP (VB bear) (P in))'), path, subpaths)
    similar = self.predicate_linking_cost.GetSimilar(tree_pattern1)

    tree_pattern2 = \
      TreePattern(tree_or_string('!fb:location.location.people_born_here'), path, subpaths)
    similarities = self.predicate_linking_cost.GetSimilarity(tree_pattern1, tree_pattern2)

    cost = 3 * self.cost
    expected_similarities = [Similarity(cost, self.relation, tree_pattern1, tree_pattern2)]
    for expected_similarity in expected_similarities:
      self.assertIn(expected_similarity, similar)
      self.assertIn(expected_similarity, similarities)

  def test_BornInHere(self):
    path, subpaths = (), []
    tree_pattern1 = \
      TreePattern(tree_or_string('(VP (VB born) (P in))'), path, subpaths)
    similar = self.predicate_linking_cost.GetSimilar(tree_pattern1)

    tree_pattern2 = \
      TreePattern(tree_or_string('!fb:location.location.people_born_here'), path, subpaths)
    similarities = \
      self.predicate_linking_cost.GetSimilarity(tree_pattern1, tree_pattern2)

    cost = 3 * self.cost
    expected_similarities = [Similarity(cost, self.relation, tree_pattern1, tree_pattern2)]
    for expected_similarity in expected_similarities:
      self.assertIn(expected_similarity, similar)
      self.assertIn(expected_similarity, similarities)

  @unittest.expectedFailure
  def test_BornInOrigin(self):
    path, subpaths = (), []
    tree_pattern1 = \
      TreePattern(tree_or_string('(VP (VB born) (P in))'), path, subpaths)
    similar = self.predicate_linking_cost.GetSimilar(tree_pattern1)

    tree_pattern2 = \
      TreePattern(tree_or_string('fb:music.artist.origin'), path, subpaths)
    similarities = \
      self.predicate_linking_cost.GetSimilarity(tree_pattern1, tree_pattern2)

    cost = 3 * self.cost
    expected_similarities = [Similarity(cost, self.relation, tree_pattern1, tree_pattern2)]
    for expected_similarity in expected_similarities:
      self.assertIn(expected_similarity, similar)
      self.assertIn(expected_similarity, similarities)

  def test_IssueNumber(self):
    path, subpaths = (), []
    tree_pattern1 = TreePattern(tree_or_string('(NN issue)'), path, subpaths)
    similar = self.predicate_linking_cost.GetSimilar(tree_pattern1)

    tree_pattern2 = \
      TreePattern(tree_or_string('!fb:comic_books.comic_book_issue.issue_number'), path, subpaths)
    similarities = self.predicate_linking_cost.GetSimilarity(tree_pattern1, tree_pattern2)

    cost = 2 * self.cost
    expected_similarities = [Similarity(cost, self.relation, tree_pattern1, tree_pattern2)]
    for expected_similarity in expected_similarities:
      self.assertIn(expected_similarity, similar)
      self.assertIn(expected_similarity, similarities)

  @unittest.expectedFailure
  def test_InstitutionPlural(self):
    path, subpaths = (), []
    tree_pattern1 = TreePattern(tree_or_string('institutions'), path, subpaths)
    similar = self.predicate_linking_cost.GetSimilar(tree_pattern1)

    tree_pattern2 = \
      TreePattern(tree_or_string('!fb:education.academic_post.institution'),
                  path, subpaths)
    similarities = self.predicate_linking_cost.GetSimilarity(tree_pattern1, tree_pattern2)

    cost = 2 * self.cost
    expected_similarities = [Similarity(cost, self.relation, tree_pattern1, tree_pattern2)]
    for expected_similarity in expected_similarities:
      self.assertIn(expected_similarity, similar)
      self.assertIn(expected_similarity, similarities)

  def test_Institution(self):
    path, subpaths = (), []
    tree_pattern1 = TreePattern(tree_or_string('institution'), path, subpaths)
    similar = self.predicate_linking_cost.GetSimilar(tree_pattern1)

    tree_pattern2 = \
      TreePattern(tree_or_string('fb:education.academic_post.institution'),
                  path, subpaths)
    similarities = self.predicate_linking_cost.GetSimilarity(tree_pattern1, tree_pattern2)

    cost = 2 * self.cost
    expected_similarities = [Similarity(cost, self.relation, tree_pattern1, tree_pattern2)]
    for expected_similarity in expected_similarities:
      self.assertIn(expected_similarity, similar)
      self.assertIn(expected_similarity, similarities)

  @unittest.expectedFailure
  def test_StopRunning(self):
    path, subpaths = (), []
    tree_pattern1 = TreePattern(tree_or_string('(VP (VB stop) (S (VP (VBG running))))'), path, subpaths)
    similar = self.predicate_linking_cost.GetSimilar(tree_pattern1)

    tree_pattern2 = \
      TreePattern(tree_or_string('!fb:tv.tv_program.air_date_of_final_episode'),
                  path, subpaths)
    similarities = self.predicate_linking_cost.GetSimilarity(tree_pattern1, tree_pattern2)

    cost = 3 * self.cost
    expected_similarities = [Similarity(cost, self.relation, tree_pattern1, tree_pattern2)]
    for expected_similarity in expected_similarities:
      self.assertIn(expected_similarity, similar)
      self.assertIn(expected_similarity, similarities)

  @unittest.expectedFailure
  def test_WithVariable(self):
    path, subpaths = (), [(1,1,1)]
    tree_pattern1 = TreePattern(
      tree_or_string('(SQ (NP (NNS companies)) (VP (VBP are) (VP (VBN traded) ?x1|PP)))'),
      path, subpaths)
    similar = self.predicate_linking_cost.GetSimilar(tree_pattern1)

    path, subpaths = (), [(1,)]
    tree_pattern2 = \
      TreePattern(tree_or_string('(ID !fb:finance.stock_exchange.companies_traded ?x1|)'),
                  path, subpaths)
    similarities = self.predicate_linking_cost.GetSimilarity(tree_pattern1, tree_pattern2)

    cost = 4 * self.cost
    expected_similarities = [Similarity(cost, self.relation, tree_pattern1, tree_pattern2)]
    for expected_similarity in expected_similarities:
      self.assertIn(expected_similarity, similar)
      self.assertIn(expected_similarity, similarities)

class BridgeLinkingTestCase(unittest.TestCase):
  """
  Here we document the bridge entity linking capabilities of the Solr index.
  """
  def setUp(self):
    self.cache_filename = '.bridge_linking_cache_test'
    self.bridge_linking_cost = BridgeLinkingCost(cache_filename=self.cache_filename)
    self.cost = 1.0
    self.extra_cost = 3.0
    self.relation = 'bridge_entity'

  def tearDown(self):
    self.bridge_linking_cost.Close()
    # os.remove(self.cache_filename)

  def test_MarshallHall(self):
    path, subpaths = (), []
    tree_pattern1 = \
      TreePattern(tree_or_string('(NP (NNP marshall) (NNP hall))'), path, subpaths)
    similar = self.bridge_linking_cost.GetSimilar(tree_pattern1)

    tree_pattern2 = \
      TreePattern(tree_or_string('(ID fb:education.academic_post.person fb:en.marshall_hall)'),
                  path, subpaths)
    similarities = self.bridge_linking_cost.GetSimilarity(tree_pattern1, tree_pattern2)

    tree_pattern3 = \
      TreePattern(tree_or_string('(ID [] fb:en.marshall_hall)'),
                  path, subpaths)
    cost = 3 * self.cost + 1 * self.extra_cost
    expected_similar = [Similarity(cost, self.relation, tree_pattern1, tree_pattern3)]
    expected_similarities = [Similarity(cost, self.relation, tree_pattern1, tree_pattern2)]
    for expected_similarity in expected_similar:
      self.assertIn(expected_similarity, similar)
    for expected_similarity in expected_similarities:
      self.assertIn(expected_similarity, similarities)

  def test_Professor(self):
    path, subpaths = (), []
    tree_pattern1 = \
      TreePattern(tree_or_string('(NP (DT a) (NN professor))'), path, subpaths)
    similar = self.bridge_linking_cost.GetSimilar(tree_pattern1)

    tree_pattern2 = \
      TreePattern(tree_or_string('(ID fb:education.academic_post.position_or_title fb:en.professor)'),
                  path, subpaths)
    similarities = self.bridge_linking_cost.GetSimilarity(tree_pattern1, tree_pattern2)

    tree_pattern3 = \
      TreePattern(tree_or_string('(ID [] fb:en.professor)'),
                  path, subpaths)
    cost = 3 * self.cost + 1 * self.extra_cost
    expected_similar = [Similarity(cost, self.relation, tree_pattern1, tree_pattern3)]
    expected_similarities = [Similarity(cost, self.relation, tree_pattern1, tree_pattern2)]
    for expected_similarity in expected_similar:
      self.assertIn(expected_similarity, similar)
    for expected_similarity in expected_similarities:
      self.assertIn(expected_similarity, similarities)

if __name__ == '__main__':
  suite1 = unittest.TestLoader().loadTestsFromTestCase(NoSimilarityQATestCase)
  suite2 = unittest.TestLoader().loadTestsFromTestCase(CountOpTestCase)
  suite3 = unittest.TestLoader().loadTestsFromTestCase(EntityLinkingTestCase)
  suite4 = unittest.TestLoader().loadTestsFromTestCase(PredicateLinkingTestCase)
  suite5 = unittest.TestLoader().loadTestsFromTestCase(BridgeLinkingTestCase)
  suite6 = unittest.TestLoader().loadTestsFromTestCase(NounPhraseTestCase)
  suite7 = unittest.TestLoader().loadTestsFromTestCase(UriSurfTestCase)
  suites = unittest.TestSuite([suite1, suite2, suite3, suite4, suite5, suite6, suite7])
  unittest.TextTestRunner(verbosity=2).run(suites)

