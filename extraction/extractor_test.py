import unittest

from itertools import product, chain

from extraction.extractor_beam import RuleExtractor, Transformation
from extraction.extractor import (GetDisjointPaths, ObtainTreePattern)
from linguistics.similarity import Similarity, SimilarityScorerEnsemble
from linguistics.similarity_costs import (LexicalSimilarity, NoSimilarity,
  LeafSimilarity, NodesDifference, TreeComplexity)
from linguistics.similarity_ordering import OrderDifference
from linguistics.similarity_costs_deps import (LexicalSimilarityDeps, NoSimilarityDeps,
                                               LeafSimilarityDeps)
from training.transductionrule import XTRule
from utils.tree_tools import Tree, tree_or_string

## Using cython:
from utils.cutils import AreDisjointPaths, GetCommonParentsAt
## Not using cython:
# from extraction.extractor import AreDisjointPaths

class TransformationTestCase(unittest.TestCase):
  def setUp(self):
    self.src_path = (0, 0)
    self.trg_path = (0, 1)
    self.src_subpaths = [(0, 0, 0), (0, 0, 1)]
    self.trg_subpaths = [(0, 1, 0), (0, 1, 1)]
    self.similarity = Similarity(1.0, 'dummy', None, None)
    self.trans1 = Transformation(self.src_path, self.trg_path, self.src_subpaths,
      self.trg_subpaths, self.similarity)
    self.trans2 = Transformation(self.trg_path, self.src_path, self.src_subpaths,
      self.trg_subpaths, self.similarity)
    self.trans3 = Transformation(self.src_path, self.trg_path, self.src_subpaths,
      self.trg_subpaths, self.similarity)

  def test_Equality(self):
    self.assertEqual(self.trans1, self.trans3)

  def test_Inequality(self):
    self.assertNotEqual(self.trans1, self.trans2)

  def test_Pertains(self):
    transformations = [self.trans1, self.trans2]
    self.assertTrue(self.trans3 in transformations)

  def test_NotPertains(self):
    transformations = [self.trans1, self.trans3]
    self.assertFalse(self.trans2 in transformations)

  def test_SetPertains(self):
    transformations = [self.trans1, self.trans3]
    self.assertEqual(2, len(transformations))
    self.assertEqual(1, len(set(transformations)))

  def test_SetNotPertains(self):
    transformations = [self.trans1, self.trans2]
    self.assertEqual(2, len(transformations))
    self.assertEqual(2, len(set(transformations)))

  def test_SetRemoveSameElement(self):
    set_transformations = set([self.trans1, self.trans2])
    self.assertEqual(2, len(set_transformations))
    set_transformations.remove(self.trans1)
    self.assertEqual(1, len(set_transformations))
    set_transformations.add(self.trans1)
    self.assertEqual(2, len(set_transformations))
    set_transformations.remove(self.trans3)
    self.assertEqual(1, len(set_transformations))

class ExtractRulesDepsTestCase(unittest.TestCase):
  def setUp(self):
    self.similarity_scorer = \
      SimilarityScorerEnsemble([LexicalSimilarityDeps(), NoSimilarityDeps()],
                               [OrderDifference()])
    self.similarity_score_guesser = \
      SimilarityScorerEnsemble([LeafSimilarityDeps()])
    self.options = {'similarity_scorer' : self.similarity_scorer,
                    'similarity_score_guesser' : self.similarity_score_guesser,
                    'cached_extractors' : {},
                    'max_running_time' : 3000,
                    'initial_state' : 'q0'}

  @unittest.expectedFailure
  def test_TerminalToTerminalDifferent(self):
    tree1 = tree_or_string('the')
    tree2 = tree_or_string('la')
    path1 = ()
    path2 = ()
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    derivations = rule_extractor.ObtainBestDerivations(1)
    expected_derivations = [[XTRule('t', 'the', 'la', {}, 1.0)]]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|'),
                           tree_or_string('?x0|'),
                           {() : 't'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.assertItemsEqual(expected_derivations, derivations)

  @unittest.expectedFailure
  def test_TerminalToTerminalEqual(self):
    tree1 = tree_or_string('the')
    tree2 = tree_or_string('the')
    path1 = ()
    path2 = ()
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    derivations = rule_extractor.ObtainBestDerivations(1)
    expected_derivations = [XTRule('copy', 'the', 'the', {}, 0.5)]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|'),
                           tree_or_string('?x0|'),
                           {() : 'copy'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.assertItemsEqual(expected_derivations, derivations)

  @unittest.expectedFailure
  def test_TerminalToTerminalSimilar(self):
    tree1 = tree_or_string('italian')
    tree2 = tree_or_string('european')
    path1 = ()
    path2 = ()
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    derivations = rule_extractor.ObtainBestDerivations(1)
    expected_derivations = [[XTRule('hypernym', 'italian', 'european', {}, 0.5)]]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|'),
                           tree_or_string('?x0|'),
                           {() : 'hypernym'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.assertItemsEqual(expected_derivations, derivations)

  def test_PreterminalToTerminalDifferent(self):
    tree1 = tree_or_string('(italian the)')
    tree2 = tree_or_string('la')
    path1 = ()
    path2 = ()
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    derivations = rule_extractor.ObtainBestDerivations(1)
    expected_derivation1 = [XTRule(self.options['initial_state'],
                            tree_or_string('?x0|italian'),
                            tree_or_string('?x0|'),
                            {() : 'q0'}, 0.0),
                            XTRule('q0', tree_or_string('(italian ?x0|)'),
                                         tree_or_string('?x0|'),
                                   {() : 't'}, 1.0),
                            XTRule('t', tree_or_string('the'),
                                        tree_or_string('la'), {}, 1.0)]
    expected_derivation2 = [XTRule(self.options['initial_state'],
                            tree_or_string('?x0|italian'),
                            tree_or_string('?x0|'),
                            {() : 't'}, 0.0),
                            XTRule('t', tree_or_string('(italian the)'),
                                         tree_or_string('la'),
                                   {}, 2.0)]
    expected_derivations = [expected_derivation1, expected_derivation2]
    for derivation in derivations:
      self.assertIn(derivation, expected_derivations)

  def test_PreterminalToTerminalEqual1(self):
    tree1 = tree_or_string('(italian the)')
    tree2 = tree_or_string('the')
    path1 = ()
    path2 = ()
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    derivations = rule_extractor.ObtainBestDerivations(1)
    expected_derivations = [[XTRule('q0', tree_or_string('(italian ?x0|)'),
                                        tree_or_string('?x0|'),
                                  {() : 'copy'}, 1.0),
                           XTRule('copy', tree_or_string('the'),
                                          tree_or_string('the'),
                                  {}, 0.5)]]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|italian'),
                           tree_or_string('?x0|'),
                           {() : 'q0'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.assertItemsEqual(expected_derivations, derivations)

  def test_PreterminalToTerminalSimilar1(self):
    tree1 = tree_or_string('(italian smart)')
    tree2 = tree_or_string('bright')
    path1 = ()
    path2 = ()
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    derivations = rule_extractor.ObtainBestDerivations(1)
    expected_derivations = [[XTRule('q0', tree_or_string('(italian ?x0|)'),
                                        tree_or_string('?x0|'),
                                  {() : 'synonym'}, 1.0),
                           XTRule('synonym', tree_or_string('smart'),
                                             tree_or_string('bright'),
                                  {}, 0.5)]]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|italian'),
                           tree_or_string('?x0|'),
                           {() : 'q0'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.assertItemsEqual(expected_derivations, derivations)

  # With the current rule extraction algorithm, we do not allow rule extractions
  # that satisfy the test below. The reason for the failure is that we need to
  # insert a variable at a node, and still consider the children of such node.
  # It seems to me (pascual@) that a modificaton of the rule extractor that
  # satisfies this test would be not trivial.
  @unittest.expectedFailure
  def test_PreterminalToTerminalEqual2(self):
    tree1 = tree_or_string('(italian the)')
    tree2 = tree_or_string('italian')
    path1 = ()
    path2 = ()
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    derivations = rule_extractor.ObtainBestDerivations(1)
    expected_derivations = [[XTRule('q0', tree_or_string('(?x0| the)'),
                                        tree_or_string('?x0|'),
                                  {() : 'copy'}, 1.0),
                           XTRule('copy', tree_or_string('italian'),
                                          tree_or_string('italian'),
                                  {}, 0.5)]]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|italian'),
                           tree_or_string('?x0|'),
                           {() : 'q0'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.assertItemsEqual(expected_derivations, derivations)

  def test_PreterminalToPreterminalEqualBoth(self):
    tree1 = tree_or_string('(italian the)')
    tree2 = tree_or_string('(italian the)')
    path1 = ()
    path2 = ()
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    derivations = rule_extractor.ObtainBestDerivations(1)
    expected_derivations = [[XTRule('copy', tree_or_string('(italian ?x0|)'),
                                          tree_or_string('(italian ?x0|)'),
                                  {(0,) : 'copy'}, 0.5),
                           XTRule('copy', tree_or_string('the'),
                                          tree_or_string('the'),
                                  {}, 0.5)]]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|italian'),
                           tree_or_string('?x0|italian'),
                           {() : 'copy'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.assertItemsEqual(expected_derivations, derivations)

  def test_PreterminalToPreterminalEqualFirst(self):
    tree1 = tree_or_string('(italian the)')
    tree2 = tree_or_string('(italian la)')
    path1 = ()
    path2 = ()
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    derivations = rule_extractor.ObtainBestDerivations(1)
    expected_derivations = [[XTRule('copy', tree_or_string('(italian ?x0|)'),
                                          tree_or_string('(italian ?x0|)'),
                                  {(0,) : 't'}, 0.5),
                           XTRule('t', tree_or_string('the'),
                                       tree_or_string('la'),
                                  {}, 1.0)]]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|italian'),
                           tree_or_string('?x0|italian'),
                           {() : 'copy'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.assertItemsEqual(expected_derivations, derivations)

  def test_PreterminalToPreterminalEqualSecond(self):
    tree1 = tree_or_string('(italian smart)')
    tree2 = tree_or_string('(french smart)')
    path1 = ()
    path2 = ()
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    derivations = rule_extractor.ObtainBestDerivations(1)
    expected_derivations = [[XTRule('q0', tree_or_string('(italian ?x0|)'),
                                        tree_or_string('(french ?x0|)'),
                                  {(0,) : 'copy'}, 1.0),
                           XTRule('copy', tree_or_string('smart'),
                                          tree_or_string('smart'),
                                  {}, 0.5)]]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|italian'),
                           tree_or_string('?x0|french'),
                           {() : 'q0'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.assertItemsEqual(expected_derivations, derivations)

  def test_PreterminalToPreterminalEqualFirstSimilarSecond(self):
    tree1 = tree_or_string('(italian smart)')
    tree2 = tree_or_string('(italian bright)')
    path1 = ()
    path2 = ()
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    derivations = rule_extractor.ObtainBestDerivations(1)
    expected_derivations = [[XTRule('copy', tree_or_string('(italian ?x0|)'),
                                          tree_or_string('(italian ?x0|)'),
                                  {(0,) : 'synonym'}, 0.5),
                           XTRule('synonym', tree_or_string('smart'),
                                             tree_or_string('bright'),
                                  {}, 0.5)]]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|italian'),
                           tree_or_string('?x0|italian'),
                           {() : 'copy'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.assertItemsEqual(expected_derivations, derivations)

  def test_PreterminalToPreterminalEqualSecondSimilarFirst(self):
    tree1 = tree_or_string('(italian smart)')
    tree2 = tree_or_string('(european smart)')
    path1 = ()
    path2 = ()
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    derivations = rule_extractor.ObtainBestDerivations(1)
    expected_derivations = [[XTRule('hypernym', tree_or_string('(italian ?x0|)'),
                                              tree_or_string('(european ?x0|)'),
                                  {(0,) : 'copy'}, 0.5),
                           XTRule('copy', tree_or_string('smart'),
                                          tree_or_string('smart'),
                                  {}, 0.5)]]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|italian'),
                           tree_or_string('?x0|european'),
                           {() : 'hypernym'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.assertItemsEqual(expected_derivations, derivations)

  def test_PreterminalToPreterminalSimilarBoth(self):
    tree1 = tree_or_string('(italian smart)')
    tree2 = tree_or_string('(european bright)')
    path1 = ()
    path2 = ()
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    derivations = rule_extractor.ObtainBestDerivations(1)
    expected_derivations = [[XTRule('hypernym', tree_or_string('(italian ?x0|)'),
                                              tree_or_string('(european ?x0|)'),
                                  {(0,) : 'synonym'}, 0.5),
                           XTRule('synonym', tree_or_string('smart'),
                                             tree_or_string('bright'),
                                  {}, 0.5)]]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|italian'),
                           tree_or_string('?x0|european'),
                           {() : 'hypernym'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.assertItemsEqual(expected_derivations, derivations)

  # This is another case (same phylosophy) where the rule extractor shows no
  # capability to extract appropriate rules.
  @unittest.expectedFailure
  def test_NonterminalToNonterminal(self):
    tree1 = tree_or_string('(eat italian smart)')
    tree2 = tree_or_string('(european bright)')
    path1 = ()
    path2 = ()
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    derivations = rule_extractor.ObtainBestDerivations(1)
    expected_derivations = [[XTRule('q0', tree_or_string('(eat ?x0| ?x1|)'),
                                        tree_or_string('(?x0| ?x1|)'),
                                  {() : 'hypernym', (0,) : 'synonym'}, 1.0),
                           XTRule('hypernym', tree_or_string('italian'),
                                              tree_or_string('european'),
                                  {}, 0.5),
                           XTRule('synonym', tree_or_string('smart'),
                                             tree_or_string('bright'),
                                  {}, 0.5)]]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|eat'),
                           tree_or_string('?x0|european'),
                           {() : 'q0'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.assertItemsEqual(expected_derivations, derivations)

class ExtractRulesTestCase(unittest.TestCase):
  def setUp(self):
    self.lexical_similarity = LexicalSimilarity()
    self.lexical_similarity.kLinguisticVariation = 0.5
    self.lexical_similarity.feature_weight = 1.0
    self.similarity_scorer = \
      SimilarityScorerEnsemble([self.lexical_similarity, NoSimilarity()],
                               [OrderDifference()])
    self.similarity_score_guesser = \
      SimilarityScorerEnsemble([LeafSimilarity()])
    self.options = {'similarity_scorer' : self.similarity_scorer,
                    'similarity_score_guesser' : self.similarity_score_guesser,
                    'cached_extractors' : {},
                    'max_running_time' : 3000,
                    'initial_state' : 'start',
                    'beam_size' : 5}

  @unittest.expectedFailure
  def test_TerminalToPreterminalEpsilonSource(self):
    tree1 = tree_or_string('the')
    tree2 = tree_or_string('(DT the)')
    path1 = ()
    path2 = ()
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    derivations = rule_extractor.ObtainBestDerivations(1)
    expected_derivations = [[XTRule('q0', tree_or_string('?x0|'),
                                       tree_or_string('(DT ?x0|)'),
                                  {(0,) : 'copy'}, 0.0),
                           XTRule('copy', tree_or_string('the'),
                                          tree_or_string('the'),
                                  {}, 0.5)]]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|'),
                           tree_or_string('?x0|DT'),
                           {() : 'q0'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.assertItemsEqual(expected_derivations, derivations)

  @unittest.expectedFailure
  def test_TerminalToPreterminalEpsilonTarget(self):
    tree1 = tree_or_string('(DT the)')
    tree2 = tree_or_string('the')
    path1 = ()
    path2 = ()
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    derivations = rule_extractor.ObtainBestDerivations(1)
    expected_derivations = [[XTRule('q0', tree_or_string('(DT ?x0|)'),
                                       tree_or_string('?x0|'),
                                  {() : 'copy'}, 0.0),
                           XTRule('copy', tree_or_string('the'),
                                          tree_or_string('the'),
                                  {}, 0.5)]]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|'),
                           tree_or_string('?x0|DT'),
                           {() : 'q0'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.assertItemsEqual(expected_derivations, derivations)

  @unittest.expectedFailure
  def test_TerminalToTerminalDifferent(self):
    tree1 = tree_or_string('the')
    tree2 = tree_or_string('la')
    path1 = ()
    path2 = ()
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    derivations = rule_extractor.ObtainBestDerivations(1)
    expected_derivations = [[XTRule('t', 'the', 'la', {}, 1.0)]]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|'),
                           tree_or_string('?x0|'),
                           {() : 't'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.assertItemsEqual([expected_derivation], derivation)

  @unittest.expectedFailure
  def test_TerminalToTerminalEqual(self):
    tree1 = tree_or_string('the')
    tree2 = tree_or_string('the')
    path1 = ()
    path2 = ()
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    derivations = rule_extractor.ObtainBestDerivations(1)
    expected_derivations = [[XTRule('copy', 'the', 'the', {}, 0.5)]]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|'),
                           tree_or_string('?x0|'),
                           {() : 'copy'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.assertItemsEqual([expected_derivation], derivation)

  @unittest.expectedFailure
  def test_TerminalToTerminalSimilar(self):
    tree1 = tree_or_string('italian')
    tree2 = tree_or_string('european')
    path1 = ()
    path2 = ()
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    derivations = rule_extractor.ObtainBestDerivations(1)
    expected_derivations = [[XTRule('hypernym', 'italian', 'european', {}, 0.5)]]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|'),
                           tree_or_string('?x0|'),
                           {() : 'hypernym'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.assertItemsEqual([expected_derivation], derivation)
    self.assertEqual(len(expected_derivation), len(derivation))

  @unittest.expectedFailure
  def test_TerminalToPreterminalEqual(self):
    tree1 = tree_or_string('the')
    tree2 = tree_or_string('(DT the)')
    path1 = ()
    path2 = ()
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    derivations = rule_extractor.ObtainBestDerivations(1)
    expected_derivations = [[XTRule('q0', tree_or_string('?x0|'),
                                         tree_or_string('(DT ?x0|)'),
                                   {(0,) : 'copy'}, 0.0),
                            XTRule('copy', tree_or_string('the'),
                                           tree_or_string('the'),
                                   {}, 0.5)]]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|'),
                           tree_or_string('?x0|DT'),
                           {() : 'q0'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    for derivation in derivations:
      self.assertIn(derivation, expected_derivations)

  @unittest.expectedFailure
  def test_TerminalToNonterminalEqual(self):
    tree1 = tree_or_string('the')
    tree2 = tree_or_string('(NP (DT the) (NN house))')
    path1 = ()
    path2 = ()
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    derivations = rule_extractor.ObtainBestDerivations(1)
    expected_derivations = [[XTRule('q0', tree_or_string('?x0|'),
                                        tree_or_string('(NP (DT ?x0|) (NN house))'), \
                                   {(0,0) : 'copy'}, 1.0),
                            XTRule('copy', tree_or_string('the'),
                                           tree_or_string('the'),
                                   {}, 0.5)]]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|'),
                           tree_or_string('?x0|NP'),
                           {() : 'q0'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    for derivation in derivations:
      self.assertIn(derivation, expected_derivations)

  def test_PreterminalToTerminalEqual(self):
    tree1 = tree_or_string('(JJ the)')
    tree2 = tree_or_string('the')
    path1 = ()
    path2 = ()
    n_best = 1
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    # from pudb import set_trace; set_trace()
    derivations = rule_extractor.ObtainBestDerivations(n_best)
    expected_derivations = [[XTRule('q0', Tree.fromstring('(JJ ?x0|)'), '?x0|', \
                                  {() : 'copy'}, 0.0),\
                           XTRule('copy', 'the', 'the', {}, 0.5)]]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|JJ'),
                           tree_or_string('?x0|'),
                           {() : 'q0'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.maxDiff = None
    self.assertEqual(n_best, len(derivations))
    self.assertItemsEqual(expected_derivations, derivations)

  def test_PreterminalToTerminalSimilar(self):
    tree1 = tree_or_string('(JJ italian)')
    tree2 = tree_or_string('european')
    path1 = ()
    path2 = ()
    n_best = 1
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    derivations = rule_extractor.ObtainBestDerivations(n_best)
    expected_derivations = [[XTRule('q0', Tree.fromstring('(JJ ?x0|)'), '?x0|', \
                                  {() : 'hypernym'}, 0.0),\
                           XTRule('hypernym', 'italian', 'european', {}, 0.5)]]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|JJ'),
                           tree_or_string('?x0|'),
                           {() : 'q0'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.assertEqual(n_best, len(derivations))
    self.assertItemsEqual(expected_derivations, derivations)

  def test_PreterminalToTerminalDifferent(self):
    tree1 = tree_or_string('(DT the)')
    tree2 = tree_or_string('la')
    path1 = ()
    path2 = ()
    n_best = 2
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    derivations = rule_extractor.ObtainBestDerivations(n_best)
    expected_derivation1 = [XTRule('q0', Tree.fromstring('(DT ?x0|)'), '?x0|', \
                                  {() : 't'}, 0.0),\
                           XTRule('t', 'the', 'la', {}, 1.0)]
    expected_derivation2 = [XTRule('t', Tree.fromstring('(DT the)'), 'la', {}, 1.0)]
    initial_rule1 = [XTRule(self.options['initial_state'],
                            tree_or_string('?x0|DT'),
                            tree_or_string('?x0|'),
                            {() : 'q0'}, 0.0)]
    initial_rule2 = [XTRule(self.options['initial_state'],
                            tree_or_string('?x0|DT'),
                            tree_or_string('?x0|'),
                            {() : 't'}, 0.0)]
    expected_derivations = [initial_rule1 + expected_derivation1,
                            initial_rule2 + expected_derivation2]
    self.assertEqual(n_best, len(derivations))
    for derivation in derivations:
      self.assertIn(derivation, expected_derivations)
    self.assertItemsEqual(expected_derivations, derivations)

  def test_PreterminalToPreterminalEqual(self):
    tree1 = tree_or_string('(DT the)')
    tree2 = tree_or_string('(DT the)')
    path1 = ()
    path2 = ()
    n_best = 3
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    derivations = rule_extractor.ObtainBestDerivations(n_best)
    expected_derivation1 = [XTRule('q0', Tree.fromstring('(DT ?x0|)'),
                                         Tree.fromstring('(DT ?x0|)'),
                                  {(0,) : 'copy'}, 0.0),
                            XTRule('copy', 'the', 'the', {}, 0.5)]
    expected_derivation2 = [XTRule('q0', Tree.fromstring('(DT ?x0|)'),
                                        tree_or_string('?x0|DT'),
                                  {() : 'q1'}, 0.0), 
                            XTRule('q1', tree_or_string('?x0|'),
                                         Tree.fromstring('(DT ?x0|)'),
                                  {(0,) : 'copy'}, 0.0),
                            XTRule('copy', 'the', 'the', {}, 0.5)]
    expected_derivation3 = [XTRule('q0', tree_or_string('?x0|DT'),
                                         Tree.fromstring('(DT ?x0|)'),
                                  {(0,) : 'q0'}, 0.0),
                            XTRule('q0', Tree.fromstring('(DT ?x0|)'),
                                        tree_or_string('?x0|'),
                                  {() : 'copy'}, 0.0), 
                            XTRule('copy', 'the', 'the', {}, 0.5)]
    expected_derivations = [expected_derivation1, expected_derivation2,
                            expected_derivation3]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|DT'),
                           tree_or_string('?x0|DT'),
                           {() : 'q0'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.assertEqual(n_best, len(derivations))
    for derivation in derivations:
      self.assertIn(derivation, expected_derivations)
    self.assertItemsEqual(expected_derivations, derivations)

  def test_PreterminalToPreterminalSimilar(self):
    tree1 = tree_or_string('(JJ italian)')
    tree2 = tree_or_string('(JJ european)')
    path1 = ()
    path2 = ()
    n_best = 3
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    derivations = rule_extractor.ObtainBestDerivations(n_best)
    expected_derivation1 = [XTRule('q0', tree_or_string('(JJ ?x0|)'),
                                         tree_or_string('(JJ ?x0|)'), \
                                  {(0,) : 'hypernym'}, 0.0),\
                           XTRule('hypernym', tree_or_string('italian'),
                                              tree_or_string('european'),
                                  {}, 0.5)]
    expected_derivation2 = [XTRule('q0', tree_or_string('(JJ ?x0|)'),
                                         tree_or_string('?x0|JJ'), \
                                  {() : 'q1'}, 0.0),\
                           XTRule('q1', tree_or_string('?x0|'),
                                        tree_or_string('(JJ ?x0|)'),
                                  {(0,) : 'hypernym'}, 0.0),
                           XTRule('hypernym', tree_or_string('italian'),
                                              tree_or_string('european'),
                                  {}, 0.5)]
    expected_derivation3 = [XTRule('q0', tree_or_string('?x0|JJ'),
                                        tree_or_string('(JJ ?x0|)'),
                                  {(0,) : 'q0'}, 0.0),
                            XTRule('q0', tree_or_string('(JJ ?x0|)'),
                                         tree_or_string('?x0|'), \
                                  {() : 'hypernym'}, 0.0),\
                           XTRule('hypernym', tree_or_string('italian'),
                                              tree_or_string('european'),
                                  {}, 0.5)]
    expected_derivations = [expected_derivation1, expected_derivation2,
                            expected_derivation3]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|JJ'),
                           tree_or_string('?x0|JJ'),
                           {() : 'q0'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.assertEqual(n_best, len(derivations))
    for derivation in derivations:
      self.assertIn(derivation, expected_derivations)
    self.assertItemsEqual(expected_derivations, derivations)

  def test_PreterminalToNonterminalEqual(self):
    tree1 = tree_or_string('(DT the)')
    tree2 = tree_or_string('(NP (DT the) (JJ italian))')
    path1 = ()
    path2 = ()
    n_best = 8
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    derivations = rule_extractor.ObtainBestDerivations(n_best)
    expected_derivation1 = [XTRule('q0', Tree.fromstring('(DT ?x0|)'), \
                                  Tree.fromstring('(NP (DT ?x0|) (JJ italian))'), \
                                  {(0, 0) : 'copy'}, 1.0),\
                           XTRule('copy', 'the', 'the', {}, 0.5)]
    expected_derivation2 = [XTRule('q0', tree_or_string('(DT ?x0|)'), \
                                         tree_or_string('(NP ?x0|DT (JJ italian))'), \
                                  {(0,) : 'q1'}, 1.0),
                            XTRule('q1', tree_or_string('?x0|'), \
                                         tree_or_string('(DT ?x0|)'), \
                                  {(0,) : 'copy'}, 0.0),
                           XTRule('copy', 'the', 'the', {}, 0.5)]
    expected_derivation3 = [XTRule('q0', tree_or_string('(DT ?x0|)'), \
                                         tree_or_string('?x0|NP'),
                                  {() : 'q1'}, 0.0),
                            XTRule('q1', tree_or_string('?x0|'), \
                                         tree_or_string('(NP (DT ?x0|) (JJ italian))'), \
                                  {(0, 0) : 'copy'}, 1.0),
                           XTRule('copy', 'the', 'the', {}, 0.5)]
    expected_derivation6 = [XTRule('q0', tree_or_string('(DT ?x0|)'),
                                         tree_or_string('?x0|NP'),
                                  {() : 'q1'}, 0.0),
                            XTRule('q1', tree_or_string('?x0|'),
                                         tree_or_string('(NP ?x0|DT (JJ italian))'),
                                  {(0,) : 'q1'}, 1.0),
                            XTRule('q1', tree_or_string('?x0|'),
                                         tree_or_string('(DT ?x0|)'),
                                  {(0,) : 'copy'}, 0.0),
                           XTRule('copy', 'the', 'the', {}, 0.5)]
    expected_derivation4 = [XTRule('q0', tree_or_string('?x0|DT'),
                                         tree_or_string('(NP ?x0|DT (JJ italian))'),
                                  {(0,) : 'q0'}, 1.0),
                            XTRule('q0', tree_or_string('?x0|DT'),
                                         tree_or_string('(DT ?x0|)'),
                                  {(0,) : 'q0'}, 0.0),
                            XTRule('q0', tree_or_string('(DT ?x0|)'),
                                         tree_or_string('?x0|'),
                                  {() : 'copy'}, 0.0),
                           XTRule('copy', 'the', 'the', {}, 0.5)]
    expected_derivation5 = [XTRule('q0', tree_or_string('?x0|DT'),
                                         tree_or_string('(NP ?x0|DT (JJ italian))'),
                                  {(0,) : 'q0'}, 1.0),
                            XTRule('q0', tree_or_string('(DT ?x0|)'),
                                         tree_or_string('(DT ?x0|)'),
                                  {(0,) : 'copy'}, 0.0),
                           XTRule('copy', 'the', 'the', {}, 0.5)]
    expected_derivation7 = [XTRule('q0', tree_or_string('?x0|DT'),
                                         tree_or_string('(NP (DT ?x0|) (JJ italian))'),
                                  {(0, 0) : 'q0'}, 1.0),
                            XTRule('q0', tree_or_string('(DT ?x0|)'),
                                         tree_or_string('?x0|'),
                                  {() : 'copy'}, 0.0),
                           XTRule('copy', 'the', 'the', {}, 0.5)]
    expected_derivation8 = [XTRule('q0', tree_or_string('?x0|DT'),
                                         tree_or_string('(NP ?x0|DT (JJ italian))'),
                                  {(0,) : 'q0'}, 1.0),
                            XTRule('q0', tree_or_string('(DT ?x0|)'),
                                         tree_or_string('?x0|DT'),
                                  {() : 'q1'}, 0.0),
                            XTRule('q1', tree_or_string('?x0|'),
                                         tree_or_string('(DT ?x0|)'),
                                  {(0,) : 'copy'}, 0.0),
                           XTRule('copy', 'the', 'the', {}, 0.5)]
    expected_derivations = [expected_derivation1, expected_derivation2,
                            expected_derivation3, expected_derivation4,
                            expected_derivation5, expected_derivation6,
                            expected_derivation7, expected_derivation8]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|DT'),
                           tree_or_string('?x0|NP'),
                           {() : 'q0'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.assertEqual(n_best, len(derivations))
    for derivation in derivations:
      self.assertIn(derivation, expected_derivations)
    self.assertItemsEqual(expected_derivations, derivations)

  def test_PreterminalToNonterminalSimilar(self):
    tree1 = tree_or_string('(JJ italian)')
    tree2 = tree_or_string('(NP (DT the) (JJ european))')
    path1 = ()
    path2 = ()
    n_best = 8
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    derivations = rule_extractor.ObtainBestDerivations(n_best)
    expected_derivation1 = [XTRule('q0', Tree.fromstring('(JJ ?x0|)'), \
                                  Tree.fromstring('(NP (DT the) (JJ ?x0|))'), \
                                  {(1, 0) : 'hypernym'}, 1.0),\
                            XTRule('hypernym', 'italian', 'european', {}, 0.5)]
    expected_derivation2 = [XTRule('q0', tree_or_string('(JJ ?x0|)'), \
                                         tree_or_string('(NP (DT the) ?x0|JJ)'), \
                                  {(1,) : 'q1'}, 1.0),\
                            XTRule('q1', tree_or_string('?x0|'), \
                                         tree_or_string('(JJ ?x0|)'), \
                                  {(0,) : 'hypernym'}, 0.0),
                            XTRule('hypernym', 'italian', 'european', {}, 0.5)]
    expected_derivation3 = [XTRule('q0', tree_or_string('(JJ ?x0|)'), \
                                         tree_or_string('?x0|NP'), \
                                  {() : 'q1'}, 0.0),\
                            XTRule('q1', tree_or_string('?x0|'), \
                                         tree_or_string('(NP (DT the) (JJ ?x0|))'), \
                                  {(1, 0) : 'hypernym'}, 1.0),
                            XTRule('hypernym', 'italian', 'european', {}, 0.5)]
    expected_derivation4 = [XTRule('q0', tree_or_string('?x0|JJ'), \
                                         tree_or_string('(NP (DT the) (JJ ?x0|))'), \
                                  {(1,0) : 'q0'}, 1.0),\
                            XTRule('q0', tree_or_string('(JJ ?x0|)'), \
                                         tree_or_string('?x0|'), \
                                  {() : 'hypernym'}, 0.0),
                            XTRule('hypernym', 'italian', 'european', {}, 0.5)]
    expected_derivation5 = [XTRule('q0', tree_or_string('(JJ ?x0|)'), \
                                         tree_or_string('?x0|NP'), \
                                  {() : 'q1'}, 0.0),\
                            XTRule('q1', tree_or_string('?x0|'), \
                                         tree_or_string('(NP (DT the) ?x0|JJ)'), \
                                  {(1,) : 'q1'}, 1.0),
                            XTRule('q1', tree_or_string('?x0|'), \
                                         tree_or_string('(JJ ?x0|)'), \
                                  {(0,) : 'hypernym'}, 0.0),
                            XTRule('hypernym', 'italian', 'european', {}, 0.5)]
    expected_derivation6 = [XTRule('q0', tree_or_string('?x0|JJ'), \
                                         tree_or_string('(NP (DT the) ?x0|JJ)'), \
                                  {(1,) : 'q0'}, 1.0),\
                            XTRule('q0', tree_or_string('?x0|JJ'), \
                                         tree_or_string('(JJ ?x0|)'), \
                                  {(0,) : 'q0'}, 0.0),
                            XTRule('q0', tree_or_string('(JJ ?x0|)'), \
                                         tree_or_string('?x0|'), \
                                  {() : 'hypernym'}, 0.0),
                            XTRule('hypernym', 'italian', 'european', {}, 0.5)]
    expected_derivation7 = [XTRule('q0', tree_or_string('?x0|JJ'), \
                                         tree_or_string('(NP (DT the) ?x0|JJ)'), \
                                  {(1,) : 'q0'}, 1.0),\
                            XTRule('q0', tree_or_string('(JJ ?x0|)'), \
                                         tree_or_string('?x0|JJ'), \
                                  {() : 'q1'}, 0.0),
                            XTRule('q1', tree_or_string('?x0|'), \
                                         tree_or_string('(JJ ?x0|)'), \
                                  {(0,) : 'hypernym'}, 0.0),
                            XTRule('hypernym', 'italian', 'european', {}, 0.5)]
    expected_derivation8 = [XTRule('q0', tree_or_string('?x0|JJ'), \
                                         tree_or_string('(NP (DT the) ?x0|JJ)'), \
                                  {(1,) : 'q0'}, 1.0),\
                            XTRule('q0', tree_or_string('(JJ ?x0|)'), \
                                         tree_or_string('(JJ ?x0|)'), \
                                  {(0,) : 'hypernym'}, 0.0),
                            XTRule('hypernym', 'italian', 'european', {}, 0.5)]
    expected_derivations = [expected_derivation1, expected_derivation2,
                            expected_derivation3, expected_derivation4,
                            expected_derivation5, expected_derivation6,
                            expected_derivation7, expected_derivation8]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|JJ'),
                           tree_or_string('?x0|NP'),
                           {() : 'q0'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.assertEqual(n_best, len(derivations))
    for derivation in derivations:
      self.assertIn(derivation, expected_derivations)
    self.assertItemsEqual(expected_derivations, derivations)

  def test_NonterminalToTerminalEqual(self):
    tree1 = tree_or_string('(NP (DT the) (JJ italian))')
    tree2 = tree_or_string('the')
    path1 = ()
    path2 = ()
    n_best = 2
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    derivations = rule_extractor.ObtainBestDerivations(n_best)
    expected_derivation1 = [XTRule('q0', Tree.fromstring('(NP ?x0|DT (JJ italian))'),
                                   tree_or_string('?x0|'), \
                                   {() : 'q1'}, 1.0),\
                            XTRule('q1', Tree.fromstring('(DT ?x0|)'), tree_or_string('?x0|'), \
                                   {() : 'copy'}, 0.0),\
                            XTRule('copy', 'the', 'the', {}, 0.5)]
    # Another alternative with the same cost:
    expected_derivation2 = [XTRule('q0', Tree.fromstring('(NP (DT ?x0|) (JJ italian))'),
                                   tree_or_string('?x0|'), \
                                   {() : 'copy'}, 1.0),\
                            XTRule('copy', 'the', 'the', {}, 0.5)]
    expected_derivations = [expected_derivation1, expected_derivation2]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|NP'),
                           tree_or_string('?x0|'),
                           {() : 'q0'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.assertEqual(n_best, len(derivations))
    for derivation in derivations:
      self.assertIn(derivation, expected_derivations)
    self.assertItemsEqual(expected_derivations, derivations)

  def test_NonterminalToTerminalSimilar(self):
    tree1 = tree_or_string('(NP (DT the) (JJ italian))')
    tree2 = tree_or_string('european')
    path1 = ()
    path2 = ()
    n_best = 2
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, self.options)
    derivations = rule_extractor.ObtainBestDerivations(n_best)
    expected_derivation1 = [XTRule('q0', Tree.fromstring('(NP (DT the) (JJ ?x0|))'),
                                   tree_or_string('?x0|'), \
                                   {() : 'hypernym'}, 1.0),\
                            XTRule('hypernym', 'italian', 'european', {}, 0.5)]
    # Another alternative with the same cost:
    expected_derivation2 = [XTRule('q0', Tree.fromstring('(NP (DT the) ?x0|JJ)'),
                                   tree_or_string('?x0|'), \
                                   {() : 'q1'}, 1.0),\
                            XTRule('q1', Tree.fromstring('(JJ ?x0|)'), tree_or_string('?x0|'), \
                                   {() : 'hypernym'}, 0.0),\
                            XTRule('hypernym', 'italian', 'european', {}, 0.5)]
    expected_derivations = [expected_derivation1, expected_derivation2]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|NP'),
                           tree_or_string('?x0|'),
                           {() : 'q0'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.assertEqual(n_best, len(derivations))
    for derivation in derivations:
      self.assertIn(derivation, expected_derivations)
    self.assertItemsEqual(expected_derivations, derivations)

  def test_NonterminalToPreterminalEqual(self):
    tree1 = tree_or_string('(NP (DT the) (JJ italian))')
    tree2 = tree_or_string('(DT the)')
    path1 = ()
    path2 = ()
    n_best = 2
    similarity_scorer = \
      SimilarityScorerEnsemble([self.lexical_similarity, NoSimilarity()],
        [NodesDifference(), OrderDifference(), TreeComplexity()])
    options = {'similarity_scorer' : similarity_scorer,
               'similarity_score_guesser' : self.similarity_score_guesser,
               'initial_state' : self.options['initial_state']}
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, options)
    # from pudb import set_trace; set_trace()
    derivations = rule_extractor.ObtainBestDerivations(n_best)
    expected_derivation1 = [XTRule('q0', Tree.fromstring('(NP ?x0|DT (JJ italian))'),
                                  tree_or_string('?x0|DT'), \
                                  {() : 'q1'}, 1.14),\
                           XTRule('q1', Tree.fromstring('(DT ?x0|)'), \
                                  tree_or_string('(DT ?x0|)'), \
                                  {(0,) : 'copy'}, 0.0),\
                           XTRule('copy', 'the', 'the', {}, 0.5)]
    expected_derivation2 = [XTRule('q0', Tree.fromstring('(NP (DT ?x0|) (JJ italian))'),
                                  tree_or_string('(DT ?x0|)'), \
                                  {(0,) : 'copy'}, 1.156666),\
                           XTRule('copy', 'the', 'the', {}, 0.5)]
    expected_derivations = [expected_derivation1, expected_derivation2]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|NP'),
                           tree_or_string('?x0|DT'),
                           {() : 'q0'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.assertEqual(n_best, len(derivations))
    for derivation in derivations:
      self.assertIn(derivation, expected_derivations)
    self.assertItemsEqual(expected_derivations, derivations)

  def test_NonterminalToPreterminalSimilar(self):
    tree1 = tree_or_string('(NP (DT the) (JJ italian))')
    tree2 = tree_or_string('(JJ european)')
    path1 = ()
    path2 = ()
    n_best = 2
    similarity_scorer = \
      SimilarityScorerEnsemble([self.lexical_similarity, NoSimilarity()],
        [NodesDifference(), OrderDifference(), TreeComplexity()])
    options = {'similarity_scorer' : similarity_scorer,
               'similarity_score_guesser' : self.similarity_score_guesser,
               'initial_state' : self.options['initial_state']}
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, options)
    derivations = rule_extractor.ObtainBestDerivations(n_best)
    expected_derivation1 = [XTRule('q0', Tree.fromstring('(NP (DT the) ?x0|JJ)'),
                                  tree_or_string('?x0|JJ'), \
                                  {() : 'q1'}, 1.14),\
                           XTRule('q1', Tree.fromstring('(JJ ?x0|)'), \
                                  tree_or_string('(JJ ?x0|)'), \
                                  {(0,) : 'hypernym'}, 0.0),\
                           XTRule('hypernym', 'italian', 'european', {}, 0.5)]
    expected_derivation2 = [XTRule('q0', Tree.fromstring('(NP (DT the) (JJ ?x0|))'),
                                  tree_or_string('(JJ ?x0|)'), \
                                  {(0,) : 'hypernym'}, 1.15666666667),
                           XTRule('hypernym', 'italian', 'european', {}, 0.5)]
    expected_derivations = [expected_derivation1, expected_derivation2]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|NP'),
                           tree_or_string('?x0|JJ'),
                           {() : 'q0'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.assertEqual(n_best, len(derivations))
    for derivation in derivations:
      self.assertIn(derivation, expected_derivations)
    self.assertItemsEqual(expected_derivations, derivations)

  def test_NonterminalToNonterminalEqual(self):
    tree1 = tree_or_string('(NP (DT the) (JJ italian))')
    tree2 = tree_or_string('(NP (DT the) (JJ italian))')
    path1 = ()
    path2 = ()
    n_best = 5
    similarity_scorer = \
      SimilarityScorerEnsemble([self.lexical_similarity, NoSimilarity()],
        [NodesDifference(), OrderDifference(), TreeComplexity()])
    options = {'similarity_scorer' : similarity_scorer,
               'similarity_score_guesser' : self.similarity_score_guesser,
               'initial_state' : self.options['initial_state']}
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, options)
    derivations = rule_extractor.ObtainBestDerivations(n_best)
    expected_derivation1 = [XTRule('q0', Tree.fromstring('(NP ?x0|DT ?x1|JJ)'),
                                  tree_or_string('(NP ?x0|DT ?x1|JJ)'), \
                                  {(0,) : 'q1', (1,) : 'q1'}, 0.0),\
                           XTRule('q1', Tree.fromstring('(DT ?x0|)'), \
                                  tree_or_string('(DT ?x0|)'), \
                                  {(0,) : 'copy'}, 0.0),\
                           XTRule('copy', 'the', 'the', {}, 0.5),\
                           XTRule('q1', Tree.fromstring('(JJ ?x0|)'), \
                                  tree_or_string('(JJ ?x0|)'), \
                                  {(0,) : 'copy'}, 0.0),\
                           XTRule('copy', 'italian', 'italian', {}, 0.5)]
    expected_derivation2 = [XTRule('q0', Tree.fromstring('(NP (DT ?x0|) ?x1|JJ)'),
                                  tree_or_string('(NP (DT ?x0|) ?x1|JJ)'), \
                                  {(0, 0) : 'copy', (1,) : 'q1'}, 0.04),\
                           XTRule('copy', 'the', 'the', {}, 0.5),\
                           XTRule('q1', Tree.fromstring('(JJ ?x0|)'), \
                                  tree_or_string('(JJ ?x0|)'), \
                                  {(0,) : 'copy'}, 0.0),\
                           XTRule('copy', 'italian', 'italian', {}, 0.5)]
    expected_derivation3 = [XTRule('q0', Tree.fromstring('(NP ?x0|DT (JJ ?x1|))'),
                                  tree_or_string('(NP ?x0|DT (JJ ?x1|))'), \
                                  {(0,) : 'q1', (1, 0) : 'copy'}, 0.04),\
                           XTRule('q1', Tree.fromstring('(DT ?x0|)'), \
                                  tree_or_string('(DT ?x0|)'), \
                                  {(0,) : 'copy'}, 0.0),\
                           XTRule('copy', 'the', 'the', {}, 0.5),\
                           XTRule('copy', 'italian', 'italian', {}, 0.5)]
    expected_derivation4 = [XTRule('q0', Tree.fromstring('(NP (DT ?x0|) (JJ ?x1|))'),
                                  tree_or_string('(NP (DT ?x0|) (JJ ?x1|))'), \
                                  {(0, 0) : 'copy', (1, 0) : 'copy'}, 0.16),\
                           XTRule('copy', 'the', 'the', {}, 0.5),\
                           XTRule('copy', 'italian', 'italian', {}, 0.5)]
    expected_derivation5 = [XTRule('q0', Tree.fromstring('(NP (DT ?x0|) ?x1|JJ)'),
                                  tree_or_string('(NP ?x0|DT ?x1|JJ)'), \
                                  {(0,) : 'q2', (1,) : 'q1'}, 0.06),\
                           XTRule('q2', tree_or_string('?x0|'), \
                                  tree_or_string('(DT ?x0|)'), \
                                  {(0,) : 'copy'}, 0.1),\
                           XTRule('copy', 'the', 'the', {}, 0.5),\
                           XTRule('q1', Tree.fromstring('(JJ ?x0|)'), \
                                  tree_or_string('(JJ ?x0|)'), \
                                  {(0,) : 'copy'}, 0.0),\
                           XTRule('copy', 'italian', 'italian', {}, 0.5)]
    expected_derivations = [expected_derivation1, expected_derivation2, \
                            expected_derivation3, expected_derivation4,
                            expected_derivation5]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|NP'),
                           tree_or_string('?x0|NP'),
                           {() : 'q0'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.maxDiff = None
    self.assertEqual(n_best, len(derivations))
    for derivation in derivations:
      self.assertIn(derivation, expected_derivations)
    self.assertItemsEqual(expected_derivations, derivations)

  def test_NonterminalToNonterminalEqualComplexityFeatures(self):
    tree1 = tree_or_string('(NP (DT the) (JJ italian))')
    tree2 = tree_or_string('(NP (DT the) (JJ italian))')
    path1 = ()
    path2 = ()
    n_best = 5
    similarity_scorer = \
      SimilarityScorerEnsemble([self.lexical_similarity, NoSimilarity()],
        [NodesDifference(), OrderDifference(), TreeComplexity()])
    options = {'similarity_scorer' : similarity_scorer,
               'similarity_score_guesser' : self.similarity_score_guesser,
               'initial_state' : self.options['initial_state']}
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, options)
    derivations = rule_extractor.ObtainBestDerivations(n_best)
    expected_derivation1 = [XTRule('q0', Tree.fromstring('(NP ?x0|DT ?x1|JJ)'),
                                  tree_or_string('(NP ?x0|DT ?x1|JJ)'), \
                                  {(0,) : 'q1', (1,) : 'q1'}, 0.0),\
                           XTRule('q1', Tree.fromstring('(DT ?x0|)'), \
                                  tree_or_string('(DT ?x0|)'), \
                                  {(0,) : 'copy'}, 0.0),\
                           XTRule('copy', 'the', 'the', {}, 0.5),\
                           XTRule('q1', Tree.fromstring('(JJ ?x0|)'), \
                                  tree_or_string('(JJ ?x0|)'), \
                                  {(0,) : 'copy'}, 0.0),\
                           XTRule('copy', 'italian', 'italian', {}, 0.5)]
    expected_derivation2 = [XTRule('q0', Tree.fromstring('(NP (DT ?x0|) ?x1|JJ)'),
                                  tree_or_string('(NP (DT ?x0|) ?x1|JJ)'), \
                                  {(0, 0) : 'copy', (1,) : 'q1'}, 0.04),\
                           XTRule('copy', 'the', 'the', {}, 0.5),\
                           XTRule('q1', Tree.fromstring('(JJ ?x0|)'), \
                                  tree_or_string('(JJ ?x0|)'), \
                                  {(0,) : 'copy'}, 0.0),\
                           XTRule('copy', 'italian', 'italian', {}, 0.5)]
    expected_derivation3 = [XTRule('q0', Tree.fromstring('(NP ?x0|DT (JJ ?x1|))'),
                                  tree_or_string('(NP ?x0|DT (JJ ?x1|))'), \
                                  {(0,) : 'q1', (1, 0) : 'copy'}, 0.04),\
                           XTRule('q1', Tree.fromstring('(DT ?x0|)'), \
                                  tree_or_string('(DT ?x0|)'), \
                                  {(0,) : 'copy'}, 0.0),\
                           XTRule('copy', 'the', 'the', {}, 0.5),\
                           XTRule('copy', 'italian', 'italian', {}, 0.5)]
    expected_derivation4 = [XTRule('q0', Tree.fromstring('(NP (DT ?x0|) (JJ ?x1|))'),
                                  tree_or_string('(NP (DT ?x0|) (JJ ?x1|))'), \
                                  {(0, 0) : 'copy', (1, 0) : 'copy'}, 0.16),\
                           XTRule('copy', 'the', 'the', {}, 0.5),\
                           XTRule('copy', 'italian', 'italian', {}, 0.5)]
    expected_derivation5 = [XTRule('q0', Tree.fromstring('(NP (DT ?x0|) ?x1|JJ)'),
                                  tree_or_string('(NP ?x0|DT ?x1|JJ)'), \
                                  {(0,) : 'q2', (1,) : 'q1'}, 0.06),\
                           XTRule('q2', tree_or_string('?x0|'), \
                                  tree_or_string('(DT ?x0|)'), \
                                  {(0,) : 'copy'}, 0.1),\
                           XTRule('copy', 'the', 'the', {}, 0.5),\
                           XTRule('q1', Tree.fromstring('(JJ ?x0|)'), \
                                  tree_or_string('(JJ ?x0|)'), \
                                  {(0,) : 'copy'}, 0.0),\
                           XTRule('copy', 'italian', 'italian', {}, 0.5)]
    expected_derivations = [expected_derivation1, expected_derivation2, \
                            expected_derivation3, expected_derivation4,
                            expected_derivation5]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|NP'),
                           tree_or_string('?x0|NP'),
                           {() : 'q0'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.maxDiff = None
    self.assertEqual(n_best, len(derivations))
    for derivation in derivations:
      self.assertIn(derivation, expected_derivations)
    self.assertItemsEqual(expected_derivations, derivations)

  def test_NonterminalToNonterminalLeftDifferent(self):
    tree1 = tree_or_string('(NP (DT the) (JJ italian))')
    tree2 = tree_or_string('(NP (DT an) (JJ italian))')
    path1 = ()
    path2 = ()
    n_best = 7
    similarity_scorer = \
      SimilarityScorerEnsemble([self.lexical_similarity, NoSimilarity()],
        [NodesDifference(), OrderDifference(), TreeComplexity()])
    options = {'similarity_scorer' : similarity_scorer,
               'similarity_score_guesser' : self.similarity_score_guesser,
               'initial_state' : self.options['initial_state']}
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, options)
    derivations = rule_extractor.ObtainBestDerivations(n_best)
    expected_derivation1 = [XTRule('q0', Tree.fromstring('(NP ?x0|DT ?x1|JJ)'),
                                  tree_or_string('(NP ?x0|DT ?x1|JJ)'), \
                                  {(0,) : 'q1', (1,) : 'q1'}, 0.00),\
                           XTRule('q1', Tree.fromstring('(DT ?x0|)'), \
                                  tree_or_string('(DT ?x0|)'), \
                                  {(0,) : 't'}, 0.00),\
                           XTRule('t', 'the', 'an', {}, 1.0),\
                           XTRule('q1', Tree.fromstring('(JJ ?x0|)'), \
                                  tree_or_string('(JJ ?x0|)'), \
                                  {(0,) : 'copy'}, 0.00),\
                           XTRule('copy', 'italian', 'italian', {}, 0.5)]
    expected_derivation2 = [XTRule('q0', Tree.fromstring('(NP ?x0|DT ?x1|JJ)'),
                                  tree_or_string('(NP ?x0|DT ?x1|JJ)'), \
                                  {(0,) : 't', (1,) : 'q1'}, 0.00),\
                           XTRule('t', Tree.fromstring('(DT the)'), \
                                  tree_or_string('(DT an)'), {}, 1.04),\
                           XTRule('q1', Tree.fromstring('(JJ ?x0|)'), \
                                  tree_or_string('(JJ ?x0|)'), \
                                  {(0,) : 'copy'}, 0.00),\
                           XTRule('copy', 'italian', 'italian', {}, 0.5)]
    expected_derivation3 = [XTRule('q0', Tree.fromstring('(NP (DT ?x0|) (JJ ?x1|))'),
                                  tree_or_string('(NP (DT ?x0|) (JJ ?x1|))'), \
                                  {(1,0) : 'copy', (0,0) : 't'}, 0.16),\
                           XTRule('t', tree_or_string('the'), \
                                  tree_or_string('an'), {}, 1.0),\
                           XTRule('copy', 'italian', 'italian', {}, 0.5)]
    expected_derivation4 = [XTRule('q0', Tree.fromstring('(NP (DT the) ?x0|JJ)'),
                                          tree_or_string('(NP (DT an) ?x0|JJ)'), \
                                  {(1,) : 'q1'}, 1.16),\
                           XTRule('q1', tree_or_string('(JJ ?x0|)'), \
                                        tree_or_string('(JJ ?x0|)'),
                                  {(0,) : 'copy'}, 0.0),\
                           XTRule('copy', 'italian', 'italian', {}, 0.5)]
    expected_derivation5 = [XTRule('q0', Tree.fromstring('(NP ?x0|DT (JJ ?x1|))'),
                                          tree_or_string('(NP ?x0|DT (JJ ?x1|))'), \
                                  {(1, 0) : 'copy', (0,) : 'q1'}, 0.04),\
                           XTRule('q1', tree_or_string('(DT ?x0|)'), \
                                        tree_or_string('(DT ?x0|)'),
                                  {(0,) : 't'}, 0.0),\
                           XTRule('t', 'the', 'an', {}, 1.0),
                           XTRule('copy', 'italian', 'italian', {}, 0.5)]
    expected_derivation6 = [XTRule('q0', Tree.fromstring('(NP (DT ?x0|) ?x1|JJ)'),
                                  tree_or_string('(NP (DT ?x0|) ?x1|JJ)'), \
                                  {(0, 0) : 't', (1,) : 'q1'}, 0.04),\
                           XTRule('t', 'the', 'an', {}, 1.0),\
                           XTRule('q1', Tree.fromstring('(JJ ?x0|)'), \
                                  tree_or_string('(JJ ?x0|)'), \
                                  {(0,) : 'copy'}, 0.00),\
                           XTRule('copy', 'italian', 'italian', {}, 0.5)]
    expected_derivation7 = [XTRule('q0', Tree.fromstring('(NP ?x0|DT (JJ ?x1|))'),
                                          tree_or_string('(NP ?x0|DT (JJ ?x1|))'), \
                                  {(1, 0) : 'copy', (0,) : 't'}, 0.04),\
                           XTRule('t', tree_or_string('(DT the)'), \
                                       tree_or_string('(DT an)'), {}, 1.04),\
                           XTRule('copy', 'italian', 'italian', {}, 0.5)]
    expected_derivations = [expected_derivation1, expected_derivation2,
                            expected_derivation3, expected_derivation4,
                            expected_derivation5, expected_derivation6,
                            expected_derivation7]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|NP'),
                           tree_or_string('?x0|NP'),
                           {() : 'q0'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.maxDiff = None
    self.assertEqual(n_best, len(derivations))
    for derivation in derivations:
      self.assertIn(derivation, expected_derivations)
    self.assertItemsEqual(expected_derivations, derivations)

  def test_NonterminalToNonterminalRightSimilar(self):
    tree1 = tree_or_string('(NP (DT the) (JJ italian))')
    tree2 = tree_or_string('(NP (DT the) (JJ european))')
    path1 = ()
    path2 = ()
    n_best = 5
    similarity_scorer = \
      SimilarityScorerEnsemble([self.lexical_similarity, NoSimilarity()],
        [NodesDifference(), OrderDifference(), TreeComplexity()])
    options = {'similarity_scorer' : similarity_scorer,
               'similarity_score_guesser' : self.similarity_score_guesser,
               'initial_state' : self.options['initial_state']}
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, options)
    derivations = rule_extractor.ObtainBestDerivations(n_best)
    expected_derivation1 = [XTRule('q0', Tree.fromstring('(NP ?x0|DT ?x1|JJ)'),
                                  tree_or_string('(NP ?x0|DT ?x1|JJ)'), \
                                  {(0,) : 'q1', (1,) : 'q1'}, 0.0),\
                           XTRule('q1', Tree.fromstring('(DT ?x0|)'), \
                                  tree_or_string('(DT ?x0|)'), \
                                  {(0,) : 'copy'}, 0.0),\
                           XTRule('copy', 'the', 'the', {}, 0.5),\
                           XTRule('q1', Tree.fromstring('(JJ ?x0|)'), \
                                  tree_or_string('(JJ ?x0|)'), \
                                  {(0,) : 'hypernym'}, 0.0),\
                           XTRule('hypernym', 'italian', 'european', {}, 0.5)]
    expected_derivation2 = [XTRule('q0', Tree.fromstring('(NP (DT ?x0|) (JJ ?x1|))'),
                                  tree_or_string('(NP (DT ?x0|) (JJ ?x1|))'), \
                                  {(0,0) : 'copy', (1,0) : 'hypernym'}, 0.16),\
                           XTRule('copy', 'the', 'the', {}, 0.5),\
                           XTRule('hypernym', 'italian', 'european', {}, 0.5)]
    expected_derivation3 = [XTRule('q0', Tree.fromstring('(NP (DT ?x0|) ?x1|JJ)'),
                                  tree_or_string('(NP (DT ?x0|) ?x1|JJ)'), \
                                  {(0,0) : 'copy', (1,) : 'q1'}, 0.04),\
                           XTRule('copy', 'the', 'the', {}, 0.5),\
                           XTRule('q1', Tree.fromstring('(JJ ?x0|)'), \
                                  tree_or_string('(JJ ?x0|)'), \
                                  {(0,) : 'hypernym'}, 0.0),\
                           XTRule('hypernym', 'italian', 'european', {}, 0.5)]
    expected_derivation4 = [XTRule('q0', Tree.fromstring('(NP ?x0|DT (JJ ?x1|))'),
                                  tree_or_string('(NP ?x0|DT (JJ ?x1|))'), \
                                  {(0,) : 'q1', (1,0) : 'hypernym'}, 0.04),\
                           XTRule('q1', Tree.fromstring('(DT ?x0|)'), \
                                  tree_or_string('(DT ?x0|)'), \
                                  {(0,) : 'copy'}, 0.0),\
                           XTRule('copy', 'the', 'the', {}, 0.5),\
                           XTRule('hypernym', 'italian', 'european', {}, 0.5)]
    expected_derivation5 = [XTRule('q0', Tree.fromstring('(NP (DT ?x0|) ?x1|JJ)'),
                                  tree_or_string('(NP ?x0|DT ?x1|JJ)'), \
                                  {(0,) : 'q2', (1,) : 'q1'}, 0.06),\
                           XTRule('q2', tree_or_string('?x0|'), \
                                  tree_or_string('(DT ?x0|)'), \
                                  {(0,) : 'copy'}, 0.1),\
                           XTRule('copy', 'the', 'the', {}, 0.5),\
                           XTRule('q1', Tree.fromstring('(JJ ?x0|)'), \
                                  tree_or_string('(JJ ?x0|)'), \
                                  {(0,) : 'hypernym'}, 0.0),\
                           XTRule('hypernym', 'italian', 'european', {}, 0.5)]
    expected_derivations = [expected_derivation1, expected_derivation2,
                            expected_derivation3, expected_derivation4,
                            expected_derivation5]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|NP'),
                           tree_or_string('?x0|NP'),
                           {() : 'q0'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.maxDiff = None
    self.assertEqual(n_best, len(derivations))
    for derivation in derivations:
      self.assertIn(derivation, expected_derivations)
    self.assertItemsEqual(expected_derivations, derivations)

  def test_NonterminalToNonterminalReordering(self):
    tree1 = tree_or_string('(NP (DT the) (JJ italian))')
    tree2 = tree_or_string('(NP (JJ italian) (DT the))')
    path1 = ()
    path2 = ()
    n_best = 5
    similarity_scorer = \
      SimilarityScorerEnsemble([self.lexical_similarity, NoSimilarity()],
        [NodesDifference(), OrderDifference(), TreeComplexity()])
    options = {'similarity_scorer' : similarity_scorer,
               'similarity_score_guesser' : self.similarity_score_guesser,
               'initial_state' : self.options['initial_state']}
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, options)
    derivations = rule_extractor.ObtainBestDerivations(n_best)
    expected_derivation1 = [XTRule('q0', Tree.fromstring('(NP ?x0|DT ?x1|JJ)'),
                                  tree_or_string('(NP ?x1|JJ ?x0|DT)'), \
                                  {(0,) : 'q1', (1,) : 'q1'}, 0.1), \
                           XTRule('q1', Tree.fromstring('(DT ?x0|)'), \
                                  tree_or_string('(DT ?x0|)'), \
                                  {(0,) : 'copy'}, 0.0),\
                           XTRule('copy', 'the', 'the', {}, 0.5),\
                           XTRule('q1', Tree.fromstring('(JJ ?x0|)'), \
                                  tree_or_string('(JJ ?x0|)'), \
                                  {(0,) : 'copy'}, 0.0),\
                           XTRule('copy', 'italian', 'italian', {}, 0.5)]
    expected_derivation2 = [XTRule('q0', Tree.fromstring('(NP (DT ?x0|) ?x1|JJ)'),
                                  tree_or_string('(NP ?x1|JJ (DT ?x0|))'), \
                                  {(1,0) : 'copy', (0,) : 'q1'}, 0.14), \
                           XTRule('copy', 'the', 'the', {}, 0.5),\
                           XTRule('q1', Tree.fromstring('(JJ ?x0|)'), \
                                  tree_or_string('(JJ ?x0|)'), \
                                  {(0,) : 'copy'}, 0.0),\
                           XTRule('copy', 'italian', 'italian', {}, 0.5)]
    expected_derivation3 = [XTRule('q0', Tree.fromstring('(NP ?x0|DT (JJ ?x1|))'),
                                  tree_or_string('(NP (JJ ?x1|) ?x0|DT)'), \
                                  {(0,0) : 'copy', (1,) : 'q1'}, 0.14), \
                           XTRule('q1', Tree.fromstring('(DT ?x0|)'), \
                                  tree_or_string('(DT ?x0|)'), \
                                  {(0,) : 'copy'}, 0.0),\
                           XTRule('copy', 'the', 'the', {}, 0.5),\
                           XTRule('copy', 'italian', 'italian', {}, 0.5)]
    expected_derivation4 = [XTRule('q0', Tree.fromstring('(NP (DT ?x0|) (JJ ?x1|))'),
                                  tree_or_string('(NP (JJ ?x1|) (DT ?x0|))'), \
                                  {(0,0) : 'copy', (1,0) : 'copy'}, 0.26), \
                           XTRule('copy', 'the', 'the', {}, 0.5),\
                           XTRule('copy', 'italian', 'italian', {}, 0.5)]
    expected_derivation5 = [XTRule('q0', Tree.fromstring('(NP (DT ?x0|) ?x1|JJ)'),
                                  tree_or_string('(NP ?x1|JJ ?x0|DT)'), \
                                  {(1,) : 'q2', (0,) : 'q1'}, 0.16), \
                           XTRule('q2', tree_or_string('?x0|'), \
                                  tree_or_string('(DT ?x0|)'), \
                                  {(0,) : 'copy'}, 0.1),\
                           XTRule('copy', 'the', 'the', {}, 0.5),\
                           XTRule('q1', Tree.fromstring('(JJ ?x0|)'), \
                                  tree_or_string('(JJ ?x0|)'), \
                                  {(0,) : 'copy'}, 0.0),\
                           XTRule('copy', 'italian', 'italian', {}, 0.5)]
    expected_derivations = [expected_derivation1, expected_derivation2,
                            expected_derivation3, expected_derivation4,
                            expected_derivation5]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|NP'),
                           tree_or_string('?x0|NP'),
                           {() : 'q0'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.maxDiff = None
    self.assertEqual(n_best, len(derivations))
    for derivation in derivations:
      self.assertIn(derivation, expected_derivations)
    self.assertItemsEqual(expected_derivations, derivations)

  def test_NonterminalToNonterminalReorderingAndSimilarity(self):
    tree1 = tree_or_string('(NP (DT the) (JJ italian))')
    tree2 = tree_or_string('(NP (JJ european) (DT the))')
    path1 = ()
    path2 = ()
    n_best = 5
    similarity_scorer = \
      SimilarityScorerEnsemble([self.lexical_similarity, NoSimilarity()],
        [NodesDifference(), OrderDifference(), TreeComplexity()])
    options = {'similarity_scorer' : similarity_scorer,
               'similarity_score_guesser' : self.similarity_score_guesser,
               'initial_state' : self.options['initial_state']}
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, options)
    derivations = rule_extractor.ObtainBestDerivations(n_best)
    expected_derivation1 = [XTRule('q0', Tree.fromstring('(NP ?x0|DT ?x1|JJ)'),
                                  tree_or_string('(NP ?x1|JJ ?x0|DT)'), \
                                  {(0,) : 'q1', (1,) : 'q1'}, 0.1), \
                           XTRule('q1', Tree.fromstring('(DT ?x0|)'), \
                                  tree_or_string('(DT ?x0|)'), \
                                  {(0,) : 'copy'}, 0.0),\
                           XTRule('copy', 'the', 'the', {}, 0.5),\
                           XTRule('q1', Tree.fromstring('(JJ ?x0|)'), \
                                  tree_or_string('(JJ ?x0|)'), \
                                  {(0,) : 'hypernym'}, 0.0),\
                           XTRule('hypernym', 'italian', 'european', {}, 0.5)]
    expected_derivation2 = [XTRule('q0', Tree.fromstring('(NP (DT ?x0|) ?x1|JJ)'),
                                  tree_or_string('(NP ?x1|JJ (DT ?x0|))'), \
                                  {(1,0) : 'copy', (0,) : 'q1'}, 0.14), \
                           XTRule('copy', 'the', 'the', {}, 0.5),\
                           XTRule('q1', Tree.fromstring('(JJ ?x0|)'), \
                                  tree_or_string('(JJ ?x0|)'), \
                                  {(0,) : 'hypernym'}, 0.0),\
                           XTRule('hypernym', 'italian', 'european', {}, 0.5)]
    expected_derivation3 = [XTRule('q0', Tree.fromstring('(NP ?x0|DT (JJ ?x1|))'),
                                  tree_or_string('(NP (JJ ?x1|) ?x0|DT)'), \
                                  {(1,) : 'q1', (0,0) : 'hypernym'}, 0.14), \
                           XTRule('q1', Tree.fromstring('(DT ?x0|)'), \
                                  tree_or_string('(DT ?x0|)'), \
                                  {(0,) : 'copy'}, 0.0),\
                           XTRule('copy', 'the', 'the', {}, 0.5),\
                           XTRule('hypernym', 'italian', 'european', {}, 0.5)]
    expected_derivation4 = [XTRule('q0', Tree.fromstring('(NP (DT ?x0|) (JJ ?x1|))'),
                                  tree_or_string('(NP (JJ ?x1|) (DT ?x0|))'), \
                                  {(1,0) : 'copy', (0,0) : 'hypernym'}, 0.26), \
                           XTRule('copy', 'the', 'the', {}, 0.5),\
                           XTRule('hypernym', 'italian', 'european', {}, 0.5)]
    expected_derivation5 = [XTRule('q0', Tree.fromstring('(NP (DT ?x0|) ?x1|JJ)'),
                                  tree_or_string('(NP ?x1|JJ ?x0|DT)'), \
                                  {(1,) : 'q2', (0,) : 'q1'}, 0.16), \
                           XTRule('q2', tree_or_string('?x0|'), \
                                  tree_or_string('(DT ?x0|)'), \
                                  {(0,) : 'copy'}, 0.1),\
                           XTRule('copy', 'the', 'the', {}, 0.5),\
                           XTRule('q1', Tree.fromstring('(JJ ?x0|)'), \
                                  tree_or_string('(JJ ?x0|)'), \
                                  {(0,) : 'hypernym'}, 0.0),\
                           XTRule('hypernym', 'italian', 'european', {}, 0.5)]
    expected_derivations = [expected_derivation1, expected_derivation2,
                            expected_derivation3, expected_derivation4,
                            expected_derivation5]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|NP'),
                           tree_or_string('?x0|NP'),
                           {() : 'q0'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.maxDiff = None
    self.assertEqual(n_best, len(derivations))
    for derivation in derivations:
      self.assertIn(derivation, expected_derivations)
    self.assertItemsEqual(expected_derivations, derivations)

  def test_NonterminalToNonterminalBranchDelete(self):
    tree1 = tree_or_string('(NP (DT the) (JJ italian))')
    tree2 = tree_or_string('(NP (DT the))')
    path1 = ()
    path2 = ()
    n_best = 3
    similarity_scorer = \
      SimilarityScorerEnsemble([self.lexical_similarity, NoSimilarity()],
        [NodesDifference(), OrderDifference(), TreeComplexity()])
    options = {'similarity_scorer' : similarity_scorer,
               'similarity_score_guesser' : self.similarity_score_guesser,
               'initial_state' : self.options['initial_state'],
               'deletions' : True}
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, options)
    derivations = rule_extractor.ObtainBestDerivations(n_best)
    expected_derivation1 = [XTRule('q0', Tree.fromstring('(NP ?x0|DT ?xx0|JJ)'),
                                  tree_or_string('(NP ?x0|DT)'), \
                                  {(0,) : 'q1'}, 1.09),\
                           XTRule('q1', Tree.fromstring('(DT ?x0|)'), \
                                  tree_or_string('(DT ?x0|)'), \
                                  {(0,) : 'copy'}, 0.0),\
                           XTRule('copy', 'the', 'the', {}, 0.5)]
    expected_derivation2 = [XTRule('q0', Tree.fromstring('(NP (DT ?x0|) ?xx0|JJ)'),
                                  tree_or_string('(NP (DT ?x0|))'), \
                                  {(0, 0) : 'copy'}, 1.19333333333),\
                           XTRule('copy', 'the', 'the', {}, 0.5)]
    expected_derivation3 = [XTRule('q0', Tree.fromstring('(NP ?x0|DT ?xx0|JJ)'),
                                  tree_or_string('(NP (DT ?x0|))'), \
                                  {(0, 0) : 'q1'}, 1.09),\
                           XTRule('q1', tree_or_string('(DT ?x0|)'), \
                                  tree_or_string('?x0|'), \
                                  {() : 'copy'}, 0.1),\
                           XTRule('copy', 'the', 'the', {}, 0.5)]
    # This derivation is not actually good, because TreeComplexity
    # is penalizing it very much the LHS of the first rule, since
    # the (DT ?x0|) and the full (JJ ...) branch count towards the size.
    expected_derivation4 = [XTRule('q0', Tree.fromstring('(NP (DT ?x0|) ?xx0|JJ)'),
                                  tree_or_string('(NP ?x0|DT)'), \
                                  {(0,) : 'q2'}, 1.19),\
                           XTRule('q2', tree_or_string('?x0|'), \
                                  tree_or_string('(DT ?x0|)'), \
                                  {(0,) : 'copy'}, 0.1),\
                           XTRule('copy', 'the', 'the', {}, 0.5)]
    expected_derivations = [expected_derivation1, expected_derivation2, \
                            expected_derivation3]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|NP'),
                           tree_or_string('?x0|NP'),
                           {() : 'q0'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.maxDiff = None
    self.assertEqual(n_best, len(derivations))
    for derivation in derivations:
      self.assertIn(derivation, expected_derivations)
    self.assertItemsEqual(expected_derivations, derivations)

  def test_NonterminalToNonterminalCrossConstituent(self):
    tree1 = tree_or_string('(S (NP (DT the) (NN house)) (VP (VBZ is) (JJ nice)))')
    tree2 = tree_or_string('(S (NP (DT the) (JJ nice)) (VP (VBZ is) (NN house)))')
    path1 = ()
    path2 = ()
    n_best = 1
    from linguistics.similarity_semantics import TreeSize
    tree_size = TreeSize()
    tree_size.feature_weight = 0.05
    similarity_scorer = \
      SimilarityScorerEnsemble([self.lexical_similarity, NoSimilarity()],
        [NodesDifference(), OrderDifference(), tree_size])
    options = {'similarity_scorer' : similarity_scorer,
               'similarity_score_guesser' : self.similarity_score_guesser,
               'max_source_branches' : 4,
               'initial_state' : self.options['initial_state']}
    rule_extractor = RuleExtractor(tree1, tree2, path1, path2, options)
    derivations = rule_extractor.ObtainBestDerivations(n_best)
    expected_derivation1 = \
      [XTRule('q0',
              tree_or_string('(S (NP ?x0|DT ?x1|NN) (VP ?x2|VBZ ?x3|JJ))'),
              tree_or_string('(S (NP ?x0|DT ?x3|JJ) (VP ?x2|VBZ ?x1|NN))'),
              {(0, 0) : 'q2',
               (0, 1) : 'q2',
               (1, 0) : 'q2',
               (1, 1) : 'q2'}, 0.95),\
       XTRule('q2',
              tree_or_string('(DT ?x0|)'), \
              tree_or_string('(DT ?x0|)'), \
              {(0,) : 'copy'}, 0.1),\
       XTRule('copy', 'the', 'the', {}, 0.6),
       XTRule('q2',
              tree_or_string('(NN ?x0|)'), \
              tree_or_string('(NN ?x0|)'), \
              {(0,) : 'copy'}, 0.1),\
       XTRule('copy', 'house', 'house', {}, 0.6),
       XTRule('q2',
              tree_or_string('(VBZ ?x0|)'), \
              tree_or_string('(VBZ ?x0|)'), \
              {(0,) : 'copy'}, 0.1),\
       XTRule('copy', 'is', 'is', {}, 0.6),
       XTRule('q2',
              tree_or_string('(JJ ?x0|)'), \
              tree_or_string('(JJ ?x0|)'), \
              {(0,) : 'copy'}, 0.1),\
       XTRule('copy', 'nice', 'nice', {}, 0.6)]
    expected_derivations = [expected_derivation1]
    initial_rule = [XTRule(self.options['initial_state'],
                           tree_or_string('?x0|S'),
                           tree_or_string('?x0|S'),
                           {() : 'q0'}, 0.0)]
    expected_derivations = \
      [[rule] + derivation \
         for rule, derivation in product(initial_rule, expected_derivations)]
    self.maxDiff = None
    self.assertEqual(n_best, len(derivations))
    for derivation in derivations:
      self.assertIn(derivation, expected_derivations)
    self.assertItemsEqual(expected_derivations, derivations)

class ObtainTreePatternTestCase(unittest.TestCase):
  def test_TerminalRootSubpath(self):
    tree = tree_or_string('the')
    path = ()
    subpaths = ((),)
    tree_pattern = ObtainTreePattern(tree, path, subpaths)
    expected_tree_pattern = tree_or_string('?x0|')
    self.assertEqual(expected_tree_pattern, tree_pattern)

  def test_TerminalNoSubpaths(self):
    tree = tree_or_string('the')
    path = ()
    subpaths = ()
    tree_pattern = ObtainTreePattern(tree, path, subpaths)
    expected_tree_pattern = tree_or_string('the')
    self.assertEqual(expected_tree_pattern, tree_pattern)

  def test_TerminalWrongPath(self):
    tree = tree_or_string('the')
    path = (0,)
    subpaths = ((),)
    with self.assertRaises(ValueError):
      tree_pattern = ObtainTreePattern(tree, path, subpaths)

  def test_TerminalWrongSubpath(self):
    tree = tree_or_string('the')
    path = ()
    subpaths = ((0,),)
    with self.assertRaises(ValueError):
      tree_pattern = ObtainTreePattern(tree, path, subpaths)

  def test_PreterminalRootSubpath(self):
    tree = tree_or_string('(DT the)')
    path = ()
    subpaths = ((),)
    tree_pattern = ObtainTreePattern(tree, path, subpaths)
    expected_tree_pattern = tree_or_string('?x0|DT')
    self.assertEqual(expected_tree_pattern, tree_pattern)

  def test_PreterminalSubpath1(self):
    tree = tree_or_string('(DT the)')
    path = ()
    subpaths = ((0,),)
    tree_pattern = ObtainTreePattern(tree, path, subpaths)
    expected_tree_pattern = tree_or_string('(DT ?x0|)')
    self.assertEqual(expected_tree_pattern, tree_pattern)

  def test_PreterminalNoSubpaths(self):
    tree = tree_or_string('(DT the)')
    path = ()
    subpaths = ()
    tree_pattern = ObtainTreePattern(tree, path, subpaths)
    expected_tree_pattern = tree_or_string('(DT the)')
    self.assertEqual(expected_tree_pattern, tree_pattern)

  def test_NonterminalRootSubpath(self):
    tree = tree_or_string('(NP (DT the) (NN house))')
    path = ()
    subpaths = ((),)
    tree_pattern = ObtainTreePattern(tree, path, subpaths)
    expected_tree_pattern = tree_or_string('?x0|NP')
    self.assertEqual(expected_tree_pattern, tree_pattern)

  def test_NonterminalNoSubpaths(self):
    tree = tree_or_string('(NP (DT the) (NN house))')
    path = ()
    subpaths = ()
    tree_pattern = ObtainTreePattern(tree, path, subpaths)
    expected_tree_pattern = tree_or_string('(NP (DT the) (NN house))')
    self.assertEqual(expected_tree_pattern, tree_pattern)

  def test_NonterminalSubpath1(self):
    tree = tree_or_string('(NP (DT the) (NN house))')
    path = ()
    subpaths = ((0,),)
    tree_pattern = ObtainTreePattern(tree, path, subpaths)
    expected_tree_pattern = tree_or_string('(NP ?x0|DT (NN house))')
    self.assertEqual(expected_tree_pattern, tree_pattern)

  def test_NonterminalSubpathToLeaf(self):
    tree = tree_or_string('(NP (DT the) (NN house))')
    path = ()
    subpaths = ((1, 0),)
    tree_pattern = ObtainTreePattern(tree, path, subpaths)
    expected_tree_pattern = tree_or_string('(NP (DT the) (NN ?x0|))')
    self.assertEqual(expected_tree_pattern, tree_pattern)

  def test_NonterminalSubpath1and2(self):
    tree = tree_or_string('(NP (DT the) (NN house))')
    path = ()
    subpaths = ((0,), (1,))
    tree_pattern = ObtainTreePattern(tree, path, subpaths)
    expected_tree_pattern = tree_or_string('(NP ?x0|DT ?x1|NN)')
    self.assertEqual(expected_tree_pattern, tree_pattern)

  def test_NonterminalMultilevel(self):
    tree = tree_or_string('(NP (DT the) (NN house))')
    path = ()
    subpaths = ((0,), (1,0))
    tree_pattern = ObtainTreePattern(tree, path, subpaths)
    expected_tree_pattern = tree_or_string('(NP ?x0|DT (NN ?x1|))')
    self.assertEqual(expected_tree_pattern, tree_pattern)

  def test_NonterminalLowerPreterminalVariable(self):
    tree = tree_or_string('(NP (DT the) (NN house))')
    path = (0,)
    subpaths = [(0,)]
    tree_pattern = ObtainTreePattern(tree, path, subpaths)
    expected_tree_pattern = tree_or_string('?x0|DT')
    self.assertEqual(expected_tree_pattern, tree_pattern)

  def test_NonterminalLowerTerminalVariable(self):
    tree = tree_or_string('(NP (DT the) (NN house))')
    path = (0,0)
    subpaths = [(0,0)]
    tree_pattern = ObtainTreePattern(tree, path, subpaths)
    expected_tree_pattern = tree_or_string('?x0|')
    self.assertEqual(expected_tree_pattern, tree_pattern)

class GetDisjointPathsTestCase(unittest.TestCase):
  def test_AreDisjointPathsSiblings(self):
    paths = [(0,), (1,)]
    self.assertTrue(AreDisjointPaths(paths))

  def test_AreDisjointPathsParentChild(self):
    paths = [(), (0,)]
    self.assertFalse(AreDisjointPaths(paths))

  def test_AreDisjointPathsChildParent(self):
    paths = [(0,), ()]
    self.assertFalse(AreDisjointPaths(paths))

  def test_AreDisjointPathsRootPath(self):
    paths = [()]
    self.assertTrue(AreDisjointPaths(paths))

  def test_AreDisjointPathsRootPathTuple(self):
    paths = ((),)
    self.assertTrue(AreDisjointPaths(paths))

  def test_TerminalPathLength0(self):
    tree = tree_or_string('the')
    disjoint_paths = GetDisjointPaths(tree, (), 0, 0)
    expected_disjoint_paths = [()]
    self.assertItemsEqual(expected_disjoint_paths, disjoint_paths)

  def test_TerminalPathLength1(self):
    tree = tree_or_string('the')
    disjoint_paths = GetDisjointPaths(tree, (), 1, 1)
    # expected_disjoint_paths = [((),)]
    expected_disjoint_paths = [((),)]
    self.assertItemsEqual(expected_disjoint_paths, disjoint_paths)

  def test_TerminalPathLength0To1(self):
    tree = tree_or_string('the')
    disjoint_paths = GetDisjointPaths(tree, (), 0, 1)
    # expected_disjoint_paths = [(), ((),)]
    expected_disjoint_paths = [(), ((),)]
    self.assertItemsEqual(expected_disjoint_paths, disjoint_paths)

  def test_TerminalFromPreterminallPathLength0(self):
    tree = Tree.fromstring('(DT the)')
    disjoint_paths = GetDisjointPaths(tree, (0,), 0, 0)
    expected_disjoint_paths = [()]
    self.assertItemsEqual(expected_disjoint_paths, disjoint_paths)

  def test_TerminalFromPreterminallPathLength1(self):
    tree = Tree.fromstring('(DT the)')
    disjoint_paths = GetDisjointPaths(tree, (0,), 1, 1)
    expected_disjoint_paths = [((0,),)]
    self.assertItemsEqual(expected_disjoint_paths, disjoint_paths)

  def test_TerminalFromPreterminallPathLength0To1(self):
    tree = Tree.fromstring('(DT the)')
    disjoint_paths = GetDisjointPaths(tree, (0,), 0, 1)
    expected_disjoint_paths = [(), ((0,),)]
    self.assertItemsEqual(expected_disjoint_paths, disjoint_paths)

  def test_PreterminalPathLength1(self):
    tree = Tree.fromstring('(DT the)')
    disjoint_paths = GetDisjointPaths(tree, (), 1, 1)
    # expected_disjoint_paths = [((),), \
    #                            ((0,),)]
    expected_disjoint_paths = [((0,),)]
    self.assertItemsEqual(expected_disjoint_paths, disjoint_paths)

  def test_PreterminalPathLength2(self):
    tree = Tree.fromstring('(DT the)')
    disjoint_paths = GetDisjointPaths(tree, (), 2, 2)
    expected_disjoint_paths = []
    self.assertItemsEqual(expected_disjoint_paths, disjoint_paths)

  def test_PreterminalPathLength1To2(self):
    tree = Tree.fromstring('(DT the)')
    disjoint_paths = GetDisjointPaths(tree, (), 1, 2)
    expected_disjoint_paths = [((0,),)]
    self.assertItemsEqual(expected_disjoint_paths, disjoint_paths)

  def test_NonterminalPathLength1(self):
    tree = Tree.fromstring('(NP (DT the) (NN house))')
    disjoint_paths = GetDisjointPaths(tree, (), 1, 1)
    expected_disjoint_paths = [((0,),), \
                               ((1,),), \
                               ((0, 0),), \
                               ((1, 0),)]
    self.assertItemsEqual(expected_disjoint_paths, disjoint_paths)

  def test_NonterminalPathLength2(self):
    tree = Tree.fromstring('(NP (DT the) (NN house))')
    disjoint_paths = GetDisjointPaths(tree, (), 1, 2)
    expected_disjoint_paths = [((0,),), \
                               ((1,),), \
                               ((0, 0),), \
                               ((1, 0),), \
                               ((0,), (1,)), \
                               ((0,), (1, 0)), \
                               ((0, 0), (1,)), \
                               ((0, 0), (1, 0))]
    self.assertItemsEqual(expected_disjoint_paths, disjoint_paths)

class GetCommonParentsAtTestCase(unittest.TestCase):
  def test_NoPaths(self):
    max_depth = 5
    paths = []
    expected_parents = [()]
    parents = GetCommonParentsAt(paths, max_depth)
    self.assertItemsEqual(expected_parents, parents)

  def test_1pathLength1(self):
    max_depth = 5
    paths = [(0,)]
    expected_parents = [(), (0,)]
    parents = GetCommonParentsAt(paths, max_depth)
    self.assertItemsEqual(expected_parents, parents)

  def test_1pathLength2(self):
    max_depth = 5
    paths = [(0,1)]
    expected_parents = [(), (0,), (0,1)]
    parents = GetCommonParentsAt(paths, max_depth)
    self.assertItemsEqual(expected_parents, parents)

  def test_1pathLength5Depth3(self):
    max_depth = 3
    paths = [(0,1,0,1,0)]
    expected_parents = [(0,1,0,1,0), (0,1,0,1), (0,1,0), (0,1)]
    parents = GetCommonParentsAt(paths, max_depth)
    self.assertItemsEqual(expected_parents, parents)

  def test_2pathsLength2and2OneCommon(self):
    max_depth = 5
    paths = [(0,1), (1,0)]
    expected_parents = [()]
    parents = GetCommonParentsAt(paths, max_depth)
    self.assertItemsEqual(expected_parents, parents)

  def test_2pathsLength2and2TwoCommon(self):
    max_depth = 5
    paths = [(0,0), (0,1)]
    expected_parents = [(), (0,)]
    parents = GetCommonParentsAt(paths, max_depth)
    self.assertItemsEqual(expected_parents, parents)

  def test_2pathsLength4and5ThreeCommon(self):
    max_depth = 3
    paths = [(0,0,0,1), (0,0,0,0,1)]
    expected_parents = [(0,0,0), (0,0)]
    parents = GetCommonParentsAt(paths, max_depth)
    self.assertItemsEqual(expected_parents, parents)

  def test_2pathsLength4and7Max4(self):
    max_depth = 4
    paths = [(0,0,0,1), (0,0,0,0,0,0,0)]
    expected_parents = [(0,0,0)]
    parents = GetCommonParentsAt(paths, max_depth)
    self.assertItemsEqual(expected_parents, parents)

  def test_2pathsLength4and7Max5(self):
    max_depth = 5
    paths = [(0,0,0,1), (0,0,0,0,0,0,0)]
    expected_parents = [(0,0,0), (0,0)]
    parents = GetCommonParentsAt(paths, max_depth)
    self.assertItemsEqual(expected_parents, parents)

  def test_2pathsLength4and7Max3(self):
    max_depth = 3
    paths = [(0,0,0,1), (0,0,0,0,0,0,0)]
    expected_parents = []
    parents = GetCommonParentsAt(paths, max_depth)
    self.assertItemsEqual(expected_parents, parents)

  def test_2pathsLength4and7Max5EarlyDisjoint(self):
    max_depth = 6
    paths = [(0,0,0,1), (0,1,0,0,0,0,0)]
    expected_parents = [(0,)]
    parents = GetCommonParentsAt(paths, max_depth)
    self.assertItemsEqual(expected_parents, parents)

if __name__ == '__main__':
  suite1  = unittest.TestLoader().loadTestsFromTestCase(GetDisjointPathsTestCase)
  suite2  = unittest.TestLoader().loadTestsFromTestCase(ObtainTreePatternTestCase)
  suite3  = unittest.TestLoader().loadTestsFromTestCase(ExtractRulesTestCase)
  suite4  = unittest.TestLoader().loadTestsFromTestCase(GetCommonParentsAtTestCase)
  # suite5  = unittest.TestLoader().loadTestsFromTestCase(ExtractRulesDepsTestCase)
  suite6  = unittest.TestLoader().loadTestsFromTestCase(TransformationTestCase)
  suites  = unittest.TestSuite([suite1, suite2, suite3, suite4, suite6])
  unittest.TextTestRunner(verbosity=2).run(suites)

