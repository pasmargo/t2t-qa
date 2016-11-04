import unittest

import copy

from training.train_perceptron import PerceptronModel
from training.transducer import xT
from training.transductionrule import *
from utils.tree_tools import tree_or_string

class TrainPerceptronTestCase(unittest.TestCase):
  def setUp(self):
    self.S = 'q0'
    rule0 = XTRule(self.S, tree_or_string('(A ?x0|)'),
                        tree_or_string('(O ?x0|)'),
                        {(0,) : self.S}, 1.0)
    rule1 = XTRule(self.S, tree_or_string('(A ?x0|)'),
                        tree_or_string('(P ?x0|)'),
                        {(0,) : self.S}, 1.0)
    rule2 = XTRule(self.S, tree_or_string('a'),
                        tree_or_string('o'),
                        {}, 1.0)
    self.rules = [rule0, rule1, rule2]
    self.transducer = xT(self.S, self.rules)
    self.model = PerceptronModel

  def test_IndependentFeatures(self):
    self.rules[0].features = [(0, 1.0), (1, 1.0)]
    self.rules[1].features = [(2, 1.0), (3, 1.0)]
    self.rules[2].features = [(4, 1.0), (5, 1.0)]
    feat_weights = {0: 1.0, 1: 1.0, 2: 2.0, 3: 1.0, 4: 1.0, 5: 1.0}
    input1, output1, weight = '(A a)', '(O o)', 1.0
    corpus = [(input1, output1, weight)]

    perceptron_model = self.model()
    perceptron_model.max_iterations = 1
    perceptron_model.learing_rate = .1
    perceptron_model.feat_weights = feat_weights
    # from pudb import set_trace; set_trace()
    perceptron_model.train(self.transducer, corpus)

    self.assertEqual(feat_weights[0], 1.1, 'feat_weights: {0}'.format(feat_weights))
    self.assertEqual(feat_weights[1], 1.1, 'feat_weights: {0}'.format(feat_weights))
    self.assertEqual(feat_weights[2], 1.9, 'feat_weights: {0}'.format(feat_weights))
    self.assertEqual(feat_weights[3], 0.9, 'feat_weights: {0}'.format(feat_weights))
    self.assertEqual(feat_weights[4], 1.0, 'feat_weights: {0}'.format(feat_weights))
    self.assertEqual(feat_weights[5], 1.0, 'feat_weights: {0}'.format(feat_weights))

  def test_SharedFeatures(self):
    self.rules[0].features = [(0, 1.0), (1, 1.0)]
    self.rules[1].features = [(1, 1.0), (3, 1.0)]
    self.rules[2].features = [(4, 1.0), (5, 1.0)]
    feat_weights = {0: 1.0, 1: 1.0, 3: 2.0, 4: 1.0, 5: 1.0}
    input1, output1, weight = '(A a)', '(O o)', 1.0
    corpus = [(input1, output1, weight)]

    perceptron_model = self.model()
    perceptron_model.max_iterations = 1
    perceptron_model.learing_rate = .1
    perceptron_model.feat_weights = feat_weights
    perceptron_model.train(self.transducer, corpus)

    self.assertEqual(feat_weights[0], 1.1, 'feat_weights: {0}'.format(feat_weights))
    self.assertEqual(feat_weights[1], 1.0, 'feat_weights: {0}'.format(feat_weights))
    self.assertEqual(feat_weights[3], 1.9, 'feat_weights: {0}'.format(feat_weights))
    self.assertEqual(feat_weights[4], 1.0, 'feat_weights: {0}'.format(feat_weights))
    self.assertEqual(feat_weights[5], 1.0, 'feat_weights: {0}'.format(feat_weights))
    self.assertTrue(2 not in feat_weights)

  def test_DifferentFeatureValues(self):
    self.rules[0].features = [(0, 1.1), (1, 1.0)]
    self.rules[1].features = [(1, 1.0), (3, 1.0)]
    self.rules[2].features = [(4, 1.0), (5, 1.0)]
    feat_weights = {0: 1.0, 1: 1.0, 3: 2.0, 4: 1.0, 5: 1.0}
    input1, output1, weight = '(A a)', '(O o)', 1.0
    corpus = [(input1, output1, weight)]

    perceptron_model = self.model()
    perceptron_model.max_iterations = 1
    perceptron_model.learing_rate = .1
    perceptron_model.feat_weights = feat_weights
    perceptron_model.train(self.transducer, corpus)

    self.assertEqual(feat_weights[0], 1.11, 'feat_weights: {0}'.format(feat_weights))
    self.assertEqual(feat_weights[1], 1.0, 'feat_weights: {0}'.format(feat_weights))
    self.assertEqual(feat_weights[3], 1.9, 'feat_weights: {0}'.format(feat_weights))
    self.assertEqual(feat_weights[4], 1.0, 'feat_weights: {0}'.format(feat_weights))
    self.assertEqual(feat_weights[5], 1.0, 'feat_weights: {0}'.format(feat_weights))
    self.assertTrue(2 not in feat_weights)

  def test_PerceptronExampleArticle(self):
    rule0 = XTRule(self.S, tree_or_string('(A ?x0| ?x1|)'),
                        tree_or_string('(A (R ?x1| ?x0|) (S X))'),
                        {(0, 0) : self.S, (0, 1) : self.S}, 1.0)
    rule1 = XTRule(self.S, tree_or_string('(B ?x0| ?x1|)'),
                        tree_or_string('U'),
                        {}, 1.0)
    rule2 = XTRule(self.S, tree_or_string('(C ?x0| ?x1|)'),
                        tree_or_string('(T ?x0| ?x1|)'),
                        {(0,) : self.S, (1,) : self.S}, 1.0)
    rule3 = XTRule(self.S, tree_or_string('(C ?x0| ?x1|)'),
                        tree_or_string('(T ?x1| ?x0|)'),
                        {(0,) : self.S, (1,) : self.S}, 1.0)
    rule4 = XTRule(self.S, tree_or_string('F'),
                        tree_or_string('V'),
                        {}, 1.0)
    rule5 = XTRule(self.S, tree_or_string('F'),
                        tree_or_string('W'),
                        {}, 1.0)
    rule6 = XTRule(self.S, tree_or_string('G'),
                        tree_or_string('V'),
                        {}, 1.0)
    rule7 = XTRule(self.S, tree_or_string('G'),
                        tree_or_string('W'),
                        {}, 1.0)
    rule8 = XTRule(self.S, tree_or_string('G'), # This rule does not apply.
                        tree_or_string('Z'),
                        {}, 1.0)
    rules = [rule0, rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8]
    rules[0].features = [(0, 1.0), (10, 1.0)]
    rules[1].features = [(1, 1.0), (11, 1.0)]
    rules[2].features = [(2, 1.0), (12, 1.0)]
    rules[3].features = [(3, 1.0), (13, 1.0)]
    rules[4].features = [(4, 1.0), (14, 1.0)]
    rules[5].features = [(5, 1.0), (15, 1.0)]
    rules[6].features = [(6, 1.0), (16, 1.0)]
    rules[7].features = [(7, 1.0), (17, 1.0)]
    rules[8].features = [(8, 1.0), (18, 1.0)]
    feat_weights = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0,
                    5: 1.0, 6: 1.0, 7: 1.0, 8: 2.0, 9: 1.0,
                    10: 1.0, 11: 1.0, 12: 2.0, 13: 1.0, 14: 1.0,
                    15: 1.0, 16: 1.0, 17: 1.0, 18: 2.0}
    transducer = xT(self.S, rules)
    input1  = '(A (B D E) (C F G))'
    output1 = '(A (R (T V W) U) (S X))'
    pair_weight = 1.0
    corpus = [(input1, output1, pair_weight)]

    perceptron_model = self.model()
    perceptron_model.max_iterations = 1
    perceptron_model.learing_rate = .1
    perceptron_model.feat_weights = copy.deepcopy(feat_weights)
    perceptron_model.train(transducer, corpus)

    self.assertEqual(perceptron_model.feat_weights[18], 1.9,
                     'feat_weights: {0}'.format(feat_weights))

    # Check that, at first, for non-estimated feature weights, the transducer
    # produces a grammar that does not obtain the desired target tree. However,
    # when running the structured perceptron to estimate the feature weights,
    # the transducer produces a grammar that obtains the desired target tree.
    transducer = xT(self.S, rules)
    wrtg = transducer.Transduce(tree_or_string(input1))
    perceptron_model.weight_wrtg(wrtg)
    trg_tree = wrtg.GenerateTrees().next()[0]
    self.assertNotEqual(repr(trg_tree), output1)

    transducer = xT(self.S, rules)
    perceptron_model = self.model()
    perceptron_model.max_iterations = 10
    perceptron_model.learing_rate = .1
    perceptron_model.feat_weights = copy.deepcopy(feat_weights)
    perceptron_model.train(transducer, corpus)

    wrtg = transducer.Transduce(tree_or_string(input1))
    perceptron_model.weight_wrtg(wrtg)
    trg_tree = wrtg.GenerateTrees().next()[0]
    self.assertEqual(repr(trg_tree), output1)

if __name__ == '__main__':
  suite1 = unittest.TestLoader().loadTestsFromTestCase(TrainPerceptronTestCase)
  suites = unittest.TestSuite([suite1])
  unittest.TextTestRunner(verbosity=2).run(suites)

