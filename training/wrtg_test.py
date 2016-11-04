import unittest

# from pudb import set_trace; set_trace() 

from training.loadrules import loadrules
from training.transducer import xT
from training.transductionrule import *
from training.ruleindex import *
from utils.tree_tools import immutable, Tree, tree_or_string
from training.wrtg import wRTG, Production, RHS
from training.wrtg import (TargetProjectionFromDerivation,
  SourceProjectionFromDerivationMix, SourceProjectionFromDerivationStrict)

class ObtainBestDerivationTestCase(unittest.TestCase):
  def setUp(self):
    self.S = 'q'

  def test_ExampleUnconstrainedOutputTerminal(self):
    rules = loadrules("training/ttt_example_fig5.yaml", fmt='yaml')
    intree = immutable(tree_or_string('F'))
    transducer = xT(self.S, rules)
    outtree = None
    q, i, o = self.S, (), ()
    initial_state = q, i, ''
    productions, non_terminals = transducer.Produce(intree, outtree, q, i, o)
    wrtg = wRTG(transducer.rules, non_terminals, initial_state, productions)
    prunned_wrtg = wrtg.Prune()
    best_derivation = wrtg.ObtainBestDerivation()
    target_weighted_tree = TargetProjectionFromDerivation(best_derivation)
    expected_weighted_trees = [(tree_or_string('V'), 0.9)]
    self.assertIn(target_weighted_tree, expected_weighted_trees)

  def test_ExampleUnconstrainedOutputNonterminal(self):
    rules = loadrules("training/ttt_example_fig5.yaml", fmt='yaml')
    intree = immutable(tree_or_string('(C F G)'))
    transducer = xT(self.S, rules)
    outtree = None
    q, i, o = self.S, (), ()
    initial_state = q, i, ''
    productions, non_terminals = transducer.Produce(intree, outtree, q, i, o)
    wrtg = wRTG(transducer.rules, non_terminals, initial_state, productions)
    prunned_wrtg = wrtg.Prune()
    best_derivation = wrtg.ObtainBestDerivation()
    # from pudb import set_trace; set_trace()
    target_weighted_tree = TargetProjectionFromDerivation(best_derivation)
    expected_weighted_trees = [(tree_or_string('(T V V)'), 0.27),
                               (tree_or_string('(T V W)'), 0.27)]
    self.assertIn(target_weighted_tree, expected_weighted_trees)

  def test_ExampleUnconstrainedOutputFull(self):
    rules = loadrules("training/ttt_example_fig5.yaml", fmt='yaml')
    intree = immutable(tree_or_string('(A (B D E) (C F G))'))
    transducer = xT(self.S, rules)
    outtree = None
    q, i, o = self.S, (), ()
    initial_state = q, i, ''
    productions, non_terminals = transducer.Produce(intree, outtree, q, i, o)
    wrtg = wRTG(transducer.rules, non_terminals, initial_state, productions)
    prunned_wrtg = wrtg.Prune()
    best_derivation = wrtg.ObtainBestDerivation()
    # from pudb import set_trace; set_trace()
    target_weighted_tree = TargetProjectionFromDerivation(best_derivation)
    expected_weighted_trees = [\
     (Tree.fromstring('(A (R (T V V) U) (S X))'), 0.27),\
     (Tree.fromstring('(A (R (T V W) U) (S X))'), 0.27)]
    self.assertIn(target_weighted_tree, expected_weighted_trees)

class GenerateTreesTestCase(unittest.TestCase):
  def setUp(self):
    self.S = 'q'

  def test_OneProductionOneTree(self):
    # Define Transducer rules. This is the alphabet Sigma of the wRTG.
    rule5 = XTRule(self.S, tree_or_string('F'),
                        tree_or_string('V'),
                        {}, 0.9)
    rules = [rule5]
    # Non-terminals (states) of wRTG.
    non_terminals = [(self.S, (), ())]

    # Productions of wRTG.
    productions = []
    deriv_rhs1 = RHS(rules[0], [])
    productions.append(\
      Production((self.S, (), ()), deriv_rhs1, rules[0].weight))

    initial_state = (self.S, (), ())
    wrtg = wRTG(rules, non_terminals, initial_state, productions)
    generated_trees = [tree for tree in wrtg.GenerateTrees()]
    expected_generated_trees = [(tree_or_string('V'), 0.9)]
    # Comparing trees.
    self.assertEqual(expected_generated_trees[0][0], generated_trees[0][0])
    # Comparing their weights (using approximation).
    self.assertAlmostEqual(expected_generated_trees[0][1], generated_trees[0][1])
    # Only should generate one tree.
    self.assertEqual(1, len(generated_trees))

  def test_TwoProductionsTwoTrees(self):
    # Define Transducer rules. This is the alphabet Sigma of the wRTG.
    rule5 = XTRule(self.S, tree_or_string('F'),
                        tree_or_string('V'),
                        {}, 0.9)
    rule6 = XTRule(self.S, tree_or_string('F'),
                        tree_or_string('W'),
                        {}, 0.8)
    rules = [rule5, rule6]
    # Non-terminals (states) of wRTG.
    non_terminals = [(self.S, (), ())]

    # Productions of wRTG.
    productions = []
    deriv_rhs1 = RHS(rules[0], [])
    productions.append(\
      Production((self.S, (), ()), deriv_rhs1, rules[0].weight))
    deriv_rhs2 = RHS(rules[1], [])
    productions.append(\
      Production((self.S, (), ()), deriv_rhs2, rules[1].weight))

    initial_state = (self.S, (), ())
    wrtg = wRTG(rules, non_terminals, initial_state, productions)
    generated_trees = [tree for tree in wrtg.GenerateTrees()]
    expected_generated_trees = [(tree_or_string('V'), 0.9),\
                                (tree_or_string('W'), 0.8)]
    # Comparing trees.
    self.assertEqual(expected_generated_trees[0][0], generated_trees[0][0])
    self.assertEqual(expected_generated_trees[1][0], generated_trees[1][0])
    # Comparing their weights (using approximation).
    self.assertAlmostEqual(expected_generated_trees[0][1], generated_trees[0][1])
    self.assertAlmostEqual(expected_generated_trees[1][1], generated_trees[1][1])
    # Only should generate two trees.
    self.assertEqual(2, len(generated_trees))

  def test_ExampleConstrainedOutput(self):
    # Define Transducer rules. This is the alphabet Sigma of the wRTG.
    rule1 = XTRule(self.S, tree_or_string('(A ?x0| ?x1|)'),
                        tree_or_string('(A (R ?x1| ?x0|) (S X))'),
                        {(0, 0) : self.S, (0, 1) : self.S}, 1.0)
    rule2 = XTRule(self.S, tree_or_string('(B ?x0| ?x1|)'),
                        tree_or_string('U'),
                        {}, 1.0)
    rule3 = XTRule(self.S, tree_or_string('(C ?x0| ?x1|)'),
                        tree_or_string('(T ?x0| ?x1|)'),
                        {(0,) : self.S, (1,) : self.S}, 0.6)
    rule4 = XTRule(self.S, tree_or_string('(C ?x0| ?x1|)'),
                        tree_or_string('(T ?x1| ?x0|)'),
                        {(0,) : self.S, (1,) : self.S}, 0.4)
    rule5 = XTRule(self.S, tree_or_string('F'),
                        tree_or_string('V'),
                        {}, 0.9)
    rule6 = XTRule(self.S, tree_or_string('F'),
                        tree_or_string('W'),
                        {}, 0.1)
    rule7 = XTRule(self.S, tree_or_string('G'),
                        tree_or_string('V'),
                        {}, 0.5)
    rule8 = XTRule(self.S, tree_or_string('G'),
                        tree_or_string('W'),
                        {}, 0.5)
    rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8]
    # Non-terminals (states) of wRTG.
    non_terminals = [(self.S, (), ()), \
                     (self.S, (0, ), (0, 1)), \
                     (self.S, (1, ), (0, 0)), \
                     (self.S, (1, 0), (0, 0, 0)), \
                     (self.S, (1, 0), (0, 0, 1)), \
                     (self.S, (1, 1), (0, 0, 0)), \
                     (self.S, (1, 1), (0, 0, 1))]

    # Productions of wRTG.
    productions = []
    deriv_rhs1 = RHS(rules[0], [(self.S, (0, ), (0, 1)), \
                                (self.S, (1, ), (0, 0))])
    productions.append(\
      Production((self.S, (), ()), deriv_rhs1, rules[0].weight))

    deriv_rhs2 = RHS(rules[1])
    productions.append(\
      Production((self.S, (0,), (0, 1)), deriv_rhs2, rules[1].weight))

    deriv_rhs3 = RHS(rules[2], [(self.S, (1, 0), (0, 0, 0)), \
                                (self.S, (1, 1), (0, 0, 1))])
    productions.append(\
      Production((self.S, (1,), (0, 0)), deriv_rhs3, rules[2].weight))

    deriv_rhs4 = RHS(rules[3], [(self.S, (1, 0), (0, 0, 1)), \
                                (self.S, (1, 1), (0, 0, 0))])
    productions.append(\
      Production((self.S, (1,), (0, 0)), deriv_rhs4, rules[3].weight))

    deriv_rhs5 = RHS(rules[4])
    productions.append(\
      Production((self.S, (1, 0), (0, 0, 0)), deriv_rhs5, rules[4].weight))

    deriv_rhs8 = RHS(rules[7])
    productions.append(\
      Production((self.S, (1, 1), (0, 0, 1)), deriv_rhs8, rules[7].weight))

    deriv_rhs6 = RHS(rules[5])
    productions.append(\
      Production((self.S, (1, 0), (0, 0, 1)), deriv_rhs6, rules[5].weight))

    deriv_rhs7 = RHS(rules[6])
    productions.append(\
      Production((self.S, (1, 1), (0, 0, 0)), deriv_rhs7, rules[6].weight))

    initial_state = (self.S, (), ())
    wrtg = wRTG(rules, non_terminals, initial_state, productions)
    generated_trees = [tree for tree in wrtg.GenerateTrees()]
    expected_generated_trees = [(Tree.fromstring('(A (R (T V W) U) (S X))'), 0.27),\
                                (Tree.fromstring('(A (R (T V W) U) (S X))'), 0.02)]
    # Comparing trees.
    self.assertEqual(expected_generated_trees[0][0], generated_trees[0][0])
    self.assertEqual(expected_generated_trees[1][0], generated_trees[1][0])
    # Comparing their weights (using approximation).
    self.assertAlmostEqual(expected_generated_trees[0][1], generated_trees[0][1])
    self.assertAlmostEqual(expected_generated_trees[1][1], generated_trees[1][1])
    # Only should generate two trees.
    self.assertEqual(2, len(generated_trees))

  def test_ExampleUnconstrainedOutput(self):
    rules = loadrules("training/ttt_example_fig5.yaml", fmt='yaml')
    intree = immutable(tree_or_string('(A (B D E) (C F G))'))
    transducer = xT(self.S, rules)
    outtree = None
    q, i, o = self.S, (), ()
    initial_state = q, i
    productions, non_terminals = transducer.Produce(intree, outtree, q, i, o)
    wrtg = wRTG(transducer.rules, non_terminals, initial_state, productions)
    prunned_wrtg = wrtg.Prune()
    expected_weighted_trees = [\
     (Tree.fromstring('(A (R (T V V) U) (S X))'), 0.27),\
     (Tree.fromstring('(A (R (T V W) U) (S X))'), 0.27),\
     (Tree.fromstring('(A (R (T W V) U) (S X))'), 0.03),\
     (Tree.fromstring('(A (R (T W W) U) (S X))'), 0.03),\
     (Tree.fromstring('(A (R (T V V) U) (S X))'), 0.180),\
     (Tree.fromstring('(A (R (T W V) U) (S X))'), 0.180),\
     (Tree.fromstring('(A (R (T V W) U) (S X))'), 0.020),\
     (Tree.fromstring('(A (R (T W W) U) (S X))'), 0.020)]
    generated_weighted_trees = [tree for tree in prunned_wrtg.GenerateTrees()]
    self.assertEqual(len(expected_weighted_trees), len(generated_weighted_trees),
                     msg='{0} != {1}. Generated trees: \n{2}'\
                     .format(len(expected_weighted_trees),
                             len(generated_weighted_trees),
                             '\n'.join([repr(x) for x in generated_weighted_trees])))
    self.maxDiff = None
    for generated_weighted_tree in generated_weighted_trees:
      generated_weighted_tree = (generated_weighted_tree[0],
                                 round(generated_weighted_tree[1], 7))
      self.assertIn(generated_weighted_tree, expected_weighted_trees)

class SourceProjectionFromDerivationMixTestCase(unittest.TestCase):
  def setUp(self):
    self.S = 'q0'

  def test_1ProdTerm2Term(self):
    rule0 = XTRule('t', tree_or_string('a'),
                        tree_or_string('b'),
                        {}, 0.8)
    rhs = RHS(rule0, [])
    start_symbol = ('t', (), ())
    production = Production(start_symbol, rhs, rule0.weight)
    weighted_tree = SourceProjectionFromDerivationMix([production])
    expected_weighted_tree = ('a', 0.8)
    self.assertEqual(expected_weighted_tree, weighted_tree)

  def test_1ProdTerm2Pre(self):
    rule0 = XTRule('t', tree_or_string('a'),
                        tree_or_string('(B b)'),
                        {}, 0.7)
    rhs = RHS(rule0, [])
    start_symbol = ('t', (), ())
    production = Production(start_symbol, rhs, rule0.weight)
    weighted_tree = SourceProjectionFromDerivationMix([production])
    expected_weighted_tree = (tree_or_string('a'), 0.7)
    self.assertEqual(expected_weighted_tree, weighted_tree) 

  def test_1ProdPre2Term(self):
    rule0 = XTRule('t', tree_or_string('(A a)'),
                        tree_or_string('b'),
                        {}, 0.7)
    rhs = RHS(rule0, [])
    start_symbol = ('t', (), ())
    production = Production(start_symbol, rhs, rule0.weight)
    weighted_tree = SourceProjectionFromDerivationMix([production])
    expected_weighted_tree = (tree_or_string('(A a)'), 0.7)
    self.assertEqual(expected_weighted_tree, weighted_tree) 

  def test_2ProdPre2PreTerm2Term(self):
    rule0 = XTRule(self.S, tree_or_string('(A ?x0|)'),
                        tree_or_string('(B ?x0|)'),
                        {(0,) : 't'}, 0.6)
    rule1 = XTRule('t', tree_or_string('a'),
                        tree_or_string('b'),
                        {}, 0.5)
    # Production 1
    rhs0 = RHS(rule0, [('t', (0,), (0,))])
    start_symbol = (self.S, (), ())
    production0 = Production(start_symbol, rhs0, rule0.weight)
    # Production 2
    rhs1 = RHS(rule1, [])
    start_symbol = ('t', (0,), (0,))
    production1 = Production(start_symbol, rhs1, rule1.weight)

    weighted_tree = SourceProjectionFromDerivationMix([production0, production1])
    expected_weighted_tree = (tree_or_string('(B a)'), 0.3)
    self.assertEqual(expected_weighted_tree, weighted_tree) 

  def test_3ProdNon2NonTerm2TermTerm2Term(self):
    # Transforming English into English with Spanish word order.
    rule0 = XTRule(self.S, tree_or_string('(NP (JJ ?x0|) (NN ?x1|))'),
                        tree_or_string('(NPP (NNN ?x1|) (JJJ ?x0|))'),
                        {(0, 0) : 't', (1, 0) : 't'}, 0.6)
    rule1 = XTRule('t', tree_or_string('nice'),
                        tree_or_string('bonita'),
                        {}, 1.0)
    rule2 = XTRule('t', tree_or_string('house'),
                        tree_or_string('casa'),
                        {}, 1.0)
    # Production 1
    rhs0 = RHS(rule0, [('t', (0,), (0,))])
    start_symbol = (self.S, (), ())
    production0 = Production(start_symbol, rhs0, rule0.weight)
    # Production 2
    rhs1 = RHS(rule1, [])
    start_symbol = ('t', (0, 0), (1, 0))
    production1 = Production(start_symbol, rhs1, rule1.weight)
    # Production 3
    rhs2 = RHS(rule2, [])
    start_symbol = ('t', (1, 0), (0, 0))
    production2 = Production(start_symbol, rhs2, rule2.weight)

    weighted_tree = \
      SourceProjectionFromDerivationMix([production0, production1, production2])
    expected_weighted_tree = (tree_or_string('(NPP (NNN house) (JJJ nice))'), 0.6)
    self.assertEqual(expected_weighted_tree, weighted_tree) 

  def test_3ProdNon2NonPre2PrePre2Term(self):
    rule0 = XTRule(self.S, tree_or_string('(A ?x0|B ?x1|C)'),
                        tree_or_string('(A ?x0| (C (D d) (E ?x1|)))'),
                        {(0,) : 't', (1, 1, 0) : 't'}, 0.6)
    rule1 = XTRule('t', tree_or_string('(B b)'),
                        tree_or_string('(B b)'),
                        {}, 0.5)
    rule2 = XTRule('t', tree_or_string('(C c)'),
                        tree_or_string('e'),
                        {}, 0.5)
    # Production 1
    rhs0 = RHS(rule0, [])
    start_symbol = (self.S, (), ())
    production0 = Production(start_symbol, rhs0, rule0.weight)
    # Production 2
    rhs1 = RHS(rule1, [])
    start_symbol = ('t', (0,), (0,))
    production1 = Production(start_symbol, rhs1, rule1.weight)
    # Production 3
    rhs2 = RHS(rule2, [])
    start_symbol = ('t', (1,), (1, 1, 0))
    production2 = Production(start_symbol, rhs2, rule2.weight)

    weighted_tree = \
      SourceProjectionFromDerivationMix([production0, production1, production2])
    expected_weighted_tree = (tree_or_string('(A (B b) (C (D d) (E (C c))))'), 0.15)
    self.assertEqual(expected_weighted_tree, weighted_tree) 

  def test_SourceParticleNotPreservedOriginalPreterminals(self):
    rule0 = XTRule(self.S, tree_or_string('(A ?x0|B (PT wo) ?x1|C)'),
                        tree_or_string('(AA ?x1| ?x0|)'),
                        {(0,) : 't', (1,) : 't'}, 0.6)
    rule1 = XTRule('t', tree_or_string('(B b)'),
                        tree_or_string('(BB bb)'),
                        {}, 0.5)
    rule2 = XTRule('t', tree_or_string('(C c)'),
                        tree_or_string('(CC cc)'),
                        {}, 0.5)
    # Production 1
    rhs0 = RHS(rule0, [])
    start_symbol = (self.S, (), ())
    production0 = Production(start_symbol, rhs0, rule0.weight)
    # Production 2
    rhs1 = RHS(rule1, [])
    start_symbol = ('t', (0,), (1,))
    production1 = Production(start_symbol, rhs1, rule1.weight)
    # Production 3
    rhs2 = RHS(rule2, [])
    start_symbol = ('t', (2,), (0,))
    production2 = Production(start_symbol, rhs2, rule2.weight)

    weighted_tree = \
      SourceProjectionFromDerivationMix([production0, production1, production2])
    expected_weighted_tree = (tree_or_string('(AA (C c) (B b))'), 0.15)
    self.assertEqual(expected_weighted_tree, weighted_tree) 

  def test_SourceParticleNotPreservedNewPreterminals(self):
    rule0 = XTRule(self.S, tree_or_string('(A ?x0|B (PT wo) ?x1|C)'),
                           tree_or_string('(AA ?x1| ?x0|)'),
                           {(0,) : 't', (1,) : 't'}, 0.6)
    rule1 = XTRule(self.S, tree_or_string('(B ?x0|)'),
                           tree_or_string('(BB ?x0|)'),
                           {(0,) : 't'}, 1.0)
    rule2 = XTRule(self.S, tree_or_string('(C ?x0|)'),
                           tree_or_string('(CC ?x0|)'),
                           {(0,) : 't'}, 1.0)
    rule3 = XTRule(self.S, tree_or_string('b'),
                           tree_or_string('bb'),
                           {}, 0.5)
    rule4 = XTRule(self.S, tree_or_string('c'),
                           tree_or_string('cc'),
                           {}, 0.5)
    # Production 1
    rhs0 = RHS(rule0, [])
    start_symbol = (self.S, (), ())
    production0 = Production(start_symbol, rhs0, rule0.weight)
    # Production 2
    rhs1 = RHS(rule1, [])
    start_symbol = (self.S, (0,), (1,))
    production1 = Production(start_symbol, rhs1, rule1.weight)
    # Production 3
    rhs2 = RHS(rule2, [])
    start_symbol = (self.S, (2,), (0,))
    production2 = Production(start_symbol, rhs2, rule2.weight)
    # Production 4
    rhs3 = RHS(rule3, [])
    start_symbol = ('t', (0,0), (1,0))
    production3 = Production(start_symbol, rhs3, rule3.weight)
    # Production 5
    rhs4 = RHS(rule4, [])
    start_symbol = ('t', (2,0), (0,0))
    production4 = Production(start_symbol, rhs4, rule4.weight)

    productions = [production0, production1, production2, production3, production4]

    weighted_tree = SourceProjectionFromDerivationMix(productions)
    expected_weighted_tree = (tree_or_string('(AA (CC c) (BB b))'), 0.15)
    self.assertEqual(expected_weighted_tree, weighted_tree) 

class SourceProjectionFromDerivationStrictTestCase(unittest.TestCase):
  def setUp(self):
    self.S = 'q0'

  def test_1ProdTerm2Term(self):
    rule0 = XTRule('t', tree_or_string('a'),
                        tree_or_string('b'),
                        {}, 0.8)
    rhs = RHS(rule0, [])
    start_symbol = ('t', (), ())
    production = Production(start_symbol, rhs, rule0.weight)
    weighted_tree = SourceProjectionFromDerivationStrict([production])
    expected_weighted_tree = ('a', 0.8)
    self.assertEqual(expected_weighted_tree, weighted_tree)

  def test_1ProdTerm2Pre(self):
    rule0 = XTRule('t', tree_or_string('a'),
                        tree_or_string('(B b)'),
                        {}, 0.7)
    rhs = RHS(rule0, [])
    start_symbol = ('t', (), ())
    production = Production(start_symbol, rhs, rule0.weight)
    weighted_tree = SourceProjectionFromDerivationStrict([production])
    expected_weighted_tree = (tree_or_string('a'), 0.7)
    self.assertEqual(expected_weighted_tree, weighted_tree) 

  def test_1ProdPre2Term(self):
    rule0 = XTRule('t', tree_or_string('(A a)'),
                        tree_or_string('b'),
                        {}, 0.7)
    rhs = RHS(rule0, [])
    start_symbol = ('t', (), ())
    production = Production(start_symbol, rhs, rule0.weight)
    weighted_tree = SourceProjectionFromDerivationStrict([production])
    expected_weighted_tree = (tree_or_string('(A a)'), 0.7)
    self.assertEqual(expected_weighted_tree, weighted_tree) 

  def test_2ProdPre2PreTerm2Term(self):
    rule0 = XTRule(self.S, tree_or_string('(A ?x0|)'),
                        tree_or_string('(B ?x0|)'),
                        {(0,) : 't'}, 0.6)
    rule1 = XTRule('t', tree_or_string('a'),
                        tree_or_string('b'),
                        {}, 0.5)
    # Production 1
    rhs0 = RHS(rule0, [('t', (0,), (0,))])
    start_symbol = (self.S, (), ())
    production0 = Production(start_symbol, rhs0, rule0.weight)
    # Production 2
    rhs1 = RHS(rule1, [])
    start_symbol = ('t', (0,), (0,))
    production1 = Production(start_symbol, rhs1, rule1.weight)

    weighted_tree = SourceProjectionFromDerivationStrict([production0, production1])
    expected_weighted_tree = (tree_or_string('(A a)'), 0.3)
    self.assertEqual(expected_weighted_tree, weighted_tree) 

  def test_3ProdNon2NonTerm2TermTerm2Term(self):
    # Transforming English into English with Spanish word order.
    rule0 = XTRule(self.S, tree_or_string('(NP (JJ ?x0|) (NN ?x1|))'),
                        tree_or_string('(NPP (NNN ?x1|) (JJJ ?x0|))'),
                        {(0, 0) : 't', (1, 0) : 't'}, 0.6)
    rule1 = XTRule('t', tree_or_string('nice'),
                        tree_or_string('bonita'),
                        {}, 1.0)
    rule2 = XTRule('t', tree_or_string('house'),
                        tree_or_string('casa'),
                        {}, 1.0)
    # Production 1
    rhs0 = RHS(rule0, [('t', (0,), (0,))])
    start_symbol = (self.S, (), ())
    production0 = Production(start_symbol, rhs0, rule0.weight)
    # Production 2
    rhs1 = RHS(rule1, [])
    start_symbol = ('t', (0, 0), (1, 0))
    production1 = Production(start_symbol, rhs1, rule1.weight)
    # Production 3
    rhs2 = RHS(rule2, [])
    start_symbol = ('t', (1, 0), (0, 0))
    production2 = Production(start_symbol, rhs2, rule2.weight)

    weighted_tree = \
      SourceProjectionFromDerivationStrict([production0, production1, production2])
    expected_weighted_tree = (tree_or_string('(NP (JJ house) (NN nice))'), 0.6)
    self.assertEqual(expected_weighted_tree, weighted_tree) 

  def test_3ProdNon2NonPre2PrePre2Term(self):
    rule0 = XTRule(self.S, tree_or_string('(A ?x0|B ?x1|C)'),
                        tree_or_string('(A ?x0| (C (D d) (E ?x1|)))'),
                        {(0,) : 't', (1, 1, 0) : 't'}, 0.6)
    rule1 = XTRule('t', tree_or_string('(B b)'),
                        tree_or_string('(B b)'),
                        {}, 0.5)
    rule2 = XTRule('t', tree_or_string('(C c)'),
                        tree_or_string('e'),
                        {}, 0.5)
    # Production 1
    rhs0 = RHS(rule0, [])
    start_symbol = (self.S, (), ())
    production0 = Production(start_symbol, rhs0, rule0.weight)
    # Production 2
    rhs1 = RHS(rule1, [])
    start_symbol = ('t', (0,), (0,))
    production1 = Production(start_symbol, rhs1, rule1.weight)
    # Production 3
    rhs2 = RHS(rule2, [])
    start_symbol = ('t', (1,), (1, 1, 0))
    production2 = Production(start_symbol, rhs2, rule2.weight)

    weighted_tree = \
      SourceProjectionFromDerivationStrict([production0, production1, production2])
    expected_weighted_tree = (tree_or_string('(A (B b) (C c))'), 0.15)
    self.assertEqual(expected_weighted_tree, weighted_tree) 

  def test_SourceParticlePreservedOriginalPreterminals(self):
    rule0 = XTRule(self.S, tree_or_string('(A ?x0|B (PT wo) ?x1|C)'),
                        tree_or_string('(AA ?x1| ?x0|)'),
                        {(0,) : 't', (1,) : 't'}, 0.6)
    rule1 = XTRule('t', tree_or_string('(B b)'),
                        tree_or_string('(BB bb)'),
                        {}, 0.5)
    rule2 = XTRule('t', tree_or_string('(C c)'),
                        tree_or_string('(CC cc)'),
                        {}, 0.5)
    # Production 1
    rhs0 = RHS(rule0, [])
    start_symbol = (self.S, (), ())
    production0 = Production(start_symbol, rhs0, rule0.weight)
    # Production 2
    rhs1 = RHS(rule1, [])
    start_symbol = ('t', (0,), (1,))
    production1 = Production(start_symbol, rhs1, rule1.weight)
    # Production 3
    rhs2 = RHS(rule2, [])
    start_symbol = ('t', (2,), (0,))
    production2 = Production(start_symbol, rhs2, rule2.weight)

    weighted_tree = \
      SourceProjectionFromDerivationStrict([production0, production1, production2])
    expected_weighted_tree = (tree_or_string('(A (C c) (PT wo) (B b))'), 0.15)
    self.assertEqual(expected_weighted_tree, weighted_tree) 

  def test_SourceParticlePreservedNewPreterminals(self):
    rule0 = XTRule(self.S, tree_or_string('(A ?x0|B (PT wo) ?x1|C)'),
                           tree_or_string('(AA ?x1| ?x0|)'),
                           {(0,) : 't', (1,) : 't'}, 0.6)
    rule1 = XTRule(self.S, tree_or_string('(B ?x0|)'),
                           tree_or_string('(BB ?x0|)'),
                           {(0,) : 't'}, 1.0)
    rule2 = XTRule(self.S, tree_or_string('(C ?x0|)'),
                           tree_or_string('(CC ?x0|)'),
                           {(0,) : 't'}, 1.0)
    rule3 = XTRule(self.S, tree_or_string('b'),
                           tree_or_string('bb'),
                           {}, 0.5)
    rule4 = XTRule(self.S, tree_or_string('c'),
                           tree_or_string('cc'),
                           {}, 0.5)
    # Production 1
    rhs0 = RHS(rule0, [])
    start_symbol = (self.S, (), ())
    production0 = Production(start_symbol, rhs0, rule0.weight)
    # Production 2
    rhs1 = RHS(rule1, [])
    start_symbol = (self.S, (0,), (1,))
    production1 = Production(start_symbol, rhs1, rule1.weight)
    # Production 3
    rhs2 = RHS(rule2, [])
    start_symbol = (self.S, (2,), (0,))
    production2 = Production(start_symbol, rhs2, rule2.weight)
    # Production 4
    rhs3 = RHS(rule3, [])
    start_symbol = ('t', (0,0), (1,0))
    production3 = Production(start_symbol, rhs3, rule3.weight)
    # Production 5
    rhs4 = RHS(rule4, [])
    start_symbol = ('t', (2,0), (0,0))
    production4 = Production(start_symbol, rhs4, rule4.weight)

    productions = [production0, production1, production2, production3, production4]

    weighted_tree = SourceProjectionFromDerivationStrict(productions)
    expected_weighted_tree = (tree_or_string('(A (C c) (PT wo) (B b))'), 0.15)
    self.assertEqual(expected_weighted_tree, weighted_tree) 

  def test_ReorderingTwoLevels(self):
    rule0 = XTRule(self.S, tree_or_string('(S ?x0|NP ?x1|VP)'),
                           tree_or_string('(S ?x1|VP ?x0|NP)'),
                           {(0,) : 'q0', (1,) : 'q0'}, 0.6)
    rule1 = XTRule(self.S, tree_or_string('(NP ?x0|JJ ?x1|NN)'),
                           tree_or_string('(NP ?x1|NN ?x0|JJ)'),
                           {(0,) : 'q0', (1,) : 'q0'}, 0.6)
    rule2 = XTRule(self.S, tree_or_string('(JJ ?x0|)'),
                           tree_or_string('(JJ ?x0|)'),
                           {(0,) : 't'}, 1.0)
    rule3 = XTRule(self.S, tree_or_string('(NN ?x0|)'),
                           tree_or_string('(NN ?x0|)'),
                           {(0,) : 't'}, 1.0)
    rule4 = XTRule(self.S, tree_or_string('house'),
                           tree_or_string('casa'),
                           {}, 0.5)
    rule5 = XTRule(self.S, tree_or_string('nice'),
                           tree_or_string('bonita'),
                           {}, 0.5)
    rule6 = XTRule(self.S, tree_or_string('(VP ?x0|VBZ ?x1|JJ)'),
                           tree_or_string('(VP ?x1|JJ ?x0|VBZ)'),
                           {(0,) : 'q0', (1,) : 'q0'}, 0.6)
    rule7 = XTRule(self.S, tree_or_string('(VBZ ?x0|)'),
                           tree_or_string('(VBZ ?x0|)'),
                           {(0,) : 't'}, 1.0)
    rule8 = XTRule(self.S, tree_or_string('(JJ ?x0|)'),
                           tree_or_string('(JJ ?x0|)'),
                           {(0,) : 't'}, 1.0)
    rule9 = XTRule(self.S, tree_or_string('is'),
                           tree_or_string('es'),
                           {}, 0.5)
    rule10 = XTRule(self.S, tree_or_string('red'),
                            tree_or_string('roja'),
                            {}, 0.5)
    # Production 0
    rhs0 = RHS(rule0, [])
    start_symbol = (self.S, (), ())
    production0 = Production(start_symbol, rhs0, rule0.weight)
    # Production 1
    rhs1 = RHS(rule1, [])
    start_symbol = (self.S, (0,), (1,))
    production1 = Production(start_symbol, rhs1, rule1.weight)
    # Production 2
    rhs2 = RHS(rule2, [])
    start_symbol = (self.S, (0,0), (1,1))
    production2 = Production(start_symbol, rhs2, rule2.weight)
    # Production 3
    rhs3 = RHS(rule3, [])
    start_symbol = (self.S, (0,1), (1,0))
    production3 = Production(start_symbol, rhs3, rule3.weight)
    # Production 4
    rhs4 = RHS(rule4, [])
    start_symbol = (self.S, (0,1,0), (1,1,0))
    production4 = Production(start_symbol, rhs4, rule4.weight)
    # Production 5
    rhs5 = RHS(rule5, [])
    start_symbol = (self.S, (0,0,0), (1,1,0))
    production5 = Production(start_symbol, rhs5, rule5.weight)
    # Production 6
    rhs6 = RHS(rule6, [])
    start_symbol = (self.S, (1,), (0,))
    production6 = Production(start_symbol, rhs6, rule6.weight)
    # Production 7
    rhs7 = RHS(rule7, [])
    start_symbol = (self.S, (1,0), (0,1))
    production7 = Production(start_symbol, rhs7, rule7.weight)
    # Production 8
    rhs8 = RHS(rule8, [])
    start_symbol = (self.S, (1,1), (0,0))
    production8 = Production(start_symbol, rhs8, rule8.weight)
    # Production 9
    rhs9 = RHS(rule9, [])
    start_symbol = (self.S, (1,0,0), (0,1,0))
    production9 = Production(start_symbol, rhs9, rule9.weight)
    # Production 10
    rhs10 = RHS(rule10, [])
    start_symbol = (self.S, (1,1,0), (0,0,0))
    production10 = Production(start_symbol, rhs10, rule10.weight)

    productions = [production0, production1, production2, production3, production4,
                   production5, production6, production7, production8, production9,
                   production10]

    weighted_tree = SourceProjectionFromDerivationStrict(productions)
    expected_tree = tree_or_string('(S (VP (JJ red) (VBZ is)) (NP (NN house) (JJ nice)))')
    self.assertEqual(expected_tree, weighted_tree[0]) 

class TargetProjectionFromDerivationTestCase(unittest.TestCase):
  def setUp(self):
    self.S = 'q0'

  def test_1ProdTerm2Term(self):
    rule0 = XTRule('t', tree_or_string('a'),
                        tree_or_string('b'),
                        {}, 0.8)
    rhs = RHS(rule0, [])
    start_symbol = ('t', (), ())
    production = Production(start_symbol, rhs, rule0.weight)
    weighted_tree = TargetProjectionFromDerivation([production])
    expected_weighted_tree = ('b', 0.8)
    self.assertEqual(expected_weighted_tree, weighted_tree)

  def test_1ProdTerm2Pre(self):
    rule0 = XTRule('t', tree_or_string('a'),
                        tree_or_string('(B b)'),
                        {}, 0.7)
    rhs = RHS(rule0, [])
    start_symbol = ('t', (), ())
    production = Production(start_symbol, rhs, rule0.weight)
    weighted_tree = TargetProjectionFromDerivation([production])
    expected_weighted_tree = (tree_or_string('(B b)'), 0.7)
    self.assertEqual(expected_weighted_tree, weighted_tree) 

  def test_1ProdPre2Term(self):
    rule0 = XTRule('t', tree_or_string('(A a)'),
                        tree_or_string('b'),
                        {}, 0.7)
    rhs = RHS(rule0, [])
    start_symbol = ('t', (), ())
    production = Production(start_symbol, rhs, rule0.weight)
    weighted_tree = TargetProjectionFromDerivation([production])
    expected_weighted_tree = (tree_or_string('b'), 0.7)
    self.assertEqual(expected_weighted_tree, weighted_tree) 

  def test_2ProdPret2Pre(self):
    rule0 = XTRule(self.S, tree_or_string('(A ?x0|)'),
                        tree_or_string('(B ?x0|)'),
                        {(0,) : 't'}, 0.6)
    rule1 = XTRule('t', tree_or_string('a'),
                        tree_or_string('b'),
                        {}, 0.5)
    # Production 1
    rhs0 = RHS(rule0, [('t', (0,), (0,))])
    start_symbol = (self.S, (), ())
    production0 = Production(start_symbol, rhs0, rule0.weight)
    # Production 2
    rhs1 = RHS(rule1, [])
    start_symbol = ('t', (0,), (0,))
    production1 = Production(start_symbol, rhs1, rule1.weight)

    weighted_tree = TargetProjectionFromDerivation([production0, production1])
    expected_weighted_tree = (tree_or_string('(B b)'), 0.3)
    self.assertEqual(expected_weighted_tree, weighted_tree) 

  def test_3ProdNon2Non(self):
    rule0 = XTRule(self.S, tree_or_string('(A ?x0|B ?x1|C)'),
                        tree_or_string('(A ?x0| (C (D d) (E ?x1|)))'),
                        {(0,) : 't', (1, 1, 0) : 't'}, 0.6)
    rule1 = XTRule('t', tree_or_string('(B b)'),
                        tree_or_string('(B b)'),
                        {}, 0.5)
    rule2 = XTRule('t', tree_or_string('(C c)'),
                        tree_or_string('e'),
                        {}, 0.5)
    # Production 1
    rhs0 = RHS(rule0, [])
    start_symbol = (self.S, (), ())
    production0 = Production(start_symbol, rhs0, rule0.weight)
    # Production 2
    rhs1 = RHS(rule1, [])
    start_symbol = ('t', (0,), (0,))
    production1 = Production(start_symbol, rhs1, rule1.weight)
    # Production 3
    rhs2 = RHS(rule2, [])
    start_symbol = ('t', (1,), (1, 1, 0))
    production2 = Production(start_symbol, rhs2, rule2.weight)

    weighted_tree = \
      TargetProjectionFromDerivation([production0, production1, production2])
    expected_weighted_tree = (tree_or_string('(A (B b) (C (D d) (E e)))'), 0.15)
    self.assertEqual(expected_weighted_tree, weighted_tree) 

class InsideOutsideTestCase(unittest.TestCase):
  def setUp(self):
    self.S = 'q0'
    # Define Transducer rules. This is the alphabet Sigma of the wRTG.
    rule1 = XTRule(self.S, tree_or_string('(A ?x0| ?x1|)'),
                        tree_or_string('(A (R ?x1| ?x0|) (S X))'),
                        {(0, 0) : self.S, (0, 1) : self.S}, 1.0)
    rule2 = XTRule(self.S, tree_or_string('(B ?x0| ?x1|)'),
                        tree_or_string('U'),
                        {}, 1.0)
    rule3 = XTRule(self.S, tree_or_string('(C ?x0| ?x1|)'),
                        tree_or_string('(T ?x0| ?x1|)'),
                        {(0,) : self.S, (1,) : self.S}, 0.6)
    rule4 = XTRule(self.S, tree_or_string('(C ?x0| ?x1|)'),
                        tree_or_string('(T ?x1| ?x0|)'),
                        {(0,) : self.S, (1,) : self.S}, 0.4)
    rule5 = XTRule(self.S, tree_or_string('F'),
                        tree_or_string('V'),
                        {}, 0.9)
    rule6 = XTRule(self.S, tree_or_string('F'),
                        tree_or_string('W'),
                        {}, 0.1)
    rule7 = XTRule(self.S, tree_or_string('G'),
                        tree_or_string('V'),
                        {}, 0.5)
    rule8 = XTRule(self.S, tree_or_string('G'),
                        tree_or_string('W'),
                        {}, 0.5)
    self.rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8]
    # Non-terminals (states) of wRTG.
    non_terminals = [(self.S, (), ()), \
                     (self.S, (0, ), (0, 1)), \
                     (self.S, (1, ), (0, 0)), \
                     (self.S, (1, 0), (0, 0, 0)), \
                     (self.S, (1, 0), (0, 0, 1)), \
                     (self.S, (1, 1), (0, 0, 0)), \
                     (self.S, (1, 1), (0, 0, 1))]

    # Productions of wRTG.
    productions = []
    deriv_rhs1 = RHS(self.rules[0], [(self.S, (0, ), (0, 1)), \
                                (self.S, (1, ), (0, 0))])
    productions.append(\
      Production((self.S, (), ()), deriv_rhs1, self.rules[0].weight))

    deriv_rhs2 = RHS(self.rules[1])
    productions.append(\
      Production((self.S, (0,), (0, 1)), deriv_rhs2, self.rules[1].weight))

    deriv_rhs3 = RHS(self.rules[2], [(self.S, (1, 0), (0, 0, 0)), \
                                (self.S, (1, 1), (0, 0, 1))])
    productions.append(\
      Production((self.S, (1,), (0, 0)), deriv_rhs3, self.rules[2].weight))

    deriv_rhs4 = RHS(self.rules[3], [(self.S, (1, 0), (0, 0, 1)), \
                                (self.S, (1, 1), (0, 0, 0))])
    productions.append(\
      Production((self.S, (1,), (0, 0)), deriv_rhs4, self.rules[3].weight))

    deriv_rhs5 = RHS(self.rules[4])
    productions.append(\
      Production((self.S, (1, 0), (0, 0, 0)), deriv_rhs5, self.rules[4].weight))

    deriv_rhs8 = RHS(self.rules[7])
    productions.append(\
      Production((self.S, (1, 1), (0, 0, 1)), deriv_rhs8, self.rules[7].weight))

    deriv_rhs6 = RHS(self.rules[5])
    productions.append(\
      Production((self.S, (1, 0), (0, 0, 1)), deriv_rhs6, self.rules[5].weight))

    deriv_rhs7 = RHS(self.rules[6])
    productions.append(\
      Production((self.S, (1, 1), (0, 0, 0)), deriv_rhs7, self.rules[6].weight))

    initial_state = (self.S, (), ())
    self.wrtg = wRTG(self.rules, non_terminals, initial_state, productions)

  def test_MaxInside(self):
    self.wrtg.ComputeMaxInsideWeights()

    self.assertAlmostEqual(0.27, self.wrtg.GetMaxInside((self.S, (), ()))[0])
    self.assertAlmostEqual(0.27, self.wrtg.GetMaxInside((self.S, (1,), (0, 0)))[0])
    self.assertAlmostEqual(1.0,  self.wrtg.GetMaxInside((self.S, (0,), (0, 1)))[0])
    self.assertAlmostEqual(0.5,  self.wrtg.GetMaxInside((self.S, (1, 1), (0, 0, 0)))[0])
    self.assertAlmostEqual(0.1,  self.wrtg.GetMaxInside((self.S, (1, 0), (0, 0, 1)))[0])
    self.assertAlmostEqual(0.5,  self.wrtg.GetMaxInside((self.S, (1, 1), (0, 0, 1)))[0])
    self.assertAlmostEqual(0.9,  self.wrtg.GetMaxInside((self.S, (1, 0), (0, 0, 0)))[0])

  def test_Inside(self):
    self.wrtg.ComputeInsideWeights()
    self.assertAlmostEqual(0.29, self.wrtg.GetInside((self.S, (), ())))
    self.assertAlmostEqual(0.29, self.wrtg.GetInside((self.S, (1,), (0, 0))))
    self.assertAlmostEqual(1.0,  self.wrtg.GetInside((self.S, (0,), (0, 1))))
    self.assertAlmostEqual(0.5,  self.wrtg.GetInside((self.S, (1, 1), (0, 0, 0))))
    self.assertAlmostEqual(0.1,  self.wrtg.GetInside((self.S, (1, 0), (0, 0, 1))))
    self.assertAlmostEqual(0.5,  self.wrtg.GetInside((self.S, (1, 1), (0, 0, 1))))
    self.assertAlmostEqual(0.9,  self.wrtg.GetInside((self.S, (1, 0), (0, 0, 0))))

  def test_InsideWithFeats(self):
    # Set sparse features for rules (feat_id, feat_val).
    self.rules[0].features = [(0, 1), (10, 1)]
    self.rules[1].features = [(1, 1), (11, 1)]
    self.rules[2].features = [(2, 1), (12, 1)]
    self.rules[3].features = [(3, 1), (13, 1)]
    self.rules[4].features = [(4, 1), (14, 1)]
    self.rules[5].features = [(5, 1), (15, 1)]
    self.rules[6].features = [(6, 1), (16, 1)]
    self.rules[7].features = [(7, 1), (17, 1)]
    # Set feature weights.
    feat_weights = [1.0, 1.0, .6, .4, .9, .1, .5, .5, .0, .0,
                    .0, .0, .0, .0, .0, .0, .0, .0]
    feat_weights = {i : v for i, v in enumerate(feat_weights)}
    # Compute Inside algorithm using rules with features.
    self.wrtg.ComputeInsideWeightsWithFeats(feat_weights)

    self.assertAlmostEqual(0.29, self.wrtg.GetInside((self.S, (), ())))
    self.assertAlmostEqual(0.29, self.wrtg.GetInside((self.S, (1,), (0, 0))))
    self.assertAlmostEqual(1.0,  self.wrtg.GetInside((self.S, (0,), (0, 1))))
    self.assertAlmostEqual(0.5,  self.wrtg.GetInside((self.S, (1, 1), (0, 0, 0))))
    self.assertAlmostEqual(0.1,  self.wrtg.GetInside((self.S, (1, 0), (0, 0, 1))))
    self.assertAlmostEqual(0.5,  self.wrtg.GetInside((self.S, (1, 1), (0, 0, 1))))
    self.assertAlmostEqual(0.9,  self.wrtg.GetInside((self.S, (1, 0), (0, 0, 0))))

    # Change feature weights that preserve the final score, and re-run Inside.
    feat_weights = [.5, .3, .6, .2, .9, .05, .1, .0, .66666, .77777,
                    .5, .7, .0, .2, .0, .05, .4, .5]
    feat_weights = {i : v for i, v in enumerate(feat_weights)}
    self.wrtg.ComputeInsideWeightsWithFeats(feat_weights)

    self.assertAlmostEqual(0.29, self.wrtg.GetInside((self.S, (), ())))
    self.assertAlmostEqual(0.29, self.wrtg.GetInside((self.S, (1,), (0, 0))))
    self.assertAlmostEqual(1.0,  self.wrtg.GetInside((self.S, (0,), (0, 1))))
    self.assertAlmostEqual(0.5,  self.wrtg.GetInside((self.S, (1, 1), (0, 0, 0))))
    self.assertAlmostEqual(0.1,  self.wrtg.GetInside((self.S, (1, 0), (0, 0, 1))))
    self.assertAlmostEqual(0.5,  self.wrtg.GetInside((self.S, (1, 1), (0, 0, 1))))
    self.assertAlmostEqual(0.9,  self.wrtg.GetInside((self.S, (1, 0), (0, 0, 0))))

  def test_Outside(self):
    self.wrtg.ComputeInsideWeights()
    self.wrtg.ComputeOutsideWeights()

    self.assertAlmostEqual(1.0,  self.wrtg.GetOutside((self.S, (), ())))
    self.assertAlmostEqual(1.0,  self.wrtg.GetOutside((self.S, (1,), (0, 0))))
    self.assertAlmostEqual(0.29, self.wrtg.GetOutside((self.S, (0,), (0, 1))))
    self.assertAlmostEqual(0.04, self.wrtg.GetOutside((self.S, (1, 1), (0, 0, 0))))
    self.assertAlmostEqual(0.2,  self.wrtg.GetOutside((self.S, (1, 0), (0, 0, 1))))
    self.assertAlmostEqual(0.54, self.wrtg.GetOutside((self.S, (1, 1), (0, 0, 1))))
    self.assertAlmostEqual(0.3,  self.wrtg.GetOutside((self.S, (1, 0), (0, 0, 0))))

  def test_OutsideWithFeats(self):
    # Set sparse features for rules (feat_id, feat_val).
    self.rules[0].features = [(0, 1), (10, 1)]
    self.rules[1].features = [(1, 1), (11, 1)]
    self.rules[2].features = [(2, 1), (12, 1)]
    self.rules[3].features = [(3, 1), (13, 1)]
    self.rules[4].features = [(4, 1), (14, 1)]
    self.rules[5].features = [(5, 1), (15, 1)]
    self.rules[6].features = [(6, 1), (16, 1)]
    self.rules[7].features = [(7, 1), (17, 1)]
    # Set feature weights.
    feat_weights = [1.0, 1.0, .6, .4, .9, .1, .5, .5, .0, .0,
                    .0, .0, .0, .0, .0, .0, .0, .0]
    feat_weights = {i : v for i, v in enumerate(feat_weights)}
    self.wrtg.ComputeInsideWeightsWithFeats(feat_weights)
    self.wrtg.ComputeOutsideWeightsWithFeats(feat_weights)

    self.assertAlmostEqual(1.0,  self.wrtg.GetOutside((self.S, (), ())))
    self.assertAlmostEqual(1.0,  self.wrtg.GetOutside((self.S, (1,), (0, 0))))
    self.assertAlmostEqual(0.29, self.wrtg.GetOutside((self.S, (0,), (0, 1))))
    self.assertAlmostEqual(0.04, self.wrtg.GetOutside((self.S, (1, 1), (0, 0, 0))))
    self.assertAlmostEqual(0.2,  self.wrtg.GetOutside((self.S, (1, 0), (0, 0, 1))))
    self.assertAlmostEqual(0.54, self.wrtg.GetOutside((self.S, (1, 1), (0, 0, 1))))
    self.assertAlmostEqual(0.3,  self.wrtg.GetOutside((self.S, (1, 0), (0, 0, 0))))

    # Change feature weights that preserve the final score, and re-run Inside.
    feat_weights = [.5, .3, .6, .2, .9, .05, .1, .0, .66666, .77777,
                    .5, .7, .0, .2, .0, .05, .4, .5]
    feat_weights = {i : v for i, v in enumerate(feat_weights)}
    self.wrtg.ComputeInsideWeightsWithFeats(feat_weights)
    self.wrtg.ComputeOutsideWeightsWithFeats(feat_weights)

    self.assertAlmostEqual(1.0,  self.wrtg.GetOutside((self.S, (), ())))
    self.assertAlmostEqual(1.0,  self.wrtg.GetOutside((self.S, (1,), (0, 0))))
    self.assertAlmostEqual(0.29, self.wrtg.GetOutside((self.S, (0,), (0, 1))))
    self.assertAlmostEqual(0.04, self.wrtg.GetOutside((self.S, (1, 1), (0, 0, 0))))
    self.assertAlmostEqual(0.2,  self.wrtg.GetOutside((self.S, (1, 0), (0, 0, 1))))
    self.assertAlmostEqual(0.54, self.wrtg.GetOutside((self.S, (1, 1), (0, 0, 1))))
    self.assertAlmostEqual(0.3,  self.wrtg.GetOutside((self.S, (1, 0), (0, 0, 0))))

  def test_SumWeightsTrees(self):
    self.wrtg.ComputeInsideWeights()
    self.wrtg.ComputeOutsideWeights()

    self.assertAlmostEqual(0.29, self.wrtg.SumWeightsTreesInvolvingProduction(self.wrtg.P[0]))
    self.assertAlmostEqual(0.29, self.wrtg.SumWeightsTreesInvolvingProduction(self.wrtg.P[1]))
    self.assertAlmostEqual(0.27, self.wrtg.SumWeightsTreesInvolvingProduction(self.wrtg.P[2]))
    self.assertAlmostEqual(0.02, self.wrtg.SumWeightsTreesInvolvingProduction(self.wrtg.P[3]))
    self.assertAlmostEqual(0.27, self.wrtg.SumWeightsTreesInvolvingProduction(self.wrtg.P[4]))
    self.assertAlmostEqual(0.27, self.wrtg.SumWeightsTreesInvolvingProduction(self.wrtg.P[5]))
    self.assertAlmostEqual(0.02, self.wrtg.SumWeightsTreesInvolvingProduction(self.wrtg.P[6]))
    self.assertAlmostEqual(0.02, self.wrtg.SumWeightsTreesInvolvingProduction(self.wrtg.P[7]))

  def test_SumWeightsTreesWithFeats(self):
    # Set sparse features for rules (feat_id, feat_val).
    self.rules[0].features = [(0, 1), (10, 1)]
    self.rules[1].features = [(1, 1), (11, 1)]
    self.rules[2].features = [(2, 1), (12, 1)]
    self.rules[3].features = [(3, 1), (13, 1)]
    self.rules[4].features = [(4, 1), (14, 1)]
    self.rules[5].features = [(5, 1), (15, 1)]
    self.rules[6].features = [(6, 1), (16, 1)]
    self.rules[7].features = [(7, 1), (17, 1)]
    # Set feature weights.
    feat_weights = [1.0, 1.0, .6, .4, .9, .1, .5, .5, .0, .0,
                    .0, .0, .0, .0, .0, .0, .0, .0]
    feat_weights = {i : v for i, v in enumerate(feat_weights)}
    self.wrtg.ComputeInsideWeightsWithFeats(feat_weights)
    self.wrtg.ComputeOutsideWeightsWithFeats(feat_weights)

    self.assertAlmostEqual(
      0.29, self.wrtg.SumWeightsTreesInvolvingProductionWithFeats(feat_weights, self.wrtg.P[0]))
    self.assertAlmostEqual(
      0.29, self.wrtg.SumWeightsTreesInvolvingProductionWithFeats(feat_weights, self.wrtg.P[1]))
    self.assertAlmostEqual(
      0.27, self.wrtg.SumWeightsTreesInvolvingProductionWithFeats(feat_weights, self.wrtg.P[2]))
    self.assertAlmostEqual(
      0.02, self.wrtg.SumWeightsTreesInvolvingProductionWithFeats(feat_weights, self.wrtg.P[3]))
    self.assertAlmostEqual(
      0.27, self.wrtg.SumWeightsTreesInvolvingProductionWithFeats(feat_weights, self.wrtg.P[4]))
    self.assertAlmostEqual(
      0.27, self.wrtg.SumWeightsTreesInvolvingProductionWithFeats(feat_weights, self.wrtg.P[5]))
    self.assertAlmostEqual(
      0.02, self.wrtg.SumWeightsTreesInvolvingProductionWithFeats(feat_weights, self.wrtg.P[6]))
    self.assertAlmostEqual(
      0.02, self.wrtg.SumWeightsTreesInvolvingProductionWithFeats(feat_weights, self.wrtg.P[7]))

    # Change feature weights that preserve the final score, and re-run Inside.
    feat_weights = [.5, .3, .6, .2, .9, .05, .1, .0, .66666, .77777,
                    .5, .7, .0, .2, .0, .05, .4, .5]
    feat_weights = {i : v for i, v in enumerate(feat_weights)}
    self.wrtg.ComputeInsideWeightsWithFeats(feat_weights)
    self.wrtg.ComputeOutsideWeightsWithFeats(feat_weights)

    self.assertAlmostEqual(
      0.29, self.wrtg.SumWeightsTreesInvolvingProductionWithFeats(feat_weights, self.wrtg.P[0]))
    self.assertAlmostEqual(
      0.29, self.wrtg.SumWeightsTreesInvolvingProductionWithFeats(feat_weights, self.wrtg.P[1]))
    self.assertAlmostEqual(
      0.27, self.wrtg.SumWeightsTreesInvolvingProductionWithFeats(feat_weights, self.wrtg.P[2]))
    self.assertAlmostEqual(
      0.02, self.wrtg.SumWeightsTreesInvolvingProductionWithFeats(feat_weights, self.wrtg.P[3]))
    self.assertAlmostEqual(
      0.27, self.wrtg.SumWeightsTreesInvolvingProductionWithFeats(feat_weights, self.wrtg.P[4]))
    self.assertAlmostEqual(
      0.27, self.wrtg.SumWeightsTreesInvolvingProductionWithFeats(feat_weights, self.wrtg.P[5]))
    self.assertAlmostEqual(
      0.02, self.wrtg.SumWeightsTreesInvolvingProductionWithFeats(feat_weights, self.wrtg.P[6]))
    self.assertAlmostEqual(
      0.02, self.wrtg.SumWeightsTreesInvolvingProductionWithFeats(feat_weights, self.wrtg.P[7]))

class PruneTestCase(unittest.TestCase):
  def setUp(self):
    self.S = 'q0'
    self.dummy_rule = XTRule(self.S, tree_or_string('(A B)'),\
                                  tree_or_string('(A B)'),\
                                  {}, 1.0)

  def test_ProductionSetDifferent(self):
    productions = []

    deriv_rhs1 = RHS(self.dummy_rule)
    productions.append(\
      Production((self.S, (9,9), (9,9)), deriv_rhs1, 1.0))

    deriv_rhs2 = RHS(self.dummy_rule)
    productions.append(\
      Production((self.S, (), ()), deriv_rhs2, 1.0))

    self.assertEqual(2, len(productions))
    self.assertEqual(2, len(set(productions)))

  def test_ProductionSetEqual(self):
    productions = []

    deriv_rhs1 = RHS(self.dummy_rule)
    productions.append(\
      Production((self.S, (), ()), deriv_rhs1, 1.0))

    deriv_rhs2 = RHS(self.dummy_rule)
    productions.append(\
      Production((self.S, (), ()), deriv_rhs2, 1.0))

    self.assertEqual(2, len(productions))
    self.assertEqual(1, len(set(productions)))

  def test_PruneUnreachable(self):
    productions = []

    # It is reachable from S, and produces non-terminals
    # that produce complete trees.
    deriv_rhs1 = RHS(self.dummy_rule)
    productions.append(\
      Production((self.S, (), ()), deriv_rhs1, 1.0))

    # It is not reachable (useless production)
    deriv_rhs5 = RHS(self.dummy_rule)
    productions.append(\
      Production((self.S, (9,), (9,)), deriv_rhs5, 1.0))

    non_terminals = [(self.S, (), ()),\
                     (self.S, (9,), (9,))]
    start_symbol = (self.S, (), ())
    wrtg = wRTG([self.dummy_rule], non_terminals, start_symbol, productions)
    prunned_wrtg = wrtg.Prune()

    useful_symbol1 = (self.S, (), (), '')
    self.assertIn(useful_symbol1, wrtg.N)
    self.assertIn(useful_symbol1, prunned_wrtg.N)

    useless_symbol = (self.S, (9,), (9,), '')
    self.assertIn(useless_symbol, wrtg.N)
    self.assertNotIn(useless_symbol, prunned_wrtg.N)

    useful_production1 = Production((self.S, (), ()), deriv_rhs1, 1.0)
    self.assertIn(useful_production1, wrtg.P)
    self.assertIn(useful_production1, prunned_wrtg.P)

    useless_production = Production((self.S, (9,), (9,)), deriv_rhs5, 1.0)
    self.assertIn(useless_production, wrtg.P)
    self.assertNotIn(useless_production, prunned_wrtg.P)

  def test_PruneNonDeriving(self):
    productions = []

    # It is reachable from S, and produces non-terminals
    # that produce complete trees.
    deriv_rhs1 = RHS(self.dummy_rule, [(self.S, (0,), (0,))])
    productions.append(\
      Production((self.S, (), ()), deriv_rhs1, 1.0))

    # It is reachable from S, and produces a tree. 
    deriv_rhs3 = RHS(self.dummy_rule)
    productions.append(\
      Production((self.S, (0,), (0,)), deriv_rhs3, 1.0))

    # It is reachable, but points to an infinite cycle.
    deriv_rhs7 = RHS(self.dummy_rule, [(self.S, (8,), (8,))])
    productions.append(\
      Production((self.S, (0,), (0,)), deriv_rhs7, 0.9))

    # It is reachable, but points to itself (infinite cycle).
    deriv_rhs8 = RHS(self.dummy_rule, [(self.S, (8,), (8,))])
    productions.append(\
      Production((self.S, (8,), (8,)), deriv_rhs8, 1.0))

    non_terminals = [(self.S, (), ()),\
                     (self.S, (0,), (0,)),\
                     (self.S, (8,), (8,))]
    start_symbol = (self.S, (), ())
    wrtg = wRTG([self.dummy_rule], non_terminals, start_symbol, productions)
    # from pudb import set_trace; set_trace()
    prunned_wrtg = wrtg.Prune()

    useful_symbol1 = (self.S, (0,), (0,), '')
    useful_symbol2 = (self.S, (), (), '')
    self.assertIn(useful_symbol1, wrtg.N)
    self.assertIn(useful_symbol2, wrtg.N)
    self.assertIn(useful_symbol1, prunned_wrtg.N)
    self.assertIn(useful_symbol2, prunned_wrtg.N)

    useless_symbol = (self.S, (8,), (8,), '')
    self.assertIn(useless_symbol, wrtg.N)
    self.assertNotIn(useless_symbol, prunned_wrtg.N)

    useful_production1 = Production((self.S, (), ()), deriv_rhs1, 1.0)
    useful_production2 = Production((self.S, (0,), (0,)), deriv_rhs3, 1.0)
    self.assertIn(useful_production1, wrtg.P)
    self.assertIn(useful_production2, wrtg.P)
    self.assertIn(useful_production1, prunned_wrtg.P)
    self.assertIn(useful_production2, prunned_wrtg.P)

    useless_production1 = Production((self.S, (0,), (0,)), deriv_rhs7, 0.9)
    useless_production2 = Production((self.S, (8,), (8,)), deriv_rhs8, 1.0)
    self.assertIn(useless_production1, wrtg.P)
    self.assertIn(useless_production2, wrtg.P)
    self.assertNotIn(useless_production1, prunned_wrtg.P)
    self.assertNotIn(useless_production2, prunned_wrtg.P)

  def test_PruneNonReachableThatProducesReachable(self):
    productions = []

    # It is reachable from S, and produces non-terminals
    # that produce complete trees.
    deriv_rhs1 = RHS(self.dummy_rule, [(self.S, (0,), (0,))])
    productions.append(\
      Production((self.S, (), ()), deriv_rhs1, 1.0))

    # It is reachable from S, and produces a tree. 
    deriv_rhs3 = RHS(self.dummy_rule)
    productions.append(\
      Production((self.S, (0,), (0,)), deriv_rhs3, 1.0))

    # Produces a reachable non-terminal, but it is not reachable.
    deriv_rhs6 = RHS(self.dummy_rule, [(self.S, (0,), (0,))])
    productions.append(\
      Production((self.S, (7,), (7,)), deriv_rhs6, 1.0))

    non_terminals = [(self.S, (), ()),\
                     (self.S, (0,), (0,)),\
                     (self.S, (7,), (7,))]
    start_symbol = (self.S, (), ())
    wrtg = wRTG([self.dummy_rule], non_terminals, start_symbol, productions)
    prunned_wrtg = wrtg.Prune()

    useful_symbol1 = (self.S, (0,), (0,), '')
    useful_symbol2 = (self.S, (), (), '')
    self.assertIn(useful_symbol1, wrtg.N)
    self.assertIn(useful_symbol2, wrtg.N)
    self.assertIn(useful_symbol1, prunned_wrtg.N)
    self.assertIn(useful_symbol2, prunned_wrtg.N)

    useless_symbol = (self.S, (7,), (7,), '')
    self.assertIn(useless_symbol, wrtg.N)
    self.assertNotIn(useless_symbol, prunned_wrtg.N)

    useful_production1 = Production((self.S, (), ()), deriv_rhs1, 1.0)
    useful_production2 = Production((self.S, (0,), (0,)), deriv_rhs3, 1.0)
    self.assertIn(useful_production1, wrtg.P)
    self.assertIn(useful_production2, wrtg.P)
    self.assertIn(useful_production1, prunned_wrtg.P)
    self.assertIn(useful_production2, prunned_wrtg.P)

    useless_production1 = Production((self.S, (7,), (7,)), deriv_rhs6, 1.0)
    self.assertIn(useless_production1, wrtg.P)
    self.assertNotIn(useless_production1, prunned_wrtg.P)

  def test_PruneReachableNotProducesValidTrees(self):
    rule0 = XTRule(self.S, tree_or_string('(C ?x0| ?x1|)'),
                        tree_or_string('(T ?x0| ?x1|)'),
                        {(0,) : self.S, (1,) : self.S}, 1.0)
    # This rule is useless.
    rule1 = XTRule(self.S, tree_or_string('(C ?x0| ?x1|)'),
                        tree_or_string('(T ?x1| ?x0|)'),
                        {(0,) : self.S, (1,) : self.S}, 1.0)
    rule2 = XTRule(self.S, tree_or_string('X'),
                        tree_or_string('X'),
                        {}, 1.0)
    rule3 = XTRule(self.S, tree_or_string('Y'),
                        tree_or_string('Y'),
                        {}, 1.0)
    rules = [rule0, rule1, rule2, rule3]
    transducer = xT(self.S, rules)
    intree  = immutable(tree_or_string('(C X Y)'))
    outtree = immutable(tree_or_string('(T X Y)'))
    # from pudb import set_trace; set_trace()
    wrtg = transducer.DerivTreeToTree(intree, outtree)

    self.assertEqual(3, len(wrtg.P))

    expected_production0 = Production((self.S, (), ()), \
                                      RHS(rule0, \
                                          [(self.S, (0,), (0,)), \
                                           (self.S, (1,), (1,))]), \
                                      rule0.weight)
    expected_production2 = Production((self.S, (0,), (0,)), \
                                      RHS(rule2,  []), \
                                      rule2.weight)
    expected_production3 = Production((self.S, (1,), (1,)), \
                                      RHS(rule3,  []), \
                                      rule3.weight)

    self.assertIn(expected_production0, wrtg.P)
    self.assertIn(expected_production2, wrtg.P)
    self.assertIn(expected_production3, wrtg.P)

    unexpected_production1 = Production((self.S, (), ()), \
                                        RHS(rule1, \
                                            [(self.S, (0,), (0,)), \
                                             (self.S, (1,), (1,))]), \
                                        rule1.weight)

    self.assertNotIn(unexpected_production1, wrtg.P)

if __name__ == '__main__':
  suite1  = unittest.TestLoader().loadTestsFromTestCase(InsideOutsideTestCase)
  suite2  = unittest.TestLoader().loadTestsFromTestCase(PruneTestCase)
  suite3  = unittest.TestLoader().loadTestsFromTestCase(GenerateTreesTestCase)
  suite4  = unittest.TestLoader().loadTestsFromTestCase(
    TargetProjectionFromDerivationTestCase)
  suite5  = unittest.TestLoader().loadTestsFromTestCase(
    SourceProjectionFromDerivationMixTestCase)
  suite6  = unittest.TestLoader().loadTestsFromTestCase(
    SourceProjectionFromDerivationStrictTestCase)
  suite7  = unittest.TestLoader().loadTestsFromTestCase(
    ObtainBestDerivationTestCase)
  suites  = unittest.TestSuite([suite1, suite2, suite3, suite4, suite5, suite6,
                                suite7])
  unittest.TextTestRunner(verbosity=2).run(suites)


