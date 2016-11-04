import unittest

# from pudb import set_trace; set_trace()

from linguistics.similarity_costs import LexicalSimilarity
from linguistics.similarity_pre import Identity
from transducer import xT
from transductionrule import *
from ruleindex import *
from utils.tree_tools import immutable, Tree, tree_or_string
from wrtg import RHS, Production

class TransduceTestCase(unittest.TestCase):
  def tearDown(self):
    xT.produce_cache = {}
  
  def test_UnseenTerminalSimilar(self):
    intree = tree_or_string('dog')
    rule0 = XTRule('hypernym', tree_or_string('italian'),
                               tree_or_string('european'),
                               {}, 1.0)
    rules = [rule0]
    rule_backoffs = [LexicalSimilarity()]
    initial_state = 'hypernym'
    transducer = xT(initial_state, rules, rule_backoffs)
    wrtg = transducer.Transduce(intree)
    outtrees = [tree for tree, _ in wrtg.GenerateNBestTrees()]
    expected_outtree = tree_or_string('canine')
    self.assertIn(expected_outtree, outtrees)

  def test_UnseenTerminalEqual(self):
    intree = tree_or_string('dog')
    rule0 = XTRule('copy', tree_or_string('italian'),
                           tree_or_string('italian'),
                           {}, 1.0)
    rules = [rule0]
    rule_backoffs = [LexicalSimilarity()]
    initial_state = 'copy'
    transducer = xT(initial_state, rules, rule_backoffs)
    wrtg = transducer.Transduce(intree)
    outtrees = [tree for tree, _ in wrtg.GenerateNBestTrees()]
    expected_outtree = tree_or_string('dog')
    self.assertIn(expected_outtree, outtrees)

  def test_UnseenTerminalDifferent(self):
    intree = tree_or_string('dog')
    rule0 = XTRule('t', tree_or_string('italian'),
                        tree_or_string('house'),
                        {}, 1.0)
    rules = [rule0]
    initial_state = 't'
    transducer = xT(initial_state, rules)
    wrtg = transducer.Transduce(intree)
    self.assertEqual(wrtg.P, [])

  def test_PreterminalUnseenTerminalSimilar(self):
    intree = tree_or_string('(NN dog)')
    rule0 = XTRule('q', tree_or_string('(NN ?x0|)'),
                        tree_or_string('(JJ ?x0|)'),
                        {(0,) : 'hypernym'}, 1.0)
    rule1 = XTRule('hypernym', tree_or_string('italian'),
                               tree_or_string('european'),
                               {}, 1.0)
    rules = [rule0, rule1]
    rule_backoffs = [LexicalSimilarity()]
    initial_state = 'q'
    transducer = xT(initial_state, rules, rule_backoffs)
    wrtg = transducer.Transduce(intree, None)
    outtrees = [tree for tree, _ in wrtg.GenerateNBestTrees()]
    expected_outtree = immutable(tree_or_string('(JJ canine)'))
    self.assertIn(expected_outtree, outtrees)

  def test_PreterminalUnseenTerminalEqual(self):
    intree = tree_or_string('(NN dog)')
    rule0 = XTRule('q', tree_or_string('(NN ?x0|)'),
                        tree_or_string('(JJ ?x0|)'),
                        {(0,) : 'copy'}, 1.0)
    rule1 = XTRule('copy', tree_or_string('italian'),
                           tree_or_string('italian'),
                           {}, 1.0)
    rules = [rule0, rule1]
    rule_backoffs = [LexicalSimilarity()]
    initial_state = 'q'
    transducer = xT(initial_state, rules, rule_backoffs)
    wrtg = transducer.Transduce(intree, None)
    outtrees = [tree for tree, _ in wrtg.GenerateNBestTrees()]
    expected_outtree = immutable(tree_or_string('(JJ dog)'))
    self.assertIn(expected_outtree, outtrees)

  def test_PreterminalUnseenTerminalDifferent(self):
    intree = tree_or_string('(NN dog)')
    rule0 = XTRule('q', tree_or_string('(NN ?x0|)'),
                        tree_or_string('(JJ ?x0|)'),
                        {(0,) : 't'}, 1.0)
    rule1 = XTRule('t', tree_or_string('italian'),
                        tree_or_string('house'),
                        {}, 1.0)
    rules = [rule0, rule1]
    initial_state = 'q'
    transducer = xT(initial_state, rules)
    wrtg = transducer.Transduce(intree, None)
    self.assertEqual(wrtg.P, [])

  def test_NonterminalUnseenTerminalEqualAndSimilar(self):
    intree = tree_or_string('(NP (DT the) (NN dog))')
    rule0 = XTRule('q', tree_or_string('(NP (DT ?x0|) (NN ?x1|))'),
                        tree_or_string('(NP (DT ?x0|) (NN ?x1|))'),
                        {(0, 0) : 'copy', (1, 0) : 'hypernym'}, 1.0)
    rule1 = XTRule('copy', tree_or_string('the'),
                           tree_or_string('the'),
                           {}, 1.0)
    rule2 = XTRule('hypernym', tree_or_string('italian'),
                               tree_or_string('european'),
                               {}, 1.0)
    rules = [rule0, rule1, rule2]
    rule_backoffs = [LexicalSimilarity()]
    initial_state = 'q'
    transducer = xT(initial_state, rules, rule_backoffs)
    wrtg = transducer.Transduce(intree, None)
    outtrees = [tree for tree, _ in wrtg.GenerateNBestTrees()]
    expected_outtree = immutable(tree_or_string('(NP (DT the) (NN canine))'))
    self.assertIn(expected_outtree, outtrees)

  def test_NonConsumingLHSAvoidsInfiniteRTG(self):
    intree = tree_or_string('(NN dog)')
    rule0 = XTRule('q', tree_or_string('?x0|NN'),
                        tree_or_string('(NN ?x0|)'),
                        {(0,) : 'q'}, 0.9)
    rule1 = XTRule('q', tree_or_string('?x0|NN'),
                        tree_or_string('(JJ ?x0|)'),
                        {(0,) : 't'}, 0.9)
    rule2 = XTRule('t', tree_or_string('(NN dog)'),
                        tree_or_string('canine'),
                        {}, 1.0)
    rules = [rule0, rule1, rule2]
    initial_state = 'q'
    transducer = xT(initial_state, rules)
    wrtg = transducer.Transduce(intree, None)
    outtrees = [tree for tree, _ in wrtg.GenerateNBestTrees()]
    expected_outtree = immutable(tree_or_string('(JJ canine)'))
    self.assertIn(expected_outtree, outtrees)

  def test_PreterminalIdentity(self):
    intree = tree_or_string('(NN dog)')
    rule1 = XTRule('q', tree_or_string('dog'),
                        tree_or_string('perro'),
                        {}, 1.0)
    rules = [rule1]
    rule_backoffs = [Identity(), LexicalSimilarity()]
    initial_state = 'q'
    transducer = xT(initial_state, rules, rule_backoffs)
    wrtg = transducer.Transduce(intree, None)
    outtrees = [tree for tree, _ in wrtg.GenerateNBestTrees()]
    expected_outtree = immutable(tree_or_string('(NN perro)'))
    self.assertIn(expected_outtree, outtrees)

  @unittest.expectedFailure
  def test_PreterminalIdentityUnseenTerminalSimilar(self):
    """
    Using the Identity back-off, the state of the parent rule
    is applied to the path of the variable in the RHS.
    However, the states of the path of the variable in the RHS
    should be more specific: "hypernym".
    """
    intree = tree_or_string('(NN dog)')
    rule1 = XTRule('hypernym', tree_or_string('italian'),
                               tree_or_string('european'),
                               {}, 1.0)
    rules = [rule1]
    rule_backoffs = [Identity(), LexicalSimilarity()]
    initial_state = 'q'
    transducer = xT(initial_state, rules, rule_backoffs)
    wrtg = transducer.Transduce(intree, None)
    outtrees = [tree for tree, _ in wrtg.GenerateNBestTrees()]
    expected_outtree = immutable(tree_or_string('(NN canine)'))
    self.assertIn(expected_outtree, outtrees)

  def test_NonterminalUnseenTerminalSimilar(self):
    intree = tree_or_string('(NP (DT the) (NN dog))')
    rule0 = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        tree_or_string('(NPP ?x0|DTT ?x1|NNN)'),
                        {(0,) : 'q', (1,) : 'q'}, 1.0)
    rule1 = XTRule('q', tree_or_string('(DT ?x0|)'),
                        tree_or_string('(DTT ?x0|)'),
                        {(0,) : 'copy'}, 1.0)
    rule2 = XTRule('copy', tree_or_string('the'),
                           tree_or_string('the'),
                           {}, 1.0)
    rule3 = XTRule('q', tree_or_string('(NN ?x0|)'),
                        tree_or_string('(NNN ?x0|)'),
                        {(0,) : 'hypernym'}, 1.0)
    rules = [rule0, rule1, rule2, rule3]
    rule_backoffs = [LexicalSimilarity()]
    initial_state = 'q'
    transducer = xT(initial_state, rules, rule_backoffs)
    wrtg = transducer.Transduce(intree, None)
    outtrees = [tree for tree, _ in wrtg.GenerateNBestTrees()]
    expected_outtree = immutable(tree_or_string('(NPP (DTT the) (NNN canine))'))
    self.assertIn(expected_outtree, outtrees)

  def test_NonterminalUnseenTerminalSimilarNoBackoff(self):
    intree = tree_or_string('(NP (DT the) (NN dog))')
    rule0 = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        tree_or_string('(NPP ?x0|DTT ?x1|NNN)'),
                        {(0,) : 'q', (1,) : 'q'}, 1.0)
    rule1 = XTRule('q', tree_or_string('(DT ?x0|)'),
                        tree_or_string('(DTT ?x0|)'),
                        {(0,) : 'copy'}, 1.0)
    rule2 = XTRule('copy', tree_or_string('the'),
                           tree_or_string('the'),
                           {}, 1.0)
    rule3 = XTRule('q', tree_or_string('(NN ?x0|)'),
                        tree_or_string('(NNN ?x0|)'),
                        {(0,) : 'hypernym'}, 1.0)
    rules = [rule0, rule1, rule2, rule3]
    rule_backoffs = []
    initial_state = 'q'
    transducer = xT(initial_state, rules, rule_backoffs)
    wrtg = transducer.Transduce(intree, None)
    outtrees = [tree for tree, _ in wrtg.GenerateNBestTrees()]
    expected_outtree = immutable(tree_or_string('(NPP (DTT the) (NNN canine))'))
    self.assertNotIn(expected_outtree, outtrees)

  @unittest.expectedFailure
  def test_NonterminalPreterminalIdentity(self):
    """
    Using the Identity back-off, the state of the parent rule
    is applied to the paths of the variables in the RHS.
    However, the states of the paths of the variables in the RHS
    should be more specific: "copy" and "hypernym".
    """
    intree = tree_or_string('(NP (DT the) (NN dog))')
    rule0 = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        tree_or_string('(NPP ?x0|DTT ?x1|NNN)'),
                        {(0,) : 'q', (1,) : 'q'}, 1.0)
    rule1 = XTRule('q', tree_or_string('(DT ?x0|)'),
                        tree_or_string('(DTT ?x0|)'),
                        {(0,) : 'copy'}, 1.0)
    rule2 = XTRule('copy', tree_or_string('the'),
                           tree_or_string('the'),
                           {}, 1.0)
    rule3 = XTRule('hypernym', tree_or_string('dog'),
                               tree_or_string('canine'),
                               {}, 1.0)
    rules = [rule0, rule1, rule2, rule3]
    rule_backoffs = [Identity()]
    initial_state = 'q'
    transducer = xT(initial_state, rules, rule_backoffs)
    wrtg = transducer.Transduce(intree, None)
    outtrees = [tree for tree, _ in wrtg.GenerateNBestTrees()]
    expected_outtree = immutable(tree_or_string('(NPP (DTT the) (NN canine))'))
    self.assertIn(expected_outtree, outtrees)

  def test_NonterminalPreterminalIdentityNoBackoff(self):
    intree = tree_or_string('(NP (DT the) (NN dog))')
    rule0 = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        tree_or_string('(NPP ?x0|DTT ?x1|NNN)'),
                        {(0,) : 'q', (1,) : 'q'}, 1.0)
    rule1 = XTRule('q', tree_or_string('(DT ?x0|)'),
                        tree_or_string('(DTT ?x0|)'),
                        {(0,) : 'copy'}, 1.0)
    rule2 = XTRule('copy', tree_or_string('the'),
                           tree_or_string('the'),
                           {}, 1.0)
    rule3 = XTRule('hypernym', tree_or_string('dog'),
                               tree_or_string('canine'),
                               {}, 1.0)
    rules = [rule0, rule1, rule2, rule3]
    rule_backoffs = []
    initial_state = 'q'
    transducer = xT(initial_state, rules, rule_backoffs)
    wrtg = transducer.Transduce(intree, None)
    outtrees = [tree for tree, _ in wrtg.GenerateNBestTrees()]
    expected_outtree = immutable(tree_or_string('(NPP (DTT the) (NN canine))'))
    self.assertNotIn(expected_outtree, outtrees)

  def test_NonterminalIdentity(self):
    intree = tree_or_string('(NP (DT the) (NN dog))')
    rule0 = XTRule('q', tree_or_string('(DT ?x0|)'),
                        tree_or_string('(DTT ?x0|)'),
                        {(0,) : 'copy'}, 1.0)
    rule1 = XTRule('copy', tree_or_string('the'),
                           tree_or_string('the'),
                           {}, 1.0)
    rule2 = XTRule('q', tree_or_string('(NN ?x0|)'),
                        tree_or_string('(NNN ?x0|)'),
                        {(0,) : 'hypernym'}, 1.0)
    rule3 = XTRule('hypernym', tree_or_string('dog'),
                               tree_or_string('canine'),
                               {}, 1.0)
    rules = [rule0, rule1, rule2, rule3]
    rule_backoffs = [Identity()]
    initial_state = 'q'
    transducer = xT(initial_state, rules, rule_backoffs)
    wrtg = transducer.Transduce(intree, None)
    outtrees = [tree for tree, _ in wrtg.GenerateNBestTrees()]
    expected_outtree = immutable(tree_or_string('(NP (DTT the) (NNN canine))'))
    self.assertIn(expected_outtree, outtrees)

  def test_NonterminalIdentityNoBackoff(self):
    intree = tree_or_string('(NP (DT the) (NN dog))')
    rule0 = XTRule('q', tree_or_string('(DT ?x0|)'),
                        tree_or_string('(DTT ?x0|)'),
                        {(0,) : 'copy'}, 1.0)
    rule1 = XTRule('copy', tree_or_string('the'),
                           tree_or_string('the'),
                           {}, 1.0)
    rule2 = XTRule('q', tree_or_string('(NN ?x0|)'),
                        tree_or_string('(NNN ?x0|)'),
                        {(0,) : 'hypernym'}, 1.0)
    rule3 = XTRule('hypernym', tree_or_string('dog'),
                               tree_or_string('canine'),
                               {}, 1.0)
    rules = [rule0, rule1, rule2, rule3]
    rule_backoffs = []
    initial_state = 'q'
    transducer = xT(initial_state, rules, rule_backoffs)
    wrtg = transducer.Transduce(intree, None)
    outtrees = [tree for tree, _ in wrtg.GenerateNBestTrees()]
    expected_outtree = immutable(tree_or_string('(NP (DTT the) (NNN canine))'))
    self.assertNotIn(expected_outtree, outtrees)

  def test_NonterminalDeleteLeftBranch(self):
    intree = tree_or_string('(NP (DT the) (NN dog))')
    rule0 = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        tree_or_string('(NPP ?x1|NNN)'),
                        {(0,) : 'q'}, 1.0)
    rule1 = XTRule('q', tree_or_string('(DT ?x0|)'),
                        tree_or_string('(DTT ?x0|)'),
                        {(0,) : 'copy'}, 1.0)
    rule1 = XTRule('q', tree_or_string('(NN ?x0|)'),
                        tree_or_string('(NNN ?x0|)'),
                        {(0,) : 'hypernym'}, 1.0)
    rule2 = XTRule('copy', tree_or_string('the'),
                           tree_or_string('the'),
                           {}, 1.0)
    rule3 = XTRule('hypernym', tree_or_string('dog'),
                               tree_or_string('canine'),
                               {}, 1.0)
    rules = [rule0, rule1, rule2, rule3]
    rule_backoffs = []
    initial_state = 'q'
    transducer = xT(initial_state, rules, rule_backoffs)
    wrtg = transducer.Transduce(intree, None)
    outtrees = [tree for tree, _ in wrtg.GenerateNBestTrees()]
    expected_outtree = immutable(tree_or_string('(NPP (NNN canine))'))
    self.assertItemsEqual([expected_outtree], outtrees)

class GetNonterminalsTestCase(unittest.TestCase):
  def setUp(self):
    rule1 = XTRule('q', tree_or_string('(A ?x0| ?x1|)'),
                        tree_or_string('(A (R ?x1| ?x0|) (S X))'),
                        {(0, 0) : 'q', (0, 1) : 'q'}, 1.0)
    rule2 = XTRule('q', tree_or_string('(B ?x0| ?x1|)'),
                        tree_or_string('U'),
                        {}, 1.0)
    rule3 = XTRule('q', tree_or_string('(C ?x0| ?x1|)'),
                        tree_or_string('(T ?x0| ?x1|)'),
                        {(0,) : 'q', (1,) : 'q'}, 0.6)
    rule4 = XTRule('q', tree_or_string('(C ?x0| ?x1|)'),
                        tree_or_string('(T ?x1| ?x0|)'),
                        {(0,) : 'q', (1,) : 'q'}, 0.4)
    rule5 = XTRule('q', tree_or_string('F'),
                        tree_or_string('V'),
                        {}, 0.9)
    rule6 = XTRule('q', tree_or_string('F'),
                        tree_or_string('W'),
                        {}, 0.1)
    rule7 = XTRule('q', tree_or_string('G'),
                        tree_or_string('V'),
                        {}, 0.5)
    rule8 = XTRule('q', tree_or_string('G'),
                        tree_or_string('W'),
                        {}, 0.5)
    self.rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8]
    self.transducer = xT('q', self.rules)

  def test_Nonterminal(self):
    input_tree = immutable(tree_or_string('(A (B D E) (C F G))'))
    output_tree = immutable(tree_or_string('(A (R (T V W) U) (S X))'))
    productions, non_terminals = \
      self.transducer.Produce(input_tree, output_tree, 'q', (), ())
    expected_non_terminals = [('q', (), (), ''),
                              ('q', (0,), (0, 1), ''),
                              ('q', (1,), (0, 0), ''),
                              ('q', (1, 0), (0, 0, 0), ''),
                              ('q', (1, 1), (0, 0, 1), ''),
                              ('q', (1, 0), (0, 0, 1), ''),
                              ('q', (1, 1), (0, 0, 0), '')]
    self.assertIn(expected_non_terminals[0], non_terminals)
    self.assertIn(expected_non_terminals[1], non_terminals)
    self.assertIn(expected_non_terminals[2], non_terminals)
    self.assertIn(expected_non_terminals[3], non_terminals)
    self.assertIn(expected_non_terminals[4], non_terminals)
    self.assertIn(expected_non_terminals[5], non_terminals)
    self.assertIn(expected_non_terminals[6], non_terminals)

class ProduceTestCase(unittest.TestCase):
  def setUp(self):
    rule1 = XTRule('q', tree_or_string('(A ?x0| ?x1|)'),
                        tree_or_string('(A (R ?x1| ?x0|) (S X))'),
                        {(0, 0) : 'q', (0, 1) : 'q'}, 1.0)
    rule2 = XTRule('q', tree_or_string('(B ?x0| ?x1|)'),
                        tree_or_string('U'),
                        {}, 1.0)
    rule3 = XTRule('q', tree_or_string('(C ?x0| ?x1|)'),
                        tree_or_string('(T ?x0| ?x1|)'),
                        {(0,) : 'q', (1,) : 'q'}, 0.6)
    rule4 = XTRule('q', tree_or_string('(C ?x0| ?x1|)'),
                        tree_or_string('(T ?x1| ?x0|)'),
                        {(0,) : 'q', (1,) : 'q'}, 0.4)
    rule5 = XTRule('q', tree_or_string('F'),
                        tree_or_string('V'),
                        {}, 0.9)
    rule6 = XTRule('q', tree_or_string('F'),
                        tree_or_string('W'),
                        {}, 0.1)
    rule7 = XTRule('q', tree_or_string('G'),
                        tree_or_string('V'),
                        {}, 0.5)
    rule8 = XTRule('q', tree_or_string('G'),
                        tree_or_string('W'),
                        {}, 0.5)
    self.rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8]
    self.transducer = xT('q', self.rules)

  def tearDown(self):
    del self.rules
    # self.transducer.Produce.cache_clear()
    self.transducer.produce_cache = {}
 
  def test_Terminal(self):
    input_tree = tree_or_string('G')
    output_tree = tree_or_string('W')
    productions, _ = self.transducer.Produce(input_tree, output_tree, 'q', (), ())
    rule8 = XTRule('q', tree_or_string('G'),
                        tree_or_string('W'),
                        {}, 0.5)
    deriv_rhs = RHS(rule8)
    # from pudb import set_trace; set_trace()
    expected_production = Production(('q', (), ()), deriv_rhs, rule8.weight)
    self.assertIn(expected_production, productions)

  def test_TerminalEmptyLHSfail(self):
    input_tree = tree_or_string('Z')
    output_tree = tree_or_string('W')
    productions, _ = self.transducer.Produce(input_tree, output_tree, 'q', (), ())
    self.assertEqual(0, len(productions))

  def test_TerminalEmptyRHSfail(self):
    input_tree = tree_or_string('G')
    output_tree = tree_or_string('Z')
    productions, _ = self.transducer.Produce(input_tree, output_tree, 'q', (), ())
    self.assertEqual(0, len(productions))

  def test_Preterminal(self):
    input_tree = immutable(tree_or_string('(B D E)'))
    output_tree = immutable(tree_or_string('U'))
    productions, _ = self.transducer.Produce(input_tree, output_tree, 'q', (), ())
    rule2 = XTRule('q', tree_or_string('(B ?x0| ?x1|)'),
                        tree_or_string('U'),
                        {}, 1.0)
    deriv_rhs = RHS(rule2)
    expected_production = Production(('q', (), ()), deriv_rhs, rule2.weight)
    self.assertIn(expected_production, productions)
    
  def test_PreterminalEmptyLHSfail(self):
    input_tree = immutable(tree_or_string('(Z D E)'))
    output_tree = immutable(tree_or_string('U'))
    productions, _ = self.transducer.Produce(input_tree, output_tree, 'q', (), ())
    self.assertEqual(0, len(productions))
    
  def test_PreterminalEmptyRHSfail(self):
    input_tree = immutable(tree_or_string('(B D E)'))
    output_tree = immutable(tree_or_string('Z'))
    productions, _ = self.transducer.Produce(input_tree, output_tree, 'q', (), ())
    self.assertEqual(0, len(productions))

  def test_Nonterminal(self):
    input_tree = immutable(tree_or_string('(A (B D E) (C F G))'))
    output_tree = immutable(tree_or_string('(A (R (T V W) U) (S X))'))
    productions, _ = self.transducer.Produce(input_tree, output_tree, 'q', (), ())
    expected_productions = []

    deriv_rhs1 = RHS(self.rules[0], [('q', (0, ), (0, 1)), \
                                     ('q', (1, ), (0, 0))])
    expected_productions.append(\
      Production(('q', (), ()), deriv_rhs1, self.rules[0].weight))

    deriv_rhs2 = RHS(self.rules[1])
    expected_productions.append(\
      Production(('q', (0,), (0, 1)), deriv_rhs2, self.rules[1].weight))

    deriv_rhs3 = RHS(self.rules[2], [('q', (1, 0), (0, 0, 0)), \
                                     ('q', (1, 1), (0, 0, 1))])
    expected_productions.append(\
      Production(('q', (1,), (0, 0)), deriv_rhs3, self.rules[2].weight))

    deriv_rhs4 = RHS(self.rules[3], [('q', (1, 0), (0, 0, 1)), \
                                     ('q', (1, 1), (0, 0, 0))])
    expected_productions.append(\
      Production(('q', (1,), (0, 0)), deriv_rhs4, self.rules[3].weight))

    deriv_rhs5 = RHS(self.rules[4])
    expected_productions.append(\
      Production(('q', (1, 0), (0, 0, 0)), deriv_rhs5, self.rules[4].weight))

    deriv_rhs8 = RHS(self.rules[7])
    expected_productions.append(\
      Production(('q', (1, 1), (0, 0, 1)), deriv_rhs8, self.rules[7].weight))

    deriv_rhs6 = RHS(self.rules[5])
    expected_productions.append(\
      Production(('q', (1, 0), (0, 0, 1)), deriv_rhs6, self.rules[5].weight))

    deriv_rhs7 = RHS(self.rules[6])
    expected_productions.append(\
      Production(('q', (1, 1), (0, 0, 0)), deriv_rhs7, self.rules[6].weight))

    self.assertEqual(len(expected_productions), len(productions))
    self.assertIn(expected_productions[0], productions)
    self.assertIn(expected_productions[1], productions)
    self.assertIn(expected_productions[2], productions)
    self.assertIn(expected_productions[3], productions)
    self.assertIn(expected_productions[4], productions)
    self.assertIn(expected_productions[5], productions)
    self.assertIn(expected_productions[6], productions)
    self.assertIn(expected_productions[7], productions)
 
  def test_OnlySourceDifferentVarTypes(self):
    rule0 = XTRule('q', tree_or_string('(A ?x0|AA)'),
                        tree_or_string('(a ?x0|aa)'),
                        {(0,) : 't'}, 1.0)
    rule1 = XTRule('t', tree_or_string('(AA AAA)'),
                        tree_or_string('(aa aaa)'),
                        {}, 1.0)
    rule2 = XTRule('t', tree_or_string('(AA AAA)'),
                        tree_or_string('(bb bbb)'),
                        {}, 1.0)
    rules = [rule0, rule1, rule2]
    self.transducer = xT('q', rules)
    input_tree = immutable(tree_or_string('(A (AA AAA))'))
    output_tree = None
    productions, _ = self.transducer.Produce(input_tree, output_tree, 'q', (), ())

    self.assertEqual(2, len(productions))
    self.assertIn(rule0, [p.rhs.rule for p in productions])
    self.assertIn(rule1, [p.rhs.rule for p in productions])
    self.assertNotIn(rule2, [p.rhs.rule for p in productions])

if __name__ == '__main__':
  suite1  = unittest.TestLoader().loadTestsFromTestCase(GetNonterminalsTestCase)
  suite2  = unittest.TestLoader().loadTestsFromTestCase(ProduceTestCase)
  suite3  = unittest.TestLoader().loadTestsFromTestCase(TransduceTestCase)
  suites  = unittest.TestSuite([suite1, suite2, suite3])
  unittest.TextTestRunner(verbosity=2).run(suites)

