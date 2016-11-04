import unittest

from training.transductionrule import ParseTiburonRule, XTRule
from training.wrtg import RHS, Production
from utils.tree_tools import Tree, tree_or_string

class ParseTiburonTestCase(unittest.TestCase):
  def test_TerminalToTerminal(self):
    tib_rule = 'q."hello" -> "hola"'
    rule = ParseTiburonRule(tib_rule)
    self.assertEqual(rule['weight'], 1.0)
    self.assertEqual(rule['state'], 'q')
    self.assertEqual(rule['lhs'], 'hello')
    self.assertEqual(rule['rhs'], 'hola')
    self.assertEqual(rule['newstates'], {})

  def test_TerminalToTerminalAtSymbolSource(self):
    tib_rule = 'q."AThello" -> "hola"'
    rule = ParseTiburonRule(tib_rule)
    self.assertEqual(rule['weight'], 1.0)
    self.assertEqual(rule['state'], 'q')
    self.assertEqual(rule['lhs'], '@hello')
    self.assertEqual(rule['rhs'], 'hola')
    self.assertEqual(rule['newstates'], {})

  def test_TerminalToTerminalAtSymbolTarget(self):
    tib_rule = 'q."hello" -> "AThola"'
    rule = ParseTiburonRule(tib_rule)
    self.assertEqual(rule['weight'], 1.0)
    self.assertEqual(rule['state'], 'q')
    self.assertEqual(rule['lhs'], 'hello')
    self.assertEqual(rule['rhs'], '@hola')
    self.assertEqual(rule['newstates'], {})

  def test_TerminalToTerminalSpecialSymbol(self):
    tib_rule = 'q."PERCENT" -> "hola"'
    rule = ParseTiburonRule(tib_rule)
    self.assertEqual(rule['weight'], 1.0)
    self.assertEqual(rule['state'], 'q')
    self.assertEqual(rule['lhs'], '%')
    self.assertEqual(rule['rhs'], 'hola')
    self.assertEqual(rule['newstates'], {})

  def test_TerminalToTerminalWeight(self):
    tib_rule = 'q."hello" -> "hola" # 0.8'
    rule = ParseTiburonRule(tib_rule)
    self.assertEqual(rule['weight'], 0.8)
    self.assertEqual(rule['state'], 'q')
    self.assertEqual(rule['lhs'], 'hello')
    self.assertEqual(rule['rhs'], 'hola')
    self.assertEqual(rule['newstates'], {})

  def test_TerminalToTerminalTied(self):
    tib_rule = 'q."hello" -> "hola" @ 2'
    rule = ParseTiburonRule(tib_rule)
    self.assertEqual(rule['weight'], 1.0)
    self.assertEqual(rule['state'], 'q')
    self.assertEqual(rule['lhs'], 'hello')
    self.assertEqual(rule['rhs'], 'hola')
    self.assertEqual(rule['newstates'], {})
    self.assertEqual(rule['tied_to'], 2)

  def test_TerminalToTerminalWeightedAndTied(self):
    tib_rule = 'q."hello" -> "hola" # 0.8 @ 2'
    rule = ParseTiburonRule(tib_rule)
    self.assertEqual(rule['weight'], 0.8)
    self.assertEqual(rule['state'], 'q')
    self.assertEqual(rule['lhs'], 'hello')
    self.assertEqual(rule['rhs'], 'hola')
    self.assertEqual(rule['newstates'], {})
    self.assertEqual(rule['tied_to'], 2)

  def test_InitialState(self):
    tib_rule = 'q0'
    rule = ParseTiburonRule(tib_rule)
    self.assertEqual(rule, None)

  def test_TiburonComment(self):
    tib_rule = '% q."hello" -> "hola" # 0.8'
    rule = ParseTiburonRule(tib_rule)
    self.assertEqual(rule, None)

  def test_TiburonComment2(self):
    tib_rule = '%% q."hello" -> "hola" # 0.8'
    rule = ParseTiburonRule(tib_rule)
    self.assertEqual(rule, None)

  def test_VarToVarWeight(self):
    tib_rule = 'q.x0:DT -> t.x0 # 0.7'
    rule = ParseTiburonRule(tib_rule)
    self.assertEqual(rule['weight'], 0.7)
    self.assertEqual(rule['state'], 'q')
    self.assertEqual(rule['lhs'], '?x0|DT')
    self.assertEqual(rule['rhs'], '?x0|')
    self.assertEqual(rule['newstates'], {() : 't'})

  def test_TerminalToTerminalDot(self):
    tib_rule = 'q."Anderson" -> "Mr.Anderson" # 0.7'
    rule = ParseTiburonRule(tib_rule)
    self.assertEqual(rule['weight'], 0.7)
    self.assertEqual(rule['state'], 'q')
    self.assertEqual(rule['lhs'], 'Anderson')
    self.assertEqual(rule['rhs'], 'Mr.Anderson')
    self.assertEqual(rule['newstates'], {})

  def test_PreterminalToVar(self):
    tib_rule = 'q.NN(x0:) -> t.x0 # 0.7'
    rule = ParseTiburonRule(tib_rule)
    self.assertEqual(rule['weight'], 0.7)
    self.assertEqual(rule['state'], 'q')
    self.assertEqual(rule['lhs'], '(NN ?x0|)')
    self.assertEqual(rule['rhs'], '?x0|')
    self.assertEqual(rule['newstates'], {() : 't'})

  def test_TypedPreterminalToVar(self):
    tib_rule = 'q.NP(x0:NN) -> t.x0 # 0.7'
    rule = ParseTiburonRule(tib_rule)
    self.assertEqual(rule['weight'], 0.7)
    self.assertEqual(rule['state'], 'q')
    self.assertEqual(rule['lhs'], '(NP ?x0|NN)')
    self.assertEqual(rule['rhs'], '?x0|')
    self.assertEqual(rule['newstates'], {() : 't'})

  def test_UntypedPreterminalToPreterminalVar(self):
    tib_rule = 'q.NP(x0:) -> NP(t.x0) # 0.7'
    rule = ParseTiburonRule(tib_rule)
    self.assertEqual(rule['weight'], 0.7)
    self.assertEqual(rule['state'], 'q')
    self.assertEqual(rule['lhs'], '(NP ?x0|)')
    self.assertEqual(rule['rhs'], '(NP ?x0|)')
    self.assertEqual(rule['newstates'], {(0,) : 't'})

  def test_UntypedAndTypedNonterminal(self):
    tib_rule = 'q.NP(x0:DT x1:) -> NP(t.x0 q1.x1) # 0.7'
    rule = ParseTiburonRule(tib_rule)
    self.assertEqual(rule['weight'], 0.7)
    self.assertEqual(rule['state'], 'q')
    self.assertEqual(rule['lhs'], '(NP ?x0|DT ?x1|)')
    self.assertEqual(rule['rhs'], '(NP ?x0| ?x1|)')
    self.assertEqual(rule['newstates'], {(0,) : 't', (1,) : 'q1'})

  def test_DeepNonterminalNoVars(self):
    tib_rule = 'q.NP(DT("the") NN("house")) -> NP(DT("la") NN("casa")) # 0.7'
    rule = ParseTiburonRule(tib_rule)
    self.assertEqual(rule['weight'], 0.7)
    self.assertEqual(rule['state'], 'q')
    self.assertEqual(rule['lhs'], '(NP (DT the) (NN house))')
    self.assertEqual(rule['rhs'], '(NP (DT la) (NN casa))')
    self.assertEqual(rule['newstates'], {})

  def test_DeepNonterminalVars(self):
    tib_rule = 'q.NP(x0:DT NN(x1:)) -> NP(t1.x0 NN(t2.x1)) # 0.7'
    rule = ParseTiburonRule(tib_rule)
    self.assertEqual(rule['weight'], 0.7)
    self.assertEqual(rule['state'], 'q')
    self.assertEqual(rule['lhs'], '(NP ?x0|DT (NN ?x1|))')
    self.assertEqual(rule['rhs'], '(NP ?x0| (NN ?x1|))')
    self.assertEqual(rule['newstates'], {(0,) : 't1', (1, 0) : 't2'})

  def test_SpecialCharInVarPOS(self):
    tib_rule = 'q.NP(x0:COLONarg0) -> NP(q1.x0) # 0.7'
    rule = ParseTiburonRule(tib_rule)
    self.assertEqual(rule['weight'], 0.7)
    self.assertEqual(rule['state'], 'q')
    self.assertEqual(rule['lhs'], '(NP ?x0|:arg0)')
    self.assertEqual(rule['rhs'], '(NP ?x0|)')
    self.assertEqual(rule['newstates'], {(0,) : 'q1'})

  def test_SpecialCharInNT(self):
    tib_rule = 'q.COLONNP(x0:DT) -> COLONNP(q1.x0) # 0.7'
    rule = ParseTiburonRule(tib_rule)
    self.assertEqual(rule['weight'], 0.7)
    self.assertEqual(rule['state'], 'q')
    self.assertEqual(rule['lhs'], '(:NP ?x0|DT)')
    self.assertEqual(rule['rhs'], '(:NP ?x0|)')
    self.assertEqual(rule['newstates'], {(0,) : 'q1'})

  def test_AtSymbolInLeaf(self):
    tib_rule = 'q.DT("ATYoshiko") -> DT("Yoshiko") # 0.7'
    rule = ParseTiburonRule(tib_rule)
    self.assertEqual(rule['weight'], 0.7)
    self.assertEqual(rule['state'], 'q')
    self.assertEqual(rule['lhs'], '(DT @Yoshiko)')
    self.assertEqual(rule['rhs'], '(DT Yoshiko)')
    self.assertEqual(rule['newstates'], {})

# TODO: check whether this works with Japanese characters too.
# TODO: check whether we are printing special characters in NT
#       when converting into tiburon.

class PrintTiburonTestCase(unittest.TestCase):
  def test_WeightNonePrintsNoWeight(self):
    rule1 = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        tree_or_string('(NPP ?x1|NN ?x0|DT)'),
                        {(0,) : 'q1', (1,) : 'q2'}, None)
    tiburon_string = rule1.PrintTiburon()
    expected_tiburon_string = \
      'q.NP(x0:DT x1:NN) -> NPP(q1.x1 q2.x0)'
    self.assertEqual(expected_tiburon_string, tiburon_string)

  def test_WeightNoneTiedPrintsNoWeight(self):
    rule1 = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        tree_or_string('(NPP ?x1|NN ?x0|DT)'),
                        {(0,) : 'q1', (1,) : 'q2'}, None)
    rule1.tied_to = 2
    tiburon_string = rule1.PrintTiburon()
    expected_tiburon_string = \
      'q.NP(x0:DT x1:NN) -> NPP(q1.x1 q2.x0) @ 2'
    self.assertEqual(expected_tiburon_string, tiburon_string)

  def test_WeightAndTiedPrintsWeightAndTied(self):
    rule1 = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        tree_or_string('(NPP ?x1|NN ?x0|DT)'),
                        {(0,) : 'q1', (1,) : 'q2'}, 0.8)
    rule1.tied_to = 2
    tiburon_string = rule1.PrintTiburon()
    expected_tiburon_string = \
      'q.NP(x0:DT x1:NN) -> NPP(q1.x1 q2.x0) # 0.8 @ 2'
    self.assertEqual(expected_tiburon_string, tiburon_string)

  def test_Newstates(self):
    rule1 = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        tree_or_string('(NPP ?x1|NN ?x0|DT)'),
                        {(0,) : 'q1', (1,) : 'q2'}, 0.8)
    tiburon_string = rule1.PrintTiburon()
    expected_tiburon_string = \
      'q.NP(x0:DT x1:NN) -> NPP(q1.x1 q2.x0) # 0.8'
    self.assertEqual(expected_tiburon_string, tiburon_string)

  def test_StringOnLHSAndRHS(self):
    rule1 = XTRule('q', tree_or_string('(NP (DT the) ?x0|NN)'),
                        tree_or_string('(NPP ?x0|NN)'),
                        {(0,) : 'q2'}, 0.8)
    tiburon_string = rule1.PrintTiburon()
    expected_tiburon_string = \
      'q.NP(DT("the") x0:NN) -> NPP(q2.x0) # 0.8'
    self.assertEqual(expected_tiburon_string, tiburon_string)

  def test_NoNewstates(self):
    rule1 = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        tree_or_string('(NPP X)'),
                        {}, 0.8)
    tiburon_string = rule1.PrintTiburon()
    expected_tiburon_string = \
      'q.NP(x0:DT x1:NN) -> NPP("X") # 0.8'
    self.assertEqual(expected_tiburon_string, tiburon_string)

  def test_TerminalRule(self):
    rule1 = XTRule('t', tree_or_string('hello'),
                        tree_or_string('hola'),
                        {}, 0.7)
    tiburon_string = rule1.PrintTiburon()
    expected_tiburon_string = 't."hello" -> "hola" # 0.7'
    self.assertEqual(expected_tiburon_string, tiburon_string)

  def test_TerminalRuleAtSymbol(self):
    rule1 = XTRule('t', tree_or_string('@hello'),
                        tree_or_string('hola'),
                        {}, 0.7)
    tiburon_string = rule1.PrintTiburon()
    expected_tiburon_string = 't."AThello" -> "hola" # 0.7'
    self.assertEqual(expected_tiburon_string, tiburon_string)

  def test_DeepState(self):
    rule1 = XTRule('q', tree_or_string('(NP (DT ?x0|) ?x1|NN)'),
                        tree_or_string('(NPP (DTT ?x0|) ?x1|NN)'),
                        {(0, 0) : 'q1', (1,) : 'q2'}, 0.8)
    tiburon_string = rule1.PrintTiburon()
    expected_tiburon_string = \
      'q.NP(DT(x0:) x1:NN) -> NPP(DTT(q1.x0) q2.x1) # 0.8'
    self.assertEqual(expected_tiburon_string, tiburon_string)

  def test_EpsilonSource(self):
    rule1 = XTRule('q', tree_or_string('?x0|NP'),
                        tree_or_string('(NPP ?x0|NP)'),
                        {(0,) : 'q1'}, 0.8)
    tiburon_string = rule1.PrintTiburon()
    expected_tiburon_string = \
      'q.x0:NP -> NPP(q1.x0) # 0.8'
    self.assertEqual(expected_tiburon_string, tiburon_string)

  def test_EpsilonTarget(self):
    rule1 = XTRule('q', tree_or_string('(NP ?x0|NN)'),
                        tree_or_string('?x0|NP'),
                        {() : 'q1'}, 0.8)
    tiburon_string = rule1.PrintTiburon()
    expected_tiburon_string = \
      'q.NP(x0:NN) -> q1.x0 # 0.8'
    self.assertEqual(expected_tiburon_string, tiburon_string)

  def test_LongRootState(self):
    rule1 = XTRule('my_state', tree_or_string('(NP ?x0|NN)'),
                               tree_or_string('(NP ?x0|NN)'),
                               {(0,) : 'q1'}, 0.8)
    tiburon_string = rule1.PrintTiburon()
    expected_tiburon_string = \
      'my_state.NP(x0:NN) -> NP(q1.x0) # 0.8'
    self.assertEqual(expected_tiburon_string, tiburon_string)

  def test_PunctuationPOStag(self):
    rule1 = XTRule('q', tree_or_string('(. ?x0|NN)'),
                        tree_or_string('(NP ?x0|NN)'),
                        {(0,) : 'q1'}, 0.8)
    tiburon_string = rule1.PrintTiburon()
    expected_tiburon_string = \
      'q.PU(x0:NN) -> NP(q1.x0) # 0.8'
    self.assertEqual(expected_tiburon_string, tiburon_string)

  def test_PercentSymbol(self):
    rule1 = XTRule('q', tree_or_string('(NN %)'),
                        tree_or_string('(NN %)'),
                        {(0,) : 'q1'}, 0.8)
    tiburon_string = rule1.PrintTiburon()
    expected_tiburon_string = \
      'q.NN("PERCENT") -> NN("PERCENT") # 0.8'
    self.assertEqual(expected_tiburon_string, tiburon_string)

  def test_TypedVar(self):
    rule1 = XTRule('q', tree_or_string('(VP ?x0|.)'),
                        tree_or_string('(VP ?x0|.)'),
                        {(0,) : 'q1'}, 0.8)
    tiburon_string = rule1.PrintTiburon()
    expected_tiburon_string = \
      'q.VP(x0:PU) -> VP(q1.x0) # 0.8'
    self.assertEqual(expected_tiburon_string, tiburon_string)

  def test_SpecialCharInVarPOS(self):
    rule1 = XTRule('q', tree_or_string('(NP ?x0|:arg0)'),
                        tree_or_string('(NP ?x0|:arg0)'),
                        {(0,) : 'q1'}, 0.7)
    tiburon_string = rule1.PrintTiburon()
    expected_tiburon_string = \
      'q.NP(x0:COLONarg0) -> NP(q1.x0) # 0.7'
    self.assertEqual(expected_tiburon_string, tiburon_string)

  def test_SpecialCharInNT(self):
    rule1 = XTRule('q', tree_or_string('(:NP ?x0|DT)'),
                        tree_or_string('(:NP ?x0|DT)'),
                        {(0,) : 'q1'}, 0.7)
    tiburon_string = rule1.PrintTiburon()
    expected_tiburon_string = \
      'q.COLONNP(x0:DT) -> COLONNP(q1.x0) # 0.7'
    self.assertEqual(expected_tiburon_string, tiburon_string)

class CopyRuleTestCase(unittest.TestCase):
  def test_List(self):
    rule = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                       tree_or_string('(NPP ?x1|NN ?x0|DT)'),
                       {(0,) : 'q1', (1,) : 'q2'}, 0.8)
    rule_list = list([rule])
    rule_list[0].state = 'new'
    self.assertEqual(rule_list[0].state, rule.state)

  def test_Set(self):
    rule = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                       tree_or_string('(NPP ?x1|NN ?x0|DT)'),
                       {(0,) : 'q1', (1,) : 'q2'}, 0.8)
    rule_set = set([rule])
    popped_rule = rule_set.pop()
    popped_rule.state = 'new'
    self.assertEqual(popped_rule.state, rule.state)

  def test_SetTwoEqualWeightEqual(self):
    rule1 = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        tree_or_string('(NPP ?x1|NN ?x0|DT)'),
                        {(0,) : 'q1', (1,) : 'q2'}, 0.8)
    rule2 = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        tree_or_string('(NPP ?x1|NN ?x0|DT)'),
                        {(0,) : 'q1', (1,) : 'q2'}, 0.8)
    rule_set = set([rule1, rule2])
    self.assertEqual(1, len(rule_set))

  def test_SetTwoEqualWeightDifferent(self):
    rule1 = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        tree_or_string('(NPP ?x1|NN ?x0|DT)'),
                        {(0,) : 'q1', (1,) : 'q2'}, 0.8)
    rule2 = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        tree_or_string('(NPP ?x1|NN ?x0|DT)'),
                        {(0,) : 'q1', (1,) : 'q2'}, 0.3)
    # from pudb import set_trace; set_trace()
    rule_set = set([rule1, rule2])
    self.assertEqual(1, len(rule_set))

  def test_ListSet(self):
    rule = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                       tree_or_string('(NPP ?x1|NN ?x0|DT)'),
                       {(0,) : 'q1', (1,) : 'q2'}, 0.8)
    rule_set = list(set([rule]))
    popped_rule = rule_set.pop()
    popped_rule.state = 'new'
    self.assertEqual(popped_rule.state, rule.state)

  def test_RHS(self):
    rule = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                       tree_or_string('(NPP ?x1|NN ?x0|DT)'),
                       {(0,) : 'q1', (1,) : 'q2'}, 0.8)
    rhs = RHS(rule)
    rhs.rule.state = 'new'
    self.assertEqual(rhs.rule.state, rule.state)

  def test_Production(self):
    rule = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                       tree_or_string('(NPP ?x1|NN ?x0|DT)'),
                       {(0,) : 'q1', (1,) : 'q2'}, 0.8)
    rhs = RHS(rule)
    production = Production(('q', (), ()), rhs, 0.0)
    production.rhs.rule.state = 'new'
    self.assertEqual(production.rhs.rule.state, rule.state)

class PrintYamlTestCase(unittest.TestCase):
  def test_Newstates(self):
    rule1 = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        tree_or_string('(NPP ?x1|NN ?x0|DT)'),
                        {(0,) : 'q1', (1,) : 'q2'}, 0.8)
    yaml_string = rule1.PrintYaml()
    expected_yaml_string = '- state: q\n' \
                         + '  lhs: "(NP ?x0|DT ?x1|NN)"\n' \
                         + '  rhs: "(NPP ?x1|NN ?x0|DT)"\n' \
                         + '  newstates:\n' \
                         + '  - [[0], q1]\n' \
                         + '  - [[1], q2]\n' \
                         + '  weight: 0.8'
    self.assertEqual(expected_yaml_string, yaml_string)

  def test_NoNewstates(self):
    rule1 = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        tree_or_string('(NPP X)'),
                        {}, 0.8)
    yaml_string = rule1.PrintYaml()
    expected_yaml_string = '- state: q\n' \
                         + '  lhs: "(NP ?x0|DT ?x1|NN)"\n' \
                         + '  rhs: "(NPP X)"\n' \
                         + '  weight: 0.8'
    self.assertEqual(expected_yaml_string, yaml_string)

  def test_TerminalRule(self):
    rule1 = XTRule('t', tree_or_string('hello'),
                        tree_or_string('hola'),
                        {}, 0.7)
    yaml_string = rule1.PrintYaml()
    expected_yaml_string = '- state: t\n' \
                         + '  lhs: "hello"\n' \
                         + '  rhs: "hola"\n' \
                         + '  weight: 0.7'
    self.assertEqual(expected_yaml_string, yaml_string)

class ApplyRuleTestCase(unittest.TestCase):
  def test_TerminalApplyTerminal(self):
    rule1 = XTRule('t', tree_or_string('hello'),
                        tree_or_string('hola'),
                        {}, 1.0)
    tree = tree_or_string('hello')
    (transformed_tree, new_statemap) = rule1.apply(tree, {() : 't'})
    expected_tree = tree_or_string('hola')
    self.assertEqual(expected_tree, transformed_tree)
    self.assertEqual(new_statemap, {})

  def test_PreterminalWithTerminalVariable(self):
    rule1 = XTRule('q', tree_or_string('(DT ?x0|)'),
                        tree_or_string('?x0|'),
                        {() : 't'}, 1.0)
    tree = tree_or_string('(DT the)')
    (transformed_tree, new_statemap) = rule1.apply(tree, {() : 'q'})
    expected_tree = tree_or_string('the')
    self.assertEqual(expected_tree, transformed_tree)
    self.assertEqual(new_statemap, {() : 't'})

  def test_PreterminalWithPreterminalVariable(self):
    rule1 = XTRule('q', tree_or_string('?x0|DT'),
                        tree_or_string('(JJ ?x0|)'),
                        {(0,) : 'q'}, 1.0)
    tree = tree_or_string('(DT the)')
    (transformed_tree, new_statemap) = rule1.apply(tree, {() : 'q'})
    expected_tree = tree_or_string('(JJ (DT the))')
    self.assertEqual(expected_tree, transformed_tree)
    self.assertEqual(new_statemap, {(0,) : 'q'})

  def test_PreterminalApplyTerminal(self):
    rule1 = XTRule('t', tree_or_string('the'),
                        tree_or_string('la'),
                        {}, 1.0)
    tree = tree_or_string('(DT the)')
    (transformed_tree, new_statemap) = rule1.apply(tree, {(0,) : 't'})
    expected_tree = tree_or_string('(DT la)')
    self.assertEqual(expected_tree, transformed_tree)
    self.assertEqual(new_statemap, {})

  def test_PreterminalApplyPreterminal(self):
    rule1 = XTRule('q', tree_or_string('(DT ?x0|)'),
                        tree_or_string('(DTT ?x0|)'),
                        {(0,) : 't'}, 1.0)
    tree = tree_or_string('(DT the)')
    (transformed_tree, new_statemap) = rule1.apply(tree, {() : 'q'})
    expected_tree = tree_or_string('(DTT the)')
    self.assertEqual(expected_tree, transformed_tree)
    self.assertEqual(new_statemap, {(0,) : 't'})

  def test_PreterminalApplyPreterminalAndTerminal(self):
    rule1 = XTRule('q', tree_or_string('(DT ?x0|)'),
                        tree_or_string('(DTT ?x0|)'),
                        {(0,) : 't'}, 1.0)
    rule2 = XTRule('t', tree_or_string('the'),
                        tree_or_string('la'),
                        {}, 1.0)
    tree = tree_or_string('(DT the)')
    (transformed_tree, new_statemap) = rule1.apply(tree, {() : 'q'})
    (transformed_tree, new_statemap) = rule2.apply(transformed_tree, new_statemap)
    expected_tree = tree_or_string('(DTT la)')
    self.assertEqual(expected_tree, transformed_tree)
    self.assertEqual(new_statemap, {})

  def test_NonterminalWithTerminalVariable(self):
    rule1 = XTRule('q', tree_or_string('(NP (DT ?x0|) (NN house))'),
                        tree_or_string('?x0|'),
                        {}, 1.0)
    tree = tree_or_string('(NP (DT the) (NN house))')
    (transformed_tree, new_statemap) = rule1.apply(tree, {() : 't'})
    expected_tree = tree_or_string('the')
    self.assertEqual(expected_tree, transformed_tree)
    self.assertEqual(new_statemap, {})

  def test_NonterminalWithTwoTerminalVariables(self):
    rule1 = XTRule('q', tree_or_string('(NP (DT ?x0|) (NN ?x1|))'),
                        tree_or_string('?x1|'),
                        {}, 1.0)
    tree = tree_or_string('(NP (DT the) (NN house))')
    (transformed_tree, new_statemap) = rule1.apply(tree, {() : 't'})
    expected_tree = tree_or_string('house')
    self.assertEqual(expected_tree, transformed_tree)
    self.assertEqual(new_statemap, {})

  def test_NonterminalWithPreterminalVariable(self):
    rule1 = XTRule('q', tree_or_string('(NP ?x0|DT (NN house))'),
                        tree_or_string('?x0|'),
                        {() : 'q'}, 1.0)
    tree = tree_or_string('(NP (DT the) (NN house))')
    (transformed_tree, new_statemap) = rule1.apply(tree, {() : 'q'})
    expected_tree = tree_or_string('(DT the)')
    self.assertEqual(expected_tree, transformed_tree)
    self.assertEqual(new_statemap, {() : 'q'})

  def test_NonterminalTransformsIntoTerminal(self):
    rule1 = XTRule('t', tree_or_string('(NP (DT the) (NN house))'),
                        tree_or_string('casa'),
                        {}, 1.0)
    tree = tree_or_string('(NP (DT the) (NN house))')
    (transformed_tree, new_statemap) = rule1.apply(tree, {() : 't'})
    expected_tree = tree_or_string('casa')
    self.assertEqual(expected_tree, transformed_tree)
    self.assertEqual(new_statemap, {})

  def test_NonterminalTransformsIntoPreterminal(self):
    rule1 = XTRule('t', tree_or_string('(NP (DT the) (NN house))'),
                        tree_or_string('(NN casa)'),
                        {}, 1.0)
    tree = tree_or_string('(NP (DT the) (NN house))')
    (transformed_tree, new_statemap) = rule1.apply(tree, {() : 't'})
    expected_tree = tree_or_string('(NN casa)')
    self.assertEqual(expected_tree, transformed_tree)
    self.assertEqual(new_statemap, {})

  def test_NonterminalTransformsIntoNonterminal(self):
    rule1 = XTRule('t', tree_or_string('(NP (DT the) (NN house))'),
                        tree_or_string('(NP (DT la) (NN casa))'),
                        {}, 1.0)
    tree = tree_or_string('(NP (DT the) (NN house))')
    (transformed_tree, new_statemap) = rule1.apply(tree, {() : 't'})
    expected_tree = tree_or_string('(NP (DT la) (NN casa))')
    self.assertEqual(expected_tree, transformed_tree)
    self.assertEqual(new_statemap, {})

  def test_NonterminalApplyTerminal(self):
    rule1 = XTRule('t', tree_or_string('the'),
                        tree_or_string('la'),
                        {}, 1.0)
    tree = tree_or_string('(NP (DT the) (NN house))')
    (transformed_tree, new_statemap) = rule1.apply(tree, {(0, 0) : 't'})
    expected_tree = tree_or_string('(NP (DT la) (NN house))')
    self.assertEqual(expected_tree, transformed_tree)
    self.assertEqual(new_statemap, {})

  def test_NonterminalApplyPreterminal(self):
    rule1 = XTRule('q', tree_or_string('(DT ?x0|)'),
                        tree_or_string('(DTT ?x0|)'),
                        {(0,) : 't'}, 1.0)
    tree = tree_or_string('(NP (DT the) (NN house))')
    (transformed_tree, new_statemap) = rule1.apply(tree, {(0,) : 'q'})
    expected_tree = tree_or_string('(NP (DTT the) (NN house))')
    self.assertEqual(expected_tree, transformed_tree)
    self.assertEqual(new_statemap, {(0, 0) : 't'})

  def test_NonterminalApplyNonterminal(self):
    rule1 = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        tree_or_string('(NPP ?x1|NN ?x0|DT)'),
                        {(0,) : 'q1', (1,) : 'q2'}, 1.0)
    tree = tree_or_string('(NP (DT the) (NN house))')
    (transformed_tree, new_statemap) = rule1.apply(tree, {() : 'q'})
    expected_tree = tree_or_string('(NPP (NN house) (DT the))')
    self.assertEqual(expected_tree, transformed_tree)
    self.assertEqual(new_statemap, {(0,) : 'q1', (1,) : 'q2'})

  def test_NonterminalApplyNonterminalAndPreterminalAndTerminal(self):
    rule1 = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        tree_or_string('(NPP ?x1|NN ?x0|DT)'),
                        {(0,) : 'q1', (1,) : 'q2'}, 1.0)
    rule2 = XTRule('q2', tree_or_string('(DT ?x0|)'),
                         tree_or_string('(DTT ?x0|)'),
                         {(0,) : 't'}, 1.0)
    rule3 = XTRule('q1', tree_or_string('(NN ?x0|)'),
                         tree_or_string('(NNN ?x0|)'),
                         {(0,) : 't'}, 1.0)
    rule4 = XTRule('t', tree_or_string('the'),
                        tree_or_string('la'),
                        {}, 1.0)
    rule5 = XTRule('t', tree_or_string('house'),
                        tree_or_string('casa'),
                        {}, 1.0)
    tree = tree_or_string('(NP (DT the) (NN house))')
    (transformed_tree, new_statemap) = rule1.apply(tree, {() : 'q'})
    self.assertEqual(new_statemap, {(0,) : 'q1', (1,) : 'q2'})
    (transformed_tree, new_statemap) = rule3.apply(transformed_tree, new_statemap)
    self.assertEqual(new_statemap, {(0, 0) : 't', (1,) : 'q2'})
    (transformed_tree, new_statemap) = rule5.apply(transformed_tree, new_statemap)
    self.assertEqual(new_statemap, {(1,) : 'q2'})
    (transformed_tree, new_statemap) = rule2.apply(transformed_tree, new_statemap)
    self.assertEqual(new_statemap, {(1, 0) : 't'})
    (transformed_tree, new_statemap) = rule4.apply(transformed_tree, new_statemap)
    self.assertEqual(new_statemap, {})
    expected_tree = tree_or_string('(NPP (NNN casa) (DTT la))')
    self.assertEqual(expected_tree, transformed_tree)
    self.assertEqual(new_statemap, {})

  def test_NonterminalApplyNonterminalPOS(self):
    rule1 = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        tree_or_string('(NP ?x1|NNN ?x0|DTT)'),
                        {(0,) : 'q1', (1,) : 'q2'}, 1.0)
    tree = tree_or_string('(NP (DT the) (NN house))')
    (transformed_tree, new_statemap) = rule1.apply(tree, {() : 'q'})
    expected_tree = tree_or_string('(NP (NN house) (DT the))')
    self.assertEqual(expected_tree, transformed_tree)
    self.assertEqual(new_statemap, {(0,) : 'q1', (1,) : 'q2'})

class RuleComparisonTestCase(unittest.TestCase):
  def test_EqualTerminalIdentity(self):
    rule1 = XTRule('t', tree_or_string('hello'),
                        tree_or_string('hello'),
                        {}, 1.0)
    rule2 = XTRule('t', tree_or_string('hello'),
                        tree_or_string('hello'),
                        {}, 1.0)
    self.assertEqual(rule1, rule2)

  def test_EqualTerminalToNonIdentity(self):
    rule1 = XTRule('t', tree_or_string('hello'),
                        tree_or_string('(NP (DT the) (NN hello))'),
                        {}, 1.0)
    rule2 = XTRule('t', tree_or_string('hello'),
                        tree_or_string('(NP (DT the) (NN hello))'),
                        {}, 1.0)
    self.assertEqual(rule1, rule2)

  def test_EqualNonTerminalToNonTerminal(self):
    rule1 = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        {(0) : 'q', (1) : 'q'}, 1.0)
    rule2 = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        {(0) : 'q', (1) : 'q'}, 1.0)
    self.assertEqual(rule1, rule2)

  def test_EqualNonTerminalToTerminal(self):
    rule1 = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        tree_or_string('(NP (DT the) (NN hello))'),
                        {}, 1.0)
    rule2 = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        tree_or_string('(NP (DT the) (NN hello))'),
                        {}, 1.0)
    self.assertEqual(rule1, rule2)

  def test_NonEqual1(self):
    rule1 = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        {(0) : 'q', (1) : 'q'}, 1.0)
    rule2 = XTRule('q', tree_or_string('(DT ?x0|)'),
                        tree_or_string('(DT ?x0|)'),
                        {(0) : 't'}, 1.0)
    self.assertNotEqual(rule1, rule2)

  def test_NonEqual2(self):
    rule1 = XTRule('q', tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        tree_or_string('(NP ?x0|DT ?x1|NN)'),
                        {(0) : 'q', (1) : 'q'}, 1.0)
    rule2 = XTRule('t', tree_or_string('the'),
                        tree_or_string('la'),
                        {}, 1.0)
    self.assertNotEqual(rule1, rule2)

  def test_NonEqualDifferentPOS(self):
    rule1 = XTRule('q', tree_or_string('(DT ?x0|)'),
                        tree_or_string('(DT ?x0|)'),
                        {(0) : 't'}, 1.0)
    rule2 = XTRule('q', tree_or_string('(NN ?x0|)'),
                        tree_or_string('(NN ?x0|)'),
                        {(0) : 't'}, 1.0)
    self.assertNotEqual(rule1, rule2)

  @unittest.skip("Now we only compare rules structurally.")
  def test_NonEqualDifferentWeights(self):
    rule1 = XTRule('q', tree_or_string('(DT ?x0|)'),
                        tree_or_string('(DT ?x0|)'),
                        {(0) : 't'}, 1.0)
    rule2 = XTRule('q', tree_or_string('(DT ?x0|)'),
                        tree_or_string('(DT ?x0|)'),
                        {(0) : 't'}, 0.8)
    self.assertNotEqual(rule1, rule2)

class MakeDeletingRuleTestCase(unittest.TestCase):
  def test_NonterminalDeleteLeftLeaf(self):
    rule = \
      XTRule('q', tree_or_string('(NP the ?x0|NN)'),
                  tree_or_string('(NP ?x0|NN)'),
                  {}, 1.0)
    expected_deleting_rule = \
      XTRule('q', tree_or_string('(NP ?xx0| ?x0|NN)'),
                  tree_or_string('(NP ?x0|NN)'),
                  {}, 1.0)
    deleting_rule = rule.MakeDeletingRule()
    self.assertEqual(expected_deleting_rule, deleting_rule)

  def test_RHSVarLeafDeleteTerminal(self):
    rule = \
      XTRule('q', tree_or_string('(NP the ?x0|NN)'),
                  tree_or_string('?x0|NN'),
                  {}, 1.0)
    expected_deleting_rule = \
      XTRule('q', tree_or_string('(NP ?xx0| ?x0|NN)'),
                  tree_or_string('?x0|NN'),
                  {}, 1.0)
    deleting_rule = rule.MakeDeletingRule()
    self.assertEqual(expected_deleting_rule, deleting_rule)

  def test_RHSVarLeafDeletePreterminal(self):
    rule = \
      XTRule('q', tree_or_string('(NP (DT the) ?x0|NN)'),
                  tree_or_string('?x0|NN'),
                  {}, 1.0)
    expected_deleting_rule = \
      XTRule('q', tree_or_string('(NP ?xx0|DT ?x0|NN)'),
                  tree_or_string('?x0|NN'),
                  {}, 1.0)
    # from pudb import set_trace; set_trace()
    deleting_rule = rule.MakeDeletingRule()
    self.assertEqual(expected_deleting_rule, deleting_rule)

  def test_NonterminalDeleteLeftBranch(self):
    rule = \
      XTRule('q', tree_or_string('(NP (DT the) ?x0|NN)'),
                  tree_or_string('(NP ?x0|NN)'),
                  {}, 1.0)
    expected_deleting_rule = \
      XTRule('q', tree_or_string('(NP ?xx0|DT ?x0|NN)'),
                  tree_or_string('(NP ?x0|NN)'),
                  {}, 1.0)
    deleting_rule = rule.MakeDeletingRule()
    self.assertEqual(expected_deleting_rule, deleting_rule)

  def test_NonterminalDeleteRightBranch(self):
    rule = \
      XTRule('q', tree_or_string('(NP ?x0|DT (NN house))'),
                  tree_or_string('(NP ?x0|DT)'),
                  {}, 1.0)
    expected_deleting_rule = \
      XTRule('q', tree_or_string('(NP ?x0|DT ?xx0|NN)'),
                  tree_or_string('(NP ?x0|DT)'),
                  {}, 1.0)
    deleting_rule = rule.MakeDeletingRule()
    self.assertEqual(expected_deleting_rule, deleting_rule)

  def test_NonterminalDeleteMiddleBranch(self):
    rule = \
      XTRule('q', tree_or_string('(NP ?x0|DT (ADJ nice) ?x1|NN)'),
                  tree_or_string('(NP ?x0|DT ?x1|NN)'),
                  {}, 1.0)
    expected_deleting_rule = \
      XTRule('q', tree_or_string('(NP ?x0|DT ?xx0|ADJ ?x1|NN)'),
                  tree_or_string('(NP ?x0|DT ?x1|NN)'),
                  {}, 1.0)
    deleting_rule = rule.MakeDeletingRule()
    self.assertEqual(expected_deleting_rule, deleting_rule)

  def test_PreterminalVariableNoDeletion(self):
    rule = \
      XTRule('q', tree_or_string('(DT ?x0|)'),
                  tree_or_string('(DT ?x0|)'),
                  {}, 1.0)
    expected_deleting_rule = \
      XTRule('q', tree_or_string('(DT ?x0|)'),
                  tree_or_string('(DT ?x0|)'),
                  {}, 1.0)
    deleting_rule = rule.MakeDeletingRule()
    self.assertEqual(expected_deleting_rule, deleting_rule)

  def test_PreterminalNoVariableNoDeletion(self):
    rule = \
      XTRule('q', tree_or_string('(DT the)'),
                  tree_or_string('(DT the)'),
                  {}, 1.0)
    expected_deleting_rule = \
      XTRule('q', tree_or_string('(DT the)'),
                  tree_or_string('(DT the)'),
                  {}, 1.0)
    deleting_rule = rule.MakeDeletingRule()
    self.assertEqual(expected_deleting_rule, deleting_rule)

  def test_NonterminalDeleteLeftBranchRightLevel2(self):
    rule = \
      XTRule('q', tree_or_string('(NP (DT the) (NN ?x0|))'),
                  tree_or_string('(NP (NN ?x0|))'),
                  {}, 1.0)
    expected_deleting_rule = \
      XTRule('q', tree_or_string('(NP ?xx0|DT (NN ?x0|))'),
                  tree_or_string('(NP (NN ?x0|))'),
                  {}, 1.0)
    deleting_rule = rule.MakeDeletingRule()
    self.assertEqual(expected_deleting_rule, deleting_rule)

  def test_NonterminalDeleteLeftBranchRightComplex(self):
    rule = \
      XTRule('q', tree_or_string('(NP (DT the) (NP ?x0|DT (NN house)))'),
                  tree_or_string('(NP (NP ?x0|DT))'),
                  {}, 1.0)
    expected_deleting_rule = \
      XTRule('q', tree_or_string('(NP ?xx0|DT (NP ?x0|DT (NN house)))'),
                  tree_or_string('(NP (NP ?x0|DT))'),
                  {}, 1.0)
    deleting_rule = rule.MakeDeletingRule()
    self.assertEqual(expected_deleting_rule, deleting_rule)

  def test_NonterminalNoDeleteLeftBranchRightComplex(self):
    rule = \
      XTRule('q', tree_or_string('(NP (DT the) (NP ?x0|DT (NN house)))'),
                  tree_or_string('(NP (NP ?x0|DT (NN house)))'),
                  {}, 1.0)
    expected_deleting_rule = \
      XTRule('q', tree_or_string('(NP (DT the) (NP ?x0|DT (NN house)))'),
                  tree_or_string('(NP (NP ?x0|DT (NN house)))'),
                  {}, 1.0)
    deleting_rule = rule.MakeDeletingRule()
    self.assertEqual(expected_deleting_rule, deleting_rule)

  def test_NonterminalNoDelete(self):
    rule = \
      XTRule('q', tree_or_string('(NP (DT ?x0|) ?x1|NN)'),
                  tree_or_string('(NP (DT ?x0|) ?x1|NN)'),
                  {}, 1.0)
    expected_deleting_rule = \
      XTRule('q', tree_or_string('(NP (DT ?x0|) ?x1|NN)'),
                  tree_or_string('(NP (DT ?x0|) ?x1|NN)'),
                  {}, 1.0)
    deleting_rule = rule.MakeDeletingRule()
    self.assertEqual(expected_deleting_rule, deleting_rule)

  def test_NonterminalDeleteTwo(self):
    rule = \
      XTRule('q', tree_or_string('(NP (DT the) (ADJ nice) ?x1|NN)'),
                  tree_or_string('(NP ?x1|NN)'),
                  {}, 1.0)
    expected_deleting_rule = \
      XTRule('q', tree_or_string('(NP ?xx0|DT ?xx1|ADJ ?x1|NN)'),
                  tree_or_string('(NP ?x1|NN)'),
                  {}, 1.0)
    deleting_rule = rule.MakeDeletingRule()
    self.assertEqual(expected_deleting_rule, deleting_rule)

class RuleToTreePatternsTestCase(unittest.TestCase):
  def test_StringString(self):
    rule = \
      XTRule('q', tree_or_string('hola'),
                  tree_or_string('hello'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    self.assertEqual(src_treep.tree, 'hola')
    self.assertEqual(trg_treep.tree, 'hello')
    self.assertEqual(src_treep.path, ())
    self.assertEqual(trg_treep.path, ())
    self.assertEqual(src_treep.subpaths, [])
    self.assertEqual(trg_treep.subpaths, [])

  def test_StringVarOneLevelVar(self):
    rule = \
      XTRule('q', tree_or_string('?x0|NP'),
                  tree_or_string('(VP ?x0|VB)'),
                  {(0,) : 'q2'}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    self.assertEqual(src_treep.tree, tree_or_string('?x0|NP'))
    self.assertEqual(trg_treep.tree, tree_or_string('(VP ?x0|VB)'))
    self.assertEqual(src_treep.path, ())
    self.assertEqual(trg_treep.path, ())
    self.assertEqual(src_treep.subpaths, [()])
    self.assertEqual(trg_treep.subpaths, [(0,)])

  def test_PreterminalString(self):
    rule = \
      XTRule('q', tree_or_string('(DT la)'),
                  tree_or_string('the'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    self.assertEqual(src_treep.tree, tree_or_string('(DT la)'))
    self.assertEqual(trg_treep.tree, 'the')
    self.assertEqual(src_treep.path, ())
    self.assertEqual(trg_treep.path, ())
    self.assertEqual(src_treep.subpaths, [])
    self.assertEqual(trg_treep.subpaths, [])

  def test_PreterminalVarString(self):
    rule = \
      XTRule('q', tree_or_string('(DT ?x0|)'),
                  tree_or_string('(DT ?x0|)'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    self.assertEqual(src_treep.tree, tree_or_string('(DT ?x0|)'))
    self.assertEqual(trg_treep.tree, tree_or_string('(DT ?x0|)'))
    self.assertEqual(src_treep.path, ())
    self.assertEqual(trg_treep.path, ())
    self.assertEqual(src_treep.subpaths, [(0,)])
    self.assertEqual(trg_treep.subpaths, [(0,)])

  def test_PreterminalDelVarString(self):
    rule = \
      XTRule('q', tree_or_string('(NP ?xx0| (NN ?x0|))'),
                  tree_or_string('(NP (NN ?x0|))'),
                  {}, 1.0)
    src_treep, trg_treep = rule.GetTreePatterns()
    self.assertEqual(src_treep.tree, tree_or_string('(NP ?xx0| (NN ?x0|))'))
    self.assertEqual(trg_treep.tree, tree_or_string('(NP (NN ?x0|))'))
    self.assertEqual(src_treep.path, ())
    self.assertEqual(trg_treep.path, ())
    self.assertEqual(src_treep.subpaths, [(0,), (1, 0)])
    self.assertEqual(trg_treep.subpaths, [(0, 0)])

if __name__ == '__main__':
  suite1  = unittest.TestLoader().loadTestsFromTestCase(ApplyRuleTestCase)
  suite2  = unittest.TestLoader().loadTestsFromTestCase(RuleComparisonTestCase)
  suite3  = unittest.TestLoader().loadTestsFromTestCase(CopyRuleTestCase)
  suite4  = unittest.TestLoader().loadTestsFromTestCase(PrintTiburonTestCase)
  suite5  = unittest.TestLoader().loadTestsFromTestCase(ParseTiburonTestCase)
  suite6  = unittest.TestLoader().loadTestsFromTestCase(MakeDeletingRuleTestCase)
  suite7  = unittest.TestLoader().loadTestsFromTestCase(RuleToTreePatternsTestCase)
  suites  = unittest.TestSuite([
    suite1, suite2, suite3, suite4, suite5, suite6, suite7])
  unittest.TextTestRunner(verbosity=2).run(suites)

