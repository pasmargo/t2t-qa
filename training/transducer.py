import codecs
from collections import defaultdict
import copy
import logging
from operator import itemgetter

from nltk import ImmutableTree
# from pudb import set_trace; set_trace()

from semirings.semiring_prob import ProbSemiRing
from training.ruleindex import RuleIndexT2T
from training.wrtg import *
from utils.debug import warning, info
from utils.tree_tools import variables_to_paths, GetLeaves, Tree, GetPosAt

class xT:
  """
  An extended Transducer xT is a 5-tuple xT = (Sigma, Delta, Q, Qi, R) where:
  Sigma: is the input vocabulary in_vocab
  Delta: is the output vocabulary out_vocab
  Q: is the set (list) of states.
  Qi: is the initial (or start) state start_state.
  R: is the list of rules.
  """
  def __init__(self, start_state, rules, rule_backoffs = None):
    self.in_vocab  = GetInputVocabulary(rules)
    self.out_vocab = GetOutputVocabulary(rules)
    self.states = set([rule.state for rule in rules])
    self.start_state = start_state
    self.rules = list(set(rules))
    self.rule_index = RuleIndexT2T(self.rules, rule_backoffs)
    self.production_index = defaultdict(set)

  def SaveRulesInFile(self, output_filename, fmt='json'):
    foutput = open(output_filename, 'w')
    if fmt not in ['json', 'yaml']:
      raise ValueError('Format to print rules not recognized: {0}'.format(fmt))
    if fmt == 'json':
      PrintRule = lambda x: x.PrintJson() + '\n'
    elif fmt == 'yaml':
      PrintRule = lambda x: x.PrintYaml() + '\n\n'
    sorted_rules = sorted(self.rules, key=lambda r: r.lhs)
    for rule in sorted_rules:
      foutput.write(PrintRule(rule))
    foutput.close()

  def DerivTreeToTree(self, intree, outtree):
    """
    Based on Algorithm 1 in "Training Tree Transducers". Takes a list of rules,
    a start state, an observed input tree and an observed output tree. Returns
    the derivation wRTG that describes all of the weighted derivation trees
    that could have produced the output tree.
    """
    q, i, o = self.start_state, (), ()
    productions, non_terminals = self.Produce(intree, outtree, q, i, o)
    wrtg = wRTG(self.rules, non_terminals, (q, i, o), productions)
    prunned_wrtg = wrtg.Prune()
    return prunned_wrtg

  def Transduce(self, intree, outtree = None, convert_to_prob=True):
    """
    Returns a prunned wRTG given an input tree and optionally an output tree.
    Weights of the RTG are not estimated.
    """
    q, i, o = self.start_state, (), ()
    initial_state = (q, i) if outtree is None else (q, i, o)
    productions, non_terminals = self.Produce(intree, outtree, q, i, o)
    if not productions and outtree:
      logging.debug("The following tree pair could not be explained by the transducer"
                    " ||| {0} ||| {1}"\
                    .format(intree, outtree))
      return wRTG(self.rules, [], initial_state, [], convert_to_prob)
    rules = list(set([production.rhs.rule for production in productions]))
    wrtg = wRTG(rules, non_terminals, initial_state, productions, convert_to_prob)
    prunned_wrtg = wrtg.Prune()
    return prunned_wrtg

  def Produce(self, intree, outtree, q, i, o):
    # Create production index (which is a tree) into self.production_index.
    self.production_index = defaultdict(set)
    # Clear cache of rule_index.
    self.rule_index.ClearCache()
    success = self.MakeProductionIndex(intree, outtree, q, i, o)
    productions = self.GetProductionsFromProductionIndex()
    non_terminals = self.GetNonterminalsFromProductionIndex()
    return productions, non_terminals

  def GetProductionsFromProductionIndex(self):
    productions = list(set(itertools.chain(*self.production_index.values())))
    productions = copy.deepcopy(productions)
    return productions

  def GetNonterminalsFromProductionIndex(self):
    """
    The keys of self.production_index are of the form (q, i, rhs_child_pos) or
    (q, i, o, rhs_child_pos). To retrieve the non-terminals, we need to discard
    the rhs_child_pos from these tuples.
    """
    # Check that all non-terminals have the same length (q, i) or (q, i, o).
    assert len(set([len(nt) for nt in self.production_index.keys()])) == 1
    result_ids = self.production_index.keys()
    # non_terminals = [r[:-1] for r in result_ids]
    non_terminals = [r for r in result_ids]
    return non_terminals

  def MakeProductionIndex(self, intree, outtree, q, i, o, rhs_child_pos=''):
    """
    This method populates self.production_index : NonTerminal -> [Production],
    where NonTerminal is a 3-tuple (state, in_path, out_path).
    self.production_index also stores key-value pairs where the key is a 2-tuple
    of the form (state, in_path).
    """
    # Check cache for already-computed solutions.
    # non_terminal = (q, i) if outtree is None else (q, i, o)
    non_terminal = (q, i, rhs_child_pos) if outtree is None else \
                   (q, i, o, rhs_child_pos)
    if non_terminal in self.production_index:
      return self.production_index[non_terminal]
    relevant_rules = self.rule_index.GetRelevantRules(
      intree, (i, q), outtree, o, rhs_child_pos)
    productions = []
    for rule in relevant_rules:
      # TODO: In the case this rule has just been made for the occasion,
      # it should inherit the newstates from its parent rule.
      if rule.newstates == None:
        rule.newstates = {}
      # When converting list of tuples (var, path) into a dictionary,
      # we are assuming that each variable appears at most once.
      # This is a limitation (see Tree Adjoining Grammars for more expressivity).
      lhs_variable_paths = rule.lhs_vars_to_paths
      rhs_variable_paths_sorted = sorted(rule.rhs_vars_to_paths.items())
      deriv_rhs = RHS(rule)
      for rhs_variable, rhs_path in rhs_variable_paths_sorted:
        lhs_path = lhs_variable_paths[rhs_variable]
        lhs_absolute_path = i + lhs_path
        rhs_absolute_path = o + rhs_path
        new_state = rule.newstates[rhs_path]
        # Avoid infinite recursion when target tree is unconstrained
        # and LHS is non-consuming.
        if outtree is None and (q, i) == (new_state, lhs_absolute_path):
          assert len(rhs_variable_paths_sorted) == 1
          break
        rhs_child_pos = GetPosAt(rule.rhs, rhs_path)
        success = self.MakeProductionIndex(
          intree, outtree, new_state, lhs_absolute_path, rhs_absolute_path,
          rhs_child_pos)
        if not success:
          break
        if outtree is None:
          child_non_terminal = (new_state, lhs_absolute_path, rhs_child_pos)
        else:
          child_non_terminal = \
            (new_state, lhs_absolute_path, rhs_absolute_path, rhs_child_pos)
        deriv_rhs.non_terminals.append(child_non_terminal)
      else:
        production = Production(non_terminal, deriv_rhs, rule.weight)
        # Store partial solutions to avoid later recomputations.
        productions.append(production)
    self.production_index[non_terminal] = set(productions)
    return self.production_index[non_terminal]

def GetInputVocabulary(rules):
  return set([l for rule in rules \
              for l in GetLeaves(rule.lhs) \
              if not l.startswith('?x')])

def GetOutputVocabulary(rules):
  return set([l for rule in rules \
              for l in GetLeaves(rule.rhs) \
              if not l.startswith('?x')])

def DerivTreeToString(rules, initial, intree, outstring):
  """
  Based on Algorithm 1 in "Training Tree Transducers". Takes a list of rules,
  a start state, an observed input tree and an observed output tree. Returns
  the derivation wRTG that describes all of the weighted derivation trees
  that could have produced the output tree.
  """

  return None
 
