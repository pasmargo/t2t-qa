from collections import namedtuple
from copy import deepcopy
from functools import total_ordering

from semirings.semiring_prob import ProbSemiRing
from utils.tree_tools import variables_to_paths, IsString

class Derivation:
  def __init__(self, derivation):
    self.derivation = derivation

@total_ordering
class RHS:
  """
  Right-hand side of a production in a wRTG.
  It consists of a rule (that acts as a label),
  and optionally some non-terminals (other states).
  """
  def __init__(self, rule, non_terminals = []):
    self.rule = rule
    nts = []
    for nt in non_terminals:
      if isinstance(nt[-1], tuple):
        nts.append(nt + ('',))
    self.non_terminals = nts # List of non-terminals
    # self.non_terminals = list(non_terminals) # List of non-terminals

  def __eq__(self, other):
    return (self.rule == other.rule and \
            self.non_terminals == other.non_terminals)

  def __eq__2(self, other):
    return (self.rule == other.rule \
            and all([(x in other.non_terminals) for x in self.non_terminals]))

  def __ne__(self, other):
    return not __eq__(other)

  def __lt__(self, other):
    return (tuple(self.non_terminals), self.rule) < \
           (tuple(other.non_terminals), other.rule)

  def __hash__(self):
    return hash((tuple(self.non_terminals), hash(self.rule)))
    # return hash(repr(self))

  def __repr__(self):
    if not self.non_terminals:
      return ('rule')
    return ('rule' + str(hash(repr(self.rule))) + '(' \
            + ','.join(map(lambda nt: PrintNonterminal(nt), self.non_terminals)) + ')')

  def __str__(self):
    return repr(self)

NonterminalNamedTuple = namedtuple('Nonterminal', 'state i o rhstag')
class Nonterminal(NonterminalNamedTuple):
  def __repr__(self):
    pass

  def __getnewargs__(self):
    pass

def PrintNonterminal(non_terminal):
  nt_str_list = []
  for n in non_terminal:
    if not isinstance(n, list) and not isinstance(n, tuple):
      nt_str_list.append(str(n))
    else:
      nt_str_list.append(''.join([str(i) for i in n]))
  return '.'.join(nt_str_list)
    
def PrintNonterminal_(non_terminal):
  nt_str = '{0}.{1}'.format(
    non_terminal[0],
    ''.join([str(i) for i in non_terminal[1]]))
  if len(non_terminal) > 2:
    nt_str += '.' + ''.join([str(i) for i in non_terminal[2]])
  return nt_str

@total_ordering
class Production:
  """
  Production rule of a weighted regular tree grammar (wRTG).
  It consists of a non-terminal symbol on the left-hand side,
  a right-hand-side (which is a rule, acting as a label) plus
  other optional non-terminal symbols (states).
  There is also a weight associated with the production,
  via the RHS.
  """
  def __init__(self, non_terminal, rhs, weight):
    if isinstance(non_terminal[-1], tuple):
      self.non_terminal = non_terminal + ('',)
    else:
      self.non_terminal = non_terminal
    self.rhs = rhs
    self.representation = None
    self.representation_no_weigth = None

  def __eq__(self, other):
    return (self.non_terminal == other.non_terminal \
            and self.rhs == other.rhs)

  def __ne__(self, other):
    return not __eq__(other)

  def __lt__(self, other):
    return (self.non_terminal, self.rhs) < (other.non_terminal, other.rhs)

  def __hash__(self):
    return hash((self.non_terminal, hash(self.rhs)))
    # return hash(self.StringifyWithoutWeight())
    # return hash(repr(self))

  def __repr__(self):
    if self.representation != None:
      return self.representation
    self.representation = \
      '{0} -> {1} # {2}'.format(PrintNonterminal(self.non_terminal),
                                self.rhs, self.rhs.rule.weight)
    return self.representation

  def __str__(self):
    return repr(self)


  def StringifyWithoutWeight(self):
    if not self.representation_no_weigth:
      self.representation_no_weight = \
        '{0} -> {1}'.format(PrintNonterminal(self.non_terminal), self.rhs)
    return self.representation_no_weigth

def SourceProjectionFromDerivationMix(derivation):
  """
  Returns a tuple (Tree, weight), given a derivation (sequence of productions).
  The Tree is the right-hand-side (target side) of transformation rules
  that have new states, and the left-hand-side (source side) of transformation
  rules that do not have new states. This mode of source projection might be
  useful to insert target-side particles (such as wo in Japanese), and to remove
  source-side particles that do not have a direct translation.
  """
  weight = ProbSemiRing(1.0)
  tree = None
  for production in derivation:
    rule = production.rhs.rule
    if not rule.newstates:
      subtree_replacement = rule.lhs
    else:
      subtree_replacement = rule.rhs
    if tree == None or IsString(tree):
      tree = deepcopy(subtree_replacement)
    else:
      q_state, in_path, out_path = production.non_terminal[:3]
      tree[out_path] = deepcopy(subtree_replacement)
    weight *= rule.weight
  return (tree, weight)

def GetRemappedRulePaths(production, remapped_rule_paths):
  in_path = production.non_terminal[1]
  in_path_remapped = remapped_rule_paths[in_path]
  rule = production.rhs.rule
  src_vars_paths = variables_to_paths(rule.lhs)
  src_vars_paths = [(var.split('|')[0], path) for var, path in src_vars_paths]
  src_vars_to_paths = {x[0] : x[1] for x in src_vars_paths}
  assert sorted(src_vars_paths, key=lambda x: x[0]) == src_vars_paths, \
    'Variables in lhs are not sorted: {0}'.format(rule.lhs)
  trg_vars_paths = variables_to_paths(rule.rhs)
  trg_vars_paths = [(var.split('|')[0], path) for var, path in trg_vars_paths]
  trg_vars_to_paths = {x[0] : x[1] for x in trg_vars_paths}
  # Check whether the lhs and rhs have the same variable names.
  src_vars = [varpath[0] for varpath in src_vars_paths]
  trg_vars = [varpath[0] for varpath in trg_vars_paths]
  assert set(src_vars) == set(trg_vars), \
    'Variables in lhs {0} and rhs {1} differ:'.format(src_vars, trg_vars)
  for src_var, trg_var in zip(src_vars, trg_vars):
    src_path = src_vars_to_paths[trg_var]
    src_path_remapped = in_path_remapped + src_path
    remapped_rule_paths[in_path + src_vars_to_paths[src_var]] = src_path_remapped
  return remapped_rule_paths

def SourceProjectionFromDerivationStrict(derivation):
  """
  Returns a tuple (Tree, weight), given a derivation (sequence of productions).
  The Tree is the left-hand-side (source side) of the transformation rules,
  where variables are sorted in target-side order.
  """
  # Build the source tree with re-mapped paths.
  remapped_rule_paths = { () : () }
  weight = ProbSemiRing(1.0)
  tree = None
  for production in derivation:
    remapped_rule_paths = GetRemappedRulePaths(production, remapped_rule_paths)
    in_path = production.non_terminal[1]
    in_path_remapped = remapped_rule_paths[in_path]
    rule = production.rhs.rule
    subtree_replacement = deepcopy(rule.lhs)
    if tree == None or IsString(tree):
      tree = subtree_replacement
    else:
      tree[in_path_remapped] = subtree_replacement
    weight *= rule.weight
  return (tree, weight)

def GetInitialStateFromDerivation(derivation):
  q_start = 'q0'
  if len(derivation) > 0:
    first_production = derivation[0]
    q_start = first_production.non_terminal[0]
  return q_start

def GetInitialInPathFromDerivation(derivation):
  in_start = ()
  if len(derivation) > 0:
    first_production = derivation[0]
    in_start = first_production.non_terminal[1]
  return in_start

# TODO: when start using namedtuples, then we should
# obtain these values using named fields.
def GetInitialOutPathFromDerivation(derivation):
  out_start = ()
  if len(derivation) > 0:
    first_production = derivation[0]
    # This is the in_path if NTs are only (q, i) or out_path if NTs are (q, i, o).
    if not isinstance(first_production.non_terminal[2], tuple):
      out_start = first_production.non_terminal[1]
  return out_start

def GetIn2OutPathMapping(derivation):
  """
  Build a dictionary : (q, in_path) -> out_path that stores
  correspondences between the absolute input paths and absolute
  output paths.
  """
  q_start = GetInitialStateFromDerivation(derivation)
  in_start = GetInitialInPathFromDerivation(derivation)
  out_start = GetInitialOutPathFromDerivation(derivation)
  in_to_out_paths = {(q_start, in_start) : out_start}
  for production in derivation:
    q, in_path = production.non_terminal[:2] # NTs are (q, i, o) or (q, i).
    out_path = in_to_out_paths[(q, in_path)]
    rule = production.rhs.rule
    in_rel_paths = rule.lhs_vars_to_paths # Relative source paths (to the rule).
    out_rel_paths = rule.rhs_vars_to_paths # Relative target paths.
    in_abs_paths = \
      {var : in_path + rel_path for (var, rel_path) in in_rel_paths.items()}
    out_abs_paths = \
      {var : out_path + rel_path for (var, rel_path) in out_rel_paths.items()}
    for var in in_abs_paths.keys():
      if var in out_abs_paths:
        var_in_abs_path = in_abs_paths[var]
        var_out_abs_path = out_abs_paths[var]
        state = rule.newstates[out_rel_paths[var]]
        assert (state, var_in_abs_path) not in in_to_out_paths
        in_to_out_paths[(state, var_in_abs_path)] = var_out_abs_path
  return in_to_out_paths

def TargetProjectionFromDerivation(derivation):
  """
  Returns a tuple (Tree, weight), given a derivation (sequence of productions).
  This version does not use the left-hand-sides of productions to figure out
  the output path o that corresponds to each input path.
  """
  weight = 1.0
  tree = None
  # Mapping between input paths i to output paths o (as in (q, i, o)).
  in_to_out_paths = GetIn2OutPathMapping(derivation)
  out_start = GetInitialOutPathFromDerivation(derivation)
  for production in derivation:
    q, in_path = production.non_terminal[:2] # NTs are (q, i, o) or (q, i).
    out_path = in_to_out_paths[(q, in_path)]
    rule = production.rhs.rule
    if tree == None or IsString(tree):
      tree = deepcopy(rule.rhs)
    else:
      tree[out_path[len(out_start):]] = deepcopy(rule.rhs)
    weight *= rule.weight
  assert not variables_to_paths(tree), \
    'Tree was left incomplete: %s for derivation %s' % (tree, derivation)
  return (tree, weight)

