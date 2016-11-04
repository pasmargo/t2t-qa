import codecs
from collections import defaultdict

from utils.tree_tools import GetLeaves, IsVariable, IsString

cvts = set()
with codecs.open('qald/cvts.txt', 'r', 'utf-8') as f:
  cvts = set([cvt.strip() for cvt in f.readlines()])

operators = ['COUNT', 'MAX', 'MIN', '[]']
def IsOperator(uri):
  return uri in operators

def FilterOutRulesWithCVT(rules):
  remaining_rules = []
  for r in rules:
    all_leaves = GetLeaves(r.rhs)
    for leaf in all_leaves:
      if IsVariable(leaf) or IsOperator(leaf):
        continue
      if leaf.lstrip('!') in cvts:
        break
    else:
      remaining_rules.append(r)
  return remaining_rules

class RuleFilter(object):
  """
  Indexes conditions used to filter rules.
  """

  def __init__(self, state_conditions_list):
    """
    state_conditions is a dictionary where keys are states,
    and values are lists of conditions. E.g.:
    {'q' : [["rhs:is_var"], ["rhs:is_str"], ["lhs:is_var", "lhs:is_str"]]}
    In this case, a rule with state "q" will be filtered out
    if the RHS is a variable, OR
    if the RHS is a string, OR
    if the LHS is a variable AND a string.
    It is possible that a state has no conditions, in which
    case a rule will be signaled unconditionally. E.g.:
    {'entity' : []}
    """
    self.sconds = self.parse(state_conditions_list)
    self.states = set(self.sconds.keys())

  def parse(self, state_conditions_list):
    sconds = defaultdict(list)
    for state_conds in state_conditions_list:
      conds = state_conds.split(',')
      assert conds
      state = conds[0]
      sconds[state].append(conds[1:])
    return sconds

  def rule_meets_conds(self, rule, conds):
    if not conds:
      return True
    for cond in conds:
      target = rule.lhs if cond.startswith('lhs:') else rule.rhs
      if cond.endswith('is_var') and not IsVariable(target):
        return False
      if cond.endswith('is_str') and not IsString(target):
        return False
    return True

  def signal_rule(self, rule):
    if rule.state not in self.states:
      return False
    conds_lists = self.sconds.get(rule.state, [])
    return any([self.rule_meets_conds(rule, conds) for conds in conds_lists])

def FilterOutRulesByStates(rules, state_conditions):
  """
  Filter out all rules whose state and rule conditions
  are specified in @state_conditions.
  """
  rule_filter = RuleFilter(state_conditions)
  return [r for r in rules if not rule_filter.signal_rule(r)]

