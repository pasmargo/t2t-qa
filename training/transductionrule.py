from copy import deepcopy
from functools import total_ordering
import simplejson
from operator import itemgetter
import re

from nltk import Tree as NLTKTree

from utils.tree_tools import (tree_index, variables_to_paths,
  GetTreeString, Quote, Unquote, Tree, IsString, get_top, IsVariable,
  tree_or_string, TreePattern, GetLeaves)
from utils.tools_generic_transducer import StanfordToTiburon
from semirings.semiring_prob import ProbSemiRing

def GetTreePattern(tree, subpaths):
  """
  Converts a rule LHS or RHS into a TreePattern.
  The tree attribute of the TreePattern would simply be the
  LHS or RHS tree.
  The path to the root (beginning) of the TreePattern would be (),
  because we do not have the real information on at what level this
  rule was originally extracted (or is being applied).
  The subpaths of the TreePattern would be the relative paths of the
  variables in the LHS or RHS.
  """
  path = ()
  if IsString(tree):
    if IsVariable(tree):
      return TreePattern(tree, path, [()])
    else:
      return TreePattern(tree, path, [])
  subpaths_sorted = sorted(subpaths)
  return TreePattern(tree, path, subpaths_sorted)

@total_ordering
class XTRule:
  """One rule in a tree transducer. It has a left-hand side (a pattern),
  a right-hand side and a weight."""

  def __init__(self, state, lhs, rhs, newstates, weight):
    self.lhs = lhs
    self.rhs = rhs
    self.state = state
    self.newstates = newstates
    self.weight = weight
    self.tied_to = None
    self.features = None
    self.stringified = None
    self.yaml_repr = None
    self.feat_descr = None
    self._lhs_vars_to_paths = None
    self._rhs_vars_to_paths = None

  def __str__(self):
    return repr(self)

  def __repr__(self):
    lhs_str = self.lhs.encode('utf-8') if IsString(self.lhs) else repr(self.lhs)
    rhs_str = self.rhs.encode('utf-8') if IsString(self.rhs) else repr(self.rhs)
    return (("<rule.\n  state: {0}\n  lhs: {1}\n  rhs: {2}\n" +
             "  newstates: {3}\n  weight: {4}>").format(
        self.state, lhs_str, rhs_str, self.newstates, self.weight))

  def __eq__2(self, other):
    if isinstance(self.weight, ProbSemiRing) or \
       isinstance(other.weight, ProbSemiRing):
      are_equal_weight = (self.weight == other.weight)
    else:
      are_equal_weight = round(self.weight, 5) == round(other.weight, 5)
    return isinstance(other, self.__class__) \
           and self.rhs == other.rhs \
           and self.lhs == other.lhs \
           and self.state == other.state \
           and self.newstates == other.newstates \
           and are_equal_weight

  def __eq__(self, other):
    return isinstance(other, self.__class__) \
           and self.rhs == other.rhs \
           and self.lhs == other.lhs \
           and self.state == other.state \
           and self.newstates == other.newstates \

  def __ne__(self, other):
    return not self.__eq__(other)

  def __lt__(self, other):
    return (self.lhs, self.rhs, self.state, self.newstates) < \
           (other.lhs, other.rhs, other.state, other.newstates)
 
  def __hash__(self):
    return hash(self.StringifyWithoutWeight())

  def StringifyWithoutWeight(self):
    if not self.stringified:
      lhs_str = self.lhs.encode('utf-8') if IsString(self.lhs) else repr(self.lhs)
      rhs_str = self.rhs.encode('utf-8') if IsString(self.rhs) else repr(self.rhs)
      self.stringified = ("<rule.\n  state: {0}\n  lhs: {1}\n  rhs: {2}\n" +
        "  newstates: {3}>").format(self.state, lhs_str, rhs_str, self.newstates)
    return self.stringified

  @property
  def lhs_vars_to_paths(self):
    if self._lhs_vars_to_paths is None:
      self._lhs_vars_to_paths = \
        {var.split('|')[0] : path for (var, path) in variables_to_paths(self.lhs)}
    return self._lhs_vars_to_paths

  @property
  def rhs_vars_to_paths(self):
    if self._rhs_vars_to_paths is None:
      self._rhs_vars_to_paths = \
        {var.split('|')[0] : path for (var, path) in variables_to_paths(self.rhs)}
    return self._rhs_vars_to_paths

  def GetTreePatterns(self):
    """
    Transforms the LHS and RHS into TreePattern(s).
    """
    lhs_vars = set(self.lhs_vars_to_paths.keys())
    rhs_vars = set(self.rhs_vars_to_paths.keys())
    lhs_subpaths = [self.lhs_vars_to_paths[v] for v in lhs_vars]
    rhs_subpaths = [self.rhs_vars_to_paths[v] for v in rhs_vars]
    src_treep = GetTreePattern(self.lhs, lhs_subpaths)
    trg_treep = GetTreePattern(self.rhs, rhs_subpaths)
    return src_treep, trg_treep

  def MakeDeletingRule(self):
    """
    If the LHS does not produce any leaf but RHS does, such rule can be
    considered as a leaf-deleting rule. It is not clear when lexicalized
    branches should be replaced by a deleting variable (it depends on the
    application). Here we replace fully lexicalied branches at level 1
    by a deleting variable, only when the RHS does not contain any leaf
    that is not a variable.
    """
    if IsString(self.lhs):
      return self
    if IsString(self.rhs) and not IsVariable(self.rhs):
      return self
    if not IsString(self.rhs):
      rhs_leaves = self.rhs.leaves()
      if rhs_leaves and any([not IsVariable(l) for l in rhs_leaves]):
        return self
    # Make generator of fresh variables.
    index_new_variable = ('?xx%d|' % i for i in xrange(20))
    # Substitute branches at level 1 if they are fully lexicalized.
    lhs_paths_prefix_1 = set([p[0] for p in self.lhs_vars_to_paths.values()])
    if not lhs_paths_prefix_1:
      return self
    for i, branch in enumerate(self.lhs):
      if i not in lhs_paths_prefix_1:
        if IsString(branch):
          self.lhs[i] = index_new_variable.next()
        else:
          self.lhs[i] = index_new_variable.next() + get_top(branch)
        self.lhs_vars_to_paths[self.lhs[i]] = (i,)
    return self

  def PrintTiburon(self):
    lhs_string = GetTreeString(self.lhs)
    rhs_string = GetTreeString(self.rhs)
    # Change LISP tree bracketing into Tiburon bracketing.
    lhs_string = StanfordToTiburon(lhs_string)
    rhs_string = StanfordToTiburon(rhs_string)
    # Replace variables ?x[0-9]+ by x[0-9]+.
    lhs_string = lhs_string.replace('?x', 'x')
    rhs_string = rhs_string.replace('?x', 'x')
    # Replace typed variables ?x[0-9]|YY by ?x[0-9]:YY or ?x[0-9]| by ?x[0-9]:
    lhs_string = lhs_string.replace('|', ':')
    rhs_string = rhs_string.replace('|', ':')
    if not isinstance(self.lhs, Tree):
      lhs_string = Quote(lhs_string)
    if not isinstance(self.rhs, Tree):
      rhs_string = Quote(rhs_string)
    lhs_string = BuildTiburonLHS(self.lhs)
    rhs_string = BuildTiburonRHS(self.rhs, self.newstates)
    weight = ' # ' + str(float(self.weight)) if self.weight is not None else ''
    tie = ' @ ' + str(self.tied_to) if self.tied_to is not None else ''
    return self.state + '.' + lhs_string + ' -> ' + rhs_string + weight + tie

  def PrintJson(self):
    json_rule = {'rule' : {'state' : self.state,
                           'lhs' : GetTreeString(self.lhs),
                           'rhs' : GetTreeString(self.rhs),
                           'newstates' : self.newstates.items(),
                           'weight' : repr(self.weight)}
                }
    if self.tied_to is not None:
      json_rule['rule']['tied_to'] = self.tied_to
    return unicode(simplejson.dumps(json_rule, ensure_ascii=False))

  def PrintYaml(self):
    if not self.yaml_repr is None:
      return self.yaml_repr
    lhs_string = GetTreeString(self.lhs)
    rhs_string = GetTreeString(self.rhs)
    yaml_string = (u'- state: {0}\n'
                    '  lhs: "{1}"\n'
                    '  rhs: "{2}"\n').format(self.state, lhs_string, rhs_string)
    if self.newstates:
      newstates_strings = []
      path_states = list([(path, state) for (path, state) in self.newstates.items()])
      path_states = sorted(path_states, key = itemgetter(0))
      for path, newstate in path_states:
        newstates_strings.append(u'  - [' + unicode(list(path)) + ', ' + newstate + ']')
      yaml_string += u'  newstates:\n'
      yaml_string += u'\n'.join(newstates_strings) + u'\n'
    yaml_string += u'  weight: {0}'.format(self.weight)
    if self.tied_to is not None:
      yaml_string += u'\n  tied_to: {0}'.format(self.tied_to)
    if self.features is not None:
      yaml_string += u'\n  features: {0}'.format(repr(self.features))
    self.yaml_repr = escape(yaml_string)
    return self.yaml_repr

  def apply(self, tree, statemap):
    """Returns a pair (new tree, new statemap) by applying this rule"""
    if not statemap:
      return (None, None)
    path_states = list([(path, state) for (path, state) in statemap.items()])
    # Sort path_state map so that we apply the rule to the left-most branch
    # consistently.
    path_states = sorted(path_states, key = itemgetter(0))
    # Get the left-most path and state.
    (path, state) = path_states[0]
    indexed = tree_index(tree, path)

    newsubtree = replace(indexed, self.lhs, self.rhs)
    newstates_with_prepend = {
     tuple(list(path) + list(rulepath)) : state
     for (rulepath, state) in self.newstates.items()
    }
    newstatemap = deepcopy(statemap)
    del newstatemap[path]
    newstatemap.update(newstates_with_prepend)

    if path == ():
      out = (newsubtree, newstatemap)
    else:
      newtree = deepcopy(tree)
      newtree[path] = newsubtree
      out = (newtree, newstatemap)
    return out
 
def replace(concrete, lhs, rhs, failok=False):
  """Given a concrete tree (ie, maybe a subtree of a larger tree) and lhs and
  rhs patterns, produce the new tree resulting from substituting the things
  in concrete in the positions determined by lhs into the rhs pattern."""
  # Get the (untyped) variable name and path from LHS and RHS.
  # Remember that variables have the form ?x0|NP for variables of type NP.
  lhs_vps = [(var.split('|')[0], path) for (var, path) in variables_to_paths(lhs)]
  rhs_vps = [(var.split('|')[0], path) for (var, path) in variables_to_paths(rhs)]

  # Check if all variables of the RHS are present in the LHS.
  lhs_vars_dict = { var : path for (var, path) in lhs_vps }
  for (rhs_var, _) in rhs_vps:
    if rhs_var not in lhs_vars_dict:
      if failok:
        return None
      else:
        raise ValueError(
        "LHS {0} missing expected variable {1}.".format(lhs, rhs_var))

  # Do replacement.
  out = deepcopy(rhs)
  for (rhs_var, rhs_path) in rhs_vps:
    lhs_path = lhs_vars_dict[rhs_var]
    if not lhs_path and not rhs_path:
      out = deepcopy(concrete)
    elif not lhs_path and rhs_path:
      out[rhs_path] = deepcopy(concrete)
    elif lhs_path and rhs_path:
      out[rhs_path] = deepcopy(concrete[lhs_path])
    elif lhs_path and not rhs_path:
      out = deepcopy(concrete[lhs_path])
  return out

# Funtions to convert from/into Tiburon format.
def escape(s):
  return re.sub(r'\\', r'\\\\', s)

def ConvertTokenToTiburon(leaf, quote=True):
  leaf_tiburon = leaf
  leaf_tiburon = leaf_tiburon.replace(r'%', 'PERCENT')
  # leaf_tiburon = leaf_tiburon.replace(r':', 'COLON')
  leaf_tiburon = leaf_tiburon.replace(r'#', 'NUMBERSIGN')
  leaf_tiburon = leaf_tiburon.replace(r'@', 'AT')
  if quote:
    leaf_tiburon = Quote(leaf_tiburon)
  return leaf_tiburon

def UnconvertTokenFromTiburon(leaf_tiburon):
  leaf = leaf_tiburon
  leaf = leaf.replace('PERCENT', '%')
  # leaf = leaf.replace('COLON', ':')
  leaf = leaf.replace('NUMBERSIGN', '#')
  leaf = leaf.replace('AT', '@')
  leaf = Unquote(leaf)
  return leaf

def ConvertPOSToTiburon(pos):
  pos_tiburon = pos.replace(r'.', 'PU')
  pos_tiburon = pos_tiburon.replace(r':', 'COLON')
  pos_tiburon = pos_tiburon.replace(r'#', 'NUMBERSIGN')
  return pos_tiburon

def UnconvertPOSFromTiburon(pos_tiburon):
  pos = pos_tiburon.replace('PU', '.')
  pos = pos.replace('COLON', ':')
  pos = pos.replace('NUMBERSIGN', r'#')
  return pos

def ConvertVarToTiburon(typed_var):
  assert '|' in typed_var, '| not found in {0}'.format(typed_var)
  assert typed_var.startswith('?x')
  var = typed_var[1:typed_var.index('|')]
  pos = typed_var[typed_var.index('|')+1:]
  pos_tiburon = ConvertPOSToTiburon(pos)
  typed_var_tiburon = var + ':' + pos_tiburon
  return typed_var_tiburon

def UnconvertVarFromTiburon(typed_var):
  assert ':' in typed_var, ': not found in {0}'.format(typed_var)
  var, pos = typed_var.split(':')
  pos = UnconvertPOSFromTiburon(pos)
  typed_var = '?' + var + '|' + pos
  return typed_var

# Convert tree_strings in Tiburon format 'NP(DT(a) NN(house))'
# into Stanford format '(NP (DT a) (NN house))'
def TiburonToStanford(tree_string):
  nts_moved_str = re.sub(r'(\S+?)\(', r'(\1 ', tree_string)
  return nts_moved_str

def IsQuotedToken(token):
  return token.startswith('"') and token.endswith('"')

def IsTiburonVariable(var_str):
  """
  Check whether a string is a tiburon variable of the form var: or var:POS
  """
  if not IsQuotedToken(var_str) and re.search(r'\S+:', var_str):
    return True
  return False

# TODO: some tokens may contain dots, but they are not applications
# of states to variables. Thus, we should have an index of what are
# the variables that already appeared in the LHS, so that we can
# better recognize what are state.variable leaves in RHS.
# Currently this is solved by recognizing whether the leaf is quoted,
# in which case, it is not a variable or a state.variable. However,
# this method will fail if we allow unquoted tokens that are leaves
# in Tiburon rules.
def IsTiburonStateVariable(var_str):
  """
  Check whether a string is a state application on a tiburon variable,
  of the form state_name.var_name.
  """
  if not IsQuotedToken(var_str) and '.' in var_str:
    return True
  return False

def BuildTiburonLHS(tree, quote_tokens=True):
  """
  1. Quote terminals,
  2. Rename variables ?x0|NP -> x0:NP
  3. Change bracketing (NP (DT the) (NN house)) -> NP(DT(the) NN(house))
  """
  lhs_str = ''
  if IsString(tree):
    if IsVariable(tree):
      lhs_str = ConvertVarToTiburon(tree)
    else:
      lhs_str = ConvertTokenToTiburon(tree, quote=quote_tokens)
  else:
    pos = get_top(tree)
    lhs_str = ConvertPOSToTiburon(pos) + '('
    lhs_str += ' '.join(
      [BuildTiburonLHS(child, quote_tokens=quote_tokens) for child in tree])
    lhs_str += ')'
  return lhs_str

def BuildTiburonRHS(tree, newstates, path = (), quote_tokens=True):
  """
  1. Quote terminals,
  2. Rename variables ?x0|NP would change into x0:NP
  3. Remove types of variables. x0:NP would change into x0.
  3. Change bracketing (NP (DT the) (NN house)) would
     change into NP(DT(the) NN(house))
  4. Apply states to variables. (NP (DT ?x0|) ?x1|NN) and
     {(0,0): 'q1', (1,) : 'q2'} would change into
     NP(DT(q1.x0) q2.x1)
  """
  rhs_str = ''
  if IsString(tree):
    if IsVariable(tree):
      assert tree.startswith('?')
      assert '|' in tree
      assert path in newstates, 'path {0} not in {1}'.format(path, newstates)
      rhs_str = newstates[path] + '.' + tree[1:tree.index('|')]
    else:
      rhs_str = ConvertTokenToTiburon(tree, quote=quote_tokens)
  else:
    pos = get_top(tree)
    rhs_str = ConvertPOSToTiburon(pos) + '('
    rhs_str += ' '.join(
      [BuildTiburonRHS(child, newstates, path + (i,), quote_tokens=quote_tokens) \
         for i, child in enumerate(tree)])
    rhs_str += ')'
  return rhs_str

## Functions to parse Tiburon rules into a dictionary.

def GetStateNameFromLHS(lhs):
  dot_position = lhs.index('.')
  return lhs[0:dot_position]

# Get weight from tiburon rule, if present.
def GetWeight(rule):
  weight = '1.0' # weight by default.
  if '#' in rule:
    weight_tie_str = rule.split(' # ')[-1].strip()
    if '@' in weight_tie_str:
      weight = weight_tie_str.split(' @ ')[0].strip()
    else:
      weight = weight_tie_str
  return float(weight)

def GetTreePatternFromTiburon(tree_str, LeafConversor):
  """
  Returns a string representation of the LHS or RHS tree pattern,
  depending on what function is passed to LeafConversor.
  Such LeafConversor is specific to LHS or RHS of rules.
  """
  if '(' not in tree_str or ')' not in tree_str:
    # We are processing a leaf.
    tree = LeafConversor(tree_str)
  else:
    # We are processing a tree pattern in tiburon format.
    # 1. Change style of parenthesis from tiburon to LISP format.
    tree_str_nltk = TiburonToStanford(tree_str)
    # 2. NLTK parse tree.
    tree_nltk = NLTKTree.fromstring(tree_str_nltk)
    # 3. Visit leaves and change them according to:
    #    3.1 If they are state.var, change state.x0 into ?x0|
    #    3.2 If they are not variables, unconvert leaf from tiburon.
    leaf_paths = tree_nltk.treepositions('leaves')
    for leaf_path in leaf_paths:
      tree_nltk[leaf_path] = LeafConversor(tree_nltk[leaf_path])
    # 4. Visit NTs and unconvert POS tags from Tiburon format.
    tree_nltk = UnconvertAllPOSFromTiburon(tree_nltk)
    tree = tree_nltk.pprint(margin=10000)
  return tree

def UnconvertLHSLeafFromTiburon(lhs_leaf_tiburon):
  """
  A leaf can be a variable or not. Different conversions
  are done depending on whether it is a LHS variable or not.
  """
  lhs_leaf = None
  if IsTiburonVariable(lhs_leaf_tiburon):
    lhs_leaf = UnconvertVarFromTiburon(lhs_leaf_tiburon)
  else:
    lhs_leaf = UnconvertTokenFromTiburon(lhs_leaf_tiburon)
  return lhs_leaf

def UnconvertRHSLeafFromTiburon(rhs_leaf_tiburon):
  """
  A leaf can be a state.variable or not. Different conversions
  are done depending on whether it is a RHS variable or not.
  """
  rhs_leaf = None
  if IsTiburonStateVariable(rhs_leaf_tiburon):
    rhs_leaf = UnconvertStateVarFromTiburon(rhs_leaf_tiburon)
  else:
    rhs_leaf = UnconvertTokenFromTiburon(rhs_leaf_tiburon)
  return rhs_leaf

def UnconvertStateVarFromTiburon(state_var):
  """
  Receives as input a state-applied variable of the form state.var.
  It returns the name of the variable.
  """
  assert '.' in state_var, '. not found in {0}'.format(state_var)
  var = '.'.join(state_var.split('.')[1:])
  pos = '' # In Tiburon, variables on RHS are not typed.
  typed_var = '?' + var + '|' + pos
  return typed_var

def UnconvertAllPOSFromTiburon(tree):
  leaf_paths = tree.treepositions('leaves')
  nt_paths = set(tree.treepositions()) - set(leaf_paths)
  for nt_path in nt_paths:
    assert not IsString(tree[nt_path])
    tiburon_pos = get_top(tree[nt_path])
    tree[nt_path].set_label(UnconvertPOSFromTiburon(tiburon_pos))
  return tree

def GetNewstatesFromRHSInTiburon(rhs_str):
  """
  Given a string representation of a RHS in Tiburon format,
  it returns a dictionary: path -> varname, where varname is converted to our
  software's variable name convention (.e.g ?x0|).
  """
  rhs_str_nltk = TiburonToStanford(rhs_str)
  rhs_nltk = tree_or_string(rhs_str_nltk)
  newstates = {}
  if IsString(rhs_nltk) and IsTiburonStateVariable(rhs_nltk):
    state = rhs_nltk[:rhs_nltk.index('.')]
    newstates[()] = state
  elif not IsString(rhs_nltk):
    for path in rhs_nltk.treepositions('leaves'):
      if IsTiburonStateVariable(rhs_nltk[path]):
        state_var = rhs_nltk[path]
        state = state_var[:state_var.index('.')]
        newstates[path] = state
  return newstates

def ParseTiburonRule(rule_str):
  if ' -> ' not in rule_str or rule_str.startswith('%'):
    return None
  # Obtain weight, or set 1.0 to default if not present.
  rule = {}
  rule['weight'] = GetWeight(rule_str)
  # Split LHS and RHS.
  [state_lhs_str, rhs_str_weight_tie] = rule_str.split(' -> ')
  # Get state name.
  state = GetStateNameFromLHS(state_lhs_str)
  rule['state'] = state
  lhs_str = state_lhs_str[state_lhs_str.index('.')+1:]
  if ' @ ' in rhs_str_weight_tie:
    rhs_str_weight, rhs_tie = [x.strip() for x in rhs_str_weight_tie.split(' @ ')]
  else:
    rhs_str_weight, rhs_tie = rhs_str_weight_tie, None
  if ' # ' in rhs_str_weight:
    rhs_str = rhs_str_weight.split(' # ')[0].strip()
  else:
    rhs_str = rhs_str_weight.strip()
  # Get string representation tree pattern of LHS and RHS converted.
  rule['lhs'] = GetTreePatternFromTiburon(lhs_str, UnconvertLHSLeafFromTiburon)
  rule['rhs'] = GetTreePatternFromTiburon(rhs_str, UnconvertRHSLeafFromTiburon)
  rule['newstates'] = GetNewstatesFromRHSInTiburon(rhs_str)
  if rhs_tie is not None:
    rule['tied_to'] = int(rhs_tie)
  return rule

