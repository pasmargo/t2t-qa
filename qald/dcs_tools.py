from copy import deepcopy
from itertools import chain
import logging
import json

from nltk import Tree

from utils.tree_tools import IsString, get_top, tree_or_string

def flatten(l):
    '''flattening depth-1 lists'''
    return list(chain.from_iterable(l))

def replaceVariable(func, arg):
    '''replace (var x) in func with arg'''
    def replaceVariableSub(f, arg):
        if f.label() == 'var':
            assert(f[0] == 'x')
            return arg
        else:
            return Tree(f.label(), map(lambda f1: replaceVariableSub(f1, arg), f))

    assert(len(func) == 2)
    assert(func.label() == 'lambda')
    assert(func[0] == 'x')
    return replaceVariableSub(func[1], arg)
    
def dcs2constituent_(dcs):
    '''convert DCS tree into constituent structure'''
    if not isinstance(dcs, Tree):
        return [dcs]
    if dcs.label() == 'and':
        return flatten(map(dcs2constituent, dcs))
    elif dcs.label() == 'count':
        assert(len(dcs) == 1)
        return [Tree("COUNT", dcs2constituent(dcs[0]))]
    elif dcs.label() == 'date':
        assert(len(dcs) == 3)
        return [Tree("DATE", ['_'.join(dcs)])]
    elif dcs.label() == 'number':
        assert(len(dcs) == 2)
        return [Tree("NUMBER", flatten(map(dcs2constituent, dcs)))]
    elif dcs.label() == '':
        # must be lambda expression
        assert(len(dcs) == 2)
        new_dcs = replaceVariable(dcs[0], dcs[1])
        return dcs2constituent(new_dcs)
    else:
        assert(len(dcs) == 1), '%s' % dcs
        dtrs = dcs2constituent(dcs[0])
        return [Tree("ID", [dcs.label()] + dtrs)]

def dcs2constituent(dcs):
    '''convert DCS tree into constituent structure'''
    # Check for a malformed tree.
    if isinstance(dcs, Tree) and len(dcs) == 0:
        return []
    if not isinstance(dcs, Tree):
        return [dcs]
    if dcs.label() == 'and':
        return flatten(map(dcs2constituent, dcs))
    elif dcs.label() == 'count':
        # assert len(dcs) == 1
        if len(dcs) != 1:
          logging.warning('Invalid {0} tree: {1}'.format(dcs.label(), dcs))
          return []
        return [Tree("ID", ["COUNT"] + dcs2constituent(dcs[0]))]
    elif dcs.label() == 'date':
        # assert len(dcs) == 3, 'Unexpected dcs for date: {0}'.format(dcs)
        if len(dcs) != 3:
          logging.warning('Invalid {0} tree: {1}'.format(dcs.label(), dcs))
          return []
        return [Tree("DATE", ['_'.join(dcs)])]
    elif dcs.label() == 'number':
        # assert len(dcs) == 2
        if len(dcs) != 2:
          logging.warning('Invalid {0} tree: {1}'.format(dcs.label(), dcs))
          return []
        return [Tree("NUMBER", flatten(map(dcs2constituent, dcs)))]
    elif dcs.label() == '':
        # must be lambda expression application
        # assert len(dcs) == 2
        if len(dcs) != 2:
          logging.warning('Invalid l-application tree: {1}'.format(dcs.label(), dcs))
          return []
        new_dcs = replaceVariable(dcs[0], dcs[1])
        return dcs2constituent(new_dcs)
    elif dcs.label() == 'var':
        # This is a variable of a lambda expression that has not been
        # substituted by any argument. I don't know how to deal with
        # these cases. For the time being, I will just remove it.
        # assert len(dcs) == 1
        # assert get_top(dcs[0]) == 'x'
        if len(dcs) != 1 or get_top(dcs[0]) != 'x':
          logging.warning('Invalid {0} tree: {1}'.format(dcs.label(), dcs))
          return []
        return []
    elif dcs.label() == 'lambda':
        # This is a lambda expression that could not be resolved.
        # Since I don't know how to deal with it either, I will remove it.
        # assert len(dcs) == 2
        # assert get_top(dcs[0]) == 'x'
        if len(dcs) != 2 or get_top(dcs[0]) != 'x':
          logging.warning('Invalid {0} tree: {1}'.format(dcs.label(), dcs))
          return []
        return dcs2constituent(dcs[1])
    else:
        # assert len(dcs) == 1, '%s' % dcs
        if len(dcs) != 1:
          logging.warning('Invalid {0} tree: {1}'.format(dcs.label(), dcs))
          return []
        dtrs = dcs2constituent(dcs[0])
        return [Tree("ID", [dcs.label()] + dtrs)]

def ConvertDCS2Constituent(dcs):
  """
  Wrapper for dcs2constituent, where we try to convert an eventual
  tree string into a tree.
  This function also retrieves the first item of the resulting list,
  which contains the final constituent structure, and transforms it
  into a utils.tree_tools.Tree object.
  """
  dcs_tree = tree_or_string(dcs)
  constituents = dcs2constituent(dcs_tree)
  # assert len(constituents) == 1, 'C: %s\nD: %s' % (constituents, dcs_tree)
  constituent_tree = tree_or_string(str(constituents[0]))
  return constituent_tree

def ConvertConstituent2DCS(constituent_tree):
  """
  Wrapper for constituent2dcs, where we try to convert an eventual
  tree string into a tree.
  This function also retrieves the first item of the resulting list,
  which contains the final constituent structure, and transforms it
  into a utils.tree_tools.Tree object.
  """
  if IsString(constituent_tree):
    constituent_tree = tree_or_string(constituent_tree)
  dcs_tree_fragments = constituent2dcs(constituent_tree)
  assert isinstance(dcs_tree_fragments, list) and len(dcs_tree_fragments) == 1
  dcs_tree = tree_or_string(str(dcs_tree_fragments[0]))
  return dcs_tree

def constituent2dcs_(tree):
  '''convert a constituent structure into a DCS tree'''
  if not isinstance(tree, Tree):
    return [tree]
  elif get_top(tree) == 'COUNT':
    assert len(tree) == 1
    return [Tree('count', constituent2dcs(tree[0]))]
  elif get_top(tree) == 'NUMBER':
    assert len(tree) == 2
    return [Tree('number', tree[:])]
  elif get_top(tree) == 'DATE':
    # tree contains a list with only one element, which is the data
    # joined with underscores. We re-establish the list.
    date_info = tree[0].split('_')
    return [Tree('date', date_info)]
  if get_top(tree) == 'ID' and len(tree) == 2:
    # The first child is the predicate. The rest are the arguments.
    assert len(tree) == 2, '%s' % tree
    return [Tree(get_top(tree[0]), flatten(map(constituent2dcs, tree[1:])))]
  if len(tree) > 2:
    # A length greater than 2 is the only signal we have for "and".
    return [Tree(get_top(tree[0]), flatten(map(constituent2dcs, tree[1:])))]
  return [tree]

def constituent2dcs(tree):
  '''convert a constituent structure into a DCS tree'''
  if not isinstance(tree, Tree):
    return [tree]
  # elif get_top(tree) == 'COUNT':
  #   assert len(tree) == 1
  #   return [Tree('count', constituent2dcs(tree[0]))]
  elif get_top(tree) == 'NUMBER':
    assert len(tree) == 2
    return [Tree('number', tree[:])]
  elif get_top(tree) == 'DATE':
    # tree contains a list with only one element, which is the data
    # joined with underscores. We re-establish the list.
    if IsString(tree[0]):
      date_info = tree[0].split('_')
      try:
        map(int, date_info)
      except ValueError:
        date_info = [tree[0]]
      return [Tree('date', date_info)]
    else:
      return [Tree(get_top(tree[0]), flatten(map(constituent2dcs, tree[1:])))]
  if get_top(tree) == 'ID' and len(tree) == 2:
    # The first child is the predicate. The rest are the arguments.
    assert len(tree) == 2, '%s' % tree
    predicate = get_top(tree[0])
    if predicate == 'COUNT':
      predicate = predicate.lower()
    return [Tree(predicate, flatten(map(constituent2dcs, tree[1:])))]
  if len(tree) > 2:
    # A length greater than 2 is the only signal we have for "and".
    return [Tree(get_top(tree[0]), [Tree('and', flatten(map(constituent2dcs, tree[1:])))])]
  return [tree]

