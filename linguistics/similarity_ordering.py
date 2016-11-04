from scipy.stats import kendalltau

from libs.functools27 import lru_cache
from nltk import Tree as NLTKTree

from linguistics.similarity import SimilarityScorer, Similarity
from utils.tree_tools import (variables_to_paths,
                              TreePattern, IsString)

def get_int_maps(items, item2int):
  """
  @item2int: dictionary with mappings from item to integer.
  Returns a list of integers that correspond to the items.
  If an item does not have any integer associated, it creates
  a new entry.
  """
  maps = []
  for item in items:
    item_id = item2int.get(item, None)
    if item_id is None:
      item_id = len(item2int) + 1
      item2int[item] = item_id
    maps.append(item_id)
  return maps

class OrderDifference(SimilarityScorer):
  """
  Compute the order differences (as given by Kendall's tau) of the branches
  with variables. E.g:
  (NP ?x0|JJ ?x1|NN) -- (NP ?x0|JJ ?x1|NN) would not be penalized.
  (NP ?x0|JJ ?x1|NN) -- (NP ?x1|NN ?x0|JJ) would be penalized (weight = -1).
  """
  def __init__(self):
    self.feature_weight = 0.1
    self.cached_tree_to_vars = {}

  @lru_cache(maxsize = None)
  def KendallTauCached(self, tree1_vars, tree2_vars):
    if len(tree1_vars) < 2 or len(tree1_vars) != len(tree2_vars):
      return None
    var2int = {}
    tree1_orders = get_int_maps(tree1_vars, var2int)
    tree2_orders = get_int_maps(tree2_vars, var2int)
    return kendalltau(tree1_orders, tree2_orders)[0]

  def MakeVariablesFromTreePattern(self, tree_pattern):
    if tree_pattern not in self.cached_tree_to_vars:
      path_to_var = \
        {path : '?x' + str(i) for i, path in enumerate(tree_pattern.subpaths)}
      subpaths_sorted = sorted(tree_pattern.subpaths)
      tree_vars = [path_to_var[path] for path in subpaths_sorted]
      self.cached_tree_to_vars[tree_pattern] = tree_vars
    return self.cached_tree_to_vars[tree_pattern]

  def GetVariables(self, tree):
    if isinstance(tree, TreePattern):
      tree_vars = self.MakeVariablesFromTreePattern(tree)
    elif isinstance(tree, NLTKTree):
      tree_vars = [var.split('|')[0] for (var, path) in variables_to_paths(tree)]
    elif IsString(tree) and tree.startswith('?x'):
      tree_vars = [tree]
    elif IsString(tree) and not tree.startswith('?x'):
      tree_vars = []
    else:
      tree_vars = None
    return tree_vars

  def GetSimilarity(self, tree1, tree2):
    # Retrieve the variables of each tree.
    tree1_vars = self.GetVariables(tree1)
    tree2_vars = self.GetVariables(tree2)
    assert len(tree1_vars) == len(tree2_vars), \
      'Number of variables differ {0} vs. {1} in trees {2} vs. {3}'.format(
      tree1_vars, tree2_vars, tree1, tree2)
    if not tree1_vars or len(tree1_vars) < 2:
      return [Similarity(0.0, 'order_difference', tree1, tree2)]
    kendall_tau = self.KendallTauCached(tuple(tree1_vars), tuple(tree2_vars))
    if kendall_tau is None:
      return []
    # kendall_tau is in the range [-1.0, 1.0]. We need to normalize it to be
    # within [0.0, 1.0] and complement it so that high scores denote similar
    # word order, and low scores denote reversed word order.
    weight = (1.0 - kendall_tau) / 2
    assert weight >= 0, 'Kendall tau value is not positive: {0}'.format(weight)
    return [Similarity(weight, 'order_difference', tree1, tree2)]
 
  def GetSimilar(self, tree):
    return []

