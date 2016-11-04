from collections import defaultdict
from copy import deepcopy
import itertools
from nltk import Tree as NLTKTree
import time

from utils.tree_tools import (tree_index, get_top, tree_or_string)

## Using cython:
from utils.cutils import AreDisjointPaths
## Not using cython:
# from extraction.extractor import AreDisjointPaths

class Transformation:
  def __init__(self, src_path, trg_path, src_subpaths, trg_subpaths, similarity):
    self.src_path = src_path
    self.trg_path = trg_path
    self.src_subpaths = tuple(src_subpaths)
    self.trg_subpaths = tuple(trg_subpaths)
    self.similarity = similarity
    self.rule = None
    self.rule_extractors = []
    self.rule_extractor_parent = None

  def __repr__(self):
    return (('<trans.\n  src_path: {0}\n  trg_path: {1}\n  subpaths1: {2}\n  subpaths2: {3}>'\
             .format(self.src_path, self.trg_path, self.src_subpaths, self.trg_subpaths)))

  def __eq__(self, other):
    return isinstance(other, self.__class__) \
       and self.src_path == other.src_path \
       and self.trg_path == other.trg_path \
       and self.src_subpaths == other.src_subpaths \
       and self.trg_subpaths == other.trg_subpaths \
       and self.similarity.relation == other.similarity.relation

  def __hash__(self):
    return hash(
      (self.src_path,
       self.trg_path,
       self.src_subpaths,
       self.trg_subpaths,
       self.similarity.relation))

def FixNewstates(derivation):
  # Collect a dictionary with certain (src_path, trg_path) : state names.
  path_states = { (rule.src_path, rule.trg_path) : rule.state for rule in derivation }
  # Add the state names in the newstates dictionary of each rule.
  derivation_copy = deepcopy(derivation)
  for rule in derivation_copy:
    for subpath1, subpath2 in zip(rule.src_subpaths, rule.trg_subpaths):
      state = path_states[(subpath1, subpath2)]
      rhs_relative_subpath = subpath2[len(rule.trg_path):]
      rule.newstates[rhs_relative_subpath] = state
  return derivation_copy

class RuleExtractor:
  """
  Given two (NLTK) trees, it finds the set of most likely chain(s) of rules
  that transform the source tree into the target tree.
  """
  def __init__(self, tree1, tree2, path1, path2, options):
    pass

  def SetOptions(self, options):
    self.similarity_scorer = options.get('similarity_scorer', None)
    self.similarity_score_guesser = options.get('similarity_score_guesser', None)
    self.initial_state = options.get('initial_state', 'q0')
    self.kMaxSourceBranches = options.get('max_source_branches', 3)
    self.kMaxSourceDepth = options.get('max_source_depth', 4)
    self.kNBestPerLevel = options.get('nbest_per_level', 30)
    self.max_running_time = options.get('max_running_time', 3000)
    self.kBeamSize = options.get('beam_size', 5)
    self.cached_extractors = options.get('cached_extractors', {})
    self.optimal_scores = options.get('optimal_scores', defaultdict(list))
    self.options['optimal_scores'] = self.optimal_scores
    self.avoid_src_empty_transitions = options.get('src_empty', False)
    self.enable_deletions = options.get('deletions', False)
    self.feat_inst = options.get('feat_inst', None)

  def IsTimeOut(self):
    if time.time() > self.time_start + self.max_running_time:
      return True
    else:
      return False

  def GetRemainingRunningTime(self):
    return self.max_running_time - (time.time() - self.time_start)

  def ObtainBestDerivations(self, n_best = 1):
    pass

def ObtainTreePattern(tree, path, subpaths):
  subtree = tree_index(tree, path)
  if not subpaths:
    return deepcopy(subtree)
  if not isinstance(subtree, NLTKTree) and (subpaths[0] == () or path == subpaths[0]):
    return '?x0|'
  if isinstance(subtree, NLTKTree) and (subpaths[0] == () or path == subpaths[0]):
    return '?x0|' + get_top(subtree)
  if not isinstance(subtree, NLTKTree) and subpaths[0] != ():
    raise(ValueError, \
          'String {0} cannot be indexed by {1}'.format(subtree, subpaths))
  depth_subtree = len(path)
  tree_pattern = deepcopy(subtree)
  for i, subpath in enumerate(subpaths):
    subpath_relative = subpath[depth_subtree:]
    branch = tree_index(tree, subpath)
    if not isinstance(branch, NLTKTree):
      tree_pattern[subpath_relative] = '?x' + str(i) + '|'
    else:
      tree_pattern[subpath_relative] = '?x' + str(i) + '|' + get_top(branch)
  return tree_pattern

def GetDisjointPaths(tree, path, min_paths, max_paths, max_depth = 4,
                     candidate_paths = None):
  """
  This function returns a list of tuples of disjoint paths. E.g.
  [((path1, path2)), ((path1, path3)), ...]. A list such as
  [()] represents a single tuple of disjoint paths, which contains
  no paths. That element "()" is important to signal tree patterns
  to have no subpaths, converting them into simple subtrees with no variables.
  candidate_paths is a variable containing a list of tuples, where
  each tuple is a valid path in the tree. If the candidate_paths
  variable is set, the disjoint paths will be generated from the
  list of candidate paths. Otherwise, the disjoint paths will be
  generated from the subpaths of the parameter "path".
  """
  subtree = tree_index(tree, path)
  # Case where the subtree is a string.
  # The disjoint paths [()] means that there is only one set of disjoint paths,
  # which is (), that is the empty set of disjoint paths.
  if not isinstance(subtree, NLTKTree):
    disjoint_paths = []
    # Add the tuple containing zero number of paths.
    if min_paths == 0:
      disjoint_paths.append(())
    if max_paths > 0:
      disjoint_paths.append((path,))
    return disjoint_paths
  # Case where the subtree is a tree.
  if candidate_paths == None:
    paths = [path + subpath for subpath in subtree.treepositions() \
                              if len(subpath) < max_depth \
                                 and len(subpath) > 0]
  else:
    paths = candidate_paths
  # Return a generator to save memory for large combinations, at the
  # expense of some speed.
  # Update: return a list instead, for caching and faster computation.
  return list(itertools.chain(
    combined_paths \
    for k in range(min_paths, max_paths + 1) \
      for combined_paths in itertools.combinations(paths, k) \
        if AreDisjointPaths(combined_paths)))

def GetDisjointPathsWithPermutations(
    tree, path, min_paths, max_paths, max_depth = 4):
  subtree = tree_index(tree, path)
  # Case where the subtree is a string.
  if not isinstance(subtree, NLTKTree):
    disjoint_paths = []
    if min_paths == 0:
      disjoint_paths.append( () )
    if min_paths < 2 and max_paths > 0:
      disjoint_paths.append( ((),) )
    return disjoint_paths
  # Case where the subtree is a tree (return generator).
  paths = [path + subpath for subpath in subtree.treepositions() \
                            if len(subpath) < max_depth]
  return itertools.chain(
    permutted_paths \
    for k in range(min_paths, max_paths + 1) \
      for combined_paths in itertools.combinations(paths, k) \
        if AreDisjointPaths(combined_paths)
          for permutted_paths in itertools.permutations(combined_paths))

"""
def AreDisjointPaths(paths):
  num_paths = len(paths)
  for i in range(num_paths):
    path1_length = len(paths[i])
    for j in range(i + 1, num_paths):
      path2_length = len(paths[j])
      min_length = min(path1_length, path2_length)
      if paths[i][0:min_length] == paths[j][0:min_length]:
        return False
  return True
"""
