from collections import defaultdict
from copy import deepcopy
import itertools
import logging
import numpy as np
import time

from extraction.extractor import ObtainTreePattern
from extraction.extractor import Transformation as GenericTransformation
from extraction.extractor import RuleExtractor as GenericRuleExtractor
from linguistics.similarity import Similarity
from training.transductionrule import XTRule
from training.wrtg import RHS, wRTG, Production
from utils.tree_tools import (IsString, TreePattern, GetLeaves,
  tstartswith, GetChildrenPaths)
from utils.priority_queue import PriorityQueue

## Using cython:
from utils.cutils import AreDisjointPaths, GetCommonParentsAt
## Not using cython:
# from extraction.extractor import AreDisjointPaths, GetCommonParentsAt

def FixNewstatesRefined(derivation, default_state=None):
  """
  This function does the same as Fixnewstates, but it avoids assigning
  default-name states.
  """
  # Collect a dictionary with certain (src_path, trg_path) : state names.
  # These paths are absolute to the tree where this rule was extracted from.
  path_states = { (rule.src_path, rule.trg_path) : rule.state for rule in derivation }
  # Add the state names in the newstates dictionary of each rule.
  derivation_copy = deepcopy(derivation)
  for rule in reversed(derivation_copy):
    if rule.state == default_state and \
       (len(rule.newstates) == 1 or len(rule.src_subpaths) == 1):
      child_state = path_states[(rule.src_subpaths[0], rule.trg_subpaths[0])]
      if child_state != default_state:
        rule.state = child_state + '_pending' \
          if not child_state.endswith('_pending') else child_state
        path_states[(rule.src_path, rule.trg_path)] = rule.state
    for subpath1, subpath2 in zip(rule.src_subpaths, rule.trg_subpaths):
      state = path_states[(subpath1, subpath2)]
      rhs_relative_subpath = subpath2[len(rule.trg_path):]
      rule.newstates[rhs_relative_subpath] = state
  return derivation_copy

def MakeStateName(relation, src_path, src_subpaths, state_id=None):
  if not relation and not src_subpaths:
    state = 't'
  elif not relation:
    if state_id == None:
      state_id = str(len(src_path)) if len(src_path) > 0 else '0'
    state = 'q' + str(state_id)
  else:
    state = relation
  return state

class Transformation(GenericTransformation):
  def BuildTransformationRule(self, tree1, tree2, state_id=None):
    """ Builds an XTRule """
    assert len(self.src_subpaths) == len(self.trg_subpaths), \
      'Length of subtree paths differ: {0} vs. {1}'.format(
      self.src_subpaths, self.trg_subpaths)
    lhs = ObtainTreePattern(tree1, self.src_path, self.src_subpaths)
    rhs = ObtainTreePattern(tree2, self.trg_path, self.trg_subpaths)
    state = MakeStateName(self.similarity.relation, self.src_path, self.src_subpaths)
    weight = self.similarity.score
    newstates = {}
    rule = XTRule(state, lhs, rhs, newstates, weight)
    rule.src_path = self.src_path
    rule.trg_path = self.trg_path
    rule.src_subpaths = self.src_subpaths
    rule.trg_subpaths = self.trg_subpaths
    return rule

def GetLeavesIndices(tree, path, subpaths):
  if IsString(tree):
    if list(subpaths) == []:
      leaves_indices = [0]
    else:
      leaves_indices = []
  else:
    path_leaves_indices = tree.path_to_leaves_indices[path]
    subpaths_leaves_indices = \
      list(itertools.chain(
        *[tuple(tree.path_to_leaves_indices[s]) for s in subpaths]))
    leaves_indices = list(set(path_leaves_indices) - set(subpaths_leaves_indices))
  return leaves_indices

def GetAlignmentFromRule(rule, src_tree, trg_tree, cost_threshold = 0.3):
  assert len(rule.src_subpaths) == len(rule.trg_subpaths), \
    'Number of rule source and target subpaths differs: {0} vs {1}' \
    .format(rule.src_subpaths, rule.trg_subpaths)
  src_subpaths = rule.src_subpaths
  trg_subpaths = rule.trg_subpaths
  src_leaves_indices = GetLeavesIndices(src_tree, rule.src_path, src_subpaths)
  trg_leaves_indices = GetLeavesIndices(trg_tree, rule.trg_path, trg_subpaths)
  num_src_and_trg_leaves = len(src_leaves_indices) + len(trg_leaves_indices)
  if num_src_and_trg_leaves == 0:
    alignment = []
  elif (rule.weight / float(num_src_and_trg_leaves)) > cost_threshold:
    alignment = []
  else:
    alignment = list(itertools.product(src_leaves_indices, trg_leaves_indices))
  return alignment

def GetAlignmentFromDerivation(derivation, src_tree, trg_tree):
  alignments = [] # list of tuples (src_leaf_index, trg_leaf_index)
  for rule in derivation:
    alignment = GetAlignmentFromRule(rule, src_tree, trg_tree)
    alignments.extend(alignment)
  alignments_unique = sorted(list(set(alignments)))
  alignments_str = ' '.join([str(src_index) + '-' + str(trg_index) \
                               for src_index, trg_index in alignments_unique])
  return alignments_str

def InstantiateFeatures(derivation, feat_inst):
  """
  Given a derivation (list of rules) and a feature instantiator,
  it extracts features of rules and populates the attribute "features"
  of each rule.
  """
  if feat_inst is None:
    return
  for rule in derivation:
    src_treep, trg_treep = rule.GetTreePatterns()
    rule.features = feat_inst.InstantiateFeatures(src_treep, trg_treep, rule)

class RuleExtractor(GenericRuleExtractor):
  """
  Given two trees, it finds the set of most likely chain(s) of rules
  that transform the source tree into the target tree.
  The trees that this class expects are defined in utils/tree_tools.py
  Optionally, a cost function guesser (similarity_score_guesser) can be
  provided (in the options parameter), to speed up the search.
  Such guesser should underestimate the cost of transforming a source
  subtree. Ideally, such underestimation will consist of the optimal
  cost of transformation, and could be computed if no syntactic constraints
  were imposed. If the underestimation falls below the optimal cost,
  then it is ignored.
  """
  def __init__(self, tree1, tree2, path1, path2, options):
    self.tree1 = tree1
    self.tree2 = tree2
    # Path to the subtrees for which we want to find a transformation chain.
    self.path1 = path1
    self.path2 = path2
    # Setting options.
    self.options = options
    self.SetOptions(options)
    # PreCache costs given the source and target trees.
    if path1 == () and path2 == ():
      self.similarity_scorer.PreCache(tree1, tree2, options)
      self.similarity_score_guesser.PreCache(tree1, tree2, options)
    # Setting time limitations.
    self.time_start = time.time()
    self.cached_tree_patterns = {}
    src_paths = self.tree1.treepositions() if not IsString(self.tree1) else [()]
    trg_paths = self.tree2.treepositions() if not IsString(self.tree2) else [()]
    # N-best lists of costs given source or target path.
    # src_path -> [(cost, transformation), ...]
    # trg_path -> [(cost, transformation), ...]
    self.costs = {p : PriorityQueue(self.kBeamSize) for p in src_paths}
    self.trg_costs = {p : PriorityQueue(self.kBeamSize) for p in trg_paths}
    self.best_transformations = \
      { (s, t) : PriorityQueue(self.kBeamSize) \
        for s, t in itertools.product(src_paths, trg_paths) }
    # Keep track of what source nodes and source-target pairs
    # have been explored.
    self.src_explored = set() # set of source paths.
    self.cached_cost_tree_patterns = {}
    self.span = None
    self.span_rec = None

  def ObtainBestDerivations(self, n_best = 1, state_id = None):
    """
    Obtains a list of lists. Each list contains a sequence of rules
    that transform the source tree into the target tree.
    """
    self.status = self.ApproximateSearch(self.kBeamSize)
    if self.status != 'success':
      best_derivations = [self.SalvageBestDerivation()]
    else:
      # Make each transformation to point correctly to its child rule extractors.
      best_derivations = GetBestDerivations(self.best_transformations,
        self.tree1, self.tree2, self.initial_state, (), (), state_id,
        enable_deletions=self.enable_deletions)
    glue_rule = MakeGlueRule(self.tree1, self.tree2, self.initial_state)
    derivations = []
    for i, best_derivation in enumerate(best_derivations):
      if i >= n_best:
        break
      best_derivation = FixNewstatesRefined(best_derivation, default_state='q')
      InstantiateFeatures(best_derivation, self.feat_inst)
      derivations.append(best_derivation)
    return derivations

  def ComputeNaiveMappings(self):
    src_tree_patterns = self.GetNaiveTreePatterns(self.tree1)
    trg_tree_patterns = self.GetNaiveTreePatterns(self.tree2)
    for src_tree_pattern in src_tree_patterns:
      for trg_tree_pattern in trg_tree_patterns:
        beam_size = self.kBeamSize
        self.UpdateTransformationCaches(
          src_tree_pattern, trg_tree_pattern, beam_size)
    return

  def GetBestTrgPathsForSrcPath(self, src_path, trg_path, k_best = 5):
    """
    Return in sorted order the best mappings between src_path and
    any other path from target tree that includes trg_path or any
    of its children.
    """
    # For child_src_path, obtain all possible mappings to paths below trg_path.
    best_trg_paths = PriorityQueue(k_best)
    child_trg_paths = [trg_path] + GetChildrenPaths(self.tree2, trg_path, np.inf)
    for child_trg_path in child_trg_paths:
      child_trg_cost = \
        self.best_transformations[(src_path, child_trg_path)].GetBestScore()
      best_trg_paths.Push(child_trg_cost, child_trg_path)
    child_src_trg_paths = \
      [((src_path,), (child_trg_path,)) \
         for _, child_trg_path in \
           best_trg_paths.GetSortedScoresAndItems()]
    return child_src_trg_paths[:k_best]

  def GenerateDisjointPaths(self, src_path, trg_path, k_best = 2):
    """
    Obtain a list of 2-tuples, where each 2-tuple contains source
    and target disjoint paths that are children of src_path and trg_path.
    Both src_path and trg_path may also be included in the disjoint paths.
    """
    self.span_rec = defaultdict(lambda: PriorityQueue(k_best))
    child_src_paths = [src_path] + GetChildrenPaths(self.tree1, src_path, np.inf)
    child_src_paths = sorted(child_src_paths, key=lambda p: len(p), reverse=True)
    for child_src_path in child_src_paths:
      self.GenerateSubpaths(src_path, trg_path, child_src_path, k_best)
    disjoint_paths = \
      GetItemsForChildSrcPath(self.span_rec, src_path, trg_path, src_path)
    return disjoint_paths

  def GenerateSubpaths(self, src_path, trg_path, child_src_path, k_best = 2):
    child_disjoint_paths = \
      self.GetBestTrgPathsForSrcPath(child_src_path, trg_path, k_best)
    num_children = 0 if IsString(self.tree1[child_src_path]) \
                     else len(self.tree1[child_src_path])
    if num_children == 0:
      self.EvaluateAndStoreDisjointPaths(self.span_rec, child_disjoint_paths,
        src_path, trg_path, child_src_path)
      return child_disjoint_paths
    # This dictionary of priority queues stores intermediate disjoint paths
    # in the sequential search across immediate children of child_src_path.
    store = defaultdict(lambda: PriorityQueue(k_best))
    for i in range(num_children):
      # 1. Obtain best disjoint paths from current immediate child.
      imm_child_src_path = child_src_path + (i,)
      imm_child_disjoint = GetItemsForChildSrcPath(
        self.span_rec, src_path, trg_path, imm_child_src_path)
      # 2. Obtain best disjoint paths from up to previous child.
      if i > 0:
        prev_child_src_path = child_src_path + (i - 1,)
        prev_child_disjoint = GetItemsForChildSrcPath(
          store, src_path, trg_path, prev_child_src_path)
        # 3. Combine those disjoint paths, and filter out invalid disjoint paths.
        disjoint_paths = self.CombineAndFilterDisjointPaths(
            prev_child_disjoint, imm_child_disjoint, src_path, trg_path)
      else:
        disjoint_paths = imm_child_disjoint
      # 4. Evaluate the quality of these combined disjoint paths, and store them
      #    in a sensible way.
      self.EvaluateAndStoreDisjointPaths(store, disjoint_paths, src_path, trg_path,
        imm_child_src_path)
    # 5. Return best disjoint paths accumulated at the last immediate child.
    last_child_path = child_src_path + (num_children - 1,)
    accumulated_child_disjoint = GetItemsForChildSrcPath(
      store, src_path, trg_path, last_child_path)
    accumulated_child_disjoint.extend(child_disjoint_paths)
    self.EvaluateAndStoreDisjointPaths(self.span_rec, accumulated_child_disjoint,
      src_path, trg_path, child_src_path)
    return accumulated_child_disjoint

  def EvaluateAndStoreDisjointPaths(self, store, disjoint_paths, src_path,
      trg_path, child_src_path):
    for src_disjoint, trg_disjoint in disjoint_paths:
      assert AreDisjointPaths(src_disjoint), '{0}'.format(src_disjoint)
      assert AreDisjointPaths(trg_disjoint), '{0}'.format(trg_disjoint)
      if self.avoid_src_empty_transitions and \
         tuple([src_path]) == tuple(src_disjoint):
        continue
      src_tree_pattern = self.GetTreePatternCached(
        self.tree1, src_path, src_disjoint)
      trg_tree_pattern = self.GetTreePatternCached(
        self.tree2, trg_path, trg_disjoint)
      cost, _ = self.GetCostTreePatterns(src_tree_pattern, trg_tree_pattern)
      num_src_leaves_covered = len(src_tree_pattern.GetExcludedLeaves())
      trg_leaves_covered = len(trg_tree_pattern.GetExcludedLeaves())
      # trg_leaves_covered = frozenset(trg_tree_pattern.GetExcludedLeaves())
      # trg_leaves_covered = frozenset(trg_tree_pattern.GetExcludedLeavesIndices())
      store[(src_path, trg_path, child_src_path,
             num_src_leaves_covered, trg_leaves_covered)]\
        .Push(cost, (src_disjoint, trg_disjoint))

  def ApproximateSearch(self, beam_size = 5):
    """
    Perform bottom-up beam search of rule extraction. This is the heavy work.
    """
    # Initialize intermediate caches of intermediate results with naive rules.
    self.ComputeNaiveMappings()
    # Obtain list of source paths in bottom-up order. Every source path
    # is matched to all target paths, before proceeding to next source path.
    src_paths = GetSortedTreePaths(self.tree1)
    trg_paths = GetSortedTreePaths(self.tree2)
    # The product of these src_paths and trg_paths makes trg_paths to iterate
    # faster.
    for src_path, trg_path in itertools.product(src_paths, trg_paths):
      # if AreTreesUnbalanced(self.tree1, src_path, self.tree2, trg_path):
      #   continue
      # if not self.IsPathPairPromising(src_path, trg_path):
      #   continue
      disjoint_paths = self.GenerateDisjointPaths(src_path, trg_path, beam_size)
      for src_subpaths, trg_subpaths in disjoint_paths:
        # We do not allow empty transitions on both source and target
        # at the same time.
        if tuple([src_path]) == tuple(src_subpaths) and \
           tuple([trg_path]) == tuple(trg_subpaths):
          continue
        assert AreDisjointPaths(src_subpaths), \
          'source subpaths not disjoint: {0}'.format(src_subpaths)
        assert AreDisjointPaths(trg_subpaths), \
          'target subpaths not disjoint: {0}'.format(trg_subpaths)
        # Check if we ran out of time.
        if self.IsTimeOut():
          return 'timed_out'
        src_tree_pattern = self.GetTreePatternCached(
          self.tree1, src_path, src_subpaths)
        trg_tree_pattern = self.GetTreePatternCached(
          self.tree2, trg_path, trg_subpaths)
        cost = self.UpdateTransformationCaches(src_tree_pattern,
          trg_tree_pattern, self.kBeamSize)
    self.src_explored.add(src_path)
    return 'success'

  def IsPathPairPromising(self, src_path, trg_path,
                          threshold = 0.8, min_leaves = 5):
    """
    We are exploring every possible pair of source and target paths. However,
    such exploration may visit path pairs that are not likely to be mappings
    "at a first glance". This method aims to filter out these pairs with a
    simple heuristic. Note that in some cases, we may filter out legitimated
    path pairs. This heuristic proceeds as follows:
    1. Obtain src_path subpaths, and get their best trg_path mappings.
    2. If the set of trg_paths covers to a high degree (according to a threshold)
       the leaves spanned by trg_path, then the path pair is promising.
    """
    if IsString(self.tree1) or IsString(self.tree2):
      return True
    num_src_leaves = self.tree1.GetNumLeaves(src_path)
    num_trg_leaves = self.tree2.GetNumLeaves(trg_path)
    if num_src_leaves < min_leaves and num_trg_leaves < min_leaves:
      return True
    src_subpaths = [s for s in self.tree1.treepositions() \
                        if tstartswith(s, src_path)]
    trg_subpaths = [trans.path2 for s in src_subpaths \
                                 for trans in self.costs[s].GetItems()]
    trg_leaves_indices = set(self.tree2.path_to_leaves_indices[trg_path])
    trg_leaves_indices_covered = set(
      itertools.chain(*[self.tree2.path_to_leaves_indices[t] \
                          for t in trg_subpaths]))
    trg_leaves_indices_not_covered = \
      trg_leaves_indices - trg_leaves_indices_covered
    num_trg_leaves_indices_not_covered = float(len(trg_leaves_indices_not_covered))
    num_trg_leaves_indices = float(len(trg_leaves_indices))
    if num_trg_leaves_indices == 0.0:
      return True
    ratio_trg_covered = num_trg_leaves_indices_not_covered \
                      / num_trg_leaves_indices
    assert 0.0 <= ratio_trg_covered <= 1.0
    if ratio_trg_covered < threshold:
      return True
    logging.debug('IsPathPairPromising False for {0} vs. {1}'\
                  .format(src_path, trg_path))
    return False

  def UpdateTransformationCaches(self,
                                 src_tree_pattern,
                                 trg_tree_pattern,
                                 beam_size = 5):
    cost, similarity = self.GetCostTreePatterns(src_tree_pattern, trg_tree_pattern)
    src_path = src_tree_pattern.path
    trg_path = trg_tree_pattern.path
    src_subpaths = src_tree_pattern.subpaths
    trg_subpaths = trg_tree_pattern.subpaths
    transformation = BuildTransformation(src_path, trg_path, \
      src_subpaths, trg_subpaths, similarity)
    self.best_transformations[(src_path, trg_path)].Push(cost, transformation)
    self.costs[src_path].Push(cost, transformation, beam_size)
    self.trg_costs[trg_path].Push(cost, transformation, beam_size)
    return cost

  def GetCostTransformation(self, src_path, trg_path, src_subpaths, trg_subpaths):
    src_tree_pattern = self.GetTreePatternCached(self.tree1, src_path, src_subpaths)
    trg_tree_pattern = self.GetTreePatternCached(self.tree2, trg_path, trg_subpaths)
    cost, similarity = self.GetCostTreePatterns(src_tree_pattern, trg_tree_pattern)
    return cost, similarity

  def GetCostTreePatterns(self, src_tree_pattern, trg_tree_pattern):
    """
    In this version of the function, we do cache the tree pattern costs
    and similarities because they may occur several times during the
    incremental computation of disjoint paths.
    """
    if (id(src_tree_pattern), id(trg_tree_pattern)) \
        not in self.cached_cost_tree_patterns:
      cost = 0.0
      src_subpaths = src_tree_pattern.subpaths
      trg_subpaths = trg_tree_pattern.subpaths
      for src_subpath, trg_subpath in zip(src_subpaths, trg_subpaths):
        accumulated_cost = \
          self.best_transformations[(src_subpath, trg_subpath)].GetBestScore()
        cost += accumulated_cost
      similarities = \
        self.similarity_scorer.GetSimilarity(src_tree_pattern, trg_tree_pattern)
      best_similarity = similarities[0]
      cost += best_similarity.score
      self.cached_cost_tree_patterns[
        (id(src_tree_pattern), id(trg_tree_pattern))] = (cost, best_similarity)
    cost, best_similarity = self.cached_cost_tree_patterns[
      (id(src_tree_pattern), id(trg_tree_pattern))]
    return cost, best_similarity

  def GetTreePatternCached(self, tree, path, subpaths):
    if not (tree, path, tuple(subpaths)) in self.cached_tree_patterns:
      tree_pattern = TreePattern(tree, path, subpaths)
      self.cached_tree_patterns[(tree, path, tuple(subpaths))] = tree_pattern
    return self.cached_tree_patterns[(tree, path, tuple(subpaths))]

  def GetNaiveTreePatterns(self, tree):
    if IsString(tree):
      positions = [()]
    else:
      positions = tree.treepositions()
    subpaths = []
    tree_patterns = [self.GetTreePatternCached(tree, position, subpaths) \
                       for position in positions]
    return tree_patterns

  def SalvageBestDerivation(self):
    """
    This function is called to obtain a reasonable derivation
    in the case of a not successful rule extraction (e.g. time-out).
    In the search of the best derivation, we cannot assume any more
    that there is an estimation of the best transformation at each
    source node. The salvage proceeds as follows:
    1. From the root node of the source tree, find the most immediate
       children for which there is 
    """
    src_paths = self.tree1.treepositions()
    trg_paths = self.tree2.treepositions()
    src_candidates = list(self.src_explored)
    src_candidates = sorted(src_candidates, key=lambda x: len(x))
    discarded_paths = set()
    for i, path in enumerate(src_candidates):
      discarded_paths.update(
        [s for s in src_candidates[(i + 1):] if tstartswith(s, path)])
    src_subpaths = tuple(set(src_candidates) - discarded_paths)
    trg_subpaths = [self.costs[s].GetBestScoreItem().path2 for s in src_subpaths]

    derivation = []
    for src_subpath, trg_subpath in zip(src_subpaths, trg_subpaths):
      child_derivation = GetBestDerivation(self.best_transformations,
        self.tree1, self.tree2, src_subpath, trg_subpath)
      derivation.extend(child_derivation)
    return derivation

  def CombineAndFilterDisjointPaths(self, prev_disjoint, imm_disjoint,
      src_path, trg_path):
    disjoint_paths = \
      [(prev_src_paths + imm_src_paths,
        prev_trg_paths + imm_trg_paths) \
          for prev_src_paths, prev_trg_paths in prev_disjoint \
            for imm_src_paths, imm_trg_paths in imm_disjoint]
    disjoint_paths.extend(prev_disjoint)
    disjoint_paths.extend(imm_disjoint)
    disjoint_paths = \
      self.FilterDisjointPathsRec(disjoint_paths, src_path, trg_path)
    return disjoint_paths

  def FilterDisjointPathsRec(self, disjoint_paths, src_path, trg_path):
    """
    Filter pairs of source and target disjoint paths according to different
    criteria. This is useful to narrow the search space, possibly at the
    expense of an eventual exclusion of legitimated disjoint paths.
    :param disjoint_paths is a list of tuples.
    """
    # Renaming variables.
    max_branches = self.kMaxSourceBranches
    max_depth = self.kMaxSourceDepth
    # Deep copy of disjoint paths.
    valid_disjoint_paths = disjoint_paths[:]
    # Remove those disjoint paths whose target subpaths are not descendants
    # of the target path.
    valid_disjoint_paths = \
      filter(lambda x: all([tstartswith(s, trg_path) for s in x[1]]),
             valid_disjoint_paths)
    # Remove those disjoint paths with more than max_branches paths.
    valid_disjoint_paths = \
      filter(lambda x: len(x[0]) <= max_branches, valid_disjoint_paths)
    # Remove those disjoint paths with any of its subpaths longer than src or trg
    # path than max_depth.
    # For source disjoint paths:
    valid_disjoint_paths = \
      filter(lambda x: all([len(p) - len(src_path) <= max_depth for p in x[0]]),
             valid_disjoint_paths)
    # For target disjoint paths:
    valid_disjoint_paths = \
      filter(lambda x: all([len(p) - len(trg_path) <= max_depth for p in x[1]]),
             valid_disjoint_paths)
    # Remove those disjoint paths whose target or source subpaths
    # are not disjoint.
    valid_disjoint_paths = \
      filter(lambda x: AreDisjointPaths(x[1]) and AreDisjointPaths(x[0]),
             valid_disjoint_paths)
    # Remove those disjoint source paths that have a low leaf coverage.
    filtered_disjoint_paths = valid_disjoint_paths[:]
    filtered_disjoint_paths = \
      filter(lambda x: not IsLowLeafCoverageIncremental(
               self.tree1, x[0],
               src_min_index=min(self.tree1.path_to_leaves_indices[x[0][0]]),
               current_index=max(self.tree1.path_to_leaves_indices[x[0][-1]]),
               min_coverage=0.5,
               min_leaves=10),
             filtered_disjoint_paths)
    # Remove those disjoint paths that have a very different leaf coverage
    # on the source and the target.
    filtered_disjoint_paths = \
      filter(lambda x: not IsDifferentLeafCoverage(x,
               self.tree1, src_path,
               self.tree2, trg_path,
               difference=0.5,
               min_leaves=10),
             filtered_disjoint_paths)
    # We have filtered out all disjoint paths, the search will fail.
    # In such case, we step-back in the filtering and return the unfiltered
    # disjoint paths.
    if not filtered_disjoint_paths:
      return valid_disjoint_paths
    return filtered_disjoint_paths

def GetBestN(cost_items, n = 100):
  best_results = PriorityQueue(n)
  for cost, items in cost_items:
    best_results.Push(cost, items)
  return best_results.GetItems()

def FilterDisjointPaths(disjoint_paths, src_tree, src_path, trg_tree, trg_path,
    src_min_index, current_src_index, max_branches, max_depth):
  """
  Filter pairs of source and target disjoint paths according to different
  criteria. This is useful to narrow the search space, possibly at the
  expense of an eventual exclusion of legitimated disjoint paths.
  :param disjoint_paths is a list of tuples.
  """
  valid_disjoint_paths = disjoint_paths[:]
  # Remove those disjoint paths whose target subpaths are not descendants
  # of the target path.
  valid_disjoint_paths = \
    filter(lambda x: all([tstartswith(s, trg_path) for s in x[1]]),
           valid_disjoint_paths)
  # Remove those disjoint paths with more than max_branches paths.
  valid_disjoint_paths = \
    filter(lambda x: len(x) <= max_branches, valid_disjoint_paths)
  # Remove those disjoint paths with any of its subpaths longer than src or trg
  # path than max_depth.
  # For source disjoint paths:
  valid_disjoint_paths = \
    filter(lambda x: all([len(p) - len(src_path) <= max_depth for p in x[0]]),
           valid_disjoint_paths)
  # For target disjoint paths:
  valid_disjoint_paths = \
    filter(lambda x: all([len(p) - len(trg_path) <= max_depth for p in x[1]]),
           valid_disjoint_paths)
  # Remove those disjoint paths whose target or source subpaths
  # are not disjoint.
  valid_disjoint_paths = \
    filter(lambda x: AreDisjointPaths(x[1]) and AreDisjointPaths(x[0]),
           valid_disjoint_paths)
  # Remove those disjoint source paths that have a low leaf coverage.
  filtered_disjoint_paths = valid_disjoint_paths[:]
  filtered_disjoint_paths = \
    filter(lambda x: not IsLowLeafCoverageIncremental(
                       src_tree, x[0], src_min_index,
                       current_index=current_src_index,
                       min_coverage=0.5,
                       min_leaves=20),
           filtered_disjoint_paths)
  # Remove those disjoint paths that have a very different leaf coverage
  # on the source and the target.
  filtered_disjoint_paths = \
    filter(lambda x: not IsDifferentLeafCoverage(
                       x, src_tree, src_path, trg_tree, trg_path,
                       difference=0.5,
                       min_leaves=10),
           filtered_disjoint_paths)
  # We have filtered out all disjoint paths, the search will fail.
  # In such case, we step-back in the filtering and return the unfiltered
  # disjoint paths.
  if not filtered_disjoint_paths:
    logging.warning('Failed to filter disjoint paths for {0} vs. {1}'\
                    .format(src_path, trg_path))
    return valid_disjoint_paths
  return filtered_disjoint_paths

def IsDifferentLeafCoverage(disjoint_paths, src_tree, src_path, trg_tree, trg_path,
  difference=1.0, min_leaves=10):
  """
  It signals whether the leaf coverage of the source tree at path src_path and
  the leaf coverage of the target tree at path trg_path is "significantly"
  different. In case difference=0.5, "significantly" means 50% different.
  """
  src_subpaths, trg_subpaths = disjoint_paths
  if IsString(src_tree) or IsString(trg_tree):
    return False
  num_src_leaves = src_tree.GetNumLeaves(src_path)
  num_trg_leaves = trg_tree.GetNumLeaves(trg_path)
  if num_src_leaves <= min_leaves or num_trg_leaves <= min_leaves:
    return False
  src_coverage = sum([src_tree.GetNumLeaves(s) for s in src_subpaths]) \
               / src_tree.GetNumLeaves(src_path)
  trg_coverage = sum([trg_tree.GetNumLeaves(s) for s in trg_subpaths]) \
               / trg_tree.GetNumLeaves(trg_path)
  if abs(src_coverage - trg_coverage) > difference:
    return True
  return False

def GetItemsForChildSrcPath(store, src_path, trg_path, child_src_path):
  """
  Obtain source and target disjoint paths for all possible
  disjoint source paths including child_src_path and its children,
  that transform trg_path or any subset of its children.
  Keys of store are 5-tuples:
  [0] src_path
  [1] trg_path
  [2] child_src_path
  [3] num_src_leaves_covered
  [4] num_trg_leaves_covered
  Values of store are 3-tuples:
  [0] source disjoint paths
  [1] target disjoint paths
  """
  items = list(itertools.chain(*[priority_queue.GetItems() \
    for key, priority_queue in store.items() \
      if key[0] == src_path and \
         key[1] == trg_path and \
         key[2] == child_src_path]))
  return items

def SortAccordingTo(items, indices):
  assert len(items) == len(indices), \
    'Length of items and indices differ: {0} vs. {1}'.format(items, indices)
  return [x[1] for x in sorted(zip(indices, items))]

def IsLowLeafCoverageIncremental(tree, subpaths, src_min_index, current_index,
    min_coverage = 0.6, min_leaves = 10):
  num_visited_leaves = float(current_index - src_min_index + 1)
  if num_visited_leaves <= min_leaves:
    return False
  # num_path_leaves = float(src_max_index - src_min_index + 1)
  num_subpath_leaves = sum([tree.path_to_num_leaves[s] for s in subpaths])
  if num_subpath_leaves / num_visited_leaves < min_coverage:
    return True
  return False

def IsLowLeafCoverage(tree, path, subpaths, min_coverage = 0.8, min_leaves = 5):
  """
  Signal if the number of leaves spanned by tree[path] is lower than min_leaves,
  or whether the percentage of leaves spanned by subpaths divided by the number
  of leaves spanned by path is below the threshold min_coverage.
  This function is useful to prune tree patterns that potentially may not be
  well explained by their corresponding target tree patterns.
  Note that tree.path_to_num_leaves is a dictionary path -> num_leaves whose
  keys are paths (tuples) and values are number of leaves (*float*).
  """
  num_path_leaves = tree.path_to_num_leaves[path] if not IsString(tree) else 1
  if num_path_leaves <= min_leaves:
    return False
  num_subpaths_leaves = sum([tree.path_to_num_leaves[s] for s in subpaths])
  if num_subpaths_leaves / num_path_leaves < min_coverage:
    return True
  return False

def FilterOptimalPaths(paths, costs, over_costs,
                       permit_suboptimal_parents = True):
  """
  Filter out paths that are below a path that is optimal.
  If permit_suboptimal_parents = True, then parents of optimal paths
  are included in the result. Such characteristic in useful when
  obtaining target subpaths.
  """
  paths_unique = sorted(tuple(set(paths)), key=lambda x: len(x))
  paths_optimal = [s for s in paths_unique \
                     if costs[s].GetBestScore() <= 1.0 * over_costs.get(s, np.inf)]
  if not permit_suboptimal_parents:
    paths_unique = paths_optimal
  if not paths_optimal:
    candidate_paths = paths_unique
  else:
    paths_optimal = sorted(paths_optimal, key=lambda x: len(x))
    discarded_paths = set()
    for path_optimal in paths_optimal:
      path_index = paths_unique.index(path_optimal)
      discarded_paths.update(
        [s for s in paths_unique[(path_index + 1):] \
           if tstartswith(s, path_optimal)])
    candidate_paths = tuple(set(paths_unique) - discarded_paths)
  # Sort in lexicographical order.
  candidate_paths = sorted(candidate_paths)
  return candidate_paths

def GetSortedTreePaths(tree):
  """
  Gets paths sorted by descending length.
  """
  if not IsString(tree):
    paths = sorted(tree.treepositions(), key=lambda x: len(x), reverse=True)
  else:
    paths = [()]
  return paths

def AreTreesUnbalanced(src_tree, src_path, trg_tree, trg_path,
                       threshold = 3, min_leaves = 5):
  if IsString(src_tree):
    num_src_leaves = 1
  else:
    num_src_leaves = src_tree.GetNumLeaves(src_path)
  if IsString(trg_tree):
    num_trg_leaves = 1
  else:
    num_trg_leaves = trg_tree.GetNumLeaves(trg_path)
  if num_src_leaves < min_leaves and num_trg_leaves < min_leaves:
    return False
  if num_src_leaves * threshold < num_trg_leaves \
     or num_trg_leaves * threshold < num_src_leaves:
    return True
  return False

def AreTreePatternsUnbalanced(tree_pattern1, tree_pattern2, threshold = 4):
  num_leaves1 = len(tree_pattern1.GetLeaves())
  num_leaves2 = len(tree_pattern2.GetLeaves())
  if abs(num_leaves1 - num_leaves2) > threshold:
    return True
  return False

def PrintCostMatrix(costs):
  print('Cost matrix:')
  for k, v in costs.items():
    print(k)
    for cost, transformation in v:
      print(cost, transformation)

def AreItemsEqual(list1, list2):
  for list1_item in list1:
    if list1_item not in list2:
      return False
  for list2_item in list2:
    if list2_item not in list1:
      return False
  return True

def BuildTransformation(path1, path2, \
                        src_path_subset, trg_path_subset, similarity):
  transformation = Transformation(path1, path2, \
    src_path_subset, trg_path_subset, similarity)
  return transformation

def MakeGlueRule(tree1, tree2, initial_state):
  path1, path2 = (), ()
  subpaths1, subpaths2 = [()], [()]
  cost = 0.0
  similarity = Similarity(cost, initial_state, None, None)
  transformation = \
    BuildTransformation(path1, path2, subpaths1, subpaths2, similarity)
  state_id = '0'
  glue_rule = transformation.BuildTransformationRule(
    tree1, tree2, state_id)
  return glue_rule

def MakeProductionFromRule(rule):
  state = rule.state
  src_path = rule.src_path
  trg_path = rule.trg_path
  non_terminal = state, src_path, trg_path
  deriv_rhs = RHS(rule)
  lhs_variable_paths = rule.lhs_vars_to_paths
  rhs_variable_paths_sorted = sorted(rule.rhs_vars_to_paths.items())
  for rhs_variable, rhs_path in rhs_variable_paths_sorted:
    lhs_path = lhs_variable_paths[rhs_variable]
    lhs_absolute_path = src_path + lhs_path
    rhs_absolute_path = trg_path + rhs_path
    new_state = rule.newstates[rhs_path]
    child_non_terminal = (new_state, lhs_absolute_path, rhs_absolute_path, '')
    deriv_rhs.non_terminals.append(child_non_terminal)
  else:
    production = Production(non_terminal, deriv_rhs, rule.weight)
  return production

def GetAbsoluteChildPathsFromRule(rule, src_path, trg_path):
  """
  Returns a list of tuples, where each tuple contains a child lhs absolute path
  and a child rhs absolute paths.
  """
  lhs_variable_paths = rule.lhs_vars_to_paths
  rhs_variable_paths_sorted = sorted(rule.rhs_vars_to_paths.items())
  child_abs_paths = []
  for rhs_variable, rhs_path in rhs_variable_paths_sorted:
    lhs_path = lhs_variable_paths[rhs_variable]
    lhs_absolute_path = src_path + lhs_path
    rhs_absolute_path = trg_path + rhs_path
    child_abs_paths.append((lhs_absolute_path, rhs_absolute_path))
  return child_abs_paths

def ExpandNewstatesForRule(rule, transformation, best_transformations):
  """
  When building a rule from a transformation instance, newstates
  (mappings from a RHS path to the next state) are not set.
  This function returns a list of rules, whose newstates for each
  RHS path are the cross-product of all possible newstates.
  """
  state = rule.state
  src_path = transformation.src_path
  trg_path = transformation.trg_path
  child_abs_paths = GetAbsoluteChildPathsFromRule(rule, src_path, trg_path)
  all_child_states = []
  for child_src_path, child_trg_path in child_abs_paths:
    transformations = \
      best_transformations[(child_src_path, child_trg_path)].GetItems()
    child_states = [MakeStateName(t.similarity.relation, t.src_path, t.src_subpaths) \
                      for t in transformations]
    child_states = list(set(child_states))
    all_child_states.append(child_states)
  child_rhs_abs_paths = [p[1] for p in child_abs_paths]
  child_rhs_rel_paths = [p[len(trg_path):] for p in child_rhs_abs_paths]
  rules_ext = []
  for state_combination in itertools.product(*all_child_states):
    assert len(child_rhs_rel_paths) == len(state_combination)
    newstates = {path : state \
                   for path, state in zip(child_rhs_rel_paths, state_combination)}
    rule_ext = XTRule(rule.state, rule.lhs, rule.rhs, newstates, rule.weight)
    rule_ext.src_path = rule.src_path
    rule_ext.trg_path = rule.trg_path
    rule_ext.src_subpaths = rule.src_subpaths
    rule_ext.trg_subpaths = rule.trg_subpaths
    rule_ext.features = rule.features
    rules_ext.append(rule_ext)
  return rules_ext

def MakeGlueProductions(best_transformations, tree1, tree2,
    initial_state, src_path = (), trg_path = ()):
  glue_rule = MakeGlueRule(tree1, tree2, initial_state)
  transformations = best_transformations[(src_path, trg_path)].GetItems()
  glue_states = [MakeStateName(t.similarity.relation, t.src_path, t.src_subpaths) \
                   for t in transformations]
  glue_rules = []
  for state in glue_states:
    # Ensure we do not create a glue rule that re-directs to itself.
    if state == initial_state:
      continue
    rule = deepcopy(glue_rule)
    rule.newstates = {trg_path : state}
    glue_rules.append(rule)
  glue_productions = [MakeProductionFromRule(glue_rule) for glue_rule in glue_rules]
  return glue_productions

def ObtainProductionsFromBestTransformations(best_transformations, tree1, tree2,
    initial_state, src_path = (), trg_path = (), state_id = None):
  productions = []
  src_and_trg_paths = best_transformations.keys()
  assert (src_path, trg_path) in src_and_trg_paths
  rules_extended = []
  for src_path, trg_path in src_and_trg_paths:
    transformations = best_transformations[(src_path, trg_path)].GetItems()
    rules = [t.BuildTransformationRule(tree1, tree2, state_id) for t in transformations]
    for rule, trans in zip(rules, transformations):
      rules_ext = ExpandNewstatesForRule(rule, trans, best_transformations)
      rules_extended.extend(rules_ext)
  productions = [MakeProductionFromRule(rule_ext) for rule_ext in rules_extended]
  glue_productions = MakeGlueProductions(best_transformations, tree1, tree2,
    initial_state, src_path = (), trg_path = ())
  productions.extend(glue_productions)
  return productions

def GetCostDerivation(derivation):
  """
  Here, a derivation is a sequence of productions.
  Since the rule extraction considers rule weights as costs,
  we need to define a different function to obtain the total
  cost of a derivation, and complement it. In that way,
  the behavior of the wRTG when searching for the best derivation
  is equivalent to the scenario when rule weights are probabilities.
  """
  return -sum([prod.rhs.rule.weight for prod in derivation])

def CombineDerivationCosts(derivations):
  """
  Here, a derivation is a sequence of productions.
  The combination of derivation costs is simply the sum of their costs.
  """
  return sum([GetCostDerivation(derivation) for derivation in derivations])

def GetBestDerivations(best_transformations, tree1, tree2, initial_state,
    src_path = (), trg_path = (), state_id = None, enable_deletions=False):
  productions = ObtainProductionsFromBestTransformations(best_transformations,
    tree1, tree2, initial_state, src_path, trg_path, state_id)
  productions = sorted(list(set(productions)))
  # productions = list(set(productions))
  rules = list(set(production.rhs.rule for production in productions))
  S = initial_state, src_path, trg_path, ''
  non_terminals_lhs = set([p.non_terminal for p in productions])
  non_terminals_rhs = set([nt for p in productions for nt in p.rhs.non_terminals])
  non_terminals = list(non_terminals_lhs.union(non_terminals_rhs))
 
  # if 'fuel' in tree1.leaves():
  #   import pickle
  #   i="2"
  #   with open('fuel.' + i + '.productions.pkl', 'wb') as fout:
  #     pickle.dump(productions, fout)
  #   with open('fuel.' + i + '.rules.pkl', 'wb') as fout:
  #     pickle.dump(rules, fout)
  #   with open('fuel.' + i + '.nts.pkl', 'wb') as fout:
  #     pickle.dump(non_terminals, fout)

  wrtg = wRTG(rules, non_terminals, S, productions, convert_to_prob=False)
  prunned_wrtg = wrtg.Prune()
  # Over-write the methods to compute the score of a derivation
  # and to combine scores of derivations.
  prunned_wrtg.ScoreDerivation = GetCostDerivation
  prunned_wrtg.CombineDerivationScores = CombineDerivationCosts
  prunned_wrtg.ClearCaches()
  for derivation in prunned_wrtg.ObtainDerivationsFromNT(S):
    if enable_deletions:
      rules = [prod.rhs.rule.MakeDeletingRule() for prod in derivation]
    else:
      rules = [prod.rhs.rule for prod in derivation]
    yield rules

def GetBestDerivation(best_transformations, tree1, tree2,
                      src_path = (), trg_path = (), state_id = None):
  derivation = []
  transformation = best_transformations[(src_path, trg_path)].GetBestScoreItem()
  rule = transformation.BuildTransformationRule(tree1, tree2, state_id)
  derivation.append(rule)
  src_subpaths = transformation.src_subpaths
  trg_subpaths = transformation.trg_subpaths
  for src_subpath, trg_subpath in zip(src_subpaths, trg_subpaths):
    child_derivation = GetBestDerivation(best_transformations,
      tree1, tree2, src_subpath, trg_subpath, state_id)
    derivation.extend(child_derivation)
  return derivation

