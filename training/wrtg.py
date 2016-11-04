from collections import defaultdict
import itertools
import logging
import numpy as np
from operator import mul

from nltk import Tree

from semirings.semiring_prob import ProbSemiRing
from utils.generators import GeneratorsList, OrderedProduct
from utils.production import (RHS, Production, TargetProjectionFromDerivation,
  SourceProjectionFromDerivationMix, SourceProjectionFromDerivationStrict)
from utils.tree_tools import immutable, IsString, variables_to_paths

class wRTG:
  """
  A weighted regular tree grammar (wRTG) is a tuple G = (Sigma, N, S, P), where:
  Sigma: alphabet (rules)
  N: set of nonterminals
  S: initial nonterminal
  P: set of weighted productions
  """
  def __init__(self, Sigma, N, S, P, convert_to_prob=True):
    self.Sigma = list(Sigma)
    self.convert_to_prob = convert_to_prob
    if convert_to_prob:
      for rule in self.Sigma:
        if not isinstance(rule.weight, ProbSemiRing):
          rule.weight = ProbSemiRing(prob=rule.weight)
    non_terminals = []
    for nt in N:
      non_terminal = nt + ('',) if isinstance(nt[-1], tuple) else nt
      non_terminals.append(non_terminal)
    self.N = non_terminals
    self.S = S + ('',) if isinstance(S[-1], tuple) else S
    self.P = list(P)
    self.production_index = self.BuildProductionIndex()
    self.production_reverse_index = self.BuildProductionReverseIndex()
    self.Inside = {}
    self.MaxInside = {}
    if convert_to_prob:
      self.Outside = {self.S : ProbSemiRing(1.0)}
    else:
      self.Outside = {self.S : 1.0}
    self.derivations_cache = defaultdict(list)
    self.best_derivation_cache = {}
    self.ScoreDerivation = GetProbabilityOfDerivation
    self.CombineDerivationScores = \
      lambda dd: reduce(mul, [self.ScoreDerivation(d) for d in dd])
    self.LMScoreDerivations = None
    self.cached_derivations_generators = {}
    self.feat_weights = None
    self.feat_inst = None

  def ClearCaches(self):
    self.derivations_cache = defaultdict(list)
    self.cached_derivations_generators = {}

  def PrintRules(self):
    rules = [p.rhs.rule for p in self.P]
    rules.sort(key=lambda r: repr(r))
    for rule in rules:
      print(rule.PrintYaml() + "\n")

  def GenerateNBestTreesMax(self, max_derivations = 50, direction = 'target'):
    tree_to_weight = defaultdict(float)
    for i, (tree, weight) in enumerate(self.GenerateTrees(direction)):
      if i > max_derivations:
        break
      tree_immutable = immutable(tree)
      current_weight = float(weight)
      if tree_immutable in tree_to_weight:
        assert tree_to_weight[tree_immutable] >= current_weight
        continue
      tree_to_weight[tree_immutable] = current_weight
      yield tree, weight

  def GenerateNBestTreesMax_(self, max_derivations = 50, direction = 'target'):
    tree_to_weight = defaultdict(float)
    for i, (tree, weight) in enumerate(self.GenerateTrees(direction)):
      if i > max_derivations:
        break
      tree_immutable = immutable(tree)
      current_weight = float(weight)
      if tree_immutable in tree_to_weight:
        assert tree_to_weight[tree_immutable] >= current_weight
        continue
      tree_to_weight[tree_immutable] = current_weight
    sorted_trees_by_weight = \
      sorted([(tree, weight) for (tree, weight) in tree_to_weight.items()], \
             key=lambda x: x[1], reverse=True)
    return sorted_trees_by_weight

  def GenerateNBestTrees(self, max_derivations = 50, direction = 'target'):
    accumulated_tree_weight = defaultdict(float)
    for i, (tree, weight) in enumerate(self.GenerateTrees(direction)):
      if i > max_derivations:
        break
      tree_immutable = immutable(tree)
      current_weight = float(weight)
      accumulated_tree_weight[tree_immutable] += current_weight
    sorted_trees_by_weight = \
      sorted([(tree, weight) for (tree, weight) in accumulated_tree_weight.items()], \
             key=lambda x: x[1], reverse=True)
    return sorted_trees_by_weight

  def GenerateTrees(self, direction = 'target'):
    for i, derivation in enumerate(self.ObtainDerivationsFromNT()):
      if direction == 'source':
        source_projection = SourceProjectionFromDerivationStrict(derivation)
        yield source_projection
      else:
        target_projection = TargetProjectionFromDerivation(derivation)
        target_projection = (target_projection[0], self.ScoreDerivation(derivation))
        logging.debug('Target projection {0:03d}: {1}'.format(i, target_projection))
        if self.feat_inst is not None:
          logging.debug('\n'.join(
            [repr(d) + '\n' + d.rhs.rule.PrintYaml() + \
             '\n' + self.feat_inst.DescribeFeatureIDs(d.rhs.rule) \
             for d in derivation]))
        else:
          logging.debug('\n'.join([repr(d) + '\n' + d.rhs.rule.PrintYaml() for d in derivation]))
        yield target_projection

  def RetrieveOrMakeDerivationsGeneratorFromNT(self, start_symbol):
    """
    This retrieves from a cache or produces a generator of derivations
    that start from a given non-terminal.
    """
    # self.PrintTotals()
    if start_symbol in self.cached_derivations_generators:
      derivations_generator = self.cached_derivations_generators[start_symbol]
    else:
      derivations = [] # Will contain lazy lists of sub-derivations.
      productions = self.production_index.get(start_symbol, [])
      for production in productions:
        production_derivations = \
          self.ObtainDerivationsFromProduction(production)
        derivations.append(production_derivations)
      # Productions_derivations is a list of generators of sorted derivations.
      # We build a GeneratorsList generator, that yields the best items among
      # all lists, in order.
      generators_list = GeneratorsList(derivations, self.ScoreDerivation)
      derivations_generator = generators_list.items()
      self.cached_derivations_generators[start_symbol] = derivations_generator
    return derivations_generator

  def RetrieveOrMakeDerivationsGeneratorFromProduction(self, production):
    """
    This produces a generator of tuple combinations. Each tuple combination
    is a list of derivations, that when concatenated, produce a derivation.
    """
    # self.PrintTotals()
    if production in self.cached_derivations_generators:
      derivations_generator = self.cached_derivations_generators[production]
    else:
      non_terminals = production.rhs.non_terminals
      if not non_terminals:
        tuple_combinations = iter([[]]) # A single empty tuple combination
        # that will combine to produce the derivation: [production].
      else:
        non_terminals = production.rhs.non_terminals
        nt_derivations = [(lambda x: \
                             (lambda: self.ObtainDerivationsFromNT(x)))(nt) \
                          for nt in non_terminals]
        tuple_combinations = OrderedProduct(*nt_derivations,
          key=self.CombineDerivationScores, lm=self.LMScoreDerivations)
      derivations_generator = \
        itertools.imap(lambda tuple_comb: \
                         list(itertools.chain([production], *tuple_comb)),
                       tuple_combinations)
      self.cached_derivations_generators[production] = derivations_generator
    return derivations_generator

  def ObtainDerivations(self, start_symbol, deriv_gen_func):
    """
    This method obtains derivations in descending order of probability,
    given a non-terminal or production.
    This algorithm is a generalization of Algorithm 1 presented in
    "An Overview of probabilistic tree transducers for natural language processing"
    where we do not only store the best derivation, but the N-best derivations
    at each RHS non-terminal or production.
    """
    i = 0
    while True:
      if __debug__:
        CheckDerivationsAreUniqueAndMonotonic(
          self.derivations_cache, start_symbol, self.ScoreDerivation)
      # Case: some derivations are already cached. Retrieving them.
      if i < len(self.derivations_cache[start_symbol]):
        derivation = self.derivations_cache[start_symbol][i]
        if derivation is None:
          return
        yield derivation
      # Case: No more derivations are cached. Recover a shared generator,
      # obtain the next derivation, cache it and yield it.
      else:
        derivations_generator = deriv_gen_func(start_symbol)
        try:
          derivation = derivations_generator.next()
          assert not IsDerivationInList(
            derivation, self.derivations_cache[start_symbol])
          self.derivations_cache[start_symbol].append(derivation)
          yield derivation
        except StopIteration:
          self.derivations_cache[start_symbol].append(None)
          return
      i += 1

  def ObtainDerivationsFromNT(self, non_terminal = None):
    """
    This method obtains derivations in descending order of probability,
    given a non-terminal.
    This algorithm is a generalization of Algorithm 1 presented in
    "An Overview of probabilistic tree transducers for natural language processing"
    where we do not only store the best derivation, but the N-best derivations
    at each RHS non-terminal.
    """
    if non_terminal is None:
      non_terminal = self.S
    derivations = self.ObtainDerivations(
      non_terminal, self.RetrieveOrMakeDerivationsGeneratorFromNT)
    return derivations

  def ObtainDerivationsFromProduction(self, production):
    derivations = self.ObtainDerivations(
      production, self.RetrieveOrMakeDerivationsGeneratorFromProduction)
    return derivations

  # TODO: remove this method, and replace it in the rest of code/tests.
  def ObtainBestDerivation(self, start_symbol = None):
    """
    From a regular tree grammar (stored in self.production_index), it produces
    the best derivation using the Viterbi Algorithm. A single derivation is a
    list of productions: [p1, p2, p3]. 
    """
    if start_symbol in self.best_derivation_cache:
      return self.best_derivation_cache[start_symbol]
    if start_symbol == None:
      start_symbol = self.S
    derivations = []
    productions = self.production_index.get(start_symbol, [])
    for production in productions:
      non_terminals = production.rhs.non_terminals
      if not non_terminals:
        derivations.append([production])
        continue
      nt_derivations = []
      for nt in non_terminals:
        nt_derivations.append(self.ObtainBestDerivation(nt))
      derivations.append(list(itertools.chain([production], *nt_derivations)))
    if not derivations:
      best_derivation = []
    else:
      best_derivation = self.GetBestDerivation(derivations)
    self.best_derivation_cache[start_symbol] = best_derivation
    return best_derivation

  def GetBestDerivation(self, derivations):
    probabilities = []
    for derivation in derivations:
      probabilities.append(reduce(mul, [p.rhs.rule.weight for p in derivation]))
    probabilities = np.array(probabilities)
    best_derivation_index = probabilities.argmax()
    return derivations[best_derivation_index] 

  def BuildProductionIndex_(self):
    """
    key: non_terminal, value: production such that non_terminal is the LHS of
    the production. 
    """
    production_index = defaultdict(list)
    for production in self.P:
      production_index[production.non_terminal].append(production)
    return production_index

  def BuildProductionIndex(self):
    """
    key: non_terminal, value: production such that non_terminal is the LHS of
    the production. 
    """
    production_index = defaultdict(list)
    for production in self.P:
      production_index[production.non_terminal].append(production)
    for nt in production_index:
      production_index[nt].sort()
    return production_index

  def BuildProductionReverseIndex(self):
    """ key: non_terminal, value: production p, such that non_terminal in p.rhs. """
    production_reverse_index = defaultdict(set)
    for production in self.P:
      non_terminals = production.rhs.non_terminals
      for non_terminal in non_terminals:
        production_reverse_index[non_terminal].add(production)
    return production_reverse_index

  def Reach(self, non_terminal, B, adjacents, r):
    """ Helper function described in Algorithm 2 "Training Tree Transducers"."""
    B[non_terminal] = True
    for production_id in adjacents[non_terminal]:
      non_terminal = self.P[production_id].non_terminal
      if not B[non_terminal]:
        r[production_id] -= 1       
        if r[production_id] == 0:
          (B, r) = self.Reach(non_terminal, B, adjacents, r)
    return (B, r) 

  def Use(self, non_terminal, B, A):
    """ Helper function described in Algorithm 2 "Training Tree Transducers"."""
    A[non_terminal] = True
    for production in self.production_index.get(non_terminal, []):
      for rhs_non_terminal in production.rhs.non_terminals:
        if not A[rhs_non_terminal] and B[rhs_non_terminal]:
          (B, A) = self.Use(rhs_non_terminal, B, A)
    return (B, A)
 
  # def Prune(self, initial_state = 'q0'):
  def Prune(self):
    """
    Prunes a wRTG following Algorithm 2 (page 405) from "Training Tree Transducers".
    Returns the prunned wRTG.
    """
    M = set()
    # B[NT] == True if from NT we can produce a complete tree that does not contain
    # non-terminals anymore.
    B = { n : False for n in self.N }
    A = { n : False for n in self.N }
    adjacents = { n : set() for n in self.N }
    # r[i] is the number of remaining NT for production with ID i.
    r = [None] * len(self.P)
    for i, production in enumerate(self.P):
      non_terminal, deriv_rhs = production.non_terminal, production.rhs
      for deriv_rhs_nt in deriv_rhs.non_terminals:
        adjacents[deriv_rhs_nt].add(i)
      if not deriv_rhs.non_terminals:
        M.add(non_terminal)
      r[i] = len(deriv_rhs.non_terminals)
    for non_terminal in M:
      (B, r) = self.Reach(non_terminal, B, adjacents, r)
    (B, A) = self.Use(self.S, B, A)
    prunned_productions = [p for p in self.P \
                           if (A[p.non_terminal] and not p.rhs.non_terminals) \
                           or (A[p.non_terminal] and p.rhs.non_terminals \
                               and all(map(lambda x: A[x], p.rhs.non_terminals)))]
    prunned_non_terminals = [n for (n, t) in A.items() if t]
    return wRTG(self.Sigma, prunned_non_terminals, self.S, prunned_productions,
                self.convert_to_prob)

  def GetMaxInside(self, non_terminal):
    if isinstance(non_terminal[-1], tuple):
      non_terminal = non_terminal + ('',)
    if non_terminal not in self.MaxInside:
      return None
    return self.MaxInside[non_terminal]

  def GetOutside(self, non_terminal):
    if isinstance(non_terminal[-1], tuple):
      non_terminal = non_terminal + ('',)
    if non_terminal not in self.Outside:
      return None
    return self.Outside[non_terminal]

  def GetInside(self, non_terminal = None):
    if non_terminal == None:
      non_terminal = self.S
    if isinstance(non_terminal[-1], tuple):
      non_terminal = non_terminal + ('',)
    if non_terminal not in self.Inside:
      return None
    return self.Inside[non_terminal]

  # TODO: This method has never been used, and it might be a good moment to
  # remove it, together with the test associated to it.
  def ComputeMaxInsideWeights(self, nt_or_rhs = None):
    """
    This method populates the self.MaxInside dictionary: NT -> (weight, child_position)
    """
    if isinstance(nt_or_rhs, RHS):
      rhs = nt_or_rhs
      weight = ProbSemiRing(1.0)
      for non_terminal in rhs.non_terminals:
        weight *= self.ComputeMaxInsideWeights(non_terminal)
      return weight
    non_terminal = nt_or_rhs
    if non_terminal == None:
      self.MaxInside = {}
      non_terminal = self.S
    if non_terminal in self.MaxInside:
      (weight, child_position) = self.MaxInside[non_terminal]
      return weight
    productions = self.production_index.get(non_terminal, [])
    weights = [0] * len(productions)
    for i, production in enumerate(productions):
      weights[i] = production.rhs.rule.weight \
                   * self.ComputeMaxInsideWeights(production.rhs)
    max_weight = max(weights)
    child_position = weights.index(max_weight)
    self.MaxInside[non_terminal] = (max_weight, child_position)
    return max_weight

  def ComputeInsideWeights(self, nt_or_rhs = None):
    """
    This method populates the self.Inside dictionary: NT | production -> weight
    """
    if isinstance(nt_or_rhs, RHS):
      rhs = nt_or_rhs
      weight = ProbSemiRing(1.0)
      for non_terminal in rhs.non_terminals:
        weight *= self.ComputeInsideWeights(non_terminal)
      return weight
    non_terminal = nt_or_rhs
    if non_terminal == None:
      self.Inside = {}
      non_terminal = self.S
    if non_terminal in self.Inside:
      return self.Inside[non_terminal]
    productions = self.production_index.get(non_terminal, [])
    weight = ProbSemiRing(0.0)
    for production in productions:
      weight += production.rhs.rule.weight \
                * self.ComputeInsideWeights(production.rhs)
    self.Inside[non_terminal] = weight
    return weight

  def ComputeRuleWeights_(self, model, feat_inst=None):
    """
    Visit every rule of the grammar, and assigns a weight to the rule
    according to the features of the rule and a model. Features of the rule
    are in the form:
    rule.features = [[feat_id1, feat_val1, ...], [feat_id2, feat_val2, ...], ...]
    """
    for p in self.P:
      rule = p.rhs.rule
      assert isinstance(rule.features, list) or feat_inst is not None
      if not isinstance(rule.features, list):
        src_treep, trg_treep = rule.GetTreePatterns()
        rule.features = feat_inst.InstantiateFeatures(src_treep, trg_treep, rule)
      rule.weight = model.weight_rule(rule)

  def ComputeInsideWeightsWithFeats(self, feat_weights, nt_or_rhs = None):
    """
    This method populates the self.Inside dictionary: NT | production -> weight.
    The parameter feat_weights defines the scaling factors of each feature.
    """
    if isinstance(nt_or_rhs, RHS):
      rhs = nt_or_rhs
      weight = ProbSemiRing(1.0)
      for non_terminal in rhs.non_terminals:
        weight *= self.ComputeInsideWeightsWithFeats(feat_weights, non_terminal)
      return weight
    non_terminal = nt_or_rhs
    if non_terminal == None:
      self.Inside = {}
      non_terminal = self.S
    if non_terminal in self.Inside:
      return self.Inside[non_terminal]
    productions = self.production_index.get(non_terminal, [])
    weight = ProbSemiRing(0.0)
    for production in productions:
      assert isinstance(production.rhs.rule.features, list)
      # from pudb import set_trace; set_trace()
      weight += MultWeightsBySparseFeats(feat_weights, production.rhs.rule.features) \
                * self.ComputeInsideWeightsWithFeats(feat_weights, production.rhs)
    self.Inside[non_terminal] = weight
    return weight

  def ComputeOutsideWeights(self, non_terminal = None):
    """
    This method populates the self.Outside dictionary: NT -> weight
    """
    if non_terminal == None:
      self.Outside = { self.S : ProbSemiRing(1.0) }
      [self.ComputeOutsideWeights(nt) for nt in self.N]
    if non_terminal in self.Outside:
      return self.Outside[non_terminal]
    productions = self.production_reverse_index.get(non_terminal, [])
    weight = ProbSemiRing(0.0)
    for p in productions:
      if not self.ComputeInsideWeights(non_terminal).is_zero():
        weight += p.rhs.rule.weight \
                  * self.ComputeOutsideWeights(p.non_terminal) \
                  * self.ComputeInsideWeights(p.rhs) \
                  / self.ComputeInsideWeights(non_terminal)
    self.Outside[non_terminal] = weight
    return weight

  def ComputeOutsideWeightsWithFeats(self, feat_weights, non_terminal = None):
    """
    This method populates the self.Outside dictionary: NT -> weight.
    The parameter feat_weights defines the scaling factors of each feature.
    """
    if non_terminal == None:
      self.Outside = { self.S : ProbSemiRing(1.0) }
      for nt in self.N:
        self.ComputeOutsideWeightsWithFeats(feat_weights, nt)
    if non_terminal in self.Outside:
      return self.Outside[non_terminal]
    productions = self.production_reverse_index.get(non_terminal, [])
    weight = ProbSemiRing(0.0)
    for p in productions:
      if not self.ComputeInsideWeightsWithFeats(feat_weights, non_terminal).is_zero():
        assert isinstance(p.rhs.rule.features, list)
        weight += MultWeightsBySparseFeats(feat_weights, p.rhs.rule.features) \
                  * self.ComputeOutsideWeightsWithFeats(feat_weights, p.non_terminal) \
                  * self.ComputeInsideWeightsWithFeats(feat_weights, p.rhs) \
                  / self.ComputeInsideWeightsWithFeats(feat_weights, non_terminal)
    self.Outside[non_terminal] = weight
    return weight

  def SumWeightsTreesInvolvingProduction(self, production):
    return self.Outside[production.non_terminal] \
           * production.rhs.rule.weight \
           * self.ComputeInsideWeights(production.rhs)

  def SumWeightsTreesInvolvingProductionWithFeats(self, feat_weights, production):
    return self.Outside[production.non_terminal] \
           * MultWeightsBySparseFeats(feat_weights, production.rhs.rule.features) \
           * self.ComputeInsideWeightsWithFeats(feat_weights, production.rhs)

def MultWeightsBySparseFeats(feat_weights, features):
  weight = ProbSemiRing(0.0)
  for feat_id, feat_val in features:
    feat_weight = feat_weights[feat_id]
    weight += ProbSemiRing(feat_weight) * feat_val
  return weight

def CheckDerivationsAreUniqueAndMonotonic(derivations_cache, key, scorer):
  if __debug__:
    derivations = derivations_cache.get(key, [])
    derivation_scores = [scorer(d) for d in derivations if d is not None]
    assert IsMonotonic(derivation_scores), ('Derivations with probabilities '
      'that are not monotonically decreasing for key {0}:\n{1}')\
      .format(key, derivation_scores)
    assert AreUniqueDerivations(derivations), ('List of derivations is not '
      'unique for key {0}:\n{1}').format(key, derivations)

def IsMonotonic(probs):
  return all([round(x, 5) >= round(y, 5) for x, y in zip(probs, probs[1:])])

def IsDerivationInList(derivation, derivations):
  frozen_derivations = [frozenset(d) for d in derivations if d is not None]
  return frozenset(derivation) in frozen_derivations

def AreUniqueDerivations(derivations):
  """
  Given a list of derivations (which in turn are a list of productions),
  tell whether those derivations are unique (there are no repetitions).
  """
  # Convert derivations into a frozen_set of productions (to be hashable).
  frozen_derivations = [frozenset(d) for d in derivations if d is not None]
  return len(frozen_derivations) == len(set(frozen_derivations))

def IsDerivation(derivation):
  return isinstance(derivation, list) and \
         all([isinstance(p, Production) for p in derivation])

def GetProbabilityOfDerivation(derivation):
  probability = reduce(mul, [p.rhs.rule.weight for p in derivation])
  return float(probability)

"""
def CombineProbabilitiesOfDerivations(derivations):
  return reduce(mul, [ScoreDerivation(d) for d in derivations])
"""
