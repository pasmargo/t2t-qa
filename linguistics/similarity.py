#!/usr/bin/env python
# -*- coding: utf-8 -*-

import abc
from collections import defaultdict
import logging

class Similarity:
  """
  A Similarity object contains the source and target objects (or pointers to them),
  their similarity score, and their type of relationship. For instance, if source
  and target are close synonyms, it would contain self.score = 0.9 and
  self.relation = 'synonym'. The attribute self.score measures the strength of
  the similarity between the source and target objects. Such score should be a
  positive number, to ensure that we can define monotonically non-decreasing
  objective functions on partial solutions in A* search algorithms. 
  """
  def __init__(self, score, relation, source, target = None):
    self.score = score
    self.relation = relation
    self.source = source
    self.target = target

  def __eq__2(self, other):
    return isinstance(other, self.__class__) \
      and self.score == other.score \
      and self.relation == other.relation \
      and self.source == other.source \
      and self.target == other.target

  def __eq__(self, other):
    return isinstance(other, self.__class__) \
      and self.relation == other.relation \
      and self.source == other.source \
      and self.target == other.target

  def __ne__(self, other):
    return not self.__eq__(other)

  def __str__(self):
    return repr(self)

  def __repr__(self):
    return (("<sim.\n  score: {0}\n  relation: {1}\n  source: {2}\n" +
             "  target: {3}>").format(
            self.score, self.relation, self.source, self.target))

  def __hash__(self):
    return hash(repr(self))


class SimilarityScorer(object):
  """
  A SimilarityScorer is a feature function \phi(s, t) such that measures
  the similarity between object s and t. It typically contains a parameter
  self.feature_weight that scales the importance of this similarity.
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def GetSimilarity(self, source, target):
    """
    GetSimilarity : (source, target) --> [Similarity].
    Given a source and a target object (subtrees, strings, etc.),
    it returns a list of Similarities (defined above).
    """
    return [Similarity(None, None, source, target)]

  def Close(self):
    pass

  def PreCache(self, source, target, options = None):
    pass

  @abc.abstractmethod
  def GetSimilar(self, source):
    """
    GetSimilar : source -> [Similarity].
    Given the source object (subtree, string, etc.), it returns a list of
    Similarities, where the Similarity.target field is populated with the
    target object that is similar to the source. E.g:
    GetSimilar('house') would return an object o such that
    o.target == 'condominium', o.relation = 'synonym' and o.score = 0.8.
    """
    return [Similarity(None, None, source, None)]

class SimilarityScorerEnsemble(SimilarityScorer):
  """
  This class implements a combination of similarity scorers that produce several
  similarity relationships (and their score) for a given source-target tuple,
  and also other similarity scorers that produce a single similarity relationship
  between the source-target tuple.
  """
  def __init__(self, scorers, scorers_global = []):
    self.scorers = list(scorers)
    self.scorers_global = list(scorers_global)
    self.kProb = 1e-7

  def Close(self):
    for scorer in self.scorers:
      scorer.Close()
    for scorer in self.scorers_global:
      scorer.Close()

  def PreCache(self, source, target, options = None):
    for scorer in self.scorers:
      scorer.PreCache(source, target, options)
    for scorer in self.scorers_global:
      scorer.PreCache(source, target, options)

  def GetGlobalScore(self, source, target):
    score = 0.0
    if not self.scorers_global:
      return score
    for scorer_global in self.scorers_global:
      scorer_similarities = scorer_global.GetSimilarity(source, target)
      if scorer_similarities:
        assert len(scorer_similarities) == 1, \
          'More than one similarity received in the global scorer.'
        score += scorer_similarities[0].score \
                 * scorer_global.feature_weight
    return score

  """
  It returns the union of similarities from all its similarity scorers,
  weighed by the scaling factor of each similarity scorer.
  """
  def GetSimilarity(self, source, target):
    global_score = self.GetGlobalScore(source, target)
    similarities = []
    for scorer in self.scorers:
      scorer_similarities = scorer.GetSimilarity(source, target)
      for scorer_similarity in scorer_similarities:
        scorer_similarity.score = scorer_similarity.score \
                                  * scorer.feature_weight \
                                  + global_score
        assert isinstance(scorer_similarity.score, float)
      similarities.extend(scorer_similarities)
    similarities = sorted(similarities, key=lambda x: x.score, reverse=False)
    return similarities

  def GetSimilar(self, source):
    global_score = 0.0
    similarities = []
    for scorer in self.scorers:
      scorer_similarities = scorer.GetSimilar(source)
      # Check that there are no repeated elements:
      assert len(scorer_similarities) == len(set(scorer_similarities))
      for scorer_similarity in scorer_similarities:
        src_treep, trg_treep = scorer_similarity.source, scorer_similarity.target
        global_score = self.GetGlobalScore(src_treep, trg_treep)
        scorer_similarity.score = scorer_similarity.score \
                                  * scorer.feature_weight \
                                  + global_score
      similarities.extend(scorer_similarities)
    return sorted(similarities, key=lambda x: x.score, reverse=True)

  def SetFeatureWeightsFromRules(self, rules):
    """
    In our applications, we use individual cost functions typically to find
    lexical relationships between source and target tree patterns in terminal
    rules. Those cost functions (that act as recognizers) can also be run
    in generation mode: given only the source side (or left-hand-side), find
    all possible target sides (or right-hand-sides).
    When estimating rule probabilities, we tie some of their parameters,
    currently according to their state.
    In this function, we use a list of rules where some of those rules have
    tied parameters, and use those rules to set the feature weight of the
    cost (probability) back-off functions (GetSimilar).
    If feature weights cannot be set for some cost functions, we set them
    to a default.
    """
    states_to_score = defaultdict(float)
    scorer_states = [scorer.relation for scorer in self.scorers]
    for rule in rules:
      if hasattr(rule, 'tied_to') and \
         rule.tied_to is not None and \
         rule.state in scorer_states:
        states_to_score[rule.state] = rule.weight
    for scorer in self.scorers:
      scorer.feature_weight = states_to_score.get(scorer.relation, self.kProb)

