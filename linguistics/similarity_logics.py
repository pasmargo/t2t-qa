

# -*- coding: utf-8 -*-
import numpy as np

from linguistics.similarity import SimilarityScorer, Similarity

class StringRulesInfiniteCost(SimilarityScorer):
  """
  Implements the abstract methods from Similarity class. 
  It returns a list with a single Similarity element. Such similarity
  element has relation q and cost equal to infinity if the
  src_treep or trg_treep are strings, and zero otherwise.
  This is to prevent rules with strings to be extracted
  when working with CCG trees, since such rules do not have
  meaning (meaning is constructed only at nodes with syntactic categories).
  """
  def __init__(self, feature_weight=1.0, side="both"):
    self.feature_weight = feature_weight
    self.side = side
    self.high_cost = 1.0
    self.low_cost = 0.0
    self.relation = 'q'

  def GetSimilarity(self, src_treep, trg_treep):
    cost = self.low_cost
    if self.side == "both" and (src_treep.IsString() or trg_treep.IsString()):
      cost = self.high_cost
    if self.side == "source" and src_treep.IsString():
      cost = self.high_cost
    if self.side == "target" and trg_treep.IsString():
      cost = self.high_cost
    similarities = [Similarity(cost, self.relation, src_treep, trg_treep)]
    return similarities

  def GetSimilar(self, src_treep):
    raise ValueError('Not implemented')
    return [Similarity(self.high_cost, None, src_treep, None)]

