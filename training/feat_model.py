#!/usr/bin/env python
# -*- coding: utf-8 -*-
import abc
from collections import defaultdict

class FeatModel(object):
  """ Model that assigns weights to rules given their features. """
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def save(self, filename):
    """ Saves model to filename. """
    pass

  @abc.abstractmethod
  def load(self, filename):
    """ Loads model from filename. """
    pass

  @abc.abstractmethod
  def weight_rule(self, rule):
    """
    Given a rule with attribute *features*:
      rule.features = [[feat_id, feat_val, ...], ...]
    it returns the weight of the rule according to the model.
    """
    pass

  def weight_wrtg(self, wrtg):
    """
    Weights all rules in a wRTG grammar.
    """
    # Clear caches because weights are going to change.
    # TODO: it might be possible to not clear the caches
    # if the weight doesn't change, and re-use previous decoding.
    wrtg.ClearCaches()
    for p in wrtg.P:
      rule = p.rhs.rule
      assert isinstance(rule.features, list)
      rule.weight = self.weight_rule(rule)

  @staticmethod
  def populate_wrtg_feats(wrtg, feat_inst=None):
    """
    Populate (instantiate) features of every rule of productions
    in wrtg.
    """
    for p in wrtg.P:
      rule = p.rhs.rule
      assert isinstance(rule.features, list) or feat_inst is not None
      populate_rule_feats(rule, feat_inst)

  @abc.abstractmethod
  def train(self, transducer, corpus, feat_inst=None):
    """
    Estimates parameters of the model given a transducer and a corpus
    of parallel trees.
    """
    pass

def populate_rule_feats(rule, feat_inst=None):
  """
  If rule has not features associated but there is a feature instantiator,
  then create the feature vector.
  """
  # if isinstance(rule.features, list) and rule.features:
  #   print('************** Features already populated ***************')
  if feat_inst is not None and not isinstance(rule.features, list):
    src_treep, trg_treep = rule.GetTreePatterns()
    rule.features = feat_inst.InstantiateFeatures(src_treep, trg_treep, rule)

# TODO: Implement a mechanism to extract non-local features from a derivation,
# given the source and the target tree
# (extraction.feat_instantiator.ExtractFeatsFromDerivation).
# It would require:
#   1. Convert list of rules in a list of tree patterns (as implemented in utils).
#   2. Instantiate features, and possibly obtain new instances. This would require
#      to extend the vector of feature weights dynamically.
def CollectFeats(derivation):
  """
  This function simply visits each rule of of the derivation (list of productions),
  and adds their features (if a feature appears in more than one rule, then their
  values are added).
  """
  feats_dict = defaultdict(float)
  for production in derivation:
    rule = production.rhs.rule
    assert isinstance(rule.features, list)
    for features in rule.features:
      feat_id, feat_val = features[0:2]
      feats_dict[feat_id] += feat_val
  return list(feats_dict.items())

