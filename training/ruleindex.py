#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
import logging

from nltk import Tree as NLTKTree

from linguistics.similarity import SimilarityScorerEnsemble
from semirings.semiring_prob import ProbSemiRing
from training.transductionrule import XTRule
from utils.tree_tools import (get_top, tree_index, IsVarTyped, IsString,
  variables_to_paths, LabelAndRank, TreeContains, TreePattern, GetPosAt,
  UntypeVars)

class RuleIndexT2T:
  def __init__(self, rules, rule_backoffs = None):
    self.index = self.MakeRuleIndex(rules)
    self.rules = rules
    # Similarity scorer used to retrieve unseen relevant rules for a source tree.
    self.rule_backoffs = rule_backoffs if rule_backoffs != None else []
    self.similarity_scorer = SimilarityScorerEnsemble(self.rule_backoffs, [])
    self.similarity_scorer.SetFeatureWeightsFromRules(self.rules)
    self.relevant_rules_cache = {}

  def MakeRuleIndex(self, rules):
    """
    Produces a dictionary indexed by the rule state, the POS of current
    non-terminal, and the POS of the children of the lhs. This index is also
    indexed by state, lhs label-and-rank, and rhs label-and-rank.
    """
    rules_index = defaultdict(list)
    for (i, rule) in enumerate(rules):
      lhs_pos, lhs_num_branches = LabelAndRank(rule.lhs)
      rhs_pos, rhs_num_branches = LabelAndRank(rule.rhs)
      rules_index[(rule.state, lhs_pos, lhs_num_branches)].append(i)
      rules_index[(rule.state, lhs_pos, lhs_num_branches,
                               rhs_pos, rhs_num_branches)].append(i)
      rule.index = i
    return rules_index

  def ClearCache(self):
    self.relevant_rules_cache = {}

  def GetRelevantRules(self, src_tree, src_path_state, \
                       trg_tree=None, trg_path=None, rhs_child_pos=None):
    """
    if trg_tree and trg_path is given, it returns rules that match the src_tree
    AND the trg_tree (at src_path and trg_path).
    """
    # Retrieving from cache, if result has already been computed.
    result_id = (src_path_state, rhs_child_pos) if trg_tree is None else \
                (src_path_state, trg_path, rhs_child_pos)
    if result_id in self.relevant_rules_cache:
      return self.relevant_rules_cache[result_id]
    src_path, state = src_path_state
    src_subtree = tree_index(src_tree, src_path)
    src_pos, src_num_branches = LabelAndRank(src_subtree)
    if trg_tree is None:
      trg_subtree, trg_pos, trg_num_branches = None, None, None
    else:
      trg_subtree = tree_index(trg_tree, trg_path)
      trg_pos, trg_num_branches = LabelAndRank(trg_subtree)
    rule_indices = self.GetCandidateRuleIndices(
      state, src_pos, src_num_branches, trg_pos, trg_num_branches)
    rules = [self.rules[i] for i in rule_indices]
    relevant_rules = FilterMatchingRules(
      rules, src_subtree, trg_subtree, rhs_child_pos)
    # If there are no relevant rules in our set, we search in other resources
    # e.g. WordNet for previously unseen (yet valid) rules for this tree pattern.
    if not relevant_rules and self.rule_backoffs:
      relevant_rules = \
        ProduceUnseenRelevantRules(state, src_subtree, self.similarity_scorer)
    relevant_rules = FilterMatchingRules(
      relevant_rules, src_subtree, trg_subtree, rhs_child_pos)
    # Storing result in cache.
    self.relevant_rules_cache[result_id] = relevant_rules
    return relevant_rules

  def GetCandidateRuleIndicesSource(self, state, src_pos, src_num_branches):
    """
    Rules that are only constrained by the LHS's state, root POS and
    number of branches.
    """
    candidate_rule_indices = []
    candidate_rule_indices.extend(
      self.index.get((state, src_pos, src_num_branches), []))
    candidate_rule_indices.extend(
      self.index.get((state, src_pos, None), []))
    candidate_rule_indices.extend(
      self.index.get((state, None, None), []))
    return set(candidate_rule_indices)

  def GetCandidateRuleIndices(self, state, src_pos, src_num_branches,
                              trg_pos, trg_num_branches):
    if trg_pos is None and trg_num_branches is None:
      return self.GetCandidateRuleIndicesSource(state, src_pos, src_num_branches)
    candidate_rule_indices = []
    # lhs and rhs have label and rank.
    candidate_rule_indices.extend(
      self.index.get(
        (state, src_pos, src_num_branches, trg_pos, trg_num_branches), []))
    # lhs have label and rank. rhs is a typed variable (thus, there is no rank).
    candidate_rule_indices.extend(
      self.index.get(
        (state, src_pos, src_num_branches, trg_pos, None), []))
    # lhs have label and rank. rhs is an untyped variable (no variable and no rank).
    candidate_rule_indices.extend(
      self.index.get(
        (state, src_pos, src_num_branches, None, None), []))
    candidate_rule_indices.extend(
      self.index.get(
        (state, src_pos, None, trg_pos, trg_num_branches), []))
    candidate_rule_indices.extend(
      self.index.get(
        (state, src_pos, None, trg_pos, None), []))
    candidate_rule_indices.extend(
      self.index.get(
        (state, src_pos, None, None, None), []))
    candidate_rule_indices.extend(
      self.index.get(
        (state, None, None, trg_pos, trg_num_branches), []))
    candidate_rule_indices.extend(
      self.index.get(
        (state, None, None, trg_pos, None), []))
    candidate_rule_indices.extend(
      self.index.get(
        (state, None, None, None, None), []))
    return set(candidate_rule_indices)

def MatchPos(pos1, pos2):
  if not pos1 or not pos2:
    return True
  return pos1 == pos2

def FilterMatchingRules(rules, src_subtree, trg_subtree, rhs_child_pos):
  if trg_subtree is None:
    relevant_rules = [rule for rule in rules \
                      if TreeContains(src_subtree, rule.lhs) \
                         and MatchPos(GetPosAt(rule.rhs, ()), rhs_child_pos)]
  else:
    relevant_rules = [rule for rule in rules \
                      if TreeContains(trg_subtree, rule.rhs)
                         and TreeContains(src_subtree, rule.lhs)]
  return relevant_rules

def ProduceUnseenRelevantRules(state, subtree, similarity_scorer):
  tree_pattern = TreePattern(subtree, (), [])
  # similarities = similarity_scorer.GetSimilar(tree_pattern)
  # Obtain similarities only for those scorers for which the state
  # matches (or is 'identity' or 'ling' relation).
  similarities = []
  for scorer in similarity_scorer.scorers:
    if scorer.relation in [state, 'identity', 'ling']:
      similarities.extend(scorer.GetSimilar(tree_pattern))
  relevant_rules = []
  for similarity in similarities:
    if similarity.relation == state or similarity.relation == 'identity':
      relevant_rule = MakeRule(similarity)
      relevant_rules.append(relevant_rule)
    if similarity.relation == 'identity':
      relevant_rule.state = state
      relevant_rule.newstates = MakeNewStates(state, relevant_rule)
      relevant_rule.rhs = UntypeVars(relevant_rule.rhs)
  return relevant_rules

def MakeRule(similarity, newstates = {}):
  source = similarity.source.ObtainTreePattern()
  target = similarity.target.ObtainTreePattern()
  # if not IsString(target) \
  #    or (IsString(target) and target.startswith('?x')):
  #   newstates = MakeNewStates(target, similarity.relation)
  rule = XTRule(similarity.relation, # state
                source, # left-hand-side
                target, # right-hand-side
                newstates, # If None, it will copy states from previous rule
                ProbSemiRing(similarity.score)) # Rule weight.
  return rule

def MakeNewStates(state, rule):
  vars_paths = variables_to_paths(rule.rhs)
  newstates = {path : state for var, path in vars_paths}
  return newstates

def MakeNewStates_(target, relation):
  newstates = {}
  vars_paths = variables_to_paths(target)
  for (var, path) in vars_paths:
    state_name = relation if IsVarTyped(var) else 't'
    newstates[path] = state_name
  return newstates

class RuleIndex:
  def __init__(self, rules):
    self.index = self.MakeRuleIndex(rules)
    self.rules = rules

  def MakeRuleIndex(self, rules):
    """
    Produces a dictionary indexed by the rule state,
    the POS of current non-terminal, and the POS of the children.
    """
    rules_index = defaultdict(list)
    for (i, rule) in enumerate(rules):
      if not isinstance(rule.lhs, NLTKTree):
        lhs_branches_pos = rule.lhs
        lhs_pos = rule.lhs
      else:
        lhs_branches_pos = ''
        for t in rule.lhs:
          pos = get_top(t).split('|')
          if len(pos) > 1 and pos[1] != '':
              lhs_branches_pos += ' ' + pos[1].strip()
          elif len(pos) == 1:
            lhs_branches_pos += ' ' + pos[0].strip()
        lhs_branches_pos = lhs_branches_pos.strip()
        lhs_pos = get_top(rule.lhs)
      rules_index[(rule.state, lhs_pos, lhs_branches_pos)].append(i)
    return rules_index

  def GetRelevantRules(self, tree, path_state):
    path, state = path_state
    subtree = tree_index(tree, path)
    relevant_rules = []
    if not isinstance(subtree, NLTKTree):
      tree_branches_pos = subtree
      tree_pos = subtree
    else:
      tree_branches_pos = \
        ' '.join([get_top(t) for t in subtree if isinstance(t, NLTKTree)])
      tree_pos = get_top(subtree)
    rules_indices = self.index[(state, tree_pos, tree_branches_pos)]
    relevant_rules = [self.rules[i] for i in rules_indices]
    return relevant_rules

