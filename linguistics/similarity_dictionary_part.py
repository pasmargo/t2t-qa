#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
from collections import defaultdict
import itertools
import re

from linguistics.similarity import SimilarityScorer, Similarity
from utils.tree_tools import GetLeaves, TreePattern, IsVariable

class DictionaryCostPart(SimilarityScorer):
  """
  Check a dictionary of the form source ||| target ||| cost.
  Source and target fields may have several tokens, separated by
  whitespaces. E.g.
  src_tok1 src_tok2 ||| trg_tok1 ||| cost.
  Leaves from source or target tree patterns may contain several
  tokens separated by a character. E.g. 'come_up'. The attributes
  self.src_token_separators and self.trg_token_separators can be
  overwritten to control how comparisons of source and target leaves
  are done with respect the dictionary (which is space separated).
  In this implementation, if *any* token of the source or
  the target appears in a dictionary entry, then the entry
  is selected (partial match).
  """
  def __init__(self, dictionary_filename, feature_weight = 1.0):
    self.feature_weight = feature_weight
    self.relation = 'dict_part'
    self.src_token_separators = '._ '
    self.trg_token_separators = '._ '
    self.trg_token_separator = '_'
    self.src_trg_cost = None
    self.src_index = None
    self.trg_index = None
    self.lowercase = True
    self.n_best = 5 # Number of target tree patterns at generation stage.
    self.BuildDictionary(dictionary_filename)
    self.src_words = None

  def get_src_index(self, query_word):
    return self.src_index.get(query_word, [])

  def BuildDictionary(self, dictionary_filename):
    """
    dictionary_filename points to a file with the following format:
    source ||| target ||| cost_of_transformation
    E.g.
    食べた ||| ate ||| 0.1
    ...
    食べた ||| eat ||| 0.2
    This method creates three dictionaries:
    self.src_trg_cost is a list of dictionary entries:
      [(source1, target1, cost1), ..., (sourceN, targetN, costN)]
    self.src_index and src.trg_index are dictionaries that map from source
    or target segments to dictionary entry indices.
    """
    finput = codecs.open(dictionary_filename, 'r', 'utf-8')
    entries = []
    for line in finput:
      if '|||' in line:
        entries.append([item.strip() for item in line.strip('\n ').split('|||')])
    finput.close()
    # Check that all lines have the same number of fields.
    assert not entries or len(set([len(entry) for entry in entries])) == 1
    # Check if cost field is present (which should be the last field).
    is_cost_present = IsCostPresent(entries)

    self.src_index = defaultdict(set)
    self.trg_index = defaultdict(set)
    self.src_trg_cost = []
    for i, entry in enumerate(entries):
      if not (1 < len(entry) <= 3):
        continue
      source, target = entry[0], entry[1]
      cost = float(entry[-1]) if is_cost_present else None
      self.src_trg_cost.append((source, target, cost))

      index_words = self.get_index_words(source, target)
      for word in index_words:
        self.src_index[word].add(i)
      target_words = self.get_target_words(target)
      for word in target_words:
        self.trg_index[word].add(i)

  def IsEligible(self, src_treep, trg_treep):
    """
    Returns True if the target tree pattern is a single leaf.
    """
    if trg_treep is not None:
      return trg_treep.IsString()
    return True

  def get_target_words(self, target):
    """
    @target: string of space separated tokens.
    """
    tokens = self.segment(target, self.trg_token_separators)
    tokens = self.normalize_tokens(tokens)
    return tokens

  def get_index_words(self, source, target):
    """
    @source: string of space separated tokens.
    @target: string of space separated tokens.
    """
    tokens = self.segment(source, self.src_token_separators)
    tokens = self.normalize_tokens(tokens)
    tokens = self.extend_tokens(tokens)
    tokens = self.filter_tokens(tokens)
    return tokens

  # TODO: when normalizing URIs, remove the "!" operator.
  def normalize_tokens(self, tokens):
    if self.lowercase:
      tokens = [token.lower() for token in tokens]
    return tokens

  def filter_tokens(self, tokens):
    return tokens

  def extend_tokens(self, tokens):
    return tokens

  def segment(self, phrase, token_separators):
    """
    @phrase: string of space separated tokens.
    """
    splitted = phrase.split()
    segments = SplitLeavesBy(splitted, token_separators)
    return segments

  def get_src_words_from_treep(self, src_treep):
    src_leaves = [leaf for leaf in GetLeaves(src_treep) if not IsVariable(leaf)]
    source = ' '.join(src_leaves)
    tokens = self.get_index_words(source, None)
    return tokens

  def get_trg_words_from_treep(self, trg_treep):
    trg_leaves = [leaf for leaf in GetLeaves(trg_treep) if not IsVariable(leaf)]
    target = ' '.join(trg_leaves)
    tokens = self.get_target_words(target)
    return tokens

  def get_src_indices_from_treep(self, src_treep):
    src_words = self.get_src_words_from_treep(src_treep)
    src_word_indices = [self.get_src_index(word) for word in src_words]
    src_indices = list(itertools.chain(*src_word_indices))
    return src_indices

  def get_trg_indices_from_treep(self, trg_treep):
    trg_words = self.get_trg_words_from_treep(trg_treep)
    trg_word_indices = [self.trg_index.get(word, []) for word in trg_words]
    trg_indices = list(itertools.chain(*trg_word_indices))
    return trg_indices

  def GetCostSimilarity(self, src_treep, trg_treep):
    """
    Computes the maximum cost among all entries that share at least one word
    with the source and the target.
    """
    src_indices = set(self.get_src_indices_from_treep(src_treep))
    trg_indices = set(self.get_trg_indices_from_treep(trg_treep))
    common_indices = src_indices.intersection(trg_indices)
    cost = None
    if not common_indices:
      return cost
    max_cost = 2.0 * max([self.src_trg_cost[i][2] for i in common_indices])
    return max_cost

  def GetScoreSimilarity(self, src_treep, trg_treep):
    """
    The probability associated to a partial match when checking only source
    is equivalent to the probability (complement to the cost) of the lexical
    entry in the dictionary.
    """
    src_indices = set(self.get_src_indices_from_treep(src_treep))
    trg_indices = set(self.get_trg_indices_from_treep(trg_treep))
    common_indices = src_indices.intersection(trg_indices)
    score = None
    if not common_indices:
      return score
    min_score = 1.0 - min([self.src_trg_cost[i][2] for i in common_indices])
    return min_score

  def GetSimilarity(self, src_treep, trg_treep):
    """
    The cost associated to a partial match when checking the source and the target sides
    is equivalent to the worst cost (max cost) among all entries where
    there is a partial match. The rationale is that this cost function
    should not have preference over the exact match implemented in
    DictionaryCost(). 
    """
    similarities = []
    if not self.IsEligible(src_treep, trg_treep):
      return similarities
    cost = self.GetCostSimilarity(src_treep, trg_treep)
    if cost is None:
      return similarities
    similarities = [Similarity(cost, self.relation, src_treep, trg_treep)]
    return similarities

  def MakeSimilar(self, src_treep, src_words, src_indices):
    similarities = []
    for i in src_indices:
      target = self.src_trg_cost[i][1].replace(' ', self.trg_token_separator)
      score = self.GetScoreSimilar(src_words, i)
      trg_treep = TreePattern(target, (), [])
      similarities.append(
        Similarity(score, self.relation, src_treep, trg_treep))
    # Remove duplicated elements (since this is not an exact match, they may occur).
    similarities = list(set(similarities))
    return similarities

  def BuildTrgTreePatterns(self, src_treep):
    src_words = self.get_src_words_from_treep(src_treep)
    src_word_indices = [self.get_src_index(word) for word in src_words]
    src_indices = set(itertools.chain(*src_word_indices))
    trg_treeps = []
    for i in src_indices:
      target = self.src_trg_cost[i][1].replace(' ', self.trg_token_separator)
      trg_treep = TreePattern(target, (), [])
      trg_treeps.append(trg_treep)
    return list(set(trg_treeps))

  def GetSimilar(self, src_treep):
    similarities = []
    if not self.IsEligible(src_treep, None):
      return similarities
    trg_treeps = self.BuildTrgTreePatterns(src_treep)
    for trg_treep in trg_treeps:
      score = self.GetScoreSimilarity(src_treep, trg_treep)
      if score is not None:
        similarity = Similarity(score, self.relation, src_treep, trg_treep)
        similarities.append(similarity)
    return sorted(similarities, key=lambda s: s.score, reverse=True)[:self.n_best]

def SplitLeavesBy(leaves, token_separators):
  tok_sep_pattern = r'[' + token_separators + ']'
  words = []
  for leaf in leaves:
    leaf_words = [word for word in re.split(tok_sep_pattern, leaf) if word != '']
    words.extend(leaf_words)
  return words

def IsCostPresent(entries):
  if not entries:
    return False
  possible_cost = entries[0][-1]
  try:
    float(possible_cost)
    return True
  except ValueError:
    return False

