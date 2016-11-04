#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
import logging
import re
from subprocess import Popen, PIPE

from linguistics.similarity import SimilarityScorer, Similarity
from utils.tree_tools import (get_top, tree_index, GetLeaves, variables_to_paths,
                              Tree, TreePattern, IsString)

# In the task of extracting transformation rules for semantic trees,
# we need to consider (at least) the following features:
# 1. Bilingual lexical similarity between leaves, possibly using
#    a distributed similarity cost or a dictionary (as computed by IBM Models).
# 2. Variable similarity. Variables are identifiers of the form [a-zA-Z]-[0-9].
#    The type of the variable seems to be related to the first character of the
#    variable ID. E.g. event variables e-1, e-2, other variables x-1, x-2.
# 3. Entity similarity. Some entities are preceeded by an '@' mark, indicating
#    that they will have a "literal" translation. Such translation could be
#    found by transliteration or similar techniques, or naive dictionary lookups.
#    The type of the variable seems to be related to the first character of the
#    variable ID. E.g. event variables e-1, e-2, other variables x-1, x-2.
# 4. Inner node similarity that checks the set difference (or bag of words)
#    between the inner nodes (excluding leaves) from source and target.
#    Depending on how this is implemented, it could also work as a regularizer.
# 5. Regularizer by tree complexity. The cost needs to be linear with the size
#    of the tree patterns. In this way, it does not matter from where the
#    cost is coming, as long as the complexity preserves.

class TreeSize(SimilarityScorer):
  """
  Implements the abstract methods from Similarity class. It measures the
  sum of absolute size of both tree patterns. In a non-leaf tree pattern,
  the first node does not count towards the size. E.g. (NP ?x0|DT ?x1|NN)
  has size 0.
  """
  def __init__(self, feature_weight = 0.5):
    self.feature_weight = feature_weight

  def PreCache(self, src_tree, trg_tree, options = None):
    pass

  def GetSimilarity(self, tree_pattern1, tree_pattern2):
    src_nodes = tree_pattern1.GetInnerNodes() 
    trg_nodes = tree_pattern2.GetInnerNodes()
    src_leaves = GetLeaves(tree_pattern1)
    trg_leaves = GetLeaves(tree_pattern2)
    src_length = len(src_nodes) + len(src_leaves)
    trg_length = len(trg_nodes) + len(trg_leaves)
    cost = max(0, src_length) ** 2 + max(0, trg_length) ** 2
    similarities = [Similarity(cost, 'tree_size', tree_pattern1, tree_pattern2)]
    return similarities

  def GetSimilar(self, tree_pattern1):
    return []

class TreeDifferenceComplexity(SimilarityScorer):
  """
  Implements the abstract methods from Similarity class. It measures the
  difference between the number of nodes from source and target that are
  not variables.
  """
  def __init__(self, feature_weight = 0.1):
    self.feature_weight = feature_weight

  def PreCache(self, src_tree, trg_tree, options = None):
    pass

  def GetSimilarity(self, tree_pattern1, tree_pattern2):
    src_nodes = tree_pattern1.GetInnerNodes() 
    trg_nodes = tree_pattern2.GetInnerNodes()
    cost = abs(len(src_nodes) - len(trg_nodes)) ** 2
    similarities = [Similarity(cost, 'tree_comp', tree_pattern1, tree_pattern2)]
    return similarities

  def GetSimilar(self, tree_pattern1):
    return []

class EntityDifference(SimilarityScorer):
  """
  Implements the abstract methods from Similarity class. It measures the
  difference between the number of nodes from source and target that are
  entities. E.g. (:arg1 @Sugita) <-> (:arg1 @Sugita).
  """
  def __init__(self, feature_weight = 0.5):
    self.feature_weight = feature_weight

  def PreCache(self, src_tree, trg_tree, options = None):
    pass

  def GetSimilarity(self, tree_pattern1, tree_pattern2):
    src_nodes = tree_pattern1.GetLeaves()
    trg_nodes = tree_pattern2.GetLeaves()
    src_vars = [v for v in src_nodes if v.startswith('@')]
    trg_vars = [v for v in trg_nodes if v.startswith('@')]
    # Bag of words for variable types. Variable types are the first character
    # of a variable. E.g. e-3 has type 'e'. We ignore variable numbers,
    # as their difference between source and target is not meaningful.
    cost = abs(len(src_vars) - len(trg_vars))
    similarities = [Similarity(cost, 'entity_diff', tree_pattern1, tree_pattern2)]
    return similarities

  def GetSimilar(self, tree_pattern1):
    return []

class EntityDifferenceIndividual(SimilarityScorer):
  """
  Implements the abstract methods from Similarity class. It signals
  with a low cost when two tree patterns consist of a single entity each. 
  E.g. @杉田 <-> @Sugita.
  The generation GetSimilar of this cost function needs a third-party
  software. You can install it doing:
  sudo apt-get install kakasi
  """
  def __init__(self, feature_weight = 0.5):
    self.feature_weight = feature_weight
    self.relation = 'entity_copy'
    self.trg_token_separator = '_'
    self.kProb = 0.863259

  def PreCache(self, src_tree, trg_tree, options = None):
    pass

  def GetSimilarity(self, tree_pattern1, tree_pattern2):
    similarities = []
    if not (tree_pattern1.IsString() and tree_pattern2.IsString()):
      return similarities
    src_leaves = tree_pattern1.GetLeaves()
    trg_leaves = tree_pattern2.GetLeaves()
    if len(src_leaves) != 1 or len(trg_leaves) != 1:
      return similarities
    src_str, trg_str = src_leaves[0], trg_leaves[0]
    if src_str.startswith('@') and trg_str.startswith('@'):
      cost = 0.0
      similarities = [Similarity(cost, self.relation, tree_pattern1, tree_pattern2)]
    return similarities

  def GetSimilar(self, tree_pattern1):
    similarities = []
    if not tree_pattern1.IsString():
      return similarities
    src_leaves = tree_pattern1.GetLeaves()
    if len(src_leaves) != 1:
      return similarities
    src_str = src_leaves[0]
    if src_str.startswith('@'):
      # Remove the @ symbol, transliterate the rest of the string,
      # and add the target token separator.
      trg_str = TransliterateJaEn(src_str[1:]).title()
      trg_str = trg_str.replace(' ', self.trg_token_separator)
      # Add the @ to the transliterated string and title-cased it.
      tree_pattern2 = TreePattern('@{0}'.format(trg_str), (), [])
      similarities = [Similarity(self.kProb, self.relation, tree_pattern1, tree_pattern2)]
    return similarities

def TransliterateJaEn(str_ja):
  """
  This function makes a call to kakasi to transliterate Japanese kanji or kana
  into English. From the shell, it is accomplished as:
  echo "日本が好きです。" | \
    iconv -f utf8 -t eucjp | \
    kakasi -i euc -w | \
    kakasi -i euc -Ha -Ka -Ja -Ea -ka
  it returns 'nippon ga suki desu .'
  """
  str_en = Popen((u'echo {0} | '
                  'iconv -f utf8 -t eucjp | '
                  'kakasi -i euc -w | '
                  'kakasi -i euc -Ha -Ka -Ja -Ea -ka').format(str_ja),
                 shell=True,
                 stdout=PIPE).stdout.read().strip()
  return str_en

class VariableDifference(SimilarityScorer):
  """
  Implements the abstract methods from Similarity class. It measures the
  difference between the number of nodes from source and target that are
  not variables.
  """
  def __init__(self, feature_weight = 0.5):
    self.feature_weight = feature_weight
    self.is_var = re.compile(r'[a-zA-Z]-[0-9]')

  def PreCache(self, src_tree, trg_tree, options = None):
    pass

  def GetSimilarity(self, tree_pattern1, tree_pattern2):
    IsVariable = self.is_var.match
    src_nodes = tree_pattern1.GetLeaves()
    trg_nodes = tree_pattern2.GetLeaves()
    src_vars = [v.lower() for v in src_nodes if IsVariable(v)]
    trg_vars = [v.lower() for v in trg_nodes if IsVariable(v)]
    # Bag of words for variable types. Variable types are the first character
    # of a variable. E.g. e-3 has type 'e'. We ignore variable numbers,
    # as their difference between source and target is not meaningful.
    bow = defaultdict(float)
    for src_var in src_vars:
      src_var_type = src_var[0]
      bow[src_var_type] += 1.0
    for trg_var in trg_vars:
      trg_var_type = trg_var[0]
      bow[trg_var_type] -= 1.0
    num_src_trg_vars = len(src_vars) + len(trg_vars)
    if num_src_trg_vars == 0:
      cost = 0.0
    else:
      cost = sum([abs(count) for count in bow.values()])
    relation = 'var_diff'
    similarities = [Similarity(cost, relation, tree_pattern1, tree_pattern2)]
    return similarities

  def GetSimilar(self, tree_pattern1):
    return []

class VariableDifferenceIndividual(SimilarityScorer):
  """
  Implements the abstract methods from Similarity class.
  This cost function only applies when the source and target
  tree patterns are only strings. It signals (with a low cost)
  whether the source string and the target string are both variables.
  """
  def __init__(self, feature_weight = 0.5):
    self.feature_weight = feature_weight
    self.is_var = re.compile(r'[a-zA-Z]-[0-9]')
    self.relation = 'var_copy'
    self.kProb = 0.274412

  def PreCache(self, src_tree, trg_tree, options = None):
    pass

  def GetSimilarity(self, tree_pattern1, tree_pattern2):
    IsVariable = self.is_var.match
    similarities = []
    if not (tree_pattern1.IsString() and tree_pattern2.IsString()):
      return similarities
    src_leaves = tree_pattern1.GetLeaves()
    trg_leaves = tree_pattern2.GetLeaves()
    if len(src_leaves) != 1 or len(trg_leaves) != 1:
      return similarities
    if IsVariable(src_leaves[0]) and IsVariable(trg_leaves[0]):
      cost = 0.0
      similarities = [Similarity(cost, self.relation, tree_pattern1, tree_pattern2)]
    return similarities

  def GetSimilar(self, tree_pattern1):
    IsVariable = self.is_var.match
    similarities = []
    if not tree_pattern1.IsString():
      return similarities
    src_leaves = tree_pattern1.GetLeaves()
    if len(src_leaves) != 1:
      return similarities
    src_str = src_leaves[0]
    if IsVariable(src_str):
      tree_pattern2 = TreePattern(src_str, (), [])
      similarities = [Similarity(self.kProb, self.relation, tree_pattern1, tree_pattern2)]
    return similarities

class InnerNodesDifference(SimilarityScorer):
  """
  Implements the abstract methods from Similarity class. It measures the
  number of inner nodes that are different between the source and target
  tree patterns. It uses a bag of words where every node identity maps
  to a counter of the token's occurrences.
  """
  def __init__(self, feature_weight = 2.0):
    self.feature_weight = feature_weight
    self.max_phrase_length = 4

  def PreCache(self, src_tree, trg_tree, options = None):
    pass

  def GetSimilarity(self, tree_pattern1, tree_pattern2):
    src_nodes = tree_pattern1.GetInnerNodes() 
    trg_nodes = tree_pattern2.GetInnerNodes()
    # Get bag of words of differences.
    bow = defaultdict(float)
    for src_node in src_nodes:
      bow[src_node] += 1.0
    for trg_node in trg_nodes:
      bow[trg_node] -= 1.0
    num_src_trg_nodes = len(src_nodes) + len(trg_nodes)
    if num_src_trg_nodes == 0:
      cost = 0.0
    else:
      cost = sum([abs(count) for count in bow.values()])
    similarities = [Similarity(cost, 'inner_node_diff', tree_pattern1, tree_pattern2)]
    return similarities

  def GetSimilar(self, tree_pattern1):
    tree1_leaves = GetLeaves(tree_pattern1)
    num_tree1_leaves = len(tree1_leaves)
    if len(tree1_leaves) > self.max_phrase_length or not tree1_leaves:
      cost = num_tree1_leaves
      similarities = [Similarity(cost, 'q0', tree_pattern1, tree_pattern2)]
    else:
      entities = self.GetLexicon(tree1_leaves, 'entity')
      unary_predicates = self.GetLexicon(tree1_leaves, 'unary')
      binary_predicates = self.GetLexicon(tree1_leaves, 'binary')
      lexicon = entities + unary_predicates + binary_predicates
      similarities = []
      cost = 0.0
      for lex, lex_type in lexicon:
        path, subpaths = (), []
        tree_pattern2 = TreePattern(lex, path, subpaths)
        similarity = Similarity(cost, lex_type, tree_pattern1, tree_pattern2)
        similarities.append(similarity)
    return similarities

