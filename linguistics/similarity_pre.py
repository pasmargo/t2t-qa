import codecs
from copy import deepcopy

from linguistics.similarity import SimilarityScorer, Similarity

from utils.tree_tools import (get_top, tree_index, GetLeaves, variables_to_paths,
  Tree, TreePattern, IsString, GetChildrenPaths)

class DictionaryCost(SimilarityScorer):
  """
  Implements the abstract methods from Similarity class. It checks WordNet and
  VerbOcean for words that show any relationship, such as "synonym" relationship
  (WordNet) or "stronger-than" relationship (VerbOcean).
  """
  def __init__(self, dictionary_filename, feature_weight = 1.0):
    self.feature_weight = feature_weight
    self.dictionary = self.BuildDictionary(dictionary_filename)

  def BuildDictionary(self, dictionary_filename):
    dictionary = {}
    finput = codecs.open(dictionary_filename, 'r', 'utf-8')
    for line in finput:
      (source, target, cost) = \
        [item.strip() for item in line.strip('\n ').split('|||')]
      dictionary[(source, target)] = float(cost)
    return dictionary

  def GetSimilarity(self, tree_pattern1, tree_pattern2):
    if not (tree_pattern1.IsString() and tree_pattern2.IsString()):
      return []
    tree1_leaves = GetLeaves(tree_pattern1)
    tree2_leaves = GetLeaves(tree_pattern2)
    phrase1 = '_'.join(tree1_leaves).lower()
    phrase2 = '_'.join(tree2_leaves).lower()
    if (phrase1, phrase2) in self.dictionary:
      cost = self.dictionary[(phrase1, phrase2)]
      return [Similarity(cost, 'dictionary', tree_pattern1, tree_pattern2)]
    return []

  def GetSimilar(self, word):
    return []

class NoSimilarityPre(SimilarityScorer):
  """
  Implements the abstract methods from Similarity class. It returns a list with
  a single element of 0.0 score with a None relationship between source and
  target. It is used to ensure that at least one transformation is produced
  between the source and the target, even if the transformation has a low score.
  """
  def __init__(self, feature_weight = 1.0):
    self.feature_weight = feature_weight
    self.kDeletionCost = 1.0
    self.kInsertionCost = 1.0
    self.kSubstitutionCost = 1.0

  def GetSimilarity(self, tree_pattern1, tree_pattern2):
    if tree_pattern1.IsString() and tree_pattern2.IsString():
      return[Similarity(self.kSubstitutionCost, None, tree_pattern1, tree_pattern2)]
    tree1_leaves = set(GetLeaves(tree_pattern1))
    tree2_leaves = set(GetLeaves(tree_pattern2))
    num_tree1_leaves = len(tree1_leaves)
    num_tree2_leaves = len(tree2_leaves)
    weight = num_tree1_leaves * self.kDeletionCost \
           + num_tree2_leaves * self.kInsertionCost
    return [Similarity(weight, None, tree_pattern1, tree_pattern2)]

  def GetSimilar(self, word):
    return [Similarity(self.kScore, None, word, None)]

class Identity(SimilarityScorer):
  """
  Implements the abstract methods from Similarity class. It returns a list with
  a single element of 0.0 score with a None relationship between source and
  target. It is used to ensure that at least one transformation is produced
  between the source and the target, even if the transformation has a low score.
  """
  def __init__(self, feature_weight = 1.0):
    self.feature_weight = feature_weight
    self.kCost = 1e-300
    self.kDefaultState = 'identity'
    self.relation = self.kDefaultState

  def GetSimilarity(self, tree1, tree2):
    similarities = []
    tree1_str = tree1 if IsString(tree1) else repr(tree1)
    tree2_str = tree2 if IsString(tree2) else repr(tree2)
    if tree1_str == tree2_str:
      similarities = [Similarity(self.kCost, self.kDefaultState, tree1, tree2)]
    return similarities

  def GetSimilar(self, src_pattern):
    """
    From a source tree pattern that contains only a subtree, such as:
    (NP (DT the) (NN house)), produce a naive tree pattern where its immediate
    children are substituted by variables:
    (NP ?x0|DT ?x1|NN).
    """
    tree = deepcopy(src_pattern.tree)
    path = src_pattern.path
    # Limit variables to immediate children.
    subpaths = GetChildrenPaths(tree, path, max_depth=1)
    trg_tree_pattern = TreePattern(tree, path, subpaths)
    src_tree_pattern = TreePattern(tree, path, subpaths)
    state = self.kDefaultState
    return [Similarity(self.kCost, state, src_tree_pattern, trg_tree_pattern)]

