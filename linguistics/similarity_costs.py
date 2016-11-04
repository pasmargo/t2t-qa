import math

from nltk import Tree as NLTKTree

from linguistics.linguistic_tools import (LinguisticRelationship,
                                          ObtainLinguisticRelationships)
from linguistics.similarity import Similarity, SimilarityScorer
from utils.tree_tools import (get_top, tree_index, GetLeaves, variables_to_paths,
                              Tree, TreePattern, IsString)

class NoSimilarity(SimilarityScorer):
  """
  Implements the abstract methods from Similarity class. It returns a list with
  a single element of 0.0 score with a None relationship between source and
  target. It is used to ensure that at least one transformation is produced
  between the source and the target, even if the transformation has a low score.
  """
  def __init__(self, feature_weight = 1.0):
    self.feature_weight = feature_weight
    self.kLinguisticVariation = 0.5
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
    min_leaves = min(num_tree1_leaves, num_tree2_leaves)
    num_deleted_leaves = max(0, (num_tree1_leaves - num_tree2_leaves))
    num_inserted_leaves = max(0, (num_tree2_leaves - num_tree1_leaves))
    weight = min_leaves * self.kSubstitutionCost \
           + num_deleted_leaves * self.kDeletionCost \
           + num_inserted_leaves * self.kInsertionCost
    return [Similarity(weight, None, tree_pattern1, tree_pattern2)]

  def GetSimilar(self, word):
    return [Similarity(self.kScore, None, word, None)]

class InsertionDeletionCost(SimilarityScorer):
  """
  Implements the abstract methods from Similarity class. It counts the leaves
  in the source tree and multiplies them by a deletion cost. Then, it counts
  the leaves in the target tree and multiplies them by an insertion cost.
  Finally, it adds up the insertion and deletion costs.
  """
  def __init__(self, feature_weight = 1.0):
    self.feature_weight = feature_weight
    self.kDeletionCost = 1.0
    self.kInsertionCost = 1.0

  def GetSimilarity(self, tree_pattern1, tree_pattern2):
    tree1_leaves = set(GetLeaves(tree_pattern1))
    tree2_leaves = set(GetLeaves(tree_pattern2))
    weight = len(tree1_leaves) * self.kDeletionCost \
           + len(tree2_leaves) * self.kInsertionCost
    return [Similarity(weight, None, tree_pattern1, tree_pattern2)]

  def GetSimilar(self, word):
    return []

class LexicalSimilarity(SimilarityScorer):
  """
  Implements the abstract methods from Similarity class. It checks WordNet and
  VerbOcean for words that show any relationship, such as "synonym" relationship
  (WordNet) or "stronger-than" relationship (VerbOcean).
  """
  def __init__(self, feature_weight = 0.1):
    self.feature_weight = feature_weight
    self.kLinguisticVariation = 0.0
    self.kDeletionCost = 1.0
    self.kInsertionCost = 1.0
    self.kSubstitutionCost = 1.0
    self.relation = 'ling'

  def GetSimilarity(self, tree_pattern1, tree_pattern2):
    if not (tree_pattern1.IsString() and tree_pattern2.IsString()):
      """
      if tree_pattern1 == tree_pattern2:
        return [Similarity(self.kLinguisticVariation, 'copy',
                           tree_pattern1, tree_pattern2)]
      """
      return []
    tree1_leaves = set(GetLeaves(tree_pattern1))
    tree2_leaves = set(GetLeaves(tree_pattern2))
    phrase1 = '_'.join(tree1_leaves)
    phrase2 = '_'.join(tree2_leaves)
    linguistic_relationships = LinguisticRelationship(phrase1, phrase2)
    similarities = []
    for relation in linguistic_relationships:
      similarity = Similarity(self.kLinguisticVariation, relation,
                              tree_pattern1, tree_pattern2)
      similarities.append(similarity)
    return similarities

  def GetSimilar(self, src_pattern):
    # TODO: If we want the ability to compute linguistic similarities between
    # tree patterns, we could concatenate the yield (except variables on leaves)
    # to produce (possibly gapped) phrases.
    if not src_pattern.IsString():
      return []
    phrase = ' '.join([leaf for leaf in src_pattern.GetLeaves()])
    linguistic_relationships = ObtainLinguisticRelationships(phrase)
    similarities = []
    for relation, lemma in linguistic_relationships:
      trg_pattern = TreePattern(lemma, (), [])
      similarity = Similarity(
        self.kLinguisticVariation, relation, src_pattern, trg_pattern)
      similarities.append(similarity)
    return similarities

class LeafSimilarity(SimilarityScorer):
  """
  Computes the optimistic cost between leaves from source and target trees.
  """
  def __init__(self, feature_weight = 1.0):
    self.feature_weight = feature_weight
    self.kDeletionCost = 1.0
    self.kInsertionCost = 1.0
    self.kSubstitutionCost = 1.0
    self.kLinguisticVariationCost = 0.5

  def GetSimilarity(self, tree_pattern1, tree_pattern2):
    assert len(tree_pattern1.subpaths) == len(tree_pattern2.subpaths), \
      'Number of subpaths from tree_pattern1 and tree_pattern2 differ: {0} vs. {1}'\
      .format(tree_pattern1, tree_pattern2)
    tree1_excluded_leaves = set(tree_pattern1.GetExcludedLeaves())
    tree2_excluded_leaves = set(tree_pattern2.GetExcludedLeaves())
    tree1_unique_leaves = tree1_excluded_leaves.difference(tree2_excluded_leaves)
    tree2_unique_leaves = tree2_excluded_leaves.difference(tree1_excluded_leaves)
    num_tree1_leaves = len(tree1_excluded_leaves)
    num_tree2_leaves = len(tree2_excluded_leaves)
    min_unique_leaves = min(len(tree1_unique_leaves), len(tree2_unique_leaves))
    num_deleted_leaves = max(0, (num_tree1_leaves - num_tree2_leaves))
    num_inserted_leaves = max(0, (num_tree2_leaves - num_tree1_leaves))
    weight = min_unique_leaves * self.kLinguisticVariationCost \
           + num_deleted_leaves * self.kDeletionCost \
           + num_inserted_leaves * self.kInsertionCost
    return [Similarity(weight, 'leaf_similarity', tree_pattern1, tree_pattern2)]
 
  def GetSimilar(self, tree):
    return []

class NodesDifference(SimilarityScorer):
  """
  Counts the (normalized) absolute number difference of inner nodes between
  tree1 and tree2. The intention is to penalize tree patterns with uneven
  complexity or size.
  """
  def __init__(self, feature_weight = 0.1):
    self.feature_weight = feature_weight

  def GetSimilarity(self, tree1, tree2):
    num_nodes_tree1 = 0 if IsString(tree1) \
                        else tree1.GetNumSubtrees()
    num_nodes_tree2 = 0 if IsString(tree2) \
                        else tree2.GetNumSubtrees()
    weight = 0.0
    if not (num_nodes_tree1 == 0 and num_nodes_tree2 == 0):
      weight = (float(abs(num_nodes_tree1 - num_nodes_tree2)) \
                / max(num_nodes_tree1, num_nodes_tree2))
    return [Similarity(weight, 'nodes_difference', tree1, tree2)]
 
  def GetSimilar(self, tree):
    return []

class TreeComplexity(SimilarityScorer):
  """
  Penalizes (assigns low score) to trees that are big (rule complexity).
  """
  def __init__(self, feature_weight = 1.0):
    self.feature_weight = feature_weight
    self.normalizer = 100.0

  def GetSimilarity(self, tree_pattern1, tree_pattern2):
    # When computing the number of subtrees to measure the tree complexity,
    # we do not penalize trees (lhs or rhs) that have only one inner node,
    # as in the case of pre-terminals or transductions that consume only
    # one inner node.
    num_nodes_tree1 = max(0, tree_pattern1.GetNumNodes() - 1)
    num_nodes_tree2 = max(0, tree_pattern2.GetNumNodes() - 1)
    weight = min(1, (num_nodes_tree1 + num_nodes_tree2)**2 / self.normalizer)
    return [Similarity(weight, 'nodes_difference', tree_pattern1, tree_pattern2)]
 
  def GetSimilar(self, tree):
    return []

