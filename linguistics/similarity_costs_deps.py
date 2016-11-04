from nltk import Tree as NLTKTree

from linguistics.linguistic_tools import (LinguisticRelationship,
                                          ObtainLinguisticRelationships)
from linguistics.similarity import SimilarityScorer, Similarity
from utils.tree_tools import (get_top, tree_index, GetLeaves, variables_to_paths,
                              Tree, TreePattern, IsString, GetNodes)

class LexicalSimilarityDeps(SimilarityScorer):
  """
  Implements the abstract methods from Similarity class. It checks WordNet and
  VerbOcean for words that show any relationship, such as "synonym" relationship
  (WordNet) or "stronger-than" relationship (VerbOcean).
  """
  def __init__(self):
    self.kScore = 1.0
    self.feature_weight = 0.5

  def GetSimilarity(self, tree_pattern1, tree_pattern2):
    if not (isinstance(tree_pattern1, TreePattern) \
            and isinstance(tree_pattern2, TreePattern)):
      return []
    nodes1 = tree_pattern1.GetNodes()
    nodes2 = tree_pattern2.GetNodes()
    # At the moment, we do not allow gapped phrases. Only single tokens.
    if len(nodes1) != 1 or len(nodes2) != 1:
      return []
    phrase1 = '_'.join(nodes1)
    phrase2 = '_'.join(nodes2)
    linguistic_relationships = LinguisticRelationship(phrase1, phrase2)
    similarities = []
    for relation in linguistic_relationships:
      similarity = Similarity(self.kScore, relation, tree_pattern1, tree_pattern2)
      similarities.append(similarity)
    return similarities

  def GetSimilar(self, tree_pattern1):
    if not isinstance(tree_pattern1, TreePattern):
      return []
    nodes1 = tree_pattern1.GetNodes()
    phrase1 = '_'.join(nodes1)
    linguistic_relationships = ObtainLinguisticRelationships(phrase1)
    similarities = []
    for relation, lemma in linguistic_relationships:
      similarity = Similarity(self.kScore, relation, tree_pattern1, lemma)
      similarities.append(similarity)
    return similarities

class NoSimilarityDeps(SimilarityScorer):
  """
  Implements the abstract methods from Similarity class. It returns a list with
  a single element of 0.0 score with a None relationship between source and
  target. It is used to ensure that at least one transformation is produced
  between the source and the target, even if the transformation has a low score.
  """
  def __init__(self):
    self.kLinguisticVariation = 0.5
    self.kDeletionCost = 1.0
    self.kInsertionCost = 1.0
    self.kSubstitutionCost = 1.0
    self.feature_weight = 1.0

  def GetSimilarity(self, tree_pattern1, tree_pattern2):
    if tree_pattern1.IsString() and tree_pattern2.IsString():
      return[Similarity(self.kSubstitutionCost, None, tree_pattern1, tree_pattern2)]
    tree1_nodes = set(GetNodes(tree_pattern1))
    tree2_nodes = set(GetNodes(tree_pattern2))
    num_tree1_nodes = len(tree1_nodes)
    num_tree2_nodes = len(tree2_nodes)
    min_nodes = min(num_tree1_nodes, num_tree2_nodes)
    num_deleted_nodes = max(0, (num_tree1_nodes - num_tree2_nodes))
    num_inserted_nodes = max(0, (num_tree2_nodes - num_tree1_nodes))
    weight = min_nodes * self.kSubstitutionCost \
           + num_deleted_nodes * self.kDeletionCost \
           + num_inserted_nodes * self.kInsertionCost
    return [Similarity(weight, None, tree_pattern1, tree_pattern2)]

  def GetSimilar(self, word):
    return [Similarity(self.kScore, None, word, None)]

class LeafSimilarityDeps(SimilarityScorer):
  """
  Computes the optimistic cost between leaves from source and target trees.
  """
  def __init__(self):
    self.kDeletionCost = 1.0
    self.kInsertionCost = 1.0
    self.kSubstitutionCost = 1.0
    self.kLinguisticVariationCost = 0.5
    self.feature_weight = 1.0

  def GetSimilarity(self, tree_pattern1, tree_pattern2):
    assert len(tree_pattern1.subpaths) == len(tree_pattern2.subpaths), \
      'Number of subpaths from tree_pattern1 and tree_pattern2 differ: {0} vs. {1}'\
      .format(tree_pattern1, tree_pattern2)
    tree1_excluded_nodes = set(tree_pattern1.GetExcludedNodes())
    tree2_excluded_nodes = set(tree_pattern2.GetExcludedNodes())
    tree1_unique_nodes = tree1_excluded_nodes.difference(tree2_excluded_nodes)
    tree2_unique_nodes = tree2_excluded_nodes.difference(tree1_excluded_nodes)
    num_tree1_nodes = len(tree1_excluded_nodes)
    num_tree2_nodes = len(tree2_excluded_nodes)
    min_unique_nodes = min(len(tree1_unique_nodes), len(tree2_unique_nodes))
    num_deleted_nodes = max(0, (num_tree1_nodes - num_tree2_nodes))
    num_inserted_nodes = max(0, (num_tree2_nodes - num_tree1_nodes))
    weight = min_unique_nodes * self.kLinguisticVariationCost \
           + num_deleted_nodes * self.kDeletionCost \
           + num_inserted_nodes * self.kInsertionCost
    return [Similarity(weight, 'leaf_similarity', tree_pattern1, tree_pattern2)]

  def GetSimilar(self, tree):
    return []


