# -*- coding: utf-8 -*-
import codecs
from collections import defaultdict

import numpy as np

from linguistics.similarity import SimilarityScorer, Similarity
from utils.tree_tools import tree_or_string, IsString

class AlignmentCost(SimilarityScorer):
  """
  Implements the abstract methods from Similarity class. 
  It returns a list with a single Similarity element. Such similarity
  element has relation q and cost equal to Infinite if there is
  an alignment violation. The cost is equal to 0 otherwise.
  """
  def __init__(self, alignment_fname, feature_weight):
    self.feature_weight = feature_weight
    self.kCost = 0.0
    self.kCostViolation = np.inf
    self.relation = 'q'
    self.alignments = LoadAlignments(alignment_fname)

  def GetSimilarity(self, src_treep, trg_treep):
    alignment = self.alignments.get(
      (str(src_treep.tree), str(trg_treep.tree)), None)
    assert alignment is not None
    src_leaves_inds = src_treep.GetLeavesIndices()
    trg_leaves_inds = trg_treep.GetLeavesIndices()
    src_to_trg_inds = alignment.get_trg_inds(src_leaves_inds)
    trg_to_src_inds = alignment.get_src_inds(trg_leaves_inds)
    cost = self.kCost
    if IsAlignmentViolated(src_to_trg_inds, trg_leaves_inds):
      cost = self.kCostViolation
    if IsAlignmentViolated(trg_to_src_inds, src_leaves_inds):
      cost = self.kCostViolation
    return [Similarity(cost, self.relation, src_treep, trg_treep)]

  def GetSimilar(self, src_treep):
    raise ValueError('Not implemented')

def IsAlignmentViolated(aligned_inds, range_inds):
  """
  Returns true if the aligned indices aligned_inds span
  indices beyond the range of indices range_inds. E.g.
  IsAlignmentedViolated([1,2], [0,2,3]) returns False.
  IsAlignmentedViolated([1,4], [0,2,3]) returns True.
  """
  if aligned_inds:
    if not range_inds:
      return True
    elif aligned_inds[0] < range_inds[0]:
      return True
    elif aligned_inds[-1] > range_inds[-1]:
      return True
  return False

class Alignment(object):
  """
  Implements some operations and structure of aligned sentences.
  """

  def __init__(self, alignment_str, src_words, trg_words):
    self.src = src_words
    self.trg = trg_words
    self.alignment = parse_alignment(alignment_str)
    self.inverted = invert_alignments(self.alignment)

  def get_trg_inds(self, src_inds):
    """
    Returns a list of target word indices.
    """
    trg_inds = []
    for i in src_inds:
      trg_inds.extend(self.alignment[i])
    return trg_inds

  def get_src_inds(self, trg_inds):
    """
    Returns a list of source word indices.
    """
    src_inds = []
    for i in trg_inds:
      src_inds.extend(self.inverted[i])
    return src_inds

def invert_alignments(alignment):
  """
  @alignment is a dictionary that maps a source word index
  into a list of target word indices.
  This function returns the inverted index.
  """
  inverted = defaultdict(list)
  for src_i, trg_is in alignment.items():
    for trg_i in trg_is:
      inverted[trg_i].append(src_i)
  for trg_i, src_is in inverted.items():
    inverted[trg_i] = sorted(src_is)
  return inverted

def parse_alignment(alignment_str):
  """
  Parses an alignment string (e.g. "0-0 0-1 1-0 2-2 3-4")
  into a dictionary.
  """
  alignment = defaultdict(list)
  als = alignment_str.split()
  for al in als:
    src_ind, trg_ind = map(int, al.split('-'))
    alignment[src_ind].append(trg_ind)
  for src_i, trg_is in alignment.items():
    alignment[src_i] = sorted(trg_is)
  return alignment

def LoadAlignments(alignment_fname):
  """
  Load a filename with the following structure:
    src_tree
    trg_tree
    alignment
    ...
    src_tree
    trg_tree
    alignment
  into a dictionary indexed by a tuple (src_tree_str, trg_tree_str),
  whose values are Alignment objects.
  """
  alignments = {}
  with codecs.open(alignment_fname, 'r', 'utf-8') as fin:
    lines = fin.readlines()
    assert len(lines) % 3 == 0, 'Lines in {0} are not a multiple of 3.'.format(
      alignment_fname)
    for i, line in enumerate(lines):
      if i % 3 == 0:
        src_tree_str = line.strip()
        src_tree = tree_or_string(src_tree_str)
        src_leaves = src_tree.leaves() if not IsString(src_tree) else [src_tree]
      if i % 3 == 1:
        trg_tree_str = line.strip()
        trg_tree = tree_or_string(trg_tree_str)
        trg_leaves = trg_tree.leaves() if not IsString(trg_tree) else [trg_tree]
      if i % 3 == 2:
        alignment_str = line.strip()
        alignment = Alignment(alignment_str, src_leaves, trg_leaves)
        alignments[(src_tree_str, trg_tree_str)] = alignment
  return alignments
