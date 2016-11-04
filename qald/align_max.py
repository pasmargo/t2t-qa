#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import codecs
from collections import defaultdict
import itertools
import json
import os
import sys

from linguistics.similarity_qa import CountOp
from utils.corpus import LoadCorpus
from utils.tree_tools import tree_or_string, TreePattern, IsString

def load_lexicon(lex_fname):
  """
  Returns a dictionary where keys are URIs, and values
  are lists of sets of source words.
  """
  lexicon = defaultdict(list)
  with codecs.open(lex_fname, 'r', 'utf-8') as fin:
    for line in fin:
      src, trg, weight = [f.strip() for f in line.strip().split(' ||| ')]
      assert len(trg.split()) == 1
      lexicon[trg].append(set(src.split()))
  return lexicon

count_op = CountOp()
def is_count_pred(trg_leaf):
  treep = TreePattern(tree_or_string('(ID {0} dummy)'.format(trg_leaf)), (), [])
  return count_op.ExpressionReturnsNumber(treep)

def align_with_lexicon(trg_leaf, src_leaves, lexicon):
  """
  Returns the source leaves indices of the longest match
  in the lexicon. If there is a tie in the longest match,
  the most contiguous is returned.
  """
  # Obtain the longest (best) matching word sequence
  # from the lexicon.
  lists_src_words = lexicon.get(trg_leaf, [])
  best_matching_words = []
  for lex_words in lists_src_words:
    if all([w in src_leaves for w in lex_words]):
      if len(lex_words) > len(best_matching_words):
        best_matching_words = lex_words
  # Obtain the word indices of the matching lexicon words
  # into a list of lists of indices. E.g. [[0], [1, 4], [3]].
  src_indices = []
  for word in best_matching_words:
    word_inds = [i for i, w in enumerate(src_leaves) if w == word]
    assert word_inds
    src_indices.append(word_inds)
  # Obtain the most contiguous matching words.
  index_combinations = list(itertools.product(*src_indices))
  most_contiguous = []
  for combination in index_combinations:
    if not most_contiguous:
      most_contiguous = combination
    elif max(combination) - min(combination) < \
         max(most_contiguous) - min(most_contiguous):
      most_contiguous = combination
  return sorted(most_contiguous)

def align(trg_leaf, src_leaves, entities_lex, predicates_lex):
  alignment_count = []
  if is_count_pred(trg_leaf) and 'how' in src_leaves and 'many' in src_leaves:
    alignment_count.append(src_leaves.index('how'))
    alignment_count.append(src_leaves.index('many'))
  alignment = align_with_lexicon(trg_leaf, src_leaves, entities_lex)
  pred_alignment = align_with_lexicon(trg_leaf, src_leaves, predicates_lex)
  if len(pred_alignment) > len(alignment):
    alignment = pred_alignment
  if alignment_count:
    alignment = alignment_count + alignment
  alignment = sorted(set(alignment))
  return alignment

def serialize_alignment(alignments):
  """
  @alignments is a list of lists. List i contains source word
  indices that align to target leaf i.
  """
  als = []
  for i, src_word_inds in enumerate(alignments):
    als.extend(zip(src_word_inds, [i] * len(src_word_inds)))
  return ' '.join(['%s-%s' % (s, t) for s, t in als])

cvts = set([
  'fb:measurement_unit.dated_integer.number',
  'fb:measurement_unit.dated_money_value.amount',
  'fb:business.employment_tenure.person',
  'fb:measurement_unit.money_value.amount',
  'fb:measurement_unit.dated_money_value.valid_date',
  'fb:measurement_unit.integer_range.high_value',
  'fb:business.stock_ticker_symbol.ticker_symbol'])

def is_cvt(trg_leaf):
  return trg_leaf.lstrip('!') in cvts

def fix_unaligned(alignments, trg_leaves):
  """
  Given a list of lists, where every list i contains the
  source word indices to which target leaf i aligns to,
  it decides on the alignments of unaligned target leaves.
  We assume here that every target leaf should be aligned
  to source words, specially those of bridged entities.
  """
  num_trg_leaves = len(alignments)
  for i, als in enumerate(alignments):
    if not als and i < (num_trg_leaves - 1) and not is_cvt(trg_leaves[i]):
      alignments[i] = alignments[i+1]
  return alignments

def get_alignment(pair_tt, entities_lex, predicates_lex):
  """
  Obtains maximum-length alignment between leaves of source and target tree.
  @pair_tt is a tuple (src_tree, trg_tree).
  @entities_lex and @predicates_lex are dictionaries, as constructed above.
  """
  assert len(pair_tt) == 2
  src_tree, trg_tree = map(tree_or_string, pair_tt)
  src_leaves = src_tree.leaves()
  if IsString(trg_tree):
    trg_leaves = [trg_tree]
  else:
    trg_leaves = trg_tree.leaves()
  # This is a list of lists. Each list will have source word indices.
  alignments = []
  for i, trg_leaf in enumerate(trg_leaves):
    alignment = align(trg_leaf, src_leaves, entities_lex, predicates_lex)
    alignments.append(alignment)
  alignments = fix_unaligned(alignments, trg_leaves)
  return alignments

def get_max_contiguous(alignment, alignments):
  """
  Given a sequence, it returns the maximum-length
  contiguous sequence. E.g.:
  For [1,3,4], returns [3,4].
  For [1,2,3], returns [1,2,3].
  If there are ties, it returns the right-most sequence (this is arbitrary).
  For [1,2,4,5], returns [4,5].
  For [1,2,3,6,7,8], returns [6,7,8].
  """
  if not alignment:
    return alignment
  alignment = sorted(alignment)
  range_len = alignment[-1] - alignment[0]
  if not (range_len > len(alignment) - 1):
    return alignment
  aligned_src_inds = set(itertools.chain(*alignments))
  # Get all contiguous segments in a list of tuples,
  # where each tuple has the start and end index of
  # each contiguous sequence.
  solutions = []
  starti, endi = 0, 0
  prev = alignment[0]
  for i in range(1, len(alignment)):
    a = alignment[i]
    if any([u in aligned_src_inds for u in range(prev+1, a)]):
      solutions.append((starti, endi))
      starti, endi = i, i
    else:
      # case: a - prev == 1, and unaligned words in the range.
      endi = i
    prev = a
  else:
    solutions.append((starti, endi))
  # Get the longest contiguous sequence.
  longest = -1
  longesti = None
  for i, (starti, endi) in enumerate(solutions):
    if endi - starti >= longest:
      longest = endi - starti
      longesti = i
  starti, endi = solutions[longesti]
  return alignment[starti:endi+1]

def make_variation(alignments, mode):
  """
  @alignments is a list of lists (as described above).
  If mode is "max", then alignments are left untouched.
  If mode is "max-contiguous", gappy alignments are substituted
  by the longest contiguous alignment. If there is a tie, then
  the right-most chunk is returned.
  """
  assert mode in ['max', 'max_contiguous'], 'Incorrect mode: %s' % mode
  if mode == 'max_contiguous':
    for i, al in enumerate(alignments):
      alignments[i] = get_max_contiguous(al, alignments)
  return alignments

def main(args = None):
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", dest="input_fname", nargs='?', type=str,
    help="Text filename with pairs of constituent trees.", default="")
  parser.add_argument("--entities", dest="entities_fname", nargs='?', type=str,
    help="Entities lexicon.", default="")
  parser.add_argument("--predicates", dest="predicates_fname", nargs='?', type=str,
    help="Predicates lexicon.", default="")
  parser.add_argument("--mode", dest="mode", nargs='?', type=str,
    help="Specifies what kind of alignment is desired (max or max_contiguous).",
    default="max")
  # parser.add_argument("--mode", action="store_true",
  #   help="Obtains maximum-length word alignments.", default=False)
  args = parser.parse_args()

  if not os.path.exists(args.input_fname):
    print('File does not exist: {0}'.format(args.input_fname))
    sys.exit(1)
  corpus = LoadCorpus(args.input_fname)
  entities_lex = load_lexicon(args.entities_fname)
  predicates_lex = load_lexicon(args.predicates_fname)

  for pair_tt in corpus:
    alignments = get_alignment(pair_tt, entities_lex, predicates_lex)
    alignments = make_variation(alignments, args.mode)
    alignments_str = serialize_alignment(alignments)
    src_tree, trg_tree = pair_tt
    print(src_tree)
    print(trg_tree)
    print(alignments_str)

if __name__ == '__main__':
  main()
