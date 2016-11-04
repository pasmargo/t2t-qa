#!/usr/bin/python
from collections import defaultdict
import simplejson

from nltk.align import AlignedSent

from utils.tree_tools import IsString

def PrintGeneralInfo(general_info, fmt='json'):
  if fmt not in ['json', 'yaml', 'tiburon']:
    raise ValueError('Format to print rule not recognized: {0}'\
                     .format(fmt))
  if fmt == 'json':
    general_info_str = PrintGeneralInfoJson(general_info)
  elif fmt == 'yaml':
    general_info_str = PrintGeneralInfoYaml(general_info)
  elif fmt == 'tiburon':
    general_info_str = PrintGeneralInfoTiburon(general_info)
  return general_info_str

def PrintGeneralInfoYaml(general_info):
  """
  This function simply converts (in an ugly way) a dictionary 'pair_info'
  into a yaml comment of the form:
  # keyword1: value1
  # keyword2: value2
  # ...
  """
  general_info_str = u'# ' \
    + u'\n# '.join([u': '.join([unicode(s) for s in item]) \
                     for item in general_info['general_info'].items()])
  return general_info_str.encode('utf-8')

def PrintGeneralInfoTiburon(general_info):
  """
  This function simply converts (in an ugly way) a dictionary 'pair_info'
  into a tiburon comment of the form:
  % keyword1: value1
  % keyword2: value2
  % ...
  It also concatenates the initial state (uncommented).
  """
  general_info_str = u'% ' \
    + u'\n% '.join([u': '.join([unicode(s) for s in item]) \
                     for item in general_info['general_info'].items()])
  general_info_str += '\n' + general_info['general_info']['initial_state']
  return general_info_str.encode('utf-8')

def PrintGeneralInfoJson(general_info):
  return simplejson.dumps(general_info, ensure_ascii=False).encode('utf-8')

def PrintRule(rule, fmt='json'):
  if fmt not in ['json', 'yaml', 'tiburon']:
    raise ValueError('Format to print rule not recognized: {0}'\
                     .format(fmt))
  if fmt == 'json':
    rule_str = rule.PrintJson().encode('utf-8')
  elif fmt == 'yaml':
    rule_str = rule.PrintYaml().encode('utf-8') + '\n'
  elif fmt == 'tiburon':
    rule.weight = None
    rule_str = rule.PrintTiburon().encode('utf-8')
  return rule_str

def PrintSentencePairInfo(tree_pair_info, fmt='json'):
  if fmt not in ['json', 'yaml', 'tiburon']:
    raise ValueError('Format to print sentence pair info not recognized: {0}'\
                     .format(fmt))
  if fmt == 'json':
    sentence_pair_info_str = \
      PrintSentencePairInfoJson(tree_pair_info)
  elif fmt == 'yaml':
    sentence_pair_info_str = \
      PrintSentencePairInfoYaml(tree_pair_info)
  elif fmt == 'tiburon':
    sentence_pair_info_str = \
      PrintSentencePairInfoTiburon(tree_pair_info)
  return sentence_pair_info_str

def PrintSentencePairInfoYaml(tree_pair_info):
  """
  This function simply converts (in an ugly way) a dictionary 'pair_info'
  into a yaml comment of the form:
  # keyword1: value1
  # keyword2: value2
  # ...
  """
  tree_pair_info_str = u'# ' \
    + u'\n# '.join([u': '.join([unicode(s) for s in item]) \
                     for item in tree_pair_info['pair_info'].items()])
  return tree_pair_info_str.encode('utf-8')

def PrintSentencePairInfoTiburon(tree_pair_info):
  """
  This function simply converts (in an ugly way) a dictionary 'pair_info'
  into a tiburon comment of the form:
  % keyword1: value1
  % keyword2: value2
  % ...
  """
  tree_pair_info_str = u'% ' \
    + u'\n% '.join([u': '.join([unicode(s) for s in item]) \
                     for item in tree_pair_info['pair_info'].items()])
  return tree_pair_info_str.encode('utf-8')

def PrintSentencePairInfoJson(tree_pair_info):
  return simplejson.dumps(tree_pair_info, ensure_ascii=False).encode('utf-8')

def PrintDefaultAlignment(tree1, tree2):
  """
  This prints:
  1-1 2-2 3-3 ... N-N
  where N is the number of leaves of the smallest tree.
  I don't expect this to happen often, and it serves as a backoff
  of alignment failure.
  """
  num_leaves_tree1 = len(tree1.GetLeaves())
  num_leaves_tree2 = len(tree2.GetLeaves())
  return ' '.join([str(x[0]) + '-' + str(x[1]) \
                     for x in zip(range(num_leaves_tree1),
                                  range(num_leaves_tree2))])

class Alignment:
  """
  This class holds information about a word-to-word (or phrase-to-phrase)
  alignments. It also implements some printing routines to make the alignments
  compatible with those produced by GIZA++.
  """
  def __init__(self):
    pass

def GetLeavePositions(tree):
  positions = []
  if IsString(tree):
    if not tree.startswith(u'?x'):
      positions.append( () )
  else:
    positions = [position for position in tree.treepositions('leaves') \
                   if not tree[position].startswith(u'?x')]
  return positions

def GetAlignment(derivation, tree1, tree2, cost_threshold = 0.3):
  """
  Given a derivation (sequence of rules that puts in correspondence tree1
  and tree2), it returns a word-to-word alignment.
  """
  src_tokens = tree1.leaves()
  trg_tokens = tree2.leaves()
  alignments_dict = defaultdict(list)
  src_leave_paths = sorted(tree1.treepositions('leaves'))
  trg_leave_paths = sorted(tree2.treepositions('leaves'))
  assert len(src_tokens) == len(src_leave_paths), \
    'Num. source leaves and leave paths mismatch: {0} vs. {1}'\
    .format(src_tokens, src_leave_paths)
  assert len(trg_tokens) == len(trg_leave_paths), \
    'Num. target leaves and leave paths mismatch: {0} vs. {1}'\
    .format(trg_tokens, trg_leave_paths)
  # Create an index that maps paths to word indices in the sentences.
  src_path_to_index = {path : index for index, path in enumerate(src_leave_paths)}
  trg_path_to_index = {path : index for index, path in enumerate(trg_leave_paths)}
  # Extract word-to-word alignments, one production at a time.
  for production in derivation:
    # Get the absolute path positions of lhs and rhs leaves.
    state, src_path, trg_path = production.non_terminal
    if state != 'dist_sim':
      continue
    lhs, rhs = production.rhs.rule.lhs, production.rhs.rule.rhs
    src_leave_abs_positions = \
      [src_path + src_leaf_path for src_leaf_path in GetLeavePositions(lhs)]
    trg_leave_abs_positions = \
      [trg_path + trg_leaf_path for trg_leaf_path in GetLeavePositions(rhs)]
    cost = production.rhs.rule.weight
    num_src_and_trg_leaves = \
      len(src_leave_abs_positions) + len(trg_leave_abs_positions)
    if (cost / float(num_src_and_trg_leaves)) > cost_threshold:
      continue
    # Set the index of each source word to align to the index of each
    # target word appearing in this production (rule).
    for src_leaf_path in src_leave_abs_positions:
      src_leaf_index = src_path_to_index[src_leaf_path]
      for trg_leaf_path in trg_leave_abs_positions:
        trg_leaf_index = trg_path_to_index[trg_leaf_path]
        alignments_dict[src_leaf_index].append(trg_leaf_index)
  # List of tuples with alignments. E.g. [(0, 0), (0, 1), (1, 2), (2, 3), ...]
  alignments = [(src_index, trg_index) \
                  for (src_index, trg_indices) in alignments_dict.items() \
                    for trg_index in trg_indices]
  alignments_str = ' '.join([str(src_index) + '-' + str(trg_index) \
                               for (src_index, trg_index) in alignments])
  aligned_sentence = AlignedSent(src_tokens, trg_tokens, alignments_str)
  return aligned_sentence

