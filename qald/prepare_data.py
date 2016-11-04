#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import json
import os
import random
from signal import signal, SIGPIPE, SIG_DFL 
import sys

from qald.dcs_tools import ConvertDCS2Constituent
from utils.tree_tools import tree_or_string

# Since we are piping the output of this script,
# we do not wish SIGPIPE to raise an exception.
signal(SIGPIPE, SIG_DFL)

def remove_qmark(tree):
  """
  Removes the question mark at the end of the tree, if it exists.
  """
  for i, s in enumerate(tree):
    subtree_leaves = s.leaves()
    if len(subtree_leaves) == 1 and subtree_leaves[0] == '?':
      del tree[i]
      return tree
    elif '?' in subtree_leaves:
      return remove_qmark(s)
  return tree

def main(args = None):
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", dest="input_fname", nargs='?', type=str,
    help="Json filename with questions, as provided by SEMPRE.", default="")
  parser.add_argument("--binarize", action="store_true",
    help="Convert source trees into Chomsky normal form.", default=False)
  parser.add_argument("--rem_qmark", action="store_true",
    help="Remove question mark from the end of the question.", default=False)
  parser.add_argument("--random", action="store_true",
    help="Randomize the order of the examples.", default=False)
  args = parser.parse_args()

  if not os.path.exists(args.input_fname):
    print('File does not exist: {0}'.format(args.input_fname))
    sys.exit(1)
  # json data as provided by SEMPRE.
  fin_json_fname = args.input_fname
  with open(fin_json_fname) as fin:
    data = json.load(fin)
    if args.random:
      # random.seed(23)
      random.shuffle(data)
    for d in data:
      if 'targetFormula' in d:
        dcs_str = d['targetFormula']
        trg_tree = ConvertDCS2Constituent(dcs_str)
      else:
        trg_tree = '(ID no target formula)'
      src_tree = tree_or_string(d['src_tree'])
      if args.rem_qmark:
        src_tree = remove_qmark(src_tree)
      if args.binarize:
        src_tree.chomsky_normal_form()
      print(src_tree)
      print(trg_tree)

if __name__ == '__main__':
  main()

