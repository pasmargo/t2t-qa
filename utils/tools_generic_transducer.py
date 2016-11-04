#!/usr/bin/python

import codecs
import itertools
from lxml import etree
import re

# For a list of tree_strings such as:
# ['(ROOT S(DT(the) NN(house)))', '(ROOT S(DT(the) NN(house)))']
# Merges (and outputs) both tree_strings into one as:
# 'ROOT(S(DT(the) NN(house)) S(DT(the) NN(house)))'
def MergeTreeStrings(tree_strings):
  if len(tree_strings) == 1:
    return tree_strings[0]
  else:
    return '(ROOT ' + ' '.join([x[6:-1] for x in tree_strings]) + ')'

# Convert tree_strings in Stanford format '(NP (DT a) (NN house))'
# into Tiburon format 'NP(DT(a) NN(house))'
def StanfordToTiburon(tree_string):
  return re.sub(r'\((\S+) ', r'\1(', tree_string)

# Converts a tree in dictionary form to tree_string.
def ConvertTreeToTreeString(tree, quotes = True):
  tree_string = ''
  if type(tree) is not dict:
    # Quotes every token.
    if quotes:
      tree_string = '"' + tree + '"'
  else:
    children = list(tree.keys())
    children.sort(key = lambda x: x[1])
    i = 0
    num_children = len(children)
    for child in children:
      (pos_tag, index) = tuple(child)
      tree_string = tree_string + pos_tag + '('
      subtree_string = ConvertTreeToTreeString(tree[child])
      tree_string = tree_string + subtree_string
      if i < num_children - 1:
        tree_string = tree_string + ') '
      else:
        tree_string = tree_string + ')'
      i = i + 1
  return tree_string

# Given a tree (in a dictionary), lowercases the leaves.
def LowercaseTree(tree):
  if type(tree) is not dict:
    tree = tree.lower()
  else:
    for child in tree.keys():
      tree[child] = LowercaseTree(tree[child])
  return tree

# Given the string of a sentence, normalize it.
# This is necessary because the transducer
# treats specially the dots (to refer to non-terminals after states).
# Also lowercasing.
def NormalizeString(sentence, quotes = True):
  sentence = re.sub(r'%', 'PERCENT', sentence)
  sentence = re.sub(r':', 'COLON', sentence)
  sentence = re.sub(r'#', 'NUMBERSIGN', sentence)
  # Quotes every token.
  if quotes:
    sentence = ' '.join(map(lambda x: '"' + x + '"', sentence.split()))
  # Lowercase.
  sentence = sentence.lower()
  return sentence

# Substitute symbols in a tree_string. So far:
# .(.) becomes PU(.)
# (%) becomes (PERCENT)
def NormalizeTreeString(tree_string):
  tree_string = re.sub(r'\.\(', 'PU(', tree_string)
  tree_string = re.sub(r'%', 'PERCENT', tree_string)
  tree_string = re.sub(r':', 'COLON', tree_string)
  tree_string = re.sub(r'#', 'NUMBERSIGN', tree_string)
  # Lowercase.
  (tree, _) = ParseTreeString(tree_string)
  tree = LowercaseTree(tree)
  tree_string = ConvertTreeToTreeString(tree, False)
  return tree_string

# Given a list of source words and a list of target words,
# it returns a set of words that appear in the target more times
# than in the source.
def ObtainWordsInTargetNotInSource(source_words, target_words):
  # Count how many times every word in the target appears.
  d = {}
  for target_word in target_words:
    if target_word not in d.keys():
      d[target_word] = 1
    else:
      d[target_word] = d[target_word] + 1
  # Substract the counts of target words using counts from source words.
  for source_word in source_words:
    if source_word in d.keys():
      d[source_word] = d[source_word] - 1
  # Collect words from target words that appear more times than
  # in the list of source words.
  words_in_target_not_in_source = set()
  for word in d.keys():
    if d[word] > 0:
      words_in_target_not_in_source.update([word])
  return list(words_in_target_not_in_source)

# Convert tree_strings in Stanford format '(NP (DT a) (NN house))'
# into Tiburon format 'NP(DT(a) NN(house))'
def StanfordToTiburon(tree_string):
  return re.sub(r'\((\S+) ', r'\1(', tree_string)

# Convert tree_strings in Tiburon format 'NP(DT(a) NN(house))'
# into Stanford format '(NP (DT a) (NN house))'
# This function also removes the quotes around the tokens.
def TiburonToStanford(tree_string):
  nt_moved = re.sub(r'(\S+?)\(', r'(\1 ', tree_string)
  quotes_removed = re.sub(r'"(\S+?)"\)', r'\1)', nt_moved)
  quotes_removed = re.sub(r' "(\S+?)" ', r' \1 ', quotes_removed)
  return quotes_removed

# The same as above, but without removing quotes from tokens.
def TiburonToStanford_(tree_string):
  return re.sub(r'(\S+?)\(', r'(\1 ', tree_string)

# Read the tree_string ||| {tree_}string pairs from a file.
# The file contains lines like:
# NP(DT(a) NN(house))
# There is a house (or its tree version)
# NP(DT(An) JJ(Italian))
# At least one Italian (or its tree version)
def LoadCorpus(training_filename):
  finput = codecs.open(training_filename, 'r', 'utf-8')
  pairs = []
  i = 0
  for line in finput.readlines():
    if i % 2 == 0:
      pair = [line.strip(' \t\n\r')]
    else:
      pair.append(line.strip(' \t\n\r'))
      pairs.append(pair)
    i = i + 1
  finput.close()
  return pairs

# Given a string representation of a tree such as:
# u'S(NP(DT(Every) NN(student)) VP(VBD(used) NP(PRP$(her) NN(workstation))))'
# returns a list of tuples (pre-terminal, token). In this case:
# [(u'DT', u'Every'), (u'NN', u'student'), (u'VBD', u'used'), ...]
def ObtainPreTerminalsFromTreeString(tree_string):
  # Gets a list like [u'DT(Every)', u'NN(student)', u'VBD(used)', ...]
  preterminal_strings = re.findall(r'([A-Z$]+?\([^)(]+?\))', tree_string)
  # Returns [[(u'DT', u'Every'), (u'NN', u'student'), (u'VBD', u'used'), ...]
  return [(x[0], x[1][:-1]) \
          for x in map(lambda x: x.split('('), preterminal_strings)]

# Given a string representation of a tree such as:
# S(NP(DT(Every) JJ(Italian) NN(man)) VP(laughs))
# returns the list of words (the yield): Every Italian man laughs
def ObtainWordsFromTreeString(tree_string):
  return re.findall(r'\S+\((\S+?)\)', tree_string)

# Returns the words (that are on the leaves) from a tree
# represented as a dictionary.
def ObtainWordsFromTree(tree):
  if type(tree) is not dict:
    return [tree]
  words = []
  children = tree.keys()
  children.sort(key = lambda x: x[1])
  for node in children:
    words.extend(ObtainWordsFromTree(tree[node]))
  return words

# Helper function to read symbols from the string representation of the tree.
# It returns (as a tuple) the trimmed symbol and the initial position of the
# next symbol.
def ReadSymbolFromTreeString(tree_string, initial_position):
  # There are no more symbols to read because we are at the end of the string.
  if initial_position == len(tree_string):
    return (None, -1)
  i = initial_position
  symbol_name = ''
  while i < len(tree_string) and tree_string[i] not in '()':
    symbol_name = symbol_name + tree_string[i]
    i = i + 1
  return (symbol_name.strip(' \t\n\r'), i)

# Transform a tree represented in dictionary form into
# a tree represented as lxml tree.
def TreeToLxmlTree(tree):
  if type(tree) is not dict:
    leaf = etree.Element('token')
    leaf.text = tree
    return [leaf]
  subtrees = []
  children = tree.keys()
  children.sort(key = lambda x: x[1])
  for child in children:
    (pos_tag, index) = child
    subtree = etree.Element('nt')
    subtree.set('pos', pos_tag)
    [subtree.append(x) for x in TreeToLxmlTree(tree[child])]
    subtrees.append(subtree)
  return subtrees

# Given the Tiburon string representation of a tree, such as:
# u'NP(DT(a) NN(house))', returns an lxml tree.
def TreeStringToLxml(tree_string, initial_position = 0):
  i = initial_position # holds the current position in the string.
  (symbol_name, i) = ReadSymbolFromTreeString(tree_string, i)
  # If we find the symbol ')' means that we are processing a terminal node,
  # and that only contains a string with no children. E.g. 'house)'
  # Thus, we return it and finish the recursion on this branch.
  if i >= len(tree_string) or tree_string[i] == ')':
    leaf = etree.Element('token')
    leaf.text = symbol_name
    if initial_position == 0:
      return leaf
    else:
      return ([leaf], i + 1)
  trees = []
  while i <= len(tree_string):
    tree = etree.Element('nt')
    tree.set('pos', symbol_name)
    (subtrees, i) = TreeStringToLxml(tree_string, i + 1)
    [tree.append(s) for s in subtrees]
    trees.append(tree)
    if i >= len(tree_string) or tree_string[i] == ')':
      break
    (symbol_name, i) = ReadSymbolFromTreeString(tree_string, i + 1)
    if symbol_name == None:
      break
  if initial_position == 0:
    return trees[0]
  else:
    return (trees, i + 1)

# Given the Tiburon string representation of a tree, such as:
# u'NP(DT(a) NN(house))'
# returns a dictionary of dictionaries, such as:
# d = {('NP', 0) : {('DT', 0) : 'a', ('NN', 1) : 'house'}}
def ParseTreeString(tree_string, initial_position = 0):
  i = initial_position # holds the current position in the string.
  (symbol_name, i) = ReadSymbolFromTreeString(tree_string, i)
  # If we find the symbol ')' means that we are processing a terminal node,
  # and that only contains a string with no children. E.g. 'house)'
  # Thus, we return it and finish the recursion on this branch.
  if tree_string[i] == ')':
    return (symbol_name, i + 1)
  tree = {} # will hold the string parsed into a tree (as dictionaries).
  symbol_order = 0
  while i <= len(tree_string):
    (subtree, i) = ParseTreeString(tree_string, i + 1)
    tree[(symbol_name, symbol_order)] = subtree
    symbol_order = symbol_order + 1
    if i >= len(tree_string) or tree_string[i] == ')':
      break
    (symbol_name, i) = ReadSymbolFromTreeString(tree_string, i + 1)
    if symbol_name == None:
      break
  return (tree, i + 1)

# Count the number of variables on a left hand side.
# E.g. q.S(x0:NP x1:VP) would return 2 (which are x0 and x1).
def CountVariables(left_hand_side):
  return len(left_hand_side.split(' '))

# Returns a generator of the powerset of the list.
# This powerset does not contain the empty element nor all elements.
def powerset(iterable):
  "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3)"
  s = list(iterable)
  return itertools.chain.from_iterable(
    itertools.combinations(s, r) for r in range(1, len(s)))

# Returns a generator of the powerset of the list, with length
# between 1 and N.
def powersetN(iterable, minimum, maximum):
  "powersetN([1,2,3], 1, 1) --> (1,) (2,) (3,)"
  s = list(iterable)
  return itertools.chain.from_iterable(
    itertools.combinations(s, r) for r in range(minimum, maximum + 1))

def powersetN_with_replacement(iterable, minimum, maximum):
  "powersetN_with_replacement([1,2,3], 1, 2) --> (1,) (2,) (3,) (1, 1) (1, 2) ..."
  s = list(iterable)
  return itertools.chain.from_iterable(
    itertools.combinations_with_replacement(s, r) \
    for r in range(minimum, maximum + 1))

def powersetN_with_permutations(iterable, minimum, maximum):
  "powersetN_with_replacement([1,2,3], 1, 2) --> (1,) (2,) (3,) (1, 1) (1, 2) ..."
  s = list(iterable)
  powerset = itertools.chain.from_iterable( \
    itertools.combinations(s, r) for r in range(minimum, maximum + 1))
  powerset_with_permutations = []
  [powerset_with_permutations.extend(itertools.permutations(p)) for p in powerset]
  return powerset_with_permutations

# Output the transductions into a file.
# The initial state is written in the first line and is "q".
def SaveTransductionsIntoFile(transductions, filename):
  foutput = codecs.open(filename, 'w', 'utf-8')
  # Initial state.
  foutput.write('q\n')
  # Transductions.
  transductions_list = list(transductions)
  transductions_list.sort()
  for t in transductions_list:
    foutput.write('%s\n' % t)
  foutput.close()

