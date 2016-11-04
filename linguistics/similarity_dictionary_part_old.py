import codecs
from collections import defaultdict
import itertools
import re

from linguistics.similarity import SimilarityScorer, Similarity
from utils.tree_tools import GetLeaves, TreePattern

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
    query_word_lemma = self.lemmatize(query_word)
    return self.src_index.get(query_word_lemma, [])

  def lemmatize(self, word):
    return word

  def BuildDictionary(self, dictionary_filename):
    finput = codecs.open(dictionary_filename, 'r', 'utf-8')
    self.src_trg_cost = []
    self.src_index = defaultdict(list)
    self.trg_index = defaultdict(list)
    entries = []
    for line in finput:
      if '|||' in line:
        entries.append([item.strip() for item in line.strip('\n ').split('|||')])
    finput.close()
    # Check that all lines have the same number of fields.
    assert not entries or len(set([len(entry) for entry in entries])) == 1
    # Check if cost field is present (which should be the last field).
    is_cost_present = IsCostPresent(entries)

    for i, entry in enumerate(entries):
      if not (1 < len(entry) <= 3):
        continue
      source, target = entry[0], entry[1]
      cost = float(entry[-1]) if is_cost_present else None

      if self.lowercase:
        source, target = source.lower(), target.lower()
      self.src_trg_cost.append((source, target, cost))
      for word in source.split():
        word_lemma = self.lemmatize(word)
        self.src_index[word_lemma].append(i)
      for word in target.split():
        self.trg_index[word].append(i)
    return

  def IsEligible(self, tree_pattern1, tree_pattern2):
    return tree_pattern2.IsString()

  def IsEligibleSrc(self, src_treep):
    return True

  def GetSimilarity(self, tree_pattern1, tree_pattern2):
    """
    The cost associated to a partial match when checking the source and the target sides
    is equivalent to the worst cost (max cost) among all entries where
    there is a partial match. The rationale is that this cost function
    should not have preference over the exact match implemented in
    DictionaryCost(). 
    """
    similarities = []
    if not self.IsEligible(tree_pattern1, tree_pattern2):
      return similarities
    tree1_leaves = GetLeaves(tree_pattern1)
    tree2_leaves = [l.lstrip('!') for l in GetLeaves(tree_pattern2)]
    # Split source and target leaves by token separators.
    src_words = SplitLeavesBy(tree1_leaves, self.src_token_separators)
    trg_words = SplitLeavesBy(tree2_leaves, self.trg_token_separators)
    if self.lowercase:
      src_words = [word.lower() for word in src_words]
      trg_words = [word.lower() for word in trg_words]
    # Obtain indices of bilingual phrases for which at least one source word
    # appears.
    src_word_indices = [self.get_src_index(word) for word in src_words]
    src_indices = set(itertools.chain(*src_word_indices))
    # The same for target words.
    trg_word_indices = [self.trg_index.get(word, []) for word in trg_words]
    trg_indices = set(itertools.chain(*trg_word_indices))

    common_indices = src_indices.intersection(trg_indices)
    if not common_indices:
      return similarities

    cost = self.GetSimilarityCost(src_words, trg_words, common_indices)
    similarities = [Similarity(cost, self.relation, tree_pattern1, tree_pattern2)]
    return similarities

  def GetSimilarityCost(self, src_words, trg_words, common_indices):
    """
    Computes the maximum cost among all entries that share at least one word
    with the source and the target.
    """
    max_cost = 2.0 * max([self.src_trg_cost[i][2] for i in common_indices])
    return max_cost

  def GetSimilarProb(self, src_words, src_index):
    """
    The probability associated to a partial match when checking only source
    is equivalent to the probability (complement to the cost) of the lexical
    entry in the dictionary.
    """
    prob = 1.0 - self.src_trg_cost[src_index][2]
    return prob

  def MakeSimilar(self, src_tree_pattern, src_words, src_indices):
    similarities = []
    for i in src_indices:
      target = self.src_trg_cost[i][1].replace(' ', self.trg_token_separator)
      prob = self.GetSimilarProb(src_words, i)
      trg_tree_pattern = TreePattern(target, (), [])
      similarities.append(
        Similarity(prob, self.relation, src_tree_pattern, trg_tree_pattern))
    # Remove duplicated elements (since this is not an exact match, they may occur).
    similarities = list(set(similarities))
    return similarities

  def GetSimilar(self, src_tree_pattern):
    similarities = []
    if not self.IsEligibleSrc(src_tree_pattern):
      return similarities
    src_leaves = GetLeaves(src_tree_pattern)
    src_words = SplitLeavesBy(src_leaves, self.src_token_separators)
    if self.lowercase:
      src_words = [word.lower() for word in src_words]
    # Obtain indices of bilingual phrases for which at least one source word
    # appears.
    src_word_indices = [self.get_src_index(word) for word in src_words]
    src_indices = set(itertools.chain(*src_word_indices))

    similarities = self.MakeSimilar(src_tree_pattern, src_words, src_indices)
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

