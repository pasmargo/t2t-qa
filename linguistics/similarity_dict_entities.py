from fuzzywuzzy import process, fuzz

from nltk.stem.porter import PorterStemmer

from extraction.feat_instantiator import get_ngram_ratio
from linguistics.similarity import Similarity
from linguistics.similarity_dictionary_part import SplitLeavesBy
from linguistics.similarity_dictionary_part_old import DictionaryCostPart
from qald.lexicon import exclude_words, filter_tokens
from utils.tree_tools import (GetLeaves, TreePattern, Tree, IsVariable,
  IsPlausibleEntityPhrase)

stemmer = PorterStemmer()

class DictEntities(DictionaryCostPart):
  """
  Check a dictionary of the form source ||| entity ||| context ||| cost.
  Source field may have several tokens, separated by whitespaces. E.g.
  src_tok1 src_tok2 ||| entity ||| word1 [src_tok1] word2 [src_tok2] ||| cost.
  Leaves from source may contain several tokens separated by a character.
  E.g. 'come_up'. The attribute self.src_token_separators can be
  overwritten to control how comparisons of source leaves
  are done with respect the dictionary (which is space separated).
  In this implementation, if *any* token of the source appears in
  a dictionary entry, then the entry is selected (partial match).
  """

  def __init__(self, dictionary_filename, feature_weight = 1.0):
    self.feature_weight = feature_weight
    self.relation = 'entity'
    self.kCost = 1.0
    self.ExtraCost = 3.0
    self.src_token_separators = '_'
    self.trg_token_separators = '|' # Invalid separator, on purpose.
    self.trg_token_separator = '|'
    self.trg_ngram_token_separators = '._: '
    self.max_src_phrase_length = 6
    self.trg_phrase_length = 1
    self.lowercase = True
    self.n_best = 200
    self.BuildDictionary(dictionary_filename)

  def IsEligibleSrc(self, src_treep):
    if src_treep.HasVariables():
      return False
    if not IsPlausibleEntityPhrase(src_treep):
      return False
    src_leaves = filter_tokens(GetLeaves(src_treep))
    if not src_leaves:
      return False
    if not len(src_leaves) <= self.max_src_phrase_length:
      return False
    return True

  def IsEligible(self, src_treep, trg_treep):
    """
    The source tree pattern should not contain any variable (hence,
    no variables in target tree pattern either), have equal or less leaves
    than self.max_src_phrase_length and the target tree pattern have
    self.trg_phrase_length leaves.
    """
    if not self.IsEligibleSrc(src_treep):
      return False
    if trg_treep is not None:
      trg_leaves = [leaf for leaf in GetLeaves(trg_treep) if not IsVariable(leaf)]
      if not len(trg_leaves) == self.trg_phrase_length:
        return False
    return True

  def IsEligible_(self, src_treep, trg_treep):
    """
    The source tree pattern should not contain any variable (hence,
    no variables in target tree pattern either), have equal or less leaves
    than self.max_src_phrase_length and the target tree pattern have
    self.trg_phrase_length leaves.
    """
    if src_treep.HasVariables():
      return False
    src_leaves = GetLeaves(src_treep)
    if not len(src_leaves) <= self.max_src_phrase_length:
      return False
    if trg_treep is not None:
      trg_leaves = [leaf for leaf in GetLeaves(trg_treep) if not IsVariable(leaf)]
      if not len(trg_leaves) == self.trg_phrase_length:
        return False
    return True

  def lemmatize(self, word):
    word_lemma = stemmer.stem(word)
    return word_lemma

  def get_src_index(self, query_word):
    if query_word in exclude_words:
      return []
    query_word_lemma = self.lemmatize(query_word)
    return self.src_index.get(query_word_lemma, [])

  def ComputeSetsCost(self, src_items, trg_items, common_cost, diff_cost):
    src_items_set = set(src_items)
    trg_items_set = set(trg_items)
    cost = len(src_items_set.intersection(trg_items_set)) * common_cost \
         + len(src_items_set.symmetric_difference(trg_items_set)) * diff_cost
    return cost

  def GetSimilarityCost_(self, src_words, trg_words, common_indices):
    """
    For every candidate entry index:
    Computes self.kCost multiplied by all source words and target words
    that had a match, and self.ExtraCost multiplied by the source and target
    words that did not match.
    It returns the best cost.
    """
    costs = []
    for common_index in common_indices:
      src_phrase = self.src_trg_cost[common_index][0].split()
      trg_phrase = self.src_trg_cost[common_index][1].split()
      # Add the costs associated to source words.
      cost = self.ComputeSetsCost(src_words, src_phrase, self.kCost, self.ExtraCost)
      cost += self.ComputeSetsCost(trg_words, trg_phrase, self.kCost, self.ExtraCost)
      costs.append(cost)
    best_cost = min(costs)
    return best_cost

  def GetSimilarityCost(self, src_words, trg_words, common_indices):
    """
    The cost per token is guaranteed to be between 0 and 1 (except for bridges).
    For every candidate entry index:
    1. Splits words according to token separators,
    2. Filter-out stopwords in source (e.g. determiners or prepositions)
       and target (e.g. "fb:", "en", "m"),
    3. Computes ngram ratio between source and target tokens,
    4. Multiplies the complement of the ratio (1.0 - ratio) by the number
       of leaves.
    5. If there are two target URIs, then add self.extra_cost.
    """
    num_src_words = len(src_words)
    num_trg_words = len(trg_words)
    src_words = filter_tokens(src_words)
    trg_words = SplitLeavesBy(trg_words[-1:], self.trg_ngram_token_separators)
    trg_words = filter_tokens(trg_words)
    costs = []
    for common_index in common_indices:
      stored_src_words = self.src_trg_cost[common_index][0].split()
      stored_trg_words = self.src_trg_cost[common_index][1].split()
      uri = stored_trg_words[-1]
      stored_trg_words = SplitLeavesBy(
        stored_trg_words, self.trg_ngram_token_separators)
      stored_src_words = filter_tokens(stored_src_words)
      stored_trg_words = filter_tokens(stored_trg_words)
      src_ngram_ratio = get_ngram_ratio(src_words, stored_src_words)
      trg_ngram_ratio = get_ngram_ratio(trg_words, stored_trg_words)
      ngram_ratio = (src_ngram_ratio + trg_ngram_ratio) / 2
      cost = (2.0 - ngram_ratio) * (num_src_words + 1)
      if num_trg_words == 2:
        cost += self.ExtraCost
      costs.append(cost)
    best_cost = min(costs)
    return best_cost

  def GetSimilarProb(self, src_words, src_index):
    """
    Obtains the probability to transform src_words.
    """
    src_phrase = self.src_trg_cost[src_index][0].split()
    prob = self.src_trg_cost[src_index][2]
    src_phrase_set = set(src_phrase)
    src_words_set = set(src_words)
    prob *= float(len(src_phrase_set.intersection(src_words_set))) \
          / len(src_phrase_set.union(src_words_set))
    # return cost
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

class DictBridgeEntities(DictEntities):
  """
  Check a dictionary of the form source ||| entity ||| context ||| cost.
  Source field may have several tokens, separated by whitespaces. E.g.
  src_tok1 src_tok2 ||| entity ||| word1 [src_tok1] word2 [src_tok2] ||| cost.
  Leaves from source may contain several tokens separated by a character.
  E.g. 'come_up'. The attribute self.src_token_separators can be
  overwritten to control how comparisons of source leaves
  are done with respect the dictionary (which is space separated).
  In this implementation, if *any* token of the source appears in
  a dictionary entry, then the entry is selected (partial match).
  """
  def __init__(self, dictionary_filename, feature_weight = 1.0):
    self.feature_weight = feature_weight
    self.relation = 'bridge_entity'
    self.kCost = 1.0
    self.ExtraCost = 3.0
    self.src_token_separators = '_'
    self.trg_token_separators = '|' # Invalid separator, on purpose.
    self.trg_token_separator = '|'
    self.trg_ngram_token_separators = '._: '
    self.max_src_phrase_length = 6
    self.trg_phrase_length = 2
    self.lowercase = True
    self.n_best = 200
    self.BuildDictionary(dictionary_filename)

  def MakeSimilar(self, src_tree_pattern, src_words, src_indices):
    similarities = []
    for i in src_indices:
      target = self.src_trg_cost[i][1].replace(' ', self.trg_token_separator)
      prob = self.GetSimilarProb(src_words, i)
      trg_tree_pattern = \
        TreePattern(Tree.fromstring(u'(ID [] {0})'.format(target)), (), [])
      similarities.append(
        Similarity(prob, self.relation, src_tree_pattern, trg_tree_pattern))
    # Remove duplicated elements (since this is not an exact match, they may occur).
    similarities = list(set(similarities))
    return similarities

class DictPredicates(DictEntities):
  """
  Check a dictionary of the form source ||| entity ||| context ||| cost.
  Source field may have several tokens, separated by whitespaces. E.g.
  src_tok1 src_tok2 ||| entity ||| word1 [src_tok1] word2 [src_tok2] ||| cost.
  Leaves from source may contain several tokens separated by a character.
  E.g. 'come_up'. The attribute self.src_token_separators can be
  overwritten to control how comparisons of source leaves
  are done with respect the dictionary (which is space separated).
  In this implementation, if *any* token of the source appears in
  a dictionary entry, then the entry is selected (partial match).
  """
  def __init__(self, dictionary_filename, feature_weight = 1.0):
    self.feature_weight = feature_weight
    self.relation = 'predicate'
    self.kCost = 1.0
    self.ExtraCost = 3.0
    self.src_token_separators = '_'
    self.trg_token_separators = '|' # Invalid separator, on purpose.
    self.trg_token_separator = '|'
    self.trg_ngram_token_separators = '._: '
    self.max_src_phrase_length = 6
    self.trg_phrase_length = 1
    self.lowercase = True
    self.n_best = 200
    self.BuildDictionary(dictionary_filename)

  def IsEligibleSrc(self, src_treep):
    if src_treep.HasVariables():
      return False
    src_leaves = filter_tokens(GetLeaves(src_treep))
    if not src_leaves:
      return False
    if not len(src_leaves) <= self.max_src_phrase_length:
      return False
    return True

  def MakeSimilar(self, src_tree_pattern, src_words, src_indices):
    similarities = []
    for i in src_indices:
      target = self.src_trg_cost[i][1].replace(' ', self.trg_token_separator)
      prob = self.GetSimilarProb(src_words, i)
      if src_tree_pattern.HasVariables():
        trg_tree_pattern = \
          TreePattern(Tree.fromstring(u'(ID {0} ?x0|)'.format(target)), (), [])
      else:
        trg_tree_pattern = TreePattern(target, (), [])
      similarities.append(
        Similarity(prob, self.relation, src_tree_pattern, trg_tree_pattern))
    # Remove duplicated elements (since this is not an exact match, they may occur).
    similarities = list(set(similarities))
    return similarities

def tokmatch_(s, t):
  """
  Computes string matching ratio between source s and target t.
  If any of the strings is smaller than min_len, the matching
  needs to be exact. Otherwise, fuzz.WRatio is used.
  """
  min_len = 3
  if len(s) <= 3 or len(t) <= 3:
    score = 100 if s == t else 0
  else:
    score = fuzz.WRatio(s, t)
  return score

def tokmatch(s, t):
  """
  Computes Han-san's metric between source s and target t.
  In this metric:
    1. Check if the longest word contains the shortest.
    2. If not, then check if len() - 1 or - 2 of the shortest word
       is contained in the longest.
  If any of the strings is smaller than min_len, the matching
  needs to be exact.
  Otherwise, fuzz.WRatio is used.
  Return 100 if perfect match, and 0 if total unmatch.
  """
  min_len = 3
  if len(s) <= 3 or len(t) <= 3:
    score = 100 if s == t else 0
  else:
    shortest_word, longest_word = (s, t) if len(s) < len(t) else (t, s)
    if shortest_word in longest_word or \
       shortest_word[:-1] in longest_word or \
       shortest_word[:-2] in longest_word:
      score = 100
    else:
      score = fuzz.WRatio(s, t)
  return score

class DictFuzzyPredicates(DictPredicates):
  """
  Words of left-hand-sides in dictionary entries are fuzzy matched
  against words from the source side of tree patterns.
  """
  def __init__(self, dictionary_filename, feature_weight = 1.0):
    super(DictFuzzyPredicates, self).__init__(dictionary_filename, feature_weight)
    self.src_words = list(set(self.src_index.keys()))
    self.min_score = 70

  def get_src_index(self, query_word, limit=5):
    # Adding exact match indices.
    indices = set(self.src_index.get(query_word, []))
    # Adding fuzzy match indices.
    fuzzy_matches = process.extract(
      query_word, self.src_words, scorer=tokmatch, limit=limit)
    for similar_word, score in fuzzy_matches:
      if score < self.min_score:
        break
      indices.update(self.src_index.get(similar_word, []))
    return indices
