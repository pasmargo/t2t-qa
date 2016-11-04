# -*- coding: utf-8 -*-
import sys
from multiprocessing import Manager

from extraction.feat_instantiator import get_ngram_ratio
from linguistics.similarity import SimilarityScorer, Similarity
from linguistics.similarity_dictionary_part import SplitLeavesBy
from qald.lexicon import exclude_words, filter_tokens
from utils.cache import TryPickleLoad as TryLoad, TryPickleDump as TryDump
from utils.tree_tools import (GetLeaves, tree_or_string, TreePattern, IsVariable,
  IsPlausibleEntityPhrase)

# TODO:
# * Setting context.
# * Mechanism to output Solr features (e.g. rank, score, etc.).

def IsURITooLong(uri):
  return len(uri) > 300

class LinkingCost(SimilarityScorer):
  """
  Implements the abstract methods from Similarity class. It measures the
  cost of mapping between a natural language phrase and entities or predicates.
  E.g. (NP (NN Marshall) (NN Hall)) -- (ID fb:education.academic_post.person fb:en.marshall_hall)
  should have a low cost.
  This class contains generic routines to operate with a Solr index.
  """

  def __init__(self, feature_weight=1.0, relation='linking',
               linker=None, cache_filename='.linking_cache'):
    self.feature_weight = feature_weight
    self.relation = relation
    self.kgen = 1000
    self.krecog = 10000
    self.linker = linker
    self.cost = 1.0
    self.extra_cost = 3.0
    self.max_src_phrase_length = 6
    self.trg_phrase_length = 1
    self.filterq = None
    self.cache_filename = cache_filename
    self.cache = TryLoad(self.cache_filename)
    self.cache_has_changed = False
    manager = Manager()
    self.queue = manager.Queue()

  def Close(self):
    # Update cache with results from workers:
    self.cache_has_changed = not self.queue.empty()
    while not self.queue.empty():
      src_leaves_str, docs = self.queue.get()
      self.cache[src_leaves_str] = docs
    if self.cache_has_changed:
      TryDump(self.cache, self.cache_filename)

  def IsEligibleSrc(self, src_treep):
    if src_treep.HasVariables():
      return False
    src_leaves = filter_tokens(GetLeaves(src_treep))
    if not src_leaves:
      return False
    if not len(src_leaves) <= self.max_src_phrase_length:
      return False
    return True

  def IsEligible(self, src_treep, trg_treep):
    if not self.IsEligibleSrc(src_treep):
      return False
    if trg_treep is not None:
      trg_leaves = [leaf for leaf in GetLeaves(trg_treep) if not IsVariable(leaf)]
      num_trg_vars = trg_treep.GetNumVariables()
      if len(trg_leaves) != self.trg_phrase_length or num_trg_vars > 1:
        return False
    return True

  def GetDocs(self, src_leaves, context=None, fields=None, filterq=None, k=1000):
    """
    Retrieves a list of URIs that may correspond to the NL phrase.
    """
    if fields is None:
      fields = ['*', 'score']
    src_leaves_str = ' '.join(src_leaves)
    if src_leaves_str in self.cache:
      return self.cache[src_leaves_str][:k]
    if not self.linker.IsConnectionAlive():
      sys.exit(1)
    docs = self.linker.GetDocsFromPhrase(
      src_leaves, k=k, context=context, fields=fields, filterq=filterq)
    self.cache[src_leaves_str] = docs
    self.queue.put((src_leaves_str, docs))
    self.cache_has_changed = True
    return docs[:k]

  def GetURIs(self, src_leaves, context=None, filterq=None, k=1000):
    uri_candidate_docs = self.GetDocs(
      src_leaves, context=None, fields=['uri'], filterq=filterq, k=k)
    uri_candidates = [
      doc['uri'] for doc in uri_candidate_docs if not IsURITooLong(doc['uri'])]
    return uri_candidates

  def GetCostSimilarity(self, src_treep, trg_treep):
    src_leaves = [leaf for leaf in GetLeaves(src_treep) if not IsVariable(leaf)]
    trg_leaves = [
      leaf for leaf in GetLeaves(trg_treep) if not IsVariable(leaf) and leaf != '[]']
    uri = trg_leaves[-1].lstrip('!')
    uri_candidates = self.GetURIs(src_leaves, filterq=self.filterq, k=self.krecog)
    try:
      cost = 1.0 - 1.0 / (uri_candidates.index(uri) + 1)
      num_leaves = len(src_leaves) + len(trg_leaves)
      cost *= num_leaves
      if len(trg_leaves) > 1:
        cost += self.extra_cost
    except ValueError:
      cost = None
    return cost

  def GetScoreSimilarity(self, src_treep, trg_treep):
    src_leaves = [leaf for leaf in GetLeaves(src_treep) if not IsVariable(leaf)]
    trg_leaves = [
      leaf for leaf in GetLeaves(trg_treep) if not IsVariable(leaf) and leaf != '[]']
    uri = trg_leaves[-1].lstrip('!')
    uri_candidates = self.GetURIs(src_leaves, filterq=self.filterq, k=self.kgen)
    try:
      score = 1.0 / (uri_candidates.index(uri) + 1)
    except ValueError:
      score = None
    return score

  def GetSimilarity(self, tree_pattern1, tree_pattern2):
    """
    Computes similarity *cost* between natural language phrases
    and constituent structures that represent lambda-DCS subtrees.
    """
    similarities = []
    if not self.IsEligible(tree_pattern1, tree_pattern2):
      return similarities
    cost = self.GetCostSimilarity(tree_pattern1, tree_pattern2)
    if cost is not None:
      similarity = Similarity(cost, self.relation, tree_pattern1, tree_pattern2)
      similarities.append(similarity)
    return similarities

  def BuildTrgTreePatterns(self, src_treep):
    raise(NotImplementedError('BuildTrgTreePatterns is not yet implemented.'))

  def GetSimilar(self, src_treep):
    similarities = []
    if not self.IsEligibleSrc(src_treep):
      return similarities
    trg_treeps = self.BuildTrgTreePatterns(src_treep)
    for trg_treep in trg_treeps:
      score = self.GetScoreSimilarity(src_treep, trg_treep)
      if score is not None:
        similarity = Similarity(score, self.relation, src_treep, trg_treep)
        similarities.append(similarity)
    return sorted(similarities, key=lambda s: s.score, reverse=True)

class EntityLinkingCost(LinkingCost):
  """
  Implements the abstract methods from Similarity class. It measures the
  cost of mapping between a natural language phrase and entities.
  E.g. (NP (NN Marshall) (NN Hall)) -- "fb:en.marshall_hall"
  should have a low cost.
  """

  def __init__(self, feature_weight=1.0, kbest=1000,
               linker=None, cache_filename='.entity_linking_cache'):
    self.relation = 'entity'
    super(EntityLinkingCost, self).__init__(
      feature_weight, self.relation, linker, cache_filename)
    self.kgen = kbest
    self.max_src_phrase_length = 6
    self.trg_phrase_length = 1

  def BuildTrgTreePatterns(self, src_treep):
    src_leaves = GetLeaves(src_treep)
    # uri_candidate_docs = self.GetDocs(src_leaves, context=None, fields=['uri'])
    uri_candidates = self.GetURIs(src_leaves, k=self.kgen)
    path, subpaths = (), []
    trg_treeps = [TreePattern(tree_or_string(uri), path, subpaths) \
                    for uri in uri_candidates]
    return trg_treeps

class PredicateLinkingCost(LinkingCost):
  """
  Implements the abstract methods from Similarity class. It measures the
  cost of mapping between a natural language phrase and predicates.
  E.g. (NN issue) -- "!fb:comic_books.comic_book_issue.issue_number"
  should have a low cost.
  """

  def __init__(self, feature_weight=1.0, kbest=1000,
               linker=None, cache_filename='.predicate_linking_cache'):
    self.relation = 'predicate'
    super(PredicateLinkingCost, self).__init__(
      feature_weight, self.relation, linker, cache_filename)
    self.kgen = kbest
    self.max_src_phrase_length = 6
    self.trg_phrase_length = 1
    self.filterq = {'role' : 'predicate'}

  def BuildTrgTreePatterns(self, src_treep):
    src_leaves = GetLeaves(src_treep)
    uri_candidates_direct = self.GetURIs(
      src_leaves, filterq=self.filterq, k=self.kgen)
    uri_candidates = []
    for uri in uri_candidates_direct:
      uri_candidates.append(uri)
      uri_candidates.append('!' + uri)
    path, subpaths = (), []
    src_has_variables = src_treep.HasVariables()
    if src_has_variables:
      trg_treeps = [TreePattern(tree_or_string(u'(ID {0} ?x0|)'.format(uri)),
                                path, subpaths) for uri in uri_candidates]
    else:
      trg_treeps = [TreePattern(tree_or_string(uri), path, subpaths) \
                      for uri in uri_candidates]
    return trg_treeps

class BridgeLinkingCost(LinkingCost):
  """
  Implements the abstract methods from Similarity class. It measures the
  cost of mapping between a natural language phrase and entities.
  E.g. (NP (NN Marshall) (NN Hall)) -- "fb:en.marshall_hall"
  should have a low cost.
  """

  def __init__(self, feature_weight=1.0, kbest=1000,
               linker=None, cache_filename='.bridge_linking_cache'):
    self.relation = 'bridge_entity'
    super(BridgeLinkingCost, self).__init__(
      feature_weight, self.relation, linker, cache_filename)
    self.kgen = kbest
    self.max_src_phrase_length = 6
    self.trg_phrase_length = 2

  def BuildTrgTreePatterns(self, src_treep):
    src_leaves = GetLeaves(src_treep)
    uri_candidates = self.GetURIs(src_leaves, k=self.kgen)
    path, subpaths = (), []
    trg_treeps = [TreePattern(
                    tree_or_string(u'(ID [] {0})'.format(uri)), path, subpaths) \
                      for uri in uri_candidates]
    return trg_treeps

# TODO: this class should also use the linker object.
class CountOp(SimilarityScorer):
  """
  Implements the abstract methods from Similarity class.
  In recognition mode, it signals co-occurrences of "how many" with a "count"
  operator or with a predicate that retrieves integers.
  In generation mode, in presence of "how many" it inserts a "count"
  operator or a "number" unpacking predicate.
  """
  def __init__(self, feature_weight = 1.0, linker=None,
               cache_filename='.count_cache'):
    self.feature_weight = feature_weight
    self.kCostNegative = 1.0
    self.kCostNeutral = 0.0
    self.kCostPositive = 0.0
    self.relation = 'count'
    self.linker = linker
    self.cache_filename = cache_filename
    self.cache = TryLoad(self.cache_filename)
    self.cache_has_changed = False
    manager = Manager()
    self.queue = manager.Queue()

  def Close(self):
    # Update cache with results from workers:
    self.cache_has_changed = not self.queue.empty()
    while not self.queue.empty():
      uri, field, content = self.queue.get()
      if uri not in self.cache:
        self.cache[uri] = {}
      self.cache[uri][field] = content
    if self.cache_has_changed:
      TryDump(self.cache, self.cache_filename)

  def GetSimilarity(self, src_treep, trg_treep):
    """
    Fires if the source tree pattern contains at least two leaves that are
    ["how", "many"] and the target tree pattern has root "COUNT".
    """
    src_leaves = src_treep.GetLeaves()
    contains_how_many = ("how" in src_leaves and "many" in src_leaves)
    contains_count = self.ExpressionReturnsNumber(trg_treep)
    cost = self.kCostNeutral
    if contains_how_many and contains_count:
      cost = self.kCostPositive
    elif not contains_how_many and contains_count:
      cost = self.kCostNegative
    elif contains_how_many and not contains_count:
      cost = self.kCostNegative
    similarities = [Similarity(cost, self.relation, src_treep, trg_treep)]
    return similarities

  def GetSimilar(self, src_treep):
    raise ValueError('Not implemented')
    return [Similarity(self.kScore, None, word, None)]

  def GetURIFieldCached(self, uri, field):
    """
    Given a URI and a field name (e.g. "role" or "text"), it returns
    the value of such field, or None if:
      a) the URI is invalid or
      b) the field does not exist.
    The returned value depends on the field to be retrieved, and might
    be a string, a list of strings, a float or something else.
    """
    if uri in self.cache and field in self.cache[uri]:
      return self.cache[uri][field]
    content = self.linker.GetURIField(uri, field)
    if uri not in self.cache:
      self.cache[uri] = {}
    self.cache[uri][field] = content
    self.queue.put((uri, field, content))
    return content

  def ExpressionReturnsNumber(self, trg_treep):
    """
    Given a target tree pattern with a constituent representation of a SPARQL
    query, check if the data it returns is of type "number".
    """
    trg_leaves = trg_treep.GetLeaves()
    if not trg_leaves:
      return False
    main_pred = trg_leaves[0]
    if main_pred.lower() == 'count':
      return True
    role = self.GetURIFieldCached(main_pred, 'role')
    if role != 'predicate':
      return False
    arg = 'object' if main_pred.startswith('!') else 'subject'
    expr_type = self.GetURIFieldCached(main_pred, 'uri_type' + '|' + arg)
    if expr_type == 'number':
      return True
    return False

class NoSimilarityQA(SimilarityScorer):
  """
  Implements the abstract methods from Similarity class. 
  It returns a list with a single Similarity element. Such similarity
  element has relation q and cost proportional to the number of
  leaves in the source and the target.
  It is used to ensure that at least one transformation is produced between
  the source and the target, even if the transformation has a low score.
  """
  def __init__(self, feature_weight = 1.0, dels=1.0, ins=3.0, subs=3.0):
    self.feature_weight = feature_weight
    self.kScore = 1.0
    self.kDeletionCost = dels
    self.kInsertionCost = ins
    self.kSubstitutionCost = subs
    self.relation = 'q'

  def GetSimilarity(self, tree_pattern1, tree_pattern2):
    if tree_pattern1.IsString() and tree_pattern2.IsString():
      return [Similarity(self.kSubstitutionCost,
                         self.relation, tree_pattern1, tree_pattern2)]
    tree1_leaves = GetLeaves(tree_pattern1)
    tree2_leaves = GetLeaves(tree_pattern2)
    num_tree1_leaves = len(tree1_leaves)
    num_tree2_leaves = len(tree2_leaves)
    num_substitution_leaves = min(num_tree1_leaves, num_tree2_leaves)
    num_deletion_leaves = max(0, num_tree1_leaves - num_substitution_leaves)
    num_insertion_leaves = max(0, num_tree2_leaves - num_substitution_leaves)
    cost = num_substitution_leaves * self.kSubstitutionCost \
         + num_deletion_leaves * self.kDeletionCost \
         + num_insertion_leaves * self.kInsertionCost
    return [Similarity(cost, self.relation, tree_pattern1, tree_pattern2)]

  def GetSimilar(self, word):
    raise ValueError('Not implemented')
    return [Similarity(self.kScore, None, word, None)]

class NounPhraseCost(SimilarityScorer):
  """
  It returns a list with a single Similarity element. Such similarity
  element has relation q and cost set to 0.0 if the top of src_tree
  is an NP, and cost 1.0 otherwise. It might be useful to favor source tree
  patterns that are noun-phrases (e.g. for entity linking).
  """

  def __init__(self, feature_weight = 1.0, cost_np=-1.0, cost_no_np=0.0):
    self.feature_weight = feature_weight
    self.cost_np = cost_np
    self.cost_no_np = cost_no_np
    self.relation = 'q'

  def GetSimilarity(self, src_treep, trg_treep):
    similarities = []
    if src_treep.HasVariables():
      return similarities
    root_category = src_treep.GetRoot()
    cost = self.cost_np if root_category == 'NP' else self.cost_no_np
    similarities = [Similarity(cost, self.relation, src_treep, trg_treep)]
    return similarities

  def GetSimilar(self, word):
    raise ValueError('Not implemented')
    return [Similarity(self.kScore, None, word, None)]

def BuildTrgTreePatterns(bridges_and_relations):
  path, subpaths = (), []
  tree_patterns_and_relations = []
  for bridge_list, relation in bridges_and_relations:
    predicate_bridge, predicate_main = SetPredicateDirection(bridge_list, relation)
    tree = tree_or_string('(ID {0} {1})'.format(predicate_bridge, predicate_main))
    tree_pattern = TreePattern(tree, path, subpaths)
    tree_patterns_and_relations.append((tree_pattern, relation))
  return tree_patterns_and_relations

class UriSurfCost(SimilarityScorer):
  """
  Implements the abstract methods from Similarity class. 
  It returns a list with a single Similarity element. Such similarity
  element has relation {predicate, entity} and cost proportional to the
  surface overlap between source and target leaves. This technique can only
  be used to enhance the predicate/entity linking at recognition stage.
  Thus, it can be used only as a "global" (non rule-backoff) cost function.
  """
  def __init__(self, feature_weight=1.0, linker=None):
    self.feature_weight = feature_weight
    self.linker = linker
    self.extra_cost = 3.0
    self.max_src_phrase_length = 6
    self.max_trg_phrase_length = 2
    self.trg_token_separators = '._: '
    self.exclude_words = exclude_words
    self.predicate_relation = 'predicate'
    self.entity_relation = 'entity'
    self.bridge_relation = 'bridge_entity'
    self.relation = 'wildcard'
    self.ngram_min_ratio = .8
    self.cache = {}

  def IsEligible(self, src_treep, trg_treep):
    if src_treep.HasVariables():
      return False
    src_leaves = [leaf for leaf in GetLeaves(src_treep) if not IsVariable(leaf)]
    if not len(src_leaves) <= self.max_src_phrase_length:
      return False
    if trg_treep is not None:
      trg_leaves = [leaf for leaf in GetLeaves(trg_treep) if not IsVariable(leaf)]
      if len(trg_leaves) > self.max_trg_phrase_length:
        return False
    return True

  def GetURIRole(self, uri):
    if uri not in self.cache:
      roles = self.linker.GetURIField(uri, 'role')
      if not roles:
        role = None
      else:
        role = self.predicate_relation if 'predicate' in roles else self.entity_relation
      self.cache[uri] = role
    return self.cache[uri]

  def GetSimilarity(self, src_treep, trg_treep):
    """
    If 'predicate' is within the roles of the target URI, then the relation
    is labelled as self.predicate_relation. Otherwise, as self.entity_relation.
    Assuming the best possible cost to be 1.0 for the transformation of each
    source or target token, this cost function cannot give costs below that.
    In case the ngram ratio is 1.0 (perfect match of source into target; note
    it is asymmetric), then the cost to transform each token will be
    (2.1 - ngram_ratio) = 1.1
    Lower ngram ratios will give higher costs.
    The minimum ngram ratio that we consider for a mapping to be eligible
    is self.ngram_min_ratio.
    """
    similarities = []
    if not self.IsEligible(src_treep, trg_treep):
      return similarities
    src_leaves = GetLeaves(src_treep)
    trg_leaves = GetLeaves(trg_treep)
    uri = trg_leaves[-1]
    num_src_leaves = len(src_leaves)
    num_trg_leaves = len(trg_leaves)
    trg_leaves = SplitLeavesBy([uri], self.trg_token_separators)
    src_leaves = filter_tokens(src_leaves)
    trg_leaves = filter_tokens(trg_leaves)
    ngram_ratio = get_ngram_ratio(src_leaves, trg_leaves)
    if ngram_ratio >= self.ngram_min_ratio:
      cost = (2.1 - ngram_ratio) * (num_src_leaves + 1)
      if num_trg_leaves == 1:
        relation = self.GetURIRole(uri)
        if not relation:
          return similarities
      else:
        cost += self.extra_cost
        relation = self.bridge_relation
      if relation in [self.entity_relation, self.bridge_relation] and \
        not IsPlausibleEntityPhrase(src_treep):
        return similarities
      similarities = [Similarity(cost, relation, src_treep, trg_treep)]
    return similarities

  def GetSimilar(self, src_treep):
    """
    Using this technique, we are not capable of generating
    URIs given a source tree pattern. Thus, we return empty lists
    of similarities.
    """
    return []

