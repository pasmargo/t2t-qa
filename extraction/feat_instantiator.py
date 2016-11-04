# -*- coding: utf-8 -*-
import codecs
from collections import defaultdict
from math import log
import multiprocessing
from multiprocessing import Manager, Value, Lock

from linguistics.similarity_dictionary_part import SplitLeavesBy
from qald.lexicon import exclude_words
from qald.grounding import _GetURIField
from utils.cache import TryJsonLoad, TryJsonDump
from utils.tree_tools import (TreePattern, get_top, IsString, variables_to_paths,
  tree_or_string)

lock = Lock()
# Will contain the shared resource that describes each feature.
# Sub-processes will inherit it.
descriptions = None

def LoadDescriptions(description_filename, manager):
  """
  Load feature descriptions from a file into a dictionary:
  d : description -> feature id.
  It returns the dictionary, a counter (generator) and
  the reverse of the dictionary.
  The counter is useful to set the unique ID of the next new feature.
  """
  global descriptions
  descriptions_unmanaged = TryJsonLoad(description_filename)
  descriptions = manager.dict()
  descriptions.update(descriptions_unmanaged)
  current_feat_counter = len(descriptions_unmanaged)
  if __debug__:
    # Check that the next feature ID follows the feature ID stored in disk.
    vals = descriptions_unmanaged.values()
    largest_feat_id = 0 if not vals else max(vals)
    assert current_feat_counter == largest_feat_id, \
      '{0} vs. {1}'.format(current_feat_counter, largest_feat_id)
  # Shared feature counter across sub-processes. It is essential to keep
  # a consistent feature_name to feature ID mapping.
  feats_counter = Value('i', current_feat_counter)
  return descriptions_unmanaged, feats_counter

class FeatureInstantiator(object):
  """ Extract features from LHS and RHS of rules. """

  feature_templates = [
    'roots',
    # 'identity',
    # 'yield_words',
    'num_variables',
    'num_del_variables',
    'num_leaves',
    'num_leaves_diff',
    'uri_role',
    'state',
    'newstates',
    'surf_to_uri_info',
    'surface_overlap',
    'trg_surface_overlap',
    'trg_surface_overlap_to_context',
    'fb_typed_size',
    # 'tf_idf',
    'count_op',
    'src_context_to_uri',
    'undeleted_how_many']
  def __init__(self, description_filename, feats_cache_filename='.feats_cache',
               feat_names_filename=''):
    self.feats_cache_filename = feats_cache_filename
    self.description_filename = description_filename
    manager = Manager()
    self.descriptions, self.feats_counter = \
      LoadDescriptions(self.description_filename, manager)
    self.id2desc = None
    self.sync_id2desc()
    self.n_grams = 3
    self.stop_words = exclude_words + ['how', 'many']
    self.uri_token_seps = ':._'
    self.feats_cache = TryJsonLoad(feats_cache_filename)
    self.feats_cache_queue = manager.Queue()
    if feat_names_filename:
      self.feature_templates = \
        [name.strip() for name in codecs.open(feat_names_filename, 'r', 'utf-8')]
    # Context under which features are extracted.
    self.context = {}
    # Feature obtained by sempre's authors.
    self.fb_typed_size_filename = 'qald/fb_typed_size_filename.json'
    self.fb_typed_size = {}
    if 'fb_typed_size' in self.feature_templates:
      self.fb_typed_size = TryJsonLoad(self.fb_typed_size_filename)

  def Close(self):
    global lock
    # Retrieve items from shared resource and dump them into file.
    with lock:
      local_descr = descriptions.copy()
    TryJsonDump(local_descr, self.description_filename)
    # First, update cache with results from sub-processes,
    # and then write them into file.
    feat_cache_has_changed = not self.feats_cache_queue.empty()
    while not self.feats_cache_queue.empty():
      uri, field, content = self.feats_cache_queue.get()
      if uri not in self.feats_cache:
        self.feats_cache[uri] = {}
      self.feats_cache[uri][field] = content
    if feat_cache_has_changed:
      feats_cache = dict(self.feats_cache)
      TryJsonDump(feats_cache, self.feats_cache_filename)

  def SetContext(self, raw_context):
    """
    Curates the raw_context dictionary into an extended and more structured
    context dictionary. At the moment, raw_context may contain the following
    entries:
    * src_tree : constituent representation of source tree (string).
    * trg_tree : constituent representation of target tree (string).
    """
    self.ClearContext()
    src_tree_str = raw_context.get('src_tree', None)
    if src_tree_str:
      src_tree = tree_or_string(src_tree_str)
      self.context['src_words'] = src_tree.GetLeaves()
      self.context['answer_type'] = GetAnswerType(self.context['src_words'])

  def ClearContext(self):
    self.context = {}

  def GetFeatureIDByName_(self, name):
    return self.descriptions.get(repr(name), -1)

  def GetFeatureIDByName(self, name):
    global lock
    name_repr = repr(name)
    feat_id = self.descriptions.get(name_repr, -1)
    if feat_id < 0:
      # Retrieve the synchronized description.
      with lock:
        if descriptions.has_key(name_repr):
          feat_id = descriptions[name_repr]
    return feat_id

  def DescribeFeatureIDs_(self, rule):
    """
    Return a tab-separated string of the form:
      feat_id    feat_description
    for all features. The parameter *rule* has an attribute *features*
    which is a list of lists:
      [[feat_id, feat_val, ...], [feat_id, feat_val, ...], ...]
    Descriptions are listed in descending feature weight (last item).
    """
    if not rule.feat_descr is None:
      return rule.feat_descr
    if rule.features is None:
      return ''
    feats = sorted(rule.features, key=lambda x: x[-1], reverse=True)
    id_descr_str = '\n'.join(
      ['\t'.join(map(str, feat)) + '\t' + \
       str(self.id2desc[feat[0]]) for feat in feats])
    rule.feat_descr = id_descr_str
    return id_descr_str

  def sync_id2desc(self):
    """
    Feature descriptions are synchronized, but not this inverse index.
    For that reason, we need to call it explicitly after running feature
    instantiation in parallel, if we want to keep id2desc up-to-date (as
    in the case of decoding, but not in training).
    """
    self.id2desc = {ind : desc for desc, ind in self.descriptions.items()}

  def DescribeFeatureIDs(self, rule):
    """
    Return a tab-separated string of the form:
      feat_id    feat_description
    for all features. The parameter *rule* has an attribute *features*
    which is a list of lists:
      [[feat_id, feat_val, ...], [feat_id, feat_val, ...], ...]
    Descriptions are listed in descending feature weight (last item).
    """
    if not rule.feat_descr is None:
      return rule.feat_descr
    if rule.features is None:
      return ''
    feats = sorted(rule.features, key=lambda x: x[-1], reverse=True)
    id_descr_strs = []
    for feat in feats:
      id_descr_str = '\t'.join(map(str, feat))
      feat_id = feat[0]
      feat_descr = self.id2desc.get(feat_id, "<no description>")
      id_descr_str += '\t' + feat_descr
      id_descr_strs.append(id_descr_str)
    return '\n'.join(id_descr_strs)

  def GetURIFieldCached(self, uri, field):
    """
    Given a URI and a field name (e.g. "role" or "text"), it returns
    the value of such field, or None if:
      a) the URI is invalid or
      b) the field does not exist.
    The returned value depends on the field to be retrieved, and might
    be a string, a list of strings, a float or something else.
    """
    if uri in self.feats_cache and field in self.feats_cache[uri]:
      return self.feats_cache[uri][field]
    content = _GetURIField(uri, field)
    if uri not in self.feats_cache:
      self.feats_cache[uri] = {}
    self.feats_cache[uri][field] = content
    self.feats_cache_queue.put((uri, field, content))
    return content

  def get_next_feat_id(self):
    with self.feats_counter.get_lock():
      self.feats_counter.value += 1
      return self.feats_counter.value

  def register_new_features(self, features):
    """
    Assign unique IDs to new features and keep dictionaries synchronized.
    In order to avoid too many calls to the shared resource that contains
    the feature descriptions, we make local copies of the dictionary into
    each process. If a key is not in the local copy of the dictionary,
    then we check the shared resource. If a new entry is to be added,
    we added to both the local copy and the shared resource.
    """
    global descriptions, lock
    for name, _ in features:
      name_repr = unicode(repr(name))
      if self.descriptions.has_key(name_repr):
        # There is no need to assign a new feature ID. Skip.
        continue
      with lock:
        is_there_key = descriptions.has_key(name_repr)
        if is_there_key:
        # We need to update the local copy of the descriptions.
          feat_id = descriptions[name_repr]
        else:
          # This is a completely new feature. We assign a feature ID to the
          # local and remote copy.
          feat_id = self.get_next_feat_id()
          descriptions[name_repr] = feat_id
      self.descriptions[name_repr] = feat_id
      self.id2desc[feat_id] = name_repr

  def code_features(self, features, allow_new):
    if allow_new:
      self.register_new_features(features)
    # Convert feature descriptions into their unique identifiers.
    coded_features = []
    for name, feat_val in features:
      name_repr = repr(name)
      if name_repr in self.descriptions:
        coded_features.append([self.descriptions[name_repr], feat_val])
    return coded_features

  def InstantiateFeatures(self, src_treep, trg_treep, rule=None, allow_new=True):
    """ Creates a list of tuples with feature_id and feature_value.

    Args:
        src_treep: TreePattern from the source tree.
        trg_treep: TreePattern from the target tree.
        This pair of tree pattern implicitly defines a rule,
        except for the state name, which is currently left out,
        but for which I have plans to include somehow in the future.
    """
    features = []
    if 'roots' in self.feature_templates:
      features.extend(self.roots(src_treep, trg_treep))
    if 'identity' in self.feature_templates:
      features.extend(self.identity(src_treep, trg_treep))
    if 'yield_words' in self.feature_templates:
      features.extend(self.yield_words(src_treep, trg_treep))
    if 'num_variables' in self.feature_templates:
      features.extend(self.num_variables(src_treep, trg_treep))
    if 'num_del_variables' in self.feature_templates:
      features.extend(self.num_del_variables(src_treep, trg_treep))
    if 'num_leaves' in self.feature_templates:
      features.extend(self.num_leaves(src_treep, trg_treep))
    if 'num_leaves_diff' in self.feature_templates:
      features.extend(self.num_leaves_diff(src_treep, trg_treep))
    if 'uri_role' in self.feature_templates:
      features.extend(self.uri_role(src_treep, trg_treep))
    if 'state' in self.feature_templates:
      features.extend(self.state(rule))
    if 'newstates' in self.feature_templates:
      features.extend(self.newstates(rule))
    if 'surf_to_uri_info' in self.feature_templates:
      features.extend(self.surf_to_uri_info(src_treep, trg_treep))
    if 'surface_overlap' in self.feature_templates:
      features.extend(self.surface_overlap(src_treep, trg_treep))
    if 'trg_surface_overlap' in self.feature_templates:
      features.extend(self.trg_surface_overlap(src_treep, trg_treep))
    if 'trg_surface_overlap_to_context' in self.feature_templates:
      features.extend(self.trg_surface_overlap_to_context(src_treep, trg_treep))
    if 'tf_idf' in self.feature_templates:
      features.extend(self.tf_idf(src_treep, trg_treep))
    if 'count_op' in self.feature_templates:
      features.extend(self.count_op(src_treep, trg_treep))
    if 'src_context_to_uri' in self.feature_templates:
      features.extend(self.src_context_to_uri(src_treep, trg_treep))
    if 'undeleted_how_many' in self.feature_templates:
      features.extend(self.undeleted_how_many(src_treep, trg_treep))
    if 'fb_typed_size' in self.feature_templates:
      features.extend(self.get_fb_typed_size(src_treep, trg_treep))
    if 'type_match' in self.feature_templates:
      features.extend(self.type_match(src_treep, trg_treep))
    coded_features = self.code_features(features, allow_new)
    return coded_features

  def roots(self, src_treep, trg_treep):
    src_root = src_treep.GetRoot()
    trg_root = trg_treep.GetRoot()
    features = [(('root: lhs', src_root), 1.0),
                (('root: rhs', trg_root), 1.0),
                (('root: lhs, rhs', (src_root, trg_root)), 1.0)]
    return features

  def identity(self, src_treep, trg_treep):
    src_repr = repr(src_treep.ObtainTreePattern())
    trg_repr = repr(trg_treep.ObtainTreePattern())
    features = [(('identity: lhs', src_repr), 1.0),
                (('identity: rhs', trg_repr), 1.0),
                (('identity: lhs, rhs', (src_repr, trg_repr)), 1.0)]
    if src_repr == trg_repr:
      features.append(('identity: lhs and rhs are equal', 1.0))
    return features

  def yield_words(self, src_treep, trg_treep):
    src_leaves = tuple(src_treep.GetLeaves())
    trg_leaves = tuple(trg_treep.GetLeaves())
    common_leaves = tuple(set(src_leaves).intersection(set(trg_leaves)))
    features = [(('yield: lhs', src_leaves), 1.0),
                (('yield: rhs', trg_leaves), 1.0),
                (('yield: lhs, rhs', (src_leaves, trg_leaves)), 1.0),
                (('yield: common in lhs and rhs', common_leaves), 1.0)]
    return features

  def num_variables(self, src_treep, trg_treep):
    src_num_vars = len(src_treep.subpaths)
    trg_num_vars = len(trg_treep.subpaths)
    src_trg_num_vars = abs(src_num_vars - trg_num_vars)
    features = [(('num_variables: lhs', src_num_vars), 1.0),
                (('num_variables: rhs', trg_num_vars), 1.0),
                (('num_variables: abs(lhs - rhs)', src_trg_num_vars), 1.0)]
    return features

  def num_del_variables(self, src_treep, trg_treep):
    src_vars = [var for var, path in variables_to_paths(src_treep.tree) \
                  if var.startswith('?xx')]
    trg_vars = [var for var, path in variables_to_paths(trg_treep.tree) \
                  if var.startswith('?xx')]
    src_num_del_vars = len(src_vars)
    trg_num_del_vars = len(trg_vars)
    src_trg_num_del_vars = abs(src_num_del_vars - trg_num_del_vars)
    features = [(('num_del_variables: lhs', src_num_del_vars), 1.0),
                (('num_del_variables: rhs', trg_num_del_vars), 1.0),
                (('num_del_variables: abs(lhs - rhs)', src_trg_num_del_vars), 1.0)]
    return features

  def num_leaves(self, src_treep, trg_treep):
    src_num_leaves = len(src_treep.GetLeaves())
    trg_num_leaves = len(trg_treep.GetLeaves())
    src_trg_num_leaves = abs(src_num_leaves - trg_num_leaves)
    features = [(('num_leaves: lhs', src_num_leaves), 1.0),
                (('num_leaves: rhs', trg_num_leaves), 1.0),
                (('num_leaves: abs(lhs - rhs)', src_trg_num_leaves), 1.0)]
    return features

  def num_leaves_diff(self, src_treep, trg_treep):
    src_num_leaves = len(src_treep.GetLeaves())
    trg_num_leaves = len(trg_treep.GetLeaves())
    src_trg_num_leaves = abs(src_num_leaves - trg_num_leaves)
    features = [(('num_leaves_diff: abs(lhs - rhs) >=', i), 1.0) \
                  for i in range(3, min(10, src_trg_num_leaves + 1), 2)]
    return features

  def uri_role(self, src_treep, trg_treep):
    trg_leaves = trg_treep.GetLeaves()
    if not trg_leaves:
      roles = tuple()
    else:
      roles = [self.GetURIFieldCached(uri, 'role') for uri in trg_leaves]
      roles = tuple([str(r) for r in roles if r is not None])
    features = [(('uri_role: rhs', roles), 1.0)]
    return features

  def state(self, rule):
    features = [(('state', rule.state), 1.0)] if rule is not None else []
    return features

  def newstates(self, rule):
    features = []
    if rule is not None:
      sorted_newstates = \
        [x[1] for x in sorted(rule.newstates.items(), key=lambda x: x[0])]
      features = [(('newstates', tuple(sorted_newstates)), 1.0)]
    return features

  def surf_to_uri_info(self, src_treep, trg_treep):
    """
    Compute overlap of self.n_grams characters of source strings and the
    information associated to the target URIs such as label, alias or description.
    Then, normalize by the number of self.n_grams in source.
    We remove stop words from source NL phrase and words from the information
    associated to the URIs.
    """
    src_leaves = [w for w in src_treep.GetLeaves() if w not in self.stop_words]
    trg_leaves = trg_treep.GetLeaves()
    features = []
    if not (1 <= len(trg_leaves) <= 2) or not src_leaves:
      return features
    texts = [self.GetURIFieldCached(uri, 'text') for uri in trg_leaves]
    doc_words = [word.lower() for text in texts if text is not None \
                   for word in text if word not in self.stop_words]
    if not doc_words:
      return features
    ngram_ratio = get_ngram_ratio(src_leaves, doc_words, n=self.n_grams+1)
    features = [('surf_to_uri_info', ngram_ratio)]
    return features

  def surface_overlap(self, src_treep, trg_treep):
    """
    Compute overlap of self.n_grams characters of source strings and target URI,
    and normalize by the number of self.n_grams in source.
    """
    src_leaves = [w for w in src_treep.GetLeaves() if w not in self.stop_words]
    trg_leaves = trg_treep.GetLeaves()
    features = []
    if not (1 <= len(trg_leaves) <= 2) or not src_leaves:
      return features
    ngram_ratio = get_ngram_ratio(src_leaves, trg_leaves, n=self.n_grams)
    features = [('surface_match', ngram_ratio)]
    return features

  def trg_surface_overlap(self, src_treep, trg_treep):
    """
    Compute overlap of self.n_grams characters of target URI and source leaves,
    and normalize by the number of self.n_grams in target.
    It uses only the right-most part of the URI, which is the
    one that actually contains the specific information to that predicate.
    """
    src_leaves = [w for w in src_treep.GetLeaves() if w not in self.stop_words]
    trg_leaves = [uri.split('.')[-1] for uri in trg_treep.GetLeaves()]
    features = []
    if not (1 <= len(trg_leaves) <= 2) or not src_leaves:
      return features
    ngram_ratio = get_ngram_ratio(trg_leaves, src_leaves, n=self.n_grams)
    features = [('trg_surface_match', ngram_ratio)]
    return features

  def trg_surface_overlap_to_context(self, src_treep, trg_treep):
    """
    Compute overlap of self.n_grams characters of target URI and source leaves,
    and normalize by the number of self.n_grams in target.
    It uses only the right-most part of the URI, which is the
    one that actually contains the specific information to that predicate.
    """
    src_leaves = self.context.get('src_words', [])
    trg_leaves = [uri.split('.')[-1] for uri in trg_treep.GetLeaves()]
    features = []
    if not (1 <= len(trg_leaves) <= 2) or not src_leaves:
      return features
    ngram_ratio = get_ngram_ratio(trg_leaves, src_leaves, n=self.n_grams)
    features = [('trg_surface_match_to_context', ngram_ratio)]
    return features

  def tf_idf(self, src_treep, trg_treep):
    features = []
    src_leaves = src_treep.GetLeaves()
    trg_leaves = trg_treep.GetLeaves()
    if not trg_leaves or not src_leaves:
      return features
    texts = [self.GetURIFieldCached(uri, 'text') for uri in trg_leaves]
    doc_words = [word for text in texts if text is not None for word in text]
    if not doc_words:
      return features
    tfs = {word : self.compute_tf(word, doc_words) for word in src_leaves}
    idfs = {word : self.compute_idf(word) for word in src_leaves}
    tfidf = sum([tfs[w] * idfs[w] for w in src_leaves]) / len(src_leaves)
    # We divide by 40 to get some sort of normalization wrt other features.
    features = [('linking_tfidf', float(int(round(tfidf))) / 40)]
    return features

  def compute_tf(self, word, doc_words):
    """
    Computes augmented term frequency to prevent bias towards longer documents:
    https://en.wikipedia.org/wiki/Tf%E2%80%93idf
    It consists in dividing the term frequency by the frequency of the term
    with maximum frequency.
    """
    freqs = defaultdict(float)
    for doc_word in doc_words:
      freqs[doc_word] += 1.0
    max_word_freq = sorted(freqs.values(), reverse=True)[0]
    tf = .5 + .5 * freqs[word] / max_word_freq
    return tf
  
  def compute_idf(self, word):
    """
    Computes the Inverse Document Frequency, that is, the logarithm
    of the number of documents divided by the number of documents in
    which that "word" appears.
    """
    num_docs_for_word = self.GetURIFieldCached(word, 'numFound')
    num_total_docs = self.GetURIFieldCached('<total>', 'numFound')
    idf = log(float(num_total_docs) / (1 + num_docs_for_word))
    return idf

  def count_op(self, src_treep, trg_treep):
    """
    Fires if the source tree pattern contains at least two leaves that are
    ["how", "many"] and the target tree pattern has root "COUNT".
    """
    features = []
    src_leaves = src_treep.GetLeaves()
    trg_leaves = trg_treep.GetLeaves()
    contains_how_many = ("how" in src_leaves and "many" in src_leaves)
    contains_count = ("COUNT" in trg_leaves or self.ExpressionReturnsNumber(trg_treep))
    if contains_how_many and contains_count:
      features = [(('count_op', True), 1.0)]
    elif not contains_how_many and contains_count:
      features = [(('count_op', False), 1.0)]
    elif contains_how_many and not contains_count:
      features = [(('count_op', False), 1.0)]
    return features

  def type_match(self, src_treep, trg_treep):
    """
    Fires if the source tree pattern contains at least two leaves that are
    ["how", "many"] and the target tree pattern has root "COUNT".
    """
    features = []
    context_src_words = self.context.get('src_words', [])
    if not context_src_words:
      return features
    main_pred = get_main_pred_from_treep(trg_treep)
    if not main_pred:
      return features
    role = self.GetURIFieldCached(main_pred, 'role')
    if role != 'predicate':
      return features
    arg = 'object' if main_pred.startswith('!') else 'subject'
    expr_type = self.GetURIFieldCached(main_pred, 'uri_type' + '|' + arg)
    if 'who' in context_src_words and expr_type == 'person':
      features = [('who_person', 1.0)]
    elif 'where' in context_src_words and expr_type == 'location':
      features = [('where_location', 1.0)]
    elif 'when' in context_src_words and expr_type == 'date':
      features = [('when_date', 1.0)]
    return features

  def ExpressionReturnsNumber(self, trg_treep):
    """
    Given a target tree pattern with a constituent representation of a SPARQL
    query, check if the data it returns is of type "number".
    """
    main_pred = get_main_pred_from_treep(trg_treep)
    if not main_pred:
      return False
    role = self.GetURIFieldCached(main_pred, 'role')
    if role != 'predicate':
      return False
    arg = 'object' if main_pred.startswith('!') else 'subject'
    expr_type = self.GetURIFieldCached(main_pred, 'uri_type' + '|' + arg)
    if expr_type == 'number':
      return True
    return False

  def src_context_to_uri(self, src_treep, trg_treep):
    """
    Fires if the source context 'answer_type' contains a keyword that is present
    in the URI surface.
    """
    features = []
    trg_leaves = trg_treep.GetLeaves()
    trg_tokens = SplitLeavesBy(trg_leaves, self.uri_token_seps)
    answer_types, trg_token_candidates = self.context.get('answer_type', ([], []))
    common_tokens = set(trg_tokens).intersection(set(trg_token_candidates))
    if common_tokens:
      features.append((('src_context_to_uri', tuple(answer_types)), 1.0))
    return features

  def undeleted_how_many(self, src_treep, trg_treep):
    """
    Fires if the source tree pattern contains "how many" in the leaves.
    """
    features = []
    src_leaves = src_treep.GetLeaves()
    contains_how_many = ("how" in src_leaves and "many" in src_leaves)
    if contains_how_many:
      features.append((('undeleted_how_many', True), 1.0))
    return features

  def get_fb_typed_size(self, src_treep, trg_treep):
    """
    Returns the FB_typed_size feature from sempre's binary predicates.
    This feature is constant for each predicate URI.
    """
    features = []
    trg_leaves = trg_treep.GetLeaves()
    if len(trg_leaves) != 1:
      return features
    uri = trg_leaves[0].strip('!')
    uri_fb_typed_size = self.fb_typed_size.get(uri, 0.0)
    norm_size = log(uri_fb_typed_size) if uri_fb_typed_size > 0.0 else 0.0
    features.append(('fb_typed_size', norm_size))
    return features

def get_main_pred_from_treep(treep):
  leaves = treep.GetLeaves()
  if not leaves:
    return False
  main_pred = leaves[0]
  return main_pred

def DiscretizeRatio(ratio):
  if 0 <= ratio < .2:
    ratio_discr = '0<=ratio<20'
  elif .2 <= ratio < .4:
    ratio_discr = '20<=ratio<40'
  elif .4 <= ratio < .6:
    ratio_discr = '40<=ratio<60'
  elif .6 <= ratio < .8:
    ratio_discr = '60<=ratio<80'
  elif .8 <= ratio:
    ratio_discr = '80<=ratio'
  return ratio_discr

def get_ngrams(words, n=2, ignore_chars='._:'):
  """
  Get the number of N-grams of length n in the list of words,
  ignoring the characters ignore_chars.
  If a word has length smaller than n, then it is counted
  as an ngram of length n.
  """
  ngrams = defaultdict(int)
  for word in words:
    if len(word) < n:
      ngrams[word] += 1
    for ngram in [word[i:i+n] for i in range(len(word) - n + 1)]:
      if not any([ic in ngram for ic in ignore_chars]):
        ngrams[ngram] += 1
  return ngrams

def get_ngram_ratio(src, trg, n=2, ignore_chars='._:'):
  """
  Compute ratio overlap_n_gram / src_ngrams.
  """
  assert isinstance(src, list) and isinstance(trg, list)
  src_ngrams = get_ngrams(src, n, ignore_chars)
  trg_ngrams = get_ngrams(trg, n, ignore_chars)
  total_src_ngrams = sum(src_ngrams.values())
  non_overlapping_ngrams = 0
  for src_ngram in src_ngrams:
    src_count = src_ngrams[src_ngram]
    trg_count = trg_ngrams[src_ngram]
    non_overlapping_ngrams += max(0, src_count - trg_count)
  if total_src_ngrams == 0:
    ngram_ratio = 0.0
  else:
    ngram_ratio = float(total_src_ngrams - non_overlapping_ngrams) / total_src_ngrams
  return ngram_ratio

def GetAnswerType(words):
  if 'when' in words:
    return (['when'], ['date'])
  if 'where' in words:
    return (['where'], ['area', 'country', 'location', 'place', 'site'])
  if 'how' in words and 'many' in words:
    return (['how', 'many'], ['number'])
  if 'how' in words and 'much' in words:
    return (['how', 'much'], ['number'])
  return ([], [])

