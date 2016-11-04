import logging
from httplib import BadStatusLine
from multiprocessing import Manager
from urllib2 import urlopen, URLError, HTTPError
import simplejson
import sys

from qald.sparql_utils import IsDisambiguator, GetQueryResults
from utils.cache import TryPickleLoad as TryLoad, TryPickleDump as TryDump
from utils.cache import update_cache_and_save
from utils.tree_tools import IsString

kPort = 8123
kBaseUrl = 'http://kmcs.nii.ac.jp/solr_pasmargo_el/select?q='

class Linker(object):
  """
  Entity/Predicate linker. It attempts to define a generic
  interface for other linking engines. Here we use a combination
  of Solr and Sparql, and cache results.
  """

  def __init__(self, uri_cache_fname='.uri_cache', disamb_cache_fname='.disamb_cache'):
    manager = Manager()
    self.uri_cache = TryLoad(uri_cache_fname)
    self.uri_cache_queue = manager.Queue()
    self.uri_cache_fname = uri_cache_fname
    self.disamb_cache = TryLoad(disamb_cache_fname)
    self.disamb_cache_queue = manager.Queue()
    self.disamb_cache_fname = disamb_cache_fname

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

  def close(self):
    update_cache_and_save(
      self.uri_cache, self.uri_cache_queue, self.uri_cache_fname, TryDump)
    update_cache_and_save(
      self.disamb_cache, self.disamb_cache_queue, self.disamb_cache_fname, TryDump)

  def GetURIField(self, uri, field):
    uri_field = (uri, field)
    if uri_field not in self.uri_cache:
      result = _GetURIField(uri, field)
      self.uri_cache_queue.put((uri_field, result))
      self.uri_cache[uri_field] = result
    return self.uri_cache[uri_field]

  def IsURIDisambiguator(self, uri):
    if uri not in self.disamb_cache:
      result = IsDisambiguator(uri)
      self.disamb_cache_queue.put((uri, result))
      self.disamb_cache[uri] = result
    return self.disamb_cache[uri]

  def GetURIText(self, uri):
    return _GetURIText(uri)

  def GetURIRole(self, uri):
    return _GetURIRole(uri)

  def GetNumDocsFound(self, terms):
    return _GetNumDocsFound(terms)

  def GetFieldFromURI(self, uri, field_name):
    return _GetFieldFromURI(uri, field_name)

  def NormalizeString(self, phrase):
    return _NormalizeString(phrase)

  def GetURIsFromPhrase(self, terms, k=1000, context=None):
    return _GetURIsFromPhrase(terms, k, context)

  def GetURIsAndScoresFromPhrase(self, terms, k=1000):
    return _GetURIsAndScoresFromPhrase(terms, k)

  def GetDocsFromPhrase(
    self, terms, k=1000, context=None, fields=None, filterq=None, explain=False):
    return _GetDocsFromPhrase(terms, k, context, fields, filterq, explain)

  def ScoreURIByTerms(self, uri, terms):
    return _ScoreURIByTerms(uri, terms)

  def GetDocsFromURLQuery(self, url_str):
    return _GetDocsFromURLQuery(url_str)

  def GetBridges(self, terms, k_best=1000, context=None):
    return _GetBridges(terms, k_best, context)

  def RankURIsByTerms(self, uris, terms):
    return _RankURIsByTerms(uris, terms)

  def IsConnectionAlive(self):
    return _IsConnectionAlive()

def _GetURIField(uri, field):
  """
  Retrieves information of URIs or words according to the index of Freebase.
  URIs that are prefixed with "!" are stripped to remove that operator.
  """
  if field == 'numFound':
    try:
      assert _IsConnectionAlive()
      words = [] if uri == '<total>' else [uri]
      num_docs = _GetNumDocsFound(words)
    except ValueError:
      num_docs = 0
    return num_docs
  if field.startswith('uri_type'):
    arg = field.split('|')[1]
    return GetURIType(uri, arg)
  if not IsString(uri):
    return None
  if field == 'role':
    return _GetURIRole(uri)
  elif field == 'text':
    return _GetURIText(uri)
  try:
    uri_field = _GetFieldFromURI(uri, field)
  except ValueError:
    uri_field = None
  return uri_field 

def GetURIType(uri, arg='object'):
  """
  If the URI is a predicate, it returns the type of its argument "arg", where
  "arg" is one of ['subject', 'object'].
  If the URI is an entity, it returns its own type.
  """
  role = _GetURIRole(uri)
  if role == "predicate":
    return GetPredicateURIType(uri, arg)
  elif role == "entity":
    return GetEntityURIType(uri)
  return None

def guess_type_from_uri_surf(uri):
  if 'location' in uri or 'country' in uri:
    return 'location'
  if 'time' in uri:
    return 'date'
  if 'people' in uri or 'person' in uri:
    return 'person'
  if uri.endswith('author'):
    return 'person'
  return None

def GetEntityURIType(entity):
  # First, try to guess from the morphology of the URI.
  # URIs of relations or entities sometimes show their
  # types in the first portions of the URI. E.g.:
  # fb:location.country or fb:time.event.start_date
  return guess_type_from_uri_surf(entity)

def get_expected_types(uri):
  if not uri.startswith('fb:'):
    uri = '<{0}>'.format(uri)
  query_string = "PREFIX fb: <http://rdf.freebase.com/ns/> " \
                 + "SELECT ?t WHERE { " \
                 + uri + " fb:type.property.expected_type ?t }"
  results = GetQueryResults(query_string)
  if not results:
    return []
  return [r[0] for r in results]

def get_included_types(uri):
  if not uri.startswith('fb:'):
    uri = '<{0}>'.format(uri)
  query_string = "PREFIX fb: <http://rdf.freebase.com/ns/> " \
                 + "SELECT ?t WHERE { " \
                 + uri + " fb:freebase.type_hints.included_types ?t }"
  results = GetQueryResults(query_string)
  if not results:
    return []
  return [r[0] for r in results]

# from linguistics.linguistic_tools import ObtainHypernyms
# def is_instance_of_person(uri):
#   e = uri.split('.')[-1].split('_')[-1]
#   hypernyms = ObtainHypernyms(e)
#   return 'person' in hypernyms

def includes_person_type(uri):
  included_types = get_included_types(uri)
  if any([it.endswith('people.person') for it in included_types]):
    return True
  return False

def include_person_type(uri):
  expected_types = get_expected_types(uri)
  if any([is_instance_of_person(e) for e in expected_types]):
    return True
  for expected_type in expected_types:
    included_types = get_included_types(expected_type)
    if any([it.endswith('people.person') for it in included_types]):
      return True
  return False

def is_number(uri):
  return uri.endswith('type.int') or uri.endswith('type.float')

def is_date(uri):
  return uri.endswith('datetime') or 'time' in uri

def is_location(uri):
  return 'location' in uri

def GetPredicateURIType(uri, arg='object'):
  guessed_type = guess_type_from_uri_surf(uri)
  if guessed_type is not None:
    return guessed_type
  # If previous guess has failed, then do some basic reasoning over the KB.
  uri = uri.strip('!')
  expected_types = []
  if arg == 'object':
    expected_types = get_expected_types(uri)
  if any([is_number(u) for u in expected_types]):
    return 'number'
  if any([is_date(u) for u in expected_types]):
    return 'date'
  if any([is_location(u) for u in expected_types]):
    return 'location'
  # if any([is_instance_of_person(u) for u in expected_types]):
  #   return 'person'
  if any([includes_person_type(u) for u in expected_types]):
    return 'person'
  return None

def GetPredicateURIType_(uri, arg='object'):
  guessed_type = guess_type_from_uri_surf(uri)
  if guessed_type is not None:
    return guessed_type
  # If previous guess has failed, then do some basic reasoning over the KB.
  uri = uri.strip('!')
  query_string = ''
  if arg == 'object':
    # expected_types = get_expected_types(uri)
    query_string = "PREFIX fb: <http://rdf.freebase.com/ns/> " \
                   + "SELECT ?t WHERE { " \
                   + uri + " fb:type.property.expected_type ?t }"
  results = GetQueryResults(query_string)
  if not results:
    return None
  uri_type = results[0][0]
  if uri_type.endswith('type.int') or uri_type.endswith('type.float'):
    return 'number'
  if uri_type.endswith('datetime'):
    return 'date'
  if include_person_type(uri):
    return 'person'
  return None

def _GetURIText(uri):
  """
  Return lowercased words in "text" field of Solr index for URI uri.
  """
  # The field "text" in Solr is a list of strings.
  try:
    assert _IsConnectionAlive()
    text = _GetFieldFromURI(uri.lstrip('!'), 'text')
  except ValueError:
    return None
  return [word.lower() for text_chunk in text for word in text_chunk.split()]

def _GetURIRole(uri):
  if uri.startswith('!'):
    return 'predicate'
  try:
    # At least one of the features requires a connection to Solr.
    # Check if such connection is active.
    assert _IsConnectionAlive()
    roles = _GetFieldFromURI(uri.lstrip('!'), 'role')
  except ValueError:
    return None
  return 'predicate' if 'predicate' in roles else 'entity' # subject or object

def _GetNumDocsFound(terms):
  """
  returns a single integer, indicating how many documents were
  found by Solr given a certain phrase (list of terms).
  If the list of terms is empty, then it returns the total number
  of documents in Solr index.
  """
  docs_found = 0
  assert isinstance(terms, list) or isinstance(terms, tuple)
  norm_terms = [_NormalizeString(term) for term in terms if term.strip() != ""]
  norm_terms = [t for t in norm_terms if t != ""]
  if not norm_terms:
    text = '*:*'
  else:
    text = '+'.join(['text:' + term for term in norm_terms])
  url_str = kBaseUrl + '{0}&wt=json&rows=0'.format(text)
  # Connect to Solr and retrieve number of documents.
  try:
    conn = urlopen(url_str)
  except HTTPError as e:
    logging.info('HTTPError {0}\nwhen querying with {1}'.format(e, url_str))
    return docs_found
  except BadStatusLine as e:
    logging.info('BadStatusLine {0}\nwhen querying with {1}'.format(e, url_str))
    return docs_found
  results = simplejson.load(conn)
  docs_found = int(results['response']['numFound'])
  return docs_found

def _GetFieldFromURI(uri, field_name):
  """
  Given a URI and a field name, it returns the value of such field.
  If the URI is not in Solr's index, it raises a ValueError.
  If the field_name is not among the URI's document attributes,
  it returns None.
  There is no attempt to convert the field_value into a common type.
  Thus, the result might be a list, a string, or a number.
  """
  url_str = kBaseUrl + 'uri%3A{0}&wt=json&rows=1&indent=true&fl={1}'\
    .format(uri.replace(':', '%5C%3A'), field_name)
  documents = _GetDocsFromURLQuery(url_str)
  assert len(documents) <= 1, 'Too many docs for {0}'.format(uri)
  if len(documents) == 0:
    raise ValueError('No results for URI {0}'.format(uri))
  field_value = documents[0].get(field_name, None)
  return field_value

def _NormalizeString(phrase):
  phrase = phrase.replace('!', '')
  phrase = phrase.replace(':', '')
  phrase = phrase.replace('#', '')
  phrase = phrase.replace('&', '%26') # After XML and URL encoding.
  phrase = phrase.strip()
  return phrase

def _GetURIsFromPhrase(terms, k=1000, context=None):
  """
  Returns a sorted list of URIs that map natural language phrases.
  It requires an active tunnel to kmcs server on kPort.
  """
  documents = GetDocsFromPhrase(terms, k, context=context, fields=['uri'])
  uris = [document['uri'] for document in documents]
  return uris

def _GetURIsAndScoresFromPhrase(terms, k=1000):
  """
  Returns a sorted list of URIs that map natural language phrases.
  It requires an active tunnel to kmcs server on kPort.
  """
  documents = _GetDocsFromPhrase(terms, k, context=None, fields=['uri', 'score'])
  uris_and_scores = [(document['uri'], document['score']) for document in documents]
  return uris_and_scores

def _GetDocsFromPhrase(terms, k=1000, context=None, fields=None, filterq=None,
  explain=False):
  """
  Query Solr's index in kmcs using fields:
  1. uri_surf (x3 boosted)
  2. label (x2 boosted)
  3. alias (x2 boosted)
  4. name (x2 boosted)
  5. description (x1 boosted).
  6. context, if any (x0.2 boosted).
  Returns a sorted list of documents that map natural language phrases.
  It requires an active tunnel to kmcs server on kPort.
  The paramter filterq is a dictionary that contains the attributes of the
  filter query. E.g. {'role' : 'predicate'} (to filter by predicate roles.
  """
  assert isinstance(terms, list) or isinstance(terms, tuple)
  if fields is None:
    fields = ['*', 'score']
  norm_terms = [_NormalizeString(term) for term in terms if term.strip() != ""]
  norm_terms = [t for t in norm_terms if t != ""]
  if not norm_terms:
    return []
  phrase = '+'.join(['uri_surf:' + term + '^3' for term in norm_terms])
  label = '+'.join(['label:' + term + '^2' for term in norm_terms])
  alias = '+'.join(['alias:' + term + '^2' for term in norm_terms])
  name = '+'.join(['name:' + term + '^2' for term in norm_terms])
  description = '+'.join(['description:' + term for term in norm_terms])
  query_params = \
    '{0}+{1}+{2}+{3}+{4}'.format(phrase, label, alias, name, description)
  if context is not None:
    context_str = '+'.join(['text:' + term + '^2' for term in norm_terms])
    query_params += '+' + context_str
  fields_str = ','.join(fields)
  url_str = kBaseUrl + '{0}&wt=json&rows={1}&indent=true&fl={2}'\
    .format(query_params, k, fields_str)
  if filterq:
    filterq_str = ''.join(['&fq={0}:{1}'.format(k, v) for k, v in filterq.items()])
    url_str += filterq_str
  if explain:
    url_str += '&debugQuery=true'
  logging.info(url_str)
  documents = _GetDocsFromURLQuery(url_str)
  return documents

def _ScoreURIByTerms(uri, terms):
  """
  This function assumes that the terms in the list are normalized.
  """
  # The text string is joined with HTML encoded whitespaces.
  text_str = '+'.join(['text:' + term for term in terms])
  url_str = kBaseUrl + 'uri%3A{0}+{1}&wt=json&rows=1&indent=true&fl=uri,score'\
    .format(uri.replace(':', '%5C%3A'), # Replace ":" by "\:"
            text_str)
  logging.info(url_str)
  documents = _GetDocsFromURLQuery(url_str)
  assert len(documents) == 1
  score = float(documents[0]['score'])
  return score
 
def _GetDocsFromURLQuery(url_str):
  documents = []
  try:
    conn = urlopen(url_str)
  except HTTPError as e:
    logging.info('HTTPError {0}\nwhen querying with {1}'.format(e, url_str))
    return documents
  except BadStatusLine as e:
    logging.info('BadStatusLine {0}\nwhen querying with {1}'.format(e, url_str))
    return documents
  results = simplejson.load(conn)
  documents = results['response']['docs']
  return documents

def _GetBridges(terms, k_best=1000, context=None):
  """
  Returns a list of bridges and target entity/predicates.
  E.g. INPUT: ['Marshall', 'Hall']
       OUTPUT: [('fb:education.academic_post.person', 'fb:en.marshall_hall'), ...]
  """
  assert isinstance(terms, list)
  bridge_fields = ['preds_when_subj', 'preds_when_obj',
    'subj_subj', 'subj_obj', 'obj_subj', 'obj_obj']
  docs = _GetDocsFromPhrase(terms, k=k_best, context=context)
  bridges_scores = []
  for doc in docs:
    uri = doc['uri']
    score = doc['score']
    for field in bridge_fields:
      candidate_bridges = doc.get(field, [])
      candidate_bridges_scores = _RankURIsByTerms(candidate_bridges, terms)
      for bridge, bridge_score in candidate_bridges_scores:
        if bridge != uri:
          bridges_scores.append(((bridge, uri), score, bridge_score, score + bridge_score))
  bridges_scores_sorted = sorted(bridges_scores, key=lambda x: x[3], reverse=True)
  return bridges_scores_sorted
  # bridges = [x[0] for x in bridges_scores_sorted[:k_best]]
  # return bridges

def _RankURIsByTerms(uris, terms):
  """
  Given a set of URIs, e.g.:
  ["fb:freebase.flag_judgment.item",
   "fb:business.employment_tenure.title",
   "fb:education.academic_post.position_or_title",
   "fb:fictional_universe.fictional_character.occupation",
   "fb:freebase.user_profile.task",
   "fb:freebase.user_profile.person",
   "fb:people.person.profession"]
  this function returns a list of tuples [(uri, score), ...] sorted
  in descending score (URIs with higher scores are better).
  """
  norm_terms = [_NormalizeString(term) for term in terms if term.strip() != ""]
  norm_terms = [t for t in norm_terms if t != ""]
  # If there are no terms, then there is no ranking.
  if not norm_terms:
    return zip(uris, [0.0] * len(uris))
  uris_scores = []
  for uri in uris:
    score = _ScoreURIByTerms(uri, norm_terms)
    uris_scores.append((uri, score))
  uris_scores_sorted = sorted(uris_scores, key=lambda x: x[1], reverse=True)
  return uris_scores_sorted

def _IsConnectionAlive():
  try:
    conn = urlopen(kBaseUrl + '*%3A*%0A&wt=xml&indent=true')
    return True
  except URLError:
    logging.error('Connection to Solr is not possible. Exiting.')
    return False
