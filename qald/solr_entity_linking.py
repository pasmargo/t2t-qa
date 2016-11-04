import logging
from httplib import BadStatusLine
from urllib2 import urlopen, URLError, HTTPError
import simplejson
import sys

from qald.grounding import GetURIType
from utils.tree_tools import IsString

kPort = 8123
kBaseUrl = 'http://kmcs.nii.ac.jp/solr_pasmargo_el/select?q='

def GetURIField(uri, field):
  """
  Retrieves information of URIs or words according to the index of Freebase.
  URIs that are prefixed with "!" are stripped to remove that operator.
  """
  if field == 'numFound':
    try:
      assert IsConnectionAlive()
      words = [] if uri == '<total>' else [uri]
      num_docs = GetNumDocsFound(words)
    except ValueError:
      num_docs = 0
    return num_docs
  if field.startswith('uri_type'):
    arg = field.split('|')[1]
    return GetURIType(uri, arg)
  if not IsString(uri):
    return None
  if field == 'role':
    return GetURIRole(uri)
  elif field == 'text':
    return GetURIText(uri)
  try:
    uri_field = GetFieldFromURI(uri, field)
  except ValueError:
    uri_field = None
  return uri_field 

def GetURIText(uri):
  """
  Return lowercased words in "text" field of Solr index for URI uri.
  """
  # The field "text" in Solr is a list of strings.
  try:
    assert IsConnectionAlive()
    text = GetFieldFromURI(uri.lstrip('!'), 'text')
  except ValueError:
    return None
  return [word.lower() for text_chunk in text for word in text_chunk.split()]

def GetURIRole(uri):
  if uri.startswith('!'):
    return 'predicate'
  try:
    # At least one of the features requires a connection to Solr.
    # Check if such connection is active.
    assert IsConnectionAlive()
    roles = GetFieldFromURI(uri.lstrip('!'), 'role')
  except ValueError:
    return None
  return 'predicate' if 'predicate' in roles else 'entity' # subject or object

def GetNumDocsFound(terms):
  """
  returns a single integer, indicating how many documents were
  found by Solr given a certain phrase (list of terms).
  If the list of terms is empty, then it returns the total number
  of documents in Solr index.
  """
  docs_found = 0
  assert isinstance(terms, list) or isinstance(terms, tuple)
  norm_terms = [NormalizeString(term) for term in terms if term.strip() != ""]
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

def GetFieldFromURI(uri, field_name):
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
  documents = GetDocsFromURLQuery(url_str)
  assert len(documents) <= 1, 'Too many docs for {0}'.format(uri)
  if len(documents) == 0:
    raise ValueError('No results for URI {0}'.format(uri))
  field_value = documents[0].get(field_name, None)
  return field_value

def NormalizeString(phrase):
  phrase = phrase.replace('!', '')
  phrase = phrase.replace(':', '')
  phrase = phrase.replace('#', '')
  phrase = phrase.replace('&', '%26') # After XML and URL encoding.
  phrase = phrase.strip()
  return phrase

def GetURIsFromPhrase(terms, k=1000, context=None):
  """
  Returns a sorted list of URIs that map natural language phrases.
  It requires an active tunnel to kmcs server on kPort.
  """
  documents = GetDocsFromPhrase(terms, k, context=context, fields=['uri'])
  uris = [document['uri'] for document in documents]
  return uris

def GetURIsAndScoresFromPhrase(terms, k=1000):
  """
  Returns a sorted list of URIs that map natural language phrases.
  It requires an active tunnel to kmcs server on kPort.
  """
  documents = GetDocsFromPhrase(terms, k, context=None, fields=['uri', 'score'])
  uris_and_scores = [(document['uri'], document['score']) for document in documents]
  return uris_and_scores

def GetDocsFromPhrase(terms, k=1000, context=None, fields=['*', 'score'], filterq=None):
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
  norm_terms = [NormalizeString(term) for term in terms if term.strip() != ""]
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
  logging.info(url_str)
  documents = GetDocsFromURLQuery(url_str)
  return documents

def ScoreURIByTerms(uri, terms):
  """
  This function assumes that the terms in the list are normalized.
  """
  # The text string is joined with HTML encoded whitespaces.
  text_str = '+'.join(['text:' + term for term in terms])
  url_str = kBaseUrl + 'uri%3A{0}+{1}&wt=json&rows=1&indent=true&fl=uri,score'\
    .format(uri.replace(':', '%5C%3A'), # Replace ":" by "\:"
            text_str)
  logging.info(url_str)
  documents = GetDocsFromURLQuery(url_str)
  assert len(documents) == 1
  score = float(documents[0]['score'])
  return score
 
def GetDocsFromURLQuery(url_str):
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

def GetBridges(terms, k_best=1000, context=None):
  """
  Returns a list of bridges and target entity/predicates.
  E.g. INPUT: ['Marshall', 'Hall']
       OUTPUT: [('fb:education.academic_post.person', 'fb:en.marshall_hall'), ...]
  """
  assert isinstance(terms, list)
  bridge_fields = ['preds_when_subj', 'preds_when_obj',
    'subj_subj', 'subj_obj', 'obj_subj', 'obj_obj']
  docs = GetDocsFromPhrase(terms, k=k_best, context=context)
  bridges_scores = []
  for doc in docs:
    uri = doc['uri']
    score = doc['score']
    for field in bridge_fields:
      candidate_bridges = doc.get(field, [])
      candidate_bridges_scores = RankURIsByTerms(candidate_bridges, terms)
      for bridge, bridge_score in candidate_bridges_scores:
        if bridge != uri:
          bridges_scores.append(((bridge, uri), score, bridge_score, score + bridge_score))
  bridges_scores_sorted = sorted(bridges_scores, key=lambda x: x[3], reverse=True)
  return bridges_scores_sorted
  # bridges = [x[0] for x in bridges_scores_sorted[:k_best]]
  # return bridges

def RankURIsByTerms(uris, terms):
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
  norm_terms = [NormalizeString(term) for term in terms if term.strip() != ""]
  norm_terms = [t for t in norm_terms if t != ""]
  # If there are no terms, then there is no ranking.
  if not norm_terms:
    return zip(uris, [0.0] * len(uris))
  uris_scores = []
  for uri in uris:
    score = ScoreURIByTerms(uri, norm_terms)
    uris_scores.append((uri, score))
  uris_scores_sorted = sorted(uris_scores, key=lambda x: x[1], reverse=True)
  return uris_scores_sorted

def IsConnectionAlive():
  try:
    conn = urlopen(kBaseUrl + '*%3A*%0A&wt=xml&indent=true')
    return True
  except URLError:
    logging.error('Connection to Solr is not possible. Exiting.')
    return False
