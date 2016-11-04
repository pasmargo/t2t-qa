from __future__ import print_function
import logging
from multiprocessing import Manager, Pool
import sys
from urllib2 import URLError

from SPARQLWrapper import SPARQLWrapper, JSON

from utils.cache import TryJsonLoad as TryLoad, TryJsonDump as TryDump
# from utils.cache import TryPickleLoad as TryLoad, TryPickleDump as TryDump
from utils.cache import update_cache_and_save
from utils.tree_tools import Tree, IsString, get_top

kSPARQLendpoint = 'http://localhost:3001/sparql'

class Query(object):
  """
  Models a Sparql query.
  """

  def __init__(self, query_str):
    self.query_str = query_str
    self.results = None
    self.query_vars = None
    self.ldcsc = None

  def get_results(self, query_manager=None):
    if self.results is None:
      if not self.query_str:
        self.results = []
      else:
        if query_manager is None:
          self.results = GetQueryResults(self.query_str)
        else:
          self.results = query_manager.GetQueryResultsCached(self.query_str)
    return self.results

  def __repr__(self):
    return self.query_str

  def __str__(self):
    return self.query_str

  @staticmethod
  def fromstring(query_str):
    self.query_str = query_str

  @staticmethod
  def fromldcsc(ldcsc, var_prefs=None):
    """
    other_query_vars_prefs is a list with the prefixes
    of the other query variable instantiations that we
    also want to retrieve. E.g. if ['p', 'r'], then
    SELECT DISTINCT ?x0 , ?p0, ?p1, ?r0, ?r1 WHERE { ...
    """
    if not isinstance(ldcsc, Tree):
      if IsString(ldcsc) and not ldcsc.startswith('(') and not ldcsc.endswith(')'):
        return None
      else:
        raise(ValueError(
          'This method expects a Tree instance. Got type {0} for instance {1}'.format(
          type(ldcsc), ldcsc)))
    try:
      statements = get_statements_from_ldcsc(ldcsc)
    except:
      logging.warning('Failed to get statements from l-dcsc: {0}'.format(str(ldcsc)))
      statements = []
    if not statements:
      return None
    operator = ldcsc[0] if is_operator(ldcsc[0]) else ""
    query_vars = get_query_vars(statements, var_prefs)
    query_str = build_query_str(
      statements, '?x0', query_vars, operator)
    query = Query(query_str)
    query.query_vars = query_vars
    query.ldcsc = ldcsc
    return query

def build_query_str(statements, trg_var, query_vars, operator, limit=10):
  query_vars_str = u', '.join(query_vars)
  if query_vars_str:
    query_vars_str = u', ' + query_vars_str
  prefix = u'PREFIX fb: <http://rdf.freebase.com/ns/>'
  if len(query_vars) > 1 or not operator:
    select = u'SELECT DISTINCT {0} as ?answer{1} WHERE {{'.format(
      trg_var, query_vars_str)
  else:
    select = u'SELECT DISTINCT {1}({0}) as ?answer{2} WHERE {{'.format(
      trg_var, operator, query_vars_str)
  # return prefix + '\n' + select + '\n' + '\n'.join(map(str, statements)) + '} limit '
  return u'{0}\n{1}\n{2}}} LIMIT {3} #'.format(
    prefix, select, u'\n'.join(map(unicode, statements)), limit)

def is_operator(op):
  if isinstance(op, Tree):
    return False
  assert IsString(op)
  return op == "COUNT" or \
         op == "MAX" or \
         op == "MIN"

def get_statements_from_date(ldcsc, var):
  assert get_top(ldcsc) == 'DATE' and len(ldcsc) == 1
  statements = []
  if not IsString(ldcsc[0]):
    return statements
  try:
    year = int(ldcsc[0].split('_')[0])
  except ValueError:
    return statements
  statements = [
    'FILTER (xsd:dateTime({0}) >= xsd:dateTime("{1}"^^xsd:datetime)) .'\
    .format(var, year),
    'FILTER (xsd:dateTime({0}) < xsd:dateTime("{1}"^^xsd:datetime)) .'\
    .format(var, year + 1)]
  return statements

def get_statements_from_number(ldcsc, var):
  statements = []
  return statements

def get_number_from_constituent(ldcsc):
  assert get_top(ldcsc) == 'NUMBER'
  dummy_number = '?n0'
  if IsString(ldcsc):
    return dummy_number
  return ldcsc[0] if IsString(ldcsc[0]) else dummy_number

def get_statements_from_ldcsc(ldcsc, var_counter=0):
  statements = []
  if not isinstance(ldcsc, Tree):
    return statements
  if get_top(ldcsc) in ['DATE', 'NUMBER']:
    return statements
  if IsString(ldcsc[0]) and not is_operator(ldcsc[0]):
    new_var = '?x' + str(var_counter)
    pred = ldcsc[0].strip('!')
    if IsString(ldcsc[1]):
      entity_or_var = ldcsc[1]
    elif get_top(ldcsc[1]) == 'DATE':
      entity_or_var = '?d0'
      statements.extend(get_statements_from_date(ldcsc[1], '?d0'))
    elif get_top(ldcsc[1]) == 'NUMBER':
      entity_or_var = get_number_from_constituent(ldcsc[1])
      # statements.extend(get_statements_from_number(ldcsc[1], '?n0'))
    else:
      entity_or_var = '?x' + str(var_counter + 1)
    if IsString(ldcsc[0]) and ldcsc[0].startswith('!'):
      subj, obj = entity_or_var, new_var
    else:
      subj, obj = new_var, entity_or_var
    s = Statement(subj, pred, obj)
    statements.append(s)
    var_counter += 1
  subtree_ini_index = 1 if IsString(ldcsc[0]) else 0
  for subtree in ldcsc[subtree_ini_index:]:
    statements.extend(get_statements_from_ldcsc(subtree, var_counter))
  return statements

def get_query_vars(statements, prefixes):
  if prefixes is None:
    prefixes = []
  query_vars = list()
  for s in statements:
    if IsString(s):
      continue
    if is_var(s.subj):
      query_vars.append(s.subj)
    if is_var(s.rel):
      query_vars.append(s.rel)
    if is_var(s.obj):
      query_vars.append(s.obj)
  out_vars = set()
  for pref in prefixes:
    for v in query_vars:
      if v.startswith(pref):
        out_vars.add(v)
  return sorted(out_vars)

class Statement(object):
  """
  Models a Sparql statement. E.g:
  ?x1 fb:married_to ?x2 .
  """

  def __init__(self, subj, rel, obj):
    self.subj = subj
    self.rel = rel
    self.obj = obj

  def __repr__(self):
    return u'\t{0}\t{1}\t{2} .'.format(self.subj, self.rel, self.obj)

  def __str__(self):
    return self.__repr__()

def is_var(entity):
  return entity.startswith('?')

def IsDisambiguator(uri, sparql_endpoint=kSPARQLendpoint):
  query_str = u'PREFIX fb: <http://rdf.freebase.com/ns/> ' \
              u'select distinct ?r where { ' \
              + uri + u' fb:freebase.property_hints.disambiguator ?r . }'
  results = GetQueryResults(query_str, sparql_endpoint)
  return len(results) > 0 and u'1' in results[0]

def IsURI(possible_uri):
  if '://' in possible_uri:
    return True
  return False

def NormalizeCamelCase(label):
  new_label = ""
  for char in label:
    if not char.islower():
      new_label = new_label + '_' + char.lower()
    else:
      new_label = new_label + char
  return new_label

def NormalizeURILabel(label):
  new_label = label.lower()
  new_label = re.sub(' ', '_', new_label)
  return new_label

def shorten_uri(uri):
  if not uri.startswith('http'):
    return uri
  content = uri.split('/')[-1]
  shortened_uri = 'fb:' + content
  return shortened_uri

def ConvertURIToLabel(uri, kSPARQLendpoint):
  sparql = SPARQLWrapper(kSPARQLendpoint)
  query_string = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> " \
                 + "SELECT ?label WHERE { " \
                 + uri + " rdfs:label ?label }"
  sparql.setQuery(query_string)
  sparql.setReturnFormat(JSON)
  results = sparql.query().convert()
  if not results['results']['bindings']:
    return None
  labels = [NormalizeURILabel(res['label']['value']) \
              for res in results['results']['bindings']]
  if isinstance(labels, list):
    return labels[0]
  return labels

"""
For fb:zoos.zoo.num_animals ?p ?o
http://rdf.freebase.com/ns/type.property.expected_type  http://rdf.freebase.com/ns/type.int
For fb:freebase.type_profile.instance_count ?p ?o
http://rdf.freebase.com/ns/type.property.expected_type  http://rdf.freebase.com/ns/type.int
For fb:cricket.cricket_umpire.test_matches_refereed ?p ?o
http://rdf.freebase.com/ns/type.property.expected_type  http://rdf.freebase.com/ns/type.int
For fb:measurement_unit.dated_integer.number ?p ?o
http://rdf.freebase.com/ns/type.property.expected_type  http://rdf.freebase.com/ns/type.int
For fb:architecture.building.floors ?p ?o:
http://rdf.freebase.com/ns/type.property.expected_type  http://rdf.freebase.com/ns/type.int
"""

class QueryManager(object):
  """
  It exposes a static method to run queries on a Sparql endpoint,
  and another method that does caching. However, when the cache
  becomes big, it takes time to load it and save it to disk.
  For that reason, caching has to be activated explicitly, for
  batched queries.
  """
  def __init__(self, cache_fname='.sparql_cache'):
    self.cache_fname = cache_fname
    self.cache_has_changed = False
    manager = Manager()
    self.cache_queue = manager.Queue()
    self.cache = TryLoad(self.cache_fname)

  def __enter__(self):
    self.cache = TryLoad(self.cache_fname)
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

  def close(self):
    update_cache_and_save(
      self.cache, self.cache_queue, self.cache_fname, TryDump)

  @staticmethod
  def GetQueryResults(query_string):
    return GetQueryResults(query_string)

  def GetQueryResultsCached(self, query_string):
    if query_string not in self.cache:
      result = GetQueryResults(query_string)
      # If result is None, there is a Sparql connection
      # problem and the result should not be cached.
      if result is None:
        return []
      self.cache[query_string] = result
      self.cache_queue.put((query_string, result))
    return self.cache[query_string]

qmanager = None
kMaxTasksPerChild = None

import time
def GetQueryResultsWithGlobalManager(query):
  tstart = time.time()
  results = query.get_results(qmanager)
  tend = time.time()
  logging.debug('{0:.1f} secs. to process the query:\n{1}\n{2}'.format(
    tend - tstart, str(query), str(query.ldcsc)))
  return results

def GetQueriesResultsSeq(queries):
  results = []
  for query in queries:
    result = GetQueryResultsWithGlobalManager(query)
    results.append(result)
  return results

def GetQueriesResultsPar(queries, ncores):
  pool = Pool(processes=ncores, maxtasksperchild=kMaxTasksPerChild)
  queries_results = pool.map(GetQueryResultsWithGlobalManager, queries)
  pool.close()
  pool.join()
  return queries_results

def GetQueriesResults(queries, query_manager, ncores):
  """
  Given a list of Query objects and a query_manager,
  it returns a list of results for each query.
  """
  global qmanager
  qmanager = query_manager
  if ncores <= 1:
    return GetQueriesResultsSeq(queries)
  return GetQueriesResultsPar(queries, ncores)

def GetQueryResults(query_string, kSPARQLendpoint=kSPARQLendpoint):
  """
  Connect to Virtuoso in localhost to retrieve results for query string.
  It returns a list of results, or empty list if the query_string
  does not yield any result.
  It outputs None if the connection to Virtuoso server fails.
  """
  output = []
  if not query_string:
    return output
  sparql = SPARQLWrapper(kSPARQLendpoint)
  sparql.setQuery(query_string + ' limit 1000000')
  sparql.setReturnFormat(JSON)
  try:
    results = sparql.query().convert()
  except URLError:
    logging.error(
      'Broken connection to virtuoso. Failed to process:\n{0}'.format(query_string))
    results = None
    return results
  except:
    results = {'results' : {'bindings': []}}
  if not results['results']['bindings'] or results['results']['bindings'] == [{}]:
    return output
  query_vars  = results['head']['vars']
  for result in results['results']['bindings']:
    try:
      variable_assignment = [result[qvar]['value'] for qvar in query_vars]
    except KeyError:
      print('KeyError in "variable_assignment = [result[qvar][value] for qvar in query_vars]"',
        file=sys.stderr)
      print('Query string: {0}'.format(query_string), file=sys.stderr)
      print('Results: {0}'.format(results), file=sys.stderr)
      variable_assignment = []
    output.append(variable_assignment)
  return output

# Retrieve transitive relations (up to any length) for a certain entity.
# select distinct ?y where {
#   <http://www4.wiwiss.fu-berlin.de/drugbank/resource/drugs/DB00437> owl:sameAs+ ?y .
# } limit 50

# Query to retrieve the URIs whose label contain a certain regular expression.
# select ?s where {
#   ?s rdfs:label ?l .
#   filter regex(?l, "nteraction")
# } limit 5

# Query to retrieve the type of the subject and object of a predicate:
# select distinct ?stype ?otype where {
#   ?s <http://www4.wiwiss.fu-berlin.de/drugbank/resource/drugbank/target> ?o .
#   ?s rdf:type ?stype .
#   ?o rdf:type ?otype .
# } 
