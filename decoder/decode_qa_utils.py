# -*- coding: utf-8 -*-
import codecs
import logging
from multiprocessing import Manager

from qald.dcs_tools import ConvertConstituent2DCS, ConvertDCS2Constituent
from qald.solr_entity_linking import GetURIField
from qald.sparql_utils import Query, QueryManager
from utils.cache import TryPickleLoad as TryLoad
from utils.cache import TryPickleDump as TryDump
from utils.production import TargetProjectionFromDerivation
from utils.tree_tools import IsString, tree_or_string

### Using third-party JAVA interface to Virtuoso ###
## Initiallize connection to Lambda-DCS query system by SEMPRE.
# from py4j.java_gateway import JavaGateway, GatewayClient
# from py4j.protocol import Py4JJavaError
# 
# gateway = JavaGateway(GatewayClient(address=u'localhost', port=3010))
# sparql_lambda_dcs_server = gateway.entry_point
# SempreQueryLambdaDCS = sparql_lambda_dcs_server.QueryLambdaDCS
# 
# def QueryLambdaDCSC(tree_str, query_manager=None):
#   assert IsString(tree_str)
#   tree_ldcs = ConvertConstituent2DCS(tree_str)
#   try:
#     results = SempreQueryLambdaDCS(str(tree_ldcs))
#   except Py4JJavaError:
#     results = None
#     logging.error('Py4JJavaError for l-dcsc: {0}'.format(tree_str))
#   return results
# 
# def QueryLambdaDCS(ldcs_tree_str):
#   assert IsString(ldcs_tree_str)
#   try:
#     results = SempreQueryLambdaDCS(ldcs_tree_str)
#   except Py4JJavaError as e:
#     results = None
#     logging.error('Py4JJavaError for l-dcs: {0}'.format(dcs_query))
#   return results

### Using native Python interface to Virtuoso ###
query_manager_global = None
def QueryLambdaDCSC(ldcsc_str, query_manager=None):
  assert IsString(ldcsc_str)
  if query_manager is None:
    query_manager = query_manager_global
  results = []
  ldcsc = tree_or_string(ldcsc_str)
  query = Query.fromldcsc(ldcsc)
  if query is not None:
    results = [r[0] for r in query.get_results(query_manager)]
  return results

def QueryLambdaDCS(ldcs_str):
  assert IsString(ldcs_str)
  ldcsc = ConvertDCS2Constituent(ldcs_str)
  return QueryLambdaDCSC(str(ldcsc))

def GetBestValidDerivations_(wrtg, cvt_inserter, nbest=1000, nvalid=100):
  """
  It obtains derivations in descending order of score from wRTG wrtg.
  It inserts CVTs when necessary.
  If a derivation produces a sparql query that retrieves an invalid
  result, then such derivation is skipped until a good derivation is found.
  The maximum number of explored derivations is given by nbest.
  It returns a list of up to nvalid valid derivations and corresponding
  constituent trees with a CVT inserted (when necessary).
  If not found, returns an empty list.
  """
  # This variable contains the result as a list of tuples.
  valid_derivations = []
  derivations = wrtg.ObtainDerivationsFromNT()
  for i, derivation in enumerate(derivations):
    if i >= nbest:
      break
    constituent, _ = TargetProjectionFromDerivation(derivation)
    if cvt_inserter:
      constituent = cvt_inserter.insert_cvt_if_needed(constituent)
    valid_derivations.append((derivation, constituent))
    if len(valid_derivations) >= nvalid:
      break
  return valid_derivations

invalid_results = [
  u'(list)', [], u'(list (number 0))', ["0"], u'BADFORMAT', u'SERVER500']

def GetBestValidDerivations_(wrtg, cvt_inserter, nbest=1000, nvalid=100):
  """
  It obtains derivations in descending order of score from wRTG wrtg.
  It inserts CVTs when necessary.
  If a derivation produces a sparql query that retrieves an invalid
  result, then such derivation is skipped until a good derivation is found.
  The maximum number of explored derivations is given by nbest.
  It returns a list of up to nvalid valid derivations and corresponding
  constituent trees with a CVT inserted (when necessary).
  If not found, returns an empty list.
  """
  # This variable contains the result as a list of tuples.
  valid_derivations = []
  invalid_derivations = []
  derivations = wrtg.ObtainDerivationsFromNT()
  for i, derivation in enumerate(derivations):
    if i >= nbest:
      break
    constituent, _ = TargetProjectionFromDerivation(derivation)
    if cvt_inserter:
      constituent = cvt_inserter.insert_cvt_if_needed(constituent)
    constituent_str = \
      constituent if IsString(constituent) else constituent.pprint(margin=10000)
    lambda_dcs_str = ConvertConstituent2DCS(constituent_str)
    try:
      query_results = QueryLambdaDCS(str(lambda_dcs_str))
    except Py4JJavaError:
      continue
    if query_results not in invalid_results:
      valid_derivations.append((derivation, constituent))
      if len(valid_derivations) >= nvalid:
        break
    else:
      invalid_derivations.append((derivation, constituent))
  # If the number of valid derivations has not reached nvalid,
  # then fill the rest of the slots with invalid derivations.
  num_remaining = nvalid - len(valid_derivations)
  valid_derivations.extend(invalid_derivations[:num_remaining])
  return valid_derivations

def GetBestValidDerivations(
  wrtg, cvt_inserter, nbest=1000, nvalid=100, query_manager=None):
  """
  It obtains derivations in descending order of score from wRTG wrtg.
  It inserts CVTs when necessary.
  If a derivation produces a sparql query that retrieves an invalid
  result, then such derivation is skipped until a good derivation is found.
  The maximum number of explored derivations is given by nbest.
  It returns a list of up to nvalid valid derivations and corresponding
  constituent trees with a CVT inserted (when necessary).
  If not found, returns an empty list.
  """
  # This variable contains the result as a list of tuples.
  valid_derivations = []
  derivations = wrtg.ObtainDerivationsFromNT()
  first_derivation, first_tree = None, None
  for i, derivation in enumerate(derivations):
    if i >= nbest:
      break
    constituent, _ = TargetProjectionFromDerivation(derivation)
    if cvt_inserter:
      constituent = cvt_inserter.insert_cvt_if_needed(constituent)
    constituent_str = \
      constituent if IsString(constituent) else constituent.pprint(margin=10000)
    query_results = QueryLambdaDCSC(constituent_str, query_manager)
    if query_results is None:
      continue
    if query_results not in invalid_results:
      valid_derivations.append((derivation, constituent))
      if len(valid_derivations) >= nvalid:
        break
    if i == 0:
      first_derivation = derivation
      first_tree = constituent
  if not valid_derivations and first_derivation is not None:
    valid_derivations.append((first_derivation, first_tree))
  return valid_derivations

def get_main_predicate_from_tree(tree):
  """
  Given a constituent representation of a sparql query,
  it returns the main predicate (as in lambda-DCS) by
  returning the left-most leaf. If "COUNT" operator is
  the left-most leaf, then it returns the leaf immediately
  on the right of the "COUNT" operator.
  """
  if IsString(tree):
    predicate = tree
  else:
    leaves = tree.leaves()
    assert leaves
    predicate = leaves[0] # left-most-leaf
    if predicate.lower() == 'count':
      predicate = leaves[1]
  return predicate

class CVTInserter(object):
  """
  Check if constituent representations of lambda-DCS sparql queries
  return a Compound Value Type (CVT). These concepts are only a KB
  construct to assert relationships between more than two concepts
  (e.g. number of employees in a company in a certain year).
  They can be heuristically recognized by checking whether the main
  predicate of the sparql expression is linked to one of the following CVTs:
    !fb:measurement_unit.dated_integer.number
    !fb:measurement_unit.dated_money_value.amount
    !fb:business.employment_tenure.person
    !fb:measurement_unit.money_value.amount
    !fb:measurement_unit.dated_money_value.valid_date
    !fb:measurement_unit.integer_range.high_value
    !fb:business.stock_ticker_symbol.ticker_symbol
    <and a few others>
  This CVTs are in full in qald/cvts.txt
  If that is the case, then the CVT is introduced as the main predicate
  of the sparql expression, and the former main predicate is subordinated
  to the CVT. Example:
  (ID main_pred entity)
  would be converted into
  (ID cvt (ID main_pred entity))
  Checking whether a predicate connects to a CVT can be done by querying
  Solr and retrieving the stored field of the main predicate {obj,subj}_subj.
  Since such query is a bit expensive (and we are going to do it millions
  of times, then it is better to cache the results in a dictionary that
  is shared across multiple processes.
  """

  def __init__(self, cache_filename='.cvt_cache'):
    self.cache_filename = cache_filename
    with codecs.open('qald/cvts.txt', 'r', 'utf-8') as f:
      self.cvts = set([cvt.strip() for cvt in f.readlines()])
    # The cache is a dictionary predicate : cvt
    # cvt is None if there is no cvt for such predicate.
    self.cache = TryLoad(self.cache_filename)
    self.cache_has_changed = False
    manager = Manager()
    self.queue = manager.Queue()

  def close(self):
    # Update cache with results from workers:
    self.cache_has_changed = not self.queue.empty()
    while not self.queue.empty():
      predicate, cvt = self.queue.get()
      self.cache[predicate] = cvt
    if self.cache_has_changed:
      TryDump(self.cache, self.cache_filename)

  def get_cvt_cached(self, uri_pred):
    if uri_pred not in self.cache:
      pred_connection = 'obj' if uri_pred.startswith('!') else 'subj'
      possible_cvts = GetURIField(uri_pred.lstrip('!'), pred_connection + '_subj')
      if not possible_cvts:
        self.cache[uri_pred] = None
      else:
        for cvt in self.cvts:
          if cvt in possible_cvts:
            self.cache[uri_pred] = cvt
            break
        else:
          self.cache[uri_pred] = None
      self.queue.put((uri_pred, self.cache[uri_pred]))
    return self.cache[uri_pred]

  def insert_cvt_if_needed(self, tree):
    predicate = get_main_predicate_from_tree(tree)
    cvt = self.get_cvt_cached(predicate)
    if cvt:
      if IsString(tree):
        tree = tree_or_string('(ID !{0} {1})'.format(cvt, tree))
      elif tree.label() == u'COUNT':
        tree = tree_or_string('(COUNT (ID !{0} {1}))'.format(cvt, tree[0]))
      elif not IsString(tree[0]):
        tree_repr = ' '.join(map(str, tree))
        tree = tree_or_string('(ID !{0} {1})'.format(cvt, tree_repr))
      else:
        tree = tree_or_string('(ID !{0} {1})'.format(cvt, tree))
    return tree

