from __future__ import print_function

import argparse
import codecs
from functools import cmp_to_key
import locale
import logging
from multiprocessing import Pool
from simplejson import dumps
import signal
import sys
import time
import traceback

from decoder.decode_qa_utils import QueryLambdaDCSC
from extraction.feat_instantiator import FeatureInstantiator
from qald.dcs_tools import ConvertConstituent2DCS
from qald.sparql_utils import QueryManager
from training.loadrules import loadrules, GetInitialState
from training.transducer import xT
from training.wrtgs import ObtainWRTGs, ObtainWRTGsAugmented
from utils.tree_tools import IsString, tree_or_string

# Timeout mechanism for decoder. Taken from:
# http://stackoverflow.com/questions/25027122/break-the-function-after-certain-time
class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException

# Change the behavior of SIGALRM
signal.signal(signal.SIGALRM, timeout_handler)

feat_inst = None
lm = None
rule_backoffs = []
timeout = None
nbest = None
cvt_inserter = None
kMaxTasksPerChild = None
linker = None
query_manager = None

invalid_results = [
  u'(list)', [], u'(list (number 0))', ["0"], u'BADFORMAT', u'SERVER500']
def DecodeInputTree(wrtg, nbest, lambda_dcs_str_list):
  """
  lambda_dcs_str_list is an output parameter, where we store the valid
  output trees (string representations of lambda-DCS trees).
  An output parameter is used in order to retrieve partial lists in
  case of timeouts.
  """
  transductions = wrtg.GenerateNBestTreesMax(nbest)
  for best_tree, optimal_weight in transductions:
    if cvt_inserter:
      best_tree = cvt_inserter.insert_cvt_if_needed(best_tree)
    constituent_str = \
      best_tree if IsString(best_tree) else best_tree.pprint(margin=10000)
    query_results = QueryLambdaDCSC(constituent_str, query_manager)
    logging.info('\nConstituent: {0}\nWeight: {1}'\
      .format(constituent_str, optimal_weight))
    if query_results is None:
      continue
    if query_results not in invalid_results:
      lambda_dcs_str = ConvertConstituent2DCS(constituent_str)
      logging.info('Found. Weight: {0}\tTransduction: {1}'\
                    .format(optimal_weight, lambda_dcs_str))
      logging.info(u'Answer: {0}'.format(query_results))
      lambda_dcs_str_list.append(str(lambda_dcs_str))
  return

class Decoder(object):

  def __init__(self, option):
    self.__input        = option.input
    self.__fmtPrint     = option.fmtPrint
    self.__model        = option.model
    self.__feat_inst    = option.feat_inst
    self.__feat_names   = option.feat_names
    # Rule back-offs
    self.__lsdictent    = option.lsdictent
    self.__lsdictbent   = option.lsdictbent
    self.__lsdictpred   = option.lsdictpred
    self.__lsent        = option.lsent
    self.__lsbent       = option.lsbent
    self.__lspred       = option.lspred
    self.__lssempred    = option.lssempred
    # Language Model
    self.__lm           = option.lm
    self.__filter_prods = option.filter_prods
    self.__insert_cvts  = option.insert_cvts
    self.__augment_wrtgs = option.augment_wrtgs
    # Decoder options
    self.__nbest        = option.nbest
    self.__timeout      = option.timeout
    self.__cores        = option.cores
    # Debugger options
    self.__debug        = option.debug

  def run(self):
    global feat_inst
    global lm
    global rule_backoffs
    global timeout 
    global nbest
    global cvt_inserter
    global linker
    global query_manager

    query_manager = QueryManager()

    timeout = self.__timeout
    nbest = self.__nbest

    # Build transducer with rule back-offs.
    rules_filename = self.__input[1]
    if self.__lsdictent:
      from linguistics.similarity_dict_entities import DictEntities
      dict_filename, feature_weight = self.__lsdictent
      rule_backoffs.append(DictEntities(dict_filename, float(feature_weight)))
    if self.__lsdictbent:
      from linguistics.similarity_dict_entities import DictBridgeEntities
      dict_filename, feature_weight = self.__lsdictbent
      rule_backoffs.append(DictBridgeEntities(dict_filename, float(feature_weight)))
    if self.__lsdictpred:
      from linguistics.similarity_dict_entities import DictPredicates
      dict_filename, feature_weight = self.__lsdictpred
      dict_predicates = DictPredicates(dict_filename, float(feature_weight))
      rule_backoffs.append(dict_predicates)

    if self.__lssempred:
      from linguistics.similarity_dict_predicates import SemprePredicates
      dict_filename, feature_weight = self.__lssempred
      sempre_predicates = SemprePredicates(dict_filename, float(feature_weight))
      rule_backoffs.append(sempre_predicates)

    if self.__lsent or self.__lsbent or self.__lspred or self.__filter_prods:
      from qald.grounding import Linker
      linker = Linker()

    rules = loadrules(rules_filename, fmt=self.__fmtPrint)
    initial_state = GetInitialState(rules_filename, self.__fmtPrint)
    transducer = xT(initial_state, list(set(rules)), rule_backoffs)
    transducer.linker = linker

    # TODO: What about passing the feature instantiator as a parameter
    # to the model?
    # Build model and set its parameters.
    model, params_filename = self.__model
    if model == 'perceptron':
      from training.train_perceptron import PerceptronModel
      model = PerceptronModel()
      model_cls = PerceptronModel
    elif model == 'perceptron_avg':
      from training.train_perceptron_avg import AveragedPerceptronModel
      model = AveragedPerceptronModel()
      model_cls = AveragedPerceptronModel
    model.load(params_filename)

    # Load previously produced feature descriptions filename, so that
    # the same feature IDs are assigned to the same feature names.
    feat_inst = FeatureInstantiator(
      description_filename=self.__feat_inst,
      feat_names_filename=self.__feat_names)

    # Load a type-checking structured language model (if requested).
    if self.__lm:
      from lm.lm_qald import GetLMScoreOfDerivations, TypeCheckLM
      lm = TypeCheckLM(cache_filename='.lm_cache')
      lm_scoring_func = \
        lambda score, state: GetLMScoreOfDerivations(lm, score, state)
    else:
      lm_scoring_func = lambda score, state: score

    if self.__insert_cvts:
      from decoder.decode_qa_utils import CVTInserter
      cvt_inserter = CVTInserter(cache_filename='.cvt_cache')

    # Read input trees and make corpus of triplets (intree, outtree, weight).
    sentences_filename = self.__input[0]
    with codecs.open(sentences_filename, 'r', 'utf-8') as finput:
      intrees_str = [intree_str for intree_str in finput]
    corpus_src = [(s, None, 1.0) for s in intrees_str]
    # Obtain their wRTGs.
    if self.__augment_wrtgs:
      wrtgs, weighted_tree_pairs = ObtainWRTGsAugmented(
        corpus_src, transducer, feat_inst, model_cls, ncores=self.__cores)
    else:
      wrtgs, weighted_tree_pairs = ObtainWRTGs(
        corpus_src, transducer, feat_inst, model_cls, ncores=self.__cores)
    feat_inst.sync_id2desc()
    # Remove productions whose predicates do not link to any entity in the grammar.
    # from pudb import set_trace; set_trace()
    if self.__filter_prods:
      from lm.lm_qald_cohesion import ProductionFilter
      prod_filter = ProductionFilter(linker, self.__cores)
      wrtgs = prod_filter.filter_prods_from_wrtgs(wrtgs, corpus_src)

    # Weight rules of wRTGs according to the statistical model.
    for wrtg in wrtgs:
      if wrtg is not None:
        model.weight_wrtg(wrtg)

    if self.__debug:
      for wrtg in wrtgs:
        if wrtg is not None:
          wrtg.feat_inst = feat_inst 

    outputs = ProduceResultsFromTrees(wrtgs, intrees_str, ncores=self.__cores)
    print(dumps(outputs, indent=2))
    return

def ProduceResultsFromTrees(wrtgs, intrees_str, ncores):
  if ncores > 1:
    outputs = ProduceResultsFromTreesParallel(wrtgs, intrees_str, ncores)
  else:
    outputs = ProduceResultsFromTreesSequential(wrtgs, intrees_str)
  return outputs

def ProduceResultsFromTreesParallel(wrtgs, intrees_str, ncores):
  pool = Pool(processes=ncores, maxtasksperchild=kMaxTasksPerChild)
  outputs = pool.map_async(
    ProduceResultsFromTree, zip(wrtgs, intrees_str)).get(9999999)
  pool.close()
  pool.join()
  return outputs

def ProduceResultsFromTreesSequential(wrtgs, intrees_str):
  outputs = []
  num_intrees = len(wrtgs)
  for wrtg_intree_str in zip(wrtgs, intrees_str):
    output = ProduceResultsFromTree(wrtg_intree_str)
    outputs.append(output)
  return outputs

def ProduceResultsFromTree(wrtg_intree_str):
  wrtg, intree_str = wrtg_intree_str
  time_start = time.time()
  lambda_dcs_str_list = []
  if wrtg is None:
    output = {'parsed_query' : intree_str.strip(),
              'sys_queries' : lambda_dcs_str_list,
              'grammar' : [],
              'time' : time.time() - time_start,
              'status' : 'empty_grammar'}
    return output
  signal.alarm(timeout)
  try:
    DecodeInputTree(wrtg, nbest, lambda_dcs_str_list)
    status = 'completed'
  except TimeoutException:
    status = 'timeout'
  else:
    signal.alarm(0)
  rules = list(set([p.rhs.rule for p in wrtg.P]))
  rules_serialized = []
  rules_repr = sorted([r.PrintYaml() for r in rules], key=cmp_to_key(locale.strcoll))
  for rule_repr in rules_repr:
    rules_serialized.append(map(lambda s: s.strip(), rule_repr.split('\n')))
  output = {'parsed_query' : intree_str.strip(),
            'sys_queries' : lambda_dcs_str_list,
            'grammar' : rules_serialized,
            'time' : time.time() - time_start,
            'status' : status}
  return output

def main(args = None):
  import textwrap
  usage = "usage: %prog [options]"
  parser = argparse.ArgumentParser(usage)
  parser = argparse.ArgumentParser(
    prefix_chars='@',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=textwrap.dedent('''\
    Usage:
        decode_qa.py <parsed_sentences.tt> <rules.yaml> <params_file> <feat descriptions> <perceptron|rf>

        parsed_sentences.tt (tree-tree) must contain trees like:
        NP(DT(a) NN(house))
        S(NP(EX(There)) VP(VBZ(is) NP(DT(a) NN(ball))))
        ...
    '''))

  parser.add_argument("input", nargs=2, type=str, default=sys.stdin, metavar="INPUT",
                      help="Input trees in lisp format and transducer.")
  parser.add_argument("@m", "@@model", dest="model", nargs='*', type=str, default=["perceptron", "params.wwfttt"],
                      help="Name of the model for which parameter values are provided.")
  parser.add_argument("@@feat_inst", dest="feat_inst", nargs='?', type=str, default="",
                      help="Description of rule features.")
  parser.add_argument("@@feat_names", dest="feat_names", nargs='?', type=str, default="",
                      help="Names of feature templates to be instantiated.")
  parser.add_argument("@@fmtPrint", dest="fmtPrint", nargs='?', type=str, default="json",
                      help="Set the printing format to yaml, json or tiburon. Default is json.")
  # Options on back-off cost functions.
  parser.add_argument("@@LSDictbent",           dest="lsdictbent",              nargs=2,        default=[],
                      help="from linguistics.similarity_dict_entities import DictBridgeEntities. Default is False.")
  parser.add_argument("@@LSDictent",            dest="lsdictent",               nargs=2,        default=[],
                      help="from linguistics.similarity_dict_entities import DictEntities. Default is False.")
  parser.add_argument("@@LSDictpred",           dest="lsdictpred",              nargs=2,        default=[],
                      help="from linguistics.similarity_dict_entities import DictPredicates. Default is False.")
  parser.add_argument("@@LSEnt",                dest="lsent",           nargs=2, default=[],
                      help="from linguistics.similarity_qa import EntityLinkingCost. Default is False.")
  parser.add_argument("@@LSBent",               dest="lsbent",          nargs=2, default=[],
                      help="from linguistics.similarity_qa import BridgeLinkingCost. Default is False.")
  parser.add_argument("@@LSPred",               dest="lspred",          nargs=2, default=[],
                      help="from linguistics.similarity_qa import PredicateLinkingCost. Default is False.")
  parser.add_argument("@@LSSempred",            dest="lssempred",               nargs=2,        default=[],
                      help="from linguistics.similarity_dict_predicates import SemprePredicates. Default is False.")
  # Options on Language Model.
  parser.add_argument("@@lm", dest="lm", action="store_true",
                      help="Activate type-checking Language Model. Default is not activated.")
  parser.add_argument("@@filter_prods", dest="filter_prods", action="store_true", default=False,
                      help="Filter productions with non-cohesive predicates. Default is not activated.")
  parser.add_argument("@@insert_cvts", dest="insert_cvts", action="store_true", default=False,
                        help="Insert CVTs when necessary. Default is not activated.")
  parser.add_argument("@@augment_wrtgs", dest="augment_wrtgs", action="store_true", default=False,
                        help="Augment wRTGs with all predicates that connect to any entity. Default is not activated.")
  # Options controlling the decoder.
  parser.add_argument("@@nbest", dest="nbest", nargs='?', type=int, default="10000",
                      help="Number of output trees per input tree. Default is 10.000")
  parser.add_argument("@@timeout", dest="timeout", nargs='?', type=int, default="800",
                      help="Timeout per input tree, in seconds. Default 800s.")
  parser.add_argument("@@cores", dest="cores", nargs='?', type=int, default="1", \
                      help="CPUs to use.")
  parser.add_argument("@@debug", dest="debug", action="store_true", default=False,
                       help="Describe features and their weights for each rule in every derivaton. Default is not activated.")

  args = parser.parse_args()

  if args.debug:
    logging.basicConfig(level=logging.DEBUG)
  else:
    logging.basicConfig(level=logging.WARNING)

  decoder = Decoder(args)
  decoder.run()

if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt:
    sys.exit(1)
  except Exception as e:
    traceback.print_exc(file=sys.stderr)
    sys.exit(255)
  finally:
    for rule_backoff in rule_backoffs:
      rule_backoff.Close()
    if feat_inst:
      feat_inst.Close()
    if lm:
      lm.close()
    if cvt_inserter:
      cvt_inserter.close()
    if linker:
      linker.close()
    if query_manager:
      query_manager.close()
