import codecs
import logging
import sys, os, argparse, traceback

from extraction.feat_instantiator import FeatureInstantiator
from qald.dcs_tools import ConvertConstituent2DCS
from training.loadrules import loadrules, GetInitialState
from utils.corpus import LoadCorpus
from utils.tree_tools import IsString
from training.transducer import xT

class trainModelParser:

  def __init__(self, option):
    self.__input                 = option.input
    self.__output                = option.output
    self.__kMaxIterations 	 = option.kMaxIterations 
    self.__kConvergenceTolerance = option.kConvToc
    self.__fmtPrint              = option.fmtPrint
    self.__cores                 = option.cores
    self.__numOccur              = option.numOccur
    self.__learning_rate         = option.learning_rate
    self.__model                 = option.model
    self.__feat_inst             = option.feat_inst
    self.__feat_names_filename   = option.feat_names
    self.__task                  = option.task
    # Language Model
    self.__filter_prods          = option.filter_prods
    self.__insert_cvts           = option.insert_cvts
    self.__augment_wrtgs         = option.augment_wrtgs
    # Rule back-offs
    self.__lsdictent             = option.lsdictent
    self.__lsdictbent            = option.lsdictbent
    self.__lsdictpred            = option.lsdictpred
    self.__lsent                 = option.lsent
    self.__lsbent                = option.lsbent
    self.__lspred                = option.lspred
    self.__lssempred             = option.lssempred

  def run(self):
    # Build feature instantiator.
    descriptions_filename = self.__feat_inst
    feat_inst = FeatureInstantiator(
      descriptions_filename, feat_names_filename=self.__feat_names_filename)

    # Corpus is a list of [tree, string]s.
    corpus_filename = self.__input[0]
    corpus = LoadCorpus(corpus_filename)
    # The algorithm for training tree transducers expects a triplet:
    #   source tree, target_tree, weight (of the pair).
    corpus = [(src_tree, trg_tree, 0.5) for (src_tree, trg_tree) in corpus]

    # Build transducer with back-off cost functions.
    rule_backoffs = []
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
    else:
      linker = None

    rules_filename = self.__input[1]
    rules = loadrules(
      rules_filename, fmt=self.__fmtPrint, num_occur=self.__numOccur)
    rules = list(set(rules))
    initial_state = GetInitialState(rules_filename, self.__fmtPrint)
    transducer = xT(initial_state, rules, rule_backoffs)
    transducer.linker = linker

    cvt_inserter = None
    if self.__insert_cvts:
      from decoder.decode_qa_utils import CVTInserter
      cvt_inserter = CVTInserter(cache_filename='.cvt_cache')

    if self.__filter_prods:
      from lm.lm_qald_cohesion import ProductionFilter
      prod_filter = ProductionFilter(linker, self.__cores)
    else:
      prod_filter = None

    if self.__model == 'perceptron_avg':
      from training.train_perceptron_avg import AveragedPerceptronModel
      model = AveragedPerceptronModel(prod_filter, cvt_inserter=cvt_inserter)
    elif self.__model == 'perceptron':
      from training.train_perceptron import PerceptronModel
      model = PerceptronModel(prod_filter, cvt_inserter=cvt_inserter)
    model.max_iterations = self.__kMaxIterations
    model.learning_rate = self.__learning_rate

    query_manager = None
    if self.__task == 'qa':
      from qald.sparql_utils import QueryManager
      query_manager = QueryManager()
      from decoder.decode_qa_utils import QueryLambdaDCSC
      def AreAnswersEqual(src_tree, trg_tree):
        src_results = QueryLambdaDCSC(src_tree)
        trg_results = QueryLambdaDCSC(trg_tree)
        return src_results == trg_results and src_results is not None
      model.trg_equals_gold = AreAnswersEqual
      from decoder.decode_qa_utils import GetBestValidDerivations
      model.GetBestValidDerivations = GetBestValidDerivations
    else:
      def AreTreesEqual(src_tree, trg_tree):
        assert IsString(src_tree) and IsString(trg_tree)
        return src_tree == trg_tree
      model.trg_equals_gold = AreTreesEqual
    model.query_manager = query_manager
    model.augment_wrtgs = self.__augment_wrtgs

    try:
      model.train(transducer, corpus, feat_inst=feat_inst, ncores=self.__cores)
    except KeyboardInterrupt:
      sys.exit(1)
    finally:
      model.save(self.__output)
      if feat_inst:
        feat_inst.Close()
      for rule_backoff in rule_backoffs:
        rule_backoff.Close()
      if cvt_inserter:
        cvt_inserter.close()
      if linker:
        linker.close()
      if query_manager:
        query_manager.close()

def main(args = None):
  import textwrap
  usage = "usage: %prog [options]"
  parser = argparse.ArgumentParser(usage)
  parser = argparse.ArgumentParser(
    prefix_chars='@',
    formatter_class=argparse.RawDescriptionHelpFormatter, 
    description=textwrap.dedent('''\
    corpus.tt (tree-tree) must contain pairs of trees like:
    ------------------------------------------------------
          NP(DT(a) NN(house))
          NP(DT(the) NN(house))
          NP(DT(a) NN(ball))
          S(NP(EX(There)) VP(VBZ(is) NP(DT(a) NN(ball))))
          ...
    '''))

  parser.add_argument('input',				nargs=2,		type=str,	default=sys.stdin,			metavar="INPUT", \
                        help="Input corpus name (i.e., corpus.tt), transduction file name (i.e., transductions.yaml).")
  parser.add_argument("@o", "@@output",			dest="output",		nargs='?',	type=str,	default="weighted_transductions.wfttt",	metavar="weighted_transductions.wfttt", \
                        help="Output weighted transduction rules (i.e., *.wfttt).")
  parser.add_argument("@feat_inst", "@@feat_inst",			dest="feat_inst",		nargs='?',	type=str,	default="", help="Description of rule features.")
  parser.add_argument("@feat_names", "@@feat_names",			dest="feat_names",		nargs='?',	type=str,	default="", help="Names of feature templates to be instantiated.")
  parser.add_argument("@@cores",			dest="cores",		nargs='?',	type=int,	default="1", \
                      help="CPUs to use.")
  parser.add_argument("@@numOccur",			dest="numOccur",	nargs='?',	type=int,	default="0", \
                      help="Number of occurrences.")
  parser.add_argument("@@kMaxIterations", 		dest="kMaxIterations", 	nargs='?',      type=int,       default="5", \
                      help="")
  parser.add_argument("@@learning_rate",	        dest="learning_rate", 	nargs='?',      type=float,	default="0.1", \
                      help="")
  parser.add_argument("@@kConvergenceTolerance",	dest="kConvToc", 	nargs='?',      type=float,	default="0.000001", \
                      help="")
  parser.add_argument("@@model",	                dest="model", 	        nargs='?',      type=str,	default="perceptron", \
                      help="")
  parser.add_argument("@@fmtPrint",			dest="fmtPrint",	nargs='?',      type=str,       default="json", \
                      help="Set the printing format, yaml or json. Default is json.")
  parser.add_argument("@@task",			dest="task",	nargs='?',      type=str,       default="qa", \
                      help="Set the task, which tells how to compare hypothesis and gold target structures.")
  parser.add_argument("@@filter_prods", dest="filter_prods", action="store_true", default=False,
                        help="Filter productions with non-cohesive predicates. Default is not activated.")
  parser.add_argument("@@insert_cvts", dest="insert_cvts", action="store_true", default=False,
                        help="Insert CVTs when necessary. Default is not activated.")
  parser.add_argument("@@augment_wrtgs", dest="augment_wrtgs", action="store_true", default=False,
                        help="Augment wRTGs with all predicates that connect to any entity. Default is not activated.")
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

  args = parser.parse_args()

  logging.basicConfig(level=logging.WARNING)

  training = trainModelParser(args)
  training.run()


if __name__ == '__main__':
  try:
    main()
  except Exception as e:
    traceback.print_exc(file=sys.stderr)
    sys.exit(255)
