#!/usr/bin/python
from __future__ import print_function

from collections import defaultdict
import logging
from multiprocessing import Pool, Lock
import sys, os, argparse, traceback
import simplejson
import time

from alignment.printing import (PrintDefaultAlignment, PrintGeneralInfo,
  PrintSentencePairInfo, PrintRule)
from extraction.extractor_beam import GetAlignmentFromDerivation
from extraction.feat_instantiator import FeatureInstantiator
from training.transducer import xT
from training.wrtg import SourceProjectionFromDerivationStrict
from utils.tools_generic_transducer import LoadCorpus
from utils.tree_tools import IsString, tree_or_string

lock = Lock()
RuleExtractor = None
options = None

class ExtractorParser:

  def __init__(self, option):
    self.__input		= option.input
    self.__cores		= option.cores
    self.__initial_state	= option.initial_state
    # Control search space and model expressivity.
    self.__kMaxSrcDepth		= option.kMaxSrcDepth
    self.__kMaxSrcBranches	= option.kMaxSrcBranches
    self.__kDefaultRunTime	= option.kDefaultRunTime
    self.__kBeamSize            = option.kBeamSize
    self.__phrase_length        = option.phrase_length
    self.__n_best		= option.n_best
    self.__dictDir		= option.dictDir
    self.__src_empty		= option.src_empty
    self.__deletions            = option.deletions
    # Control feature extraction for discriminative training.
    self.__feat_inst            = option.feat_inst
    # Individual features.
    self.__lscsse		= option.lscsse
    self.__lscls		= option.lscls
    self.__lscnd		= option.lscnd
    self.__lscind		= option.lscind
    self.__lsctc		= option.lsctc
    self.__lslex		= option.lslex
    self.__lsnsqa		= option.lsnsqa
    self.__lsapprox		= option.lsapprox
    self.__lsdict               = option.lsdict
    self.__lsdictp              = option.lsdictp
    ## Entity, predicate and bridging using dictionaries.
    self.__lsdictent		= option.lsdictent
    self.__lsdictbent		= option.lsdictbent
    self.__lsdictpred		= option.lsdictpred
    self.__lsfuzzypred		= option.lsfuzzypred
    self.__lssempred		= option.lssempred
    self.__lssempredfuzzy	= option.lssempredfuzzy
    ## Entity, predicate and bridging using Solr index.
    self.__lsent		= option.lsent
    self.__lsbent		= option.lsbent
    self.__lspred		= option.lspred
    self.__lsvari		= option.lsvari
    self.__lsenti		= option.lsenti
    self.__lswildrecog		= option.lswildrecog
    # General features.
    self.__lsnp		        = option.lsnp
    self.__lsvar		= option.lsvar
    self.__lsentdiff		= option.lsentdiff
    self.__lscomp		= option.lscomp
    self.__lssize		= option.lssize
    self.__lscount		= option.lscount
    self.__lspdc		= option.lspdc
    self.__lspnsp		= option.lspnsp
    self.__lsphdwc		= option.lsphdwc
    self.__lsphpc		= option.lsphpc
    self.__lsphpca		= option.lsphpca
    self.__lsrules		= option.lsrules
    self.__nostrings		= option.nostrings
    self.__lsalign		= option.lsalign
    # Guesser features.
    self.__lsphpcag		= option.lsphpcag
    self.__lsfree		= option.lsfree
    self.__lsurisurf		= option.lsurisurf
    self.__lsbridge		= option.lsbridge
    self.__lsphnsp		= option.lsphnsp
    self.__lsphlsp		= option.lsphlsp
    self.__lsphpcg		= option.lsphpcg
    self.__fmtPrint             = option.fmtPrint
    self.__similarity_scorer 	= None
    self.__similarity_score_guesser = None

  def run(self):

    # Corpus is a list of [src_tree, trg_tree]s.
    corpus = LoadCorpus(self.__input)
    # Adding the sentence number to act as identifier.
    corpus = [(i, src, trg) for i, (src, trg) in enumerate(corpus)]

    # Loading features:
    individual_features = []
    general_features = []
    guesser_features = []
 
    if self.__lscsse:
      from linguistics.similarity import SimilarityScorerEnsemble

    if self.__lscls:
      from linguistics.similarity_costs import LeafSimilarity
      guesser_features.append(LeafSimilarity(self.__lscls))
    if self.__lscnd:
      from linguistics.similarity_costs import NodesDifference
      general_features.append(NodesDifference(self.__lscnd))
    if self.__lscind:
      from linguistics.similarity_semantics import InnerNodesDifference
      general_features.append(InnerNodesDifference(self.__lscind))
    if self.__lsctc:
      from linguistics.similarity_costs import TreeComplexity
      general_features.append(TreeComplexity(self.__lsctc))
    if self.__nostrings:
      from linguistics.similarity_logics import StringRulesInfiniteCost
      feature_weight, side = self.__nostrings
      general_features.append(
        StringRulesInfiniteCost(float(feature_weight), side))
    if self.__lsalign:
      from linguistics.similarity_align import AlignmentCost
      align_fname, feature_weight = self.__lsalign
      general_features.append(AlignmentCost(align_fname, float(feature_weight)))

    if self.__lslex:
      from linguistics.similarity_costs import LexicalSimilarity
      individual_features.append(LexicalSimilarity(self.__lslex))
    if self.__lsvari:
      from linguistics.similarity_semantics import VariableDifferenceIndividual
      individual_features.append(VariableDifferenceIndividual(self.__lsvari))
    if self.__lsenti:
      from linguistics.similarity_semantics import EntityDifferenceIndividual
      individual_features.append(EntityDifferenceIndividual(self.__lsenti))
    if self.__lsvar:
      from linguistics.similarity_semantics import VariableDifference
      general_features.append(VariableDifference(self.__lsvar))
    if self.__lsentdiff:
      from linguistics.similarity_semantics import EntityDifference
      general_features.append(EntityDifference(self.__lsentdiff))
    if self.__lscomp:
      from linguistics.similarity_semantics import TreeDifferenceComplexity
      general_features.append(TreeDifferenceComplexity(self.__lscomp))
    if self.__lssize:
      from linguistics.similarity_semantics import TreeSize
      general_features.append(TreeSize(self.__lssize))
    if self.__lsnp:
      from linguistics.similarity_qa import NounPhraseCost
      feature_weight, cost_np, cost_no_np = map(float, self.__lsnp)
      general_features.append(NounPhraseCost(
        feature_weight=feature_weight, cost_np=cost_np, cost_no_np=cost_no_np))
    if self.__lsrules:
      from linguistics.similarity_rules import DictionaryRules
      dict_filename, feature_weight = self.__lsrules
      individual_features.append(DictionaryRules(dict_filename, float(feature_weight)))
    if self.__lspdc:
      from linguistics.similarity_pre import DictionaryCost
      individual_features.append(DictionaryCost(self.__dictDir, self.__lspdc))
    if self.__lspnsp:
      from linguistics.similarity_pre import NoSimilarityPre
      individual_features.append(NoSimilarityPre(self.__lspnsp))
    if self.__lsnsqa:
      from linguistics.similarity_qa import NoSimilarityQA
      feat_weight, dels_cost, ins_cost, subs_cost = map(float, self.__lsnsqa)
      individual_features.append(
        NoSimilarityQA(feat_weight, dels_cost, ins_cost, subs_cost))
    if self.__lsdict:
      from linguistics.similarity_dictionary import DictionaryCost
      dict_filename, feature_weight = self.__lsdict
      individual_features.append(DictionaryCost(dict_filename, float(feature_weight)))

    if self.__lsdictp:
      from linguistics.similarity_dictionary_part import DictionaryCostPart
      dict_filename, feature_weight = self.__lsdictp
      individual_features.append(DictionaryCostPart(dict_filename, float(feature_weight)))
    if self.__lsdictent:
      from linguistics.similarity_dict_entities import DictEntities
      dict_filename, feature_weight = self.__lsdictent
      individual_features.append(DictEntities(dict_filename, float(feature_weight)))
    if self.__lsdictbent:
      from linguistics.similarity_dict_entities import DictBridgeEntities
      dict_filename, feature_weight = self.__lsdictbent
      individual_features.append(DictBridgeEntities(dict_filename, float(feature_weight)))
    if self.__lsdictpred:
      from linguistics.similarity_dict_entities import DictPredicates
      dict_filename, feature_weight = self.__lsdictpred
      dict_predicates = DictPredicates(dict_filename, float(feature_weight))
      individual_features.append(dict_predicates)
    if self.__lsfuzzypred:
      from linguistics.similarity_dict_entities import DictFuzzyPredicates
      dict_filename, feature_weight = self.__lsfuzzypred
      fuzzy_predicates = DictFuzzyPredicates(dict_filename, float(feature_weight))
      individual_features.append(fuzzy_predicates)
    if self.__lssempred:
      from linguistics.similarity_dict_predicates import SemprePredicates
      dict_filename, feature_weight = self.__lssempred
      sempre_predicates = SemprePredicates(dict_filename, float(feature_weight))
      individual_features.append(sempre_predicates)
    if self.__lssempredfuzzy:
      from linguistics.similarity_dict_predicates import SemprePredicatesFuzzy
      dict_filename, feature_weight = self.__lssempredfuzzy
      sempre_predicates_fuzzy = SemprePredicatesFuzzy(dict_filename, float(feature_weight))
      individual_features.append(sempre_predicates_fuzzy)

    # Entity/Predicate linking using an inverted index (Solr).
    if self.__lsent or self.__lsbent or self.__lspred or \
       self.__lscount or self.__lswildrecog:
      from qald.grounding import Linker
      linker = Linker()
    else:
      linker = None
    if self.__lsent:
      from linguistics.similarity_qa import EntityLinkingCost
      feature_weight, kbest = float(self.__lsent[0]), int(self.__lsent[1])
      individual_features.append(
        EntityLinkingCost(feature_weight, kbest, linker=linker))
    if self.__lsbent:
      from linguistics.similarity_qa import BridgeLinkingCost
      feature_weight, kbest = float(self.__lsbent[0]), int(self.__lsbent[1])
      individual_features.append(
        BridgeLinkingCost(feature_weight, kbest, linker))
    if self.__lspred:
      from linguistics.similarity_qa import PredicateLinkingCost
      feature_weight, kbest = float(self.__lspred[0]), int(self.__lspred[1])
      individual_features.append(
        PredicateLinkingCost(feature_weight, kbest, linker))
    # These cost functions also need access to the linker.
    if self.__lscount:
      from linguistics.similarity_qa import CountOp
      general_features.append(CountOp(self.__lscount, linker))
    if self.__lswildrecog:
      from linguistics.similarity_qa import UriSurfCost
      feature_weight = float(self.__lswildrecog)
      individual_features.append(UriSurfCost(feature_weight, linker))

    if self.__lsphdwc:
      from linguistics.similarity_phrases import DistributedWordCost, DistributedSimilarity
      distributed_similarity = DistributedSimilarity(self.__dictDir, self.__phrase_length)
      individual_features.append(DistributedWordCost(distributed_similarity, self.__lsphdwc))
    if self.__lsphpc:
      from linguistics.similarity_phrases import PhraseCost, DistributedSimilarity
      distributed_similarity = DistributedSimilarity(self.__dictDir, self.__phrase_length)
      individual_features.append(PhraseCost(distributed_similarity, self.__lsphpc))
    if self.__lsphpca:
      from linguistics.similarity_phrases_ag import DistributedSimilarity
      from linguistics.similarity_phrases_ag import PhraseCost
      feature_weight, src_lang, trg_lang = self.__lsphpca
      distributed_similarity = \
        DistributedSimilarity(self.__dictDir, self.__phrase_length,
                              str(src_lang), str(trg_lang))
      individual_features.append(
        PhraseCost(distributed_similarity, float(feature_weight)))
    if self.__lsapprox:
      from linguistics.similarity_approx import ApproximateMatch
      individual_features.append(ApproximateMatch(self.__lsapprox))
    if self.__lsfree:
      from linguistics.similarity_freebase import EntityLinkingCost
      individual_features.append(EntityLinkingCost(self.__lsfree))
    if self.__lsurisurf:
      from linguistics.similarity_urisurf import URISurfCost
      individual_features.append(URISurfCost(self.__lsurisurf))
    if self.__lsbridge:
      from linguistics.similarity_bridge import BridgeCost
      individual_features.append(BridgeCost(self.__lsbridge))
    if self.__lsphnsp:
      from linguistics.similarity_phrases import NoSimilarityPhrases
      individual_features.append(NoSimilarityPhrases(self.__lsphnsp))
    if self.__lsphlsp:
      from linguistics.similarity_phrases import LeafSimilarityPhrases
      guesser_features.append(LeafSimilarityPhrases(self.__lsphlsp))
    if self.__lsphpcag:
      from linguistics.similarity_phrases_ag import PhraseCostGuesser
      guesser_features.append(PhraseCostGuesser(distributed_similarity, self.__lsphpcag))
    if self.__lsphpcg:
      from linguistics.similarity_phrases import PhraseCostGuesser, DistributedSimilarity
      # We are currently using the same distributd_similarity object
      # as for PhraseCost similarity function. This is to avoid pre-caching
      # the same phrases for the similarity function and the similarity guesser.
      # distributed_similarity = DistributedSimilarity(self.__dictDir)
      guesser_features.append(PhraseCostGuesser(distributed_similarity, self.__lsphpcg))

    # Using either exact search (if no beam size is specified), or approximated search.
    global RuleExtractor
    if self.__kBeamSize > 0:
      from extraction.extractor_beam import RuleExtractor as RuleExtractorApprox
      RuleExtractor = RuleExtractorApprox
    else:
      from extraction.extractor_exact import RuleExtractor as RuleExtractorExact
      RuleExtractor = RuleExtractorExact

    if self.__feat_inst:
      self.__feat_instantiator = FeatureInstantiator(self.__feat_inst)
    else:
      self.__feat_instantiator = None
 
    command_used = 'python -m alignment.align ' + ' '.join(sys.argv[1:])
    general_info = {'general_info' : {'kMaxSrcDepth' : self.__kMaxSrcDepth,
                                     'kMaxSrcBranches' : self.__kMaxSrcBranches,
                                     'kDefaultRunTime' : self.__kDefaultRunTime,
                                     'kBeamSize' : self.__kBeamSize,
                                     'phrase_length' : self.__phrase_length,
                                     'initial_state' : self.__initial_state,
                                     'command' : command_used}
                   }
    general_info_str = PrintGeneralInfo(general_info, self.__fmtPrint)
    print(general_info_str, end='\n\n')

    if self.__lscsse:
      self.__similarity_scorer = \
        SimilarityScorerEnsemble(individual_features, general_features)
      self.__similarity_score_guesser = \
        SimilarityScorerEnsemble(guesser_features)
    else:
      self.__similarity_scorer = individual_features[0]
      self.__similarity_score_guesser = guesser_features[0]

    global options
    options = {'similarity_scorer' : self.__similarity_scorer,
               'similarity_score_guesser' : self.__similarity_score_guesser,
               'max_source_branches' : self.__kMaxSrcBranches,
               'max_source_depth' : self.__kMaxSrcDepth,
               'max_running_time' : self.__kDefaultRunTime,
               'beam_size' : self.__kBeamSize,
               'cached_extractors' : {},
               'initial_state' : self.__initial_state,
               'src_empty' : self.__src_empty,
               'deletions' : self.__deletions,
               'feat_inst' : self.__feat_instantiator,
               'nbest' : self.__n_best,
               'fmt' : self.__fmtPrint}

    try:
      if self.__cores == 1:
        self.__ObtainTransductionsFromCorpus(corpus)
      else:
        self.__ObtainTransductionsFromCorpusParallel(corpus)
    except KeyboardInterrupt:
      sys.exit(1)
    except Exception as e:
      traceback.print_exc(file=sys.stderr)
      sys.exit(255)
    finally:
      self.__similarity_scorer.Close()
      self.__similarity_score_guesser.Close()
      if self.__feat_instantiator:
        self.__feat_instantiator.Close()
      if linker:
        linker.close()

  def __ObtainTransductionsFromCorpusParallel(self, corpus):
    pool = Pool(processes=self.__cores, maxtasksperchild=1)
    pool.map_async(ExtractRulesFromPair, corpus).get(9999999)
    pool.close()
    pool.join()
    return
  
  def __ObtainTransductionsFromCorpus(self, corpus):
    for numbered_tree_pair in corpus:
      ExtractRulesFromPair(numbered_tree_pair)
    return

def ExtractRulesFromPair(numbered_tree_pair):
  global lock

  pair_num, tree_string1, tree_string2 = numbered_tree_pair
  tree1 = tree_or_string(tree_string1)
  tree2 = tree_or_string(tree_string2)
  path1, path2 = (), ()
  rule_extractor = RuleExtractor(tree1, tree2, path1, path2, options)

  derivations = \
    rule_extractor.ObtainBestDerivations(options['nbest'], state_id='')
  time_spent = time.time() - rule_extractor.time_start
  if not derivations:
    derivation_cost = 1000
    alignment_str = PrintDefaultAlignment(tree1, tree2)
    tree_pair_info = {'pair_info' : \
                        {'pair_num' : pair_num,
                         'n_best' : -1,
                         'source' : tree_string1,
                         'target' : tree_string2,
                         'alignment' : alignment_str,
                         'cost' : derivation_cost,
                         'status' : rule_extractor.status,
                         'time' : time_spent}
                     }
  else:
    for i, derivation in enumerate(derivations):
      derivation_cost = sum([rule.weight for rule in derivation])
      alignment_str = GetAlignmentFromDerivation(derivation, tree1, tree2)
      tree_pair_info = {'pair_info' : \
                          {'pair_num' : pair_num,
                           'n_best' : i,
                           'source' : tree_string1,
                           'target' : tree_string2,
                           'alignment' : alignment_str,
                           'cost' : derivation_cost,
                           'status' : rule_extractor.status,
                           'time' : time_spent}
                       }
      tree_pair_info_str = \
        PrintSentencePairInfo(tree_pair_info, fmt=options['fmt'])
      lock.acquire()
      print(tree_pair_info_str)
      for rule in derivation:
        print(PrintRule(rule, options['fmt']))
      sys.stdout.flush()
      lock.release()
  return

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

  parser.add_argument('input',			nargs='?',		type=str,			default=sys.stdin,	metavar="INPUT", \
			help="Input corpus.tt contains parallel source and target constituent trees.")
  parser.add_argument("@@cores",		dest="cores",		nargs='?',	type=int,	default="1", \
		      help="CPUs to use")
  parser.add_argument("@@initial_state",	dest="initial_state",	nargs='?',	type=str,	default="start", \
		      help="CPUs to use")
  parser.add_argument("@@kMaxSourceDepth",	dest="kMaxSrcDepth",	nargs='?',	type=int,	default="2", \
		      help="")
  parser.add_argument("@@kMaxSourceBranches",	dest="kMaxSrcBranches",	nargs='?',	type=int,	default="3", \
		      help="")
  parser.add_argument("@@kDefaultRunTime",	dest="kDefaultRunTime",	nargs='?',	type=int,	default="100", \
		      help="Set the running time in second.")
  parser.add_argument("@@kBeamSize",  	dest="kBeamSize",	nargs='?',	type=int,	default="0", \
		      help="Set the beam size of the approximated search. If not set, exact search is performed.")
  parser.add_argument("@@src_empty",  	dest="src_empty",	nargs='?',	type=bool,	default=True, \
		      help="Forbids empty (epsilon) transitions on source tree (left-hand-sides of rules). Default is True.")
  parser.add_argument("@@feat_inst",  	dest="feat_inst",	nargs='?',	default='', \
		      help="Enables feature instantiation for discriminative training. Default is False.")
  parser.add_argument("@@deletions",  	dest="deletions",	nargs='?',	type=bool,	default=True, \
		      help="Enables deletion operations. Default is True.")
  parser.add_argument("@@phrase_length",	dest="phrase_length",	nargs='?',	type=int,	default="1", \
		      help="Maximum phrase length for similarity function.")
  parser.add_argument("@@n_best",	dest="n_best",	nargs='?',	type=int,	default="1", \
		      help="Maximum number of derivations to be extracted.")
  parser.add_argument("@@fmtPrint",		dest="fmtPrint",	nargs='?',	type=str,	default="json", \
		      help="Set the printing format, yaml or json. Default is json.")
  parser.add_argument("@@dictDir",		dest="dictDir",		nargs='?',	type=str,	default="/home/yulin/PHD/source/pascual/transducers/lex.j2e.cost.small", \
		      help="The directory of the lexicon cost dictionary")
  parser.add_argument("@@LSCSSE",		dest="lscsse",		action="store_true", \
		      help="from linguistics.similarity import SimilarityScorerEnsemble. Default is False.")
  parser.add_argument("@@LSCLS",		dest="lscls",		action="store_true", \
		      help="from linguistics.similarity_costs import LeafSimilarity. Default is False.")
  parser.add_argument("@@LSCND",		dest="lscnd",		nargs='?',	type=float, 	default=0.0,
		      help="from linguistics.similarity_costs import NodesDifference. Default is False.")
  parser.add_argument("@@LSCIND",		dest="lscind",		nargs='?',	type=float, 	default=0.0,
		      help="from linguistics.similarity_semantics import InnerNodesDifference. Default is False.")
  parser.add_argument("@@LSCTC",		dest="lsctc",		nargs='?',	type=float, 	default=0.0,
		      help="from linguistics.similarity_costs import TreeComplexity. Default is False.")
  parser.add_argument("@@LSPDC",		dest="lspdc",		nargs='?',	type=float, 	default=0.0,
		      help="from linguistics.similarity_pre import DictionaryCost. Default is False.")
  parser.add_argument("@@LSDict",		dest="lsdict",		nargs=2,	default=[],
		      help="from linguistics.similarity_dictionary import DictionaryCost. Default is False.")
  parser.add_argument("@@LSDictp",		dest="lsdictp",		nargs=2,	default=[],
		      help="from linguistics.similarity_dictionary_part import DictionaryCostPart. Default is False.")
  parser.add_argument("@@LSDictbent",		dest="lsdictbent",		nargs=2,	default=[],
		      help="from linguistics.similarity_dict_entities import DictBridgeEntities. Default is False.")
  parser.add_argument("@@LSDictent",		dest="lsdictent",		nargs=2,	default=[],
		      help="from linguistics.similarity_dict_entities import DictEntities. Default is False.")
  parser.add_argument("@@LSDictpred",		dest="lsdictpred",		nargs=2,	default=[],
		      help="from linguistics.similarity_dict_entities import DictPredicates. Default is False.")
  parser.add_argument("@@LSFuzzypred",		dest="lsfuzzypred",		nargs=2,	default=[],
		      help="from linguistics.similarity_dict_entities import DictFuzzyPredicates. Default is False.")
  parser.add_argument("@@LSSempred",		dest="lssempred",		nargs=2,	default=[],
		      help="from linguistics.similarity_dict_predicates import SemprePredicates. Default is False.")
  parser.add_argument("@@LSSempredfuzzy",	dest="lssempredfuzzy",		nargs=2,	default=[],
		      help="from linguistics.similarity_dict_predicates import SemprePredicatesFuzzy. Default is False.")
  parser.add_argument("@@LSEnt",		dest="lsent",		nargs=2, default=[],
		      help="from linguistics.similarity_qa import EntityLinkingCost. Default is False.")
  parser.add_argument("@@LSBent",		dest="lsbent",		nargs=2, default=[],
		      help="from linguistics.similarity_qa import BridgeLinkingCost. Default is False.")
  parser.add_argument("@@LSPred",		dest="lspred",		nargs=2, default=[],
		      help="from linguistics.similarity_qa import PredicateLinkingCost. Default is False.")
  parser.add_argument("@@LSWildrecog",		dest="lswildrecog",	nargs='?', default=0.0,
		      help="from linguistics.similarity_qa import UriSurfCost. Default is 0.0.")
  parser.add_argument("@@LSRules",		dest="lsrules",		nargs=2,	default=[],
		      help="from linguistics.similarity_rules import DictionaryRules. Default is False.")
  parser.add_argument("@@LSLex",		dest="lslex",		nargs='?',	type=float, 	default=0.0,
		      help="from linguistics.similarity_costs import LexicalSimilarity. Default is False.")
  parser.add_argument("@@LSVari",		dest="lsvari",		nargs='?',	type=float, 	default=0.0,
		      help="from linguistics.similarity_semantics import VariableDifferenceIndividual. Default is False.")
  parser.add_argument("@@LSEnti",		dest="lsenti",		nargs='?',	type=float, 	default=0.0,
		      help="from linguistics.similarity_semantics import EntityDifferenceIndividual. Default is False.")
  parser.add_argument("@@LSVar",		dest="lsvar",		nargs='?',	type=float, 	default=0.0,
		      help="from linguistics.similarity_semantics import VariableDifference. Default is False.")
  parser.add_argument("@@LSEntDiff",		dest="lsentdiff",		nargs='?',	type=float, 	default=0.0,
		      help="from linguistics.similarity_semantics import EntityDifference. Default is False.")
  parser.add_argument("@@LSComp",		dest="lscomp",		nargs='?',	type=float, 	default=0.0,
		      help="from linguistics.similarity_semantics import TreeDifferenceComplexity. Default is False.")
  parser.add_argument("@@LSSize",		dest="lssize",		nargs='?',	type=float, 	default=0.0,
		      help="from linguistics.similarity_semantics import TreeSize. Default is False.")
  parser.add_argument("@@LSCount",		dest="lscount",		nargs='?',	type=float, 	default=0.0,
		      help="from linguistics.similarity_qa import CountOp. Default is False.")
  parser.add_argument("@@NoStrings",		dest="nostrings",	nargs=2,	default=[],
		      help="from linguistics.similarity_logics import StringRulesInfiniteCost. Default is False.")
  parser.add_argument("@@LSAlign",		dest="lsalign",		nargs=2,	default=[],
		      help="from linguistics.similarity_align import AlignmentCost. Default is False.")
  parser.add_argument("@@LSNP",		dest="lsnp",		nargs='*',	default=0.0,
		      help="from linguistics.similarity_qa import NounPhraseCost. Default is False.")
  parser.add_argument("@@LSPNSP",		dest="lspnsp",		nargs='?',	type=float, default=[],
		      help="from linguistics.similarity_pre import NoSimilarityPre. Default is False.")
  parser.add_argument("@@LSNSQA",		dest="lsnsqa",		nargs='*',	default=[],
		      help="from linguistics.similarity_qa import NoSimilarityQA. Default is False.")
  parser.add_argument("@@LSPHDWC",		dest="lsphdwc",		nargs='?',	type=float, 	default=0.0,
		      help="from linguistics.similarity_phrases import DistributedWordCost. Default is False.")
  parser.add_argument("@@LSPHPC",		dest="lsphpc",		nargs='?',	type=float, 	default=0.0,
		      help="from linguistics.similarity_phrases import PhraseCost. Default is False.")
  parser.add_argument("@@LSPHPCA",		dest="lsphpca",		nargs=3,	default=[],
		      help="from linguistics.similarity_phrases_ag import PhraseCost. Default is False.")
  parser.add_argument("@@LSApprox",		dest="lsapprox",	nargs='?',	type=float, 	default=0.0,
		      help="from linguistics.similarity_approx import ApproximateMatch. Default is False.")
  parser.add_argument("@@LSFree",		dest="lsfree",		nargs='?',	type=float, 	default=0.0,
		      help="from linguistics.similarity_freebase import EntityLinkingCost. Default is False.")
  parser.add_argument("@@LSURISurf",		dest="lsurisurf",		nargs='?',	type=float, 	default=0.0,
		      help="from linguistics.similarity_urisurf import URISurfCost. Default is False.")
  parser.add_argument("@@LSBridge",		dest="lsbridge",		nargs='?',	type=float, 	default=0.0,
		      help="from linguistics.similarity_bridge import BridgeCost. Default is False.")
  parser.add_argument("@@LSPHNSP",		dest="lsphnsp",		nargs='?',	type=float, 	default=0.0,
		      help="from linguistics.similarity_phrases import NoSimilarityPhrases. Default is False.")
  parser.add_argument("@@LSPHLSP",		dest="lsphlsp",		nargs='?',	type=float, 	default=0.0,
		      help="from linguistics.similarity_phrases import LeafSimilarityPhrases. Default is False.")
  parser.add_argument("@@LSPHPCAG",		dest="lsphpcag",	nargs='?',	type=float, 	default=0.0,
		      help="from linguistics.similarity_phrases_ag import PhraseCostGuesser. Default is False.")
  parser.add_argument("@@LSPHPCG",		dest="lsphpcg",		nargs='?',	type=float, 	default=0.0,
                      help="from linguistics.similarity_phrases import PhraseCostGuesser. Default is False.")

  args = parser.parse_args()

  logging.basicConfig(level=logging.WARNING)

  ruleExtractor = ExtractorParser(args)
  ruleExtractor.run()


if __name__ == '__main__':
  try:
    main()
  except Exception as e:
    traceback.print_exc(file=sys.stderr)
    sys.exit(255)
