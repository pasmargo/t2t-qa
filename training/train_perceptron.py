from __future__ import print_function
import codecs
from collections import defaultdict
import logging
import sys

from utils.production import TargetProjectionFromDerivation
from training.feat_model import FeatModel, CollectFeats
from training.wrtgs import ObtainWRTGs, ObtainWRTGsAugmented
from utils.tree_tools import IsString

class PerceptronModel(FeatModel):
  """ Implements a structured perceptron model.

  It contains a vector (default dictionary) of feature weights.
  Weights are estimated using the training procedure of the structured perceptron.
  Rule weights are computed by the dot product of the vector
  of feature weights and the feature values of the rule.
  """

  def __init__(self, prod_filter=None, cvt_inserter=None):
    self.feat_weights = defaultdict(lambda: 0.0)
    self.max_iterations = 5
    self.learning_rate = .1
    # Function that compares the decoded target latent structure
    # to the gold target structure. This comparison depends on the application.
    # E.g. typically, the function will return True if the trees are equal.
    # In QA, it returns true if the answers they produce are equal.
    self.trg_equals_gold = lambda x, y: x == y
    # Function to retrieve the best valid derivations.
    # This is application-dependent. By default,
    # we use all possible derivations (do not filter out any derivation).
    self.GetBestValidDerivations = GetBestValidDerivations
    # If self.filter_prods is True, then productions with non-cohesive
    # predicates (with the entities in the grammar) are filtered out.
    # This filtering is not very precise at this point, and legitimated
    # predicates might be filtered-out. Use with care.
    self.prod_filter = prod_filter
    self.cvt_inserter = cvt_inserter
    self.augment_wrtgs = False
    self.query_manager = None
    # Variables to collect basic statistics at every iteration.
    self.correct_predictions_per_iter = 0

  def save(self, filename):
    with codecs.open(filename, 'w', 'utf-8') as foutput:
      for feat_id, feat_weight in sorted(self.feat_weights.items()):
        foutput.write('{0}\t{1}\n'.format(feat_id, feat_weight))

  def load(self, filename):
    feat_weights = defaultdict(float)
    with codecs.open(filename, 'r', 'utf-8') as finput:
      for line in finput:
        feat_id, feat_weight = line.strip().split()
        feat_weights[int(feat_id)] = float(feat_weight)
    self.feat_weights = feat_weights

  def weight_rule(self, rule):
    assert isinstance(rule.features, list)
    weight = 0.0
    for i in range(len(rule.features)):
      feature = rule.features[i]
      feat_id, feat_val = feature[0:2]
      rule.features[i] = [feat_id, feat_val, self.feat_weights.get(feat_id, None)]
      weight += self.feat_weights[feat_id] * feat_val
    return weight

  def train(self, transducer, corpus, feat_inst=None, ncores=1):
    """
    Implements Structured Perceptron. To obtain the correct tree, it obtains
    a wRTG that is constrained on the source and the target tree (for every pair).
    To obtain the best tree of the model, it obtains a wRTG that is only
    constrained on the source tree. Then, if the best tree of the latter grammar
    is different from the gold target tree, we adjust the feature weights
    (learning step). Otherwise, skip sample. At the moment, features are extracted
    locally from each rule that forms the derivation. In the future, it should
    also contain global features.
    This function does NOT modify the weights of the rules of the transducer.
    Instead, it estimates the feature weights and returns them.

    Args:
      transducer xT
      corpus: a list of 3-tuples [(src_tree, trg_tree, pair_weight), ...]
      feat_inst: object Feature Instantiator to extract features from new rules.
    """
    wrtgs_src_trg, wrtgs_src = self.produce_grammars(
      corpus, transducer, feat_inst, ncores)
    for i in range(self.max_iterations):
      for train_ind in range(len(wrtgs_src_trg)):
        # print('Iteration {0}, instance {1}'.format(i, train_ind))
        self.decode_and_maybe_adjust(train_ind, wrtgs_src_trg, wrtgs_src, corpus)
      print(' Accuracy = {0}'.format(
        self.correct_predictions_per_iter / float(len(wrtgs_src_trg))),
        end='', file=sys.stderr)
      print('', file=sys.stderr)
      self.correct_predictions_per_iter = 0

  def produce_grammars(self, corpus, transducer, feat_inst, ncores):
    """
    Since the structure of the grammars do not change through the iterative
    parameter estimation, they are computed once at the beginning of the
    procedure. Two lists of grammars are produced:
    wrtgs_src_trg are a list of source- and target-constrained wRTGs,
      and they produce the same target tree using different derivations.
    wrtgs_src are a list of source-constrained wRTGs, and they produce
      different target trees.

    Args:
      corpus: list of weighted tree pairs [(src_tree1, trg_tree1, weight1), ...].
      transducer: contains all xT rules.
      feat_inst: feature instantiator, used to populate features for rules.
      ncores: number of cores used to produce grammars in parallel.
    """
    # Obtain G1: a grammar that only obtains correct target trees.
    print('', file=sys.stderr)
    if self.augment_wrtgs:
      wrtgs_src_trg, weighted_tree_pairs = ObtainWRTGsAugmented(
        corpus, transducer, feat_inst, PerceptronModel, ncores)
    else:
      wrtgs_src_trg, weighted_tree_pairs = ObtainWRTGs(
        corpus, transducer, feat_inst, PerceptronModel, ncores)
    corpus_src = [(s, None, w) for s, t, w in corpus]
    # Obtain G2: a grammar only constrained on the source tree.
    if self.augment_wrtgs:
      wrtgs_src, _ = ObtainWRTGsAugmented(
        corpus_src, transducer, feat_inst, PerceptronModel, ncores)
    else:
      wrtgs_src, _ = ObtainWRTGs(
        corpus_src, transducer, feat_inst, PerceptronModel, ncores)

    # Remove productions whose predicates do not link to any entity in the grammar.
    if self.prod_filter:
      # wrtgs_src_trg = self.prod_filter.filter_prods_from_wrtgs(wrtgs_src_trg, corpus)
      wrtgs_src = self.prod_filter.filter_prods_from_wrtgs(wrtgs_src, corpus)

    for wrtg in wrtgs_src:
      if wrtg is not None:
        wrtg.feat_inst = feat_inst

    assert len(wrtgs_src_trg) == len(wrtgs_src) == len(weighted_tree_pairs), \
      'Failed test: {0} == {1} == {2}'\
      .format(len(wrtgs_src_trg), len(wrtgs_src), len(weighted_tree_pairs))
    return wrtgs_src_trg, wrtgs_src

  def decode_and_maybe_adjust(self, train_ind, wrtgs_src_trg, wrtgs_src, corpus):
    """
    This is the essence of each perceptron iteration:
    1. Re-weight wRTGs with current model parameters.
    2. Decode the best solution y* with that leads to the correct answer.
    3. Decode the best solution y' without knowledge of the correct answer.
    4. If y* == y', then no parameter value adjustments are necessary.
       Otherwise, adjust parameters.
    """
    error = 0.0
    wrtg_src_trg, wrtg_src = wrtgs_src_trg[train_ind], wrtgs_src[train_ind]
    if wrtg_src_trg is None or wrtg_src is None:
      print('-', end='', file=sys.stderr)
      return error
    # Update weights of rules given feature weight vector.
    self.weight_wrtg(wrtg_src)
    # Obtain the best derivation d2 from G2 that produces target tree t2.
    derivs_and_trees = self.GetBestValidDerivations(
      wrtg_src, self.cvt_inserter, nbest=1000, nvalid=1,
      query_manager=self.query_manager)
    if not derivs_and_trees:
      logging.warning(
        'Non-empty wRTG produced no derivations at {0}:\n{1}\n{2}'.format(
        train_ind, corpus[train_ind][0], corpus[train_ind][1]))
      print('-', end='', file=sys.stderr)
      return error
    derivation_src, trg_tree_hypo = derivs_and_trees[0]
    trg_tree_hypo = str(trg_tree_hypo)
    # If t1 == t2, do nothing. Otherwise, compute features of (source_tree, d1)
    # phi1 and (source_tree, d2) phi2.
    trg_tree = corpus[train_ind][1]
    if not self.trg_equals_gold(trg_tree_hypo, trg_tree):
      print('.', end='', file=sys.stderr)
      # Obtain the best derivation d0 from G1 that produces target tree t1.
      self.weight_wrtg(wrtg_src_trg)
      derivation_src_trg = wrtg_src_trg.ObtainDerivationsFromNT().next()
      feats_src_trg = CollectFeats(derivation_src_trg)
      feats_src = CollectFeats(derivation_src)
      score_src_trg = sum(fv * self.feat_weights[fid] for (fid, fv) in feats_src_trg)
      score_src = sum(fv * self.feat_weights[fid] for (fid, fv) in feats_src)
      error = score_src - score_src_trg
      # Adjust feature weights feat_weights += phi1 - phi2
      weights_diff = subtract_feats(feats_src_trg, feats_src, self.learning_rate)
      add_weights_to(self.feat_weights, weights_diff)
    else:
      print('~', end='', file=sys.stderr)
      self.correct_predictions_per_iter += 1
    return error

  def decode_and_maybe_adjust_(self, train_ind, wrtgs_src_trg, wrtgs_src, corpus):
    """
    This is the essence of each perceptron iteration:
    1. Re-weight wRTGs with current model parameters.
    1. Decode the best solution y* with that leads to the correct answer.
    3. Decode the best solution y' without knowledge of the correct answer.
    4. If y* == y', then no parameter value adjustments are necessary.
       Otherwise, adjust parameters.
    """
    wrtg_src_trg, wrtg_src = wrtgs_src_trg[train_ind], wrtgs_src[train_ind]
    if wrtg_src_trg is None or wrtg_src is None:
      print('-', end='', file=sys.stderr)
      return
    # Update weights of rules given feature weight vector.
    self.weight_wrtg(wrtg_src)
    # Obtain the best derivation d2 from G2 that produces target tree t2.
    derivation_src = wrtg_src.ObtainDerivationsFromNT().next()
    trg_tree_hypo, weight = TargetProjectionFromDerivation(derivation_src)
    target_projection = (trg_tree_hypo, weight)
    logging.debug('Target projection {0}'.format(target_projection))
    if wrtg_src.feat_inst is not None:
      logging.debug('\n'.join(
        [repr(d) + '\n' + d.rhs.rule.PrintYaml() + \
         '\n' + wrtg_src.feat_inst.DescribeFeatureIDs(d.rhs.rule) \
         for d in derivation_src]))
    if self.cvt_inserter:
      trg_tree_hypo = self.cvt_inserter.insert_cvt_if_needed(trg_tree_hypo)
    trg_tree_hypo = str(trg_tree_hypo)
    # print('Trg tree hypo: {0}'.format(trg_tree_hypo))
    # If t1 == t2, do nothing. Otherwise, compute features of (source_tree, d1)
    # phi1 and (source_tree, d2) phi2.
    trg_tree = corpus[train_ind][1]
    # print('Trg tree gold: {0}'.format(trg_tree))
    if not self.trg_equals_gold(trg_tree_hypo, trg_tree):
      print('.', end='', file=sys.stderr)
      # Obtain the best derivation d1 from G1 that produces target tree t1.
      self.weight_wrtg(wrtg_src_trg)
      derivation_src_trg = wrtg_src_trg.ObtainDerivationsFromNT().next()
      trg_tree_const, _ = TargetProjectionFromDerivation(derivation_src_trg)
      # print('Trg tree cons: {0}'.format(trg_tree_const))
      feats_src_trg = CollectFeats(derivation_src_trg)
      feats_src = CollectFeats(derivation_src)
      # Adjust feature weights feat_weights += phi1 - phi2
      weights_diff = subtract_feats(feats_src_trg, feats_src, self.learning_rate)
      add_weights_to(self.feat_weights, weights_diff)
    else:
      print('~', end='', file=sys.stderr)
      self.correct_predictions_per_iter += 1

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
  wrtg.ClearCaches()
  derivations = wrtg.ObtainDerivationsFromNT()
  for i, derivation in enumerate(derivations):
    if i >= nbest:
      break
    constituent, _ = TargetProjectionFromDerivation(derivation)
    constituent_str = \
      constituent if IsString(constituent) else constituent.pprint(margin=10000)
    valid_derivations.append((derivation, constituent))
    if i == 0:
      first_derivation = derivation
      first_tree = constituent
    if len(valid_derivations) >= nvalid:
      break
  if not valid_derivations:
    valid_derivations.append((first_derivation, first_tree))
  return valid_derivations

def add_weights_to(weights, new_weights):
  for feat_id, feat_val in new_weights.items():
    weights[feat_id] += feat_val

def subtract_feats(feats1, feats2, learning_rate=1.0):
  feats = defaultdict(float)
  for feat_id, feat_val in feats1:
    feats[feat_id] += feat_val * learning_rate
  for feat_id, feat_val in feats2:
    feats[feat_id] -= feat_val * learning_rate
  return feats

