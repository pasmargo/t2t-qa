from __future__ import print_function
import copy
from multiprocessing import Pool
import sys

from training.transducer import xT
from training.transductionrule import XTRule
from utils.production import Production, RHS
from utils.tree_tools import immutable, tree_or_string, IsString, IsVariable

# A global transducer is helpful when multiprocessing,
# since child sub-processes inherit this variable faster
# than the pickling/unpickling associated to passing it
# as an argument to a target function of a Process.
transducer = None
feat_inst = None
model_class = None
# When multiprocessing, this variable defines how many tasks
# the processes will complete before being replaced by a new
# process. It can be helpful to release resources or clear memory.
kMaxTasksPerChild = None

def CombineScoresOfDerivations(derivations):
  score = sum([GetScoreOfDerivation(d) for d in derivations])
  return float(score)

def GetScoreOfDerivation(derivation):
  score = sum([p.rhs.rule.weight for p in derivation])
  return float(score)

def ObtainWRTG(weighted_tree_pair, print_result=True):
  """
  Given a transducer and a weighted source/target tree, it returns a tuple
  that contains the wRTG and the weighted pair. If the transducer fails at
  explaining the source/target tree with the rules it has, then it returns
  a tuple (None, None). The weights of the RTG are not estimated here.

  global variables used here (bad practice, but need for parallelization):
    * transducer
    * feat_inst
    * model_class
    * GetScoreOfDerivation
    * CombineScoresOfDerivations
  """
  intree_str, outtree_str, pair_weight = weighted_tree_pair
  intree  = immutable(tree_or_string(intree_str))
  outtree = None if outtree_str is None else immutable(tree_or_string(outtree_str))
  wrtg = transducer.Transduce(intree, outtree, convert_to_prob=False)
  sys.stdout.flush()
  if not wrtg.P:
    output = (None, None)
    result_str = 'x'
  else:
    wrtg.ScoreDerivation = GetScoreOfDerivation
    wrtg.CombineDerivationScores = CombineScoresOfDerivations
    if feat_inst:
      feat_inst.SetContext({'src_tree' : intree_str})
    model_class.populate_wrtg_feats(wrtg, feat_inst)
    output = (wrtg, weighted_tree_pair)
    result_str = 'o'
  if print_result:
    result_str = result_str if outtree is not None else result_str.upper()
    print(result_str, end='', file=sys.stderr)
  return output

def ObtainWRTGsSequential(corpus):
  """
  Given a transducer and a corpus, it returns a tuple that contains
  a list of wRTGs and weighted tree pairs that could been explained by transducer.
  """
  wrtgs = []
  weighted_pairs = []
  for weighted_tree_pair in corpus:
    wrtg, weighted_tree_pair = ObtainWRTG(weighted_tree_pair)
    wrtgs.append(wrtg)
    weighted_pairs.append(weighted_tree_pair)
  return wrtgs, weighted_pairs

import cProfile, os
def ObtainWRTGBis(weighted_tree_pair):
  return cProfile.runctx('ObtainWRTG(weighted_tree_pair)',
                  globals(), locals(), 'prof%d.prof' % os.getpid())

def ObtainWRTGsParallel(corpus, num_cores=2):
  pool = Pool(processes=num_cores, maxtasksperchild=kMaxTasksPerChild)
  wrtgs_and_pairs = pool.map_async(ObtainWRTG, corpus).get(9999999)
  pool.close()
  pool.join()
  if not wrtgs_and_pairs:
    wrtgs, weighted_pairs = [], []
  else: 
    wrtgs, weighted_pairs = zip(*wrtgs_and_pairs)
  return wrtgs, weighted_pairs

def ObtainWRTGs(corpus, transducer_l, feat_inst_l, model_class_l, ncores=1):
  """
  Given a transducer and a corpus, it returns a tuple that contains
  a list of wRTGs and weighted tree pairs that could been explained by transducer.
  """
  global transducer, feat_inst, model_class
  transducer, feat_inst, model_class = transducer_l, feat_inst_l, model_class_l
  if ncores > 1:
    results = ObtainWRTGsParallel(corpus, ncores)
  else:
    results = ObtainWRTGsSequential(corpus)
  print('', file=sys.stderr)
  return results

def ObtainWRTGAugmented(weighted_tree_pair):
  """
  Given a transducer and a weighted source/target tree, it returns a tuple
  that contains the wRTG and the weighted pair. If the transducer fails at
  explaining the source/target tree with the rules it has, then it returns
  a tuple (None, None). The weights of the RTG are not estimated here.

  global variables used here (bad practice, but need for parallelization):
    * transducer
    * feat_inst
    * model_class
    * GetScoreOfDerivation
    * CombineScoresOfDerivations
  """
  global transducer
  intree_str, outtree_str, pair_weight = weighted_tree_pair
  wrtg = ObtainWRTG((intree_str, None, pair_weight), print_result=False)[0]
  if not wrtg or not wrtg.P:
    output = (None, None)
    result_str = 'X'
  else:
    productions = AddAllPredicatesForEntitiesFromProds(wrtg.P, transducer.linker)
    rules = list(set(p.rhs.rule for p in productions))
    rules_augmented = list(set(transducer.rules[:] + rules))
    transducer_aug = xT(
      transducer.start_state, rules_augmented, transducer.rule_index.rule_backoffs)
    transducer_back = transducer
    transducer = transducer_aug
    wrtg = ObtainWRTG((intree_str, outtree_str, pair_weight), print_result=False)[0]
    transducer = transducer_back
    if not wrtg or not wrtg.P:
      output = (None, None)
      result_str = 'x'
    else:
      wrtg.ScoreDerivation = GetScoreOfDerivation
      wrtg.CombineDerivationScores = CombineScoresOfDerivations
      if feat_inst:
        feat_inst.SetContext({'src_tree' : intree_str})
      model_class.populate_wrtg_feats(wrtg, feat_inst)
      output = (wrtg, weighted_tree_pair)
      result_str = 'o'
  sys.stdout.flush()
  result_str = result_str if outtree_str is not None else result_str.upper()
  print(result_str, end='', file=sys.stderr)
  return output

def ObtainWRTGsAugmentedParallel(corpus, num_cores=2):
  pool = Pool(processes=num_cores, maxtasksperchild=kMaxTasksPerChild)
  wrtgs_and_pairs = pool.map_async(ObtainWRTGAugmented, corpus).get(9999999)
  pool.close()
  pool.join()
  if not wrtgs_and_pairs:
    wrtgs, weighted_pairs = [], []
  else: 
    wrtgs, weighted_pairs = zip(*wrtgs_and_pairs)
  return wrtgs, weighted_pairs

def ObtainWRTGsAugmentedSequential(corpus):
  """
  Given a transducer and a corpus, it returns a tuple that contains
  a list of wRTGs and weighted tree pairs that could been explained by transducer.
  """
  wrtgs = []
  weighted_pairs = []
  for weighted_tree_pair in corpus:
    wrtg, weighted_tree_pair = ObtainWRTGAugmented(weighted_tree_pair)
    wrtgs.append(wrtg)
    weighted_pairs.append(weighted_tree_pair)
  return wrtgs, weighted_pairs

def ObtainWRTGsAugmented(corpus, transducer_l, feat_inst_l, model_class_l, ncores=1):
  """
  Given a transducer and a corpus, it returns a tuple that contains
  a list of wRTGs and weighted tree pairs that could been explained by transducer.
  """
  global transducer, feat_inst, model_class
  transducer, feat_inst, model_class = transducer_l, feat_inst_l, model_class_l
  if ncores > 1:
    results = ObtainWRTGsAugmentedParallel(corpus, ncores)
  else:
    results = ObtainWRTGsAugmentedSequential(corpus)
  print('', file=sys.stderr)
  return results

def AddAllPredicatesForEntitiesFromProds(productions, linker):
  """
  This function is designed to compensate lack of coverage
  on our predicate linker. We collect all entities from rules,
  and then we obtain all predicates that connect to those entities.
  Then, we extend the rules by adding one more rule for each extra predicate.
  """
  rules = list(set([production.rhs.rule for production in productions]))
  # Get entities.
  entities = GetEntitiesFromRules(rules, linker)
  # Get the predicates (with reverse operation if needed) for entities.
  predicates = set(
    p for e in entities for p in GetEntityPredicates(e, linker, with_reverse_op=True))
  # Get a dictionary of productions with predicates.
  # We characterize a production by its non-terminal and non-terminals of RHS.
  productions_with_preds = {}
  for p in productions:
    if p.rhs.rule.state == 'predicate':
      productions_with_preds[(p.non_terminal, tuple(p.rhs.non_terminals))] = p
  # Go through all these productions and duplicate productions with
  # a different predicate as their rule's rhs.
  extended_productions = list()
  for prod in productions_with_preds.values():
    if prod.non_terminal[0] == 'predicate':
      rule = prod.rhs.rule
      for pred in predicates:
        new_rule = XTRule(
          rule.state, rule.lhs, tree_or_string(pred), rule.newstates, rule.weight)
        deriv_rhs = RHS(new_rule, prod.rhs.non_terminals)
        new_prod = Production(prod.non_terminal, deriv_rhs, None)
        extended_productions.append(new_prod)
  extended_productions.extend(productions)
  return list(set(extended_productions))

def GetEntityPredicates(entity, linker, with_reverse_op=False):
  preds_when_obj = linker.GetURIField(entity, 'preds_when_obj')
  preds_when_subj = linker.GetURIField(entity, 'preds_when_subj')
  preds = set()
  if preds_when_obj:
    preds.update(preds_when_obj)
  if preds_when_subj:
    if with_reverse_op:
      preds.update('!' + p for p in preds_when_subj)
    else:
      preds.update(preds_when_subj)
  preds = set(p for p in preds if not IsFBPredicate(p))
  return preds

def IsFBPredicate(pred_uri):
  return pred_uri.startswith('fb:type') or \
         pred_uri.startswith('fb:common') or \
         pred_uri.startswith('fb:freebase') or \
         pred_uri.startswith('fb:user') or \
         pred_uri.startswith('fb:dataworld') or \
         pred_uri.startswith('fb:base.yupgrade') or \
         pred_uri.startswith('fb:base.wordnet') or \
         pred_uri.startswith('fb:base.ontologies') or \
         pred_uri.startswith('fb:community') or \
         pred_uri.startswith('fb:base.skosbase')

def GetEntitiesFromRules(rules, linker):
  entities_from_entity_state = set(r.rhs for r in rules if r.state == 'entity')
  uris_from_other_states = GetURIsFromRules(rules)
  entities_from_other_states = FilterEntitiesFromURIs(uris_from_other_states, linker)
  entities = entities_from_entity_state.union(entities_from_other_states)
  return entities

def GetURIsFromRules(rules):
  uris = set()
  for rule in rules:
    if IsString(rule.rhs) and not IsVariable(rule.rhs):
      uris.add(rule.rhs)
    if not IsString(rule.rhs):
      uris.update(u for u in rule.rhs.leaves() if not IsVariable(u))
  return uris

def FilterEntitiesFromURIs(uris, linker):
  entities = set()
  for uri in uris:
    roles = linker.GetURIField(uri, 'role')
    if roles and not 'predicate' in roles:
      entities.add(uri)
  return entities
