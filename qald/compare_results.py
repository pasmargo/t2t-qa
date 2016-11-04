#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import json
import logging
import sys

from decoder.decode_qa_utils import QueryLambdaDCS, QueryLambdaDCSC
from utils.evaluation import compute_f1_score
from utils.tree_tools import IsString

parser = argparse.ArgumentParser()
parser.add_argument(
  "--by", nargs='?', type=str, default="entities",
  help="Target value for comparison between gold and system. Options: entities, labels.")
parser.add_argument(
  "--metric", nargs='?', type=str, default="accuracy",
  help="Metric (accuracy or fscore).")
parser.add_argument(
  "sys_file", nargs='?', type=argparse.FileType('r'), default=sys.stdin)
parser.add_argument(
  "ref_file", nargs='?', type=argparse.FileType('r'), default=sys.stdin)
args = parser.parse_args()

logging.basicConfig(level=logging.WARNING)

def get_entity_label(entity):
  assert IsString(entity)
  if ' ' in entity:
    return entity
  label_results = QueryLambdaDCSC(u'(ID !fb:type.object.name <{0}>)'.format(entity))
  if not label_results:
    return entity
  if len(label_results) > 1:
    logging.warning(
      u'More than one label results for entity {0} = {1}'.format(
      entity, ', '.join(label_results)))
  return label_results[0]

# Retrieve results computed by training system.
# Labels are also retrieved if --by "labels" is activated
# in the command line.
sys_output = json.load(args.sys_file)
for out in sys_output:
  out['sys_results'] = []
  if args.by == "labels":
    out['sys_results_labels'] = []
  for dcs_query in out['sys_queries']:
    result = QueryLambdaDCS(dcs_query)
    if result is None:
      continue
    out['sys_results'].append(result)
    if args.by == "labels":
      labels = [get_entity_label(e) for e in result]
      out['sys_results_labels'].append(labels)

# Query gold lambda-DCS expressions and retrieve results.
pairs = json.load(args.ref_file)
num_sys_outputs = len(sys_output)
gold_results = []
for i, pair in enumerate(pairs):
  if i >= num_sys_outputs:
    break
  sys_output[i]['utterance'] = pair['utterance']
  if "targetFormula" in pair:
    sys_output[i]['gold_query'] = pair["targetFormula"]
    sys_output[i]['gold_result'] = QueryLambdaDCS(sys_output[i]['gold_query'])
  else:
    sys_output[i]['gold_query'] = ""
    sys_output[i]['gold_result'] = []
  if args.by == "labels":
    if "answer_labels" in pair:
      sys_output[i]['answer_labels'] = pair['answer_labels']
    else:
      labels = [get_entity_label(e) for e in sys_output[i]['gold_result']]
      sys_output[i]['answer_labels'] = labels

def evaluate_sys_accuracy(out, sys_field, ref_field):
  try:
    result_found_at_position = out[sys_field].index(out[ref_field])
  except ValueError:
    result_found_at_position = None
  one_best = 1 if result_found_at_position == 0 else 0
  oracle = 1 if result_found_at_position is not None else 0
  return one_best, oracle, result_found_at_position

def evaluate_sys_fscore(out, sys_field, ref_field):
  assert sys_field in out and ref_field in out, \
    u"Missing information {0} and {1} in:\n{2}".format(
    sys_field, ref_field, json.dumps(out, indent=2))
  result_found_at_position = None
  one_best, oracle = 0.0, 0.0
  if not out[sys_field]:
    return one_best, oracle, result_found_at_position
  one_best = compute_f1_score(out[sys_field][0], out[ref_field])
  f1_scores = [
    compute_f1_score(answers, out[ref_field]) for answers in out[sys_field]]
  oracle = max(f1_scores)
  result_found_at_position = f1_scores.index(oracle)
  return one_best, oracle, result_found_at_position

def evaluate_sys(out, by, metric="accuracy"):
  if by == "entities":
    sys_field, ref_field = "sys_results", "gold_result"
  else:
    sys_field, ref_field = "sys_results_labels", "answer_labels"
  if metric == "accuracy":
    return evaluate_sys_accuracy(out, sys_field, ref_field)
  else:
    return evaluate_sys_fscore(out, sys_field, ref_field)

# Compare training and gold results in terms of entities or their labels.
num_answered = 0
one_best_total = 0.0
oracle_total = 0.0

for i, out in enumerate(sys_output):
  if args.by == "labels" and not out["answer_labels"]:
    out['correct_query'] = "n/a"
    out['correct_result'] = "n/a"
    out['result_at_position'] = "n/a"
    continue
  if args.by == "entities" and out['gold_result'] in ["(list)", []]:
    out['correct_query'] = "n/a"
    out['correct_result'] = "n/a"
    out['result_at_position'] = "n/a"
    continue
  num_answered += 1
  one_best, oracle, result_found_at_position = evaluate_sys(out, args.by, args.metric)
  one_best_total += one_best
  oracle_total += oracle
  if result_found_at_position is not None:
    out['correct_query'] = out['sys_queries'][result_found_at_position]
    out['correct_result'] = out['sys_results'][result_found_at_position]
    out['result_at_position'] = result_found_at_position
  else:
    out['correct_query'] = None
    out['correct_result'] = None
    out['result_at_position'] = None
print(json.dumps(sys_output, indent=2))

if num_answered != 0:
  one_best_accuracy = float(one_best_total) / num_answered
  oracle_accuracy = float(oracle_total) / num_answered
else:
  one_best_accuracy = 0.0
  oracle_accuracy = 0.0
print('1-best accuracy: {0:.1f} / {1} = {2:,.2f}'\
      .format(one_best_total, num_answered, one_best_accuracy),
      file=sys.stderr)
print('oracle accuracy: {0:.1f} / {1} = {2:,.2f}'\
      .format(oracle_total, num_answered, oracle_accuracy),
      file=sys.stderr)

