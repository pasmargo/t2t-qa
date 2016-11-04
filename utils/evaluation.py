#!/usr/bin/python
# -*- coding: utf-8 -*-

# TODO: Since @items is not a set, it is possible that the recall
# goes beyond 1.0. At the moment, I am simply limiting the max recall,
# but we should find a better way to do this.
def compute_f1_score(items, gold_items):
  """
  @items is a list with our predictions (may contain repetitions).
  @gold_items is a set with the gold answers.
  """
  hits, failed = 0.0, 0.0
  hits = float(sum([item in gold_items for item in items]))
  if not items:
    precision = 0.0
  else:
    precision = hits / len(items)
  if not gold_items:
    recall = 0.0
  else:
    recall = min(1.0, hits / len(gold_items))
  if precision == 0.0 and recall == 0.0:
    f1_score = 0.0
  else:
    f1_score = 2.0 * precision * recall / (precision + recall)
  return f1_score

