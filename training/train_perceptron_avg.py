from __future__ import print_function
from collections import defaultdict
import logging
import sys

from training.train_perceptron import PerceptronModel, add_weights_to

class AveragedPerceptronModel(PerceptronModel):
  """ Implements an averaged structured perceptron model.

  It contains a vector (default dictionary) of feature weights.
  Weights are estimated using the training procedure of the structured perceptron.
  In the averaged perceptron, hyperplanes (feature weights) that survive longer
  (that are not updated in a long time) should have larger impact on the final
  estimation of model parameters.
  """

  def train(self, transducer, corpus, feat_inst=None, ncores=1):
    """
    Implements averaged Structured Perceptron.
    Feature weights (separating hyperplane) is accumulated into a dictionary
    after every observation. At the end of the iterations, we divide each weight
    by num_iterations * num_observations. Popular hyperplane values will bias
    the final estimations.

    Args:
      transducer xT
      corpus: a list of 3-tuples [(src_tree, trg_tree, pair_weight), ...]
      feat_inst: object Feature Instantiator to extract features from new rules.
    """
    wrtgs_src_trg, wrtgs_src = self.produce_grammars(
      corpus, transducer, feat_inst, ncores)
    acc_feat_weights, total_acc_updates = defaultdict(float), 0
    for i in range(self.max_iterations):
      error = 0.0
      for train_ind in range(len(wrtgs_src_trg)):
        error_instance = self.decode_and_maybe_adjust(
          train_ind, wrtgs_src_trg, wrtgs_src, corpus)
        error += error_instance
        add_weights_to(acc_feat_weights, self.feat_weights)
        total_acc_updates += 1
        # print('Instance error: {0}'.format(error))
      print(' Accuracy = {0}, error = {1}'.format(
        self.correct_predictions_per_iter / float(len(wrtgs_src_trg)),
        error), end='', file=sys.stderr)
      print('', file=sys.stderr)
      self.correct_predictions_per_iter = 0
    divide_weights_by(acc_feat_weights, total_acc_updates)
    self.feat_weights = acc_feat_weights

def divide_weights_by(weights, constant):
  for feat_id, weight in weights.items():
    weights[feat_id] /= constant

