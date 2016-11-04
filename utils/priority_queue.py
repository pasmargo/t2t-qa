import heapq
import numpy as np

class PriorityQueue:
  """
  Implements a fixed-length queue, where only the N-best (max_length)
  elements are kept. Elements (or items) are tuples, where the first element
  is the score (what is used to decide what are the best elements), and the
  second element of the tuple is the content.
  Currently, the PriorityQueue relies on heapq, which is a min-heap.
  For that reason, when using it with costs, costs will be stored negatively.
  The rest of the methods will have to convert back the costs to positive.
  """
  def __init__(self, max_length = 1, reverse = False):
    self.queue = []
    self.length = 0
    self.max_length = max_length
    self.threshold = 3.0
    # Case of N-best lists for probabilties (only highest probabilities are kept).
    if reverse:
      self.reverse = True
      self.GetBest = max
      self.normalize = lambda x: x
      self.denormalize = lambda x: x
    # Case of N-best lists for costs (only lowest costs are kept).
    else:
      self.reverse = False
      self.GetBest = lambda queue: min(queue, key=lambda x: abs(x[0]))
      self.normalize = lambda x: - abs(x)
      self.denormalize = lambda x: abs(x)

  def GetBestScore(self):
    """
    Get the best scoring item (which is the smallest cost or highest probability),
    and return its absolute value.
    """
    if not self.queue:
      return np.inf
    best_score_and_item = self.GetBest(self.queue)
    return self.denormalize(best_score_and_item[0])

  def GetBestScoreItem(self):
    """
    Get the best scoring item item (which is the smallest cost or highest probability),
    and return its absolute value.
    """
    if not self.queue:
      return np.inf
    best_score_and_item = self.GetBest(self.queue)
    return best_score_and_item[1]

  def GetBestScoreAndItem(self):
    """
    Get the best scoring item item (which is the smallest cost or highest probability),
    and return its absolute value.
    """
    if not self.queue:
      return (np.inf, None)
    (best_score, item) = self.GetBest(self.queue)
    return (self.denormalize(best_score), item)

  def GetItems(self):
    """
    Returns the list of items that are tupled to scores
    (not the scores themselves).
    """
    return [x[1] for x in self.queue]

  def GetScoresAndItems(self):
    return [(self.denormalize(x[0]), x[1]) for x in self.queue]

  def GetSortedScoresAndItems(self):
    sorted_scores_and_items = \
      sorted(self.queue, key=lambda x: self.denormalize(x[0]),
             reverse=self.reverse)
    return [(self.denormalize(x[0]), x[1]) for x in sorted_scores_and_items]

  def Push(self, score, transformation, new_max_length = np.inf):
    assert len(self.queue) <= self.max_length, \
      'Length of queue went over max: {0}'.format(self.queue)
    # Check if the item is already in the Queue. If so, then remember
    # its index. Its score will be updated later and the heap heapified.
    update_succeed = False
    replace_index = None
    norm_score = self.normalize(score)
    for i, (stored_score, stored_trans) in enumerate(self.queue):
      if stored_trans == transformation:
        if stored_score < norm_score:
          replace_index = i
          break
        else:
          return update_succeed
    item = (norm_score, transformation)
    # Case: when the item was already in the queue, its score should be
    # updated (if it is better score). Finally, heap is heapified.
    if replace_index != None:
      self.queue[replace_index] = item
      heapq.heapify(self.queue)
      update_succeed = True
    # Case: the queue is still small. More items fit, regardless of their score.
    elif len(self.queue) < self.max_length and len(self.queue) < new_max_length:
      heapq.heappush(self.queue, item)
      update_succeed = True
    # Case: queue is already full. Item is pushed in only if its score is better
    # than the worst item in the queue.
    else:
      old_score, old_item = heapq.heappushpop(self.queue, item)
      if old_score != norm_score:
        update_succeed = False
    return update_succeed

  def PushNoRepetitionCheck(self, score, transformation, new_max_length = np.inf):
    item = (self.normalize(score), transformation)
    if len(self.queue) < self.max_length and len(self.queue) < new_max_length:
      heapq.heappush(self.queue, item)
    else:
      heapq.heappushpop(self.queue, item)
    return True

  def FilterCache(self):
    """
    Remove elements that have twice the score of the minimum item.
    """
    if not self.queue:
      return
    best_score_and_item = self.GetBestScoreAndItem()
    best_score = best_score_and_item[0]
    threshold = best_score + best_score * self.threshold
    self.queue = \
      filter(lambda x: self.denormalize(x[0]) <= threshold, self.queue)
    self.length = len(self.queue)
    heapq.heapify(self.queue)
    return self.queue

