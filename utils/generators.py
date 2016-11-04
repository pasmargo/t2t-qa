#!/usr/bin/env python
# -*- coding: utf-8 -*-

import heapq
import itertools
import numpy as np

class GeneratorsList:
  """
  Given a list of generators that produce items in descending order,
  this class allows several operations on them.
  The first operation is the generation of a single sequence of items
  sorted in descending order, according to a user-defined function.
  """
  def __init__(self, generators_list, order_func):
    self.gen_list = []
    for gen in generators_list:
      self.gen_list.append(gen() if callable(gen) else iter(gen))
    self.order_func = order_func
    # self.current_items contains a list of tuples [(score, item), ...].
    self.current_items = []
    if __debug__:
      self.log = []

  def items(self):
    for generator in self.gen_list:
      scored_item = GetScoredItemFromGenerator(generator, self.order_func)
      self.current_items.append(scored_item)
    prev_score = np.inf
    while True:
      if not self.current_items:
        raise StopIteration
      scores = [x[0] for x in self.current_items]
      max_score = max(scores)
      if max_score is None:
        raise StopIteration
      prev_score = CheckDecreasingAndReturnCurrent(max_score, prev_score)
      max_score_index = scores.index(max_score)
      assert max_score_index < len(self.current_items)
      item = self.current_items[max_score_index][1]
      yield item
      scored_item = GetScoredItemFromGenerator(
        self.gen_list[max_score_index], self.order_func)
      self.current_items[max_score_index] = scored_item
      if __debug__:
        self.log.append((max_score_index, scored_item))

def CheckDecreasingAndReturnCurrent(current_score, previous_score):
  if abs(current_score) == np.inf and abs(previous_score) == np.inf:
    return current_score
  assert round(previous_score - current_score, 7) >= 0.0, ('Non monotonic list: '
    'current_score {0} >= prev_score {1}.').format(current_score, previous_score)
  return current_score

def GetScoredItemFromGenerator(generator, order_func):
  try:
    item = generator.next()
    score = order_func(item)
    scored_item = (score, item)
  except StopIteration:
    scored_item = (None, None)
  return scored_item

# Obtained from:
# http://stackoverflow.com/questions/12093364/cartesian-product-of-large-iterators-itertools
def product(*iterables, **kwargs):
    """
    Obtains cartesian product from iterables, in the same way as itertools.product
    does. However, this function does not expand the iterables in memory.
    """
    if len(iterables) == 0:
        yield ()
    else:
        iterables = iterables * kwargs.get('repeat', 1)
        it = iterables[0]
        for item in it() if callable(it) else iter(it):
            for items in product(*iterables[1:]):
                yield (item, ) + items

# Testing:
# import itertools
# g = product(lambda: itertools.permutations(xrange(100)),
#             lambda: itertools.permutations(xrange(100)))
# print next(g)
# print sum(1 for _ in g)

class State:
  def __init__(self, score, items, iterables, order_func, state_id, history_items={}):
    self.score = score
    self.items = items
    self.iterables = iterables
    self.order_func = order_func
    # self.next_scores is a heapq.
    self.state_id = state_id
    self.history_items = history_items
    if not self.history_items:
      self.history_items = {i : {} for i in range(len(self.iterables))}
    self.next_scores = []

  def __repr__(self):
    return str(self.score)

  def __eq__(self, other):
    return isinstance(other, self.__class__) and \
           self.score == other.score

  def __ne__(self, other):
    return not self.eq(other)

  def __gt__(self, other):
    return isinstance(other, self.__class__) and \
           self.score > other.score

  def __ge__(self, other):
    return isinstance(other, self.__class__) and \
           self.score >= other.score

  def __lt__(self, other):
    return isinstance(other, self.__class__) and \
           self.score < other.score

  def __le__(self, other):
    return isinstance(other, self.__class__) and \
           self.score <= other.score

  def GetItemsFromIterablesCached(self):
    """
    Get cached items from iterables. These cached items are stored
    in a dictionary of dictionaries
    self.history_items: iter_num x iter_counter -> item
    If self.history_items[iter_num][iter_counter] is "unexplored",
    then a new item is drawn from the iterable and stored in the cache.
    If self.history_items[iter_num][iter_counter] is None, it means
    that there are no more elements in the iterable.
    It is assumed here that the state_id is a tuple where each component
    is the current index in each iterable.
    """
    items = []
    assert len(self.state_id) == len(self.iterables)
    for iter_num, iter_counter in enumerate(self.state_id):
      item = self.history_items[iter_num].get(iter_counter+1, "unexplored")
      if item == "unexplored":
        try:
          item = self.iterables[iter_num].next()
        except StopIteration:
          item = None
        self.history_items[iter_num][iter_counter+1] = item
      items.append(item)
    return items

  def GetItemAt(self, iter_num, iter_counter):
    """
    Returns the item at position iter_counter from iterable iter_num.
    First we try to retrieve it from the cache. If we fail, then
    retrieve it from the iterable.
    Returns None if no more items are left.
    """
    assert iter_num < len(self.iterables)
    item = self.history_items[iter_num].get(iter_counter, "unexplored")
    if item == "unexplored":
      try:
        item = self.iterables[iter_num].next()
      except StopIteration:
        item = None
      self.history_items[iter_num][iter_counter+1] = item
    return item

  def GetNextScores(self):
    heads = self.GetItemsFromIterablesCached()
    next_scores = []
    assert not self.next_scores
    for i, head in enumerate(heads):
      if head is None:
        continue
      next_score = self.order_func(self.items[:i] + [head] + self.items[i+1:])
      next_score = - float(next_score)
      next_scores.append((next_score, i))
    heapq.heapify(next_scores)
    return next_scores

  def ProduceNewStates(self):
    new_states = []
    assert not self.next_scores
    self.next_scores = self.GetNextScores()
    while self.next_scores:
      _, i = heapq.heappop(self.next_scores)
      assert i < len(self.iterables)
      head = self.GetItemAt(i, self.state_id[i]+1)
      # If we have exhausted iterable i, we move to second-best iterable.
      if head is None:
        continue
      new_items = self.items[:i] + [head] + self.items[i+1:]
      score = - float(self.order_func(new_items))
      assert round(score - self.score, 7) >= 0 or \
             score == np.inf or \
             self.score == np.inf, \
        ('New item in OrderedProduct has larger score'
         ' than previous item at state_id {0}. score: {1} vs. prev_score {2}')\
         .format(self.state_id, abs(score), abs(self.score))
      new_state_id = self.state_id[:i] \
                   + (self.state_id[i] + 1,) \
                   + self.state_id[i+1:]
      new_state = State(score, new_items, self.iterables, self.order_func,
                        new_state_id, self.history_items)
      new_states.append(new_state)
    return new_states

def GetHeadsFromIterables(iterables):
  new_iterables = []
  head_items = []
  for iterable in iterables:
    head, new_iterable = PeekIterable(iterable)
    head_items.append(head)
    new_iterables.append(new_iterable)
  return head_items, new_iterables

def TeeIterables(iterables):
  iters1, iters2 = [], []
  for iterable in iterables:
    iter1, iter2 = itertools.tee(iterable)
    iters1.append(iter1)
    iters2.append(iter2)
  return iters1, iters2

def PeekIterable(iterable):
  """
  Returns a tuple with the first element of the iterable,
  and the iterable. It returns None if the iterable is exhausted.
  """
  try:
    head_element = iterable.next()
    new_iterable = itertools.chain([head_element], iterable)
    return head_element, new_iterable
  except StopIteration:
    return None, iterable

def MoveItem(src_queue, trg_queue, order_func):
  """
  Move the top of src_queue to trg_queue, after computig order_func on it
  and its score.
  """
  score, item = heapq.heappop(src_queue)
  score = - float(order_func(- score, item))
  heapq.heappush(trg_queue, (score, item))
  return item

def ExpandTopInto(src_queue, trg_queue, cached_states, min_bound=1.0):
  """
  Expand top of src_queue into trg_queue.
  """
  _, best_state = src_queue[0]
  # Produce more candidate items.
  new_states = best_state.ProduceNewStates()
  for new_state in new_states:
    if new_state.state_id not in cached_states:
      score = new_state.score * min_bound
      heapq.heappush(trg_queue, (score, new_state))
      cached_states.add(new_state.state_id)

def IsItemInBuffGuaranteedMin(buff, candidate_states):
  """
  This function tells whether the best item (top of queue) of buff
  has a score better than the best lower bound of candidate_states.
  Base case: if there are not candidate_states, we return False,
             because we cannot guarantee that the best item in buff
             is better than any other future item in candidate_states.
  """
  assert buff
  if not candidate_states:
    return False
  best_lower_bound = - float(candidate_states[0][0])
  best_score = - float(buff[0][0])
  return best_score >= best_lower_bound

def OrderedProduct(*iterables, **kwargs):
  cached_states = set() # to ensure that we do not create repeated elements.
  assert len(iterables) > 0
  order_func = kwargs.get('key', lambda x: x)
  lm_order_func = kwargs.get('lm', lambda x, y: x)
  if lm_order_func is None:
    lm_order_func = lambda x, y: x
  min_bound = kwargs.get('bound', 1.0)
  
  iters = [it() if callable(it) else iter(it) for it in iterables]
  # assert AreIterablesSorted(iterables, order_func), 'Iterables are not sorted: '
  first_items, new_iterables = GetHeadsFromIterables(iters)
  # Consume one element from each iterable.
  [it.next() for it in new_iterables]
  score = order_func(first_items)
  score = - float(score)
  state_id = (0,) * len(iterables)
  history_items = {i : {0 : item} for i, item in enumerate(first_items)}
  # The state's score should be computed without the LM nor the LM estimation.
  state = State(score, first_items, new_iterables, order_func, state_id, history_items)
  cached_states.add(state_id)
  # candidate_states will contain a min-heap of -LM items plus a lower bound.
  # It should be guaranteed that any future items that are to be pushed into
  # this heap should have a lower bound that is greater or equal than the
  # worst item currently in candidate_states (monotonicity of lower bound).
  candidate_states = []
  # buff will contain items with their true score, waiting for being popped-out
  # and yielded.
  buff = []
  # The score to rank the state should count with the lower (constant) bound
  # for this hyperedge.
  heapq.heappush(candidate_states, (score, state))
  prev_score = np.inf
  while buff or candidate_states:
    if not buff:
      MoveItem(candidate_states, buff, lm_order_func)
    else:
      if IsItemInBuffGuaranteedMin(buff, candidate_states):
        score, best_state = heapq.heappop(buff)
        if score != np.inf:
          prev_score = CheckDecreasingAndReturnCurrent(- score, prev_score)
          yield tuple(best_state.items)
        ExpandTopInto([(None, best_state)], candidate_states, cached_states, min_bound)
      else:
        if candidate_states:
          item = MoveItem(candidate_states, buff, lm_order_func)
          ExpandTopInto([(None, item)], candidate_states, cached_states, min_bound)
        else:
          ExpandTopInto(buff, candidate_states, cached_states, min_bound)
          # if not candidate_states and len(buff) == 1:
          if not candidate_states:
            score, best_state = heapq.heappop(buff)
            if score != np.inf:
              prev_score = CheckDecreasingAndReturnCurrent(- score, prev_score)
              yield tuple(best_state.items)

def IsIterableSorted(iterable, score_func):
  prev_score = np.inf
  for element in iterable:
    score = score_func([element])
    print('Score: {0}'.format(score))
    if score > prev_score:
      print('End')
      return False
    prev_score = score
  print('End')
  return True

def AreIterablesSorted(iterables, score_func):
  iters = [it() if callable(it) else iter(it) for it in iterables]
  return all([IsIterableSorted(it, score_func) for it in iters])
